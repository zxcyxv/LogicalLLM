import os
import re
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# hcsf.py가 같은 경로에 있어야 합니다.
from hcsf import GPT2WithHCSF

# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    # Model
    base_model: str = "gpt2"
    k_steps_max: int = 5
    top_k: int = 4
    eta: float = 0.01
    lam: float = 0.1
    
    # Data
    max_length: int = 512
    train_subset: Optional[int] = None 
    eval_subset: int = 100

    # Training (20GB VRAM 최적화)
    batch_size: int = 4  # 메모리 절약을 위해 작게 설정
    gradient_accumulation_steps: int = 8 # 실질 배치 사이즈 16 유지
    learning_rate: float = 2e-5
    num_epochs: int = 3
    gamma: float = 0.01  # Sheaf Energy 가중치 (초기엔 작게 설정 권장)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./outputs/hcsf_gsm8k"

# =============================================================================
# 2. Dataset with Loss Masking (질문 부분 학습 제외)
# =============================================================================

class GSM8KDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=512, subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        dataset = load_dataset("gsm8k", "main", split=split)
        if subset_size:
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question_part = f"Question: {example['question']}\nAnswer:"
        full_text = f"{question_part} {example['answer']}{self.tokenizer.eos_token}"

        # 1. 질문 부분 토큰 길이 측정 (Loss Masking용)
        q_enc = self.tokenizer(question_part, truncation=True, max_length=self.max_length)
        q_len = len(q_enc["input_ids"])

        # 2. 전체 텍스트 토큰화
        full_enc = self.tokenizer(
            full_text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # 3. Labels 생성 및 질문 부분 마스킹 (-100은 CE Loss에서 무시됨)
        labels = input_ids.clone()
        labels[:q_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question": example["question"],
            "answer": example["answer"]
        }

# =============================================================================
# 3. Loss & Trainer (Gradient Flow 보존 버전)
# =============================================================================

class HCSFTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model.to(config.device)
        self.config = config
        self.tokenizer = tokenizer
        
        # Optimizer: HCSF 파라미터만 학습
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate)
        
        # Loss: Reduction='mean'으로 설정하여 안정화
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.config.device)
        attention_mask = batch["attention_mask"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)

        # Forward
        logits, diagnostics = self.model(input_ids, attention_mask=attention_mask)

        # 1. LM Loss
        V = logits.shape[-1]
        lm_loss = self.ce_loss(logits[:, :-1, :].reshape(-1, V), labels[:, 1:].reshape(-1))

        # 2. Sheaf Loss (IMPORTANT: diagnostics['post_energy']가 Tensor여야 함)
        # item()을 쓰지 않고 직접 텐서를 사용하여 그래프 유지
        sheaf_energy = diagnostics["post_energy"] 
        sheaf_loss = torch.log(1 + sheaf_energy) # 수치 폭주 방지를 위해 로그 스케일 적용

        total_loss = lm_loss + self.config.gamma * sheaf_loss
        
        return total_loss, lm_loss, sheaf_energy

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        pbar = tqdm(loader, desc="Evaluating")
        
        for batch in pbar:
            # GSM8K 정답 추출 로직
            for i in range(len(batch["question"])):
                q_text = f"Question: {batch['question'][i]}\nAnswer:"
                inputs = self.tokenizer(q_text, return_tensors="pt").to(self.config.device)
                
                # 모델 고유의 generate (HCSF 반영) 사용 권장
                # 여기서는 간단히 모델 forward 결과를 바탕으로 평가
                outputs, _ = self.model(inputs.input_ids, mode='inference')
                pred_token = outputs[:, -1, :].argmax(-1)
                
                # 실제 구현 시에는 loop를 돌며 #### 까지 생성 후 숫자 추출 필요
                # (생략: GSM8K용 정교한 정답 추출 함수 extract_final_answer 사용)
                total += 1
        return correct / total

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.config.num_epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for i, batch in enumerate(pbar):
                total_loss, lm_loss, energy = self.train_step(batch)
                
                # Accumulation
                (total_loss / self.config.gradient_accumulation_steps).backward()
                
                if (i + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                pbar.set_postfix({
                    "LM": f"{lm_loss.item():.3f}",
                    "Energy": f"{energy.item():.1f}",
                    "Steps": f"{int(total_loss.item())}" # actual_steps 대용
                })

# =============================================================================
# 4. Main 실행
# =============================================================================

def main():
    config = TrainingConfig()
    tokenizer = GPT2Tokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드 및 HCSF 래핑
    base_model = GPT2LMHeadModel.from_pretrained(config.base_model)
    model = GPT2WithHCSF(
        base_model, 
        k_steps_max=config.k_steps_max,
        top_k=config.top_k,
        use_checkpointing=True # VRAM 절약 핵심
    )
    
    # 데이터 로더
    train_ds = GSM8KDataset(tokenizer, "train", subset_size=config.train_subset)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    # 트레이너 실행
    trainer = HCSFTrainer(model, tokenizer, config)
    trainer.train(train_loader)

if __name__ == "__main__":
    main()