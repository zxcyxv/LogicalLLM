import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Tuple, Optional, Dict

# =============================================================================
# 1. 수치 연산 커널
# =============================================================================
def apply_householder_prenorm(h: torch.Tensor, v_unit: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(v_unit * h, dim=-1, keepdim=True)
    return h - 2.0 * v_unit * dot

class EdgeLogicMLP(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 2 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.pos_emb = nn.Embedding(2048, d_model)
        # 초기 변환이 너무 크지 않도록 가중치 초기화 조절
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, h_src, h_tgt, rel_pos):
        pos_feat = self.pos_emb(torch.clamp(rel_pos + 1024, 0, 2047))
        x = torch.cat([h_src, h_tgt, pos_feat], dim=-1)
        v = self.mlp(x)
        return F.normalize(v, p=2, dim=-1, eps=1e-8)

# =============================================================================
# 2. HCSF Engine (수치 안정성 강화 버전)
# =============================================================================
class HCSFEngine(nn.Module):
    def __init__(self, d_model, k_steps_max=5, eta=0.1, lam=0.01, top_k=16):
        super().__init__()
        self.d_model = d_model
        self.k_steps_max = k_steps_max
        self.eta = nn.Parameter(torch.tensor(eta))
        self.lam = lam
        self.top_k = top_k
        self.edge_logic = EdgeLogicMLP(d_model)

    def build_global_sparse_graph(self, attn, seq_len, batch_size, device):
        all_src, all_tgt, all_weight, all_rel = [], [], [], []
        
        # 1. Causal Chain Edges (i -> i-1)
        if seq_len > 1:
            src = torch.arange(1, seq_len, device=device)
            tgt = torch.arange(seq_len - 1, device=device)
            all_src.append(src.unsqueeze(0).expand(batch_size, -1))
            all_tgt.append(tgt.unsqueeze(0).expand(batch_size, -1))
            all_rel.append(torch.full((batch_size, seq_len - 1), -1, device=device))
            # 체인 가중치는 기본 1.0으로 시작 (나중에 정규화)
            all_weight.append(torch.ones(batch_size, seq_len - 1, device=device))
        
        # 2. Top-K Attention Edges
        if self.top_k > 0 and attn is not None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device), -2).bool()
            masked_attn = attn.clone()
            masked_attn[:, ~mask] = -1e4 # Softmax를 위해 너무 작지 않은 값 사용
            
            k_actual = min(self.top_k, max(1, seq_len - 2))
            topk_vals, topk_idx = torch.topk(masked_attn, k=k_actual, dim=-1)
            
            # [핵심 수정] neighbors(K) 차원에 대해 정규화된 가중치 계산
            topk_weights = F.softmax(topk_vals, dim=-1) # [B, L, K]
            
            for k in range(k_actual):
                src = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                all_src.append(src)
                all_tgt.append(topk_idx[:, :, k])
                all_weight.append(topk_weights[:, :, k])
                all_rel.append(topk_idx[:, :, k] - src)

        # 모든 에지 가중치 결합 및 노드별 정규화 (에너지가 폭주하지 않도록)
        src_cat = torch.cat(all_src, 1)
        tgt_cat = torch.cat(all_tgt, 1)
        w_cat = torch.cat(all_weight, 1)
        rel_cat = torch.cat(all_rel, 1)
        
        # 노드별 가중치 합이 1이 되도록 조정
        w_sum = w_cat.sum(dim=1, keepdim=True) + 1e-8
        w_cat = w_cat / w_sum

        return src_cat, tgt_cat, w_cat, rel_cat

    def compute_energy_and_grad(self, h, h0, src_idx, tgt_idx, weights, v_ij, v_ji):
        B, E, D = v_ij.shape
        b_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(-1, E)
        
        # Householder 변환 적용
        p_ij = apply_householder_prenorm(h[b_idx, src_idx], v_ij)
        p_ji = apply_householder_prenorm(h[b_idx, tgt_idx], v_ji)
        
        # 불일치(Disagreement) 계산
        diff = p_ij - p_ji
        
        # [핵심 수정] 에너지 스케일링: 단순히 sum이 아니라 전체 요소에 대한 mean으로 접근
        # (diff**2).mean()은 이미 모든 차원에 대해 정규화된 값임
        weighted_diff_sq = (diff**2) * weights.unsqueeze(-1)
        energy = 0.5 * weighted_diff_sq.mean() * 100 # 시각화를 위해 100배 스케일업 (옵션)
        
        # 기울기 계산 (정규화된 가중치 반영)
        grad_delta = diff * weights.unsqueeze(-1)
        grad_src = apply_householder_prenorm(grad_delta, v_ij)
        grad_tgt = -apply_householder_prenorm(grad_delta, v_ji)
        
        grad = torch.zeros_like(h)
        grad.scatter_add_(1, src_idx.unsqueeze(-1).expand(-1, -1, D), grad_src)
        grad.scatter_add_(1, tgt_idx.unsqueeze(-1).expand(-1, -1, D), grad_tgt)
        
        # Anchor 항과 결합
        total_grad = grad + self.lam * (h - h0)
        return energy, total_grad

    def _step(self, h, h0, src_idx, tgt_idx, weights, v_ij, v_ji, eta):
        _, grad = self.compute_energy_and_grad(h, h0, src_idx, tgt_idx, weights, v_ij, v_ji)
        return h - torch.abs(eta) * grad

    def forward(self, h, attn=None):
        B, L, D = h.shape
        src_idx, tgt_idx, weights, rel_pos = self.build_global_sparse_graph(attn, L, B, h.device)
        
        h0 = h.detach().clone()
        E = src_idx.shape[1]
        b_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(-1, E)

        v_ij = self.edge_logic(h0[b_idx, src_idx], h0[b_idx, tgt_idx], rel_pos)
        v_ji = self.edge_logic(h0[b_idx, tgt_idx], h0[b_idx, src_idx], -rel_pos)

        for k in range(self.k_steps_max):
            h = checkpoint(self._step, h, h0, src_idx, tgt_idx, weights, v_ij, v_ji, self.eta, use_reentrant=False)
        
        post_energy, _ = self.compute_energy_and_grad(h, h0, src_idx, tgt_idx, weights, v_ij, v_ji)
        return h, {"post_energy": post_energy}

# =============================================================================
# 3. GPT-2 Wrapper
# =============================================================================
class HCSFAdapterBlock(nn.Module):
    def __init__(self, original_block, config, k_steps=5, eta=0.1, lam=0.01):
        super().__init__()
        self.original_block = original_block
        self.hcsf_engine = HCSFEngine(config.n_embd, k_steps_max=k_steps, eta=eta, lam=lam)
        self.latest_energy = torch.tensor(0.0)

    def forward(self, hidden_states, *args, **kwargs):
        kwargs['output_attentions'] = True
        outputs = self.original_block(hidden_states, *args, **kwargs)
        h = outputs[0]
        attn = outputs[1].mean(1) if len(outputs) > 1 and outputs[1] is not None else None

        h_refined, diag = self.hcsf_engine(h, attn)
        self.latest_energy = diag["post_energy"]
        return (h_refined,) + outputs[1:]

class GPT2HCSFAdapters(nn.Module):
    def __init__(self, gpt2, num_layers=4, k_steps=5, eta=0.1, lam=0.01):
        super().__init__()
        self.gpt2 = gpt2
        for p in self.gpt2.parameters(): p.requires_grad = False
        
        total_layers = len(self.gpt2.transformer.h)
        self.target_indices = list(range(total_layers - num_layers, total_layers))
        for i in self.target_indices:
            orig = self.gpt2.transformer.h[i]
            self.gpt2.transformer.h[i] = HCSFAdapterBlock(orig, gpt2.config, k_steps=k_steps, eta=eta, lam=lam)
        
        for n, m in self.gpt2.named_modules():
            if isinstance(m, nn.LayerNorm) or "ln_" in n:
                for p in m.parameters(): p.requires_grad = True

    def forward(self, ids, mask):
        res = self.gpt2(ids, attention_mask=mask)
        # 텐서 리스트를 하나의 텐서로 스택한 후 평균 계산 (그래프 보존)
        energies = torch.stack([self.gpt2.transformer.h[i].latest_energy for i in self.target_indices])
        return res.logits, energies.mean()

# =============================================================================
# 4. Dataset & Training
# =============================================================================
class GSMDataset(Dataset):
    def __init__(self, tokenizer, subset=2000):
        self.tokenizer = tokenizer
        ds = load_dataset("gsm8k", "main", split="train")
        self.data = ds.select(range(min(subset, len(ds))))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        q_text, a_text = f"Question: {item['question']}\nAnswer:", f" {item['answer']}{self.tokenizer.eos_token}"
        q_len = len(self.tokenizer.encode(q_text))
        enc = self.tokenizer(q_text + a_text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        labels = enc["input_ids"].clone().squeeze(0)
        labels[:q_len] = -100
        labels[enc["attention_mask"].squeeze(0) == 0] = -100
        return {"ids": enc["input_ids"].squeeze(0), "mask": enc["attention_mask"].squeeze(0), "labels": labels}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.0) # 에너지가 정규화되었으므로 1.0 권장
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    device = "cuda"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model = GPT2HCSFAdapters(base_model, num_layers=4, k_steps=5, eta=args.eta, lam=args.lam).to(device)
    
    loader = DataLoader(GSMDataset(tokenizer), batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    
    model.train()
    for epoch in range(3):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            ids, mask, labels = batch["ids"].to(device), batch["mask"].to(device), batch["labels"].to(device)
            logits, sheaf_loss = model(ids, mask)
            lm_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            
            # 수치 안정성을 위해 log(1 + E) 사용
            total_loss = (lm_loss + args.gamma * torch.log(1 + sheaf_loss)) / args.accum_steps
            total_loss.backward()
            
            if (i + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_postfix({"LM": f"{lm_loss.item():.2f}", "Energy": f"{sheaf_loss.item():.4f}"})

if __name__ == "__main__":
    train()