import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_loader import get_dataloader
from collections import defaultdict
import os

from models import (
    GPT2WithSheafHead, 
    GPT2WithSheafAdapters, 
    GPT2WithLinearAdapters,
    GPT2WithRecurrentSheaf
)

def evaluate(model, dataloader, device, desc="Validation"):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=50256)
    
    total_ce = 0
    total_acc_correct = 0
    total_acc_total = 0
    
    # Diagnostics accumulation
    total_pre_energy = 0
    total_post_energy = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            diagnostics = {}
            
            # --- Forward Pass Handling ---
            if hasattr(model, "sheaf_layer"):
                refined, diagnostics = model.sheaf_layer(
                    model.backbone(input_ids, attention_mask=attention_mask).last_hidden_state, 
                    mode='train'
                )
                logits = model.lm_head(refined)
                
            elif isinstance(model, GPT2WithSheafAdapters):
                logits, diagnostics = model(input_ids, attention_mask=attention_mask)
            
            elif isinstance(model, GPT2WithRecurrentSheaf):
                logits, diagnostics = model(input_ids, attention_mask=attention_mask)
                
            elif isinstance(model, GPT2WithLinearAdapters):
                logits, _ = model(input_ids, attention_mask=attention_mask)
                
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # --- Loss & Accuracy ---
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            ce_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            preds = shift_logits.argmax(dim=-1)
            mask = (shift_labels != 50256)
            correct = (preds == shift_labels) & mask
            
            total_ce += ce_loss.item()
            total_acc_correct += correct.sum().item()
            total_acc_total += mask.sum().item()
            
            total_pre_energy += diagnostics.get("pre_energy", 0).item() if isinstance(diagnostics.get("pre_energy", 0), torch.Tensor) else diagnostics.get("pre_energy", 0)
            total_post_energy += diagnostics.get("post_energy", 0).item() if isinstance(diagnostics.get("post_energy", 0), torch.Tensor) else diagnostics.get("post_energy", 0)

    avg_ce = total_ce / len(dataloader)
    avg_acc = total_acc_correct / total_acc_total * 100 if total_acc_total > 0 else 0
    
    print(f"\n[{desc}] Final Results:")
    print(f"CE Loss       : {avg_ce:.4f}")
    print(f"Token Acc     : {avg_acc:.2f}%")
    if total_pre_energy > 0:
        avg_pre = total_pre_energy / len(dataloader)
        avg_post = total_post_energy / len(dataloader)
        print(f"Pre-Energy    : {avg_pre:.4f}")
        print(f"Post-Energy   : {avg_post:.4f}")
    print("-" * 30)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    data_path = f"data/tasks_1-20_v1-2/en-10k/qa15_basic-deduction_{'train' if args.mode != 'test' else 'test'}.txt"
    dataloader = get_dataloader(data_path, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    
    test_path = "data/tasks_1-20_v1-2/en-10k/qa15_basic-deduction_test.txt"
    test_loader = get_dataloader(test_path, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    
    # 2. Load Model
    print("Loading GPT-2...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    optimizer = None
    model = None

    if args.model_type == 'sheaf':
        print(f"Initializing Sheaf Model (Frozen Backbone, Steps={args.diffusion_steps})...")
        model = GPT2WithSheafHead(base_model.transformer, hidden_dim=768, freeze=True, diffusion_steps=args.diffusion_steps)
        model.to(device)
        model.lm_head = base_model.lm_head 
        optimizer = AdamW(model.sheaf_layer.parameters(), lr=args.lr)
        
    elif args.model_type == 'sheaf_finetune':
        print(f"Initializing Sheaf Model (Unfrozen Backbone)...")
        model = GPT2WithSheafHead(base_model.transformer, hidden_dim=768, freeze=False, diffusion_steps=args.diffusion_steps)
        model.to(device)
        model.lm_head = base_model.lm_head 
        optimizer = AdamW(model.parameters(), lr=args.lr)
        
    elif args.model_type == 'sheaf_adapter':
        print(f"Initializing Internal Sheaf Adapters (Layers {args.target_layers}, Win={args.window_size})...")
        model = GPT2WithSheafAdapters(base_model, hidden_dim=768, window_size=args.window_size, mlp_hidden=args.mlp_hidden, diffusion_steps=args.diffusion_steps, target_layers=args.target_layers)
        model.to(device)
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = AdamW(trainable_params, lr=args.lr)
        
    elif args.model_type == 'linear_adapter':
        print(f"Initializing Linear Adapters (Layers {args.target_layers})...")
        model = GPT2WithLinearAdapters(base_model, target_layers=args.target_layers)
        model.to(device)
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = AdamW(trainable_params, lr=args.lr)
    
    elif args.model_type == 'recurrent_sheaf':
        print(f"Initializing Recurrent Sheaf Engine (Layers {args.target_layers})...")
        model = GPT2WithRecurrentSheaf(base_model, target_layers=args.target_layers)
        model.to(device)
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params)}")
        optimizer = AdamW(trainable_params, lr=args.lr)
        
    elif args.model_type == 'finetune':
        print("Using Fine-tuned GPT-2 (Unfrozen)...")
        model = base_model
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else: 
        print("Using Baseline (Frozen GPT-2)...")
        model = base_model
        optimizer = None
        
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    if args.model_type == 'baseline':
        model.eval()
    else:
        model.train()
    
    # 3. Training Loop with Gradient Accumulation
    for epoch in range(args.epochs):
        metrics = defaultdict(float)
        chain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        batches = 0
        
        # Reset gradients initially
        if optimizer: optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            chain_lengths = batch.get("chain_length", torch.zeros(input_ids.size(0))).to(device)
            
            diagnostics = {}
            logits = None
            
            # --- Forward Pass ---
            if args.model_type in ['sheaf', 'sheaf_finetune']:
                refined_states, diagnostics = model(input_ids, attention_mask=attention_mask)
                logits = model.lm_head(refined_states)
            
            elif args.model_type in ['sheaf_adapter', 'linear_adapter', 'recurrent_sheaf']:
                logits, diagnostics = model(input_ids, attention_mask=attention_mask)
                
            elif args.model_type == 'finetune':
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
            else: 
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits

            # --- Loss Calculation ---
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss = ce_loss
            
            cons_loss = diagnostics.get("pre_energy", torch.tensor(0.0, device=device))
            if args.model_type in ['sheaf', 'sheaf_finetune', 'sheaf_adapter'] and args.lbda > 0:
                loss = ce_loss + args.lbda * cons_loss

            # --- Backward & Optimizer Step (Gradient Accumulation) ---
            if optimizer:
                # Normalize loss by accumulation steps
                loss = loss / args.grad_accum
                loss.backward()
                
                # Step only after accumulating enough gradients
                if (i + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # --- Metrics ---
            preds = shift_logits.argmax(dim=-1)
            mask = (shift_labels != tokenizer.pad_token_id)
            correct = (preds == shift_labels) & mask
            acc_total = mask.sum().item()
            acc_correct = correct.sum().item()
            
            metrics["total_ce"] += ce_loss.item()
            metrics["total_acc_correct"] += acc_correct
            metrics["total_acc_tokens"] += acc_total
            
            if "pre_energy" in diagnostics:
                pre_e = diagnostics["pre_energy"]
                metrics["total_pre_energy"] += pre_e.item() if isinstance(pre_e, torch.Tensor) else pre_e
                
                post_e = diagnostics.get("post_energy", 0)
                metrics["total_post_energy"] += post_e.item() if isinstance(post_e, torch.Tensor) else post_e
                
                up_n = diagnostics.get("update_norm", 0)
                metrics["total_update_norm"] += up_n.item() if isinstance(up_n, torch.Tensor) else up_n

            for k in range(input_ids.size(0)):
                c_len = int(chain_lengths[k].item())
                chain_stats[c_len]["correct"] += correct[k].sum().item()
                chain_stats[c_len]["total"] += mask[k].sum().item()

            batches += 1
            
            if (i + 1) % args.grad_accum == 0 and (batches // args.grad_accum) % 10 == 0:
                # Logging effective batches
                batch_acc = acc_correct / acc_total if acc_total > 0 else 0.0
                effective_batch = batches // args.grad_accum
                log_str = f"Epoch {epoch+1} | Step {effective_batch} | CE: {ce_loss.item():.4f}"
                if args.model_type in ['sheaf_adapter']:
                    log_str += f" | Cons: {cons_loss.item():.2f}"
                log_str += f" | Acc: {batch_acc*100:.2f}%"
                print(log_str)
            
            if args.max_steps > 0 and batches >= args.max_steps:
                break
        
        # --- Epoch Summary ---
        avg_ce = metrics["total_ce"] / batches
        avg_acc = metrics["total_acc_correct"] / metrics["total_acc_tokens"]
        
        print("\n" + "="*50)
        print(f"Epoch {epoch+1} Results - Model: {args.model_type}")
        print("="*50)
        print(f"CE Loss       : {avg_ce:.4f}")
        print(f"Token Acc     : {avg_acc*100:.2f}%")
        
        if metrics["total_pre_energy"] > 0:
            avg_pre = metrics["total_pre_energy"] / batches
            avg_post = metrics["total_post_energy"] / batches
            reduction = ((avg_pre - avg_post) / avg_pre) * 100 if avg_pre > 0 else 0
            print(f"Pre-Energy    : {avg_pre:.4f}")
            print(f"Post-Energy   : {avg_post:.4f}")
            print(f"Energy Reduct : {reduction:.2f}%")
            print(f"Update Norm   : {metrics['total_update_norm']/batches:.4f}")
            
        print("-" * 30)
        
        evaluate(model, test_loader, device, desc=f"Epoch {epoch+1} Test Set Eval")

    if args.save_path:
        save_dict = None
        if args.model_type == 'sheaf_adapter':
            save_dict = {k: v for k, v in model.state_dict().items() if 'sheaf_adapter' in k}
            save_dest = args.save_path.replace('.pt', '_adapter_only.pt')
        elif args.model_type == 'linear_adapter':
            save_dict = {k: v for k, v in model.state_dict().items() if 'adapter' in k}
            save_dest = args.save_path.replace('.pt', '_linear_adapter.pt')
        elif args.model_type == 'recurrent_sheaf':
            save_dict = {k: v for k, v in model.state_dict().items() if 'engine' in k}
            save_dest = args.save_path.replace('.pt', '_engine_only.pt')
        elif args.model_type == 'sheaf':
            save_dict = model.sheaf_layer.state_dict()
            save_dest = args.save_path
        else:
            save_dict = model.state_dict()
            save_dest = args.save_path
            
        torch.save(save_dict, save_dest)
        print(f"Model saved to {save_dest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_type", type=str, default="sheaf", 
                        choices=["baseline", "sheaf", "finetune", "sheaf_finetune", "sheaf_adapter", "linear_adapter", "recurrent_sheaf"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lbda", type=float, default=0.001)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="checkpoint.pt")
    parser.add_argument("--diffusion_steps", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--target_layers", nargs='+', type=int, default=[8, 9, 10, 11])
    
    # NEW ARGUMENT: Gradient Accumulation
    parser.add_argument("--grad_accum", type=int, default=1, help="Number of steps to accumulate gradients")
    
    args = parser.parse_args()
    train(args)