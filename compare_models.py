import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models import GPT2WithSheafHead
from data_loader import get_dataloader
import argparse

def evaluate(model, dataloader, device, model_name="Model"):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=50256) # Hardcoded pad_token_id for GPT2
    
    total_ce = 0
    total_acc = 0
    total_tokens = 0
    
    # Sheaf Diagnostics
    total_pre_energy = 0
    total_post_energy = 0
    total_update_norm = 0
    total_cos_sim = 0
    
    batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            diagnostics = {}
            
            # Forward
            if hasattr(model, "sheaf_layer"):
                # Sheaf Model
                refined, diagnostics = model(input_ids, attention_mask=attention_mask)
                logits = model.lm_head(refined)
            else:
                # Baseline
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Accuracy
            preds = shift_logits.argmax(dim=-1)
            mask = (shift_labels != 50256)
            correct = (preds == shift_labels) & mask
            
            total_ce += ce_loss.item()
            total_acc += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Diagnostics
            if hasattr(model, "sheaf_layer"):
                total_pre_energy += diagnostics.get("pre_energy", 0)
                total_post_energy += diagnostics.get("post_energy", 0)
                total_update_norm += diagnostics.get("update_norm", 0)
                total_cos_sim += diagnostics.get("cos_sim", 0)
                
            batches += 1
            if batches % 10 == 0:
                print(f"[{model_name}] Evaluating batch {batches}...", end="\r")

    avg_ce = total_ce / batches
    avg_acc = total_acc / total_tokens * 100
    
    print(f"\n[{model_name}] Evaluation Complete.")
    
    results = {
        "ce_loss": avg_ce,
        "accuracy": avg_acc
    }
    
    if hasattr(model, "sheaf_layer"):
        avg_pre = total_pre_energy / batches
        avg_post = total_post_energy / batches
        metrics = {
            "pre_energy": avg_pre if isinstance(avg_pre, float) else avg_pre.item(),
            "post_energy": avg_post if isinstance(avg_post, float) else avg_post.item(),
            "update_norm": (total_update_norm / batches) if isinstance(total_update_norm, float) else (total_update_norm / batches).item(),
            "cos_sim": (total_cos_sim / batches) if isinstance(total_cos_sim, float) else (total_cos_sim / batches).item()
        }
        results.update(metrics)
        
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data - Use Test Set for Comparison
    data_path = f"data/tasks_1-20_v1-2/en-10k/qa15_basic-deduction_test.txt"
    print(f"Loading Test Data from {data_path}")
    dataloader = get_dataloader(data_path, tokenizer, batch_size=32, max_length=128)
    
    # 1. Load Finetune Baseline
    print("\nLoading Finetuned Baseline (sheaf_layer_finetuned.pt)...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Load weights
    try:
        base_model.load_state_dict(torch.load("sheaf_layer_finetuned.pt", map_location=device))
        base_model.to(device)
        res_ft = evaluate(base_model, dataloader, device, "Finetune")
    except Exception as e:
        print(f"Failed to load Finetune checkpoint: {e}")
        res_ft = None

    # 2. Load Sheaf + Finetune
    print("\nLoading Sheaf+Finetune (sheaf_layer_sheaf_ft.pt)...")
    base_model_2 = GPT2LMHeadModel.from_pretrained("gpt2")
    sheaf_model = GPT2WithSheafHead(base_model_2.transformer, hidden_dim=768, freeze=False)
    sheaf_model.lm_head = base_model_2.lm_head
    
    try:
        # State dict contains "backbone...", "lm_head...", "sheaf_layer..."
        # Our model structure matches this if saved via model.state_dict()
        sheaf_model.load_state_dict(torch.load("sheaf_layer_sheaf_ft.pt", map_location=device))
        sheaf_model.to(device)
        res_sheaf = evaluate(sheaf_model, dataloader, device, "Sheaf+FT")
    except Exception as e:
        print(f"Failed to load Sheaf checkpoint: {e}")
        res_sheaf = None

    # Compare
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Finetune Baseline':<20} | {'Sheaf + Finetune':<20}")
    print("-" * 60)
    
    metrics = ["ce_loss", "accuracy", "pre_energy", "post_energy", "update_norm", "cos_sim"]
    
    for m in metrics:
        v1 = res_ft.get(m, "N/A") if res_ft else " Err"
        v2 = res_sheaf.get(m, "N/A") if res_sheaf else " Err"
        
        if isinstance(v1, float): v1 = f"{v1:.4f}"
        if isinstance(v2, float): v2 = f"{v2:.4f}"
        
        print(f"{m:<20} | {v1:<20} | {v2:<20}")
    print("="*60)

if __name__ == "__main__":
    main()
