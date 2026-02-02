import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_loader import get_babi_loaders
import argparse
from models import GPT2WithSheafHead, GPT2WithSheafAdapters, GPT2WithLinearAdapters

def evaluate(model, dataloader, device, args):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=50256)  # GPT-2 pad token
    
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward
            if hasattr(model, "sheaf_layer"):
                # Sheaf Model (training mode=False for inference usually, but here we want to measure Energy which requires computing it.
                # models.py forward(mode='train') returns diagnostics. 'inference' does auto-regressive generation.
                # For evaluation, we use mode='train' to get diagnostics but don't update weights.
                from models import GPT2WithSheafHead
                refined, diagnostics = model.sheaf_layer(model.backbone(input_ids, attention_mask=attention_mask).last_hidden_state, mode='train')
                logits = model.lm_head(refined)
            elif hasattr(model, "target_layers"):
                 # Phase 2: Internal Adapters (Sheaf or Linear)
                 logits, diagnostics = model(input_ids, attention_mask=attention_mask)
            else:
                # Baseline
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            
            # Compute accuracy
            preds = shift_logits.argmax(dim=-1)
            mask = (shift_labels != 50256)
            correct = ((preds == shift_labels) & mask).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return avg_loss, accuracy

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and data
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_loader, test_loader = get_babi_loaders(
        task_id=args.task_id,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        max_length=128
    )
    
    # Load base model
    print("Loading GPT-2...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Initialize model based on type
    if args.model_type == 'baseline':
        # Frozen GPT-2 baseline
        model = base_model
        for param in model.parameters():
            param.requires_grad = False
        model.to(device)
        optimizer = None  # No training
        
    elif args.model_type == 'finetune':
        # Full fine-tuning
        model = base_model
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        
    elif args.model_type == 'sheaf':
        # Sheaf head only (frozen backbone)
        from models import GPT2WithSheafHead
        model = GPT2WithSheafHead(base_model, hidden_dim=768, window_size=16, mlp_hidden=128, diffusion_steps=args.diffusion_steps)
        model.to(device)
        # Only train sheaf layer and lm_head
        trainable_params = list(model.sheaf_layer.parameters()) + list(model.lm_head.parameters())
        optimizer = AdamW(trainable_params, lr=args.lr)
        
    elif args.model_type == 'sheaf_finetune':
        # Sheaf head + fine-tune backbone
        from models import GPT2WithSheafHead
        model = GPT2WithSheafHead(base_model, hidden_dim=768, window_size=16, mlp_hidden=128, diffusion_steps=args.diffusion_steps)
        # Unfreeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = True
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        
    elif args.model_type == 'sheaf_adapter':
        from models import GPT2WithSheafAdapters
        print(f"Initializing Internal Sheaf Adapters (Layers 8-11, Steps={args.diffusion_steps})...")
        model = GPT2WithSheafAdapters(base_model, hidden_dim=768, window_size=16, mlp_hidden=128, diffusion_steps=args.diffusion_steps)
        model.to(device)
        # Verify and Filter Frozen Params
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params)}")
        optimizer = AdamW(trainable_params, lr=args.lr)
        
    elif args.model_type == 'linear_adapter':
        from models import GPT2WithLinearAdapters
        print(f"Initializing Linear Adapters (Layers 8-11, Baseline)...")
        model = GPT2WithLinearAdapters(base_model, hidden_dim=768)
        model.to(device)
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params)}")
        optimizer = AdamW(trainable_params, lr=args.lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Debug: Check labels
    sample_batch = next(iter(train_loader))
    print(f"DEBUG: Pad Token ID: {tokenizer.pad_token_id}")
    print(f"DEBUG: Labels shape: {sample_batch[2].shape}")
    print(f"DEBUG: Labels sample: {sample_batch[2][0, :20]}")
    print(f"DEBUG: Labels end: {sample_batch[2][0, -20:]}")
    print(f"DEBUG: Num pads in batch: {(sample_batch[2] == tokenizer.pad_token_id).sum()} / {sample_batch[2].numel()}")
    
    # Training loop
    for epoch in range(args.epochs):
        if args.model_type == 'baseline':
            # Skip training for baseline
            break
            
        model.train()
        metrics = {
            "total_loss": 0,
            "total_ce_loss": 0,
            "total_cons_loss": 0,
            "total_correct": 0,
            "total_tokens": 0,
            "total_pre_energy": 0,
            "total_post_energy": 0,
            "total_update_norm": 0,
            "total_cos_sim": 0,
        }
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            if args.max_steps and batch_idx >= args.max_steps:
                break
                
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if args.model_type in ['sheaf', 'sheaf_finetune']:
                # Forward
                refined, diagnostics = model.sheaf_layer(model.backbone(input_ids, attention_mask=attention_mask).last_hidden_state, mode='train')
                logits = model.lm_head(refined)
                
                # Shift logits and labels for CLM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                ce_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Consistency loss (pre-energy)
                cons_loss = diagnostics.get("pre_energy", torch.tensor(0.0, device=device))
                
                # Total loss
                loss = ce_loss + args.lbda * cons_loss
                
                loss.backward()
                optimizer.step()
                
            elif args.model_type in ['sheaf_adapter', 'linear_adapter']:
                # Forward (Adapter returns logits directly)
                logits, diagnostics = model(input_ids, attention_mask=attention_mask)
                
                # Shift logits and labels for CLM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                ce_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Loss = CE + Lambda * Pre-Energy (only for sheaf_adapter)
                if args.model_type == 'sheaf_adapter':
                    cons_loss = diagnostics.get("pre_energy", torch.tensor(0.0, device=device))
                    loss = ce_loss + args.lbda * cons_loss
                else:  # linear_adapter
                    cons_loss = torch.tensor(0.0, device=device)
                    loss = ce_loss
                
                loss.backward()
                optimizer.step()

            elif args.model_type == 'finetune':
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                loss = ce_loss
                loss.backward()
                optimizer.step()
                cons_loss = torch.tensor(0.0, device=device)

            else:
                # Baseline
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    ce_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = ce_loss
                    cons_loss = torch.tensor(0.0, device=device)
            
            # Metrics
            metrics["total_loss"] += loss.item()
            metrics["total_ce_loss"] += ce_loss.item()
            metrics["total_cons_loss"] += cons_loss.item() if isinstance(cons_loss, torch.Tensor) else cons_loss
            
            # Accuracy
            preds = shift_logits.argmax(dim=-1)
            mask = (shift_labels != tokenizer.pad_token_id)
            correct = ((preds == shift_labels) & mask).sum().item()
            metrics["total_correct"] += correct
            metrics["total_tokens"] += mask.sum().item()
            
            # Sheaf-specific metrics
            if args.model_type in ['sheaf', 'sheaf_finetune', 'sheaf_adapter']:
                pre_e = diagnostics.get("pre_energy", 0)
                metrics["total_pre_energy"] += pre_e.item() if isinstance(pre_e, torch.Tensor) else pre_e
                
                post_e = diagnostics.get("post_energy", 0)
                metrics["total_post_energy"] += post_e.item() if isinstance(post_e, torch.Tensor) else post_e
                
                up_n = diagnostics.get("update_norm", 0)
                metrics["total_update_norm"] += up_n.item() if isinstance(up_n, torch.Tensor) else up_n
                
                c_sim = diagnostics.get("cos_sim", 0)
                metrics["total_cos_sim"] += c_sim.item() if isinstance(c_sim, torch.Tensor) else c_sim
            
            # Logging
            if (batch_idx + 1) % 10 == 0:
                avg_loss = metrics["total_ce_loss"] / (batch_idx + 1)
                avg_acc = 100.0 * metrics["total_correct"] / metrics["total_tokens"] if metrics["total_tokens"] > 0 else 0.0
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} | CE: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")
        
        # Epoch summary
        num_batches = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
        avg_loss = metrics["total_ce_loss"] / num_batches
        avg_cons = metrics["total_cons_loss"] / num_batches
        avg_acc = 100.0 * metrics["total_correct"] / metrics["total_tokens"] if metrics["total_tokens"] > 0 else 0.0
        
        print("\n" + "="*50)
        print(f"Epoch {epoch+1} Results - Model: {args.model_type}")
        print("="*50)
        print(f"CE Loss       : {avg_loss:.4f}")
        print(f"Token Acc     : {avg_acc:.2f}%")
        
        if args.model_type in ['sheaf', 'sheaf_finetune', 'sheaf_adapter']:
            avg_pre = metrics["total_pre_energy"] / num_batches
            avg_post = metrics["total_post_energy"] / num_batches
            avg_update = metrics["total_update_norm"] / num_batches
            avg_cos = metrics["total_cos_sim"] / num_batches
            print(f"Cons Loss     : {avg_cons:.4f}")
            print(f"Pre-Energy    : {avg_pre:.4f}")
            print(f"Post-Energy   : {avg_post:.4f}")
            print(f"Update Norm   : {avg_update:.4f}")
            print(f"Cos Sim       : {avg_cos:.4f}")
        
        print("-" * 30)
        
        # Chain length analysis
        print("Chain Length Analysis (Acc):")
        for chain_len in [2]:  # Can extend to [2, 3, 4] if needed
            chain_correct = 0
            chain_tokens = 0
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
                if args.max_steps and batch_idx >= args.max_steps:
                    break
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    if hasattr(model, "sheaf_layer"):
                        refined, _ = model.sheaf_layer(model.backbone(input_ids, attention_mask=attention_mask).last_hidden_state, mode='inference')
                        logits = model.lm_head(refined)
                    elif hasattr(model, "target_layers"):
                        logits, _ = model(input_ids, attention_mask=attention_mask)
                    else:
                        outputs = model(input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    preds = shift_logits.argmax(dim=-1)
                    mask = (shift_labels != tokenizer.pad_token_id)
                    correct = ((preds == shift_labels) & mask).sum().item()
                    chain_correct += correct
                    chain_tokens += mask.sum().item()
            
            chain_acc = 100.0 * chain_correct / chain_tokens if chain_tokens > 0 else 0.0
            print(f"Len {chain_len}: {chain_acc:.1f}% ({chain_tokens} toks)")
        
        print("="*50)
        print("\n")
    
    # Final evaluation on test set
    print("[Epoch 1 Test Set Eval] Final Results:")
    test_loss, test_acc = evaluate(model, test_loader, device, args)
    print(f"CE Loss       : {test_loss:.4f}")
    print(f"Token Acc     : {test_acc:.2f}%")
    print("-" * 30)
    
    # Save model
    if args.save_path:
        if args.model_type == 'sheaf_adapter':
            adapter_state = {k: v for k, v in model.state_dict().items() if 'sheaf_adapter' in k}
            save_dest = args.save_path.replace('.pt', '_adapter_only.pt')
            torch.save(adapter_state, save_dest)
            print(f"Saved Adapter weights only ({len(adapter_state)} keys) to {save_dest}")
        elif args.model_type == 'linear_adapter':
            adapter_state = {k: v for k, v in model.state_dict().items() if 'adapter' in k}
            save_dest = args.save_path.replace('.pt', '_linear_adapter.pt')
            torch.save(adapter_state, save_dest)
            print(f"Saved Linear Adapter weights only ({len(adapter_state)} keys) to {save_dest}")
        elif args.model_type == 'sheaf':
            torch.save(model.sheaf_layer.state_dict(), args.save_path)
            print(f"Saved Sheaf Layer to {args.save_path}")
        else:
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved model to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_type", type=str, default="sheaf", choices=["baseline", "sheaf", "finetune", "sheaf_finetune", "sheaf_adapter", "linear_adapter"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lbda", type=float, default=0.002)
    parser.add_argument("--task_id", type=int, default=15)
    parser.add_argument("--save_path", type=str, default="sheaf_layer.pt")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=1, help="Number of diffusion steps for Sheaf layer")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
