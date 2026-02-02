import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import defaultdict
import os

from model_v2 import (
    GPT2WithRecurrentSheaf,
    GPT2WithDeltaRule,
    ModelConfig,
    create_model,
    count_trainable_params,
)
from proofwriter_loader import get_proofwriter_loader


def evaluate(model, dataloader, device, desc="Validation", use_amp=False):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=50256)

    total_ce = 0.0
    total_acc_correct = 0
    total_acc_total = 0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=use_amp):
                if isinstance(model, (GPT2WithRecurrentSheaf, GPT2WithDeltaRule)):
                    logits, _ = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                ce_loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            preds = shift_logits.argmax(dim=-1)
            mask = (shift_labels != 50256)
            correct = (preds == shift_labels) & mask

            # Accumulate without .item() in inner loop
            total_ce += ce_loss.detach()
            total_acc_correct += correct.sum().detach()
            total_acc_total += mask.sum().detach()
            batches += 1

    # Only call .item() at the end
    avg_ce = (total_ce / batches).item()
    avg_acc = (total_acc_correct / total_acc_total * 100).item() if total_acc_total > 0 else 0

    print(f"\n[{desc}] Final Results:")
    print(f"CE Loss       : {avg_ce:.4f}")
    print(f"Token Acc     : {avg_acc:.2f}%")
    print("-" * 30)

    return {"ce_loss": avg_ce, "accuracy": avg_acc}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load ProofWriter Data
    print(f"\nLoading ProofWriter data (depth={args.max_depth})...")
    train_loader = get_proofwriter_loader(
        tokenizer,
        split="train",
        max_depth=args.max_depth,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        cache_tokenization=True,
    )

    test_loader = get_proofwriter_loader(
        tokenizer,
        split="validation",
        max_depth=args.max_depth,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False,
        cache_tokenization=True,
    )

    # 3. Get Model Configuration
    config = ModelConfig.from_name(args.base_model)
    target_layers = args.target_layers if args.target_layers else config.target_layers

    print(f"\nModel Config: {args.base_model}")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Target Layers: {target_layers}")

    # 4. Create Model
    model, config = create_model(
        model_type=args.model_type,
        base_model_name=args.base_model,
        target_layers=target_layers,
        device=device,
    )

    trainable_params = count_trainable_params(model)
    print(f"Trainable Parameters: {trainable_params:,}")

    # 5. Setup Optimizer
    optimizer = None
    if args.model_type != 'baseline':
        trainable_params_list = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = AdamW(trainable_params_list, lr=args.lr)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 6. Setup AMP (Mixed Precision)
    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")

    # 7. Training Loop
    effective_batch = args.batch_size * args.grad_accum
    print(f"\nStarting Training...")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Grad Accum: {args.grad_accum}")
    print(f"  Effective Batch: {effective_batch}")
    print(f"  Epochs: {args.epochs}")

    if args.model_type == 'baseline':
        model.eval()
    else:
        model.train()

    for epoch in range(args.epochs):
        # Use tensors for accumulation (avoid .item() overhead)
        running_ce = torch.tensor(0.0, device=device)
        running_correct = torch.tensor(0, device=device, dtype=torch.long)
        running_total = torch.tensor(0, device=device, dtype=torch.long)
        batches = 0

        if optimizer:
            optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with optional AMP
            with autocast(enabled=use_amp):
                if isinstance(model, (GPT2WithRecurrentSheaf, GPT2WithDeltaRule)):
                    logits, _ = model(input_ids, attention_mask=attention_mask)
                else:
                    with torch.no_grad():
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            # Backward with gradient accumulation
            if optimizer:
                norm_loss = loss / args.grad_accum
                scaler.scale(norm_loss).backward()

                if (i + 1) % args.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            # Metrics (no .item() calls)
            with torch.no_grad():
                preds = shift_logits.argmax(dim=-1)
                mask = (shift_labels != tokenizer.pad_token_id)
                correct = (preds == shift_labels) & mask

                running_ce += loss.detach()
                running_correct += correct.sum()
                running_total += mask.sum()

            batches += 1

            # Logging (infrequent .item() calls)
            if (i + 1) % args.grad_accum == 0 and (batches // args.grad_accum) % args.log_interval == 0:
                eff_step = batches // args.grad_accum
                batch_acc = (correct.sum() / mask.sum() * 100).item() if mask.sum() > 0 else 0.0
                print(f"Epoch {epoch+1} | Step {eff_step} | CE: {loss.item():.4f} | Acc: {batch_acc:.2f}%")

            if args.max_steps > 0 and batches >= args.max_steps:
                break

        # Epoch Summary (only .item() at epoch end)
        avg_ce = (running_ce / batches).item()
        avg_acc = (running_correct / running_total * 100).item() if running_total > 0 else 0

        print("\n" + "=" * 50)
        print(f"Epoch {epoch+1} Results - Model: {args.model_type} ({args.base_model})")
        print("=" * 50)
        print(f"CE Loss       : {avg_ce:.4f}")
        print(f"Token Acc     : {avg_acc:.2f}%")
        print("-" * 30)

        # Evaluate on test set
        evaluate(model, test_loader, device, desc=f"Epoch {epoch+1} Validation", use_amp=use_amp)

    # 8. Save Model
    if args.save_path:
        save_dict = None
        if args.model_type == 'recurrent_sheaf':
            save_dict = {k: v for k, v in model.state_dict().items() if 'engine' in k}
            save_dest = args.save_path.replace('.pt', '_sheaf_engine.pt')
            print(f"Saved Recurrent Sheaf Engine to {save_dest}")

        elif args.model_type == 'delta_rule':
            save_dict = {k: v for k, v in model.state_dict().items() if 'engine' in k}
            save_dest = args.save_path.replace('.pt', '_delta_engine.pt')
            print(f"Saved Delta Rule Engine to {save_dest}")

        else:
            save_dest = args.save_path

        if save_dict:
            torch.save(save_dict, save_dest)
        elif args.model_type != 'baseline':
            torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on ProofWriter dataset")

    # Model configuration
    parser.add_argument("--model_type", type=str, default="recurrent_sheaf",
                        choices=["baseline", "recurrent_sheaf", "delta_rule"],
                        help="Model type to train")
    parser.add_argument("--base_model", type=str, default="gpt2-medium",
                        choices=["gpt2", "gpt2-small", "gpt2-medium"],
                        help="Base GPT-2 model to use")
    parser.add_argument("--target_layers", nargs='+', type=int, default=None,
                        help="Target layers for engine (default: auto based on model)")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="Disable automatic mixed precision")

    # Data configuration
    parser.add_argument("--max_depth", type=int, default=5,
                        help="ProofWriter question depth (QDep)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    # Other
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 for full epoch)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N effective steps")
    parser.add_argument("--save_path", type=str, default="checkpoint.pt",
                        help="Path to save model checkpoint")
    parser.add_argument("--mode", type=str, default="train",
                        help="Mode (train/test)")

    args = parser.parse_args()
    train(args)
