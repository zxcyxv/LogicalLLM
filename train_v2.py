import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import defaultdict
import os
from tqdm import tqdm

from model_v2 import (
    GPT2WithRecurrentSheaf,
    GPT2WithDeltaNet,
    ModelConfig,
    create_model,
    count_trainable_params,
)
from proofwriter_loader import get_proofwriter_loader
from gsm8k_loader import get_gsm8k_loader


def evaluate(model, dataloader, device, desc="Validation", use_amp=False):
    model.eval()
    # Use reduction='sum' to manually handle division by zero
    criterion = nn.CrossEntropyLoss(ignore_index=50256, reduction='sum')

    total_ce = 0.0
    total_acc_correct = 0
    total_acc_total = 0
    batches = 0

    # Diagnostics accumulators
    total_energy = 0.0
    total_gate = 0.0
    total_rho_complexity = 0.0
    has_diagnostics = False

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            answer_start = batch["answer_start"].to(device)  # [B]

            with autocast('cuda', enabled=use_amp):
                if isinstance(model, (GPT2WithRecurrentSheaf, GPT2WithDeltaNet)):
                    logits, diagnostics = model(input_ids, attention_mask=attention_mask)
                    if diagnostics:
                        has_diagnostics = True
                        total_energy += diagnostics.get("avg_energy", 0.0)
                        total_gate += diagnostics.get("avg_gate", 0.0)
                        total_rho_complexity += diagnostics.get("rho_complexity", 0.0)
                else:
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Create answer-only mask: only count tokens after answer_start
                # Note: After shift, we need to adjust answer_start by -1
                B, T = shift_labels.shape
                positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]

                # Adjust answer_start for shift operation (shift removes first position)
                # Also ensure it's within bounds
                adjusted_answer_start = torch.clamp(answer_start - 1, min=0, max=T-1)

                answer_mask = (positions >= adjusted_answer_start.unsqueeze(1)) & (shift_labels != 50256)

                # CE loss only on answer tokens (with safe division)
                num_valid_tokens = answer_mask.sum()

                if num_valid_tokens > 0:
                    ce_loss_sum = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        torch.where(answer_mask, shift_labels, torch.tensor(50256, device=device)).view(-1)
                    )
                    ce_loss = ce_loss_sum / num_valid_tokens
                else:
                    # Skip this batch if no valid answer tokens
                    ce_loss = torch.tensor(0.0, device=device)
                    continue

            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels) & answer_mask

            # Accumulate without .item() in inner loop
            total_ce += ce_loss.detach()
            total_acc_correct += correct.sum().detach()
            total_acc_total += answer_mask.sum().detach()
            batches += 1

    # Only call .item() at the end
    avg_ce = (total_ce / batches).item()
    avg_acc = (total_acc_correct / total_acc_total * 100).item() if total_acc_total > 0 else 0

    print(f"\n[{desc}] Final Results:")
    print(f"CE Loss (ans) : {avg_ce:.4f}")
    print(f"Answer Acc    : {avg_acc:.2f}%")

    # Print diagnostics if available
    if has_diagnostics and batches > 0:
        print(f"\n[NSD Diagnostics]")
        print(f"  Logical Energy      : {total_energy / batches:.4f}  (Lower = Better prediction)")
        print(f"  Gate Activity       : {total_gate / batches:.4f}  (Lower = More selective)")
        print(f"  Rho Complexity      : {total_rho_complexity / batches:.4f}  (Higher = More non-trivial)")

    print("-" * 30)

    return {
        "ce_loss": avg_ce,
        "accuracy": avg_acc,
        "avg_energy": total_energy / batches if has_diagnostics and batches > 0 else 0.0,
        "avg_gate": total_gate / batches if has_diagnostics and batches > 0 else 0.0,
        "rho_complexity": total_rho_complexity / batches if has_diagnostics and batches > 0 else 0.0,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset
    if args.dataset == "gsm8k":
        print("\n[GSM8K Dataset - Grade School Math]")
        train_loader = get_gsm8k_loader(
            tokenizer,
            split="train",
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=True,
            cache_tokenization=True,
        )
        test_loader = get_gsm8k_loader(
            tokenizer,
            split="test",
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=False,
            cache_tokenization=True,
        )
    elif args.dataset == "proofwriter":
        # ProofWriter Data (OOD Setting: Train Shallow, Test Deep)
        if args.ood:
            print("\n[ProofWriter OOD MODE] Train: Shallow (depth 0-2) / Test: Deep (depth 5)")
            train_loader = get_proofwriter_loader(
                tokenizer,
                split="train",
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=True,
                cache_tokenization=True,
                depth_filter='shallow',  # Depth 0, 1, 2 only
            )
            test_loader = get_proofwriter_loader(
                tokenizer,
                split="validation",
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=False,
                cache_tokenization=True,
                depth_filter='deep',  # Depth 5 only (OOD)
            )
        else:
            print(f"\n[ProofWriter Standard Mode] Train & Test on depth={args.max_depth}")
            train_loader = get_proofwriter_loader(
                tokenizer,
                split="train",
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=True,
                cache_tokenization=True,
                exact_depth=args.max_depth,
            )
            test_loader = get_proofwriter_loader(
                tokenizer,
                split="validation",
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=False,
                cache_tokenization=True,
                exact_depth=args.max_depth,
            )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose 'gsm8k' or 'proofwriter'")

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

    # Use reduction='sum' to manually handle division by zero
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')

    # 6. Setup AMP (Mixed Precision)
    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)
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

        # Diagnostics accumulators
        running_energy = 0.0
        running_gate = 0.0
        running_rho_complexity = 0.0
        has_diagnostics = False

        if optimizer:
            optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            answer_start = batch["answer_start"].to(device)  # [B]

            # Forward pass with optional AMP
            with autocast('cuda', enabled=use_amp):
                if isinstance(model, (GPT2WithRecurrentSheaf, GPT2WithDeltaNet)):
                    logits, diagnostics = model(input_ids, attention_mask=attention_mask)
                    if diagnostics:
                        has_diagnostics = True
                        running_energy += diagnostics.get("avg_energy", 0.0)
                        running_gate += diagnostics.get("avg_gate", 0.0)
                        running_rho_complexity += diagnostics.get("rho_complexity", 0.0)
                else:
                    with torch.no_grad():
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Create answer-only mask
                # Note: After shift, we need to adjust answer_start by -1
                B, T = shift_labels.shape
                positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

                # Adjust answer_start for shift operation
                adjusted_answer_start = torch.clamp(answer_start - 1, min=0, max=T-1)

                answer_mask = (positions >= adjusted_answer_start.unsqueeze(1)) & (shift_labels != tokenizer.pad_token_id)

                # Loss on answer tokens only (with safe division)
                num_valid_tokens = answer_mask.sum()

                if num_valid_tokens == 0:
                    # Skip this batch if no valid answer tokens
                    tqdm.write(f"Warning: Batch {i} has no valid answer tokens. Skipping...")
                    continue

                answer_labels = torch.where(answer_mask, shift_labels, torch.tensor(tokenizer.pad_token_id, device=device))
                loss_sum = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    answer_labels.view(-1)
                )
                loss = loss_sum / num_valid_tokens

            # Backward with gradient accumulation
            if optimizer:
                norm_loss = loss / args.grad_accum
                scaler.scale(norm_loss).backward()

                if (i + 1) % args.grad_accum == 0:
                    # Unscale before clipping
                    scaler.unscale_(optimizer)
                    # Gradient clipping to prevent NaN
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        args.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            # Metrics on answer tokens only
            with torch.no_grad():
                preds = shift_logits.argmax(dim=-1)
                correct = (preds == shift_labels) & answer_mask

                running_ce += loss.detach()
                running_correct += correct.sum()
                running_total += answer_mask.sum()

            batches += 1

            # Update progress bar with current metrics
            batch_acc = (correct.sum() / answer_mask.sum() * 100).item() if answer_mask.sum() > 0 else 0.0
            postfix = {
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.1f}%'
            }
            if has_diagnostics and diagnostics:
                postfix['energy'] = f'{diagnostics.get("avg_energy", 0.0):.3f}'
                postfix['gate'] = f'{diagnostics.get("avg_gate", 0.0):.3f}'
            pbar.set_postfix(postfix)

            # Detailed logging at intervals
            if (i + 1) % args.grad_accum == 0 and (batches // args.grad_accum) % args.log_interval == 0:
                eff_step = batches // args.grad_accum
                log_msg = f"Step {eff_step} | CE: {loss.item():.4f} | Ans Acc: {batch_acc:.2f}%"

                # Add diagnostics to log if available
                if has_diagnostics and diagnostics:
                    log_msg += f" | Energy: {diagnostics.get('avg_energy', 0.0):.4f}"
                    log_msg += f" | Gate: {diagnostics.get('avg_gate', 0.0):.4f}"
                    log_msg += f" | Rho: {diagnostics.get('rho_complexity', 0.0):.4f}"

                tqdm.write(log_msg)

            if args.max_steps > 0 and batches >= args.max_steps:
                break

        # Epoch Summary (only .item() at epoch end)
        avg_ce = (running_ce / batches).item()
        avg_acc = (running_correct / running_total * 100).item() if running_total > 0 else 0

        print("\n" + "=" * 50)
        print(f"Epoch {epoch+1} Results - Model: {args.model_type} ({args.base_model})")
        print("=" * 50)
        print(f"CE Loss (ans) : {avg_ce:.4f}")
        print(f"Answer Acc    : {avg_acc:.2f}%")

        # Print NSD diagnostics if available
        if has_diagnostics and batches > 0:
            print(f"\n[NSD Diagnostics - Training]")
            print(f"  Logical Energy      : {running_energy / batches:.4f}  (Lower = Better prediction)")
            print(f"  Gate Activity       : {running_gate / batches:.4f}  (Lower = More selective)")
            print(f"  Rho Complexity      : {running_rho_complexity / batches:.4f}  (Higher = More non-trivial)")

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
                        choices=["baseline", "recurrent_sheaf", "deltanet"],
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="Disable automatic mixed precision")

    # Data configuration
    parser.add_argument("--dataset", type=str, default="proofwriter",
                        choices=["gsm8k", "proofwriter"],
                        help="Dataset to use (gsm8k or proofwriter)")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="ProofWriter question depth (QDep) - used when --ood is off")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length (default: 1024 for GSM8K)")
    parser.add_argument("--ood", action="store_true", default=False,
                        help="OOD mode for ProofWriter: Train on shallow (0-2), Test on deep (5)")

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
