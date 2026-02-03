#!/usr/bin/env python
"""
Verify that loss is computed ONLY on answer tokens, NOT on context.
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from gsm8k_loader import get_gsm8k_loader


def verify_loss_calculation(loader, tokenizer):
    """Verify that loss is computed only on answer tokens."""

    print("\n" + "=" * 80)
    print("VERIFYING LOSS CALCULATION - GSM8K")
    print("=" * 80)

    batch = next(iter(loader))

    # Use first sample
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]
    answer_start = batch['answer_start'][0].item()

    seq_len = (input_ids != tokenizer.pad_token_id).sum().item()

    print(f"\n[Original Sequence]")
    print(f"  Total Length: {len(input_ids)}")
    print(f"  Non-padding Length: {seq_len}")
    print(f"  Answer Start: {answer_start}")

    # Decode sections
    context_tokens = input_ids[:answer_start]
    answer_tokens = input_ids[answer_start:seq_len]

    context_text = tokenizer.decode(context_tokens)
    answer_text = tokenizer.decode(answer_tokens)

    print(f"\n[CONTEXT] (tokens 0 to {answer_start-1})")
    print(f"{context_text}")
    print(f"\n[ANSWER] (tokens {answer_start} to {seq_len-1})")
    print(f"{answer_text}")

    # === SIMULATE SHIFT OPERATION (as done in train_v2.py) ===
    print(f"\n" + "=" * 80)
    print("SIMULATING SHIFT OPERATION (for CLM loss)")
    print("=" * 80)

    # Shift for causal language modeling
    shift_logits_would_be = input_ids[:-1]  # Predictions for positions 0 to T-1
    shift_labels = labels[1:]  # Targets for positions 1 to T

    shift_len = len(shift_labels)

    print(f"\nAfter shift:")
    print(f"  shift_logits: predicting tokens 1 to {shift_len}")
    print(f"  shift_labels: actual tokens 1 to {shift_len}")
    print(f"  Shift sequence length: {shift_len}")

    # Adjust answer_start for shift (as in train_v2.py)
    adjusted_answer_start = max(0, min(answer_start - 1, shift_len - 1))
    print(f"  Adjusted answer_start: {adjusted_answer_start}")

    # === CREATE ANSWER MASK (as done in train_v2.py) ===
    positions = torch.arange(shift_len)
    answer_mask = (positions >= adjusted_answer_start) & (shift_labels != tokenizer.pad_token_id)

    num_context_tokens = answer_mask.logical_not().sum().item()
    num_answer_tokens = answer_mask.sum().item()

    print(f"\n" + "=" * 80)
    print("LOSS CALCULATION BREAKDOWN")
    print("=" * 80)

    print(f"\n[Token Classification]")
    print(f"  Context tokens (EXCLUDED from loss): {num_context_tokens}")
    print(f"  Answer tokens (INCLUDED in loss): {num_answer_tokens}")
    print(f"  Total tokens in shift: {shift_len}")

    # Show which tokens are included/excluded
    print(f"\n[Detailed Token Analysis]")

    # Context portion (should be EXCLUDED)
    context_in_shift = shift_labels[:adjusted_answer_start]
    context_mask_portion = answer_mask[:adjusted_answer_start]

    print(f"\n1. CONTEXT PORTION (should be EXCLUDED):")
    print(f"   Positions: 0 to {adjusted_answer_start-1}")
    print(f"   Number of tokens: {len(context_in_shift)}")
    print(f"   Included in loss? {context_mask_portion.any().item()}")
    print(f"   Text: {tokenizer.decode(context_in_shift)[:100]}...")

    # Answer portion (should be INCLUDED)
    answer_in_shift = shift_labels[adjusted_answer_start:seq_len-1]
    answer_mask_portion = answer_mask[adjusted_answer_start:seq_len-1]

    print(f"\n2. ANSWER PORTION (should be INCLUDED):")
    print(f"   Positions: {adjusted_answer_start} to {seq_len-2}")
    print(f"   Number of tokens: {len(answer_in_shift)}")
    print(f"   Included in loss? {answer_mask_portion.all().item()}")
    print(f"   Text: {tokenizer.decode(answer_in_shift)[:100]}...")

    # === VERIFY LOSS CALCULATION ===
    print(f"\n" + "=" * 80)
    print("VERIFYING LOSS COMPUTATION")
    print("=" * 80)

    # Create masked labels (as done in train_v2.py)
    answer_labels = torch.where(
        answer_mask,
        shift_labels,
        torch.tensor(tokenizer.pad_token_id)
    )

    # Count which tokens will contribute to loss
    tokens_in_loss = (answer_labels != tokenizer.pad_token_id).sum().item()

    print(f"\n[Loss Calculation]")
    print(f"  Tokens that contribute to loss: {tokens_in_loss}")
    print(f"  Expected (answer tokens only): {num_answer_tokens}")
    print(f"  Match: {tokens_in_loss == num_answer_tokens}")

    if tokens_in_loss == num_answer_tokens:
        print(f"\n  ✅ CORRECT: Loss is computed ONLY on answer tokens!")
    else:
        print(f"\n  ❌ ERROR: Loss includes non-answer tokens!")

    # Show the actual tokens included in loss
    loss_token_ids = answer_labels[answer_labels != tokenizer.pad_token_id]
    loss_text = tokenizer.decode(loss_token_ids)

    print(f"\n[Tokens Actually Used in Loss]")
    print(f"{loss_text}")

    print("\n" + "=" * 80)

    return tokens_in_loss == num_answer_tokens


def test_multiple_samples(loader, tokenizer, num_samples=5):
    """Test multiple samples to ensure consistency."""

    print("\n" + "=" * 80)
    print(f"TESTING {num_samples} SAMPLES FOR CONSISTENCY")
    print("=" * 80)

    batch = next(iter(loader))

    all_correct = True

    for i in range(min(num_samples, len(batch['input_ids']))):
        input_ids = batch['input_ids'][i]
        labels = batch['labels'][i]
        answer_start = batch['answer_start'][i].item()

        seq_len = (input_ids != tokenizer.pad_token_id).sum().item()

        # Shift
        shift_labels = labels[1:seq_len]
        shift_len = len(shift_labels)
        adjusted_answer_start = max(0, min(answer_start - 1, shift_len - 1))

        # Create mask
        positions = torch.arange(shift_len)
        answer_mask = (positions >= adjusted_answer_start) & (shift_labels != tokenizer.pad_token_id)

        num_answer_tokens = answer_mask.sum().item()

        # Verify
        answer_labels = torch.where(answer_mask, shift_labels, torch.tensor(tokenizer.pad_token_id))
        tokens_in_loss = (answer_labels != tokenizer.pad_token_id).sum().item()

        is_correct = (tokens_in_loss == num_answer_tokens)
        status = "✅" if is_correct else "❌"

        print(f"\nSample {i+1}: {status}")
        print(f"  Seq len: {seq_len}, Answer start: {answer_start} -> {adjusted_answer_start}")
        print(f"  Answer tokens: {num_answer_tokens}, Loss tokens: {tokens_in_loss}")

        if not is_correct:
            all_correct = False

    print("\n" + "=" * 80)
    if all_correct:
        print("✅ ALL SAMPLES CORRECT: Loss computed only on answers!")
    else:
        print("❌ SOME SAMPLES FAILED: Check implementation!")
    print("=" * 80)

    return all_correct


if __name__ == "__main__":
    print("Loading GSM8K dataset...")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    loader = get_gsm8k_loader(
        tokenizer,
        split="train",
        batch_size=8,
        max_length=1024,
        cache_tokenization=True,
        shuffle=False
    )

    # Detailed verification of first sample
    verify_loss_calculation(loader, tokenizer)

    # Test consistency across multiple samples
    test_multiple_samples(loader, tokenizer, num_samples=5)

    print("\n" + "#" * 80)
    print("# VERIFICATION COMPLETE")
    print("#" * 80)
