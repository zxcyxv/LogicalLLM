#!/usr/bin/env python
"""Test that the answer_start fix works correctly."""

import torch
from transformers import GPT2Tokenizer
from proofwriter_loader import get_proofwriter_loader

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Loading data...")
train_loader = get_proofwriter_loader(
    tokenizer,
    split="train",
    batch_size=4,
    max_length=512,
    depth_filter='shallow',
    cache_tokenization=True,
    shuffle=False
)

batch = next(iter(train_loader))
device = 'cpu'

input_ids = batch['input_ids']
labels = batch['labels']
answer_start = batch['answer_start']

# Simulate shift operation
shift_labels = labels[..., 1:].contiguous()
B, T = shift_labels.shape

print("\n" + "="*80)
print("TESTING ANSWER_START FIX")
print("="*80)

for i in range(min(2, B)):
    orig_start = answer_start[i].item()
    seq_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
    shift_len = seq_len - 1

    print(f"\n--- Sample {i} ---")
    print(f"Original sequence length: {seq_len}")
    print(f"Shift sequence length: {shift_len}")
    print(f"Original answer_start: {orig_start}")

    # Apply fix
    adjusted_start = max(0, min(orig_start - 1, T - 1))
    print(f"Adjusted answer_start: {adjusted_start}")

    # Check valid tokens
    positions = torch.arange(T)
    answer_mask = (positions >= adjusted_start) & (shift_labels[i] != tokenizer.pad_token_id)
    num_valid = answer_mask.sum().item()

    print(f"Valid answer tokens: {num_valid}")

    if num_valid > 0:
        print("✓ SUCCESS: Has valid answer tokens!")
        # Show answer tokens
        answer_tokens = shift_labels[i][answer_mask]
        answer_text = tokenizer.decode(answer_tokens)
        print(f"Answer text: {answer_text}")
    else:
        print("✗ FAILED: Still no valid answer tokens!")

print("\n" + "="*80)
