#!/usr/bin/env python
"""Test script to inspect ProofWriter data and answer_start calculation."""

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

print("\n" + "="*80)
print("INSPECTING FIRST BATCH")
print("="*80)

batch = next(iter(train_loader))

for i in range(min(2, len(batch['input_ids']))):
    input_ids = batch['input_ids'][i]
    answer_start = batch['answer_start'][i].item()

    # Decode full text
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Check sequence length
    seq_len = (input_ids != tokenizer.pad_token_id).sum().item()

    print(f"\n--- Sample {i} ---")
    print(f"Sequence Length (non-pad): {seq_len}")
    print(f"Answer Start Position: {answer_start}")
    print(f"Answer Start >= Seq Len: {answer_start >= seq_len}")
    print(f"\nFull Text:\n{full_text[:500]}...")

    # After shift (for loss calculation)
    shift_len = seq_len - 1
    print(f"\nAfter shift (:-1, 1:):")
    print(f"  Shift Length: {shift_len}")
    print(f"  Answer Start (original): {answer_start}")
    print(f"  Valid tokens in answer range: {max(0, shift_len - answer_start)}")

    if answer_start >= seq_len:
        print(f"  ⚠️  PROBLEM: Answer was TRUNCATED! No answer tokens in sequence.")
    elif answer_start >= shift_len:
        print(f"  ⚠️  PROBLEM: After shift, no answer tokens remain!")

print("\n" + "="*80)
