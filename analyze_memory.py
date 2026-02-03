#!/usr/bin/env python
"""Analyze DeltaNet memory usage."""

import torch

print("=" * 80)
print("DELTANET MEMORY ANALYSIS")
print("=" * 80)

# Settings
batch_size = 4
seq_len = 512  # GSM8K average
hidden_dim = 1024  # GPT-2 medium
num_layers = 4

print(f"\n[Configuration]")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Target layers: {num_layers}")

# Calculate W size
W_size_per_batch = batch_size * hidden_dim * hidden_dim
W_size_bytes = W_size_per_batch * 4  # float32
W_size_MB = W_size_bytes / 1024 / 1024

print(f"\n[Memory Matrix W - Single Timestep]")
print(f"  Shape: [{batch_size}, {hidden_dim}, {hidden_dim}]")
print(f"  Size: {W_size_MB:.2f} MB")

# THE PROBLEM: We need to store W for EVERY timestep for backprop!
total_W_per_layer = W_size_MB * seq_len
total_W_all_layers = total_W_per_layer * num_layers

print(f"\n[CRITICAL: W stored for ALL timesteps (for backprop)]")
print(f"  W per timestep: {W_size_MB:.2f} MB")
print(f"  Timesteps: {seq_len}")
print(f"  Total per layer: {total_W_per_layer:.2f} MB = {total_W_per_layer/1024:.2f} GB")
print(f"  Total all {num_layers} layers: {total_W_all_layers:.2f} MB = {total_W_all_layers/1024:.2f} GB")

# Plus gradients
with_gradients = total_W_all_layers * 2
print(f"\n[With Gradients (×2)]")
print(f"  Total: {with_gradients:.2f} MB = {with_gradients/1024:.2f} GB")

# Plus other activations
gpt2_medium_size = 355 * 4  # 355M params × 4 bytes
other_activations = 2000  # Rough estimate in MB
total_memory = with_gradients + gpt2_medium_size + other_activations

print(f"\n[Total Estimated Memory Usage]")
print(f"  DeltaNet W (with grad): {with_gradients/1024:.2f} GB")
print(f"  GPT-2 Medium params: {gpt2_medium_size/1024:.2f} GB")
print(f"  Other activations: ~{other_activations/1024:.2f} GB")
print(f"  TOTAL: ~{total_memory/1024:.2f} GB")

print(f"\n{'='*80}")
print(f"DIAGNOSIS")
print(f"{'='*80}")

print(f"\n❌ PROBLEM: DeltaNet stores W for EVERY timestep!")
print(f"   - Recurrent updates mean W changes at each step")
print(f"   - PyTorch needs to store all intermediate W for backprop")
print(f"   - Memory = O(batch × seq_len × D²) = HUGE!")

print(f"\n✅ SOLUTIONS:")
print(f"   1. GRADIENT CHECKPOINTING (recommended)")
print(f"      - Recompute W during backward pass")
print(f"      - Trade compute for memory")
print(f"   ")
print(f"   2. REDUCE SEQUENCE LENGTH")
print(f"      - Use max_length=256 instead of 1024")
print(f"      - Memory: {(with_gradients * 256 / 512 / 1024):.2f} GB")
print(f"   ")
print(f"   3. REDUCE BATCH SIZE")
print(f"      - batch_size=2: {(with_gradients * 2 / 4 / 1024):.2f} GB")
print(f"      - batch_size=1: {(with_gradients * 1 / 4 / 1024):.2f} GB")
print(f"   ")
print(f"   4. USE LINEAR ATTENTION APPROXIMATION")
print(f"      - Don't store W explicitly")
print(f"      - Compute online using cumsum tricks")

print(f"\n{'='*80}")
print(f"COMPARISON: DeltaNet vs Recurrent Sheaf")
print(f"{'='*80}")

# Recurrent Sheaf memory
sheaf_state_size = batch_size * hidden_dim * 4 / 1024 / 1024  # [B, D]
sheaf_total = sheaf_state_size * seq_len * num_layers
sheaf_with_grad = sheaf_total * 2

print(f"\nRecurrent Sheaf:")
print(f"  State: [{batch_size}, {hidden_dim}] = {sheaf_state_size:.2f} MB")
print(f"  Total (with grad): {sheaf_with_grad:.2f} MB = {sheaf_with_grad/1024:.2f} GB")

print(f"\nDeltaNet:")
print(f"  State: [{batch_size}, {hidden_dim}, {hidden_dim}] = {W_size_MB:.2f} MB")
print(f"  Total (with grad): {with_gradients:.2f} MB = {with_gradients/1024:.2f} GB")

print(f"\nDeltaNet uses {with_gradients/sheaf_with_grad:.1f}× MORE memory than Sheaf!")

print(f"\n{'='*80}")
