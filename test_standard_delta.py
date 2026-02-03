#!/usr/bin/env python
"""Test Standard Delta Rule implementation and compare parameter counts."""

import torch
from model_v2 import create_model, count_trainable_params

device = torch.device('cpu')

print("=" * 80)
print("PARAMETER COMPARISON: Standard Delta vs Delta (Matched) vs Recurrent Sheaf")
print("=" * 80)

models_to_test = [
    ("baseline", "Baseline (Frozen GPT-2)"),
    ("standard_delta", "Standard Delta Rule"),
    ("delta_rule", "Delta Rule (Parameter-Matched)"),
    ("recurrent_sheaf", "Recurrent Sheaf"),
]

results = []

for model_type, name in models_to_test:
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    model, config = create_model(
        model_type=model_type,
        base_model_name="gpt2-medium",
        device=device
    )

    trainable = count_trainable_params(model)
    total = sum(p.numel() for p in model.parameters())

    print(f"\n[Parameter Count]")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Frozen: {total - trainable:,}")

    if model_type != "baseline":
        overhead = trainable / 355_000_000 * 100
        print(f"  Overhead vs GPT-2 Medium: {overhead:.2f}%")

    results.append({
        "name": name,
        "trainable": trainable,
        "total": total,
        "type": model_type
    })

# Summary Table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

print(f"\n{'Model':<35} {'Trainable Params':<20} {'Overhead':<15}")
print("-" * 70)

for r in results:
    trainable_str = f"{r['trainable']:,}"
    if r['type'] == 'baseline':
        overhead_str = "0%"
    else:
        overhead_pct = r['trainable'] / 355_000_000 * 100
        overhead_str = f"{overhead_pct:.2f}%"

    print(f"{r['name']:<35} {trainable_str:<20} {overhead_str:<15}")

# Detailed breakdown for Standard Delta
print("\n" + "=" * 80)
print("DETAILED BREAKDOWN: Standard Delta Rule Layer")
print("=" * 80)

hidden_dim = 1024
print(f"\nPer-layer parameters (hidden_dim={hidden_dim}):")
print(f"  1. Decay parameter: {hidden_dim:,}")
print(f"  2. Alpha gate (Linear): {hidden_dim * hidden_dim:,}")
print(f"  Total per layer: {hidden_dim + hidden_dim * hidden_dim:,}")
print(f"\n4 layers × {hidden_dim + hidden_dim * hidden_dim:,} = {4 * (hidden_dim + hidden_dim * hidden_dim):,}")

print("\n" + "=" * 80)
print("✅ Standard Delta Rule successfully added!")
print("=" * 80)
