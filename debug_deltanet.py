#!/usr/bin/env python
"""Debug DeltaNet NaN issue."""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from gsm8k_loader import get_gsm8k_loader
from model_v2 import DeltaNetLayer

print("=" * 80)
print("DEBUGGING DELTANET NaN ISSUE")
print("=" * 80)

# Load real data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

loader = get_gsm8k_loader(
    tokenizer,
    split="train",
    batch_size=2,
    max_length=512,  # Shorter for debugging
    cache_tokenization=True,
    shuffle=False
)

batch = next(iter(loader))
input_ids = batch['input_ids'][:2]  # Just 2 samples

print(f"\n[Data Info]")
seq_lens = (input_ids != tokenizer.pad_token_id).sum(dim=1)
print(f"  Sequence lengths: {seq_lens.tolist()}")

# Create fake embeddings (simulate GPT-2 output)
B, L = input_ids.shape
D = 1024
x = torch.randn(B, L, D) * 0.02  # Small initial values

print(f"  Input shape: {x.shape}")
print(f"  Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")

# Test DeltaNetLayer
layer = DeltaNetLayer(D)

print(f"\n[Testing DeltaNet Forward]")
print("-" * 80)

# Manual forward with debugging
W = torch.zeros(B, D, D)

max_steps = min(50, L)  # Only test first 50 steps
print(f"Testing first {max_steps} steps...\n")

for t in range(max_steps):
    x_t = x[:, t, :]

    # Project
    k_t = layer.proj_k(x_t)
    v_t = layer.proj_v(x_t)

    # Feature map
    phi_k = F.elu(k_t) + 1

    # Check for issues
    if torch.isnan(phi_k).any():
        print(f"❌ Step {t}: NaN in phi_k!")
        break

    # Retrieve
    v_bar = torch.bmm(W, phi_k.unsqueeze(-1)).squeeze(-1)

    if torch.isnan(v_bar).any():
        print(f"❌ Step {t}: NaN in v_bar!")
        break

    # Error
    error = v_t - v_bar

    # Gate
    beta = torch.sigmoid(layer.beta_gate(x_t))

    # Update
    outer = torch.bmm(
        (beta * error).unsqueeze(-1),
        phi_k.unsqueeze(1)
    )

    if torch.isnan(outer).any():
        print(f"❌ Step {t}: NaN in outer product!")
        break

    W_prev_norm = W.norm().item()
    W = W + outer
    W_new_norm = W.norm().item()

    if torch.isnan(W).any():
        print(f"❌ Step {t}: NaN in W after update!")
        print(f"  W_prev_norm: {W_prev_norm:.2f}")
        print(f"  outer_norm: {outer.norm().item():.2f}")
        print(f"  beta: {beta.mean().item():.6f}")
        print(f"  error_norm: {error.norm().item():.2f}")
        break

    # Print stats every 10 steps
    if t % 10 == 0 or W_new_norm > 1e6:
        print(f"Step {t:3d}: W_norm={W_new_norm:10.2f}, error_norm={error.norm().item():8.4f}, "
              f"beta={beta.mean().item():.4f}, phi_k_norm={phi_k.norm().item():8.4f}")

    # Early stop if exploding
    if W_new_norm > 1e8:
        print(f"\n❌ W is EXPLODING at step {t}!")
        print(f"  W norm: {W_new_norm:.2e}")
        print(f"  This will cause NaN!")
        break

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Check projection weights
proj_k_norm = layer.proj_k.weight.norm().item()
proj_v_norm = layer.proj_v.weight.norm().item()
beta_gate_norm = layer.beta_gate.weight.norm().item()

print(f"\n[Layer Weights]")
print(f"  proj_k weight norm: {proj_k_norm:.4f}")
print(f"  proj_v weight norm: {proj_v_norm:.4f}")
print(f"  beta_gate weight norm: {beta_gate_norm:.4f}")

print(f"\n[Potential Issues]")
print(f"  1. Memory matrix W accumulates without decay")
print(f"  2. For long sequences, W can grow to 1e8+ causing NaN")
print(f"  3. Outer product (error ⊗ phi_k) has shape [B, D, D] = massive")
print(f"  4. ELU(x)+1 feature map may amplify values")

print(f"\n[Solutions]")
print(f"  1. Add memory decay: W = decay * W + beta * (error ⊗ phi_k)")
print(f"  2. Normalize W periodically")
print(f"  3. Use smaller beta initialization")
print(f"  4. Clip W norm")
print(f"  5. Use different feature map (layer norm?)")
