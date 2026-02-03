#!/usr/bin/env python
"""
Verify that datasets are correctly formatted and fed to the model.
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gsm8k_loader import get_gsm8k_loader
from proofwriter_loader import get_proofwriter_loader


def verify_dataset(dataset_name, loader, tokenizer, num_samples=3):
    """Verify dataset formatting and model input."""

    print("\n" + "=" * 80)
    print(f"VERIFYING {dataset_name.upper()} DATASET")
    print("=" * 80)

    batch = next(iter(loader))

    print(f"\n[Batch Shape Info]")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Attention Mask: {batch['attention_mask'].shape}")
    print(f"  Labels: {batch['labels'].shape}")
    print(f"  Answer Start: {batch['answer_start'].shape}")

    for i in range(min(num_samples, len(batch['input_ids']))):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*80}")

        input_ids = batch['input_ids'][i]
        labels = batch['labels'][i]
        answer_start = batch['answer_start'][i].item()

        # Calculate sequence length (non-padding)
        seq_len = (input_ids != tokenizer.pad_token_id).sum().item()

        print(f"\n[Sequence Info]")
        print(f"  Total Length: {len(input_ids)}")
        print(f"  Non-padding Length: {seq_len}")
        print(f"  Answer Start Position: {answer_start}")
        print(f"  Answer Tokens: {seq_len - answer_start}")

        # Decode full text
        full_text = tokenizer.decode(input_ids[:seq_len], skip_special_tokens=False)

        # Split into context and answer
        if answer_start < seq_len:
            context_tokens = input_ids[:answer_start]
            answer_tokens = input_ids[answer_start:seq_len]

            context_text = tokenizer.decode(context_tokens, skip_special_tokens=False)
            answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=False)

            print(f"\n[CONTEXT] (0 to {answer_start})")
            print(f"{context_text[:300]}{'...' if len(context_text) > 300 else ''}")

            print(f"\n[ANSWER] ({answer_start} to {seq_len})")
            print(f"{answer_text[:200]}{'...' if len(answer_text) > 200 else ''}")
        else:
            print(f"\n[WARNING] Answer start ({answer_start}) >= sequence length ({seq_len})")
            print(f"Full text:\n{full_text[:500]}...")

        # Verify input_ids == labels
        labels_match = torch.equal(input_ids, labels)
        print(f"\n[Verification]")
        print(f"  input_ids == labels: {labels_match}")

        # Simulate shift operation (for loss calculation)
        shift_labels = labels[1:seq_len].contiguous()
        shift_len = len(shift_labels)
        adjusted_answer_start = max(0, min(answer_start - 1, shift_len - 1))

        print(f"\n[After Shift Operation]")
        print(f"  Shift Length: {shift_len}")
        print(f"  Adjusted Answer Start: {adjusted_answer_start}")
        print(f"  Answer Tokens (after shift): {shift_len - adjusted_answer_start}")

        if shift_len - adjusted_answer_start <= 0:
            print(f"  ⚠️  WARNING: No answer tokens after shift!")
        else:
            print(f"  ✓ OK: {shift_len - adjusted_answer_start} answer tokens available")

    print("\n" + "=" * 80)


def test_model_forward(dataset_name, loader, device='cpu'):
    """Test that data can be fed to model without errors."""

    print("\n" + "=" * 80)
    print(f"TESTING MODEL FORWARD PASS - {dataset_name.upper()}")
    print("=" * 80)

    # Load a small model for testing
    print(f"\nLoading GPT-2 small for testing...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    batch = next(iter(loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    print(f"\n[Model Outputs]")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Loss: {outputs.loss.item():.4f}")

    # Verify logits shape
    expected_shape = (input_ids.shape[0], input_ids.shape[1], model.config.vocab_size)
    assert outputs.logits.shape == expected_shape, f"Logits shape mismatch! Expected {expected_shape}, got {outputs.logits.shape}"

    print(f"  ✓ Model forward pass successful!")
    print("=" * 80)


if __name__ == "__main__":
    # Setup
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test GSM8K
    print("\n" + "#" * 80)
    print("# GSM8K DATASET VERIFICATION")
    print("#" * 80)

    gsm8k_loader = get_gsm8k_loader(
        tokenizer,
        split="train",
        batch_size=4,
        max_length=1024,
        cache_tokenization=True,
        shuffle=False
    )

    verify_dataset("GSM8K", gsm8k_loader, tokenizer, num_samples=2)
    test_model_forward("GSM8K", gsm8k_loader, device)

    # Test ProofWriter
    print("\n" + "#" * 80)
    print("# PROOFWRITER DATASET VERIFICATION")
    print("#" * 80)

    proofwriter_loader = get_proofwriter_loader(
        tokenizer,
        split="train",
        batch_size=4,
        max_length=512,
        cache_tokenization=True,
        depth_filter='shallow',
        shuffle=False
    )

    verify_dataset("ProofWriter", proofwriter_loader, tokenizer, num_samples=2)
    test_model_forward("ProofWriter", proofwriter_loader, device)

    print("\n" + "#" * 80)
    print("# ALL VERIFICATIONS PASSED ✓")
    print("#" * 80)
