import torch
from transformers import GPT2LMHeadModel
from models import GPT2WithSheafAdapters

def test_adapter_structure():
    """Test that adapters are correctly injected and trainable."""
    print("Testing Sheaf Adapter Structure...")
    
    # Load base model
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Create adapter model
    model = GPT2WithSheafAdapters(
        base_model,
        hidden_dim=768,
        window_size=16,
        mlp_hidden=128,
        target_layers=[8, 9, 10, 11],
        diffusion_steps=1
    )
    
    # Check that backbone is frozen
    frozen_params = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    total_params = sum(1 for p in model.backbone.parameters())
    print(f"Frozen backbone params: {frozen_params}/{total_params}")
    
    # Check that adapters are trainable
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters ({len(trainable_params)}):")
    for name in trainable_params[:10]:  # Show first 10
        print(f"  - {name}")
    
    # Check adapter injection
    for i in [8, 9, 10, 11]:
        block = model.backbone.transformer.h[i]
        print(f"Layer {i}: {type(block).__name__}")
        assert hasattr(block, 'sheaf_adapter'), f"Layer {i} missing sheaf_adapter!"
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_ids = torch.randint(0, 50257, (2, 10)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    logits, diagnostics = model(input_ids, attention_mask)
    
    print(f"\nForward pass successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Diagnostics keys: {list(diagnostics.keys())}")
    print(f"Pre-energy: {diagnostics.get('pre_energy', 'N/A')}")
    print(f"Post-energy: {diagnostics.get('post_energy', 'N/A')}")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_adapter_structure()
