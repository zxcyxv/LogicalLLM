import torch
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from models import GPT2WithSheafHead

def verify():
    print("Initializing dummy GPT-2...")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=768)
    gpt2_backend = GPT2Model(config)
    
    print("Initializing Sheaf Model...")
    model = GPT2WithSheafHead(gpt2_backend)
    
    # Dummy Input
    input_ids = torch.randint(0, 1000, (2, 32)) # B=2, L=32
    
    print("Forward Pass...")
    refined_states, loss = model(input_ids)
    
    print("Output Shape:", refined_states.shape)
    print("Consistency Loss:", loss.item())
    
    # Check Gradients
    print("Checking Gradients after backward...")
    loss.backward()
    
    print("Alpha Grad:", model.sheaf_layer.alpha.grad)
    print("MLP Weight Grad:", model.sheaf_layer.restriction_mlp[0].weight.grad.mean())
    print("Backbone Grad (Should be None/Zero):", list(model.backbone.parameters())[0].grad)

if __name__ == "__main__":
    verify()
