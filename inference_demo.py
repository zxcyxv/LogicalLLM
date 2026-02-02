import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models import GPT2WithSheafHead

def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load Baseline
    print("Loading Baseline...")
    baseline = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    baseline.eval()
    
    # Load Sheaf Model
    print("Loading Sheaf Model...")
    sheaf_model = GPT2WithSheafHead(baseline.transformer, hidden_dim=768).to(device)
    sheaf_model.lm_head = baseline.lm_head
    
    # Load weights if available
    try:
        sheaf_model.sheaf_layer.load_state_dict(torch.load("sheaf_layer.pt", map_location=device))
        print("Loaded trained Sheaf weights.")
    except Exception as e:
        print(f"Could not load weights: {e}. Running with random init (or untrained).")

    sheaf_model.eval()

    # Test Case
    text = "Wolves are afraid of mice. Sheep are afraid of wolves. Mice are afraid of cats. What are sheep afraid of?"
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    
    print(f"\nContext: {text}")
    
    # 1. Baseline Generation
    print("\n--- Baseline Generation ---")
    with torch.no_grad():
        out = baseline.generate(input_ids, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    print(f"Baseline: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # 2. Sheaf Generation (Manual Loop for Diagnostics)
    print("\n--- Sheaf Generation (Diagnostic) ---")
    curr_ids = input_ids.clone()
    
    # Reset buffer
    sheaf_model.sheaf_layer.history_buffer = []
    # Pre-fill buffer with context?
    # No, inference mode assumes we feed one by one?
    # Or strict teacher forcing for context?
    # Our `_forward_inference` logic appends to buffer.
    # We should feed context tokens first to fill buffer.
    
    with torch.no_grad():
        # Feed context to build buffer
        # We need to run forward on context tokens one by one?
        # Or just manually append to buffer?
        # Hidden states:
        context_out = baseline.transformer(curr_ids)
        hidden = context_out.last_hidden_state # [1, L, D]
        # Append all to buffer
        for i in range(hidden.size(1)):
            sheaf_model.sheaf_layer.history_buffer.append(hidden[:, i:i+1, :])
            if len(sheaf_model.sheaf_layer.history_buffer) > 16:
                sheaf_model.sheaf_layer.history_buffer.pop(0)
                
        print("Context processed. generating...")
        
        for _ in range(5):
            # 1. Get next token candidate (from current state? No, Causal LM predicts next from past)
            # The model is: Context -> Hidden -> Filter -> Head -> Next Token
            # We already have Context Hidden.
            # We need to filter the *Last* hidden state of context to predict next token.
            
            # Current hidden state to decode from is the last one in buffer?
            # No, `forward` takes input_ids.
            # Let's trust pure generation loop logic or simulate step by step.
            
            # Standard generate:
            # logit = model(curr_ids) -> last token
            pass

    # Actually, simpler to just use .generate() if we hook the layer to print?
    # Let's add a print hook or just manually run one step for the critical prediction.
    
    # Manual prediction of NEXT token
    with torch.no_grad():
        # Full forward pass with sheaf
        # Note: model() calls sheaf_layer(mode='train') by default unless we change it.
        # But we want inference mode?
        # `models.py` uses `mode='train'` in forward().
        # We need to switch or use the inference logic.
        # Training logic uses `unfold` on full sequence. It SHOULD give same result for the last token as inference logic if causal masking is correct.
        
        refined, diag = sheaf_model(curr_ids) # forward calls _forward_train
        logits = sheaf_model.lm_head(refined)
        next_token_logits = logits[0, -1, :]
        next_token = next_token_logits.argmax()
        print(f"Sheaf Prediction: {tokenizer.decode(next_token)}")
        
        print("\nDiagnostics (Last Step):")
        # To get specific diagnostics of the last token, we look at the diag dict.
        # Diag dict contains means.
        # Logic in models.py:
        # `diagnostics` contains means over Batch and Window.
        # We want the specific energy of the last token.
        # The training forward computes for ALL windows.
        
        # Let's modify models.py one last time or just trust the mean if L is large?
        # Actually for demo, we want precise "Why did you change this?".
        # Maybe just print the `update_norm` and `pre_energy` from the dict.
        print(diag)

if __name__ == "__main__":
    run_demo()
