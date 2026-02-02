
# Linear Adapter Baseline (Simple comparison to Sheaf)
class LinearAdapterBlock(nn.Module):
    """Minimal adapter: just a single linear projection."""
    def __init__(self, original_block, config):
        super().__init__()
        self.block = original_block
        # Single linear layer: hidden_dim -> hidden_dim
        self.adapter = nn.Linear(config.n_embd, config.n_embd)
        # Initialize to near-identity (small random weights)
        nn.init.normal_(self.adapter.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                use_cache=False, output_attentions=False):
        
        # SDPA Compatibility Fix
        if attention_mask is not None and attention_mask.dtype == torch.int64:
            attention_mask = attention_mask.bool()

        # 1. Execute Original GPT-2 Block
        outputs = self.block(
            x,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        hidden_states = outputs[0]

        # 2. Apply Linear Adapter (residual connection)
        adapted = self.adapter(hidden_states)
        refined_states = hidden_states + adapted
        
        # 3. Tuple Repacking
        new_outputs = (refined_states,) + outputs[1:]
        
        return new_outputs


class GPT2WithLinearAdapters(nn.Module):
    """GPT-2 with simple linear adapters (baseline for comparison)."""
    def __init__(self, frozen_gpt2, target_layers=[8, 9, 10, 11]):
        super().__init__()
        self.config = frozen_gpt2.config
        self.backbone = frozen_gpt2
        self.lm_head = frozen_gpt2.lm_head
        self.target_layers = set(target_layers)
        
        # 1. Backbone Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Inject Linear Adapters
        for i in target_layers:
            original_block = self.backbone.transformer.h[i]
            self.backbone.transformer.h[i] = LinearAdapterBlock(original_block, self.config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # No diagnostics for linear adapter (just return empty dict for compatibility)
        return logits, {}
