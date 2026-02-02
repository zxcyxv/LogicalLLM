
class SheafAdapterBlock(nn.Module):
    def __init__(self, original_block, config, window_size=16, mlp_hidden=128, diffusion_steps=1):
        super().__init__()
        self.block = original_block  # Original GPT2Block
        
        # Sheaf Layer Initialization (Alpha=0.0 for stability)
        self.sheaf_adapter = SheafWindowLayer(
            hidden_dim=config.n_embd,
            window_size=window_size,
            mlp_hidden=mlp_hidden,
            alpha=0.0, 
            diffusion_steps=diffusion_steps
        )

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                use_cache=False, output_attentions=False):
        
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
        
        # outputs[0]: hidden_states (B, L, D)
        # outputs[1]: present_key_values (for cache)
        # outputs[2]: attentions (optional)
        
        hidden_states = outputs[0]

        # 2. Apply Sheaf Logic Adapter
        # Using mode='train' for now as we are in training/validation phase.
        # Note: For genuine inference with generate(), we might need to handle 'inference' mode statefully.
        refined_states, diagnostics = self.sheaf_adapter(hidden_states, mode='train')
        
        # 3. Tuple Repacking
        # Replace hidden_states with refined_states
        new_outputs = (refined_states,) + outputs[1:]
        
        # We lose diagnostics here unless we store it or pass it.
        # For now, let's attach it to the block so it can be retrieved if needed, 
        # or simplified: we ignore it for internal layers to avoid signature change issues of the block.
        # However, we want to train the adapter, so we need the loss?
        # Wait, the loss calculation in train.py uses the diagnostics from the *final* layer in Phase 1.
        # In Phase 2, if we have multiple adapters, we need to collect their losses.
        # But the standard GPT2 forward doesn't return intermediate diagnostics.
        # We might need a global collection mechanism or just rely on the fact that 
        # the refined_states carry the gradient info for the structural loss (if any)?
        # Actually, SheafWindowLayer computes 'pre_energy' but doesn't return a loss tensor, it returns diagnostics.
        # Training relies on `loss = ce_loss + args.lbda * cons_loss`.
        # WE NEED TO AGGREGATE CONSISTENCY LOSS FROM ALL ADAPTERS.
        
        # Quick hack: Attach diagnostics to the module itself, and collect it in the main model.
        self.latest_diagnostics = diagnostics
        
        return new_outputs

class GPT2WithSheafAdapters(nn.Module):
    def __init__(self, frozen_gpt2, hidden_dim=768, window_size=16, mlp_hidden=128, diffusion_steps=1, target_layers=[8, 9, 10, 11]):
        super().__init__()
        self.config = frozen_gpt2.config
        self.backbone = frozen_gpt2
        self.lm_head = frozen_gpt2.lm_head
        self.target_layers = set(target_layers)
        
        # 1. Backbone Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Inject Adapters
        for i in target_layers:
            original_block = self.backbone.transformer.h[i]
            # Replace with AdapterBlock
            self.backbone.transformer.h[i] = SheafAdapterBlock(
                original_block, 
                self.config, 
                window_size=window_size,
                mlp_hidden=mlp_hidden,
                diffusion_steps=diffusion_steps
            )

    def forward(self, input_ids, attention_mask=None):
        # GPT2Model executes transformer.h blocks sequentially.
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        # Collect Diagnostics from all adapters
        aggregated_diagnostics = {}
        total_pre_energy = 0
        total_post_energy = 0
        total_update_norm = 0
        count = 0
        
        for i in self.target_layers:
            block = self.backbone.transformer.h[i]
            if hasattr(block, 'latest_diagnostics'):
                diag = block.latest_diagnostics
                # Aggregate (Sum or Average?)
                # Summing energy across layers makes sense (total inconsistency).
                pre = diag.get("pre_energy", 0)
                post = diag.get("post_energy", 0)
                up = diag.get("update_norm", 0)
                
                total_pre_energy += pre if isinstance(pre, torch.Tensor) else torch.tensor(pre, device=input_ids.device)
                total_post_energy += post if isinstance(post, torch.Tensor) else torch.tensor(post, device=input_ids.device)
                total_update_norm += up if isinstance(up, torch.Tensor) else torch.tensor(up, device=input_ids.device)
                count += 1
        
        if count > 0:
            aggregated_diagnostics["pre_energy"] = total_pre_energy / count # Average per adapter? or Sum? Let's Average.
            aggregated_diagnostics["post_energy"] = total_post_energy / count
            aggregated_diagnostics["update_norm"] = total_update_norm / count
            aggregated_diagnostics["cos_sim"] = 0 # Not tracking for internal now
            
        return logits, aggregated_diagnostics
