import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

class SheafWindowLayer(nn.Module):
    def __init__(self, hidden_dim=768, window_size=64, mlp_hidden=128, alpha=0.1, diffusion_steps=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Trainable step size
        self.diffusion_steps = diffusion_steps
        
        # Positional embeddings for window
        self.pos_emb = nn.Parameter(torch.randn(window_size, hidden_dim))
        
        # MLP for restriction maps (2*hidden_dim because we concat [h_i, h_j])
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_dim)
        )
    
    def forward(self, hidden_states, mode='train'):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mode: 'train' or 'inference'
        Returns:
            refined_states: (batch, seq_len, hidden_dim)
            diagnostics: dict with energy metrics
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Add positional embeddings (broadcast across batch)
        pos_emb_expanded = self.pos_emb[:seq_len].unsqueeze(0)  # (1, seq_len, hidden_dim)
        x = hidden_states + pos_emb_expanded
        
        # Initialize diagnostics
        diagnostics = {}
        
        # Compute initial energy (before diffusion)
        if mode == 'train':
            pre_energy = self._compute_energy(x)
            diagnostics['pre_energy'] = pre_energy
        
        # Multi-step diffusion
        for step in range(self.diffusion_steps):
            x = self._diffusion_step(x)
        
        # Compute post-diffusion energy
        if mode == 'train':
            post_energy = self._compute_energy(x)
            diagnostics['post_energy'] = post_energy
            
            # Track update magnitude
            update_norm = torch.norm(x - hidden_states, dim=-1).mean()
            diagnostics['update_norm'] = update_norm
            
            # Cosine similarity between original and refined
            cos_sim = F.cosine_similarity(hidden_states.flatten(0, 1), x.flatten(0, 1), dim=-1).mean()
            diagnostics['cos_sim'] = cos_sim
        
        return x, diagnostics
    
    def _diffusion_step(self, x):
        """Single diffusion step over the sequence."""
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # For each position, aggregate from window
        updates = []
        for i in range(seq_len):
            # Define window around position i
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2)
            
            # Get neighbors in window
            neighbors = x[:, start:end, :]  # (batch, window_len, hidden_dim)
            
            # Compute restriction maps for each neighbor
            h_i = x[:, i:i+1, :].expand(-1, neighbors.size(1), -1)  # (batch, window_len, hidden_dim)
            concat = torch.cat([h_i, neighbors], dim=-1)  # (batch, window_len, 2*hidden_dim)
            
            # MLP produces edge features
            edge_features = self.mlp(concat)  # (batch, window_len, hidden_dim)
            
            # Aggregate (mean pooling)
            aggregated = edge_features.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
            updates.append(aggregated)
        
        # Stack updates
        updates = torch.cat(updates, dim=1)  # (batch, seq_len, hidden_dim)
        
        # Diffusion update: x_new = x + alpha * (aggregated - x)
        x_new = x + self.alpha * (updates - x)
        
        return x_new
    
    def _compute_energy(self, x):
        """Compute sheaf energy (disagreement between neighbors)."""
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        total_energy = 0.0
        count = 0
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2)
            
            neighbors = x[:, start:end, :]
            h_i = x[:, i:i+1, :].expand(-1, neighbors.size(1), -1)
            
            concat = torch.cat([h_i, neighbors], dim=-1)
            edge_features = self.mlp(concat)
            
            # Energy = ||R_ij(h_i) - h_j||^2
            energy = torch.norm(edge_features - neighbors, dim=-1).pow(2).sum()
            total_energy += energy
            count += neighbors.size(1) * batch_size
        
        return total_energy / count if count > 0 else torch.tensor(0.0, device=device)


class GPT2WithSheafHead(nn.Module):
    """GPT-2 with Sheaf Layer as post-processing (Phase 1)."""
    def __init__(self, base_model, hidden_dim=768, window_size=16, mlp_hidden=128, diffusion_steps=1):
        super().__init__()
        self.backbone = base_model
        self.sheaf_layer = SheafWindowLayer(hidden_dim, window_size, mlp_hidden, diffusion_steps=diffusion_steps)
        self.lm_head = nn.Linear(hidden_dim, base_model.config.vocab_size, bias=False)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, mode='train'):
        # Get hidden states from frozen GPT-2
        outputs = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Apply Sheaf refinement
        refined_states, diagnostics = self.sheaf_layer(hidden_states, mode=mode)
        
        # Project to vocabulary
        logits = self.lm_head(refined_states)
        
        return logits, diagnostics


class SheafAdapterBlock(nn.Module):
    """Wraps a GPT2Block and injects Sheaf logic adapter."""
    def __init__(self, original_block, hidden_dim=768, window_size=16, mlp_hidden=128, diffusion_steps=1):
        super().__init__()
        self.block = original_block
        self.sheaf_adapter = SheafWindowLayer(hidden_dim, window_size, mlp_hidden, diffusion_steps=diffusion_steps)
        self.latest_diagnostics = {}
    
    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                use_cache=False, output_attentions=False):
        
        # SDPA Fix: Cast attention_mask to boolean if it's int64
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

        # 2. Apply Sheaf Logic Adapter
        refined_states, diagnostics = self.sheaf_adapter(hidden_states, mode='train')
        
        # 3. Tuple Repacking
        new_outputs = (refined_states,) + outputs[1:]
        
        # Store diagnostics for aggregation
        self.latest_diagnostics = diagnostics
        
        return new_outputs


class GPT2WithSheafAdapters(nn.Module):
    """GPT-2 with Sheaf Adapters injected into specific layers (Phase 2)."""
    def __init__(self, base_model, hidden_dim=768, window_size=16, mlp_hidden=128, 
                 target_layers=[8, 9, 10, 11], diffusion_steps=1):
        super().__init__()
        self.backbone = base_model  # GPT2LMHeadModel
        self.target_layers = target_layers
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Inject adapters into target layers
        for i in target_layers:
            original_block = self.backbone.transformer.h[i]
            adapter_block = SheafAdapterBlock(original_block, hidden_dim, window_size, mlp_hidden, diffusion_steps)
            self.backbone.transformer.h[i] = adapter_block
    
    def forward(self, input_ids, attention_mask=None):
        # Backbone is GPT2LMHeadModel, so it returns CausalLMOutput with logits.
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
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
                # Aggregate (Average)
                pre = diag.get("pre_energy", 0)
                post = diag.get("post_energy", 0)
                up = diag.get("update_norm", 0)
                
                total_pre_energy += pre if isinstance(pre, torch.Tensor) else torch.tensor(pre, device=input_ids.device)
                total_post_energy += post if isinstance(post, torch.Tensor) else torch.tensor(post, device=input_ids.device)
                total_update_norm += up if isinstance(up, torch.Tensor) else torch.tensor(up, device=input_ids.device)
                count += 1
        
        if count > 0:
            aggregated_diagnostics["pre_energy"] = total_pre_energy / count 
            aggregated_diagnostics["post_energy"] = total_post_energy / count
            aggregated_diagnostics["update_norm"] = total_update_norm / count
            aggregated_diagnostics["cos_sim"] = torch.tensor(0.0, device=input_ids.device)  # Not tracking for internal now
            
        return logits, aggregated_diagnostics


class LinearAdapterBlock(nn.Module):
    """Wraps a GPT2Block and applies a simple linear adapter."""
    def __init__(self, original_block, hidden_dim=768):
        super().__init__()
        self.block = original_block
        # Simple linear transformation with near-identity initialization
        self.adapter = nn.Linear(hidden_dim, hidden_dim)
        # Initialize to near-identity
        nn.init.eye_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        self.adapter.weight.data *= 0.01  # Small perturbation
        
    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False):
        
        # SDPA Fix: Cast attention_mask to boolean if it's int64
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
        
        # 2. Apply Linear Adapter with residual
        adapted = self.adapter(hidden_states)
        refined_states = hidden_states + adapted
        
        # 3. Tuple Repacking
        new_outputs = (refined_states,) + outputs[1:]
        
        return new_outputs


class GPT2WithLinearAdapters(nn.Module):
    """GPT-2 with Linear Adapters injected into specific layers (Baseline)."""
    def __init__(self, base_model, hidden_dim=768, target_layers=[8, 9, 10, 11]):
        super().__init__()
        self.backbone = base_model  # GPT2LMHeadModel
        self.target_layers = target_layers
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Inject linear adapters into target layers
        for i in target_layers:
            original_block = self.backbone.transformer.h[i]
            adapter_block = LinearAdapterBlock(original_block, hidden_dim)
            self.backbone.transformer.h[i] = adapter_block
    
    def forward(self, input_ids, attention_mask=None):
        # Backbone is GPT2LMHeadModel
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Return empty diagnostics for compatibility
        return logits, {}
