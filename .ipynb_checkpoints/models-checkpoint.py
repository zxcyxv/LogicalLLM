import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model

# =============================================================================
# Core Module: Sheaf Window Layer (Phase 1 & 2 Engine)
# =============================================================================
class SheafWindowLayer(nn.Module):
    def __init__(self, hidden_dim=768, window_size=64, mlp_hidden=128, alpha=0.0, diffusion_steps=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.alpha = nn.Parameter(torch.tensor(alpha)) # Trainable step size
        self.diffusion_steps = diffusion_steps
        
        # Position Embedding for relative positions in the window
        self.pos_embedding = nn.Embedding(window_size, hidden_dim) 
        
        # Restriction Map (Logic Constraint Calculator)
        self.restriction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden), # Input: [x_u, pos_emb]
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_dim) # Output: Predicted x_v
        )
        
        self.history_buffer = [] # For autoregressive inference

    def forward(self, x, mode='train'):
        """
        x: [Batch, SeqLen, Dim] (Train) or [Batch, 1, Dim] (Validation/Inference step)
        """
        if mode == 'inference':
            return self._forward_inference(x)
        else:
            return self._forward_train(x)

    def _forward_train(self, x):
        # x: [B, L, D]
        # Multi-step Diffusion Loop
        x_curr = x.clone()
        
        # We need to track the initial pre-energy and final post-energy
        first_pre_energy = 0
        final_post_energy = 0
        total_update_norm = 0
        total_cos_sim = 0
        
        for step in range(self.diffusion_steps):
            B, L, D = x_curr.size()
            K = self.window_size
            
            if L < K:
                return x_curr, {} # Not enough context
            
            # 1. Create Windows using unfold
            x_perm = x_curr.permute(0, 2, 1) # [B, D, L]
            windows = x_perm.unfold(2, K, 1) # [B, D, NumWindows, K]
            windows_feat = windows.permute(0, 2, 3, 1) # [B, W, K, D]
            
            context_nodes = windows_feat[:, :, :-1, :] # [B, W, K-1, D]
            current_node = windows_feat[:, :, -1, :].unsqueeze(2) # [B, W, 1, D]
            
            # 2. Position Embeddings
            pos_embeds = self.pos_embedding(torch.arange(K-1, device=x.device)).unsqueeze(0).unsqueeze(0)
            
            # 3. Compute Restrictions (MLP)
            mlp_input = torch.cat([context_nodes, pos_embeds.expand(B, windows_feat.size(1), K-1, D)], dim=-1)
            predictions = self.restriction_mlp(mlp_input)
            
            # 4. Compute Disagreement (Delta)
            deltas = current_node - predictions
            mean_delta = deltas.mean(dim=2) # [B, W, D]
            
            # 5. Update Vector
            update_vector = - (self.alpha * mean_delta)
            
            # 6. Apply Update (Safe Clone)
            x_next = x_curr.clone()
            x_next[:, K-1:, :] = x_curr[:, K-1:, :] + update_vector
            
            # --- Diagnostics ---
            pre_energy = (mean_delta ** 2).sum(dim=-1).mean()
            up_norm = update_vector.norm(dim=-1).mean()
            
            # Stability Check
            cos_sim = F.cosine_similarity(current_node.squeeze(2), current_node.squeeze(2) + update_vector, dim=-1).mean()
            
            if step == 0:
                first_pre_energy = pre_energy
                
            total_update_norm += up_norm
            total_cos_sim += cos_sim
            
            x_curr = x_next
            
            # Analytical Post Energy
            alpha_val = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
            post_energy = ((1.0 - alpha_val) ** 2) * pre_energy
            
            if step == self.diffusion_steps - 1:
                final_post_energy = post_energy
        
        diagnostics = {
            "pre_energy": first_pre_energy,
            "post_energy": final_post_energy,
            "update_norm": total_update_norm / self.diffusion_steps,
            "cos_sim": total_cos_sim / self.diffusion_steps
        }
        
        return x_curr, diagnostics

    def _forward_inference(self, x):
        # x: [Batch, 1, Dim]
        self.history_buffer.append(x)
        if len(self.history_buffer) > self.window_size:
            self.history_buffer.pop(0)
            
        curr_len = len(self.history_buffer)
        if curr_len < 2:
            return x 
            
        window_seq = torch.cat(self.history_buffer, dim=1) # [B, L_curr, D]
        target = window_seq[:, -1:, :] 
        context = window_seq[:, :-1, :]
        
        pos_ids = torch.arange(curr_len - 1, device=x.device)
        pos_embeds = self.pos_embedding(pos_ids).unsqueeze(0)
        
        mlp_input = torch.cat([context, pos_embeds.expand(x.size(0), -1, -1)], dim=-1)
        predictions = self.restriction_mlp(mlp_input)
        
        deltas = target - predictions
        mean_delta = deltas.mean(dim=1, keepdim=True)
        
        x_updated = x - (self.alpha * mean_delta)
        return x_updated


# =============================================================================
# Phase 1 Model: Frozen GPT-2 + Output Logic Filter
# =============================================================================
class GPT2WithSheafHead(nn.Module):
    def __init__(self, frozen_gpt2, hidden_dim=768, freeze=True, diffusion_steps=1):
        super().__init__()
        self.backbone = frozen_gpt2
        self.lm_head = None # Assigned externally (shared)
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            
        self.sheaf_layer = SheafWindowLayer(hidden_dim=hidden_dim, diffusion_steps=diffusion_steps, alpha=0.1) # Phase 1 default alpha
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        refined_states, diagnostics = self.sheaf_layer(hidden_states, mode='train')
        return refined_states, diagnostics


# =============================================================================
# Phase 2 Components: Internal Sheaf Adapters
# =============================================================================
class SheafAdapterBlock(nn.Module):
    def __init__(self, original_block, config, window_size=16, mlp_hidden=128, diffusion_steps=1):
        super().__init__()
        self.block = original_block
        
        # Initialize Alpha=0.0 for stability in internal layers
        self.sheaf_adapter = SheafWindowLayer(
            hidden_dim=config.n_embd,
            window_size=window_size,
            mlp_hidden=mlp_hidden,
            alpha=0.0, 
            diffusion_steps=diffusion_steps
        )
        self.latest_diagnostics = {}

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                use_cache=False, output_attentions=False):
        
        # SDPA Compatibility Fix
        if attention_mask is not None and attention_mask.dtype == torch.int64:
            attention_mask = attention_mask.bool()

        outputs = self.block(
            x, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache, output_attentions=output_attentions,
        )
        
        hidden_states = outputs[0]
        
        # Apply Sheaf Logic Adapter
        refined_states, diagnostics = self.sheaf_adapter(hidden_states, mode='train')
        
        # Tuple Repacking
        new_outputs = (refined_states,) + outputs[1:]
        
        # Save diagnostics for main model to collect
        self.latest_diagnostics = diagnostics
        
        return new_outputs

class GPT2WithSheafAdapters(nn.Module):
    def __init__(self, frozen_gpt2, hidden_dim=768, window_size=16, mlp_hidden=128, diffusion_steps=1, target_layers=[8, 9, 10, 11]):
        super().__init__()
        self.config = frozen_gpt2.config
        self.backbone = frozen_gpt2
        self.lm_head = frozen_gpt2.lm_head
        self.target_layers = set(target_layers)
        
        # 1. Freeze Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Inject Sheaf Adapters
        for i in target_layers:
            original_block = self.backbone.transformer.h[i]
            self.backbone.transformer.h[i] = SheafAdapterBlock(
                original_block, self.config, 
                window_size=window_size, mlp_hidden=mlp_hidden, diffusion_steps=diffusion_steps
            )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Collect Diagnostics
        aggregated_diagnostics = {}
        total_pre = 0; total_post = 0; total_up = 0; count = 0
        
        for i in self.target_layers:
            block = self.backbone.transformer.h[i]
            if hasattr(block, 'latest_diagnostics'):
                diag = block.latest_diagnostics
                # Safe Tensor Conversion
                pre = diag.get("pre_energy", 0)
                post = diag.get("post_energy", 0)
                up = diag.get("update_norm", 0)
                
                total_pre += pre if isinstance(pre, torch.Tensor) else torch.tensor(pre, device=input_ids.device)
                total_post += post if isinstance(post, torch.Tensor) else torch.tensor(post, device=input_ids.device)
                total_up += up if isinstance(up, torch.Tensor) else torch.tensor(up, device=input_ids.device)
                count += 1
        
        if count > 0:
            aggregated_diagnostics["pre_energy"] = total_pre / count 
            aggregated_diagnostics["post_energy"] = total_post / count
            aggregated_diagnostics["update_norm"] = total_up / count
        else:
            aggregated_diagnostics["pre_energy"] = torch.tensor(0.0, device=input_ids.device)
            
        return logits, aggregated_diagnostics


# =============================================================================
# Baseline: Linear Adapter (Simple Control Group)
# =============================================================================
class LinearAdapterBlock(nn.Module):
    def __init__(self, original_block, config):
        super().__init__()
        self.block = original_block
        self.adapter = nn.Linear(config.n_embd, config.n_embd)
        
        # Near-identity initialization
        nn.init.normal_(self.adapter.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                use_cache=False, output_attentions=False):
        
        if attention_mask is not None and attention_mask.dtype == torch.int64:
            attention_mask = attention_mask.bool()

        outputs = self.block(
            x, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache, output_attentions=output_attentions,
        )
        hidden_states = outputs[0]

        # Apply Linear Adapter (Point-wise)
        adapted = self.adapter(hidden_states)
        refined_states = hidden_states + adapted
        
        new_outputs = (refined_states,) + outputs[1:]
        return new_outputs

class GPT2WithLinearAdapters(nn.Module):
    def __init__(self, frozen_gpt2, target_layers=[8, 9, 10, 11]):
        super().__init__()
        self.config = frozen_gpt2.config
        self.backbone = frozen_gpt2
        self.lm_head = frozen_gpt2.lm_head
        self.target_layers = set(target_layers)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for i in target_layers:
            original_block = self.backbone.transformer.h[i]
            self.backbone.transformer.h[i] = LinearAdapterBlock(original_block, self.config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        return outputs.logits, {}

class RecurrentSheafLayer(nn.Module):
    """
    Phase 3 Fixed: Diagonal Gated Sheaf (DGS)
    Stable SSM-like structure with Logic Error Correction.
    Memory Complexity: O(D), Time Complexity: O(L)
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Decay Factor (Forget Gate)
        # Learnable decay initialized close to 1 (long memory) but < 1 (stability)
        self.decay = nn.Parameter(torch.ones(hidden_dim) * 0.9)
        
        # 2. Logic Projector (Restriction Map)
        # S_prev를 보고 현재 x가 어떠해야 하는지 예측
        self.restriction = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. Input/Update Gate
        # 얼마나 수정할지 결정 (Kalman Gain 역할)
        self.update_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Normalization for stability
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        B, L, D = x.size()
        
        # Initial State h_0 = 0
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        
        # Sigmoid applied to gates ensures stability (0~1)
        decay_factor = torch.sigmoid(self.decay) 
        
        # Recurrent Loop (Diagonal/Element-wise)
        # This is fast enough in Python for B=8, L=128. 
        # (Ideally written in CUDA/Triton for production)
        for t in range(L):
            x_t = x[:, t, :] # [B, D]
            
            # 1. Prediction: "과거 문맥 h가 보기에 x_t는 이래야 한다"
            pred_x = self.restriction(h)
            
            # 2. Logical Error: "실제 x_t와의 불일치"
            error = x_t - pred_x
            
            # 3. Gated Update
            # z_t: 얼마나 반영할지 결정하는 게이트
            z_t = torch.sigmoid(self.update_gate(x_t))
            
            # 4. State Transition (SSM Style)
            # h_new = decay * h_old + (1-decay) * gate * error
            # This convex combination guarantees stability.
            h = decay_factor * h + (1 - decay_factor) * (z_t * error)
            
            outputs.append(h)
            
        # Stack & Project
        y = torch.stack(outputs, dim=1) # [B, L, D]
        y = self.ln(y) # Final LayerNorm for safety
        
        return self.out_proj(y)

class RecurrentSheafBlock(nn.Module):
    def __init__(self, original_block, config):
        super().__init__()
        self.block = original_block
        # [수정] alpha 인자 제거 (Diagonal Gated Sheaf는 alpha를 쓰지 않음)
        self.engine = RecurrentSheafLayer(hidden_dim=config.n_embd)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                use_cache=False, output_attentions=False):
        
        # SDPA Compatibility Fix
        if attention_mask is not None and attention_mask.dtype == torch.int64:
            attention_mask = attention_mask.bool()

        # 1. Original Block
        outputs = self.block(
            x, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache, output_attentions=output_attentions,
        )
        hidden_states = outputs[0]

        # 2. Apply Recurrent Sheaf Engine (Residual Add)
        correction = self.engine(hidden_states)
        refined_states = hidden_states + correction
        
        new_outputs = (refined_states,) + outputs[1:]
        return new_outputs

class GPT2WithRecurrentSheaf(nn.Module):
    def __init__(self, frozen_gpt2, target_layers=[8, 9, 10, 11]):
        super().__init__()
        self.config = frozen_gpt2.config
        self.backbone = frozen_gpt2
        self.lm_head = frozen_gpt2.lm_head
        self.target_layers = set(target_layers)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for i in target_layers:
            original_block = self.backbone.transformer.h[i]
            self.backbone.transformer.h[i] = RecurrentSheafBlock(original_block, self.config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # Recurrent layer doesn't return diagnostics in this simple ver.
        # But we can add energy logging if needed.
        return outputs.logits, {}