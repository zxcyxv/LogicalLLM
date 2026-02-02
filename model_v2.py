import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# Model Configuration for GPT-2 Variants
# =============================================================================
@dataclass
class ModelConfig:
    """Configuration for different GPT-2 model sizes."""
    name: str
    hidden_dim: int
    n_layers: int
    target_layers: List[int]

    @classmethod
    def gpt2_small(cls) -> 'ModelConfig':
        """GPT-2 Small (124M params): 12 layers, 768 hidden dim"""
        return cls(
            name="gpt2",
            hidden_dim=768,
            n_layers=12,
            target_layers=[8, 9, 10, 11]  # Top 4 layers (~33%)
        )

    @classmethod
    def gpt2_medium(cls) -> 'ModelConfig':
        """GPT-2 Medium (355M params): 24 layers, 1024 hidden dim"""
        return cls(
            name="gpt2-medium",
            hidden_dim=1024,
            n_layers=24,
            target_layers=[20, 21, 22, 23]  # Top 4 layers (~17%)
        )

    @classmethod
    def from_name(cls, name: str) -> 'ModelConfig':
        """Get config by model name."""
        configs = {
            "gpt2": cls.gpt2_small,
            "gpt2-small": cls.gpt2_small,
            "gpt2-medium": cls.gpt2_medium,
        }
        if name not in configs:
            raise ValueError(f"Unknown model: {name}. Available: {list(configs.keys())}")
        return configs[name]()


# =============================================================================
# 1. The Engine: Recurrent Sheaf (DGS) - [Winner Model]
# =============================================================================
class RecurrentSheafLayer(nn.Module):
    """
    Phase 3: Diagonal Gated Sheaf (DGS)
    Logic: h_t = decay * h_{t-1} + (1-decay) * gate * (x_t - rho(h_{t-1}))
    Features:
      - Restriction Map (rho): Logic transformation
      - Adaptive Gating (gate): Importance weighting
      - Learnable Decay: Memory management
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Decay Factor (Forget Gate)
        # Learnable decay initialized close to 1 (long memory)
        self.decay = nn.Parameter(torch.ones(hidden_dim) * 0.9)
        
        # 2. Logic Projector (Restriction Map)
        # "문맥 h를 현재 시점의 x공간으로 변환했을 때의 기대값"
        self.restriction = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. Input/Update Gate
        # "이 에러가 얼마나 중요한가?"
        self.update_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        B, L, D = x.size()
        
        # Initial State h_0 = 0
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        
        decay_factor = torch.sigmoid(self.decay) 
        
        # Recurrent Loop (O(L))
        for t in range(L):
            x_t = x[:, t, :] # [B, D]
            
            # Prediction
            pred_x = self.restriction(h)
            
            # Logical Error (Innovation)
            error = x_t - pred_x
            
            # Gating
            z_t = torch.sigmoid(self.update_gate(x_t))
            
            # State Update (Convex Combination)
            # DGS Update Rule
            h = decay_factor * h + (1 - decay_factor) * (z_t * error)
            
            outputs.append(h)
            
        y = torch.stack(outputs, dim=1) # [B, L, D]
        y = self.ln(y)
        
        return self.out_proj(y)


# =============================================================================
# 2. The Baseline: Simple Delta Rule (EMA)
# =============================================================================
class DeltaRuleLayer(nn.Module):
    """
    Augmented Baseline: Gated Linear RNN (Matched Parameters)
    Logic: h_t = decay * h_{t-1} + (1-decay) * gate * transform(x_t)

    Improvements for Fair Comparison:
      - Added 'input_transform' (Matches Sheaf's 'restriction' params)
      - Added 'update_gate' (Matches Sheaf's 'update_gate' params)
      - Total Params are now IDENTICAL to RecurrentSheafLayer.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Decay (Same as Sheaf)
        self.decay = nn.Parameter(torch.ones(hidden_dim) * 0.9)
        
        # 2. Input Transform (Matches Sheaf's Restriction Linear)
        # Sheaf는 h를 변환하지만, Delta는 x를 변환합니다. (단순 정보 가공)
        self.input_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. Input Gate (Matches Sheaf's Gate Linear)
        self.update_gate = nn.Linear(hidden_dim, hidden_dim)
        
        # 4. Out Proj (Same as Sheaf)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, L, D = x.size()
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        
        decay_factor = torch.sigmoid(self.decay)
        
        for t in range(L):
            x_t = x[:, t, :]
            
            # --- Logic Difference is Here ---
            # Sheaf: error = x_t - restriction(h)  (State Feedback)
            # Delta: value = input_transform(x_t)  (Feedforward)
            
            value = self.input_transform(x_t)
            z_t = torch.sigmoid(self.update_gate(x_t))
            
            # Update Rule
            h = decay_factor * h + (1 - decay_factor) * (z_t * value)
            
            outputs.append(h)
            
        y = torch.stack(outputs, dim=1)
        y = self.ln(y)
        return self.out_proj(y)


# =============================================================================
# 3. Model Wrappers (Blocks & GPT-2)
# =============================================================================

class RecurrentSheafBlock(nn.Module):
    def __init__(self, original_block, config):
        super().__init__()
        self.block = original_block
        self.engine = RecurrentSheafLayer(hidden_dim=config.n_embd)

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

        # Apply Sheaf Engine
        correction = self.engine(hidden_states)
        refined_states = hidden_states + correction
        
        return (refined_states,) + outputs[1:]


class DeltaRuleBlock(nn.Module):
    def __init__(self, original_block, config):
        super().__init__()
        self.block = original_block
        self.engine = DeltaRuleLayer(hidden_dim=config.n_embd)

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

        # Apply Delta Rule Engine
        correction = self.engine(hidden_states)
        refined_states = hidden_states + correction
        
        return (refined_states,) + outputs[1:]


# --- Main Model Classes ---

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
        # Diagnostics not implemented for recurrent efficiency
        return outputs.logits, {}


class GPT2WithDeltaRule(nn.Module):
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
            self.backbone.transformer.h[i] = DeltaRuleBlock(original_block, self.config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        return outputs.logits, {}


# =============================================================================
# 4. Model Factory Functions
# =============================================================================

def create_model(
    model_type: str,
    base_model_name: str = "gpt2",
    target_layers: Optional[List[int]] = None,
    device: Optional[torch.device] = None
):
    """
    Factory function to create models with proper configuration.

    Args:
        model_type: One of "baseline", "recurrent_sheaf", "delta_rule"
        base_model_name: HuggingFace model name ("gpt2", "gpt2-medium")
        target_layers: Optional override for target layers
        device: Optional device to place model on

    Returns:
        Tuple of (model, config)
    """
    from transformers import GPT2LMHeadModel

    # Get configuration for the base model
    config = ModelConfig.from_name(base_model_name)

    # Use provided target_layers or default from config
    layers = target_layers if target_layers is not None else config.target_layers

    # Load base model
    print(f"Loading {base_model_name} (hidden_dim={config.hidden_dim}, layers={config.n_layers})...")
    base_model = GPT2LMHeadModel.from_pretrained(base_model_name)

    if device is not None:
        base_model = base_model.to(device)

    # Create the appropriate model
    if model_type == "baseline":
        # Freeze all parameters for baseline
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model, config

    elif model_type == "recurrent_sheaf":
        print(f"Initializing Recurrent Sheaf Engine (target layers: {layers})...")
        model = GPT2WithRecurrentSheaf(base_model, target_layers=layers)
        if device is not None:
            model = model.to(device)
        return model, config

    elif model_type == "delta_rule":
        print(f"Initializing Delta Rule Baseline (target layers: {layers})...")
        model = GPT2WithDeltaRule(base_model, target_layers=layers)
        if device is not None:
            model = model.to(device)
        return model, config

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Available: baseline, recurrent_sheaf, delta_rule")


def count_trainable_params(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)