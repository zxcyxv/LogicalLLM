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

        # [Diagnostics Containers]
        total_energy = torch.tensor(0.0, device=x.device)
        total_gate = torch.tensor(0.0, device=x.device)

        decay_factor = torch.sigmoid(self.decay)

        # Recurrent Loop (O(L))
        for t in range(L):
            x_t = x[:, t, :] # [B, D]

            # Prediction
            pred_x = self.restriction(h)

            # [Metric 1] Logical Inconsistency Energy (Pre-update)
            # Measures "surprise" - how unexpected is x_t given past context h
            raw_error = x_t - pred_x
            energy = raw_error.norm(p=2, dim=-1).mean()  # L2 norm averaged over batch
            total_energy += energy

            # Stabilized Error for Update (using tanh for numerical stability)
            error = torch.tanh(raw_error)

            # Gating
            z_t = torch.sigmoid(self.update_gate(x_t))

            # [Metric 3] Gate Sparsity (Information Filtering Strength)
            # Measures how selective the model is in accepting new information
            total_gate += z_t.mean()

            # State Update (Convex Combination)
            # DGS Update Rule
            h = decay_factor * h + (1 - decay_factor) * (z_t * error)

            outputs.append(h)

        y = torch.stack(outputs, dim=1) # [B, L, D]
        y = self.ln(y)

        # [Metric 2] Restriction Non-Triviality (Transformation Complexity)
        # Measures how different the restriction map is from identity (simple copy)
        w_rho = self.restriction.weight  # [D, D]
        identity = torch.eye(self.hidden_dim, device=x.device)
        rho_complexity = torch.norm(w_rho - identity, p='fro')

        diagnostics = {
            "avg_energy": (total_energy / L).item(),      # Lower = Better prediction
            "avg_gate": (total_gate / L).item(),          # Lower = More selective
            "rho_complexity": rho_complexity.item(),      # Higher = More non-trivial logic
        }

        return self.out_proj(y), diagnostics


# =============================================================================
# 2. The Baseline: DeltaNet (Schlag et al., 2021)
# =============================================================================

class DeltaNetLayer(nn.Module):
    """
    DeltaNet: Associative Memory with Delta Rule Updates (Stabilized)

    Reference: Schlag et al., "Linear Transformers are Secretly Fast Weight Programmers" (2021)

    Mathematical Definition:
        W_t = decay * W_{t-1} + σ(β_t) * (v_t - v̄_t) ⊗ φ(k_t)
        where v̄_t = W_{t-1} @ φ(k_t)

    Stabilizations:
      - Memory decay: Prevents W from exploding in long sequences
      - Feature normalization: Keeps phi_k bounded
      - Small beta init: Conservative learning rate
      - Gradient clipping: Additional safety

    Parameters:
      - proj_k, proj_v: 2 × (d×d) for key/value projections
      - beta_gate: d×1 for learning rate (init very small)
      - decay: learnable memory decay
      - out_proj: d×d for output
      Total: ~3M params per layer
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Query, Key, Value projections (Q for READ, K/V for WRITE)
        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Beta gate (learning rate) - initialize VERY SMALL to prevent explosion
        self.beta_gate = nn.Linear(hidden_dim, 1)
        # Initialize beta to produce ~0.01 initial learning rate
        nn.init.constant_(self.beta_gate.weight, 0.0)
        nn.init.constant_(self.beta_gate.bias, -4.0)  # sigmoid(-4) ≈ 0.018

        # Memory decay factor (learnable, init close to 1 for long memory)
        self.decay = nn.Parameter(torch.ones(1) * 0.99)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # LayerNorm for stability
        self.ln = nn.LayerNorm(hidden_dim)
        self.ln_phi = nn.LayerNorm(hidden_dim)  # Normalize feature map

    def forward(self, x):
        """
        Args:
            x: [B, L, D] input sequence

        Returns:
            y: [B, L, D] output sequence

        DeltaNet Operations:
            1. READ: o_t = W_{t-1} @ φ(q_t)  (retrieve from memory using query)
            2. WRITE: W_t = decay * W_{t-1} + β * (v_t - W_{t-1} @ φ(k_t)) ⊗ φ(k_t)
        """
        B, L, D = x.size()

        # Initialize associative memory matrix W ∈ R^{B, D, D}
        W = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)

        # Memory decay factor (applied each step)
        decay_factor = torch.sigmoid(self.decay)

        # Pre-compute all projections for efficiency
        q_all = self.proj_q(x)  # [B, L, D] - for READ
        k_all = self.proj_k(x)  # [B, L, D] - for WRITE (update)
        v_all = self.proj_v(x)  # [B, L, D] - for WRITE (target)

        # Apply feature map to all queries and keys
        phi_q_all = self.ln_phi(F.elu(q_all) + 1)  # [B, L, D]
        phi_k_all = self.ln_phi(F.elu(k_all) + 1)  # [B, L, D]

        outputs = []

        for t in range(L):
            # Current timestep projections
            phi_q = phi_q_all[:, t, :]  # [B, D] - for READ
            phi_k = phi_k_all[:, t, :]  # [B, D] - for WRITE
            v_t = v_all[:, t, :]        # [B, D] - for WRITE

            # === READ OPERATION ===
            # Retrieve information from memory using QUERY
            # o_t = W_{t-1} @ φ(q_t)
            retrieved = torch.bmm(W, phi_q.unsqueeze(-1)).squeeze(-1)  # [B, D]

            # === WRITE OPERATION (Delta Rule Update) ===

            # 1. Predict using KEY
            v_bar = torch.bmm(W, phi_k.unsqueeze(-1)).squeeze(-1)  # [B, D]

            # 2. Compute prediction error
            error = v_t - v_bar  # [B, D]

            # 3. Compute gated learning rate
            beta = torch.sigmoid(self.beta_gate(x[:, t, :]))  # [B, 1]

            # 4. Update memory: W_t = decay * W_{t-1} + β * (error ⊗ φ(k))
            outer_product = torch.bmm(
                (beta * error).unsqueeze(-1),  # [B, D, 1]
                phi_k.unsqueeze(1)             # [B, 1, D]
            )

            W = decay_factor * W + outer_product  # [B, D, D]

            # === OUTPUT ===
            # Use retrieved content (NOT v_t!)
            outputs.append(retrieved)

        # Stack and post-process
        y = torch.stack(outputs, dim=1)  # [B, L, D]
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
        self.diagnostics = {}  # Store latest diagnostics

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

        # Apply Sheaf Engine with diagnostics
        correction, self.diagnostics = self.engine(hidden_states)
        refined_states = hidden_states + correction

        return (refined_states,) + outputs[1:]


class DeltaNetBlock(nn.Module):
    def __init__(self, original_block, config):
        super().__init__()
        self.block = original_block
        self.engine = DeltaNetLayer(hidden_dim=config.n_embd)
        self.use_checkpoint = True  # Enable gradient checkpointing by default

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

        # Apply DeltaNet Engine with gradient checkpointing to save memory
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            correction = checkpoint(self.engine, hidden_states, use_reentrant=False)
        else:
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

        # Aggregate diagnostics from all Sheaf layers
        diagnostics = {
            "avg_energy": 0.0,
            "avg_gate": 0.0,
            "rho_complexity": 0.0,
        }

        num_sheaf_layers = 0
        for i in self.target_layers:
            block = self.backbone.transformer.h[i]
            if isinstance(block, RecurrentSheafBlock) and hasattr(block, 'diagnostics'):
                for key in diagnostics.keys():
                    diagnostics[key] += block.diagnostics.get(key, 0.0)
                num_sheaf_layers += 1

        # Average across all Sheaf layers
        if num_sheaf_layers > 0:
            for key in diagnostics.keys():
                diagnostics[key] /= num_sheaf_layers

        return outputs.logits, diagnostics


class GPT2WithDeltaNet(nn.Module):
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
            self.backbone.transformer.h[i] = DeltaNetBlock(original_block, self.config)

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
        model_type: One of "baseline", "recurrent_sheaf", "deltanet"
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

    elif model_type == "deltanet":
        print(f"Initializing DeltaNet (Associative Memory, target layers: {layers})...")
        model = GPT2WithDeltaNet(base_model, target_layers=layers)
        if device is not None:
            model = model.to(device)
        return model, config

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Available: baseline, recurrent_sheaf, deltanet")


def count_trainable_params(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)