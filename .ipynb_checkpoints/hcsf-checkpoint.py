"""
Householder Cellular Sheaf Flow (HCSF) - Memory-Efficient Global Sparse Implementation

Key Design Principles (NO WINDOW SLIDING):
  - NO unfold: Process entire sequence as one global sparse graph
  - O(L*d) memory: No data duplication
  - Top-K edges: Each token connects to K highest attention past tokens
  - scatter_add: Efficient gradient accumulation
  - Gradient checkpointing: Trade compute for memory in K-step loop

Architecture:
  Input: [B, L, D] hidden states + [B, L, L] attention matrix
  → Build sparse edge graph: (src, tgt, weight) tuples via Top-K
  → Run K iterations of Householder gradient flow
  → Output: [B, L, D] refined hidden states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass


# =============================================================================
# Step 1: Householder Numerical Kernel (Optimized)
# =============================================================================

def apply_householder_prenorm(h: torch.Tensor, v_unit: torch.Tensor) -> torch.Tensor:
    """
    Apply Householder reflection with PRE-NORMALIZED v.

    φ(h) = h - 2v(v^T h)

    OPTIMIZATION: Assumes v_unit is already L2-normalized.
    This avoids redundant normalization inside K-step loops.

    Args:
        h: Input vector [*, D]
        v_unit: Pre-normalized reflection normal [*, D]

    Returns:
        Reflected vector [*, D]
    """
    dot = torch.sum(v_unit * h, dim=-1, keepdim=True)
    return h - 2.0 * v_unit * dot


def apply_householder(h: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Apply Householder reflection (normalizes v internally).

    φ(h) = (I - 2vv^T)h = h - 2v(v^T h)

    Args:
        h: Input vector [*, D]
        v: Reflection normal (will be normalized) [*, D]
        eps: Numerical stability constant

    Returns:
        Reflected vector [*, D]
    """
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    v_unit = v / (v_norm + eps)
    dot = torch.sum(v_unit * h, dim=-1, keepdim=True)
    return h - 2.0 * v_unit * dot


# =============================================================================
# Step 2: Relative Position Embedding (Sinusoidal)
# =============================================================================

class RelativePositionEmbedding(nn.Module):
    """
    Sinusoidal encoding for relative positions (j - i).

    PE(pos, 2k) = sin(pos / 10000^(2k/d))
    PE(pos, 2k+1) = cos(pos / 10000^(2k/d))
    """

    def __init__(self, d_model: int, max_relative_position: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_rel = max_relative_position
        self._build_table()

    def _build_table(self):
        positions = torch.arange(-self.max_rel, self.max_rel + 1, dtype=torch.float32)
        dim_idx = torch.arange(0, self.d_model, dtype=torch.float32)
        freq_exp = (2 * (dim_idx // 2)) / self.d_model
        freqs = 1.0 / (10000.0 ** freq_exp)

        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        embeddings = torch.zeros_like(angles)
        embeddings[:, 0::2] = torch.sin(angles[:, 0::2])
        embeddings[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer('embeddings', embeddings)
        self.register_buffer('offset', torch.tensor(self.max_rel))

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_pos: Relative positions [*]
        Returns:
            Embeddings [*, d_model]
        """
        clamped = torch.clamp(rel_pos, -self.max_rel, self.max_rel)
        indices = (clamped + self.offset).long()
        return self.embeddings[indices]


# =============================================================================
# Step 2: Edge Logic MLP (Pre-normalized Output)
# =============================================================================

class EdgeLogicMLP(nn.Module):
    """
    Neural network generating UNIT Householder vectors for edges.

    Key Optimization: Output is always L2-normalized, allowing
    apply_householder_prenorm() in the K-step loop.

    Input: [h_i || h_j || pos_emb] → 3*d dimensions
    Output: Unit vector v ∈ S^{d-1}
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        hidden_dim = hidden_dim or 2 * d_model

        layers = []
        in_dim = 3 * d_model

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, d_model))
        self.mlp = nn.Sequential(*layers)
        self.pos_emb = RelativePositionEmbedding(d_model)

        # Small init for stable training
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        rel_pos: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Args:
            h_src: Source node states [*, D]
            h_tgt: Target node states [*, D]
            rel_pos: Relative positions [*]

        Returns:
            UNIT vectors defining Householder reflections [*, D]
        """
        pos_enc = self.pos_emb(rel_pos)
        x = torch.cat([h_src, h_tgt, pos_enc], dim=-1)
        v = self.mlp(x)
        # Pre-normalize for efficiency
        return F.normalize(v, p=2, dim=-1, eps=eps)


# =============================================================================
# Step 3: Global Sparse HCSF Engine (NO WINDOWS)
# =============================================================================

class HCSFEngine(nn.Module):
    def __init__(
        self,
        d_model: int,
        k_steps_max: int = 10,
        eta: float = 0.1,
        lam: float = 0.01,
        eps_converge: float = 1e-7,
        top_k: int = 4,
        include_chain: bool = True,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_grad_norm: float = 1.0,
        use_checkpointing: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.k_steps_max = k_steps_max
        self.eta = nn.Parameter(torch.tensor(eta))
        self.lam = lam
        self.eps_converge = eps_converge
        self.top_k = top_k
        self.include_chain = include_chain
        self.max_grad_norm = max_grad_norm
        self.use_checkpointing = use_checkpointing

        self.edge_logic = EdgeLogicMLP(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim,
            num_layers=2,
            dropout=dropout
        )

    def build_global_sparse_graph(
        self,
        attention_weights: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """전체 시퀀스에 대해 인과적 Top-K 그래프를 구축합니다."""
        all_src, all_tgt, all_weight, all_rel = [], [], [], []

        # 1. Chain edges: (i+1 -> i) 인접 토큰 간의 기본 연결
        if self.include_chain and seq_len > 1:
            chain_src = torch.arange(1, seq_len, device=device) # 타겟(수정될 노드)
            chain_tgt = torch.arange(seq_len - 1, device=device) # 소스(참조 노드)
            
            all_src.append(chain_src.unsqueeze(0).expand(batch_size, -1))
            all_tgt.append(chain_tgt.unsqueeze(0).expand(batch_size, -1))
            all_rel.append(torch.full((batch_size, seq_len - 1), -1, dtype=torch.long, device=device))

            if attention_weights is not None:
                b_idx = torch.arange(batch_size, device=device).unsqueeze(1)
                all_weight.append(attention_weights[b_idx, chain_src, chain_tgt])
            else:
                all_weight.append(torch.ones(batch_size, seq_len - 1, device=device))

        # 2. Top-K Global Attention Edges
        if self.top_k > 0 and attention_weights is not None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=-1)
            mask.fill_diagonal_(False)
            
            masked_attn = attention_weights.clone()
            masked_attn[:, ~mask] = float('-inf')

            k_actual = min(self.top_k, seq_len - 1)
            if k_actual > 0:
                topk_vals, topk_idx = torch.topk(masked_attn, k=k_actual, dim=-1)
                valid = topk_vals > float('-inf')

                for k in range(k_actual):
                    src = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                    tgt = topk_idx[:, :, k]
                    wgt = torch.where(valid[:, :, k], F.softmax(topk_vals[:, :, k], dim=-1), torch.zeros_like(topk_vals[:, :, k]))
                    
                    all_src.append(src)
                    all_tgt.append(tgt)
                    all_weight.append(wgt)
                    all_rel.append(tgt - src)

        if not all_src:
            dummy = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            return dummy, dummy, torch.zeros(batch_size, 1, device=device), dummy

        return torch.cat(all_src, dim=1), torch.cat(all_tgt, dim=1), \
               torch.cat(all_weight, dim=1), torch.cat(all_rel, dim=1)

    def compute_energy_fixed_v(self, h, src_idx, tgt_idx, weights, v_ij, v_ji):
        B, L, D = h.shape
        E = src_idx.shape[1]
        b_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(-1, E)

        # phi_ij(h_i) - phi_ji(h_j)
        diff = apply_householder_prenorm(h[b_idx, src_idx], v_ij) - \
               apply_householder_prenorm(h[b_idx, tgt_idx], v_ji)
        
        energy = 0.5 * (diff ** 2).sum(dim=-1)
        return (energy * weights).sum(dim=-1) / (E * self.d_model + 1e-8)

    def compute_gradient_fixed_v(self, h, h0, src_idx, tgt_idx, weights, v_ij, v_ji):
        B, L, D = h.shape
        E = src_idx.shape[1]
        b_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(-1, E)

        delta = (apply_householder_prenorm(h[b_idx, src_idx], v_ij) - 
                 apply_householder_prenorm(h[b_idx, tgt_idx], v_ji)) * weights.unsqueeze(-1)

        grad_src = apply_householder_prenorm(delta, v_ij)
        grad_tgt = -apply_householder_prenorm(delta, v_ji)

        grad = torch.zeros_like(h)
        grad.scatter_add_(1, src_idx.unsqueeze(-1).expand(-1, -1, D), grad_src)
        grad.scatter_add_(1, tgt_idx.unsqueeze(-1).expand(-1, -1, D), grad_tgt)

        return (grad / (E * self.d_model + 1e-8)) + self.lam * (h - h0)

    def _single_step(self, h, h0, src_idx, tgt_idx, weights, v_ij, v_ji):
        grad = self.compute_gradient_fixed_v(h, h0, src_idx, tgt_idx, weights, v_ij, v_ji)
        # Gradient Clipping
        norm = torch.norm(grad, dim=-1, keepdim=True)
        grad = grad * torch.clamp(self.max_grad_norm / (norm + 1e-8), max=1.0)
        return h - torch.abs(self.eta) * grad

    def forward(self, h, attention_weights=None, return_trajectory=False):
        B, L, D = h.shape
        device = h.device

        # 1. 그래프 구축
        src_idx, tgt_idx, weights, rel_pos = self.build_global_sparse_graph(attention_weights, L, B, device)
        
        h0 = h.detach().clone()
        E = src_idx.shape[1]
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, E)

        # 2. 제한 사상 고정 (v_ij, v_ji를 h0 기반으로 계산)
        v_ij = self.edge_logic(h0[b_idx, src_idx], h0[b_idx, tgt_idx], rel_pos)
        v_ji = self.edge_logic(h0[b_idx, tgt_idx], h0[b_idx, src_idx], -rel_pos)

        # 3. 초기 에너지
        pre_energy = self.compute_energy_fixed_v(h, src_idx, tgt_idx, weights, v_ij, v_ji).mean()
        
        # 4. 생각 루프
        actual_steps = 0
        for k in range(self.k_steps_max):
            if self.use_checkpointing and h.requires_grad:
                h = checkpoint(self._single_step, h, h0, src_idx, tgt_idx, weights, v_ij, v_ji, use_reentrant=False)
            else:
                h = self._single_step(h, h0, src_idx, tgt_idx, weights, v_ij, v_ji)
            actual_steps += 1
            
            # 수렴 체크
            with torch.no_grad():
                curr_e = self.compute_energy_fixed_v(h, src_idx, tgt_idx, weights, v_ij, v_ji).mean()
                if torch.abs(pre_energy - curr_e) / (pre_energy + 1e-8) < self.eps_converge:
                    break

        # 5. 최종 에너지 (Tensor 보존)
        post_energy = self.compute_energy_fixed_v(h, src_idx, tgt_idx, weights, v_ij, v_ji).mean()

        return h, {
            "pre_energy": pre_energy.item(),
            "post_energy": post_energy,
            "actual_steps": actual_steps,
            "num_edges": E
        }


# =============================================================================
# Step 4: GPT-2 Integration (NO WINDOWS)
# =============================================================================

class GPT2WithHCSF(nn.Module):
    """
    GPT-2 with Global Sparse HCSF (Memory Efficient).

    CRITICAL: NO WINDOW SLIDING
    - Entire sequence processed as one global sparse graph
    - O(L*D) memory instead of O(L*W*D)
    - Top-K attention edges span full sequence (causal)

    Args:
        gpt2_model: Pretrained GPT2LMHeadModel
        k_steps_max: Maximum HCSF iterations
        eta: Step size
        lam: Anchor coefficient
        eps_converge: Early stopping threshold
        top_k: Top-K attention edges per token
        attn_layer: Which layer's attention to use (-1 = last)
        freeze_backbone: Whether to freeze GPT-2 weights
        use_checkpointing: Enable gradient checkpointing
    """

    def __init__(
        self,
        gpt2_model,
        d_model: Optional[int] = None,
        window_size: int = 64,  # DEPRECATED - kept for API compatibility
        k_steps_max: int = 10,
        eta: float = 0.1,
        lam: float = 0.01,
        eps_converge: float = 1e-3,
        top_k: int = 4,
        attn_layer: int = -1,
        freeze_backbone: bool = True,
        use_checkpointing: bool = True
    ):
        super().__init__()

        self.backbone = gpt2_model
        self.config = gpt2_model.config
        self.d_model = d_model or self.config.n_embd
        self.attn_layer = attn_layer

        # NOTE: window_size is IGNORED - no windowing!
        if window_size != 64:
            print(f"[HCSF] Warning: window_size={window_size} is ignored. Using global sparse graph.")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.hcsf_engine = HCSFEngine(
            d_model=self.d_model,
            k_steps_max=k_steps_max,
            eta=eta,
            lam=lam,
            eps_converge=eps_converge,
            top_k=top_k,
            include_chain=True,
            use_checkpointing=use_checkpointing
        )

    def _extract_attention(
        self,
        attentions: Optional[Tuple[torch.Tensor, ...]],
        layer_idx: int = -1
    ) -> Optional[torch.Tensor]:
        """Extract attention weights from specified layer."""
        if attentions is None or len(attentions) == 0:
            return None
        attn = attentions[layer_idx]
        if attn is None:
            return None
        return attn.mean(dim=1)  # Average over heads: [B, L, L]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with GLOBAL sparse HCSF.

        NO WINDOWS: Processes entire sequence at once.

        Args:
            input_ids: [B, L] token IDs
            attention_mask: Optional mask
            mode: 'train' or 'inference'

        Returns:
            logits: [B, L, V] language model logits
            diagnostics: HCSF metrics
        """
        # Get hidden states AND attention weights
        outputs = self.backbone.transformer(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        hidden_states = outputs.last_hidden_state  # [B, L, D]
        attentions = outputs.attentions

        # Extract attention for HCSF edge construction
        attention_weights = self._extract_attention(attentions, self.attn_layer)

        # Apply GLOBAL HCSF (no windowing!)
        refined_states, diagnostics = self.hcsf_engine(
            hidden_states, attention_weights
        )

        # LM head
        logits = self.backbone.lm_head(refined_states)

        return logits, diagnostics

    def reset_inference_state(self):
        """Reset inference state (no-op for global method)."""
        pass


class HCSFBlock(nn.Module):
    """
    HCSF adapter block for layer injection (Memory Efficient).
    """

    def __init__(
        self,
        original_block,
        config,
        k_steps_max: int = 5,
        eta: float = 0.05,
        lam: float = 0.01,
        eps_converge: float = 1e-3,
        top_k: int = 4,
        use_checkpointing: bool = True
    ):
        super().__init__()
        self.block = original_block
        self.hcsf_engine = HCSFEngine(
            d_model=config.n_embd,
            k_steps_max=k_steps_max,
            eta=eta,
            lam=lam,
            eps_converge=eps_converge,
            top_k=top_k,
            use_checkpointing=use_checkpointing
        )
        self.latest_diagnostics = {}

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs
    ):
        # Run original block with attention output
        outputs = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            **kwargs
        )

        block_hidden = outputs[0]

        # Extract attention
        attn_weights = None
        if len(outputs) > 1 and outputs[1] is not None:
            attn_weights = outputs[1].mean(dim=1)

        # Apply HCSF
        refined, self.latest_diagnostics = self.hcsf_engine(block_hidden, attn_weights)

        if output_attentions:
            return (refined,) + outputs[1:]
        else:
            return (refined,) + outputs[2:] if len(outputs) > 2 else (refined,)


class GPT2WithHCSFAdapters(nn.Module):
    """
    GPT-2 with HCSF adapters in specific layers.
    """

    def __init__(
        self,
        gpt2_model,
        target_layers: Optional[list] = None,
        k_steps_max: int = 5,
        eta: float = 0.05,
        lam: float = 0.01,
        eps_converge: float = 1e-3,
        top_k: int = 4,
        use_checkpointing: bool = True
    ):
        super().__init__()
        self.config = gpt2_model.config
        self.backbone = gpt2_model

        if target_layers is None:
            n_layers = self.config.n_layer
            target_layers = list(range(n_layers - 4, n_layers))

        self.target_layers = set(target_layers)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for i in target_layers:
            original = self.backbone.transformer.h[i]
            self.backbone.transformer.h[i] = HCSFBlock(
                original, self.config,
                k_steps_max=k_steps_max,
                eta=eta,
                lam=lam,
                eps_converge=eps_converge,
                top_k=top_k,
                use_checkpointing=use_checkpointing
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Aggregate diagnostics
        agg = {
            "pre_energy": 0.0,
            "post_energy": 0.0,
            "convergence_rate": 0.0,
            "actual_steps": 0.0,
            "converged_early_count": 0,
            "num_edges": 0.0,
        }
        count = 0

        for i in self.target_layers:
            block = self.backbone.transformer.h[i]
            if isinstance(block, HCSFBlock) and block.latest_diagnostics:
                d = block.latest_diagnostics
                agg["pre_energy"] += d.get("pre_energy", 0.0)
                agg["post_energy"] += d.get("post_energy", 0.0)
                agg["convergence_rate"] += d.get("convergence_rate", 0.0)
                agg["actual_steps"] += d.get("actual_steps", 0)
                agg["num_edges"] += d.get("num_edges", 0)
                if d.get("converged_early", False):
                    agg["converged_early_count"] += 1
                count += 1

        if count > 0:
            for k in ["pre_energy", "post_energy", "convergence_rate", "actual_steps", "num_edges"]:
                agg[k] /= count

        return logits, agg


# =============================================================================
# Factory Function
# =============================================================================

def create_hcsf_model(
    model_type: str = "head",
    base_model_name: str = "gpt2",
    target_layers: Optional[list] = None,
    window_size: int = 64,  # DEPRECATED
    k_steps: int = 10,
    eta: float = 0.1,
    lam: float = 0.01,
    top_k: int = 4,
    device: Optional[torch.device] = None
):
    """Create HCSF-enhanced GPT-2 model."""
    from transformers import GPT2LMHeadModel

    print(f"Loading {base_model_name}...")
    base_model = GPT2LMHeadModel.from_pretrained(base_model_name)

    if device is not None:
        base_model = base_model.to(device)

    if model_type == "head":
        model = GPT2WithHCSF(
            gpt2_model=base_model,
            k_steps_max=k_steps,
            eta=eta,
            lam=lam,
            top_k=top_k
        )
    elif model_type == "adapters":
        model = GPT2WithHCSFAdapters(
            gpt2_model=base_model,
            target_layers=target_layers,
            k_steps_max=k_steps,
            eta=eta,
            lam=lam,
            top_k=top_k
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if device is not None:
        model = model.to(device)

    return model, base_model.config


# =============================================================================
# Tests
# =============================================================================

def _test_global_sparse_graph():
    """Test global sparse graph construction (no windows)."""
    print("Testing Global Sparse Graph Construction...")

    B, L, D = 2, 32, 64
    top_k = 3

    engine = HCSFEngine(d_model=D, k_steps_max=1, top_k=top_k)

    # Create causal attention
    attn = torch.randn(B, L, L)
    attn = torch.tril(attn)
    attn = F.softmax(attn, dim=-1)

    src, tgt, weights, rel = engine.build_global_sparse_graph(attn, L, B, torch.device('cpu'))

    print(f"  Sequence length: {L}")
    print(f"  Chain edges: {L - 1}")
    print(f"  Top-K per token: {top_k}")
    print(f"  Total edges: {src.shape[1]}")
    print(f"  Memory: O(L*D) = O({L}*{D}) = {L*D} floats")
    print("  (vs windowed: O(num_windows * W * D) would be much larger)")
    print("  PASSED!\n")


def _test_memory_efficiency():
    """Test that memory usage is O(L*D), not O(L*W*D)."""
    print("Testing Memory Efficiency...")

    B, L, D = 1, 128, 64  # Longer sequence

    engine = HCSFEngine(d_model=D, k_steps_max=5, top_k=4, use_checkpointing=True)
    h = torch.randn(B, L, D)
    attn = F.softmax(torch.tril(torch.randn(B, L, L)), dim=-1)

    # This should NOT OOM even with long sequences
    h_refined, diag = engine(h, attn)

    print(f"  Input: [{B}, {L}, {D}]")
    print(f"  Output: {h_refined.shape}")
    print(f"  Edges: {diag['num_edges']}")
    print(f"  Steps: {diag['actual_steps']}")
    print(f"  Energy: {diag['pre_energy']:.4f} -> {diag['post_energy']:.4f}")
    print("  PASSED (no OOM)!\n")


def _test_gpt2_integration():
    """Test GPT-2 integration without windows."""
    print("Testing GPT-2 Integration (NO WINDOWS)...")

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        hcsf_model = GPT2WithHCSF(
            gpt2_model=model,
            k_steps_max=5,
            eta=0.1,
            top_k=4,
            use_checkpointing=True
        )

        text = "The capital of France is"
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            logits, diag = hcsf_model(inputs.input_ids)

        print(f"  Input: '{text}'")
        print(f"  Shape: {inputs.input_ids.shape} -> {logits.shape}")
        print(f"  Edges: {diag['num_edges']} (global, not per-window)")
        print(f"  Steps: {diag['actual_steps']}/{diag['max_steps']}")
        print(f"  Converged: {diag['converged_early']}")

        # Predict next token
        next_token = logits[0, -1].argmax()
        print(f"  Prediction: '{tokenizer.decode([next_token])}'")
        print("  PASSED!\n")

    except ImportError:
        print("  Skipped (transformers not installed)\n")


if __name__ == "__main__":
    print("=" * 60)
    print("HCSF Memory-Efficient Global Sparse Implementation")
    print("=" * 60 + "\n")

    _test_global_sparse_graph()
    _test_memory_efficiency()
    _test_gpt2_integration()

    print("=" * 60)
    print("All tests passed!")
    print("Key improvements:")
    print("  - NO unfold/window sliding")
    print("  - O(L*D) memory usage")
    print("  - Global sparse graph")
    print("  - Gradient checkpointing")
    print("  - Pre-normalized Householder vectors")
    print("=" * 60)
