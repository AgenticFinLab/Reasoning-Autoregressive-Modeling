"""NLCP (Next-Level Concept Pyramid) Core Modules.

This module implements the core components of NLCP architecture.
Reference: concept-pyramid.md Section 3 - Core Mechanisms Detailed Design
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from examples.nlcp.base import NLCPModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: concept-pyramid.md Section 3.4
    "RMSNorm stabilizes heterogeneous statistics (DLCM Eq.16)"

    RMSNorm normalizes without mean centering, which is more efficient
    and works well for stabilizing attention across different levels.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Dimension Flow:
            Input: [B, L, D] or [B, num_heads, L, head_dim]
                ↓
            Compute RMS: sqrt(mean(x^2) + eps)
                ↓
            Normalize: x / RMS * weight
                ↓
            Output: [B, L, D] or [B, num_heads, L, head_dim]

        Args:
            x: Input tensor of any shape with last dimension hidden_dim

        Returns:
            Normalized tensor with same shape as input
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class DepthGate(nn.Module):
    """Dynamic Depth Gate for controlling pyramid depth.

    Reference: concept-pyramid.md Section 3.2
    Dynamic Depth Gate formula:
        p_cont^(k) = σ(MLP_2(GELU(MLP_1(Pool(H_k)))))

    This module evaluates whether the current latent representation
    is sufficient to support final decoding, or if more refinement
    through additional levels is needed.

    Attributes:
        pool: Learnable global attention pooling layer
        mlp1: First MLP layer
        mlp2: Second MLP layer producing the probability
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        # Learnable pooling via attention mechanism
        # Output: [B, 1, D] global representation
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_key = nn.Linear(hidden_dim, hidden_dim)
        self.pool_value = nn.Linear(hidden_dim, hidden_dim)

        # MLP layers per Section 3.2 formula
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mlp2 = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute continuation probability.

        Dimension Flow:
            H_k: [B, L_k, D] level hidden states
                ↓
            Pool: attention pooling over sequence
                ↓
            pooled: [B, 1, D] global representation
                ↓
            MLP1 + GELU: [B, 1, 2D]
                ↓
            MLP2 + Sigmoid: [B, 1, 1]
                ↓
            p_cont: scalar probability ∈ [0, 1]

        Args:
            hidden_states: [B, L_k, D] level hidden representations
            attention_mask: Optional mask for padding positions

        Returns:
            p_cont: [B, 1] continuation probability
        """
        B, L, D = hidden_states.shape

        # Attention-based pooling
        # Query: [1, 1, D] -> expand to [B, 1, D]
        pool_q = self.pool_query.expand(B, -1, -1)

        # Keys and Values from hidden states
        pool_k = self.pool_key(hidden_states)  # [B, L, D]
        pool_v = self.pool_value(hidden_states)  # [B, L, D]

        # Attention scores: [B, 1, L]
        attn_scores = torch.matmul(pool_q, pool_k.transpose(-2, -1)) / math.sqrt(D)

        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Pooled representation: [B, 1, D]
        pooled = torch.matmul(attn_weights, pool_v)

        # MLP per Section 3.2 formula
        # MLP_1 + GELU
        hidden = F.gelu(self.mlp1(pooled))
        hidden = self.dropout(hidden)

        # MLP_2 + Sigmoid
        p_cont = torch.sigmoid(self.mlp2(hidden))

        return p_cont.squeeze(-1)  # [B, 1]


class ExpansionPredictor(nn.Module):
    """Content-Adaptive Expansion Rate Predictor.

    Reference: concept-pyramid.md Section 3.3
    Expansion rate formula:
        λ_k = Softplus(MLP(H_k)) ∈ [1, ∞)^{L_k}
        expand_mask_k = ⌊λ_k⌋
        L_{k+1} = Σ expand_mask_k[i]

    This module predicts the expansion granularity for each position
    in the coarse level, determining how many fine-level slots each
    position should expand into.

    Attributes:
        mlp: MLP network for expansion prediction
        expansion_min: Minimum expansion rate (default 1)
        expansion_max: Maximum expansion rate (default 8)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_min: int,
        expansion_max: int,
        dropout: float,
    ):
        super().__init__()
        self.expansion_min = expansion_min
        self.expansion_max = expansion_max

        # MLP for expansion rate prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict expansion rates for each position.

        Dimension Flow:
            H_k: [B, L_k, D] level hidden states
                ↓
            MLP: [B, L_k, 1]
                ↓
            Softplus: [B, L_k, 1] positive values
                ↓
            Clamp: [B, L_k, 1] bounded to [expansion_min, expansion_max]
                ↓
            floor: [B, L_k] discrete expansion counts
                ↓
            L_{k+1} = sum(expand_mask) total next level length

        Args:
            hidden_states: [B, L_k, D] level hidden representations
            temperature: Temperature for softening predictions during training

        Returns:
            expand_mask: [B, L_k] integer expansion counts per position
            lambda_k: [B, L_k] continuous expansion rates (for loss computation)
        """
        # MLP prediction
        logits = self.mlp(hidden_states).squeeze(-1)  # [B, L_k]

        # Softplus to ensure positive values, then apply temperature
        lambda_k = F.softplus(logits / temperature)

        # Clamp to valid range
        lambda_k = torch.clamp(lambda_k, self.expansion_min, self.expansion_max)

        # Discrete expansion mask (floor operation)
        expand_mask = torch.floor(lambda_k).long()

        # Ensure at least expansion_min
        expand_mask = torch.clamp(expand_mask, min=self.expansion_min)

        return expand_mask, lambda_k


class CrossLevelCausalAttention(nn.Module):
    """Cross-Level Causal Attention mechanism.

    Reference: concept-pyramid.md Section 3.4
    Causal Cross-Level Attention with Concept Replication.

    This module implements the cross-attention between fine level (query)
    and coarse level (key/value), using DLCM's Concept Replication trick
    to align irregular L_k × L_{k+1} mappings to standard L_{k+1} × L_{k+1}
    causal attention.

    Key insight from Section 3.4:
        "repeat_interleave makes irregular mapping degenerate to standard
        L_{k+1} × L_{k+1} Causal Mask"

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        q_proj: Query projection (from fine level)
        k_proj: Key projection (from coarse level)
        v_proj: Value projection (from coarse level)
        o_proj: Output projection
        q_norm: RMSNorm for query (DLCM Eq.16)
        k_norm: RMSNorm for key (DLCM Eq.16)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections per Section 3.4 code
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # RMSNorm for QK normalization (DLCM Eq.16)
        self.q_norm = RMSNorm(hidden_dim)
        self.k_norm = RMSNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def _repeat_interleave_batch(
        self,
        x: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Repeat interleave with batch dimension handling.

        Dimension Flow:
            x: [B, L_k, D] coarse level tensor
            expand_mask: [B, L_k] expansion counts per position
                ↓
            For each batch element:
                repeat_interleave along sequence dimension
                ↓
            Result: [B, L_{k+1}, D] where L_{k+1} = sum(expand_mask)

        Args:
            x: [B, L_k, D] input tensor
            expand_mask: [B, L_k] expansion counts

        Returns:
            output: [B, L_{k+1}, D] repeated tensor
        """
        batch_size = x.size(0)
        results = []

        for b in range(batch_size):
            # Get expansion counts for this batch element
            repeats = expand_mask[b].cpu()  # Move to CPU for repeat_interleave
            # Ensure repeats are non-negative integers
            repeats = torch.clamp(repeats, min=0).long()

            # Apply repeat_interleave for this batch element
            repeated = torch.repeat_interleave(x[b], repeats, dim=0)
            results.append(repeated)

        # Pad to maximum length across batch
        max_len = max(r.size(0) for r in results)
        padded_results = []
        for r in results:
            if r.size(0) < max_len:
                padding = torch.zeros(
                    max_len - r.size(0), r.size(-1), device=r.device, dtype=r.dtype
                )
                r = torch.cat([r, padding], dim=0)
            padded_results.append(r)

        return torch.stack(padded_results, dim=0)

    def forward(
        self,
        hidden_states_fine: torch.Tensor,
        hidden_states_coarse: torch.Tensor,
        expand_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-level causal attention.

        Dimension Flow (from Section 3.4):
            Coarse H_k: [B, L_k, D]
                ↓
            K_k = H_k @ W_K: [B, L_k, D]
            V_k = H_k @ W_V: [B, L_k, D]
                ↓
            K_rep = repeat_interleave(K_k, expand_mask): [B, L_{k+1}, D]
            V_rep = repeat_interleave(V_k, expand_mask): [B, L_{k+1}, D]
                ↓
            Fine Q_{k+1} = H_{k+1} @ W_Q: [B, L_{k+1}, D]
                ↓
            Q' = RMSNorm(Q), K' = RMSNorm(K_rep)
                ↓
            AttnOut = FlashAttn(Q', K', V_rep, causal=True): [B, L_{k+1}, D]
                ↓
            Output = AttnOut @ W_O + H_{k+1}: [B, L_{k+1}, D]

        Args:
            hidden_states_fine: [B, L_{k+1}, D] fine level hidden states
            hidden_states_coarse: [B, L_k, D] coarse level hidden states
            expand_mask: [B, L_k] expansion counts per coarse position
            attention_mask: Optional causal mask

        Returns:
            output: [B, L_{k+1}, D] attention output with residual
        """
        B = hidden_states_fine.size(0)

        # Project queries from fine level
        q = self.q_proj(hidden_states_fine)  # [B, L_{k+1}, D]

        # Project keys and values from coarse level
        k_coarse = self.k_proj(hidden_states_coarse)  # [B, L_k, D]
        v_coarse = self.v_proj(hidden_states_coarse)  # [B, L_k, D]

        # Concept Replication: repeat_interleave to align with fine level
        # This is the core DLCM trick from Eq.17
        # Handle batch dimension: repeat_interleave each batch element separately
        k_rep = self._repeat_interleave_batch(k_coarse, expand_mask)  # [B, L_{k+1}, D]
        v_rep = self._repeat_interleave_batch(v_coarse, expand_mask)  # [B, L_{k+1}, D]

        # RMSNorm for QK stabilization (DLCM Eq.16) - apply BEFORE reshaping
        q = self.q_norm(q)  # [B, L_{k+1}, D]
        k = self.k_norm(k_rep)  # [B, L_{k+1}, D]

        # Reshape for multi-head attention
        # [B, L, D] -> [B, num_heads, L, head_dim]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_rep.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation with causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask (upper triangular = -inf)
        L_fine = attn_weights.size(-2)
        causal_mask = torch.triu(
            torch.full((L_fine, L_fine), float("-inf"), device=attn_weights.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, L_{k+1}, head_dim]

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, -1, self.hidden_dim)
        )

        # Output projection + residual (DLCM Eq.14)
        output = self.o_proj(attn_output) + hidden_states_fine

        return output


class SelfAttentionBlock(nn.Module):
    """Self-Attention Block with Causal Masking.

    Reference: concept-pyramid.md Section 3.4
    "Standard FlashAttention (Varlen compatible)"
    "Fine level Self-Attn Query"

    Standard causal self-attention for within-level processing.
    Used in Next-Level Generator for fine-grained reasoning within each level.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        q_proj, k_proj, v_proj: QKV projections
        o_proj: Output projection
        q_norm, k_norm: RMSNorm for QK normalization
        mlp: Feed-forward network
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Self-attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # RMSNorm
        self.q_norm = RMSNorm(hidden_dim)
        self.k_norm = RMSNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # MLP (standard 4x expansion)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.ln1 = RMSNorm(hidden_dim)
        self.ln2 = RMSNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Apply causal self-attention.

        Dimension Flow:
            H: [B, L, D] hidden states
                ↓
            Self-Attn with causal mask: [B, L, D]
                ↓
            + residual: [B, L, D]
                ↓
            MLP: [B, L, D]
                ↓
            + residual: [B, L, D]

        Args:
            hidden_states: [B, L, D] input hidden states
            kv_cache: Optional list of [K_cache, V_cache] for incremental generation
            use_cache: Whether to return updated KV cache

        Returns:
            output: [B, L, D] output hidden states
            kv_cache: Updated KV cache if use_cache=True
        """
        B, L, D = hidden_states.shape
        residual = hidden_states

        # Pre-norm
        hidden_states = self.ln1(hidden_states)

        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Handle KV cache for incremental generation
        if kv_cache is not None and len(kv_cache) == 2:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        new_kv_cache = None
        if use_cache:
            new_kv_cache = [k, v]

        # RMSNorm for QK - apply BEFORE reshaping
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape for multi-head attention
        # [B, L, D] -> [B, num_heads, L, head_dim]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        seq_len = attn_weights.size(-2)
        kv_len = attn_weights.size(-1)
        causal_mask = torch.triu(
            torch.full((seq_len, kv_len), float("-inf"), device=attn_weights.device),
            diagonal=kv_len - seq_len + 1,
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, D)

        # Output projection + residual
        hidden_states = self.o_proj(attn_output) + residual

        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual

        return hidden_states, new_kv_cache


class NextLevelGenerator(nn.Module):
    """Next-Level Generator for hierarchical concept generation.

    Reference: concept-pyramid.md Section 3.4
    "Fine level generation is not coarse level upsampling, but a
    strictly conditional autoregressive process on coarse level"

    Reference: concept-pyramid.md Section 2.2 Table
    "With coarse level as condition, autoregressively generate
    fine level concept representations"

    This module generates the next level's hidden representations
    conditioned on the current level through cross-level attention
    and self-attention blocks.

    Attributes:
        cross_attn: Cross-level causal attention
        self_attn_layers: Stack of self-attention layers
        ln: Final layer norm
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.cross_attn = CrossLevelCausalAttention(hidden_dim, num_heads, dropout)
        self.self_attn_layers = nn.ModuleList(
            [
                SelfAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = RMSNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        coarse_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
        kv_cache: Optional[List[List[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[List[torch.Tensor]]]]:
        """Generate next level hidden states.

        Dimension Flow:
            H_{k+1} init: [B, L_{k+1}, D] (typically zeros or learned embedding)
                ↓
            Cross-Level Attn(H_{k+1}, H_k, expand_mask): [B, L_{k+1}, D]
                ↓
            Self-Attn layers × N: [B, L_{k+1}, D]
                ↓
            RMSNorm: [B, L_{k+1}, D]

        Args:
            hidden_states: [B, L_{k+1}, D] initial fine level states
            coarse_hidden_states: [B, L_k, D] coarse level states
            expand_mask: [B, L_k] expansion counts
            kv_cache: Optional KV caches for each self-attention layer
            use_cache: Whether to return updated caches

        Returns:
            output: [B, L_{k+1}, D] generated fine level representations
            new_kv_cache: Updated KV caches if use_cache=True
        """
        # Cross-level attention injects coarse level prior
        hidden_states = self.cross_attn(
            hidden_states,
            coarse_hidden_states,
            expand_mask,
        )

        # Self-attention for fine-grained reasoning
        new_kv_cache = [] if use_cache else None
        for i, self_attn in enumerate(self.self_attn_layers):
            layer_kv = (
                kv_cache[i] if kv_cache is not None and i < len(kv_cache) else None
            )
            hidden_states, layer_new_kv = self_attn(
                hidden_states,
                kv_cache=layer_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_cache.append(layer_new_kv)

        # Final normalization
        hidden_states = self.ln(hidden_states)

        return hidden_states, new_kv_cache


class TokenDecoder(nn.Module):
    """Token Decoder for vocabulary projection.

    Reference: concept-pyramid.md Section 2.2 Table
    "Latent space → discrete vocabulary mapping"

    Reference: concept-pyramid.md Section 4.2
    "Output layer scaling: logits = (1/s_token)(H_K @ W_unemb^T)
    ensures logits magnitude is O(1) (DLCM Eq.21)"

    This module projects the final level's hidden states to vocabulary
    logits for autoregressive token generation.

    Attributes:
        lm_head: Linear projection to vocabulary
        muP_scale: Output scaling factor for μP
    """

    def __init__(self, hidden_dim: int, vocab_size: int, muP_scale: float):
        super().__init__()
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.muP_scale = muP_scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Dimension Flow:
            H_K: [B, L_K, D] final level hidden states
                ↓
            Linear projection: [B, L_K, V]
                ↓
            μP scaling: [B, L_K, V] logits with O(1) magnitude

        Args:
            hidden_states: [B, L_K, D] final level hidden representations

        Returns:
            logits: [B, L_K, V] vocabulary logits
        """
        # Linear projection to vocabulary
        logits = self.lm_head(hidden_states)

        # μP scaling per DLCM Eq.21
        logits = logits / self.muP_scale

        return logits


class LightweightEncoder(nn.Module):
    """Lightweight Encoder for initial token encoding.

    Reference: concept-pyramid.md Section 2.1
    "Input: Question Q (Token IDs) ↓ [Lightweight Encoder]"

    Reference: concept-pyramid.md Section 2.2 Table
    "Encoder: Input x ∈ [1, L_q] → Output H_0 ∈ [1, L_0, D]"
    "Extract fine-grained local representations, initialize global intent"

    This encoder processes input tokens and produces the initial
    Level 0 hidden representations.

    Attributes:
        token_embedding: Token embedding layer
        pos_embedding: Positional embeddings
        encoder_layers: Transformer encoder layers
        ln: Final layer norm
        proj_to_l0: Projection to Level 0 length
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        l0_length: int,
        dropout: float,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.l0_length = l0_length

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                SelfAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = RMSNorm(hidden_dim)

        # Pooling to L_0 length
        self.pool_to_l0 = nn.AdaptiveAvgPool1d(l0_length)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input tokens to Level 0 hidden states.

        Dimension Flow:
            input_ids: [B, L_q] token IDs
                ↓
            Token embedding: [B, L_q, D]
                ↓
            + Position embedding: [B, L_q, D]
                ↓
            Encoder layers: [B, L_q, D]
                ↓
            Pool to L_0: [B, D, L_q] → [B, D, L_0] → [B, L_0, D]
                ↓
            H_0: [B, L_0, D] Level 0 hidden states

        Args:
            input_ids: [B, L_q] input token IDs
            attention_mask: Optional attention mask

        Returns:
            H_0: [B, L_0, D] Level 0 hidden representations
        """
        B, L = input_ids.shape

        # Token + position embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        hidden_states = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Encoder layers
        for layer in self.encoder_layers:
            hidden_states, _ = layer(hidden_states)

        hidden_states = self.ln(hidden_states)

        # Pool to Level 0 length
        # [B, L_q, D] -> [B, D, L_q] -> pool -> [B, D, L_0] -> [B, L_0, D]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pool_to_l0(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states
