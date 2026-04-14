"""NLCP V2 Concept Transformer Module.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2.2.3: Concept Transformer
    - Section 2.1.3: Training Data Flow (Stage 3)

PURPOSE:
    Learn to predict hierarchical concepts from coarse to fine.
    Implements VAR-style "Next-Scale" pattern:
    - Inter-level causal: H_k must be fully generated before H_{k+1} starts
    - Intra-level parallel: All positions in H_k generated in parallel

ARCHITECTURE PRINCIPLE:
    Unlike standard transformers that generate token-by-token,
    Concept Transformer generates concept-level-by-concept-level.
    Each level has its own concept count L_k and is generated as a whole.

LEVEL-LEVEL CAUSALITY:
    This is the core design principle from VAR adapted to concepts:
    - Level k is treated as a single unit
    - Level k+1 can only attend to level k (not within level k+1)
    - Within level k, all L_k concepts are generated in parallel

DIMENSION FLOW:
    Input:  H_0 [B, L_0, D] from Encoder (inference) or Attentive Pooling (training)
        ↓
    Level 0: Transformer processing → H_0 refined [B, L_0, D]
        ↓
    Level 1: Project + Generate → H_1 [B, L_1, D]
        ↓
    ...
        ↓
    Level K-1: Project + Generate → H_{K-1} [B, L_{K-1}, D]
        ↓
    Output: List of [B, L_k, D] for k = 0, 1, ..., K-1
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV2.config import NLCPV2Config
from nlcpV2.utils import create_causal_mask, rms_norm


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.3

    PURPOSE:
        Implements causal self-attention where each position can only attend to
        previous positions, maintaining autoregressive property.

    MATHEMATICAL FORMULATION:
        Given input X ∈ R^{L×D}:

        1. Linear projections:
           Q = X @ W_q^T    [B, L, D]
           K = X @ W_k^T    [B, L, D]
           V = X @ W_v^T    [B, L, D]

        2. Reshape for multi-head:
           Q = reshape(Q)   [B, H, L, d_head]
           K = reshape(K)   [B, H, L, d_head]
           V = reshape(V)   [B, H, L, d_head]
           where d_head = D / H

        3. Attention scores:
           scores = Q @ K^T / sqrt(d_head)    [B, H, L, L]

        4. Apply causal mask:
           scores[i, j] = -inf if j > i (future positions masked)

        5. Softmax and weighted sum:
           attn = softmax(scores)             [B, H, L, L]
           output = attn @ V                  [B, H, L, d_head]

        6. Reshape and project:
           output = reshape(output)           [B, L, D]
           output = output @ W_o^T            [B, L, D]

    DIMENSION LEGEND:
        B: Batch size
        L: Sequence length (concept count at current level)
        D: Hidden dimension (config.hidden_dim)
        H: Number of heads (config.num_heads)
        d_head: Per-head dimension (D / H)

    Attributes:
        hidden_dim: Model hidden dimension D
        num_heads: Number of attention heads H
        head_dim: Dimension per head d_head = D / H
        w_q: Query projection [D, D]
        w_k: Key projection [D, D]
        w_v: Value projection [D, D]
        w_o: Output projection [D, D]
    """

    def __init__(self, config: NLCPV2Config):
        """Initialize multi-head attention.

        INITIALIZATION:
            Creates four linear projections (Q, K, V, O) and dropout.
            All projections map D → D (total 4D² parameters).

        Args:
            config: NLCPV2Config with hidden_dim and num_heads
        """
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.w_q = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.w_k = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.w_v = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.w_o = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking.

        DIMENSION FLOW:
            Input x: [B, L, D]
                ↓
            Q, K, V projections: [B, L, D]
                ↓
            Reshape: [B, L, H, d_head] → [B, H, L, d_head]
                ↓
            Attention scores: [B, H, L, L]
                ↓
            Apply causal mask + softmax: [B, H, L, L]
                ↓
            Weighted sum: [B, H, L, d_head]
                ↓
            Reshape: [B, L, D]
                ↓
            Output projection: [B, L, D]

        Args:
            x: [B, L, D] Input tensor
            causal_mask: [L, L] Boolean causal mask
                True = can attend, False = cannot attend

        Returns:
            [B, L, D] Attention output
        """
        batch_size, seq_len, _ = x.shape

        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)

        output = self.w_o(output)

        return output


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with SwiGLU activation.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.1 (Encoder)

    PURPOSE:
        Standard transformer FFN using SwiGLU activation.
        SwiGLU is more expressive than standard ReLU/GELU.

    MATHEMATICAL FORMULATION:
        SwiGLU(x) = (x @ W_gate^T * SiLU(x @ W_gate^T)) @ W_down^T

        Or equivalently:
        gate = SiLU(x @ W_gate^T)    [B, L, 4D]
        up = x @ W_up^T              [B, L, 4D]
        hidden = gate * up           [B, L, 4D] (element-wise)
        output = hidden @ W_down^T   [B, L, D]

    DIMENSION FLOW:
        Input:  [B, L, D]
            ↓
        Gate projection: [B, L, 4D]
        Up projection:   [B, L, 4D]
            ↓
        SiLU activation on gate
            ↓
        Element-wise multiply: [B, L, 4D]
            ↓
        Down projection: [B, L, D]

    Attributes:
        w_gate: Gate projection [D, 4D]
        w_up: Up projection [D, 4D]
        w_down: Down projection [4D, D]
    """

    def __init__(self, config: NLCPV2Config):
        """Initialize FFN.

        INITIALIZATION:
            Creates three linear projections:
            - w_gate: D → 4D
            - w_up: D → 4D
            - w_down: 4D → D

        Args:
            config: NLCPV2Config with hidden_dim
        """
        super().__init__()
        hidden_dim = config.hidden_dim
        intermediate_dim = 4 * hidden_dim

        self.w_gate = nn.Linear(hidden_dim, intermediate_dim)
        self.w_up = nn.Linear(hidden_dim, intermediate_dim)
        self.w_down = nn.Linear(intermediate_dim, hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU.

        DIMENSION FLOW:
            Input x: [B, L, D]
                ↓
            gate = SiLU(W_gate(x)): [B, L, 4D]
            up = W_up(x): [B, L, 4D]
                ↓
            hidden = gate * up: [B, L, 4D]
                ↓
            output = W_down(hidden): [B, L, D]

        Args:
            x: [B, L, D] Input tensor

        Returns:
            [B, L, D] Output tensor
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.w_down(hidden)

        return output


class ConceptTransformerLayer(nn.Module):
    """Single transformer layer for concept generation.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.3

    ARCHITECTURE:
        Standard transformer layer with pre-normalization:
        1. RMSNorm → Multi-head self-attention (causal) → Residual
        2. RMSNorm → Feed-forward network (SwiGLU) → Residual

    PRE-NORMALIZATION:
        Unlike post-norm (original Transformer), pre-norm normalizes before
        each sublayer, improving training stability for deep networks.

    DIMENSION FLOW:
        Input: [B, L, D]
            ↓
        Norm1 + Attention + Residual
            ↓
        [B, L, D]
            ↓
        Norm2 + FFN + Residual
            ↓
        Output: [B, L, D]
    """

    def __init__(self, config: NLCPV2Config):
        """Initialize transformer layer.

        INITIALIZATION:
            Creates attention, FFN, two layer norms, and dropout.

        Args:
            config: NLCPV2Config
        """
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(config)
        self.ffn = FeedForwardNetwork(config)

        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        DIMENSION FLOW:
            Input x: [B, L, D]
                ↓
            normed = Norm1(x): [B, L, D]
            attn_out = Attention(normed, mask): [B, L, D]
            x = x + Dropout(attn_out): [B, L, D]
                ↓
            normed = Norm2(x): [B, L, D]
            ffn_out = FFN(normed): [B, L, D]
            x = x + Dropout(ffn_out): [B, L, D]
                ↓
            Output: [B, L, D]

        Args:
            x: [B, L, D] Input tensor
            causal_mask: [L, L] Causal attention mask

        Returns:
            [B, L, D] Output tensor
        """
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, causal_mask)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x


class ConceptTransformer(nn.Module):
    """Concept Transformer for hierarchical concept generation.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.3

    PURPOSE:
        Generates concepts level by level with level-level causality.

        Training: Learns to predict C_{k+1} from H_k (supervised by Attentive Pooling)
        Inference: Generates H_{k+1} from H_k (autoregressive generation)

    VAR-STYLE "NEXT-SCALE" PATTERN:
        Inter-level causal: H_k must be fully generated before H_{k+1} starts
        Intra-level parallel: All L_k positions in H_k generated simultaneously

        This is different from token-by-token generation:
        - Standard LLM: x_1 → x_2 → x_3 → ... (token-level causal)
        - NLCP: H_0 → H_1 → H_2 → ... (level-level causal)

    LEVEL TRANSITION:
        When moving from level k to k+1:
        1. Project H_k to match dimension
        2. Initialize H_{k+1} (zeros or from projection)
        3. Apply transformer layers with causal mask
        4. Add residual connection from H_k

    Attributes:
        config: NLCPV2Config instance
        layers: List of K transformer layers (one per level)
        level_projs: Projections for level transitions [K-1, D, D]
    """

    def __init__(self, config: NLCPV2Config):
        """Initialize concept transformer.

        INITIALIZATION:
            Creates K transformer layers and K-1 level projection layers.
            Each level has its own transformer layer for level-specific processing.

        Args:
            config: NLCPV2Config
        """
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [ConceptTransformerLayer(config) for _ in range(config.num_levels)]
        )

        self.level_projs = nn.ModuleList(
            [
                nn.Linear(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_levels - 1)
            ]
        )

    def forward(
        self,
        H_0: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Generate all level concepts from initial H_0.

        DESIGN SOURCE: concept-pyramid-V2.md Section 2.1.4 (Inference Data Flow)

        ALGORITHM:
            Input: H_0 [B, L_0, D]
            Initialize: hidden_states_list = [H_0], current = H_0
            For k = 0 to K-2:
                next = generate_level_sequence(current, k)
                hidden_states_list.append(next)
                current = next
            Return hidden_states_list

        DIMENSION FLOW:
            Input: H_0 [B, L_0, D]
                ↓
            Level 0: H_0 (passed through)
                ↓
            Level 1: generate_level_sequence(H_0, 0) → [B, L_1, D]
                ↓
            Level 2: generate_level_sequence(H_1, 1) → [B, L_2, D]
                ↓
            ...
                ↓
            Level K-1: generate_level_sequence(H_{K-2}, K-2) → [B, L_{K-1}, D]
                ↓
            Output: [H_0, H_1, ..., H_{K-1}] (list of K tensors)

        LEVEL-LEVEL CAUSALITY:
            Each H_{k+1} is generated only after H_k is complete.
            This ensures coarse-to-fine generation order.

        Args:
            H_0: [B, L_0, D] Initial concept from encoder
                B: Batch size
                L_0: Concept count at level 0 (config.level_lengths[0])
                D: Hidden dimension

        Returns:
            List of K tensors [B, L_k, D] for k = 0, 1, ..., K-1
                Each tensor represents concepts at that level
        """
        hidden_states_list = [H_0]
        current_hidden = H_0

        for level_idx in range(self.config.num_levels - 1):
            next_hidden = self.generate_level_sequence(
                current_hidden,
                level_idx,
            )
            hidden_states_list.append(next_hidden)
            current_hidden = next_hidden

        return hidden_states_list

    def generate_level_sequence(
        self,
        H_k: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        """Generate next level concept sequence.

        DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.3

        ALGORITHM:
            1. Project H_k to prepare for next level
            2. Initialize H_{k+1} as zeros
            3. Apply transformer layers with causal attention
            4. Add residual from projected H_k

        DIMENSION FLOW:
            Input H_k: [B, L_k, D]
                ↓
            Project: H_k_expanded = Proj(H_k): [B, L_k, D]
                ↓
            Initialize H_next: [B, L_{k+1}, D] (zeros)
                ↓
            Create causal mask: [L_{k+1}, L_{k+1}]
                ↓
            For each layer:
                H_next = Layer(H_next, mask): [B, L_{k+1}, D]
                ↓
            Add residual: H_next = H_next + mean(H_k_expanded): [B, L_{k+1}, D]
                ↓
            Output: [B, L_{k+1}, D]

        WHY MEAN FOR RESIDUAL?
            H_k has L_k concepts, H_{k+1} has L_{k+1} concepts.
            Taking mean of H_k provides a single vector summary,
            which is broadcast to all positions in H_{k+1}.

        Args:
            H_k: [B, L_k, D] Current level hidden states
                B: Batch size
                L_k: Concept count at level k
                D: Hidden dimension
            level_idx: Current level index k (0, 1, ..., K-2)

        Returns:
            [B, L_{k+1}, D] Next level hidden states
                L_{k+1} = config.level_lengths[level_idx + 1]
        """
        batch_size = H_k.size(0)
        L_next = self.config.level_lengths[level_idx + 1]
        device = H_k.device

        H_next = torch.zeros(
            batch_size,
            L_next,
            self.config.hidden_dim,
            device=device,
            dtype=H_k.dtype,
        )

        H_k_expanded = self.level_projs[level_idx](H_k)

        causal_mask = create_causal_mask(L_next, device)

        for layer in self.layers:
            H_next = layer(H_next, causal_mask)

        H_next = H_next + H_k_expanded.mean(dim=1, keepdim=True)

        return H_next
