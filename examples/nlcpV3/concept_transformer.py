"""NLCP V3 Concept Transformer: VAR-Style with Level-Level Causality.

USAGE:
    from nlcpV3.concept_transformer import ConceptTransformer
    from nlcpV3.config import NLCPV3Config

    config = NLCPV3Config(...)
    transformer = ConceptTransformer(config)

    # Refine hierarchical concepts
    refined_concepts = transformer(concepts)
    # refined_concepts = [C'_0, C'_1, ..., C'_K]

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2.2.4: Concept Transformer
    - Section 3.4: Stage 3 - Concept Transformer Training

PURPOSE:
    Refine hierarchical concepts with level-level causality.

    Key constraint: Level-level causality
    - Concept at level k can only attend to levels <= k
    - Within-level: Full attention (parallel)
    - Across-level: Causal (lower levels can see higher levels)

    This is analogous to VAR's scale-level causality but applied to
    concept hierarchy.

ARCHITECTURE:
    VAR-Style Transformer with:
    - Level embedding: Distinguish concept levels
    - Position embedding: Within-level position
    - Level-level causal mask: Enforce hierarchy
    - AdaLN (optional): Adaptive layer normalization

DIMENSION FLOW:
    Input: concepts [C_0, C_1, ..., C_K] where C_k [B, L_k, D]

    Process:
        1. Concatenate all concepts: x [B, sum(L_k), D]
        2. Add level embedding + position embedding
        3. Apply transformer blocks with level-level causal mask
        4. Split back into per-level concepts

    Output: refined_concepts [C'_0, C'_1, ..., C'_K]

CAUSAL MASK:
    concepts = [C_0(1), C_1(2), C_2(4), C_3(8)]  # L_k concepts at level k

    Position:  0 | 1 2 | 3 4 5 6 | 7 8 9 10 11 12 13 14
    Level:     0 | 1 1 | 2 2 2 2 | 3 3 3  3  3  3  3  3

    Mask: C_k[i] can attend to C_j[l] iff j <= k

    This ensures coarse concepts (low k) influence fine concepts (high k)
    but not vice versa.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV3.config import NLCPV3Config


class ConceptTransformer(nn.Module):
    """VAR-Style Transformer for hierarchical concept refinement.

    PURPOSE:
        Refine concepts with level-level causality.
        Coarse concepts influence fine concepts but not vice versa.

    ATTRIBUTES:
        config: NLCPV3Config instance
        level_embedding: Embedding for concept levels
        position_embedding: Positional encoding within levels
        blocks: Transformer blocks with causal attention
        norm: Final layer normalization

    DIMENSION FLOW:
        Constructor:
            config → initializes embeddings and transformer blocks

        Forward:
            concepts [C_0, ..., C_K] → refined_concepts [C'_0, ..., C'_K]
    """

    def __init__(self, config: NLCPV3Config):
        """Initialize Concept Transformer.

        Args:
            config: NLCPV3Config with hidden_dim, num_heads, num_levels, etc.
        """
        super().__init__()
        self.config = config

        # Level embedding: distinguish different concept levels
        self.level_embedding = nn.Embedding(config.num_levels, config.hidden_dim)

        # Position embedding: within-level position encoding
        max_concepts = max(config.level_lengths)
        self.position_embedding = nn.Embedding(max_concepts, config.hidden_dim)

        # Transformer blocks
        self.num_blocks = 4  # Number of transformer layers
        self.blocks = nn.ModuleList(
            [ConceptTransformerBlock(config) for _ in range(self.num_blocks)]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(config.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.level_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, concepts: list[torch.Tensor]) -> list[torch.Tensor]:
        """Refine hierarchical concepts.

        PURPOSE:
            Apply transformer with level-level causality to refine concepts.

        DIMENSION FLOW:
            Input:
                concepts: List [C_0, ..., C_K] where C_k [B, L_k, D]

            Process:
                1. Concatenate concepts: x [B, total_L, D]
                2. Create level indices and position indices
                3. Add level embedding + position embedding
                4. Apply causal mask and transformer blocks
                5. Split back into per-level concepts

            Output:
                refined_concepts: List [C'_0, ..., C'_K]

        Args:
            concepts: List of concept tensors [C_0, ..., C_K]

        Returns:
            refined_concepts: List of refined concept tensors [C'_0, ..., C'_K]
        """
        batch_size = concepts[0].shape[0]

        # Concatenate all concepts
        x = torch.cat(concepts, dim=1)  # [B, total_L, D]
        total_L = x.shape[1]

        # Create level indices
        level_indices = []
        position_indices = []
        for level_idx, C_k in enumerate(concepts):
            L_k = C_k.shape[1]
            level_indices.extend([level_idx] * L_k)
            position_indices.extend(list(range(L_k)))

        level_indices = torch.tensor(level_indices, device=x.device)  # [total_L]
        position_indices = torch.tensor(position_indices, device=x.device)  # [total_L]

        # Expand for batch
        level_indices = level_indices.unsqueeze(0).expand(
            batch_size, -1
        )  # [B, total_L]
        position_indices = position_indices.unsqueeze(0).expand(
            batch_size, -1
        )  # [B, total_L]

        # Add embeddings
        x = (
            x
            + self.level_embedding(level_indices)
            + self.position_embedding(position_indices)
        )

        # Create level-level causal mask
        attn_mask = self._create_level_causal_mask(concepts)  # [total_L, total_L]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Final normalization
        x = self.norm(x)

        # Split back into per-level concepts
        refined_concepts = []
        start_idx = 0
        for C_k in concepts:
            L_k = C_k.shape[1]
            C_k_refined = x[:, start_idx : start_idx + L_k, :]  # [B, L_k, D]
            refined_concepts.append(C_k_refined)
            start_idx += L_k

        return refined_concepts

    def _create_level_causal_mask(self, concepts: list[torch.Tensor]) -> torch.Tensor:
        """Create level-level causal attention mask.

        PURPOSE:
            Create mask where concept at level k can only attend to levels <= k.

        DIMENSION FLOW:
            concepts: List [C_0, ..., C_K]

            Returns:
                mask: [total_L, total_L] where mask[i, j] = True if i can attend to j

        Args:
            concepts: List of concept tensors

        Returns:
            attn_mask: Boolean mask [total_L, total_L]
        """
        # Get level boundaries
        level_boundaries = []
        start = 0
        for C_k in concepts:
            L_k = C_k.shape[1]
            level_boundaries.append((start, start + L_k))
            start += L_k

        total_L = start

        # Create mask
        mask = torch.zeros(total_L, total_L, dtype=torch.bool)

        for i, (start_i, end_i) in enumerate(level_boundaries):
            # Concepts at level i can attend to all concepts at levels <= i
            for j, (start_j, end_j) in enumerate(level_boundaries):
                if j <= i:  # Causal: can attend to same or lower levels
                    mask[start_i:end_i, start_j:end_j] = True

        return mask


class ConceptTransformerBlock(nn.Module):
    """Single transformer block with level-level causal attention.

    PURPOSE:
        Self-attention + FFN with residual connections and layer normalization.

    ATTRIBUTES:
        norm1: Pre-attention layer normalization
        attn: Multi-head self-attention
        norm2: Pre-FFN layer normalization
        ffn: Feed-forward network
    """

    def __init__(self, config: NLCPV3Config):
        """Initialize transformer block."""
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Multi-head attention
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor [B, L, D]
            attn_mask: Attention mask [L, L]

        Returns:
            x: Output tensor [B, L, D]
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        # QKV projection
        qkv = self.qkv_proj(x)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each [B, L, D]

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, H, L, L]

        # Apply mask (where mask is False, set to -inf)
        if attn_mask is not None:
            mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
            scores = scores.masked_fill(~mask_expanded, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, L, d]

        # Reshape and project
        out = (
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )
        out = self.out_proj(out)

        x = residual + out

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x
