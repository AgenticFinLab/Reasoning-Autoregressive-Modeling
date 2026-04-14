"""NLCP V3 Concept Generator: Inference-Only Concept Generation from Q.

USAGE:
    from nlcpV3.concept_generator import ConceptGenerator
    from nlcpV3.config import NLCPV3Config

    config = NLCPV3Config(...)
    generator = ConceptGenerator(config, encoder_hidden_dim=896)

    # Inference: Generate concepts from H (from Q only, no CoT!)
    concepts = generator(H)
    # concepts = [C_0, C_1, ..., C_K]

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2.2.3: Concept Generator (Inference)
    - Section 3.3: Stage 2 - Concept Generator Distillation

PURPOSE:
    Generate hierarchical concepts from Q during inference (NO CoT available).
    This module replaces AttentivePooling at inference time.

    Training: Distill from AttentivePooling to learn concept generation
    Inference: Generate concepts autoregressively (next-level)

ARCHITECTURE:
    Next-Level Autoregressive Generation:

    Step 0: C_0 = Generator_0(H)           [B, 1, D]
    Step 1: C_1 = Generator_1(H, C_0)      [B, 2, D]
    Step 2: C_2 = Generator_2(H, C_0, C_1) [B, 4, D]
    ...
    Step K: C_K = Generator_K(H, C_<K)     [B, 2^K, D]

    Each Generator_k is a cross-attention layer that attends to H and
    previously generated concepts.

DIMENSION FLOW:
    Input: H [B, L, D_encoder] (from Encoder, Q only)

    Process:
        For k in [0, 1, ..., K-1]:
            Context = Concat(H, C_0, C_1, ..., C_{k-1})
            Q_k = LearnableQuery(k)  [B, L_k, D]
            C_k = CrossAttention(Q_k, Context, Context)  [B, L_k, D]

    Output: concepts [C_0, C_1, ..., C_K] where C_k [B, L_k, D]

CRITICAL NOTE:
    This module is ONLY used during inference. During training, we use
    AttentivePooling to extract concepts from CoT, and train ConceptGenerator
    to match AttentivePooling output (distillation).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV3.config import NLCPV3Config


class ConceptGenerator(nn.Module):
    """Concept Generator for inference-time concept generation from Q.

    PURPOSE:
        Generate hierarchical concepts from Q (no CoT) during inference.
        Uses next-level autoregressive generation.

    ATTRIBUTES:
        config: NLCPV3Config instance
        input_proj: Projection from encoder dim to concept dim
        level_generators: List of level-specific generators

    DIMENSION FLOW:
        Constructor:
            config, encoder_hidden_dim → initializes generators

        Forward:
            H [B, L, D_encoder] → concepts [C_0, ..., C_K]
    """

    def __init__(self, config: NLCPV3Config, encoder_hidden_dim: int):
        """Initialize Concept Generator.

        Args:
            config: NLCPV3Config with hidden_dim, num_levels, level_lengths
            encoder_hidden_dim: Hidden dimension from encoder

        PURPOSE:
            Initialize level-specific concept generators.
        """
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = encoder_hidden_dim

        # Project from encoder hidden dim to concept hidden dim
        self.input_proj = nn.Linear(encoder_hidden_dim, config.hidden_dim)

        # Level-specific generators
        self.level_generators = nn.ModuleList(
            [
                LevelConceptGenerator(
                    level_idx=k,
                    num_concepts=config.level_lengths[k],
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                )
                for k in range(config.num_levels)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, H: torch.Tensor) -> list[torch.Tensor]:
        """Generate hierarchical concepts from hidden states.

        PURPOSE:
            Generate concepts level by level (next-level autoregressive).
            Each level's concepts depend on all previous levels.

        DIMENSION FLOW:
            Input:
                H: [B, L, D_encoder] - Hidden states from encoder (Q only)

            Process:
                1. Project H to concept dimension: H_proj [B, L, D]
                2. For each level k:
                    - Concatenate H_proj with all previous concepts
                    - Generate C_k using level-specific generator

            Output:
                concepts: List of [C_0, C_1, ..., C_K] where C_k [B, L_k, D]

        Args:
            H: Hidden states [B, L, D_encoder]

        Returns:
            concepts: List of concept tensors [C_0, ..., C_K]
        """
        batch_size = H.shape[0]

        # Project to concept dimension
        H_proj = self.input_proj(H)  # [B, L, D]

        concepts = []

        # Generate concepts level by level
        for level_idx, generator in enumerate(self.level_generators):
            # Prepare context: H + all previous concepts
            if level_idx == 0:
                context = H_proj  # [B, L, D]
            else:
                prev_concepts = torch.cat(concepts, dim=1)  # [B, sum(L_<k), D]
                context = torch.cat(
                    [H_proj, prev_concepts], dim=1
                )  # [B, L + sum(L_<k), D]

            # Generate concepts for this level
            C_k = generator(context)  # [B, L_k, D]
            concepts.append(C_k)

        return concepts


class LevelConceptGenerator(nn.Module):
    """Level-specific concept generator.

    PURPOSE:
        Generate concepts for a specific level using cross-attention.

    ATTRIBUTES:
        level_idx: Level index (k)
        num_concepts: Number of concepts for this level (L_k)
        hidden_dim: Concept dimension (D)
        queries: Learnable queries [L_k, D]
        attention: Multi-head cross-attention
    """

    def __init__(
        self, level_idx: int, num_concepts: int, hidden_dim: int, num_heads: int
    ):
        """Initialize level concept generator.

        Args:
            level_idx: Level index
            num_concepts: Number of concepts to generate
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.level_idx = level_idx
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim

        # Learnable queries for this level
        self.queries = nn.Parameter(torch.randn(num_concepts, hidden_dim))

        # Multi-head attention
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.queries)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Generate concepts for this level.

        PURPOSE:
            Use cross-attention to generate concepts from context.

        DIMENSION FLOW:
            Input:
                context: [B, L_ctx, D] - H + previous concepts

            Process:
                1. Expand queries: [B, L_k, D]
                2. Compute Q, K, V projections
                3. Multi-head attention
                4. Output projection + residual

            Output:
                C_k: [B, L_k, D] - Concepts for this level

        Args:
            context: Context tensor [B, L_ctx, D]

        Returns:
            C_k: Concepts [B, L_k, D]
        """
        batch_size = context.shape[0]
        L_k = self.num_concepts

        # Expand queries for batch
        Q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L_k, D]

        # Project Q, K, V
        Q = self.q_proj(Q)  # [B, L_k, D]
        K = self.k_proj(context)  # [B, L_ctx, D]
        V = self.v_proj(context)  # [B, L_ctx, D]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, L_k, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L_k, d]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L_ctx, d]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L_ctx, d]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, H, L_k, L_ctx]
        attn = F.softmax(scores, dim=-1)  # [B, H, L_k, L_ctx]

        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, H, L_k, d]

        # Reshape back
        out = (
            out.transpose(1, 2).contiguous().view(batch_size, L_k, self.hidden_dim)
        )  # [B, L_k, D]

        # Output projection
        out = self.out_proj(out)  # [B, L_k, D]

        # Residual connection and layer norm
        out = self.layer_norm(
            out + Q.transpose(1, 2).contiguous().view(batch_size, L_k, self.hidden_dim)
        )

        return out
