"""NLCP V3 Attentive Pooling: Training-Only Concept Extraction from CoT.

USAGE:
    from nlcpV3.attentive_pooling import ResidualAttentivePooling
    from nlcpV3.config import NLCPV3Config

    config = NLCPV3Config(...)
    pooling = ResidualAttentivePooling(config, encoder_hidden_dim=896)

    # Training: Extract concepts from H (from Q+CoT)
    concepts, H_hat, H_rest = pooling(H)
    # concepts = [C_0, C_1, ..., C_K]

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2.2.2: Attentive Pooling (Training Only)
    - Section 3.2: Stage 1 - Concept Extraction

PURPOSE:
    Extract hierarchical concepts from CoT representation during training.
    This module is TRAINING-ONLY because it requires CoT as input!

    During inference, ConceptGenerator is used instead (generates concepts
    from Q without CoT).

ARCHITECTURE:
    Residual Attentive Pooling with soft boundaries:

    For each level k:
        1. Compute attention: A_k = softmax(H_rest @ W_k)
        2. Extract concepts: C_k = A_k @ H_rest
        3. Reconstruct: H_recon_k = C_k @ A_k^T
        4. Update: H_hat += H_recon_k, H_rest -= H_recon_k

    This creates a residual decomposition where each level extracts
    concepts from the remaining un-encoded information.

DIMENSION FLOW:
    Input: H [B, L, D_encoder] (from Encoder)

    Process:
        H_rest = H (initial residual)
        H_hat = 0 (initial reconstruction)

        For k in [0, 1, ..., K-1]:
            H_proj = Linear(D_encoder → D)(H_rest)  [B, L, D]
            A_k = softmax(H_proj @ Q_k)  [B, L_k, L]
            C_k = A_k @ H_proj  [B, L_k, D]
            H_recon_k = C_k @ A_k^T  [B, L, D]
            H_hat += H_recon_k
            H_rest -= H_recon_k

    Output:
        concepts: [C_0, C_1, ..., C_K] where C_k [B, L_k, D]
        H_hat: [B, L, D] (accumulated reconstruction)
        H_rest: [B, L, D] (final residual)

CRITICAL NOTE:
    This module is ONLY used during training. At inference, we use
    ConceptGenerator to generate concepts from Q (no CoT available).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV3.config import NLCPV3Config


class ResidualAttentivePooling(nn.Module):
    """Residual Attentive Pooling for hierarchical concept extraction.

    PURPOSE:
        Extract hierarchical concepts from CoT during training using
        residual attentive pooling with soft boundaries.

    ATTRIBUTES:
        config: NLCPV3Config instance
        input_proj: Projection from encoder dim to concept dim
        concept_queries: Learnable queries for each level [K, max_L, D]
        scale: Temperature scaling for attention

    DIMENSION FLOW:
        Constructor:
            config, encoder_hidden_dim → initializes projections and queries

        Forward:
            H [B, L, D_encoder] → concepts [C_0, ..., C_K], H_hat, H_rest
    """

    def __init__(self, config: NLCPV3Config, encoder_hidden_dim: int):
        """Initialize Attentive Pooling module.

        Args:
            config: NLCPV3Config with hidden_dim, num_levels, level_lengths
            encoder_hidden_dim: Hidden dimension from encoder (e.g., 896)

        PURPOSE:
            Initialize projection layers and learnable concept queries.
        """
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = encoder_hidden_dim

        # Project from encoder hidden dim to concept hidden dim
        self.input_proj = nn.Linear(encoder_hidden_dim, config.hidden_dim)

        # Learnable concept queries for each level
        # Each level k has L_k queries of dimension D
        self.concept_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(length, config.hidden_dim))
                for length in config.level_lengths
            ]
        )

        # Temperature scaling for attention
        self.scale = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform.

        PURPOSE:
            Proper initialization for stable training.
        """
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for queries in self.concept_queries:
            nn.init.xavier_uniform_(queries)

    def forward(
        self, H: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Extract hierarchical concepts from hidden states.

        PURPOSE:
            Perform residual attentive pooling to extract K levels of concepts.
            Each level extracts concepts from the residual of previous levels.

        DIMENSION FLOW:
            Input:
                H: [B, L, D_encoder] - Hidden states from encoder

            Process:
                1. Project H to concept dimension: H_proj [B, L, D]
                2. Initialize H_rest = H_proj, H_hat = 0
                3. For each level k:
                    - Compute attention: A_k = softmax(Q_k @ H_rest^T / scale)
                    - Extract concepts: C_k = A_k @ H_rest [B, L_k, D]
                    - Reconstruct: H_recon = C_k @ A_k^T [B, L, D]
                    - Update: H_hat += H_recon, H_rest -= H_recon

            Output:
                concepts: List of [C_0, C_1, ..., C_K] where C_k [B, L_k, D]
                H_hat: [B, L, D] - Accumulated reconstruction
                H_rest: [B, L, D] - Final residual

        Args:
            H: Hidden states [B, L, D_encoder]

        Returns:
            concepts: List of concept tensors [C_0, ..., C_K]
            H_hat: Reconstructed hidden states [B, L, D]
            H_rest: Residual hidden states [B, L, D]
        """
        batch_size = H.shape[0]
        seq_len = H.shape[1]

        # Project to concept dimension
        H_proj = self.input_proj(H)

        # Initialize residual and reconstruction
        H_rest = H_proj
        H_hat = torch.zeros_like(H_proj)

        concepts = []

        # Extract concepts level by level
        for level_idx in range(self.config.num_levels):
            # Get queries for this level
            Q_k = self.concept_queries[level_idx]  # [L_k, D]
            L_k = Q_k.shape[0]

            # Expand queries for batch
            Q_k_batch = Q_k.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L_k, D]

            # Compute attention scores: Q @ H_rest^T
            scores = torch.bmm(Q_k_batch, H_rest.transpose(1, 2))  # [B, L_k, L]
            scores = scores / (math.sqrt(self.config.hidden_dim) * self.scale)

            # Softmax over sequence dimension
            A_k = F.softmax(scores, dim=-1)  # [B, L_k, L]

            # Extract concepts: C_k = A_k @ H_rest
            C_k = torch.bmm(A_k, H_rest)  # [B, L_k, D]
            concepts.append(C_k)

            # Reconstruct: H_recon = C_k @ A_k^T
            H_recon = torch.bmm(C_k, A_k.transpose(1, 2))  # [B, L, D]

            # Update accumulated reconstruction and residual
            H_hat = H_hat + H_recon
            H_rest = H_rest - H_recon

        return concepts, H_hat, H_rest

    def get_concept_shapes(self) -> list[tuple[int, int]]:
        """Get shapes of concepts for each level.

        PURPOSE:
            Utility for debugging and buffer allocation.

        Returns:
            List of (L_k, D) tuples for each level
        """
        return [
            (length, self.config.hidden_dim) for length in self.config.level_lengths
        ]
