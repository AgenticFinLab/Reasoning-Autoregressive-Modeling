"""NLCP V2 Residual Attentive Pooling Module.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2.2.2: Attentive Pooling
    - Section 1.4: NLCP Solution - Residual Attentive Pooling

PURPOSE:
    Extract hierarchical concept targets from encoder output during training.
    These targets supervise the Concept Transformer to learn level-by-level
    concept generation.

CORE INNOVATION - Residual Attentive Pooling:
    Unlike hard boundary detection (DLCM) or fixed-scale quantization (VAR),
    NLCP uses soft attention to extract concepts with overlapping boundaries.

MATHEMATICAL FORMULATION:
    For each level k = 0, 1, ..., K-1:

    1. Attention weights:
       A_k = softmax(Q_k @ H_rest^T / sqrt(d))     [B, L_k, L]

    2. Concept extraction (pooling):
       C_k = A_k @ H_rest                          [B, L_k, D]

    3. Reconstruction (projection back to token dim):
       H_k_recon = A_k^T @ C_k                     [B, L, D]

    4. Residual update:
       H_hat = H_hat + H_k_recon                   (accumulate reconstruction)
       H_rest = H_rest - H_k_recon                 (remove encoded information)

DIMENSION LEGEND:
    B: Batch size
    L: Token sequence length (from encoder)
    L_k: Concept count at level k (L_k << L for compression)
    D: Hidden dimension (config.hidden_dim)
    D_encoder: Encoder hidden dimension (may differ from D)

KEY INSIGHT - A_k^T @ C_k:
    This operation projects concepts back to token dimension [L, D],
    similar to VAR's upsampling operation. It enables:
    1. Dimension alignment between concepts and tokens
    2. Soft boundary via attention weights
    3. Differentiable end-to-end training

WHY RESIDUAL?
    Each level encodes "information not expressed by previous levels".
    H_rest gradually shrinks as H_hat accumulates, ensuring:
    - Coarse-to-fine hierarchy
    - No information duplication across levels
    - Complete reconstruction: H_hat ≈ H_proj

TRAINING-ONLY MODULE:
    Attentive Pooling is ONLY used during training because:
    - It requires ground-truth CoT to extract concept targets
    - Inference has no CoT, so Concept Transformer predicts directly
    - This is the key training-inference separation in NLCP V2
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV2.config import NLCPV2Config


class ResidualAttentivePooling(nn.Module):
    """Residual Attentive Pooling for hierarchical concept extraction.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.2

    ARCHITECTURE:
        Implements iterative residual decomposition using attention:
        - Level 0 extracts coarsest concepts from full H
        - Level 1 extracts finer concepts from remaining H_rest
        - ... until Level K-1 extracts finest concepts

    SOFT BOUNDARY PROPERTY:
        Unlike hard segmentation, attention allows overlapping boundaries:
        - One token can contribute to multiple concepts
        - One concept can aggregate from multiple tokens
        - No explicit boundary detection needed

    RESIDUAL PROPERTY:
        Each level encodes new information not in previous levels:
        - H_rest starts as full H_proj
        - After each level: H_rest = H_rest - H_k_recon
        - Final H_rest should be near zero (all information encoded)

    DIMENSION HANDLING:
        Encoder may output different dimension (D_encoder) than concept dim (D).
        Input projection aligns dimensions: [B, L, D_encoder] → [B, L, D]

    Attributes:
        config: NLCPV2Config instance with level_lengths and hidden_dim
        num_levels: Number of pyramid levels K
        input_proj: Projection from encoder dim to concept dim (if needed)
        concept_queries: List of learnable query tensors [L_k, D] for each level
    """

    def __init__(self, config: NLCPV2Config, encoder_hidden_dim: int = None):
        """Initialize attentive pooling module.

        INITIALIZATION PROCESS:
            1. Store configuration
            2. Create input projection if encoder dim differs from concept dim
            3. Initialize learnable concept queries for each level

        CONCEPT QUERIES:
            Each level k has L_k learnable queries, each of dimension D.
            These queries attend to token representations to extract concepts.

            Shape: concept_queries[k] = [L_k, D]

        Args:
            config: NLCPV2Config with level_lengths and hidden_dim
            encoder_hidden_dim: Actual hidden dim from encoder (may differ from config)
                If None, assumes encoder_hidden_dim = config.hidden_dim
        """
        super().__init__()
        self.config = config
        self.num_levels = config.num_levels

        self.encoder_hidden_dim = encoder_hidden_dim or config.hidden_dim
        if self.encoder_hidden_dim != config.hidden_dim:
            self.input_proj = nn.Linear(self.encoder_hidden_dim, config.hidden_dim)
        else:
            self.input_proj = nn.Identity()

        self.concept_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(length, config.hidden_dim))
                for length in config.level_lengths
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Extract hierarchical concepts via residual attentive pooling.

        DESIGN SOURCE: concept-pyramid-V2.md Section 2.1.3 (Training Data Flow)

        ALGORITHM:
            Input: H from encoder [B, L, D_encoder]
            1. Project: H_proj = Proj(H) [B, L, D]
            2. Initialize: H_rest = H_proj, H_hat = 0
            3. For k = 0 to K-1:
                a. Extract concept C_k from H_rest
                b. Reconstruct H_k_recon from C_k
                c. Accumulate: H_hat += H_k_recon
                d. Update residual: H_rest -= H_k_recon
            4. Return concepts, H_hat, H_rest

        DIMENSION FLOW:
            Input:  [B, L, D_encoder]
                ↓
            Projection: [B, L, D]
                ↓
            Iteration k:
                - Extract: [B, L, D] → [B, L_k, D] (concept C_k)
                - Reconstruct: [B, L_k, D] → [B, L, D] (H_k_recon)
            Output:
                - concepts: List of [B, L_k, D] for k = 0, ..., K-1
                - H_hat: [B, L, D] (accumulated reconstruction)
                - H_rest: [B, L, D] (final residual, should be ≈ 0)

        Args:
            hidden_states: [B, L, D_encoder] Token-level hidden states from encoder
                B: Batch size
                L: Token sequence length
                D_encoder: Encoder hidden dimension

        Returns:
            Tuple of (concepts, H_hat, H_rest):
                concepts: List of K tensors, each [B, L_k, D]
                    Concept targets for each level
                H_hat: [B, L, D]
                    Accumulated reconstruction of H_proj
                H_rest: [B, L, D]
                    Final residual (H_proj - H_hat, should be near zero)
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        H_proj = self.input_proj(hidden_states)

        H_rest = H_proj.clone()
        H_hat = torch.zeros_like(H_proj)
        concepts = []

        for level_idx in range(self.num_levels):
            C_k, H_k_recon = self._extract_level(H_rest, level_idx, batch_size, device)

            concepts.append(C_k)
            H_hat = H_hat + H_k_recon
            H_rest = H_rest - H_k_recon

        return concepts, H_hat, H_rest

    def _extract_level(
        self,
        H_rest: torch.Tensor,
        level_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract concept for a single level via attention pooling.

        MATHEMATICAL OPERATIONS:
            1. Query preparation:
               Q_k = concept_queries[level_idx]  [L_k, D]
               Q_k = expand(Q_k, batch_size)     [B, L_k, D]

            2. Attention computation:
               scores = Q_k @ H_rest^T / sqrt(D)  [B, L_k, L]
               A_k = softmax(scores, dim=-1)      [B, L_k, L]

            3. Concept extraction (pooling):
               C_k = A_k @ H_rest                 [B, L_k, D]

            4. Reconstruction (projection back):
               H_k_recon = A_k^T @ C_k            [B, L, D]

        WHY SOFTMAX OVER L (NOT L_k)?
            Each concept attends to all L tokens (soft boundary).
            Softmax over L ensures attention weights sum to 1 per concept.

        WHY A_k^T @ C_k?
            Projects concepts [B, L_k, D] back to token dimension [B, L, D].
            This is the "dual" operation of attention pooling.

        DIMENSION FLOW:
            Input H_rest: [B, L, D]
                ↓
            Q_k: [B, L_k, D]
                ↓
            scores = Q_k @ H_rest^T: [B, L_k, L]
                ↓
            A_k = softmax(scores): [B, L_k, L]
                ↓
            C_k = A_k @ H_rest: [B, L_k, D] (concept extraction)
                ↓
            H_k_recon = A_k^T @ C_k: [B, L, D] (reconstruction)

        Args:
            H_rest: [B, L, D] Residual hidden states
                Contains information not yet encoded by previous levels
            level_idx: Current level index k (0, 1, ..., K-1)
            batch_size: Batch size B
            device: Computation device (CPU/GPU)

        Returns:
            Tuple of (C_k, H_k_recon):
                C_k: [B, L_k, D] Concept target for level k
                    L_k = config.level_lengths[level_idx]
                    Each concept is a weighted sum of tokens
                H_k_recon: [B, L, D] Reconstruction projected back to token dimension
                    Used for residual accumulation
        """
        L_k = self.config.level_lengths[level_idx]

        Q_k = self.concept_queries[level_idx].unsqueeze(0).expand(batch_size, -1, -1)

        scores = torch.matmul(Q_k, H_rest.transpose(1, 2))
        scores = scores / (self.config.hidden_dim**0.5)
        A_k = F.softmax(scores, dim=-1)

        C_k = torch.matmul(A_k, H_rest)

        H_k_recon = torch.matmul(A_k.transpose(1, 2), C_k)

        return C_k, H_k_recon
