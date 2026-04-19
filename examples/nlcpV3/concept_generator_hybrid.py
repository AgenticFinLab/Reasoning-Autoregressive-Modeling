"""NLCP V3 Hybrid Concept Generator: Best-of-All-Worlds Design.

USAGE:
    from nlcpV3.concept_generator_hybrid import HybridConceptGenerator
    from nlcpV3.config import NLCPV3Config

    config = NLCPV3Config(
        hidden_dim=256,
        num_levels=6,
        level_lengths=[1, 2, 4, 8, 16, 32],
        ...
    )

    # Training: Extract concepts from CoT
    generator = HybridConceptGenerator(config, encoder_hidden_dim=896)
    concepts, aux = generator(encoder_hidden_states)  # concepts = [C_0, ..., C_K]

    # Inference: Generate concepts from Q (level by level)
    C_0, aux_0 = generator(encoder_hidden_states, target_level_index=0)
    C_1, aux_1 = generator(encoder_hidden_states, target_level_index=1, previous_level_concepts=[C_0])
    ...

DESIGN SOURCE:
    Based on comprehensive analysis in: examples/nlcpV3/generator-analysis.md
    Combines three best methods identified through innovation × effectiveness ranking:

    1. ResidualAttentivePoolingConceptGenerator (Rank #1)
       - Contribution: VAR-style residual decomposition for coarse-to-fine guarantee
       - Source: concept_generator.py lines 305-486
       - Innovation: Residual decomposition adapted from VAR to text domain

    2. MonotonicSoftAssignmentConceptGenerator (Rank #3)
       - Contribution: Cross-attention with causal context accumulation
       - Source: concept_generator.py lines 1307-1512
       - Innovation: Level-level causal dependency through context accumulation

    3. AutoregressiveSoftBoundaryConceptGenerator (Rank #2)
       - Contribution: Strictly increasing boundary prediction for ordering
       - Source: concept_generator.py lines 1975-2210
       - Innovation: AR boundary prediction with monotonic constraint

RESEARCH GOAL:
    Compress Chain-of-Thought (CoT) into a hierarchical concept space where reasoning
    operates via a coarse-to-fine process. The hybrid design ensures:

    1. Hierarchical abstraction: Coarse concepts (L_0=1) capture high-level reasoning;
       fine concepts (L_K=32) capture details
    2. Causal ordering: Text is sequential — extraction respects CoT causality
    3. Full coverage: Every CoT position contributes to some concept
    4. Training-Inference consistency: Extraction from CoT aligns with generation from Q
    5. Differentiability: All operations end-to-end trainable

ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    HYBRID CONCEPT GENERATOR                             │
    │                                                                         │
    │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
    │  │   RESIDUAL      │    │   CROSS-ATTN    │    │   BOUNDARY      │     │
    │  │   DECOMPOSITION │◄──►│   CONTEXT ACCUM │◄──►│   CONSTRAINT    │     │
    │  │   (Backbone)    │    │   (Bridge)      │    │   (Ordering)    │     │
    │  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
    │           │                      │                      │              │
    │           ▼                      ▼                      ▼              │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │              UNIFIED FORWARD INTERFACE                          │   │
    │  │  Training: forward(H) -> [C_0, ..., C_K], aux                   │   │
    │  │  Inference: forward(H, k, [C_0..C_{k-1}]) -> C_k, aux           │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

MATHEMATICS:
    Let H ∈ ℝ^{B×L×D_encoder} be encoder hidden states from Q+CoT.
    Let K be number of levels, with L_0 < L_1 < ... < L_{K-1} concepts per level.

    VAR Principles (from VAR.md Section 5.2.2):
        f_rest: "what still needs encoding" — decreases each level
        f_hat:  "what has been encoded"    — accumulates each level
        Constraint: f_hat + f_rest = H_proj (exact decomposition)

    CRITICAL DESIGN: Commit vs Refinement Separation
        The residual flow (f_rest, f_hat) must remain pure. Only base-extracted
        concepts (C_k_base) enter the residual flow. Cross-attention refinement
        (refined_k) improves concept quality for the decoder but does NOT
        participate in residual decomposition. This prevents double-counting:

        - C_k_base = level_proj(A_k @ H_rest_k)    # Commit: new info, enters f_rest
        - refined_k = cross_attn(Q_k, context)       # Refine: context-aware, no f_rest
        - C_k = C_k_base + refined_k                 # Output: goes to decoder
        - R_k = A_k^T @ C_k_base                     # Reconstruct from BASE only

    Training Path (All Levels - Parallel):
        H_proj = Linear(H) ∈ ℝ^{B×L×D}                           # Project to concept space

        # Residual Decomposition (from ResidualAttentivePooling)
        H_rest_0 = H_proj
        H_hat_0 = 0

        For k = 0 to K-1:
            Q_k ∈ ℝ^{L_k×D} = concept_queries[k]                 # Learnable queries
            Q_k' = expand(Q_k, batch_size) ∈ ℝ^{B×L_k×D}

            # Attention over residual
            A_k = softmax(Q_k' @ H_rest_k^T / (√D × τ)) ∈ ℝ^{B×L_k×L}
            C_k_base = level_proj(A_k @ H_rest_k) ∈ ℝ^{B×L_k×D}  # BASE concept

            # Reconstruct from BASE only (commit path)
            H_recon_k = A_k^T @ C_k_base ∈ ℝ^{B×L×D}
            H_hat_{k+1} = H_hat_k + H_recon_k                   # f_hat accumulation
            H_rest_{k+1} = H_rest_k - H_recon_k                  # f_rest update

            # Cross-Attention Context (from MonotonicSoftAssignment)
            If k > 0:
                context = concat([H_proj, C_0, ..., C_{k-1}]) ∈ ℝ^{B×(L+ΣL_i)×D}
                refined_k = CrossAttn(Q_k', context, context) ∈ ℝ^{B×L_k×D}
                C_k = C_k_base + refined_k                         # Refined output
            Else:
                C_k = C_k_base

        # Auxiliary Losses
        L_recon = ||H_hat_K - H_proj||²                          # Reconstruction loss

        # Boundary Constraint (from AutoregressiveSoftBoundary)
        Intra-level ordering constraint following VAR scale-level causality:
        Consecutive concept slots within each level are ordered by CoT position:
           L_order = Σ_level Σ_j ReLU(exp_pos[j] - exp_pos[j+1] + margin)

        NOTE: Inter-level ordering (last of level k < first of level k+1) is
        intentionally NOT enforced. The relationship between levels is
        granularity (coarse-to-fine), not sequential ordering. Level k+1
        provides a finer partition of the SAME space that level k covers,
        so level k+1's first concept can attend to earlier positions than
        level k's last concept. Coarse-to-fine is already guaranteed by
        the residual flow and f_hat/f_rest decomposition.

        Total Loss = L_recon + λ_order × L_order

    Inference Path (Next-Level - Sequential):
        Matches training computation but level-by-level.
        Uses _cached_attentions and _cached_base_concepts to compute
        residual from previous levels (f_hat accumulation).

DIMENSION FLOW:
    Input:  H ∈ ℝ^{B×L×D_encoder}
    Output: [C_0, ..., C_{K-1}] where C_k ∈ ℝ^{B×L_k×D}

    Level k processing:
        H_rest_k:      [B, L, D]          (residual hidden states)
        Q_k:           [L_k, D]           (learnable queries)
        Q_k':          [B, L_k, D]        (expanded queries)
        A_k:           [B, L_k, L]        (attention weights)
        C_k:           [B, L_k, D]        (extracted concepts)
        H_recon_k:     [B, L, D]          (reconstruction from C_k)
        context:       [B, L+ΣL_i, D]     (accumulated context for cross-attn)

KEY DIFFERENCES FROM INDIVIDUAL METHODS:

    vs. ResidualAttentivePooling:
        + Added cross-attention refinement (from MonotonicSoftAssignment)
        + Added boundary constraint loss (from AutoregressiveSoftBoundary)
        + Better training-inference alignment through unified interface

    vs. MonotonicSoftAssignment:
        + Added residual decomposition for coarse-to-fine guarantee
        + Added reconstruction loss for information preservation
        + Added boundary constraint for stronger ordering

    vs. AutoregressiveSoftBoundary:
        + Parallel level extraction during training (not sequential)
        + Residual decomposition instead of boundary prediction
        + Cross-attention context instead of boundary masking

IMPLEMENTATION NOTES:
    1. The hybrid uses residual decomposition as the PRIMARY mechanism (best coarse-to-fine)
    2. Cross-attention is used as SECONDARY refinement (best alignment with inference)
    3. Boundary constraint is used as AUXILIARY loss (strongest ordering)
    4. All three mechanisms are DIFFERENTIABLE and jointly trainable

REFERENCES:
    - VAR.md Section 5.2.2: Residual decomposition (f_hat + f_rest)
    - VAR.md Section 6: Next-level autoregressive generation
    - concept-pyramid-V3.md Section 3: Training (Concept Extraction)
    - concept-pyramid-V3.md Section 4: Inference (Concept Generation)
"""

import math
from typing import Optional, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV3.config import NLCPV3Config


# =============================================================================
# Hybrid Concept Generator
# =============================================================================


class HybridConceptGenerator(nn.Module):
    """Hybrid concept generator combining three best methods.

    PURPOSE:
        Implement the optimal hybrid design identified through comprehensive
        analysis of 11 concept generator methods. Combines:
        1. ResidualAttentivePooling (coarse-to-fine backbone)
        2. MonotonicSoftAssignment (cross-attention bridge)
        3. AutoregressiveSoftBoundary (boundary ordering constraint)

    ATTRIBUTES:
        config: NLCPV3Config with hyperparameters
        encoder_hidden_dim: Input dimension from encoder
        input_proj: Projection from encoder_dim to hidden_dim
        concept_queries: Learnable queries for each level [K levels]
        temperature: Learnable attention temperature
        level_projs: Level-specific output projections
        level_attn: Cross-attention layers for context accumulation
        boundary_predictor: MLP for boundary constraint (auxiliary)

    DIMENSION FLOW:
        Constructor:
            config, encoder_hidden_dim → initializes all components

        Training (forward_all_levels):
            H [B, L, D_encoder] → [C_0, ..., C_K] + aux (with losses)

        Inference (forward_next_level):
            H [B, L, D_encoder], k, [C_0..C_{k-1}] → C_k
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        order_loss_weight: float,
        order_margin: float,
        use_positional_query_init: bool,
    ):
        """Initialize Hybrid Concept Generator.

        Args:
            config: NLCPV3Config with all hyperparameters
            encoder_hidden_dim: Dimension of encoder hidden states
            order_loss_weight: Weight for boundary ordering loss (λ_order)
            order_margin: Margin for ordering constraint
            use_positional_query_init: If True, initialize concept queries with
                positional priors so query j within level k is biased toward
                position j/L_k (segment-scoped initialization). If False, use
                random initialization (Xavier uniform). This is an experimental
                option for ablation: positional init may accelerate convergence
                by providing DLCM-style segment-concept correspondence as a
                starting point, while random init lets the model discover
                position structure purely from training signal.

        DIMENSION FLOW:
            Input: config, encoder_hidden_dim
            Output: initialized generator with all parameters
        """
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = encoder_hidden_dim
        self.order_loss_weight = order_loss_weight
        self.order_margin = order_margin
        self.use_positional_query_init = use_positional_query_init

        # =====================================================================
        # Component 1: Projection (shared across all methods)
        # =====================================================================
        self.input_proj = nn.Linear(encoder_hidden_dim, config.hidden_dim)

        # =====================================================================
        # Component 2: Learnable Concept Queries (shared across all methods)
        # =====================================================================
        # Each level has its own set of learnable queries
        # Shape: [L_k, D] for level k, where L_k = config.level_lengths[k]
        self.concept_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(length, config.hidden_dim))
                for length in config.level_lengths
            ]
        )

        # =====================================================================
        # Component 3: ResidualAttentivePooling Components
        # =====================================================================
        # Learnable temperature for attention scaling
        self.temperature = nn.Parameter(torch.ones(1))

        # Level-specific projections for refined concept representation
        self.level_projs = nn.ModuleList(
            [
                nn.Linear(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_levels)
            ]
        )

        # Cache for attention weights and base concepts (used in residual computation)
        # _cached_attentions: stores A_k for each level (used to compute f_rest)
        # _cached_base_concepts: stores C_k_base for each level (used to compute f_hat)
        # Both are populated during inference (forward_next_level)
        self._cached_attentions: List[torch.Tensor] = []
        self._cached_base_concepts: List[torch.Tensor] = []

        # =====================================================================
        # Component 4: MonotonicSoftAssignment Components
        # =====================================================================
        # Cross-attention layers for context accumulation
        # Each level has its own cross-attention mechanism
        self.level_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    batch_first=True,
                )
                for _ in range(config.num_levels)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights using Xavier uniform initialization.

        Concept queries have two initialization modes controlled by
        self.use_positional_query_init:

        1. Random (False): Xavier uniform — no positional prior.
           The model must discover segment structure from training signal.

        2. Positional (True): Xavier uniform + positional embedding.
           Each query j within level k receives a positional component
           proportional to j/L_k, following DLCM's segment-concept
           correspondence principle. This provides a starting point
           where query j is biased toward attending to the j-th segment
           of the sequence.

           Concretely: Q_k[j] = xavier_init + α × PE(j/L_k)
           where PE(p) is a sinusoidal positional encoding at position p,
           and α is a scaling factor controlling the init signal strength.
        """
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Initialize concept queries
        if self.use_positional_query_init:
            # Positional initialization: add segment-scoped positional prior
            # α controls the strength of positional signal vs random signal
            # A moderate α lets the model quickly discover position structure
            # while still allowing training to override the prior
            positional_init_alpha = 0.5

            for level_idx, queries in enumerate(self.concept_queries):
                L_k = queries.shape[0]
                D = queries.shape[1]

                # Step 1: Xavier uniform base
                nn.init.xavier_uniform_(queries)

                # Step 2: Add positional component
                # Normalized positions: [0, 1/L_k, 2/L_k, ..., (L_k-1)/L_k]
                positions_norm = torch.arange(L_k, dtype=torch.float32) / L_k

                # Sinusoidal positional encoding at normalized positions
                # Dividing by 2i for the standard PE formula
                dim_half = D // 2
                pe = torch.zeros(L_k, D)
                div_term = torch.exp(
                    torch.arange(0, dim_half, dtype=torch.float32)
                    * -(math.log(10000.0) / dim_half)
                )
                pe[:, 0::2] = torch.sin(
                    positions_norm.unsqueeze(1) * div_term.unsqueeze(0)
                )
                pe[:, 1::2] = torch.cos(
                    positions_norm.unsqueeze(1) * div_term.unsqueeze(0)
                )

                # Add positional signal to queries
                # Q_k[j] += α * PE(j/L_k)
                with torch.no_grad():
                    queries.add_(positional_init_alpha * pe)
        else:
            # Random initialization: pure Xavier uniform
            for queries in self.concept_queries:
                nn.init.xavier_uniform_(queries)

        for proj in self.level_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

        # Cross-attention weights are initialized by PyTorch default

    def forward_next_level(
        self,
        encoder_hidden_states: torch.Tensor,
        previous_level_concepts: Optional[List[torch.Tensor]],
        target_level_index: int,
    ) -> torch.Tensor:
        """Generate concepts for next level (inference mode).

        PURPOSE:
            Core method implementing next-level generation paradigm.
            Called iteratively to build the concept pyramid level by level.

        HYBRID DESIGN:
            Combines residual decomposition (primary) with cross-attention
            refinement (secondary) for optimal inference.

        DIMENSION FLOW:
            Input:
                encoder_hidden_states: [B, L, D_encoder] - Hidden states from encoder
                previous_level_concepts: [C_0, ..., C_{k-1}] or None for level 0
                target_level_index: Current level index k (0-indexed)

            Process:
                1. Project: H_proj = input_proj(H) ∈ [B, L, D]
                2. Compute residual: H_rest = H_proj - reconstruct(previous)
                3. Extract concepts from residual using attention
                4. Refine with cross-attention over accumulated context

            Output:
                level_concepts: [B, L_k, D] - Concepts for level k

        NEXT-LEVEL PARADIGM:
            Level 0: No previous_level_concepts, generates from H directly
            Level k: Uses previous_level_concepts [C_0, ..., C_{k-1}] as context
            Each level generates multiple concepts in PARALLEL (intra-level)
            Levels are generated SEQUENTIALLY (inter-level causal)

        Args:
            encoder_hidden_states: Hidden states [B, L, D_encoder]
            previous_level_concepts: Previous concepts or None for level 0
            target_level_index: Level to generate (0-indexed)

        Returns:
            level_concepts: Concepts for this level [B, L_k, D]
        """
        batch_size, seq_len, _ = encoder_hidden_states.shape

        # =================================================================
        # Step 1: Project encoder hidden states to concept dimension
        # =================================================================
        # H_proj: [B, L, D] where D = config.hidden_dim
        projected_hidden = self.input_proj(encoder_hidden_states)

        # =================================================================
        # Step 2: Compute residual from previous levels (ResidualAttentivePooling)
        # =================================================================
        # Following VAR Section 5.2.2:
        #   f_rest = H_proj - f_hat (what still needs encoding)
        #   f_hat = Σ A_prev^T @ C_prev_base (what has been encoded)
        #
        # CRITICAL: reconstruction uses C_prev_BASE only (not refined),
        # ensuring residual flow is clean without double-counting.
        #
        if previous_level_concepts is None or len(previous_level_concepts) == 0:
            # Level 0: No previous concepts, use full projected hidden states
            residual_hidden = projected_hidden
        else:
            # Level k > 0: Subtract reconstruction from previous BASE concepts
            # Reconstruct H from previous levels using cached attentions
            # NOTE: _cached_base_concepts stores the BASE (pre-refinement) concepts
            # that entered the residual flow during training
            reconstructed_hidden = torch.zeros_like(projected_hidden)

            for prev_level_idx, (prev_base_concept, prev_attention) in enumerate(
                zip(self._cached_base_concepts, self._cached_attentions)
            ):
                # prev_attention: [B, L_prev, L]
                # prev_base_concept: [B, L_prev, D] (BASE, not refined)
                # Reconstruction: A^T @ C_base -> [B, L, L_prev] @ [B, L_prev, D] = [B, L, D]
                reconstructed_hidden = reconstructed_hidden + torch.bmm(
                    prev_attention.transpose(1, 2), prev_base_concept
                )

            # Residual = Original - Reconstructed (f_rest = H_proj - f_hat)
            residual_hidden = projected_hidden - reconstructed_hidden

        # =================================================================
        # Step 3: Extract BASE concepts from residual using attention (commit)
        # =================================================================
        # Get learnable queries for this level
        level_queries = self.concept_queries[target_level_index]  # [L_k, D]

        # Expand queries for batch: [L_k, D] -> [B, L_k, D]
        expanded_queries = level_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute attention scores: Q @ H_rest^T
        # [B, L_k, D] @ [B, D, L] = [B, L_k, L]
        attention_scores = torch.bmm(expanded_queries, residual_hidden.transpose(1, 2))
        attention_scores = attention_scores / (
            math.sqrt(self.config.hidden_dim) * self.temperature
        )

        # Softmax attention weights
        level_attention = F.softmax(attention_scores, dim=-1)  # [B, L_k, L]

        # Cache attention for future residual computation
        if target_level_index >= len(self._cached_attentions):
            self._cached_attentions.append(level_attention.detach())
        else:
            self._cached_attentions[target_level_index] = level_attention.detach()

        # Extract BASE concepts: A @ H_rest
        # [B, L_k, L] @ [B, L, D] = [B, L_k, D]
        level_concepts_base = torch.bmm(level_attention, residual_hidden)

        # Apply level-specific projection to base concepts
        level_concepts_base = self.level_projs[target_level_index](level_concepts_base)

        # Cache BASE concepts for future residual computation
        # (only BASE enters f_rest, not refined)
        if target_level_index >= len(self._cached_base_concepts):
            self._cached_base_concepts.append(level_concepts_base.detach())
        else:
            self._cached_base_concepts[target_level_index] = (
                level_concepts_base.detach()
            )

        # =================================================================
        # Step 4: Refine with cross-attention (MonotonicSoftAssignment)
        # =================================================================
        # Cross-attention adds context-aware refinement that does NOT enter f_rest.
        # This matches training: refined C_k goes to decoder, not residual flow.
        if target_level_index > 0 and previous_level_concepts is not None:
            # Build accumulated context: [H_proj, C_0, ..., C_{k-1}]
            # previous_level_concepts are REFINED concepts (what decoder sees)
            # Shape: prev_concepts_cat [B, ΣL_i, D]
            prev_concepts_cat = torch.cat(previous_level_concepts, dim=1)
            # Shape: context [B, L+ΣL_i, D]
            context = torch.cat([projected_hidden, prev_concepts_cat], dim=1)

            # Cross-attention: queries attend to accumulated context
            # Query: [B, L_k, D], Key/Value: [B, L+ΣL_i, D], Output: [B, L_k, D]
            refined_concepts, _ = self.level_attn[target_level_index](
                expanded_queries,
                context,
                context,
            )

            # Residual connection: base extraction + cross-attention refinement
            level_concepts = level_concepts_base + refined_concepts
        else:
            level_concepts = level_concepts_base

        return level_concepts

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        target_level_index: Optional[int],
        previous_level_concepts: Optional[List[torch.Tensor]],
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Dict[str, Any]]:
        """Unified forward interface for concept generation.

        PURPOSE:
            Supports both training (all levels parallel) and inference
            (next-level sequential) modes through unified interface.

        USAGE MODES:
            1. Training (all levels):
               concepts, aux = generator(encoder_hidden_states)
               # Returns: [C_0, C_1, ..., C_K], aux with losses

            2. Inference (single level):
               C_k, aux = generator(encoder_hidden_states, target_level_index=k,
                                    previous_level_concepts=[C_0, ..., C_{k-1}])
               # Returns: C_k (single tensor), aux

        HYBRID COMPUTATION:
            Combines three mechanisms:
            1. Residual decomposition (primary): ensures coarse-to-fine hierarchy
            2. Cross-attention refinement (secondary): aligns with inference generator
            3. Boundary ordering loss (auxiliary): enforces position ordering

        DIMENSION FLOW:
            All levels mode:
                Input: encoder_hidden_states [B, L, D_encoder]
                Process: For k in 0..K-1:
                    C_k = forward_next_level(H, [C_0..C_{k-1}], k)
                Output: [C_0, C_1, ..., C_K], aux with losses

            Single level mode:
                Input: encoder_hidden_states [B, L, D_encoder], k, previous
                Output: C_k [B, L_k, D], aux

        Args:
            encoder_hidden_states: Hidden states from encoder [B, L, D_encoder]
            target_level_index: If provided, only generate this level (inference)
            previous_level_concepts: Previous concepts for target_level_index > 0

        Returns:
            All levels: (List[C_0, ..., C_K], aux)
            Single level: (C_k, aux)
        """
        if target_level_index is not None:
            # =================================================================
            # Inference Mode: Single Level Generation
            # =================================================================
            level_concepts = self.forward_next_level(
                encoder_hidden_states, previous_level_concepts, target_level_index
            )

            aux = {
                "target_level_index": target_level_index,
                "method": "hybrid_next_level",
            }
            return level_concepts, aux

        # =================================================================
        # Training Mode: All Levels Generation (Parallel Optimization)
        # =================================================================
        batch_size = encoder_hidden_states.shape[0]

        # Clear caches for fresh computation
        self._cached_attentions = []
        self._cached_base_concepts = []

        # Project to concept dimension
        projected_hidden = self.input_proj(encoder_hidden_states)  # [B, L, D]

        # Initialize residual decomposition
        residual_hidden = projected_hidden.clone()  # H_rest_0
        reconstructed_accumulator = torch.zeros_like(projected_hidden)  # H_hat_0

        all_level_concepts: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        # =================================================================
        # Extract all levels with residual decomposition
        # =================================================================
        #
        # CRITICAL DESIGN (following VAR Section 5.2.2):
        #   The residual flow (f_rest) must ONLY subtract base-extracted concepts,
        #   NOT the cross-attention refined concepts. This is because:
        #
        #   1. f_rest tells each level "what still needs encoding" (VAR 5.2.3)
        #   2. Cross-attention reads from [H_proj, C_0..C_{k-1}], which includes
        #      information already reconstructed by previous levels
        #   3. If refined C_k enters f_rest, it double-counts this information,
        #      violating the f_hat ≈ H_proj reconstruction target
        #
        #   Therefore we split:
        #     C_k_base = A_k @ H_rest_k (commit to residual, enters f_rest)
        #     C_k = C_k_base + refined_k (output concept, goes to decoder)
        #     R_k = A_k^T @ C_k_base (only base concept reconstructs)
        #
        for level_idx in range(self.config.num_levels):
            # Get queries for this level
            level_queries = self.concept_queries[level_idx]  # [L_k, D]

            # Expand queries: [L_k, D] -> [B, L_k, D]
            expanded_queries = level_queries.unsqueeze(0).expand(batch_size, -1, -1)

            # Compute attention over residual
            attention_scores = torch.bmm(
                expanded_queries, residual_hidden.transpose(1, 2)
            )  # [B, L_k, L]
            attention_scores = attention_scores / (
                math.sqrt(self.config.hidden_dim) * self.temperature
            )

            level_attention = F.softmax(attention_scores, dim=-1)  # [B, L_k, L]
            all_attentions.append(level_attention)

            # Extract BASE concepts from residual (commit path)
            # C_k_base: only new information from H_rest, enters residual flow
            level_concepts_base = torch.bmm(
                level_attention, residual_hidden
            )  # [B, L_k, D]

            # Apply level-specific projection to base concepts
            level_concepts_base = self.level_projs[level_idx](level_concepts_base)

            # Reconstruct from BASE concepts only (not refined)
            # This is the VAR f_hat update: f_hat += A_k^T @ C_k_base
            reconstruction = torch.bmm(
                level_attention.transpose(1, 2), level_concepts_base
            )  # [B, L, D]

            # Update residual flow BEFORE cross-attention refinement
            # f_hat += R_k (accumulate what's been encoded)
            # f_rest -= R_k (tell next level what still needs encoding)
            reconstructed_accumulator = reconstructed_accumulator + reconstruction
            residual_hidden = residual_hidden - reconstruction

            # Cross-attention refinement (if not level 0)
            # refined_k adds context-aware information that does NOT enter f_rest
            # This is free to incorporate accumulated context without
            # double-counting, because it only affects the output C_k,
            # not the residual flow
            if level_idx > 0:
                # Context includes H_proj + all previous REFINED concepts
                # (refined concepts are what the decoder sees, so cross-attn
                # should align with the same information the decoder uses)
                prev_concepts_cat = torch.cat(all_level_concepts, dim=1)  # [B, ΣL_i, D]
                context = torch.cat([projected_hidden, prev_concepts_cat], dim=1)

                refined_concepts, _ = self.level_attn[level_idx](
                    expanded_queries, context, context
                )
                level_concepts = level_concepts_base + refined_concepts
            else:
                level_concepts = level_concepts_base

            all_level_concepts.append(level_concepts)

        # =================================================================
        # Compute Auxiliary Losses
        # =================================================================

        # Loss 1: Reconstruction Loss (from ResidualAttentivePooling)
        recon_loss = F.mse_loss(reconstructed_accumulator, projected_hidden)

        # Loss 2: Boundary Ordering Loss (from AutoregressiveSoftBoundary)
        # Compute expected position for each concept and enforce ordering
        order_loss = self._compute_ordering_loss(all_attentions, encoder_hidden_states)

        # Total loss (for training)
        total_loss = recon_loss + self.order_loss_weight * order_loss

        # =================================================================
        # Build Output
        # =================================================================
        aux = {
            # Reconstruction info
            "reconstructed_hidden": reconstructed_accumulator,
            "residual_hidden": residual_hidden,
            "recon_loss": recon_loss,
            # Ordering info
            "order_loss": order_loss,
            "total_loss": total_loss,
            # Metadata
            "num_levels": self.config.num_levels,
            "level_lengths": self.config.level_lengths,
            "method": "hybrid",
        }

        return all_level_concepts, aux

    def _compute_ordering_loss(
        self,
        all_attentions: List[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary ordering loss (from AutoregressiveSoftBoundary).

        PURPOSE:
            Enforce that concepts respect CoT positional causality:
            - Cross-level: earlier levels attend to earlier positions
              (coarse concepts cover the beginning, fine concepts cover details)
            - Intra-level: within each level, concepts are ordered by position
              (earlier concept slots attend to earlier CoT positions)

        MATHEMATICS:
            Following VAR's scale-level causality (Section 5.3.1):
            - Level k sees Level 0..k (inter-level causal)
            - Within level k, all positions are parallel (intra-level parallel)

            However, unlike images with natural spatial coordinates, text
            concepts need an explicit ordering signal. We enforce:

            For each concept j, compute expected position:
                exp_pos[j] = Σ_t A[j, t] × t

            Intra-level ordering constraint:
                L_order = Σ_level Σ_j ReLU(exp_pos[j] - exp_pos[j+1] + margin)

            This ensures consecutive concept slots within each level attend
            to non-decreasing positions in the CoT.

            NOTE: Inter-level ordering is intentionally NOT enforced.
            The relationship between levels is granularity (coarse-to-fine),
            not sequential ordering. Level k+1 provides a finer partition
            of the same space, so its first concept can legitimately attend
            to earlier positions than level k's last concept.
            Coarse-to-fine structure is guaranteed by residual flow.

        DIMENSION FLOW:
            Input:
                all_attentions: List of [B, L_k, L] attention matrices
                encoder_hidden_states: [B, L, D_encoder] (for device/shape)

            Output:
                order_loss: Scalar tensor

        Args:
            all_attentions: Attention weights for all levels
            encoder_hidden_states: Encoder hidden states (for device)

        Returns:
            order_loss: Ordering constraint loss (scalar)
        """
        seq_len = encoder_hidden_states.shape[1]
        device = encoder_hidden_states.device

        # Position indices: [L]
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)

        total_violation = torch.tensor(0.0, device=device)

        # =================================================================
        # Constraint 1: Intra-level ordering
        # Within each level, enforce that consecutive concept slots
        # attend to non-decreasing positions.
        # This is a soft version of VAR's spatial position ordering.
        # =================================================================
        for level_idx, level_attention in enumerate(all_attentions):
            # level_attention: [B, L_k, L]
            L_k = level_attention.shape[1]

            # Compute expected position for each concept in this level
            # exp_pos[b, j] = Σ_t A[b, j, t] × t
            expected_positions = torch.sum(
                level_attention * positions.view(1, 1, seq_len),
                dim=-1,
            )  # [B, L_k]

            if L_k > 1:
                # Enforce monotonicity: exp_pos[j] < exp_pos[j+1]
                current_pos = expected_positions[:, :-1]  # [B, L_k-1]
                next_pos = expected_positions[:, 1:]  # [B, L_k-1]

                violation = F.relu(current_pos - next_pos + self.order_margin)
                total_violation = total_violation + violation.mean()

        # =================================================================
        # NOTE: Inter-level ordering is intentionally REMOVED.
        # The relationship between levels is granularity (coarse-to-fine),
        # NOT sequential ordering. Level k+1 provides a finer partition of
        # the SAME space that level k covers. For example:
        #   Level 1 (2 concepts): C_{1,0} ~ [0, L/2],  C_{1,1} ~ [L/2, L]
        #   Level 2 (4 concepts): C_{2,0} ~ [0, L/4],  C_{2,1} ~ [L/4, L/2], ...
        # Enforcing exp_pos[C_{1,1}] < exp_pos[C_{2,0}] would require
        # L/2 < L/4, which is impossible. The coarse-to-fine structure is
        # already guaranteed by the residual flow and f_hat/f_rest decomposition.
        # =================================================================

        return total_violation

    def get_level_config(self, level_idx: int) -> Dict[str, Any]:
        """Get configuration for a specific level.

        Args:
            level_idx: Level index

        Returns:
            Dictionary with level configuration
        """
        return {
            "level_idx": level_idx,
            "num_concepts": self.config.level_lengths[level_idx],
            "hidden_dim": self.config.hidden_dim,
            "query_shape": tuple(self.concept_queries[level_idx].shape),
        }

    def get_total_concepts(self) -> int:
        """Get total number of concepts across all levels."""
        return sum(self.config.level_lengths)
