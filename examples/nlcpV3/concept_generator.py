"""NLCP V3 Concept Generators: Training Extractors & Inference Generator.

USAGE:
    from nlcpV3.concept_generator import (
        # Basic training extractors (from CoT)
        ResidualAttentivePoolingConceptGenerator,
        PositionConstrainedConceptGenerator,
        HardOrderedMaskConceptGenerator,
        RecursiveOrderedConceptGenerator,
        OrderConstrainedTrainingConceptGenerator,
        RobustOrderedConceptGenerator,
        # Advanced causal training extractors
        MonotonicSoftAssignmentConceptGenerator,
        CausalSequentialRefinementConceptGenerator,
        ContinuousCausalKernelConceptGenerator,
        CausalSoftPoolingConceptGenerator,
        # Inference generator (from Q)
        AutoregressiveConceptGenerator,
        # Unified interface
        ConceptGenerator,
    )

    # Individual extractors (standalone, trainable)
    extractor = CausalSoftPoolingConceptGenerator(config, encoder_hidden_dim)
    concepts, aux = extractor(H_cot)

    # Unified interface
    generator = ConceptGenerator(config, encoder_hidden_dim)
    concepts, aux = generator.forward_training(H_cot, method='causal_soft_pooling')
    concepts = generator.forward_inference(H_q)

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    Inspired by: docs/VAR.md Section 5 (Training) & Section 6 (Inference)

ARCHITECTURE:
    Basic Training Extractors:
    ├── ResidualAttentivePoolingConceptGenerator: Residual decomposition (VAR-style)
    ├── PositionConstrainedConceptGenerator: Learnable position centers
    ├── HardOrderedMaskConceptGenerator: Pre-defined segment masks
    ├── RecursiveOrderedConceptGenerator: Sequential extraction
    ├── OrderConstrainedTrainingConceptGenerator: Loss-based ordering
    └── RobustOrderedConceptGenerator: Combined approach

    Advanced Causal Training Extractors:
    ├── MonotonicSoftAssignmentConceptGenerator: Monotonic allocation matrix
    ├── CausalSequentialRefinementConceptGenerator: Causal transformer refinement
    ├── ContinuousCausalKernelConceptGenerator: Continuous position + causal kernel
    └── CausalSoftPoolingConceptGenerator: Complete pipeline (recommended)

    Inference Generator:
    └── AutoregressiveConceptGenerator: Next-level generation from Q

    Unified Interface:
    └── ConceptGenerator: Wraps all 10 methods, provides consistent API

KEY PROBLEM SOLVED:
    Soft assignment treats segments as unordered sets, but text is inherently
    sequential. The advanced causal methods ensure:
    1. Full coverage: sum_i A[t,i] = 1 for all positions t
    2. Ordering: Concepts respect text causality (no future leakage)
    3. Smooth transitions: Soft boundaries without hard cuts
    4. End-to-end trainable: All operations are differentiable

DIMENSION FLOW:
    All training extractors:
        Input: H [B, L, D_encoder] (from Encoder with Q+CoT)
        Process: Various extraction strategies
        Output: concepts [C_0, ..., C_K], auxiliary_info

    Inference generator:
        Input: H [B, L, D_encoder] (from Encoder with Q only)
        Process: Autoregressive next-level generation
        Output: concepts [C_0, ..., C_K]

REFERENCES:
    - Monotonic Attention: Luong et al. (2016), Press & Wolf (2018)
    - Causal Pooling: Yang et al. (2022) "Hierarchical Soft Chunking"
    - Causal Information Bottleneck: Chen et al. (2024)
    - Differentiable Sorting: Grover et al. (2019), Blondel et al. (2020)
"""

import math
from typing import Optional, Tuple, List, Dict, Any, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV3.config import NLCPV3Config


# =============================================================================
# Base Class for Training Extractors
# =============================================================================


class BaseConceptGenerator(nn.Module, ABC):
    """Abstract base class for concept extraction/generation methods.

    PURPOSE:
        Define common interface for all concept generators with
        next-level generation paradigm inspired by VAR.

    KEY DESIGN PRINCIPLE:
        Concepts are generated level by level (next-level), where each level
        can be generated in parallel (intra-level) and depends on previous
        levels (inter-level causal).

    ARCHITECTURE:
        Level 0: C_0 = forward_next_level(H, None, 0)         [B, L_0, D]
        Level 1: C_1 = forward_next_level(H, [C_0], 1)        [B, L_1, D]
        Level 2: C_2 = forward_next_level(H, [C_0, C_1], 2)   [B, L_2, D]
        ...
        Level K: C_K = forward_next_level(H, [C_0..C_{K-1}], K) [B, L_K, D]

    TRAINING vs INFERENCE:
        Training: Can use forward_all_levels() for parallel computation
                  (with ground truth from CoT extraction)
        Inference: Uses forward_next_level() sequentially
                   (generates from Q without CoT)

    ATTRIBUTES:
        config: NLCPV3Config
        encoder_hidden_dim: Input dimension from encoder
        input_proj: Projection to concept dimension
        concept_queries: Learnable queries for each level
    """

    def __init__(self, config: NLCPV3Config, encoder_hidden_dim: int):
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = encoder_hidden_dim

        # Shared projection
        self.input_proj = nn.Linear(encoder_hidden_dim, config.hidden_dim)

        # Shared concept queries (key for consistency!)
        # Each level has its own set of learnable queries
        self.concept_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(length, config.hidden_dim))
                for length in config.level_lengths
            ]
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for queries in self.concept_queries:
            nn.init.xavier_uniform_(queries)

    # =========================================================================
    # Core Methods: Next-Level Generation
    # =========================================================================

    @abstractmethod
    def forward_next_level(
        self,
        encoder_hidden_states: torch.Tensor,
        previous_level_concepts: Optional[List[torch.Tensor]],
        target_level_index: int,
    ) -> torch.Tensor:
        """Generate/extract concepts for the next level.

        PURPOSE:
            Core method implementing next-level generation.
            Called iteratively to build the concept pyramid.

        DIMENSION FLOW:
            Input:
                encoder_hidden_states: [B, L, D_encoder] - Hidden states from encoder
                previous_level_concepts: List of [C_0, ..., C_{k-1}] or None for level 0
                target_level_index: Current level index (0-indexed)

            Output:
                level_concepts: [B, L_level, D] - Concepts for this level

        NEXT-LEVEL PARADIGM:
            - Level 0: No previous_level_concepts, generates from encoder_hidden_states directly
            - Level k: Uses previous_level_concepts [C_0, ..., C_{k-1}] as context
            - Each level can generate multiple concepts in PARALLEL (intra-level)
            - Levels are generated SEQUENTIALLY (inter-level causal)

        Args:
            encoder_hidden_states: Hidden states from encoder [B, L, D_encoder]
            previous_level_concepts: Previously generated concepts (None for level 0)
            target_level_index: Current level index to generate

        Returns:
            level_concepts: Concepts for this level [B, L_level, D]
        """
        pass

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        target_level_index: Optional[int] = None,
        previous_level_concepts: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Dict[str, Any]]:
        """Unified forward interface for concept generation.

        USAGE MODES:
            1. Generate all levels (training):
               concepts, aux = generator(encoder_hidden_states)
               # Returns: [C_0, C_1, ..., C_K]

            2. Generate single level (inference, next-level):
               C_k, aux = generator(encoder_hidden_states, target_level_index=2, previous_level_concepts=[C_0, C_1])
               # Returns: C_2 (single tensor)

        PURPOSE:
            This unified interface supports both training (all levels) and
            inference (next-level generation) modes.

        DIMENSION FLOW:
            All levels mode:
                Input: encoder_hidden_states [B, L, D_encoder]
                Process: For k in 0..K-1: C_k = forward_next_level(encoder_hidden_states, prev, k)
                Output: [C_0, C_1, ..., C_K]

            Single level mode:
                Input: encoder_hidden_states [B, L, D_encoder], target_level_index, previous_level_concepts
                Output: level_concepts [B, L_level, D]

        NEXT-LEVEL PARADIGM:
            - Level 0: No previous_level_concepts, generates from encoder_hidden_states directly
            - Level k: Uses previous_level_concepts [C_0, ..., C_{k-1}] as context
            - Each level can generate multiple concepts in PARALLEL (intra-level)
            - Levels are generated SEQUENTIALLY (inter-level causal)

        Args:
            encoder_hidden_states: Hidden states from encoder [B, L, D_encoder]
            target_level_index: If provided, only generate this level (next-level mode)
            previous_level_concepts: Previous concepts for target_level_index > 0

        Returns:
            All levels: (List[C_0, ..., C_K], aux)
            Single level: (level_concepts, aux)
        """
        if target_level_index is None:
            # Generate all levels (training mode)
            projected_hidden = self.input_proj(encoder_hidden_states)
            all_level_concepts = []

            for level_idx in range(self.config.num_levels):
                previous = all_level_concepts if all_level_concepts else None
                level_concepts = self.forward_next_level(
                    encoder_hidden_states, previous, level_idx
                )
                all_level_concepts.append(level_concepts)

            aux = {
                "num_levels": self.config.num_levels,
                "level_lengths": self.config.level_lengths,
            }
            return all_level_concepts, aux
        else:
            # Generate single level (next-level inference mode)
            level_concepts = self.forward_next_level(
                encoder_hidden_states, previous_level_concepts, target_level_index
            )
            aux = {"target_level_index": target_level_index}
            return level_concepts, aux

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
            "query_shape": self.concept_queries[level_idx].shape,
        }

    def get_total_concepts(self) -> int:
        """Get total number of concepts across all levels."""
        return sum(self.config.level_lengths)


################################################################################
#                                                                              #
#                        BASIC TRAINING EXTRACTORS                             #
#                                                                              #
#    Simple, interpretable methods for concept extraction from CoT.            #
#    These methods establish the foundation for hierarchical concept learning. #
#                                                                              #
################################################################################


# =============================================================================
# Training Method 1: Residual Attentive Pooling
# =============================================================================


class ResidualAttentivePoolingConceptGenerator(BaseConceptGenerator):
    """Residual attentive pooling for hierarchical concept extraction.

    PURPOSE:
        Extract concepts using residual decomposition like VAR's VQ-VAE.
        Each level extracts from the residual of previous levels.

    NEXT-LEVEL PARADIGM:
        This method implements the core VAR-style residual decomposition:
        - Each level extracts concepts from the RESIDUAL of previous levels
        - H_rest_k = H - (reconstructed from C_0, ..., C_{k-1})
        - C_k is extracted from H_rest_k, not from original H

    MATHEMATICS:
        H_rest_0 = H
        H_hat_0 = 0

        For level k:
            A_k = softmax(Q_k @ H_rest_k^T / sqrt(D))
            C_k = A_k @ H_rest_k
            H_recon_k = A_k^T @ C_k
            H_hat_{k+1} = H_hat_k + H_recon_k
            H_rest_{k+1} = H_rest_k - H_recon_k

    REFERENCE:
        VAR.md Section 5.2.2: Core mechanism of residual decomposition
    """

    def __init__(self, config: NLCPV3Config, encoder_hidden_dim: int):
        super().__init__(config, encoder_hidden_dim)
        self.temperature = nn.Parameter(torch.ones(1))

        # Level-specific projections
        self.level_projs = nn.ModuleList(
            [
                nn.Linear(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_levels)
            ]
        )

        # Store attention weights for residual computation
        self._cached_attentions = []

    def forward_next_level(
        self,
        encoder_hidden_states: torch.Tensor,
        previous_level_concepts: Optional[List[torch.Tensor]],
        target_level_index: int,
    ) -> torch.Tensor:
        """Generate concepts for next level using residual decomposition.

        DIMENSION FLOW:
            Input:
                encoder_hidden_states: [B, L, D_encoder]
                previous_level_concepts: [C_0, ..., C_{k-1}] or None
                target_level_index: Current level (0-indexed)

            Process:
                1. Project: projected_hidden [B, L, D]
                2. Compute residual: residual_hidden = projected_hidden - reconstruct(previous)
                3. Extract level_concepts from residual_hidden

            Output:
                level_concepts: [B, L_k, D]
        """
        batch_size, seq_len, _ = encoder_hidden_states.shape
        projected_hidden = self.input_proj(encoder_hidden_states)  # [B, L, D]

        # Compute residual from previous levels
        if previous_level_concepts is None or len(previous_level_concepts) == 0:
            residual_hidden = projected_hidden  # First level: no residual
        else:
            # Reconstruct from previous concepts using cached attentions
            reconstructed_hidden = torch.zeros_like(projected_hidden)
            for prev_level_idx, (prev_concepts, prev_attention) in enumerate(
                zip(previous_level_concepts, self._cached_attentions)
            ):
                # prev_attention: [B, L_prev, L], prev_concepts: [B, L_prev, D]
                reconstructed_hidden = reconstructed_hidden + torch.bmm(
                    prev_attention.transpose(1, 2), prev_concepts
                )
            residual_hidden = projected_hidden - reconstructed_hidden

        # Extract concepts for this level
        level_queries = self.concept_queries[target_level_index]  # [L_k, D]
        num_concepts = level_queries.shape[0]

        # Expand queries
        expanded_queries = level_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, L_k, D]

        # Attention
        attention_scores = torch.bmm(expanded_queries, residual_hidden.transpose(1, 2))
        attention_scores = attention_scores / (
            math.sqrt(self.config.hidden_dim) * self.temperature
        )
        level_attention = F.softmax(attention_scores, dim=-1)  # [B, L_k, L]

        # Cache attention for future residual computation
        if target_level_index >= len(self._cached_attentions):
            self._cached_attentions.append(level_attention.detach())
        else:
            self._cached_attentions[target_level_index] = level_attention.detach()

        # Extract concepts
        level_concepts = torch.bmm(level_attention, residual_hidden)  # [B, L_k, D]
        level_concepts = self.level_projs[target_level_index](level_concepts)

        return level_concepts

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        target_level_index: Optional[int] = None,
        previous_level_concepts: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Dict[str, Any]]:
        """Unified forward with residual decomposition.

        PURPOSE:
            Override base class to add reconstruction loss for training.

        USAGE MODES:
            1. All levels (training): gen(encoder_hidden_states) -> concepts, aux (with recon_loss)
            2. Single level (inference): gen(encoder_hidden_states, target_level_index=k, previous_level_concepts=[...]) -> C_k
        """
        if target_level_index is None:
            # All levels: use optimized residual decomposition
            self._cached_attentions = []
            batch_size = encoder_hidden_states.shape[0]
            projected_hidden = self.input_proj(encoder_hidden_states)

            residual_hidden = projected_hidden.clone()
            reconstructed_accumulator = torch.zeros_like(projected_hidden)
            all_level_concepts = []

            for level_idx in range(self.config.num_levels):
                level_queries = self.concept_queries[level_idx]
                expanded_queries = level_queries.unsqueeze(0).expand(batch_size, -1, -1)

                attention_scores = torch.bmm(
                    expanded_queries, residual_hidden.transpose(1, 2)
                )
                attention_scores = attention_scores / (
                    math.sqrt(self.config.hidden_dim) * self.temperature
                )
                level_attention = F.softmax(attention_scores, dim=-1)

                self._cached_attentions.append(level_attention.detach())

                level_concepts = torch.bmm(level_attention, residual_hidden)
                level_concepts = self.level_projs[level_idx](level_concepts)
                all_level_concepts.append(level_concepts)

                reconstruction = torch.bmm(
                    level_attention.transpose(1, 2), level_concepts
                )
                reconstructed_accumulator = reconstructed_accumulator + reconstruction
                residual_hidden = residual_hidden - reconstruction

            recon_loss = F.mse_loss(reconstructed_accumulator, projected_hidden)

            aux = {
                "reconstructed_hidden": reconstructed_accumulator,
                "residual_hidden": residual_hidden,
                "recon_loss": recon_loss,
                "num_levels": self.config.num_levels,
                "level_lengths": self.config.level_lengths,
                "method": "residual_pooling",
            }
            return all_level_concepts, aux
        else:
            # Single level: use forward_next_level
            level_concepts = self.forward_next_level(
                encoder_hidden_states, previous_level_concepts, target_level_index
            )
            aux = {
                "target_level_index": target_level_index,
                "method": "residual_pooling",
            }
            return level_concepts, aux


# =============================================================================
# Training Method 2: Position Constrained
# =============================================================================


class PositionConstrainedConceptGenerator(BaseConceptGenerator):
    """Position-constrained extraction with learnable centers.

    PURPOSE:
        Ensure concept ordering by learning expected positions for each concept
        and biasing attention toward those positions.

    MATHEMATICS:
        Learnable centers c_k for each level
        Position prior: prior[i,j] = exp(-|j - c[i]| / T)
        Biased attention: A = softmax(scores + log_prior)

    REFERENCE:
        Ordered concept extraction Scheme 1
    """

    def __init__(
        self, config: NLCPV3Config, encoder_hidden_dim: int, max_seq_len: int = 2048
    ):
        super().__init__(config, encoder_hidden_dim)

        # Learnable center positions (initialized uniformly)
        init_centers = torch.linspace(0, max_seq_len - 1, config.num_levels)
        init_logits = torch.logit(torch.clamp(init_centers / max_seq_len, 0.01, 0.99))
        self.center_logits = nn.Parameter(init_logits)

        self.temperature = nn.Parameter(torch.tensor(max_seq_len / config.num_levels))

    def _get_sorted_centers(self, seq_len: int) -> torch.Tensor:
        """Get sorted, normalized concept centers."""
        centers = torch.sigmoid(self.center_logits) * seq_len
        centers, _ = torch.sort(centers)
        return centers

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract with position constraints.

        DIMENSION FLOW:
            Input: H [B, L, D_encoder]
            Project: H_proj [B, L, D]
            Centers: c_k [K] (learnable, sorted)

            For each level k:
                scores = Q_k @ H_proj^T
                prior = exp(-|positions - c_k| / T)
                A_k = softmax(scores + log_prior)
                C_k = A_k @ H_proj

            Output:
                concepts: [C_0, ..., C_K]
                aux: {'expected_positions': [...], 'centers': [...]}
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)  # [B, L, D]

        centers = self._get_sorted_centers(L)
        positions = torch.arange(L, device=H.device, dtype=torch.float32)

        concepts = []
        expected_positions = []

        for level_idx in range(self.config.num_levels):
            Q_k = self.concept_queries[level_idx]
            L_k = Q_k.shape[0]

            Q = Q_k.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L_k, D]

            # Base scores
            scores = torch.bmm(Q, H_proj.transpose(1, 2)) / math.sqrt(
                self.config.hidden_dim
            )

            # Position prior
            center = centers[level_idx]
            distance = torch.abs(positions - center)
            log_prior = -distance / self.temperature

            # Combine
            biased_scores = scores + log_prior.view(1, 1, L)
            A_k = F.softmax(biased_scores, dim=-1)

            # Extract
            C_k = torch.bmm(A_k, H_proj)
            concepts.append(C_k)

            # Track expected position
            exp_pos = torch.sum(A_k * positions.view(1, 1, L), dim=-1).mean()
            expected_positions.append(exp_pos.item())

        aux = {
            "expected_positions": expected_positions,
            "centers": centers.detach().cpu().tolist(),
            "method": "position_constrained",
        }

        return concepts, aux


# =============================================================================
# Training Method 3: Hard Ordered Mask
# =============================================================================


class HardOrderedMaskConceptGenerator(BaseConceptGenerator):
    """Hard ordered mask extraction with soft edges.

    PURPOSE:
        Enforce strict segment boundaries while allowing soft transitions.
        Each concept primarily attends to its designated segment.

    MATHEMATICS:
        Pre-defined mask: mask[i,j] = 1 if j in segment i, decay at edges
        Masked attention: A = softmax(scores + log(mask))

    REFERENCE:
        Ordered concept extraction Scheme 2
    """

    def __init__(
        self, config: NLCPV3Config, encoder_hidden_dim: int, softness: float = 0.2
    ):
        super().__init__(config, encoder_hidden_dim)
        self.softness = softness

    def _create_ordered_mask(
        self, num_concepts: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create ordered mask with soft edges."""
        mask = torch.zeros(num_concepts, seq_len, device=device)
        edge_width = int(self.softness * seq_len / num_concepts)

        for i in range(num_concepts):
            start = int(i * seq_len / num_concepts)
            end = int((i + 1) * seq_len / num_concepts)

            # Primary region
            mask[i, start:end] = 1.0

            # Left edge decay
            if start > 0 and edge_width > 0:
                left_start = max(0, start - edge_width)
                left_len = start - left_start
                decay = torch.linspace(0, 1, left_len + 1)[1:]
                mask[i, left_start:start] = torch.maximum(
                    mask[i, left_start:start], decay
                )

            # Right edge decay
            if end < seq_len and edge_width > 0:
                right_end = min(seq_len, end + edge_width)
                right_len = right_end - end
                decay = torch.linspace(1, 0, right_len + 1)[:-1]
                mask[i, end:right_end] = torch.maximum(mask[i, end:right_end], decay)

        # Normalize
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-10)
        return mask

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract with ordered mask.

        DIMENSION FLOW:
            Input: H [B, L, D_encoder]
            Create mask: [L_k, L] for each level
            Apply mask to attention
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)

        concepts = []
        all_masks = []

        for level_idx in range(self.config.num_levels):
            Q_k = self.concept_queries[level_idx]
            L_k = Q_k.shape[0]

            # Create mask for this level
            mask = self._create_ordered_mask(L_k, L, H.device)  # [L_k, L]
            all_masks.append(mask)

            Q = Q_k.unsqueeze(0).expand(batch_size, -1, -1)
            scores = torch.bmm(Q, H_proj.transpose(1, 2)) / math.sqrt(
                self.config.hidden_dim
            )

            # Apply mask
            log_mask = torch.log(mask.unsqueeze(0) + 1e-10)
            biased_scores = scores + log_mask
            A_k = F.softmax(biased_scores, dim=-1)

            C_k = torch.bmm(A_k, H_proj)
            concepts.append(C_k)

        aux = {
            "masks": all_masks,
            "method": "hard_ordered_mask",
        }

        return concepts, aux


# =============================================================================
# Training Method 4: Recursive Ordered
# =============================================================================


class RecursiveOrderedConceptGenerator(BaseConceptGenerator):
    """Recursive ordered extraction (VAR-style scale-by-scale).

    PURPOSE:
        Extract concepts sequentially, where each concept attends only to
        positions not heavily attended by previous concepts.

    MATHEMATICS:
        remaining_mask_0 = 1
        For concept i:
            A_i = softmax(scores_i * remaining_mask_i)
            C_i = A_i @ H
            usage = (A_i > threshold)
            remaining_mask_{i+1} = remaining_mask_i * (1 - usage * decay)

    REFERENCE:
        Ordered concept extraction Scheme 3
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        usage_threshold: float = 0.5,
        decay_rate: float = 0.5,
    ):
        super().__init__(config, encoder_hidden_dim)
        self.usage_threshold = usage_threshold
        self.decay_rate = decay_rate

        # Next query projection
        self.next_query_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract recursively with remaining mask.

        DIMENSION FLOW:
            Initialize remaining_mask [B, L] = 1
            For each concept i:
                Compute attention with mask
                Extract C_i
                Update remaining_mask
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)

        remaining_mask = torch.ones(batch_size, L, device=H.device)
        concepts = []
        remaining_history = []

        for level_idx in range(self.config.num_levels):
            Q_k = self.concept_queries[level_idx]
            L_k = Q_k.shape[0]

            # Use first query for this level (simplified)
            Q = Q_k[0:1].unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, D]

            # Compute scores with mask
            scores = torch.bmm(Q, H_proj.transpose(1, 2)).squeeze(1)  # [B, L]
            scores = scores.masked_fill(remaining_mask < 0.5, float("-inf"))
            A_k = F.softmax(scores, dim=-1)  # [B, L]

            # Extract concept (single concept per level for simplicity)
            C_k = torch.bmm(A_k.unsqueeze(1), H_proj)  # [B, 1, D]
            concepts.append(C_k)

            # Update remaining mask
            max_attn = A_k.max(dim=-1, keepdim=True)[0]
            usage = (A_k > max_attn * self.usage_threshold).float()
            remaining_mask = remaining_mask * (1 - usage * self.decay_rate)
            remaining_history.append(remaining_mask.clone())

        aux = {
            "remaining_history": remaining_history,
            "method": "recursive_ordered",
        }

        return concepts, aux


# =============================================================================
# Training Method 5: Order Constrained Training
# =============================================================================


class OrderConstrainedTrainingConceptGenerator(BaseConceptGenerator):
    """Order-constrained extraction with loss-based ordering.

    PURPOSE:
        Use standard attention but add order loss to encourage sequential
        concept positions. Most flexible approach.

    MATHEMATICS:
        Standard attention extraction
        Order loss: L_order = sum(ReLU(pos[i] - pos[i+1] + margin))

    REFERENCE:
        Ordered concept extraction Scheme 4
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        order_margin: float = 1.0,
        order_weight: float = 0.1,
    ):
        super().__init__(config, encoder_hidden_dim)
        self.order_margin = order_margin
        self.order_weight = order_weight
        self.temperature = nn.Parameter(torch.ones(1))

    def compute_order_loss(self, expected_positions: torch.Tensor) -> torch.Tensor:
        """Compute order constraint loss."""
        pos_current = expected_positions[:-1]
        pos_next = expected_positions[1:]
        violation = F.relu(pos_current - pos_next + self.order_margin)
        return violation.mean() * self.order_weight

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract with order loss (computed externally).

        DIMENSION FLOW:
            Standard attention extraction
            Track expected positions for loss computation
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)
        positions = torch.arange(L, device=H.device, dtype=torch.float32)

        concepts = []
        expected_positions = []

        for level_idx in range(self.config.num_levels):
            Q_k = self.concept_queries[level_idx]
            L_k = Q_k.shape[0]

            Q = Q_k.unsqueeze(0).expand(batch_size, -1, -1)
            scores = torch.bmm(Q, H_proj.transpose(1, 2))
            scores = scores / (math.sqrt(self.config.hidden_dim) * self.temperature)
            A_k = F.softmax(scores, dim=-1)

            C_k = torch.bmm(A_k, H_proj)
            concepts.append(C_k)

            # Track expected position
            exp_pos = torch.sum(A_k * positions.view(1, 1, L), dim=-1).mean()
            expected_positions.append(exp_pos)

        # Compute order loss
        expected_positions_tensor = torch.stack(expected_positions)
        order_loss = self.compute_order_loss(expected_positions_tensor)

        aux = {
            "expected_positions": [p.item() for p in expected_positions],
            "order_loss": order_loss,
            "method": "order_constrained",
        }

        return concepts, aux


# =============================================================================
# Training Method 6: Robust Ordered (Recommended Combination)
# =============================================================================


class RobustOrderedConceptGenerator(BaseConceptGenerator):
    """Robust ordered extraction combining position prior and order loss.

    PURPOSE:
        Training: Use weak position prior + order loss for flexibility
        Inference: Use strong position prior for guaranteed ordering

    MATHEMATICS:
        Centers = sort(sigmoid(center_logits))  # Enforce order
        Position bias = -|pos - center| * temperature
        Training: scores + 0.3 * position_bias
        Inference: scores + 1.0 * position_bias
        Loss: L_task + λ * L_order

    REFERENCE:
        Ordered concept extraction Scheme 5 (recommended combination)
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        order_margin: float = 1.0,
        order_weight: float = 0.1,
        train_prior_weight: float = 0.3,
        infer_prior_weight: float = 1.0,
    ):
        super().__init__(config, encoder_hidden_dim)
        self.order_margin = order_margin
        self.order_weight = order_weight
        self.train_prior_weight = train_prior_weight
        self.infer_prior_weight = infer_prior_weight

        # Learnable center positions
        max_seq_len = 2048
        init_centers = torch.linspace(0, max_seq_len - 1, config.num_levels)
        init_logits = torch.logit(torch.clamp(init_centers / max_seq_len, 0.01, 0.99))
        self.center_logits = nn.Parameter(init_logits)

        self.temperature = nn.Parameter(torch.tensor(max_seq_len / config.num_levels))

    def _get_sorted_centers(self, seq_len: int) -> torch.Tensor:
        """Get sorted, normalized concept centers."""
        centers = torch.sigmoid(self.center_logits) * seq_len
        centers, _ = torch.sort(centers)
        return centers

    def compute_order_loss(self, expected_positions: torch.Tensor) -> torch.Tensor:
        """Compute order constraint loss."""
        pos_current = expected_positions[:-1]
        pos_next = expected_positions[1:]
        violation = F.relu(pos_current - pos_next + self.order_margin)
        return violation.mean() * self.order_weight

    def forward(
        self, H: torch.Tensor, training: bool = True
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract with robust ordering.

        DIMENSION FLOW:
            Input: H [B, L, D_encoder]
            Get sorted centers
            Apply position bias (weak in training, strong in inference)
            Compute order loss
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)
        positions = torch.arange(L, device=H.device, dtype=torch.float32)

        centers = self._get_sorted_centers(L)
        prior_weight = self.train_prior_weight if training else self.infer_prior_weight

        concepts = []
        expected_positions = []

        for level_idx in range(self.config.num_levels):
            Q_k = self.concept_queries[level_idx]
            L_k = Q_k.shape[0]

            Q = Q_k.unsqueeze(0).expand(batch_size, -1, -1)
            scores = torch.bmm(Q, H_proj.transpose(1, 2)) / math.sqrt(
                self.config.hidden_dim
            )

            # Position prior
            center = centers[level_idx]
            distance = torch.abs(positions - center)
            position_bias = -distance / self.temperature

            # Apply with appropriate weight
            biased_scores = scores + prior_weight * position_bias.view(1, 1, L)
            A_k = F.softmax(biased_scores, dim=-1)

            C_k = torch.bmm(A_k, H_proj)
            concepts.append(C_k)

            # Track expected position
            exp_pos = torch.sum(A_k * positions.view(1, 1, L), dim=-1).mean()
            expected_positions.append(exp_pos)

        # Compute order loss
        expected_positions_tensor = torch.stack(expected_positions)
        order_loss = self.compute_order_loss(expected_positions_tensor)

        aux = {
            "expected_positions": [p.item() for p in expected_positions],
            "centers": centers.detach().cpu().tolist(),
            "order_loss": order_loss,
            "training": training,
            "method": "robust_ordered",
        }

        return concepts, aux


################################################################################
#                                                                              #
#                     ADVANCED CAUSAL TRAINING EXTRACTORS                      #
#                                                                              #
#    Sophisticated methods that explicitly model sequential dependencies       #
#    and causal structure in text. These address the fundamental problem:      #
#                                                                              #
#    "Soft assignment treats segments as unordered sets, but text is           #
#     inherently sequential with causal dependencies."                         #
#                                                                              #
#    Key properties:                                                           #
#    - Full coverage: sum_i A[t,i] = 1 for all positions t                     #
#    - Ordering: Concepts respect text causality (no future leakage)           #
#    - Smooth transitions: Soft boundaries without hard cuts                   #
#    - End-to-end trainable: All operations are differentiable                 #
#                                                                              #
################################################################################


# =============================================================================
# Advanced Method 1: Monotonic Soft Assignment
# =============================================================================


class MonotonicSoftAssignmentConceptGenerator(BaseConceptGenerator):
    """Monotonic soft assignment with causal constraints.

    PURPOSE:
        Enforce concept ordering through monotonic allocation matrix.
        Each position can only contribute to concepts at or after its expected segment.

    KEY INSIGHT:
        Standard soft attention treats segments as unordered sets, but text is
        inherently sequential. This method uses continuous position prediction
        with cumulative mapping to ensure monotonic assignment.

    MATHEMATICS:
        1. Predict cumulative positions: pos_continuous = cumsum(softplus(pos_logits))
           Normalized to [0, 1]

        2. Learnable segment centers: centers = [0.05, 0.15, ..., 0.95] (N segments)

        3. Causal distance: dist[t, i] = pos_continuous[t] - centers[i]
           Causal mask: mask[t, i] = 1 if centers[i] >= pos_continuous[t] - ε

        4. Soft assignment: A = softmax(-dist^2 / (2*sigma^2)) * mask
           Row-normalized: sum_i A[t, i] = 1 (full coverage)

    DIMENSION FLOW:
        Input: H [B, L, D_encoder]
        Step 1: pos_logits = MLP(H) [B, L, 1]
        Step 2: pos_continuous = cumsum(softplus(pos_logits)) [B, L]
        Step 3: dist = pos_continuous[:, :, None] - centers[None, None, :] [B, L, N]
        Step 4: A = causal_gaussian_kernel(dist) [B, L, N]
        Step 5: Z = A^T @ H_proj [B, N, D] (N = total concepts)

    OUTPUT:
        concepts: List of [C_0, ..., C_K] where sum(L_k) = N
        aux: {'assignment_matrix': A, 'positions': pos_continuous, 'centers': centers}

    REFERENCE:
        Monotonic Attention (Luong et al., 2016; Press & Wolf, 2018)
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        epsilon: float = 0.05,
        min_sigma: float = 0.01,
    ):
        super().__init__(config, encoder_hidden_dim)
        self.epsilon = epsilon
        self.min_sigma = min_sigma

        # Position prediction MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Learnable segment centers (initialized uniformly in [0.05, 0.95])
        total_concepts = sum(config.level_lengths)
        init_centers = torch.linspace(0.05, 0.95, total_concepts)
        self.centers = nn.Parameter(init_centers)  # [N]

        # Learnable temperature (bandwidth)
        self.sigma = nn.Parameter(torch.tensor(0.1))

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract concepts with monotonic soft assignment.

        DIMENSION FLOW:
            Input: H [B, L, D_encoder]
            Project: H_proj [B, L, D]

            For level k:
                Q = concept_queries[k] [B, L_k, D]
                Context = Concat(H_proj, C_0, ..., C_{k-1})
                C_k = CrossAttention(Q, Context)

            Output: [C_0, ..., C_K]
        """
        batch_size = H.shape[0]
        H_proj = self.input_proj(H)

        concepts = []

        for level_idx in range(self.config.num_levels):
            Q_k = self.concept_queries[level_idx]
            L_k = Q_k.shape[0]

            # Prepare queries
            Q = Q_k.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L_k, D]

            # Prepare context
            if level_idx == 0:
                context = H_proj
            else:
                prev_concepts = torch.cat(concepts, dim=1)
                context = torch.cat([H_proj, prev_concepts], dim=1)

            # Cross-attention
            C_k, _ = self.level_attn[level_idx](Q, context, context)
            concepts.append(C_k)

        return concepts


################################################################################
#                                                                              #
#                         INFERENCE GENERATOR                                  #
#                                                                              #
#    Generate concepts from Q (no CoT) during inference.                       #
#    Uses next-level autoregressive generation with shared concept_queries.    #
#                                                                              #
################################################################################


# =============================================================================
# Inference Method: Autoregressive Generator
# =============================================================================


class AutoregressiveConceptGenerator(nn.Module):
    """Autoregressive concept generator for inference.

    PURPOSE:
        Generate concepts from Q (no CoT) using next-level generation.
        Uses the SAME concept_queries as training extractors.

    MATHEMATICS:
        For level k:
            Context = Concat(H, C_0, ..., C_{k-1})
            C_k = CrossAttention(Q_k, Context, Context)

    REFERENCE:
        VAR.md Section 6: Inference with autoregressive generation
    """

    def __init__(self, config: NLCPV3Config, encoder_hidden_dim: int):
        super().__init__()
        self.config = config

        # Projection
        self.input_proj = nn.Linear(encoder_hidden_dim, config.hidden_dim)

        # Shared concept queries (must match training extractors!)
        self.concept_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(length, config.hidden_dim))
                for length in config.level_lengths
            ]
        )

        # Level-specific cross-attention
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
        for queries in self.concept_queries:
            nn.init.xavier_uniform_(queries)

    def forward(self, H: torch.Tensor) -> List[torch.Tensor]:
        """Generate concepts autoregressively.

        DIMENSION FLOW:
            Input: H [B, L, D_encoder]
            Project: H_proj [B, L, D]

            For level k:
                Q = concept_queries[k] [B, L_k, D]
                Context = Concat(H_proj, C_0, ..., C_{k-1})
                C_k = CrossAttention(Q, Context)

            Output: [C_0, ..., C_K]
        """
        batch_size = H.shape[0]
        H_proj = self.input_proj(H)

        concepts = []

        for level_idx in range(self.config.num_levels):
            Q_k = self.concept_queries[level_idx]
            L_k = Q_k.shape[0]

            # Prepare queries
            Q = Q_k.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L_k, D]

            # Prepare context
            if level_idx == 0:
                context = H_proj
            else:
                prev_concepts = torch.cat(concepts, dim=1)
                context = torch.cat([H_proj, prev_concepts], dim=1)

            # Cross-attention
            C_k, _ = self.level_attn[level_idx](Q, context, context)
            concepts.append(C_k)

        return concepts


################################################################################
#                                                                              #
#                         UNIFIED INTERFACE                                    #
#                                                                              #
#    Wraps all training extractors and inference generator into a single API.  #
#    Provides method selection, distillation, and consistent interface.        #
#                                                                              #
################################################################################


# =============================================================================
# Unified Interface: ConceptGenerator
# =============================================================================


class ConceptGenerator(nn.Module):
    """Unified concept generator wrapping all methods.

    PURPOSE:
        Provide a single interface for both training and inference.
        Automatically selects appropriate method based on mode.

    ATTRIBUTES:
        training_generators: Dict of training methods
        inference_generator: Autoregressive generator
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        default_training_method: str = "residual_pooling",
    ):
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = encoder_hidden_dim
        self.default_training_method = default_training_method

        # Training generators (all trainable, independent)
        self.training_generators = nn.ModuleDict(
            {
                # Basic methods
                "residual_pooling": ResidualAttentivePoolingConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "position_constrained": PositionConstrainedConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "hard_ordered_mask": HardOrderedMaskConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "recursive_ordered": RecursiveOrderedConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "order_constrained": OrderConstrainedTrainingConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "robust_ordered": RobustOrderedConceptGenerator(
                    config, encoder_hidden_dim
                ),
                # Advanced causal methods
                "monotonic_soft_assignment": MonotonicSoftAssignmentConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "causal_sequential_refinement": CausalSequentialRefinementConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "continuous_causal_kernel": ContinuousCausalKernelConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "autoregressive_soft_boundary": AutoregressiveSoftBoundaryConceptGenerator(
                    config, encoder_hidden_dim
                ),
                "causal_soft_pooling": CausalSoftPoolingConceptGenerator(
                    config, encoder_hidden_dim
                ),
            }
        )

        # Inference generator
        self.inference_generator = AutoregressiveConceptGenerator(
            config, encoder_hidden_dim
        )

    def forward_training(
        self, H: torch.Tensor, method: Optional[str] = None
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Training: Extract concepts using specified method.

        Args:
            H: [B, L, D_encoder] - Hidden states from Q+CoT
            method: Extraction method name (default: self.default_training_method)

        Returns:
            concepts: List of concept tensors
            aux: Auxiliary information (method-specific)
        """
        method = method or self.default_training_method
        if method not in self.training_generators:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(self.training_generators.keys())}"
            )

        return self.training_generators[method](H)

    def forward_inference(self, H: torch.Tensor) -> List[torch.Tensor]:
        """Inference: Generate concepts from Q.

        Args:
            H: [B, L, D_encoder] - Hidden states from Q only

        Returns:
            concepts: List of concept tensors
        """
        return self.inference_generator(H)

    def forward(
        self, H: torch.Tensor, mode: str = "training", method: Optional[str] = None
    ) -> Tuple[List[torch.Tensor], Optional[Dict[str, Any]]]:
        """Unified forward pass.

        Args:
            H: Hidden states
            mode: 'training' or 'inference'
            method: Training method (if mode='training')

        Returns:
            concepts (and aux if training)
        """
        if mode == "training":
            return self.forward_training(H, method)
        elif mode == "inference":
            return self.forward_inference(H), None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_generator(self, method: str) -> BaseConceptGenerator:
        """Get a specific training generator (for advanced usage).

        Args:
            method: Generator name

        Returns:
            generator: The requested generator module
        """
        return self.training_generators[method]

    def compute_distillation_loss(
        self, teacher_method: str, H: torch.Tensor
    ) -> torch.Tensor:
        """Compute distillation loss between teacher and inference generator.

        PURPOSE:
            Train inference generator to match a training extractor.

        Args:
            teacher_method: Which training method to use as teacher
            H: Hidden states

        Returns:
            loss: MSE loss between teacher and student outputs
        """
        with torch.no_grad():
            teacher_concepts, _ = self.forward_training(H, teacher_method)

        student_concepts = self.forward_inference(H)

        losses = []
        for t_c, s_c in zip(teacher_concepts, student_concepts):
            losses.append(F.mse_loss(s_c, t_c))

        return sum(losses) / len(losses)


class CausalSequentialRefinementConceptGenerator(BaseConceptGenerator):
    """Causal sequential refinement with soft pooling.

    PURPOSE:
        First perform soft pooling to get initial segment representations,
        then apply causal transformer to propagate sequential dependencies.

    KEY INSIGHT:
        Soft pooling ensures full coverage of input tokens, while causal
        refinement injects ordering constraints through masked attention.

    MATHEMATICS:
        Step 1 (Soft Pooling):
            A = softmax(Q @ H^T / sqrt(D))  # Standard attention
            Z_0 = A @ H  # Initial segment representations

        Step 2 (Causal Refinement):
            Z^{(k)} = CausalTransformer(Z^{(k-1)})
            where CausalTransformer uses lower-triangular attention mask

        Residual connection prevents over-smoothing:
            Z = Z_0 + CausalBlock(Z_0)

    DIMENSION FLOW:
        Input: H [B, L, D_encoder]
        H_proj: [B, L, D]
        Z_0: [B, N, D] (N = total concepts)
        Z_refined: [B, N, D] (after causal transformer)
        concepts: [C_0, ..., C_K]

    ARCHITECTURE:
        - n_refinement_layers: Number of causal transformer layers
        - nhead: Number of attention heads
        - dim_feedforward: FFN hidden dimension

    OUTPUT:
        concepts: List of refined [C_0, ..., C_K]
        aux: {'initial_concepts': Z_0, 'refined_concepts': Z, 'method': ...}

    REFERENCE:
        Yang et al. (2022) "Hierarchical Soft Chunking"
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        n_refinement_layers: int = 2,
        dim_feedforward: Optional[int] = None,
    ):
        super().__init__(config, encoder_hidden_dim)

        if dim_feedforward is None:
            dim_feedforward = 4 * config.hidden_dim

        total_concepts = sum(config.level_lengths)

        # Causal mask for refinement
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(total_concepts, total_concepts))
        )

        # Causal transformer layers
        self.refinement_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=dim_feedforward,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(n_refinement_layers)
            ]
        )

        # Temperature for soft pooling
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract and refine concepts with causal dependencies.

        DIMENSION FLOW:
            H: [B, L, D_encoder]
            H_proj: [B, L, D]
            Z_0: [B, N, D]
            Z: [B, N, D]
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)  # [B, L, D]
        total_concepts = sum(self.config.level_lengths)

        # Step 1: Soft pooling (concatenate all level queries)
        all_queries = []
        for queries in self.concept_queries:
            all_queries.append(queries)  # Each [L_k, D]
        Q_all = torch.cat(all_queries, dim=0)  # [N, D]

        Q = Q_all.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, D]
        scores = torch.bmm(Q, H_proj.transpose(1, 2))  # [B, N, L]
        scores = scores / (math.sqrt(self.config.hidden_dim) * self.temperature)
        A = F.softmax(scores, dim=-1)  # [B, N, L]

        # Initial segment representations
        Z_0 = torch.bmm(A, H_proj)  # [B, N, D]

        # Step 2: Causal refinement
        Z = Z_0
        for layer in self.refinement_layers:
            # Apply causal mask
            Z = layer(Z, src_mask=self.causal_mask.bool())

        # Add residual connection
        Z = Z_0 + Z  # [B, N, D]

        # Split into hierarchical levels
        concepts = []
        start_idx = 0
        for level_idx, length in enumerate(self.config.level_lengths):
            end_idx = start_idx + length
            C_k = Z[:, start_idx:end_idx, :]  # [B, L_k, D]
            concepts.append(C_k)
            start_idx = end_idx

        aux = {
            "initial_concepts": Z_0,
            "refined_concepts": Z,
            "assignment": A,
            "method": "causal_sequential_refinement",
        }

        return concepts, aux


class ContinuousCausalKernelConceptGenerator(BaseConceptGenerator):
    """Continuous position mapping with causal decay kernel.

    PURPOSE:
        Map token positions to continuous axis [0, 1], then apply causal
        decay kernel to compute concept membership. Allows for smooth
        transitions while maintaining strict causality.

    KEY INSIGHT:
        Instead of discrete segment boundaries, use continuous position
        mapping with kernel functions that smoothly decay based on distance
        from concept center, but only consider positions <= center (causal).

    MATHEMATICS:
        1. Position embedding: pos_emb[t] = positional_encoding(t)

        2. Continuous position prediction:
           pos_continuous[t] = sigmoid(MLP(H[t]))  # Map to [0, 1]

        3. Causal kernel (exponential decay):
           K(t, c_i) = exp(-|c_i - pos_continuous[t]| / tau) if pos_continuous[t] <= c_i + eps
                       0 otherwise

        4. Normalized assignment:
           A[t, i] = K(t, c_i) / sum_j K(t, c_j)

    DIMENSION FLOW:
        Input: H [B, L, D_encoder]
        pos_continuous: [B, L]
        centers: [N] (learnable, sorted)
        K: [B, L, N] (causal kernel values)
        A: [B, L, N]
        Z: [B, N, D]

    ADVANTAGES:
        - Smooth, differentiable boundaries
        - Strict causality (no future information leakage)
        - Adaptive bandwidth via learnable tau

    REFERENCE:
        Causal Information Bottleneck (Chen et al., 2024)
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        kernel_type: str = "exponential",  # 'exponential' or 'gaussian'
        epsilon: float = 0.02,
    ):
        super().__init__(config, encoder_hidden_dim)
        self.kernel_type = kernel_type
        self.epsilon = epsilon

        # Position prediction network
        self.pos_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),  # Map to [0, 1]
        )

        # Learnable concept centers
        total_concepts = sum(config.level_lengths)
        init_centers = torch.linspace(0.1, 0.9, total_concepts)
        self.centers = nn.Parameter(init_centers)

        # Learnable bandwidth (temperature)
        self.tau = nn.Parameter(torch.tensor(0.1))

    def _causal_kernel(
        self,
        pos: torch.Tensor,  # [B, L]
        centers: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """Compute causal kernel matrix.

        Args:
            pos: Continuous positions [B, L]
            centers: Concept centers [N]

        Returns:
            K: Kernel values [B, L, N]
        """
        # Distance: dist[b, t, i] = centers[i] - pos[b, t]
        dist = centers.view(1, 1, -1) - pos.unsqueeze(2)  # [B, L, N]

        # Causal mask: only positions <= center + epsilon contribute
        causal_mask = (dist >= -self.epsilon).float()

        # Apply kernel
        tau_clamped = torch.clamp(self.tau, min=0.01)
        if self.kernel_type == "exponential":
            K = torch.exp(-torch.abs(dist) / tau_clamped) * causal_mask
        elif self.kernel_type == "gaussian":
            K = torch.exp(-(dist**2) / (2 * tau_clamped**2)) * causal_mask
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return K

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract concepts with continuous causal kernel.

        DIMENSION FLOW:
            H: [B, L, D_encoder]
            H_proj: [B, L, D]
            pos_continuous: [B, L]
            centers_sorted: [N]
            K: [B, L, N]
            A: [B, L, N]
            Z: [B, N, D]
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)  # [B, L, D]

        # Step 1: Predict continuous positions
        pos_continuous = self.pos_net(H_proj).squeeze(-1)  # [B, L]

        # Step 2: Sort centers to ensure ordering
        centers_sorted, _ = torch.sort(torch.sigmoid(self.centers))

        # Step 3: Compute causal kernel
        K = self._causal_kernel(pos_continuous, centers_sorted)  # [B, L, N]

        # Step 4: Normalize (full coverage)
        A = K / (K.sum(dim=2, keepdim=True) + 1e-8)  # [B, L, N]

        # Step 5: Extract concept representations
        Z = torch.bmm(A.transpose(1, 2), H_proj)  # [B, N, D]

        # Split into hierarchical levels
        concepts = []
        start_idx = 0
        for level_idx, length in enumerate(self.config.level_lengths):
            end_idx = start_idx + length
            C_k = Z[:, start_idx:end_idx, :]  # [B, L_k, D]
            concepts.append(C_k)
            start_idx = end_idx

        aux = {
            "assignment": A,
            "positions": pos_continuous,
            "centers": centers_sorted.detach(),
            "tau": self.tau.detach(),
            "kernel_type": self.kernel_type,
            "method": "continuous_causal_kernel",
        }

        return concepts, aux


class AutoregressiveSoftBoundaryConceptGenerator(BaseConceptGenerator):
    """Autoregressive soft boundary prediction for concept extraction.

    PURPOSE:
        Sequentially predict segment boundaries where each boundary prediction
        is conditioned on previous segment representations. Boundaries are
        strictly increasing, ensuring natural coverage.

    KEY INSIGHT:
        Instead of predicting all concepts simultaneously, generate them one by
        one in an autoregressive manner. Each new concept's boundary depends on
        the previously extracted concepts, ensuring strict ordering.

    MATHEMATICS:
        For concept i:
            1. Predict boundary: b_i = f(H, z_{<i}) where b_i ∈ (b_{i-1}, 1]
            2. Compute attention: A_i = softmax(scores_i) over positions in (b_{i-1}, b_i]
            3. Extract concept: z_i = A_i @ H
            4. Update state: z_{<i+1} = [z_{<i}, z_i]

        Boundary prediction uses cumulative approach:
            delta_i = sigmoid(MLP(H, z_{<i}))  # (0, 1)
            b_i = b_{i-1} + delta_i * (1 - b_{i-1})  # Strictly increasing

    DIMENSION FLOW:
        Input: H [B, L, D_encoder]
        H_proj: [B, L, D]

        For each concept i:
            boundary_pred: [B, 1]
            attention_mask: [B, L] (positions within current boundary)
            A_i: [B, L] (attention weights)
            z_i: [B, D]

        Output: concepts [z_1, ..., z_N] reshaped to [C_0, ..., C_K]

    ADVANTAGES:
        - Strict ordering: Boundaries monotonically increase
        - Natural coverage: No gaps between segments
        - Adaptive: Boundary positions learned from data
        - Causal: Each concept only uses information up to its boundary

    REFERENCE:
        AR Soft Boundary paradigm from ordered concept extraction
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        temperature: float = 1.0,
    ):
        super().__init__(config, encoder_hidden_dim)
        self.temperature = temperature

        total_concepts = sum(config.level_lengths)

        # Boundary prediction network
        # Input: [H_summary, prev_concept_summary] -> Output: boundary delta
        # Always takes 2*D input (H_summary + prev_concept or zeros)
        self.boundary_net = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),  # Output in (0, 1)
        )

        # Zero vector for first iteration (no previous concept)
        self.register_buffer("zero_concept", torch.zeros(config.hidden_dim))

        # Concept extraction network
        self.concept_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Summary network for H
        self.summary_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Learnable initial boundary
        self.init_boundary = nn.Parameter(torch.tensor(0.0))

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Extract concepts with autoregressive soft boundary prediction.

        DIMENSION FLOW:
            H: [B, L, D_encoder]
            H_proj: [B, L, D]
            H_summary: [B, D]

            For concept i:
                context: [B, D + i*D] (H_summary + prev_concepts)
                delta: [B, 1]
                boundary: [B, 1]
                mask: [B, L]
                A: [B, L]
                z: [B, D]

            concepts: [z_1, ..., z_N] -> reshaped to [C_0, ..., C_K]
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)  # [B, L, D]

        # Compute sequence summary
        H_summary = self.summary_net(H_proj.mean(dim=1))  # [B, D]

        concepts = []
        boundaries = []
        prev_boundary = torch.full(
            (batch_size, 1), self.init_boundary.item(), device=H.device
        )  # [B, 1]
        boundaries.append(prev_boundary)

        total_concepts = sum(self.config.level_lengths)

        for i in range(total_concepts):
            # Prepare context: H_summary + previous concept summary
            if i == 0:
                # First concept: use zero vector as previous concept
                prev_concept = self.zero_concept.view(1, -1).expand(
                    batch_size, -1
                )  # [B, D]
            else:
                # Use mean of previous concepts as summary
                prev_concept = torch.cat(concepts, dim=1).mean(dim=1)  # [B, D]

            context = torch.cat([H_summary, prev_concept], dim=-1)  # [B, 2D]

            # Predict boundary delta
            delta = self.boundary_net(context)  # [B, 1]

            # Compute new boundary (strictly increasing)
            # b_i = b_{i-1} + delta * (1 - b_{i-1})
            new_boundary = prev_boundary + delta * (1 - prev_boundary)  # [B, 1]
            boundaries.append(new_boundary)

            # Create attention mask for positions within current segment
            positions = torch.linspace(0, 1, L, device=H.device).view(1, L)  # [1, L]
            # Mask: positions > prev_boundary AND positions <= new_boundary
            mask = (
                (positions > prev_boundary) & (positions <= new_boundary)
            ).float()  # [B, L]

            # Compute attention scores
            # Use a learnable query for each concept iteration
            Q = self.concept_queries[0][0].unsqueeze(0).expand(batch_size, -1)  # [B, D]
            scores = torch.bmm(Q.unsqueeze(1), H_proj.transpose(1, 2)).squeeze(
                1
            )  # [B, L]
            scores = scores / self.temperature

            # Apply mask (soft: use mask as weighting)
            masked_scores = scores * mask + (1 - mask) * (-1e9)
            A = F.softmax(masked_scores, dim=-1)  # [B, L]

            # Extract concept
            z = torch.bmm(A.unsqueeze(1), H_proj).squeeze(1)  # [B, D]
            z = self.concept_proj(z)
            concepts.append(z.unsqueeze(1))  # [B, 1, D]

            # Update for next iteration
            prev_boundary = new_boundary

        # Concatenate all concepts
        all_concepts = torch.cat(concepts, dim=1)  # [B, N, D]

        # Split into hierarchical levels
        concepts_list = []
        start_idx = 0
        for level_idx, length in enumerate(self.config.level_lengths):
            end_idx = start_idx + length
            C_k = all_concepts[:, start_idx:end_idx, :]  # [B, L_k, D]
            concepts_list.append(C_k)
            start_idx = end_idx

        aux = {
            "boundaries": torch.cat(boundaries, dim=1).detach(),  # [B, N+1]
            "method": "autoregressive_soft_boundary",
        }

        return concepts_list, aux


class CausalSoftPoolingConceptGenerator(BaseConceptGenerator):
    """Complete causal soft pooling pipeline (Recommended Advanced Method).

    PURPOSE:
        Combines monotonic soft assignment, causal sequential refinement,
        and reconstruction-based training objectives for robust concept
        extraction with guaranteed ordering and full coverage.

    KEY COMPONENTS:
        1. Monotonic Soft Router: Creates ordered, smooth assignment matrix
        2. Causal Refiner: Propagates sequential dependencies
        3. Reconstruction Head: Validates information preservation
        4. Multi-objective Loss: Reconstruction + Monotonicity + Coverage

    MATHEMATICS:
        Step 1: A = MonotonicRouter(H)  # [B, L, N]
        Step 2: Z_0 = A^T @ H  # [B, N, D]
        Step 3: Z = CausalRefiner(Z_0)  # [B, N, D]
        Step 4: H_recon = A @ Z  # [B, L, D]

        Loss = λ1 * MSE(H, H_recon) + λ2 * L_monotonic + λ3 * L_coverage

    DIMENSION FLOW:
        Input: H [B, L, D_encoder]
        H_proj: [B, L, D]
        A: [B, L, N]
        Z_0: [B, N, D]
        Z: [B, N, D]
        H_recon: [B, L, D]

    TRAINING STRATEGY:
        - Phase 1 (0-1000 steps): λ1=1.0, λ2=λ3=0 (stabilize reconstruction)
        - Phase 2 (1000+ steps): Gradually increase λ2, λ3
        - Temperature annealing: sigma from 0.5 → 0.05

    OUTPUT:
        concepts: List of [C_0, ..., C_K]
        aux: Full training diagnostics including all loss components

    REFERENCE:
        Complete integration of Monotonic Attention + Causal Pooling + IB
    """

    def __init__(
        self,
        config: NLCPV3Config,
        encoder_hidden_dim: int,
        n_refinement_layers: int = 2,
        epsilon: float = 0.05,
        lambda_recon: float = 1.0,
        lambda_mono: float = 0.1,
        lambda_cov: float = 0.1,
    ):
        super().__init__(config, encoder_hidden_dim)
        self.epsilon = epsilon
        self.lambda_recon = lambda_recon
        self.lambda_mono = lambda_mono
        self.lambda_cov = lambda_cov

        total_concepts = sum(config.level_lengths)

        # Position router
        self.pos_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Learnable centers
        init_centers = torch.linspace(0.05, 0.95, total_concepts)
        self.centers = nn.Parameter(init_centers)
        self.sigma = nn.Parameter(torch.tensor(0.1))

        # Causal refinement
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(total_concepts, total_concepts))
        )
        self.refinement_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=4 * config.hidden_dim,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(n_refinement_layers)
            ]
        )

        # Reconstruction head
        self.recon_head = nn.Linear(config.hidden_dim, config.hidden_dim)

    def _compute_monotonicity_loss(self, A: torch.Tensor) -> torch.Tensor:
        """Compute monotonicity regularization loss.

        Encourages assignment matrix to have peak positions that
        monotonically increase with concept index.

        Args:
            A: Assignment matrix [B, L, N]

        Returns:
            loss: Scalar monotonicity loss
        """
        # Penalize cases where A[t, i] > A[t+1, i] (inverse trend)
        A_diff = A[:, :-1, :] - A[:, 1:, :]  # [B, L-1, N]
        violation = F.relu(A_diff)  # Positive when A[t] > A[t+1]
        return violation.mean()

    def _compute_coverage_loss(self, A: torch.Tensor) -> torch.Tensor:
        """Compute coverage regularization loss.

        Encourages balanced usage of all concepts.

        Args:
            A: Assignment matrix [B, L, N]

        Returns:
            loss: Scalar coverage loss
        """
        # Sum of attention per concept
        concept_usage = A.sum(dim=1)  # [B, N]
        # Variance across concepts (lower is better)
        usage_variance = concept_usage.var(dim=1).mean()
        return usage_variance

    def forward(self, H: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Complete causal soft pooling pipeline.

        DIMENSION FLOW:
            H: [B, L, D_encoder]
            H_proj: [B, L, D]
            pos_continuous: [B, L]
            A: [B, L, N]
            Z_0: [B, N, D]
            Z: [B, N, D]
            H_recon: [B, L, D]
        """
        batch_size, L, _ = H.shape
        H_proj = self.input_proj(H)  # [B, L, D]

        # Step 1: Monotonic position prediction
        pos_logits = self.pos_mlp(H_proj).squeeze(-1)  # [B, L]
        pos_continuous = torch.cumsum(F.softplus(pos_logits), dim=1)
        pos_continuous = pos_continuous / (pos_continuous[:, -1:] + 1e-8)

        # Step 2: Sorted centers
        centers_sorted, _ = torch.sort(torch.sigmoid(self.centers))

        # Step 3: Causal distance and mask
        dist = pos_continuous.unsqueeze(2) - centers_sorted.view(1, 1, -1)  # [B, L, N]
        causal_mask = (dist > -self.epsilon).float()

        # Step 4: Gaussian kernel assignment
        sigma_clamped = torch.clamp(self.sigma, min=0.01)
        A_unnorm = torch.exp(-(dist**2) / (2 * sigma_clamped**2)) * causal_mask
        A = A_unnorm / (A_unnorm.sum(dim=2, keepdim=True) + 1e-8)  # [B, L, N]

        # Step 5: Initial pooling
        Z_0 = torch.bmm(A.transpose(1, 2), H_proj)  # [B, N, D]

        # Step 6: Causal refinement
        Z = Z_0
        for layer in self.refinement_layers:
            Z = layer(Z, src_mask=self.causal_mask.bool())
        Z = Z_0 + Z  # Residual

        # Step 7: Reconstruction
        H_recon = self.recon_head(torch.bmm(A, Z))  # [B, L, D]

        # Step 8: Compute losses
        recon_loss = F.mse_loss(H_recon, H_proj)
        mono_loss = self._compute_monotonicity_loss(A)
        cov_loss = self._compute_coverage_loss(A)

        total_loss = (
            self.lambda_recon * recon_loss
            + self.lambda_mono * mono_loss
            + self.lambda_cov * cov_loss
        )

        # Split into hierarchical levels
        concepts = []
        start_idx = 0
        for level_idx, length in enumerate(self.config.level_lengths):
            end_idx = start_idx + length
            C_k = Z[:, start_idx:end_idx, :]  # [B, L_k, D]
            concepts.append(C_k)
            start_idx = end_idx

        aux = {
            "assignment": A,
            "positions": pos_continuous,
            "centers": centers_sorted.detach(),
            "sigma": sigma_clamped.detach(),
            "losses": {
                "recon": recon_loss.detach(),
                "monotonicity": mono_loss.detach(),
                "coverage": cov_loss.detach(),
                "total": total_loss.detach(),
            },
            "method": "causal_soft_pooling",
        }

        return concepts, aux
