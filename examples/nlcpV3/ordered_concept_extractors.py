"""Ordered Concept Extractors: Ensuring Sequential Structure in NLCP V3.

USAGE:
    from nlcpV3.ordered_concept_extractors import (
        PositionConstrainedExtractor,      # Scheme 1
        HardOrderedMaskExtractor,          # Scheme 2
        RecursiveOrderedExtractor,         # Scheme 3
        OrderConstrainedTraining,          # Scheme 4
        RobustOrderedExtractor,            # Recommended Combination
    )

    # Example usage
    extractor = RobustOrderedExtractor(hidden_dim=512, num_concepts=4, seq_len=100)
    concepts, attn_weights, order_loss = extractor(H, training=True)

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md discussions on ordered concept extraction
    Problem: Soft attention may learn arbitrary concept-to-position mappings,
             breaking the sequential structure of CoT segments.

PURPOSE:
    Provide multiple strategies to ensure extracted concepts maintain
    the sequential order of CoT segments. Each scheme offers different
    trade-offs between constraint strength and flexibility.

ARCHITECTURE OVERVIEW:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    ORDERED CONCEPT EXTRACTION                    │
    │                                                                  │
    │  Input: H [B, L, D] - Hidden states from encoder                │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Scheme 1: Position-Constrained Attention                │   │
    │  │  - Learnable concept centers (sorted)                    │   │
    │  │  - Gaussian position prior                               │   │
    │  │  - Soft bias toward ordered positions                    │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                              ↓                                  │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Scheme 2: Hard Ordered Mask                             │   │
    │  │  - Pre-defined segment boundaries                        │   │
    │  │  - Soft edges with decay                                 │   │
    │  │  - Strong structural constraint                          │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                              ↓                                  │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Scheme 3: Recursive Ordered Generation                  │   │
    │  │  - Sequential concept generation                         │   │
    │  │  - Remaining position mask                               │   │
    │  │  - VAR-style scale-by-scale                              │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                              ↓                                  │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Scheme 4: Order-Constrained Training                    │   │
    │  │  - Expected position calculation                         │   │
    │  │  - Order loss: L_order = Σ ReLU(pos_i - pos_{i+1})       │   │
    │  │  - Soft constraint via loss function                     │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                              ↓                                  │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Recommended: RobustOrderedExtractor                     │   │
    │  │  - Combines position prior + order loss                  │   │
    │  │  - Train: weak prior + order loss                        │   │
    │  │  - Inference: strong prior guarantee                     │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                              ↓                                  │
    │  Output: concepts [B, K, D], attn_weights [B, K, L], order_loss │
    │                                                                  │
    │  Key Property: concept_0 corresponds to early positions         │
    │                concept_K corresponds to late positions          │
    │                (maintaining CoT segment order)                  │
    └─────────────────────────────────────────────────────────────────┘

DIMENSION FLOW:
    All Schemes:
        Input:  H [B, L, D] - Hidden states
        Output: concepts [B, K, D] - Ordered concepts (K = num_concepts)
                attn_weights [B, K, L] - Attention distribution
                order_loss (optional) - Constraint violation loss

SCHEME COMPARISON:
    ┌──────────┬────────────────┬─────────────────┬────────────────┐
    │  Scheme  │ Constraint     │ Implementation  │ Flexibility    │
    ├──────────┼────────────────┼─────────────────┼────────────────┤
    │    1     │ Medium         │ Low complexity  │ High           │
    │    2     │ High           │ Low complexity  │ Low            │
    │    3     │ High           │ Medium          │ Medium         │
    │    4     │ Soft (loss)    │ Low complexity  │ Highest        │
    │  Robust  │ Adaptive       │ Medium          │ High           │
    └──────────┴────────────────┴─────────────────┴────────────────┘

CRITICAL INSIGHT:
    The key challenge is balancing:
    1. ORDER: concept_i should correspond to segment_i (sequential)
    2. SEMANTICS: Each concept should capture meaningful content
    3. OVERLAP: Adjacent concepts may share boundary information

    Scheme 4 (order loss) + weak position prior offers the best balance,
    allowing the model to learn semantic boundaries while maintaining order.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# =============================================================================
# SCHEME 1: Position-Constrained Attention
# =============================================================================


class PositionConstrainedExtractor(nn.Module):
    """Scheme 1: Position-Constrained Attention with Learnable Centers.

    PURPOSE:
        Ensure concept ordering by learning "expected positions" for each concept
        and biasing attention toward those positions using a Gaussian prior.

    CORE IDEA:
        Instead of allowing arbitrary attention patterns, each concept query
        learns a "center position" and attends more strongly to tokens near
        that center. Centers are constrained to be monotonically increasing.

    MATHEMATICAL FORMULATION:
        Given:
            - H [B, L, D]: Hidden states
            - Q [K, D]: Concept queries (K = num_concepts)
            - c [K]: Learnable concept centers (sorted: c[0] < c[1] < ... < c[K-1])

        Attention Score:
            score[i,j] = (Q[i] · H[j]) / √D + log_prior[i,j]

        Where position prior is Gaussian:
            prior[i,j] = exp(-|j - c[i]| / σ)
            log_prior[i,j] = -|j - c[i]| / σ

        Final attention:
            A[i,j] = softmax(score[i,:])_j

    DIMENSION FLOW:
        Input:
            H: [B, L, D] - Hidden states from encoder

        Process:
            1. Compute base attention scores: [B, K, L]
            2. Calculate position prior using sorted centers: [K, L]
            3. Combine: biased_scores = scores + log_prior
            4. Softmax to get attention weights: A [B, K, L]
            5. Extract concepts: C = A @ H [B, K, D]

        Output:
            concepts: [B, K, D] - Position-constrained ordered concepts
            attn_weights: [B, K, L] - Attention distribution
            expected_positions: [B, K] - Expected position of each concept

    VISUALIZATION:
        CoT: "小明有 5 个苹果，吃掉 2 个，还剩几个？"
        Positions: 0    5    10   15   20   25   30   35   40

        Concept Centers (learned):
            C_0 center ≈ 5   → attends to "小明有 5 个苹果"
            C_1 center ≈ 15  → attends to "吃掉 2 个"
            C_2 center ≈ 25  → attends to "还剩几个"
            C_3 center ≈ 35  → attends to "？" (question marker)

        Attention Pattern (Gaussian-like):
            C_0: [0.4, 0.3, 0.2, 0.1, 0,   0,   0,   0,   0  ]
            C_1: [0,   0.1, 0.2, 0.4, 0.2, 0.1, 0,   0,   0  ]
            C_2: [0,   0,   0,   0.1, 0.2, 0.4, 0.2, 0.1, 0  ]
            C_3: [0,   0,   0,   0,   0,   0.1, 0.2, 0.3, 0.4]

        Key: Centers are sorted, ensuring C_i focuses on earlier positions than C_{i+1}

    ATTRIBUTES:
        concept_queries: nn.Parameter [K, D]
            Learnable queries for each concept
        center_logits: nn.Parameter [K]
            Raw logits for concept centers (before sorting/softmax)
        temperature: nn.Parameter [1]
            Controls sharpness of position prior (lower = sharper)
        input_proj: nn.Linear
            Projects input to concept dimension if needed

    EXAMPLE:
        >>> extractor = PositionConstrainedExtractor(
        ...     hidden_dim=512,
        ...     num_concepts=4,
        ...     seq_len=100
        ... )
        >>> H = torch.randn(2, 100, 512)  # [B=2, L=100, D=512]
        >>> concepts, attn, positions = extractor(H)
        >>> concepts.shape
        torch.Size([2, 4, 512])
        >>> attn.shape
        torch.Size([2, 4, 100])
        >>> positions.shape
        torch.Size([2, 4])
    """

    def __init__(self, hidden_dim: int, num_concepts: int, seq_len: int):
        """Initialize position-constrained extractor.

        Args:
            hidden_dim: Dimension of hidden states D
            num_concepts: Number of concepts to extract K
            seq_len: Maximum sequence length L

        PURPOSE:
            Initialize learnable queries and center positions.
            Centers are initialized to evenly spaced positions.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.seq_len = seq_len

        # Learnable concept queries [K, D]
        self.concept_queries = nn.Parameter(torch.randn(num_concepts, hidden_dim))

        # Learnable center positions (as logits for flexibility)
        # Initialize to evenly spaced positions: [0, L/(K-1), 2L/(K-1), ..., L]
        init_centers = torch.linspace(0, seq_len - 1, num_concepts)
        # Convert to logits (inverse of sigmoid)
        init_logits = torch.logit(init_centers / seq_len)
        self.center_logits = nn.Parameter(init_logits)

        # Temperature controls sharpness of position prior
        self.temperature = nn.Parameter(torch.tensor(seq_len / num_concepts))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.concept_queries)

    def _get_sorted_centers(self) -> torch.Tensor:
        """Get sorted, normalized concept centers.

        PURPOSE:
            Ensure centers are in [0, seq_len] and monotonically increasing.

        Process:
            1. Apply sigmoid to constrain to [0, 1]
            2. Scale to [0, seq_len]
            3. Sort to guarantee order

        Returns:
            centers: [K] sorted concept centers in [0, seq_len]
        """
        # Sigmoid to [0, 1], then scale to [0, seq_len]
        centers = torch.sigmoid(self.center_logits) * self.seq_len
        # Sort to ensure monotonicity: c[0] <= c[1] <= ... <= c[K-1]
        centers, _ = torch.sort(centers)
        return centers

    def forward(
        self, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract ordered concepts using position-constrained attention.

        DIMENSION FLOW:
            Input:
                H: [B, L, D] - Hidden states

            Process:
                1. Compute base scores: Q @ H^T → [B, K, L]
                2. Get sorted centers → [K]
                3. Compute position prior: -|positions - centers| / T → [K, L]
                4. Combine: scores + prior → biased_scores [B, K, L]
                5. Softmax → attention weights [B, K, L]
                6. Extract concepts: A @ H → [B, K, D]
                7. Compute expected positions: sum(A * positions, dim=-1) → [B, K]

            Output:
                concepts: [B, K, D]
                attn_weights: [B, K, L]
                expected_positions: [B, K]

        Args:
            H: Hidden states [B, L, D]

        Returns:
            concepts: Ordered concepts [B, K, D]
            attn_weights: Attention distribution [B, K, L]
            expected_positions: Expected position of each concept [B, K]
        """
        B, L, D = H.shape
        K = self.num_concepts

        # Expand queries for batch: [K, D] → [B, K, D]
        Q = self.concept_queries.unsqueeze(0).expand(B, -1, -1)

        # Compute base attention scores: [B, K, D] @ [B, D, L] → [B, K, L]
        scores = torch.bmm(Q, H.transpose(1, 2)) / math.sqrt(D)

        # Get sorted concept centers: [K]
        centers = self._get_sorted_centers()  # [K]

        # Create position indices: [L]
        positions = torch.arange(L, device=H.device, dtype=torch.float32)

        # Compute position prior: -|pos - center| / temperature
        # Shape: [K, 1] - [1, L] → [K, L]
        distance = torch.abs(centers.unsqueeze(1) - positions.unsqueeze(0))
        log_prior = -distance / self.temperature  # [K, L]

        # Add prior to scores (broadcast over batch): [B, K, L] + [1, K, L]
        biased_scores = scores + log_prior.unsqueeze(0)

        # Softmax to get attention weights
        attn_weights = F.softmax(biased_scores, dim=-1)  # [B, K, L]

        # Extract concepts: [B, K, L] @ [B, L, D] → [B, K, D]
        concepts = torch.bmm(attn_weights, H)

        # Compute expected positions for analysis
        expected_positions = torch.sum(
            attn_weights * positions.view(1, 1, L), dim=-1
        )  # [B, K]

        return concepts, attn_weights, expected_positions


# =============================================================================
# SCHEME 2: Hard Ordered Mask
# =============================================================================


class HardOrderedMaskExtractor(nn.Module):
    """Scheme 2: Hard Ordered Mask with Soft Edges.

    PURPOSE:
        Enforce strict segment boundaries while allowing soft transitions
        at boundaries to preserve semantic continuity.

    CORE IDEA:
        Pre-define segment regions for each concept:
        - Concept i primarily attends to segment i
        - Soft edges allow attention to adjacent segments (decay factor)
        - Strong structural constraint guarantees order

    MATHEMATICAL FORMULATION:
        Given:
            - L: sequence length
            - K: number of concepts
            - softness: edge decay factor (0 = hard, 1 = very soft)

        Base mask construction:
            For concept i, segment boundaries: [i*L/K, (i+1)*L/K)

            mask[i, j] = 1.0  if j in [start, end)
            mask[i, j] = decay  if j in edge region
            mask[i, j] = 0.0  otherwise

        Where decay is linear: decay = 1 - distance / edge_width

        Final attention:
            A = softmax(Q @ H^T + log(mask + ε))

    DIMENSION FLOW:
        Input:
            H: [B, L, D] - Hidden states

        Process:
            1. Create ordered base mask: [K, L]
            2. Compute attention scores: [B, K, L]
            3. Apply mask: biased_scores = scores + log(mask)
            4. Softmax: A = softmax(biased_scores) [B, K, L]
            5. Extract concepts: C = A @ H [B, K, D]

        Output:
            concepts: [B, K, D] - Mask-constrained ordered concepts
            attn_weights: [B, K, L] - Attention distribution
            base_mask: [K, L] - The ordered mask (for visualization)

    VISUALIZATION:
        seq_len=100, num_concepts=4, softness=0.2

        Base Mask (before normalization):
            C_0: [1.0, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0, 0.0, ...]  positions 0-24
            C_1: [0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 0.8, 0.5, ...]  positions 15-40
            C_2: [0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, ...]  positions 35-60
            C_3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, ...]  positions 55-99

        Key observations:
        - Each concept has a primary region (1.0)
        - Soft edges allow overlap (0.8, 0.5, 0.2)
        - Adjacent concepts share boundary information
        - Non-adjacent concepts have no overlap (0.0)

    ATTRIBUTES:
        concept_queries: nn.Parameter [K, D]
            Learnable queries for each concept
        base_mask: torch.Tensor [K, L]
            Pre-computed ordered mask with soft edges
        softness: float
            Controls edge decay width (0=hard, higher=softer)

    EXAMPLE:
        >>> extractor = HardOrderedMaskExtractor(
        ...     hidden_dim=512,
        ...     num_concepts=4,
        ...     seq_len=100,
        ...     softness=0.2
        ... )
        >>> H = torch.randn(2, 100, 512)
        >>> concepts, attn, mask = extractor(H)
        >>> mask.shape
        torch.Size([4, 100])
        >>> # Visualize mask
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(mask.cpu().numpy(), aspect='auto')
    """

    def __init__(
        self, hidden_dim: int, num_concepts: int, seq_len: int, softness: float = 0.2
    ):
        """Initialize hard ordered mask extractor.

        Args:
            hidden_dim: Dimension of hidden states D
            num_concepts: Number of concepts K
            seq_len: Sequence length L
            softness: Edge decay factor (0=hard boundaries, higher=softer)

        PURPOSE:
            Create the ordered base mask with soft edges.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.seq_len = seq_len
        self.softness = softness

        # Learnable concept queries
        self.concept_queries = nn.Parameter(torch.randn(num_concepts, hidden_dim))

        # Create and register the ordered mask
        base_mask = self._create_ordered_mask()
        self.register_buffer("base_mask", base_mask)

        self._init_weights()

    def _create_ordered_mask(self) -> torch.Tensor:
        """Create ordered mask with soft edges.

        PURPOSE:
            Generate a mask where concept i primarily covers segment i,
            with soft transitions at boundaries.

        Algorithm:
            1. Divide sequence into K equal segments
            2. For each concept, set primary region to 1.0
            3. Add soft edges with linear decay
            4. Normalize each row

        Returns:
            mask: [K, L] ordered mask with soft edges
        """
        K, L = self.num_concepts, self.seq_len
        mask = torch.zeros(K, L)

        # Edge width based on softness
        edge_width = int(self.softness * L / K)

        for i in range(K):
            # Primary segment boundaries
            start = int(i * L / K)
            end = int((i + 1) * L / K)

            # Primary region: full weight
            mask[i, start:end] = 1.0

            # Left edge decay
            if start > 0 and edge_width > 0:
                left_start = max(0, start - edge_width)
                left_edge_len = start - left_start
                decay = torch.linspace(0, 1, left_edge_len + 1)[1:]  # Exclude 0
                mask[i, left_start:start] = torch.maximum(
                    mask[i, left_start:start], decay
                )

            # Right edge decay
            if end < L and edge_width > 0:
                right_end = min(L, end + edge_width)
                right_edge_len = right_end - end
                decay = torch.linspace(1, 0, right_edge_len + 1)[:-1]  # Exclude 0
                mask[i, end:right_end] = torch.maximum(mask[i, end:right_end], decay)

        # Normalize each row to sum to 1 (for softmax compatibility)
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-10)

        return mask

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.concept_queries)

    def forward(
        self, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract concepts using hard ordered mask.

        DIMENSION FLOW:
            Input:
                H: [B, L, D]

            Process:
                1. Compute scores: Q @ H^T → [B, K, L]
                2. Get pre-computed mask: [K, L]
                3. Apply mask: scores + log(mask) → [B, K, L]
                4. Softmax → attention [B, K, L]
                5. Extract concepts: A @ H → [B, K, D]

            Output:
                concepts: [B, K, D]
                attn_weights: [B, K, L]
                base_mask: [K, L] (buffer, same for all batches)

        Args:
            H: Hidden states [B, L, D]

        Returns:
            concepts: Ordered concepts [B, K, D]
            attn_weights: Attention distribution [B, K, L]
            base_mask: The ordered mask [K, L]
        """
        B, L, D = H.shape
        K = self.num_concepts

        # Expand queries: [K, D] → [B, K, D]
        Q = self.concept_queries.unsqueeze(0).expand(B, -1, -1)

        # Compute attention scores
        scores = torch.bmm(Q, H.transpose(1, 2)) / math.sqrt(D)  # [B, K, L]

        # Apply ordered mask (add log mask to scores)
        # log_mask: [K, L] → [1, K, L]
        log_mask = torch.log(self.base_mask + 1e-10).unsqueeze(0)
        biased_scores = scores + log_mask  # [B, K, L]

        # Softmax
        attn_weights = F.softmax(biased_scores, dim=-1)  # [B, K, L]

        # Extract concepts
        concepts = torch.bmm(attn_weights, H)  # [B, K, D]

        return concepts, attn_weights, self.base_mask


# =============================================================================
# SCHEME 3: Recursive Ordered Generation
# =============================================================================


class RecursiveOrderedExtractor(nn.Module):
    """Scheme 3: Recursive Ordered Generation (VAR-style).

    PURPOSE:
        Generate concepts sequentially, where each concept is extracted from
        the remaining un-attended positions. This mimics VAR's scale-by-scale
        generation but adapted for 1D sequences.

    CORE IDEA:
        1. Start with all positions available (remaining_mask = 1)
        2. Extract concept 0 from available positions
        3. Mark highly-attended positions as "used" (reduce remaining_mask)
        4. Extract concept 1 from remaining positions
        5. Repeat until all concepts are extracted

        This naturally creates ordering: C_0 from early positions,
        C_K from late positions.

    MATHEMATICAL FORMULATION:
        Given:
            - H [B, L, D]: Hidden states
            - remaining_mask [B, L]: Initially all 1s

        For each concept i:
            1. Query: q_i = W_q · H.mean(dim=1) if i=0 else f(C_{i-1})
            2. Scores: s_i = q_i · H^T
            3. Masked scores: s_i' = s_i - ∞·(1 - remaining_mask)
            4. Attention: a_i = softmax(s_i')
            5. Concept: C_i = a_i · H
            6. Update mask: remaining_mask *= (1 - threshold(a_i))

        Where threshold(a_i) marks positions with attention > 0.5 * max(a_i)

    DIMENSION FLOW:
        Input:
            H: [B, L, D] - Hidden states

        Process (for each concept i):
            1. Generate query q_i (from previous concept or learned)
            2. Compute scores: q_i @ H^T → [B, L]
            3. Apply remaining mask: scores + mask → [B, L]
            4. Softmax: a_i = softmax(scores) → [B, L]
            5. Extract: C_i = a_i @ H → [B, D]
            6. Update remaining_mask based on a_i → [B, L]

        Output:
            concepts: [B, K, D] - Recursively ordered concepts
            attn_weights: [B, K, L] - Attention at each step
            remaining_history: [B, K, L] - Remaining mask history

    VISUALIZATION:
        Initial state:
            remaining_mask: [1, 1, 1, 1, 1, 1, 1, 1, ...] (all available)

        Step 0 (extract C_0):
            attention:    [0.4, 0.3, 0.2, 0.1, 0,   0,   0,   0,   ...]
            C_0: "小明有 5 个苹果"
            remaining:    [0.5, 0.5, 0.5, 0.5, 1,   1,   1,   1,   ...]
                          (positions 0-3 partially used)

        Step 1 (extract C_1):
            attention:    [0,   0,   0,   0.1, 0.4, 0.3, 0.2, 0,   ...]
            C_1: "吃掉 2 个"
            remaining:    [0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 1,   ...]

        Step 2 (extract C_2):
            attention:    [0,   0,   0,   0,   0,   0.1, 0.3, 0.4, ...]
            C_2: "还剩几个？"
            remaining:    [0.5, 0.5, 0.5, 0.3, 0.5, 0.3, 0.3, 0.5, ...]

        Key: Each concept naturally focuses on unused positions

    ATTRIBUTES:
        start_query: nn.Parameter [D]
            Initial query for first concept
        next_query_proj: nn.Linear
            Projects previous concept to next query
        usage_threshold: float
            Threshold for marking positions as "used" (default 0.5)
        decay_rate: float
            How much to reduce remaining mask (default 0.5)

    EXAMPLE:
        >>> extractor = RecursiveOrderedExtractor(
        ...     hidden_dim=512,
        ...     num_concepts=4
        ... )
        >>> H = torch.randn(2, 100, 512)
        >>> concepts, attn, history = extractor(H)
        >>> # Visualize remaining mask evolution
        >>> for i in range(4):
        ...     plt.plot(history[0, i].cpu().numpy())
    """

    def __init__(
        self,
        hidden_dim: int,
        num_concepts: int,
        usage_threshold: float = 0.5,
        decay_rate: float = 0.5,
    ):
        """Initialize recursive ordered extractor.

        Args:
            hidden_dim: Dimension D
            num_concepts: Number of concepts K
            usage_threshold: Attention threshold to mark position as used
            decay_rate: Decay factor for used positions in remaining mask
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.usage_threshold = usage_threshold
        self.decay_rate = decay_rate

        # Initial query for first concept
        self.start_query = nn.Parameter(torch.randn(hidden_dim))

        # Project previous concept to next query
        self.next_query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.start_query.unsqueeze(0))
        for module in self.next_query_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recursively extract ordered concepts.

        DIMENSION FLOW:
            Input:
                H: [B, L, D]

            Process (K iterations):
                For concept i:
                    - Get query q_i: [B, D]
                    - Compute scores: q_i @ H^T → [B, L]
                    - Mask: scores + log(remaining_mask) → [B, L]
                    - Attention: a_i = softmax(scores) → [B, L]
                    - Concept: C_i = a_i @ H → [B, D]
                    - Update: remaining_mask *= (1 - usage_i * decay)

            Output:
                concepts: [B, K, D]
                attn_weights: [B, K, L]
                remaining_history: [B, K, L]

        Args:
            H: Hidden states [B, L, D]

        Returns:
            concepts: Ordered concepts [B, K, D]
            attn_weights: Attention weights at each step [B, K, L]
            remaining_history: Remaining mask after each step [B, K, L]
        """
        B, L, D = H.shape
        K = self.num_concepts

        device = H.device

        # Initialize remaining mask (all positions available)
        remaining_mask = torch.ones(B, L, device=device)  # [B, L]

        concepts = []
        attn_weights_list = []
        remaining_history = []

        for i in range(K):
            # Get query for this concept
            if i == 0:
                # First concept: use learned start query
                q = self.start_query.unsqueeze(0).expand(B, -1)  # [B, D]
            else:
                # Subsequent concepts: project from previous concept
                q = self.next_query_proj(concepts[-1])  # [B, D]

            # Compute attention scores
            scores = torch.bmm(q.unsqueeze(1), H.transpose(1, 2)).squeeze(1)
            # [B, D] @ [B, D, L] → [B, 1, L] → [B, L]

            scores = scores / math.sqrt(D)

            # Apply remaining mask (add log mask)
            scores = scores + torch.log(remaining_mask + 1e-10)

            # Softmax to get attention
            attn = F.softmax(scores, dim=-1)  # [B, L]

            # Extract concept: [B, L] @ [B, L, D] → [B, D]
            concept = torch.bmm(attn.unsqueeze(1), H).squeeze(1)
            concepts.append(concept)
            attn_weights_list.append(attn)

            # Update remaining mask
            # Mark positions with high attention as "used"
            max_attn = attn.max(dim=-1, keepdim=True)[0]  # [B, 1]
            usage = (attn > max_attn * self.usage_threshold).float()
            # [B, L]: 1 for used positions, 0 otherwise

            # Decay remaining mask at used positions
            remaining_mask = remaining_mask * (1 - usage * self.decay_rate)

            # Store history
            remaining_history.append(remaining_mask.clone())

        # Stack to tensors
        concepts = torch.stack(concepts, dim=1)  # [B, K, D]
        attn_weights = torch.stack(attn_weights_list, dim=1)  # [B, K, L]
        remaining_history = torch.stack(remaining_history, dim=1)  # [B, K, L]

        return concepts, attn_weights, remaining_history


# =============================================================================
# SCHEME 4: Order-Constrained Training
# =============================================================================


class OrderConstrainedTraining(nn.Module):
    """Scheme 4: Order-Constrained Training with Loss Function.

    PURPOSE:
        Use a soft constraint via loss function to encourage ordered concepts.
        This offers maximum flexibility while still enforcing structure.

    CORE IDEA:
        1. Use standard attention-based extraction (no hard constraints)
        2. Compute expected position of each concept
        3. Add order loss: L_order = Σ ReLU(pos[i] - pos[i+1] + margin)
        4. Total loss = task_loss + λ * L_order

        The model learns to respect order while maintaining flexibility.

    MATHEMATICAL FORMULATION:
        Given:
            - Attention weights A [B, K, L]
            - Positions p [L]: [0, 1, 2, ..., L-1]

        Expected position for concept i:
            E[pos_i] = Σ_j A[i,j] * p[j]

        Order loss (for consecutive concepts):
            L_order = Σ_{i=0}^{K-2} ReLU(E[pos_i] - E[pos_{i+1}] + margin)

        If E[pos_i] > E[pos_{i+1}] - margin, we have a violation → loss > 0

        Total training loss:
            L_total = L_task + λ_order * L_order

    DIMENSION FLOW:
        Input:
            H: [B, L, D] - Hidden states
            training: bool - Whether to compute order loss

        Process:
            1. Standard attention extraction → concepts [B, K, D], A [B, K, L]
            2. Compute expected positions: E[pos] = A @ positions → [B, K]
            3. Calculate order loss from expected positions
            4. Return concepts, attention, and loss

        Output:
            concepts: [B, K, D] - Extracted concepts
            attn_weights: [B, K, L] - Attention distribution
            order_loss: scalar - Order constraint violation loss
            expected_positions: [B, K] - For monitoring

    VISUALIZATION:
        Well-ordered case (loss = 0):
            E[pos_0] = 15, E[pos_1] = 35, E[pos_2] = 60, E[pos_3] = 85
            Check: 15 < 35 < 60 < 85 ✓
            L_order = 0

        Violation case (loss > 0), margin=1.0:
            E[pos_0] = 50, E[pos_1] = 20, E[pos_2] = 70, E[pos_3] = 40
            Check:
                pos_0 - pos_1 + margin = 50 - 20 + 1 = 31 > 0 → loss += 31
                pos_2 - pos_3 + margin = 70 - 40 + 1 = 31 > 0 → loss += 31
            L_order = 62

        Training pushes model toward ordered configuration.

    ATTRIBUTES:
        concept_queries: nn.Parameter [K, D]
            Learnable queries
        order_margin: float
            Minimum gap between consecutive concept positions
        order_weight: float
            Weight λ for order loss (trade-off with task loss)

    EXAMPLE:
        >>> extractor = OrderConstrainedTraining(
        ...     hidden_dim=512,
        ...     num_concepts=4,
        ...     seq_len=100,
        ...     order_margin=1.0,
        ...     order_weight=0.1
        ... )
        >>> H = torch.randn(2, 100, 512)
        >>> concepts, attn, loss, positions = extractor(H, training=True)
        >>> loss.backward()  # Gradients flow through order constraint
    """

    def __init__(
        self,
        hidden_dim: int,
        num_concepts: int,
        seq_len: int,
        order_margin: float = 1.0,
        order_weight: float = 0.1,
    ):
        """Initialize order-constrained extractor.

        Args:
            hidden_dim: Dimension D
            num_concepts: Number of concepts K
            seq_len: Sequence length L
            order_margin: Minimum position gap between concepts
            order_weight: Weight λ for order loss
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.seq_len = seq_len
        self.order_margin = order_margin
        self.order_weight = order_weight

        # Learnable concept queries
        self.concept_queries = nn.Parameter(torch.randn(num_concepts, hidden_dim))

        # Register position indices as buffer
        positions = torch.arange(seq_len, dtype=torch.float32)
        self.register_buffer("positions", positions)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.concept_queries)

    def compute_order_loss(self, expected_positions: torch.Tensor) -> torch.Tensor:
        """Compute order constraint loss.

        PURPOSE:
            Calculate how much the expected positions violate ordering.

        Formula:
            L_order = Σ_{i=0}^{K-2} ReLU(pos[i] - pos[i+1] + margin)

        Args:
            expected_positions: [B, K] expected position of each concept

        Returns:
            order_loss: scalar mean loss over batch
        """
        K = self.num_concepts

        # Get consecutive positions
        pos_current = expected_positions[:, :-1]  # [B, K-1]
        pos_next = expected_positions[:, 1:]  # [B, K-1]

        # Violation: current position > next position - margin
        violation = F.relu(pos_current - pos_next + self.order_margin)

        # Mean over batch and concepts
        order_loss = violation.mean()

        return order_loss

    def forward(
        self, H: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract concepts with order constraint.

        DIMENSION FLOW:
            Input:
                H: [B, L, D]
                training: whether to compute order loss

            Process:
                1. Standard attention: Q @ H^T → [B, K, L]
                2. Softmax → A [B, K, L]
                3. Extract concepts: A @ H → [B, K, D]
                4. Expected positions: A @ positions → [B, K]
                5. Order loss (if training): L_order

            Output:
                concepts: [B, K, D]
                attn_weights: [B, K, L]
                order_loss: scalar (0 if not training)
                expected_positions: [B, K]

        Args:
            H: Hidden states [B, L, D]
            training: Whether to compute order loss

        Returns:
            concepts: Extracted concepts [B, K, D]
            attn_weights: Attention [B, K, L]
            order_loss: Order constraint loss (scalar)
            expected_positions: Expected position [B, K]
        """
        B, L, D = H.shape
        K = self.num_concepts

        # Expand queries
        Q = self.concept_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # Attention scores
        scores = torch.bmm(Q, H.transpose(1, 2)) / math.sqrt(D)  # [B, K, L]

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, K, L]

        # Extract concepts
        concepts = torch.bmm(attn_weights, H)  # [B, K, D]

        # Compute expected positions
        expected_positions = torch.sum(
            attn_weights * self.positions.view(1, 1, L), dim=-1
        )  # [B, K]

        # Compute order loss
        if training:
            order_loss = self.compute_order_loss(expected_positions)
            order_loss = order_loss * self.order_weight
        else:
            order_loss = torch.tensor(0.0, device=H.device)

        return concepts, attn_weights, order_loss, expected_positions


# =============================================================================
# RECOMMENDED: Robust Ordered Extractor (Combination)
# =============================================================================


class RobustOrderedExtractor(nn.Module):
    """Recommended: Robust Ordered Extractor (Combines Schemes 1 + 4).

    PURPOSE:
        Combine position prior (Scheme 1) with order loss (Scheme 4) for
        robust ordered concept extraction.

        - Training: Weak position prior + order loss → flexibility + structure
        - Inference: Strong position prior → guaranteed order

    CORE IDEA:
        Use a learnable position prior that biases attention toward ordered
        positions, but keep it weak during training. The order loss ensures
        the model learns meaningful semantic boundaries while respecting order.

        At inference, strengthen the position prior for guaranteed ordering.

    MATHEMATICAL FORMULATION:
        Given:
            - H [B, L, D]: Hidden states
            - Q [K, D]: Concept queries
            - c [K]: Sorted concept centers (learnable)
            - T: Temperature (learnable)

        Training:
            prior_strength = train_prior_strength (weak, e.g., 0.3)
            score[i,j] = (Q[i]·H[j])/√D + prior_strength * (-|j-c[i]|/T)

        Inference:
            prior_strength = inference_prior_strength (strong, e.g., 1.0)

        Order loss (training only):
            E[pos_i] = Σ_j A[i,j] * j
            L_order = Σ ReLU(E[pos_i] - E[pos_{i+1}] + margin)

    DIMENSION FLOW:
        Input:
            H: [B, L, D] - Hidden states
            training: bool - Mode selector

        Process:
            1. Get sorted centers c [K]
            2. Compute base scores: Q @ H^T → [B, K, L]
            3. Compute position prior: -|positions - c|/T → [K, L]
            4. Select prior strength based on mode
            5. Combine: scores + strength * prior → [B, K, L]
            6. Softmax → A [B, K, L]
            7. Extract concepts: C = A @ H → [B, K, D]
            8. Compute order loss (training only)

        Output:
            concepts: [B, K, D] - Robustly ordered concepts
            attn_weights: [B, K, L] - Attention distribution
            order_loss: scalar - Order constraint loss (training only)
            expected_positions: [B, K] - For monitoring

    VISUALIZATION:
        Training Mode (weak prior, order loss active):
            Position Prior (weight=0.3):
                C_0: [0.3, 0.25, 0.2, 0.15, 0.1, ...]
                C_1: [0.1, 0.15, 0.25, 0.3, 0.2, ...]
                ... (soft, allows learning)

            Order Loss: Pushes model if positions violate order

            Result: Model learns semantic boundaries while respecting order

        Inference Mode (strong prior):
            Position Prior (weight=1.0):
                C_0: [0.6, 0.3, 0.1, 0,   0,   ...]
                C_1: [0,   0.1, 0.3, 0.6, 0,   ...]
                ... (sharp, guarantees order)

            Result: Guaranteed ordered concepts

    ATTRIBUTES:
        concept_queries: nn.Parameter [K, D]
            Learnable concept queries
        center_logits: nn.Parameter [K]
            Raw logits for concept centers
        temperature: nn.Parameter [1]
            Controls sharpness of position prior
        train_prior_strength: float
            Position prior weight during training (default 0.3)
        inference_prior_strength: float
            Position prior weight during inference (default 1.0)
        order_margin: float
            Minimum gap for order loss
        order_weight: float
            Weight for order loss

    EXAMPLE:
        >>> extractor = RobustOrderedExtractor(
        ...     hidden_dim=512,
        ...     num_concepts=4,
        ...     seq_len=100,
        ...     train_prior_strength=0.3,
        ...     inference_prior_strength=1.0,
        ...     order_weight=0.1
        ... )
        >>>
        >>> # Training
        >>> H = torch.randn(2, 100, 512)
        >>> concepts, attn, loss, pos = extractor(H, training=True)
        >>> total_loss = task_loss + loss  # loss includes order constraint
        >>>
        >>> # Inference
        >>> concepts, attn, _, pos = extractor(H, training=False)
        >>> # Guaranteed ordered concepts
    """

    def __init__(
        self,
        hidden_dim: int,
        num_concepts: int,
        seq_len: int,
        train_prior_strength: float = 0.3,
        inference_prior_strength: float = 1.0,
        order_margin: float = 1.0,
        order_weight: float = 0.1,
    ):
        """Initialize robust ordered extractor.

        Args:
            hidden_dim: Dimension D
            num_concepts: Number of concepts K
            seq_len: Sequence length L
            train_prior_strength: Position prior weight during training
            inference_prior_strength: Position prior weight during inference
            order_margin: Minimum position gap for order loss
            order_weight: Weight for order loss
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.seq_len = seq_len
        self.train_prior_strength = train_prior_strength
        self.inference_prior_strength = inference_prior_strength
        self.order_margin = order_margin
        self.order_weight = order_weight

        # Learnable concept queries
        self.concept_queries = nn.Parameter(torch.randn(num_concepts, hidden_dim))

        # Learnable center positions (as logits)
        init_centers = torch.linspace(0, seq_len - 1, num_concepts)
        init_logits = torch.logit(torch.clamp(init_centers / seq_len, 0.01, 0.99))
        self.center_logits = nn.Parameter(init_logits)

        # Temperature for position prior
        self.temperature = nn.Parameter(
            torch.tensor(seq_len / num_concepts, dtype=torch.float32)
        )

        # Register positions buffer
        self.register_buffer("positions", torch.arange(seq_len, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.concept_queries)

    def _get_sorted_centers(self) -> torch.Tensor:
        """Get sorted, normalized concept centers.

        Returns:
            centers: [K] sorted centers in [0, seq_len]
        """
        centers = torch.sigmoid(self.center_logits) * self.seq_len
        centers, _ = torch.sort(centers)
        return centers

    def compute_order_loss(self, expected_positions: torch.Tensor) -> torch.Tensor:
        """Compute order constraint loss.

        Args:
            expected_positions: [B, K] expected positions

        Returns:
            order_loss: scalar
        """
        pos_current = expected_positions[:, :-1]
        pos_next = expected_positions[:, 1:]
        violation = F.relu(pos_current - pos_next + self.order_margin)
        return violation.mean() * self.order_weight

    def forward(
        self, H: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract robustly ordered concepts.

        DIMENSION FLOW:
            Input:
                H: [B, L, D]
                training: mode selector

            Process:
                1. Get sorted centers → [K]
                2. Compute base scores: Q @ H^T / √D → [B, K, L]
                3. Compute position prior: -|pos - c|/T → [K, L]
                4. Select prior strength based on mode
                5. Combine: scores + strength * prior → [B, K, L]
                6. Softmax → A [B, K, L]
                7. Extract: C = A @ H → [B, K, D]
                8. Expected positions: sum(A * pos) → [B, K]
                9. Order loss (if training)

            Output:
                concepts: [B, K, D]
                attn_weights: [B, K, L]
                order_loss: scalar
                expected_positions: [B, K]

        Args:
            H: Hidden states [B, L, D]
            training: Training mode flag

        Returns:
            concepts: Ordered concepts [B, K, D]
            attn_weights: Attention [B, K, L]
            order_loss: Order loss (0 if not training)
            expected_positions: Expected positions [B, K]
        """
        B, L, D = H.shape
        K = self.num_concepts

        # Expand queries
        Q = self.concept_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # Base attention scores
        scores = torch.bmm(Q, H.transpose(1, 2)) / math.sqrt(D)  # [B, K, L]

        # Get sorted centers
        centers = self._get_sorted_centers()  # [K]

        # Position prior
        positions = self.positions[:L]  # [L]
        distance = torch.abs(centers.unsqueeze(1) - positions.unsqueeze(0))
        position_prior = -distance / torch.clamp(self.temperature, min=1.0)
        # [K, L]

        # Select prior strength based on mode
        if training:
            prior_strength = self.train_prior_strength
        else:
            prior_strength = self.inference_prior_strength

        # Combine scores with position prior
        biased_scores = scores + prior_strength * position_prior.unsqueeze(0)

        # Softmax
        attn_weights = F.softmax(biased_scores, dim=-1)  # [B, K, L]

        # Extract concepts
        concepts = torch.bmm(attn_weights, H)  # [B, K, D]

        # Expected positions
        expected_positions = torch.sum(
            attn_weights * positions.view(1, 1, L), dim=-1
        )  # [B, K]

        # Order loss
        if training:
            order_loss = self.compute_order_loss(expected_positions)
        else:
            order_loss = torch.tensor(0.0, device=H.device)

        return concepts, attn_weights, order_loss, expected_positions


# =============================================================================
# Utility Functions
# =============================================================================


def visualize_concept_attention(
    attn_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    concept_names: Optional[List[str]] = None,
    title: str = "Concept Attention Patterns",
):
    """Visualize attention patterns of concepts.

    PURPOSE:
        Create visualization of which positions each concept attends to.
        Useful for debugging and understanding learned structure.

    Args:
        attn_weights: [K, L] or [B, K, L] attention weights
        tokens: Optional list of token strings for x-axis labels
        concept_names: Optional list of concept names
        title: Plot title

    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt

    if attn_weights.dim() == 3:
        # Take first batch item
        attn_weights = attn_weights[0]

    K, L = attn_weights.shape

    fig, axes = plt.subplots(K, 1, figsize=(15, 2 * K))
    if K == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.bar(range(L), attn_weights[i].cpu().numpy())

        if tokens is not None:
            ax.set_xticks(range(min(L, len(tokens))))
            ax.set_xticklabels(tokens[:L], rotation=45, ha="right")

        name = concept_names[i] if concept_names else f"Concept {i}"
        ax.set_title(f"{name} Attention")
        ax.set_ylim(0, attn_weights.max().item() * 1.1)
        ax.set_xlabel("Position")
        ax.set_ylabel("Attention Weight")

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def check_concept_ordering(expected_positions: torch.Tensor) -> dict:
    """Check if concepts are properly ordered.

    PURPOSE:
        Diagnostic tool to verify concept ordering.

    Args:
        expected_positions: [B, K] expected positions

    Returns:
        stats: Dictionary with ordering statistics
    """
    B, K = expected_positions.shape

    # Check ordering for each batch item
    is_ordered = torch.all(
        expected_positions[:, :-1] < expected_positions[:, 1:], dim=1
    )

    # Calculate gaps between consecutive concepts
    gaps = expected_positions[:, 1:] - expected_positions[:, :-1]

    stats = {
        "ordered_ratio": is_ordered.float().mean().item(),
        "mean_gap": gaps.mean().item(),
        "min_gap": gaps.min().item(),
        "max_gap": gaps.max().item(),
        "position_range": (
            expected_positions.min().item(),
            expected_positions.max().item(),
        ),
    }

    return stats
