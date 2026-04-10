"""NLCP (Next-Level Concept Pyramid) Base Configurations.

This module defines configuration dataclasses for NLCP architecture.

DESIGN SOURCE:
    - concept-pyramid.md Section 3.1 - Basic Configuration and Tensor Conventions
    - concept-pyramid-critic.md - Critical analysis of design choices

CONFIGURATION PARAMETERS (Section 3.1 Table):
    ┌─────────────────────┬────────────────────────────────┬─────────────┐
    │ Symbol              │ Meaning                        │ Default     │
    ├─────────────────────┼────────────────────────────────┼─────────────┤
    │ d                   │ Hidden dimension               │ 1024        │
    │ H                   │ Attention heads                │ 16          │
    │ L_0                 │ Level 0 length                 │ 8           │
    │ L_k                 │ Dynamic length                 │ [4, 512]    │
    │ K_max               │ Maximum pyramid depth          │ 4           │
    │ τ                   │ Depth gate threshold           │ 0.35~0.45   │
    │ R_target            │ Target expansion ratio         │ 3~5         │
    │ λ_k                 │ Expansion rate per position    │ [1, 8]      │
    └─────────────────────┴────────────────────────────────┴─────────────┘

CRITICAL CONSIDERATIONS (from concept-pyramid-critic.md):

    ISSUE 5 - Scaling Law Validation:
        Current configuration uses fixed d=1024 for all levels.
        No empirical validation that this is optimal.

        Recommendation: Experiment with heterogeneous widths per level
        (see critic Improvement 3: Dynamic Width per Level)

        Example configuration:
            Level 0: d_0 = 512   (coarse concepts need less detail)
            Level 1: d_1 = 768   (intermediate)
            Level 2: d_2 = 1024  (fine details need more capacity)
            Level 3: d_3 = 1024

        With μP learning rate scaling:
            η_k = η_base * (d_k / d_base)^{-1}
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class NLCPModelConfig:
    """Configuration for NLCP Model.

    DESIGN SOURCE - concept-pyramid.md Section 3.1:
        Basic Configuration and Tensor Conventions table.
        All parameters directly map to symbols in the design document.

    PARAMETER DETAILS:
        hidden_dim (d = 1024):
            Hidden dimension shared across all levels.
            Standard choice for modern LLMs (LLaMA, Mistral).
            d_head = d / H = 1024 / 16 = 64 dimensions per head.

        num_heads (H = 16):
            Number of parallel attention heads.
            Enables multi-head diversity in representation.

        vocab_size (V = 128000):
            Aligns with mainstream base models.
            Supports multilingual and code tokens.

        max_depth (K_max = 4):
            Maximum pyramid depth from Section 7.3 (risk mitigation).
            Prevents excessive expansion on complex inputs.
            Typical reasoning needs 2-3 levels.

        depth_gate_threshold (τ = 0.4):
            From Section 3.1, recommended range 0.35~0.45.
            Balances depth vs computational efficiency.
            Lower τ → deeper pyramids, more computation.
            Higher τ → shallower pyramids, faster inference.

        l0_length (L_0 = 8):
            Level 0 initial length for macro intent abstraction.
            Sufficient to capture problem structure.

        l_max (512):
            Maximum sequence length per level from Section 3.1.
            Range L_k ∈ [4, 512] for safety.

        dropout (0.1):
            Standard regularization.
            Applied in attention and MLP layers.

        expansion_min (λ_min = 1):
            Minimum expansion rate from Section 3.3.
            Every coarse position gets at least 1 fine slot.

        expansion_max (λ_max = 8):
            Maximum expansion rate from Section 3.3.
            Prevents excessive expansion of single positions.

    CRITICAL CONSIDERATIONS (from concept-pyramid-critic.md):
        Heterogeneous Widths (Improvement 3):
            Current implementation uses fixed hidden_dim for all levels.
            Consider experimenting with level-specific dimensions:

            Example heterogeneous config:
                level_dims = [512, 768, 1024, 1024]  # Per level

            Rationale:
                - Level 0 (coarse): Concepts are abstract, need less capacity
                - Level 2+ (fine): Details are complex, need more capacity

            Implementation requires:
                1. Projection layers between levels
                2. μP learning rate scaling per level
                3. Careful initialization

    Attributes:
        hidden_dim: Hidden dimension d, shared across all levels.
            Default: 1024 (from Section 3.1 table)
        num_heads: Number of attention heads H.
            Default: 16 (d_head = d/H = 64)
        vocab_size: Vocabulary size V.
            Default: 128000 (align with mainstream base models)
        max_depth: Maximum pyramid depth K.
            Default: 4 (from Section 7.3 risk mitigation)
        depth_gate_threshold: Depth gate threshold τ.
            Default: 0.4 (from Section 3.1, range 0.35~0.45)
        l0_length: Level 0 initial length L_0.
            Default: 8 (macro intent abstraction)
        l_max: Maximum sequence length per level.
            Default: 512 (from Section 3.1 L_k range [4, 512])
        dropout: Dropout rate for regularization.
        expansion_min: Minimum expansion rate.
            Default: 1 (no compression below this)
        expansion_max: Maximum expansion rate.
            Default: 8 (prevent explosion)

        # Component Selection (from concept-pyramid-critic.md solutions)
        expansion_predictor_type: Type of expansion predictor to use.
            Options: "floor" (original), "gumbel" (Solution 1A, recommended),
                     "reinforce" (Solution 1B), "soft" (Solution 1C)
            Default: "gumbel"
        depth_gate_type: Type of depth gate to use.
            Options: "standard" (original, non-causal), "causal" (Solution 3B)
            Default: "causal"
        cross_attention_type: Type of cross-level attention to use.
            Options: "standard" (original, rigid), "relaxed" (Solution 4A),
                     "hybrid" (Solution 4B)
            Default: "relaxed"
        consistency_loss_type: Type of consistency loss to use.
            Options: "standard" (original, strict L2), "directional" (Solution 2A),
                     "residual" (Solution 2B), "mi" (Solution 2C)
            Default: "directional"
    """

    hidden_dim: int
    num_heads: int
    vocab_size: int
    max_depth: int
    depth_gate_threshold: float
    l0_length: int
    l_max: int
    dropout: float
    expansion_min: int
    expansion_max: int

    # Component selection
    expansion_predictor_type: str = "gumbel"  # "floor", "gumbel", "reinforce", "soft"
    depth_gate_type: str = "causal"  # "standard", "causal"
    cross_attention_type: str = "relaxed"  # "standard", "relaxed", "hybrid"
    consistency_loss_type: str = (
        "directional"  # "standard", "directional", "residual", "mi"
    )


@dataclass
class NLCPTrainingConfig:
    """Configuration for NLCP Training.

    Reference: concept-pyramid.md Section 4
    Pretraining Strategy and Objective Functions.

    Attributes:
        lambda_consist: Weight for cross-scale consistency loss.
            Default: 0.1 (from Section 4.1)
        lambda_depth: Weight for expansion rate regularization loss.
            Default: 0.05 (from Section 4.1)
        lambda_ce: Weight for final token alignment loss.
            Default: 1.0 (from Section 4.1)
        target_expansion_ratio: Target expansion ratio R_target.
            Default: 4.0 (from Section 3.3, range [3, 5])
        learning_rate: Base learning rate η_base.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        grad_clip_norm: Gradient clipping norm.
            Default: 1.0 (from Section 7.2)
        muP_scale: Output scaling factor for μP.
            Reference: DLCM Eq.21

        # Component-specific hyperparameters (from concept-pyramid-critic.md)
        gumbel_temperature: Temperature for Gumbel-Softmax (Solution 1A).
            Default: 0.5 (lower = more discrete)
        gumbel_hard: Whether to use straight-through estimator.
            Default: True
        reinforce_baseline_weight: Weight for baseline loss in REINFORCE.
            Default: 0.5
        directional_epsilon: Epsilon for DirectionalConsistencyLoss (Solution 2A).
            Default: 0.5 (allow deviation up to 0.5 in L2 norm)
        mi_temperature: Temperature for MutualInformationConsistency (Solution 2C).
            Default: 0.07
    """

    lambda_consist: float
    lambda_depth: float
    lambda_ce: float
    target_expansion_ratio: float
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    grad_clip_norm: float
    muP_scale: float

    # Component-specific hyperparameters
    gumbel_temperature: float = 0.5
    gumbel_hard: bool = True
    reinforce_baseline_weight: float = 0.5
    directional_epsilon: float = 0.5
    mi_temperature: float = 0.07


@dataclass
class NLCPInferenceConfig:
    """Configuration for NLCP Inference.

    Reference: concept-pyramid.md Section 5
    Inference Pipeline and Causal Guarantees.

    Attributes:
        max_depth: Maximum generation depth.
        depth_threshold: Threshold τ for depth gate.
        temperature: Sampling temperature for expansion predictor.
            Default: 0.5 (from Section 7.2)
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        early_exit: Whether to enable early exit.
            Reference: Section 5.3
    """

    max_depth: int
    depth_threshold: float
    temperature: float
    top_k: int
    top_p: float
    early_exit: bool


@dataclass
class LevelState:
    """State container for a single pyramid level.

    Reference: concept-pyramid.md Section 2.2
    Module Tasks and Connection Logic.

    This dataclass holds the intermediate representations
    at each level of the pyramid during forward pass.

    Attributes:
        hidden_states: Hidden representations H_k.
            Shape: [batch_size, L_k, hidden_dim]
        length: Sequence length L_k for this level.
        expand_mask: Expansion mask for transitioning to next level.
            Shape: [batch_size, L_k] containing expansion counts
        depth_gate_prob: Probability from depth gate.
            Scalar: p_cont^(k) ∈ [0, 1]
        kv_cache_self: KV cache for self-attention at this level.
    """

    hidden_states: object
    length: int
    expand_mask: object
    depth_gate_prob: float
    kv_cache_self: object


@dataclass
class NLCPOutput:
    """Output container for NLCP forward pass.

    Reference: concept-pyramid.md Section 4.1
    Complete Loss Function.

    Attributes:
        logits: Final vocabulary logits.
            Shape: [batch_size, L_final, vocab_size]
        level_states: List of LevelState for each level.
        total_loss: Combined loss value.
        ntp_loss: Next token prediction loss sum.
        consist_loss: Cross-scale consistency loss.
        depth_loss: Expansion rate regularization loss.
        ce_loss: Final token alignment loss.
    """

    logits: object
    level_states: object
    total_loss: float
    ntp_loss: float
    consist_loss: float
    depth_loss: float
    ce_loss: float
