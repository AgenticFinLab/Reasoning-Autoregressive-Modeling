"""NLCP (Next-Level Concept Pyramid) Base Configurations.

This module defines configuration dataclasses for NLCP architecture.
Reference: concept-pyramid.md Section 3.1 - Basic Configuration and Tensor Conventions
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class NLCPModelConfig:
    """Configuration for NLCP Model.

    Reference: concept-pyramid.md Section 3.1
    Basic Configuration and Tensor Conventions table.

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
