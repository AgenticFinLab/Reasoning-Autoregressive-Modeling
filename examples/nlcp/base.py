"""NLCP (Next-Level Concept Pyramid) Base Configurations.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V1.md
    - Section 3.1: Basic Configuration and Tensor Conventions
    - Section 2.2: Module Tasks and Connection Logic
    - Section 4.1: Complete Loss Function and Training Data Format

CONFIGURATION PARAMETERS TABLE (Section 3.1):
    Symbol    | Meaning                    | Default    | Notes
    ──────────┼───────────────────────────┼────────────┼─────────────────
    d         | Hidden dimension           | 1024       | Shared across levels
    H         | Attention heads            | 16         | d_head = d/H = 64
    L_q       | Question encoding length   | 64         | Fixed padding
    L_0       | Level 0 length             | 8          | Macro intent abstraction
    L_k       | Level k length             | [4, 512]   | Dynamic via expand_mask
    K_max     | Maximum pyramid depth      | 4          | From Section 7.3
    τ         | Depth gate threshold       | 0.35~0.45  | Inference adjustment
    V         | Vocabulary size            | 128,000    | Mainstream base models
    R_target  | Target expansion ratio     | [3, 5]     | Global regularizer
    λ_min     | Min expansion rate         | 1          | Every position gets >= 1 slot
    λ_max     | Max expansion rate         | 8          | Prevent explosion

KEY INSIGHT FROM V1 (Section 1.2-1.4):
    NLCP differs from VAR and DLCM in how it ensures "layer-wise approximation to CoT":

    - VAR: Uses f_rest (residual) to tell model "what to encode" at each scale
    - DLCM: Concept = Token Pool, naturally contains reconstruction information
    - NLCP: Uses implicit learning via gradient backprop + consistency constraints

    This is the core design choice that shapes all components below.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class NLCPModelConfig:
    """Configuration for NLCP Model Architecture.

    DESIGN SOURCE: concept-pyramid-V1.md Section 3.1

    This configuration directly maps to symbols in the design document.
    All parameters are required (no defaults in code per llm-coding-rules.md).

    PARAMETER EXPLANATIONS:

    hidden_dim (d):
        Hidden dimension shared across all pyramid levels.
        Standard choice for modern LLMs (LLaMA-7B uses 4096, Mistral-7B uses 4096).
        With num_heads=16, each head operates on d_head = 1024/16 = 64 dimensions.

    num_heads (H):
        Number of parallel attention heads for multi-head attention.
        Enables representation diversity across different semantic subspaces.

    vocab_size (V):
        Vocabulary size for token embedding and output projection.
        Aligns with mainstream base models (Llama-3 uses 128K tokens).

    max_depth (K_max):
        Maximum pyramid depth K from Section 7.3 (risk mitigation).
        Prevents excessive expansion on complex inputs.
        Typical reasoning tasks need 2-3 levels; max 4 provides safety margin.

    depth_gate_threshold (tau):
        Threshold for depth continuation decision.
        Range 0.35~0.45 from Section 3.1.
        Lower tau -> deeper pyramids (more computation).
        Higher tau -> shallower pyramids (faster inference).

    l0_length (L_0):
        Level 0 initial sequence length.
        Represents "macro intent abstraction" - the problem's overall structure.
        Typically 4-8 concepts sufficient for most problems.

    l_max (L_k max):
        Maximum sequence length per level.
        Range L_k in [4, 512] from Section 3.1.
        Prevents memory issues from unbounded expansion.

    dropout:
        Dropout rate for regularization.
        Applied in attention layers and MLP blocks.

    expansion_min (lambda_min):
        Minimum expansion rate per position.
        From Section 3.3: every coarse position gets at least 1 fine slot.
        Value 1 ensures no information loss during expansion.

    expansion_max (lambda_max):
        Maximum expansion rate per position.
        From Section 3.3: prevents single position from dominating.
        Value 8 provides reasonable bound for complex reasoning steps.

    depth_gate_type:
        Type of depth gate to use.
        Options: "standard" (original), "causal" (critic.md Solution 3B).
        Causal version fixes train/test mismatch in pooling.

    expansion_predictor_type:
        Type of expansion predictor to use.
        Options: "floor" (original), "gumbel" (Solution 1A),
                 "reinforce" (Solution 1B), "soft" (Solution 1C).
        Gumbel-Softmax is recommended for differentiable training.

    cross_attention_type:
        Type of cross-level attention to use.
        Options: "standard" (original), "relaxed" (Solution 4A),
                 "hybrid" (Solution 4B).
        Relaxed allows fine positions to attend to multiple coarse parents.

    consistency_loss_type:
        Type of consistency loss to use in training.
        Options: "standard" (original L2), "directional" (Solution 2A),
                 "residual" (Solution 2B), "mi" (Solution 2C).
        Directional allows epsilon deviation for new information.

    Attributes:
        hidden_dim: Hidden dimension d (Section 3.1 default: 1024).
        num_heads: Number of attention heads H (Section 3.1 default: 16).
        vocab_size: Vocabulary size V (Section 3.1 default: 128000).
        max_depth: Maximum pyramid depth K_max (Section 7.3 default: 4).
        depth_gate_threshold: Depth gate threshold tau (Section 3.1 range: 0.35-0.45).
        l0_length: Level 0 length L_0 (Section 3.1 default: 8).
        l_max: Maximum sequence length L_k max (Section 3.1 range: 4-512).
        dropout: Dropout rate for regularization.
        expansion_min: Minimum expansion rate lambda_min.
        expansion_max: Maximum expansion rate lambda_max.
        depth_gate_type: Type of depth gate ("standard" or "causal").
        expansion_predictor_type: Type of expansion predictor ("floor", "gumbel", "reinforce", "soft").
        cross_attention_type: Type of cross-level attention ("standard", "relaxed", "hybrid").
        consistency_loss_type: Type of consistency loss ("standard", "directional", "residual", "mi").
        encoder_model_name: HuggingFace model name for encoder (e.g., "Qwen/Qwen2.5-0.5B").
        encoder_num_layers: Number of encoder layers to use (None = use all).
        encoder_freeze: Whether to freeze encoder weights.
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
    depth_gate_type: str
    expansion_predictor_type: str
    cross_attention_type: str
    consistency_loss_type: str
    encoder_model_name: str
    encoder_num_layers: Optional[int]
    encoder_freeze: bool


@dataclass
class NLCPTrainingConfig:
    """Configuration for NLCP Training.

    DESIGN SOURCE: concept-pyramid-V1.md Section 4
    Pretraining Strategy and Objective Functions.

    LOSS FUNCTION (Section 4.1.2):
        L_total = L_NTP(H_K → C) + λ₁·L_consist + λ₂·L_depth

        Where:
        - L_NTP: Next-token prediction at final level (ONLY level with text supervision)
        - L_consist: Cross-level consistency (||MeanPool(H_{k+1}) - H_k||²)
        - L_depth: Expansion rate regularization ((L_{k+1}/L_k - R_target)²)

    KEY INSIGHT (Section 4.1.2 "Why Intermediate Layers Have No Text Supervision"):
        Unlike VAR (f_rest provides per-layer supervision) or DLCM (Concept = Token Pool),
        NLCP intermediate layers H_0, H_1, ..., H_{K-1} have NO direct text supervision.

        They are shaped through:
        1. Gradient backpropagation: L_NTP → ∂L/∂H_K → ... → ∂L/∂H_0
        2. Consistency constraints: Provide "pseudo-residual" signal
        3. Conditional generation dependency: H_{k+1} depends on H_k via Cross-Attn

    MU P SCALING (Section 4.2):
        For heterogeneous width per level:
        η_k = η_base · (d_k / d_base)^{-1}

        Output scaling: logits = (H_K @ W_unemb^T) / s_μP

    Attributes:
        lambda_consist: Weight for consistency loss L_consist.
            From Section 4.1, default 0.1.
        lambda_depth: Weight for depth regularization L_depth.
            From Section 4.1, default 0.05.
        lambda_ce: Weight for final cross-entropy alignment.
            From Section 4.1, default 1.0.
        target_expansion_ratio: Target expansion ratio R_target.
            From Section 3.3, range [3, 5].
        learning_rate: Base learning rate η_base.
        weight_decay: Weight decay for AdamW optimizer.
        warmup_steps: Number of warmup steps for LR scheduler.
        max_steps: Total training steps.
        grad_clip_norm: Gradient clipping norm.
            From Section 7.2, default 1.0.
        muP_scale: Output scaling factor for μP.
            Reference: DLCM Eq.21.
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

    DESIGN SOURCE: concept-pyramid-V1.md Section 5
    Inference Pipeline and Causal Guarantees.

    INFERENCE FLOW (Section 5.1):
        1. Encode Q → H_0
        2. Loop: depth_gate → expansion_predictor → next_level_generator
        3. Token decode: H_K → logits → autoregressive_decode

    EARLY EXIT (Section 5.4.1):
        If p_cont < τ at any level, terminate early.
        Simple problems (e.g., "2+2=?") can exit at Level 0 or 1.
        Complex problems may need all K_max levels.

    Attributes:
        max_depth: Maximum generation depth K_max.
        depth_threshold: Threshold τ for depth gate decisions.
        temperature: Sampling temperature for token generation.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        early_exit: Whether to enable early exit optimization.
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

    DESIGN SOURCE: concept-pyramid-V1.md Section 2.2
    Module Tasks and Connection Logic.

    During forward pass, this dataclass holds intermediate representations
    at each level of the pyramid. Used for:
    1. Loss computation (consistency between adjacent levels)
    2. Analysis and visualization
    3. KV cache management

    TENSOR SHAPES:
        hidden_states: [batch_size, L_k, hidden_dim]
        expand_mask: [batch_size, L_k] - expansion count per position
        depth_gate_prob: scalar - p_cont^(k) ∈ [0, 1]

    Attributes:
        hidden_states: Hidden representations H_k.
            Shape: [B, L_k, D] where D = hidden_dim.
        length: Sequence length L_k for this level.
        expand_mask: Expansion mask for transitioning to next level.
            Shape: [B, L_k], each entry is expansion count for that position.
            Value 0 means this position won't be expanded.
        depth_gate_prob: Continuation probability from depth gate.
            Scalar probability ∈ [0, 1].
        kv_cache_self: Optional KV cache for self-attention.
            Tuple of (K_cache, V_cache) for efficient autoregressive generation.
    """

    hidden_states: torch.Tensor
    length: int
    expand_mask: Optional[torch.Tensor]
    depth_gate_prob: float
    kv_cache_self: Optional[tuple]


@dataclass
class NLCPOutput:
    """Output container for NLCP forward pass.

    DESIGN SOURCE: concept-pyramid-V1.md Section 4.1
    Complete Loss Function.

    LOSS COMPONENTS:
        L_total = L_NTP + λ₁·L_consist + λ₂·L_depth + λ₃·L_CE

        From Section 4.1.2:
        - L_NTP: Only final layer H_K has text supervision
        - L_consist: ||MeanPool(H_{k+1}) - H_k||² enforces refinement
        - L_depth: Regularizes expansion rates toward R_target
        - L_CE: Final cross-entropy with target CoT

    Attributes:
        logits: Final vocabulary logits.
            Shape: [B, L_final, V] where V = vocab_size.
        level_states: List of LevelState for each pyramid level.
            Length equals actual depth used (may be less than max_depth).
        total_loss: Combined weighted loss value.
        ntp_loss: Next-token prediction loss (only H_K has this).
        consist_loss: Cross-scale consistency loss sum over all transitions.
        depth_loss: Expansion rate regularization loss.
        ce_loss: Final cross-entropy alignment loss.
    """

    logits: torch.Tensor
    level_states: List[LevelState]
    total_loss: float
    ntp_loss: float
    consist_loss: float
    depth_loss: float
    ce_loss: float
