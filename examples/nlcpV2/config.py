"""NLCP V2 Configuration.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2.1: Framework Overview
    - Section 2.2: Module Detailed Analysis

PURPOSE:
    Centralized configuration for all NLCP V2 hyperparameters.
    Ensures consistency across Encoder, Attentive Pooling, Concept Transformer,
    and Token Decoder modules.

CONFIGURATION PARAMETERS TABLE:
    Symbol    | Meaning                    | Example    | Notes
    ──────────┼───────────────────────────┼────────────┼─────────────────
    d         | Hidden dimension           | 1024       | Shared across levels
    H         | Attention heads            | 16         | d_head = d/H = 64
    L_k       | Level k concept count      | [4,16,64]  | L_k << L (token count)
    K         | Number of levels           | 4          | Fixed for inference
    V         | Vocabulary size            | 128,000    | Standard LLM vocab
    α         | Concept loss weight        | 0.1        | L_total = L_NTP + α*L_concept
    β         | Reconstruction loss weight | 0.05       | + β*L_recon
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NLCPV2Config:
    """Configuration for NLCP V2 Model.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.1

    ARCHITECTURE OVERVIEW:
        NLCP V2 consists of four core modules:
        1. Encoder: Q+CoT → H (token-level representations)
        2. Attentive Pooling: H → C_0, C_1, ..., C_K (hierarchical concepts)
        3. Concept Transformer: H_0 → H_1 → ... → H_K (next-level generation)
        4. Token Decoder: H_K → tokens (cross-attention to concepts)

    DIMENSION FLOW:
        Input: [B, L] token IDs
            ↓
        Encoder: [B, L, D_encoder] (D_encoder from pretrained model, e.g., 896)
            ↓
        Projection: [B, L, D] (D = hidden_dim, e.g., 256)
            ↓
        Attentive Pooling: List of [B, L_k, D] for k = 0, 1, ..., K-1
            ↓
        Concept Transformer: List of [B, L_k, D] (predicted concepts)
            ↓
        Token Decoder: [B, L_K, V] (vocabulary logits)

    Attributes:
        hidden_dim: Hidden dimension D shared across all concept levels
        num_heads: Number of attention heads H for multi-head attention
        vocab_size: Vocabulary size V for token embedding and output projection
        num_levels: Number of pyramid levels K (fixed for inference)
        level_lengths: List of concept counts [L_0, L_1, ..., L_{K-1}]
            Constraint: L_k << L (token count) for semantic compression
            Constraint: L_0 < L_1 < ... < L_{K-1} (monotonic expansion)
        max_seq_len: Maximum sequence length for training (typically 512 or 1024)
        dropout: Dropout rate for regularization (typically 0.1)
        rms_norm_eps: Epsilon for RMSNorm numerical stability (typically 1e-6)
        encoder_model_name: HuggingFace model name for encoder (e.g., "Qwen/Qwen2.5-0.5B")
        encoder_num_layers: Number of layers in encoder (None for all layers)
        encoder_freeze: Whether to freeze encoder weights during training
        ntp_loss_weight: Weight λ for Next Token Prediction loss (typically 1.0)
        concept_loss_weight: Weight α for concept prediction loss (typically 0.1)
        recon_loss_weight: Weight β for reconstruction loss (typically 0.05)
        muP_scale: Scaling factor for muP (maximal update parameterization)
    """

    hidden_dim: int
    num_heads: int
    vocab_size: int
    num_levels: int
    level_lengths: List[int]
    max_seq_len: int
    dropout: float
    rms_norm_eps: float
    encoder_model_name: str
    encoder_num_layers: Optional[int]
    encoder_freeze: bool
    ntp_loss_weight: float
    concept_loss_weight: float
    recon_loss_weight: float
    muP_scale: float

    def __post_init__(self):
        """Validate configuration parameters.

        VALIDATION RULES:
            1. num_levels must match len(level_lengths)
            2. hidden_dim must be divisible by num_heads (for multi-head attention)
            3. All level_lengths must be positive
            4. level_lengths must be strictly increasing (L_k < L_{k+1})
        """
        if self.num_levels != len(self.level_lengths):
            raise ValueError(
                f"num_levels ({self.num_levels}) must match "
                f"len(level_lengths) ({len(self.level_lengths)})"
            )

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        for i, length in enumerate(self.level_lengths):
            if length <= 0:
                raise ValueError(f"level_lengths[{i}] must be positive, got {length}")

        for i in range(1, len(self.level_lengths)):
            if self.level_lengths[i] <= self.level_lengths[i - 1]:
                raise ValueError(
                    f"level_lengths must be strictly increasing: "
                    f"{self.level_lengths[i-1]} >= {self.level_lengths[i]}"
                )

    @property
    def head_dim(self) -> int:
        """Calculate per-head dimension for multi-head attention.

        FORMULA:
            d_head = hidden_dim / num_heads

        EXAMPLE:
            hidden_dim = 1024, num_heads = 16
            d_head = 1024 / 16 = 64

        Returns:
            Per-head dimension d_head
        """
        return self.hidden_dim // self.num_heads

    @property
    def max_concepts(self) -> int:
        """Get maximum concept count across all levels.

        PURPOSE:
            Used for memory allocation and buffer sizing in Concept Transformer.

        Returns:
            max(L_0, L_1, ..., L_{K-1})
        """
        return max(self.level_lengths)
