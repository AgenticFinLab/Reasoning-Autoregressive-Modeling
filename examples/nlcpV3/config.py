"""NLCP V3 Configuration.

USAGE:
    from nlcpV3.config import NLCPV3Config

    config = NLCPV3Config(
        hidden_dim=256,
        num_levels=6,
        level_lengths=[1, 2, 4, 8, 16, 32]
    )

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2.2: Architecture Components
    - Section 3: Training Configuration

PURPOSE:
    Centralized configuration for NLCP V3 model. All hyperparameters are
    explicitly defined here with no default values (per llm-coding-rules.md).

DIMENSION FLOW:
    Configuration parameters flow through the architecture:
    - hidden_dim: Concept dimension (D) used across all modules
    - num_levels: Number of hierarchical levels (K)
    - level_lengths: Concepts per level [1, 2, 4, ..., 2^(K-1)]
    - encoder_model_name: Pretrained encoder for Q+CoT/Q encoding
"""

from dataclasses import dataclass
from typing import List


@dataclass
class NLCPV3Config:
    """Configuration for NLCP V3 model.

    PURPOSE:
        Define all hyperparameters for NLCP V3 architecture. No default
        values are provided to ensure explicit configuration.

    ATTRIBUTES:
        Pyramid Configuration:
            hidden_dim: Dimension of concept vectors (D)
            num_heads: Number of attention heads for multi-head attention
            num_levels: Number of hierarchical levels (K)
            level_lengths: List of concept counts per level [L_0, L_1, ..., L_{K-1}]
            max_seq_len: Maximum sequence length for encoder

        Encoder Configuration:
            encoder_model_name: HuggingFace model name for encoder
            encoder_num_layers: Number of layers in encoder (if customizing)
            encoder_freeze: Whether to freeze encoder parameters

        Decoder Configuration:
            vocab_size: Vocabulary size for solution token prediction
            dropout: Dropout rate for regularization
            rms_norm_eps: Epsilon for RMS normalization
            muP_scale: Scaling factor for muP (maximal update parameterization)

        Loss Weights:
            ntp_loss_weight: Weight for next-token prediction loss
            concept_loss_weight: Weight for concept extraction/generation loss
            recon_loss_weight: Weight for reconstruction loss
    """

    hidden_dim: int
    num_heads: int
    num_levels: int
    level_lengths: List[int]
    max_seq_len: int

    encoder_model_name: str
    encoder_num_layers: int
    encoder_freeze: bool

    vocab_size: int
    dropout: float
    rms_norm_eps: float
    muP_scale: float

    ntp_loss_weight: float
    concept_loss_weight: float
    recon_loss_weight: float

    def __post_init__(self):
        """Validate configuration parameters.

        PURPOSE:
            Ensure configuration values are valid and consistent.

        VALIDATION:
            - hidden_dim must be divisible by num_heads
            - len(level_lengths) must equal num_levels
            - level_lengths should follow doubling pattern
        """
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if len(self.level_lengths) != self.num_levels:
            raise ValueError(
                f"len(level_lengths) ({len(self.level_lengths)}) must equal "
                f"num_levels ({self.num_levels})"
            )

    @property
    def head_dim(self) -> int:
        """Calculate dimension per attention head.

        PURPOSE:
            Compute head dimension for multi-head attention.

        MATHEMATICAL FORMULATION:
            head_dim = hidden_dim / num_heads

        Returns:
            Dimension of each attention head
        """
        return self.hidden_dim // self.num_heads

    @property
    def total_concepts(self) -> int:
        """Calculate total number of concepts across all levels.

        PURPOSE:
            Compute total concept count for buffer allocation.

        MATHEMATICAL FORMULATION:
            total = sum(level_lengths)

        Returns:
            Total number of concepts
        """
        return sum(self.level_lengths)
