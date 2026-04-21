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
    - reason_model_name: Pretrained decoder-only model for CoT extraction & Solution generation
    - use_positional_query_init: Whether to init concept queries with positional priors
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


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

        Reason Model Configuration:
            reason_model_name: HuggingFace model name for the decoder-only
                Transformer used as both CoT feature extractor and Solution
                generator. Builder uses AutoModel backbone; Predictor uses
                lm_head for autoregressive generation.
            reason_model_num_layers: Number of layers to use (-1 = all)
            reason_model_freeze: Whether to freeze backbone parameters
            reason_model_lora: Optional LoRA config dict for reason_model.
                If non-None, applies PEFT LoRA adapters to the backbone.
                Example: {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}

        Solution Decoder Configuration:
            decoder_model_name: HuggingFace model name for the solution decoder.
                This is the SAME base model as reason_model but loaded with
                AutoModelForCausalLM (includes lm_head). Used for Solution
                generation from Q + concept pyramid. Can differ from
                reason_model_name for model distillation scenarios.
            decoder_freeze: Whether to freeze the decoder backbone parameters
            decoder_lora: Optional LoRA config dict for the solution decoder.
                If non-None, applies PEFT LoRA adapters.
            vocab_size: Derived from decoder_model_name.config.vocab_size
                (no manual setting needed — the pretrained model defines its
                own vocabulary size)
            dropout / rms_norm_eps / muP_scale: Derived from the pretrained
                model's config. Not set manually — the model already defines
                its own normalization epsilon, dropout, etc.

        Builder Options:
            use_positional_query_init: If True, initialize concept queries with
                positional priors (hybrid-analysis.md Section 6). This biases
                C_{k,j} toward the j-th segment of the sequence.

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

    reason_model_name: str
    reason_model_num_layers: int
    reason_model_freeze: bool

    # Solution Decoder Configuration
    decoder_model_name: str  # Default: same as reason_model_name (set in __post_init__)
    decoder_freeze: bool  # Freeze decoder backbone by default

    # Builder-specific options
    use_positional_query_init: bool  # Initialize concept queries with positional priors

    ntp_loss_weight: float
    concept_loss_weight: float
    recon_loss_weight: float

    # Optional fields (must come after all required fields in dataclass)
    reason_model_lora: Optional[Dict[str, Any]] = None
    decoder_lora: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration parameters.

        PURPOSE:
            Ensure configuration values are valid and consistent.

        VALIDATION:
            - hidden_dim must be divisible by num_heads
            - len(level_lengths) must equal num_levels
            - decoder_model_name defaults to reason_model_name if empty
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

        # Default decoder_model_name to reason_model_name if not specified
        if not self.decoder_model_name:
            self.decoder_model_name = self.reason_model_name

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
