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
                Transformer. Loaded as AutoModelForCausalLM (includes lm_head)
                so a single model serves BOTH roles:
                (1) Backbone: CoT hidden state extraction (encode_cot)
                (2) lm_head: NTP / reasoning loss on solution tokens
            reason_model_num_layers: Number of layers to use (-1 = all)
            reason_model_freeze: Whether to freeze backbone parameters
            reason_model_lora: Optional LoRA config dict for reason_model.
                If non-None, applies PEFT LoRA adapters to the backbone.
                Example: {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}

        Builder Options:
            use_positional_query_init: If True, initialize concept queries with
                positional priors (hybrid-analysis.md Section 6). This biases
                C_{k,j} toward the j-th segment of the sequence.
            use_reasoning_loss: If True, enable NTP / reasoning loss during
                Builder training. Adds back_proj (D → D_encoder) and computes
                cross-entropy on solution tokens given Q + concept pyramid.
                This validates that the extracted pyramid supports effective
                reasoning — arguably more important than recon_loss alone.
                When False, only recon_loss + ordering_loss + residual_loss are used.

        Loss Weights:
            ntp_loss_weight: Weight for NTP / reasoning loss.
                The reason_model (AutoModelForCausalLM) includes lm_head.
                Given Q + concept pyramid, can it generate the correct solution?
                This is the essential validation: a pyramid that reconstructs
                CoT hidden states but cannot support reasoning is useless.
                Currently 0.0 in configs (not yet implemented in training loop).
            concept_loss_weight: Weight for concept ordering loss.
                Encourages concepts within each level to attend to sequential
                positions in the CoT (intra-level ordering constraint via Gaussian
                soft targets on attention weights).
            recon_loss_weight: Weight for CoT reconstruction loss.
                Measures MSE between the pyramid-reconstructed hidden states
                (f_hat_K = sum_k A_k^T @ C_k_base) and the projected CoT hidden
                states. Ensures the pyramid preserves CoT information.
    """

    hidden_dim: int
    num_heads: int
    num_levels: int
    level_lengths: List[int]
    max_seq_len: int

    reason_model_name: str
    reason_model_num_layers: int
    reason_model_freeze: bool

    # Builder-specific options
    use_positional_query_init: bool  # Initialize concept queries with positional priors
    use_reasoning_loss: bool  # Enable NTP reasoning loss (Q + pyramid → solution)

    # ntp_loss: NTP / reasoning loss. The reason_model includes lm_head
    # so it can generate solutions from Q + concept pyramid. This loss
    # validates that the pyramid supports effective reasoning.
    # Currently 0.0 (not yet implemented in training loop).
    ntp_loss_weight: float

    # concept_loss: Intra-level ordering loss. Encourages concepts within
    # each level to attend to sequential CoT positions (Gaussian soft targets).
    concept_loss_weight: float

    # recon_loss: CoT reconstruction loss. MSE between pyramid-reconstructed
    # hidden states and the original projected CoT hidden states.
    recon_loss_weight: float

    # Optional fields (must come after all required fields in dataclass)
    reason_model_lora: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, yaml_dict: dict) -> "NLCPV3Config":
        """Construct NLCPV3Config from a nested YAML dict.

        The YAML structure is:
          model.pyramid:       hidden_dim, num_heads, num_levels, ...
          model.reason_model:  reason_model_name, reason_model_num_layers, ...
          model.builder:       use_positional_query_init, use_reasoning_loss
          training.loss_weights: ntp_loss_weight, concept_loss_weight, ...
        """
        m = yaml_dict["model"]
        rm = m["reason_model"]
        pyr = m["pyramid"]
        bld = m["builder"]
        tr = yaml_dict["training"]
        lw = tr["loss_weights"]
        return cls(
            hidden_dim=pyr["hidden_dim"],
            num_heads=pyr["num_heads"],
            num_levels=pyr["num_levels"],
            level_lengths=pyr["level_lengths"],
            max_seq_len=pyr["max_seq_len"],
            reason_model_name=rm["reason_model_name"],
            reason_model_num_layers=rm["reason_model_num_layers"],
            reason_model_freeze=rm["reason_model_freeze"],
            reason_model_lora=rm["reason_model_lora"],
            use_positional_query_init=bld["use_positional_query_init"],
            use_reasoning_loss=bld["use_reasoning_loss"],
            ntp_loss_weight=lw["ntp_loss_weight"],
            concept_loss_weight=lw["concept_loss_weight"],
            recon_loss_weight=lw["recon_loss_weight"],
        )

    def __post_init__(self):
        """Validate configuration parameters.

        PURPOSE:
            Ensure configuration values are valid and consistent.

        VALIDATION:
            - hidden_dim must be divisible by num_heads
            - len(level_lengths) must equal num_levels
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
