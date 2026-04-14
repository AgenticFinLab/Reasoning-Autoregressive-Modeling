"""NLCP V2 (Next-Level Concept Pyramid V2) Implementation.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md
    - Section 2: Architecture Overview
    - Section 3: Training Methodology
    - Section 4: Inference Pipeline

PURPOSE:
    NLCP V2 implements hierarchical latent space autoregressive modeling
    with Residual Attentive Pooling for concept extraction.

    This is a novel architecture that bridges the gap between:
    - DLCM's concept-level generation (but with soft boundaries)
    - VAR's multi-scale generation (but with semantic concepts)

ARCHITECTURE OVERVIEW:
    NLCP V2 consists of four core modules:

    1. Encoder (nlcpV2.encoder)
       - Uses pretrained HuggingFace models (e.g., Qwen)
       - Training: Encodes Q+CoT to H
       - Inference: Encodes Q and pools/projects to H_0

    2. Attentive Pooling (nlcpV2.attentive_pooling)
       - Implements Residual Attentive Pooling algorithm
       - Training only: Extracts concept targets from Q+CoT
       - Produces C_0, C_1, ..., C_K as supervision signals

    3. Concept Transformer (nlcpV2.concept_transformer)
       - Generates concepts level by level
       - VAR-style "Next-Scale" pattern
       - Inter-level causal, intra-level parallel

    4. Token Decoder (nlcpV2.token_decoder)
       - Causal cross-attention: tokens (Q) attend to concepts (KV)
       - Projects H_K to vocabulary logits
       - Generates tokens autoregressively

MODULES:
    config: NLCPV2Config
        Configuration class with all hyperparameters

    encoder: NLCPV2Encoder
        Encoder with training/inference separation

    attentive_pooling: ResidualAttentivePooling
        Residual Attentive Pooling for concept extraction

    concept_transformer: ConceptTransformer
        Hierarchical concept generation with level-level causality

    token_decoder: TokenDecoder
        Vocabulary projection via causal cross-attention

    model: NLCPV2Model
        Complete model implementation orchestrating all modules

    types: NLCPV2Output
        Type definitions for model outputs

    utils: Utility functions
        create_causal_mask, compute_ntp_loss, rms_norm

QUICK START:
    >>> from nlcpV2 import NLCPV2Config, NLCPV2Model
    >>>
    >>> # Create configuration
    >>> config = NLCPV2Config(
    ...     hidden_dim=256,
    ...     num_heads=8,
    ...     vocab_size=128000,
    ...     num_levels=4,
    ...     level_lengths=[4, 16, 64, 256],
    ...     max_seq_len=512,
    ...     dropout=0.1,
    ...     rms_norm_eps=1e-6,
    ...     encoder_model_name="Qwen/Qwen2.5-0.5B",
    ...     encoder_num_layers=None,
    ...     encoder_freeze=True,
    ...     ntp_loss_weight=1.0,
    ...     concept_loss_weight=0.1,
    ...     recon_loss_weight=0.05,
    ...     muP_scale=1.0,
    ... )
    >>>
    >>> # Initialize model
    >>> model = NLCPV2Model(config)
    >>>
    >>> # Training
    >>> output = model.forward_training(input_ids, target_ids, padding_id=0)
    >>> loss = output.total_loss
    >>> loss.backward()
    >>>
    >>> # Inference/Generation
    >>> generated = model.generate(input_ids, max_new_tokens=100)

DIMENSION CONVENTIONS:
    B: Batch size
    L: Token sequence length
    L_k: Concept count at level k
    L_q: Question token count
    L_K: Final level concept count
    D: Hidden dimension (config.hidden_dim)
    D_encoder: Encoder hidden dimension
    V: Vocabulary size
    K: Number of levels
    T: Target sequence length

TRAINING-INFERENCE SEPARATION:
    Training (requires Q+CoT):
        1. Encoder: Q+CoT → H
        2. Attentive Pooling: H → C_k (concept targets)
        3. Concept Transformer: H_0 → H_k (predicted)
        4. Token Decoder: H_K → logits
        5. Compute losses (L_NTP, L_concept, L_recon)

    Inference (Q only):
        1. Encoder: Q → H_0 (no Attentive Pooling)
        2. Concept Transformer: H_0 → H_K
        3. Token Decoder: H_K → generated tokens
"""

from nlcpV2.config import NLCPV2Config
from nlcpV2.model import NLCPV2Model

__version__ = "2.0.0"

__all__ = [
    "NLCPV2Config",
    "NLCPV2Model",
]
