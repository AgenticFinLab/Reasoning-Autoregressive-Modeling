"""TAR Loss Functions.

This module provides loss functions for Text AutoRegressive (TAR) models,
including reconstruction losses, VQ losses, and combined losses.

Loss Function Selection Guide:
==============================

1. Same Tokenizer (T5, BART):
   - Use: ReconstructionLoss(same_tokenizer=True)
   - Or: compute_reconstruction_loss()
   - Target: input_ids from shared tokenizer

2. Different Tokenizers (BERT + GPT2):
   - Use: DualTokenizerReconstructionLoss
   - Or: DualTokenizerVQAELoss
   - Target: re-tokenize texts with DECODER's tokenizer!

3. VQ-AE Training:
   - Use: VQAELoss (same tokenizer)
   - Or: DualTokenizerVQAELoss (different tokenizers)
   - Combines: recon_loss + λ * vq_loss

4. VQ Loss Only:
   - Use: VQLoss or compute_vq_loss()
   - For: standalone quantizer training/debugging

Registry Usage:
===============
    # Get loss by type name
    loss_fn = get_loss("vqae", config)

    # Build from config with validation
    loss_fn, warnings = build_loss_from_config(config, enc_tok, dec_tok)

    # Validate config consistency
    warnings = validate_loss_config(config, enc_tok, dec_tok)

Tokenizer Validation:
=====================
    >>> from ram.losses import validate_tokenizer_compatibility
    >>> result = validate_tokenizer_compatibility(enc_tokenizer, dec_tokenizer)
    >>> print(result["recommendation"])

Classes:
    ReconstructionLoss          - Standard cross-entropy reconstruction
    DualTokenizerReconstructionLoss - For different encoder/decoder tokenizers
    VQLoss                      - Vector quantization loss
    MultiScaleVQLoss            - Multi-scale VQ loss (VAR-style)
    VQAELoss                    - Combined VQ-AE loss (same tokenizer)
    DualTokenizerVQAELoss       - Combined loss (different tokenizers)

Functions:
    compute_reconstruction_loss  - Functional reconstruction loss
    compute_vq_loss             - Functional VQ loss
    compute_vqae_loss           - Functional combined loss
    validate_tokenizer_compatibility - Check tokenizer compatibility
    straight_through_estimator  - STE for gradient flow through quantization
    get_loss                    - Get loss by type name from registry
    build_loss_from_config      - Build loss from config with validation
    validate_loss_config        - Validate loss config against tokenizers
    infer_loss_type             - Auto-infer appropriate loss type
"""

# Reconstruction losses
from .reconstruction import (
    ReconstructionLoss,
    DualTokenizerReconstructionLoss,
    compute_reconstruction_loss,
    validate_tokenizer_compatibility,
)

# VQ losses
from .vq_loss import (
    VQLoss,
    MultiScaleVQLoss,
    compute_vq_loss,
    straight_through_estimator,
)

# Combined losses
from .combined import (
    VQAELoss,
    DualTokenizerVQAELoss,
    compute_vqae_loss,
)

# Registry
from .registry import (
    get_loss,
    get_available_losses,
    validate_loss_config,
    infer_loss_type,
    build_loss_from_config,
)

__all__ = [
    # Reconstruction
    "ReconstructionLoss",
    "DualTokenizerReconstructionLoss",
    "compute_reconstruction_loss",
    "validate_tokenizer_compatibility",
    # VQ
    "VQLoss",
    "MultiScaleVQLoss",
    "compute_vq_loss",
    "straight_through_estimator",
    # Combined
    "VQAELoss",
    "DualTokenizerVQAELoss",
    "compute_vqae_loss",
    # Registry
    "get_loss",
    "get_available_losses",
    "validate_loss_config",
    "infer_loss_type",
    "build_loss_from_config",
]
