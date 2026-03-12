"""Loss Function Registry with Configuration Validation.

This module provides a registry for loss functions and automatic validation
to ensure the configured loss matches the encoder-decoder tokenizer setup.

Registry Usage:
===============
    from ram.losses import get_loss, validate_loss_config

    # Get loss function by name
    loss_fn = get_loss("vqae", config)

    # Validate config consistency
    warnings = validate_loss_config(config, enc_tokenizer, dec_tokenizer)

Supported Loss Types:
=====================
    - "reconstruction": ReconstructionLoss (same tokenizer)
    - "dual_tokenizer_reconstruction": DualTokenizerReconstructionLoss
    - "vqae": VQAELoss (same tokenizer + VQ)
    - "dual_tokenizer_vqae": DualTokenizerVQAELoss (different tokenizers + VQ)

Configuration Format (in train section of YAML):
================================================
    train:
      loss:
        type: "vqae"              # Loss type name
        vq_weight: 1.0            # Weight for VQ loss
        beta: 0.25                # Commitment cost
        label_smoothing: 0.0      # Label smoothing
        # For dual_tokenizer types, dec_tokenizer is passed separately
"""

import warnings
from typing import Optional, Dict, Any, Tuple, List

import torch.nn as nn

from .reconstruction import (
    ReconstructionLoss,
    DualTokenizerReconstructionLoss,
    validate_tokenizer_compatibility,
)
from .vq_loss import VQLoss
from .combined import VQAELoss, DualTokenizerVQAELoss


# =============================================================================
# Loss Registry
# =============================================================================

_LOSS_REGISTRY = {
    "reconstruction": ReconstructionLoss,
    "dual_tokenizer_reconstruction": DualTokenizerReconstructionLoss,
    "vq": VQLoss,
    "vqae": VQAELoss,
    "dual_tokenizer_vqae": DualTokenizerVQAELoss,
}

# Loss types that require same tokenizer
_SAME_TOKENIZER_LOSSES = {"reconstruction", "vqae"}

# Loss types that require different tokenizers (with dec_tokenizer)
_DUAL_TOKENIZER_LOSSES = {"dual_tokenizer_reconstruction", "dual_tokenizer_vqae"}


def get_available_losses() -> List[str]:
    """Get list of available loss types."""
    return list(_LOSS_REGISTRY.keys())


def get_loss(
    loss_type: str,
    config: Dict[str, Any],
    dec_tokenizer=None,
    dec_vocab_size: Optional[int] = None,
) -> nn.Module:
    """Get loss function by type name.

    Args:
        loss_type: Loss type name (e.g., "vqae", "dual_tokenizer_vqae")
        config: Loss configuration dict with parameters
        dec_tokenizer: Decoder tokenizer (required for dual_tokenizer types)
        dec_vocab_size: Decoder vocab size (required for dual_tokenizer types)

    Returns:
        Configured loss module

    Raises:
        ValueError: If loss_type is unknown or required parameters missing

    Example:
        >>> config = {"vq_weight": 1.0, "beta": 0.25}
        >>> loss_fn = get_loss("vqae", config)

        >>> # For different tokenizers
        >>> loss_fn = get_loss("dual_tokenizer_vqae", config,
        ...                    dec_tokenizer=gpt2_tok, dec_vocab_size=50257)
    """
    if loss_type not in _LOSS_REGISTRY:
        available = ", ".join(get_available_losses())
        raise ValueError(f"Unknown loss type: '{loss_type}'. Available: {available}")

    loss_class = _LOSS_REGISTRY[loss_type]

    # Build kwargs based on loss type
    # Each loss type has specific accepted parameters:
    #   - VQLoss: beta, reduction (NO ignore_index, NO label_smoothing)
    #   - ReconstructionLoss: same_tokenizer, ignore_index, label_smoothing, reduction
    #   - DualTokenizerReconstructionLoss: dec_tokenizer, dec_vocab_size, ignore_index,
    #                                      max_length, label_smoothing
    #   - VQAELoss: vq_weight, beta, same_tokenizer, ignore_index, label_smoothing
    #   - DualTokenizerVQAELoss: dec_tokenizer, dec_vocab_size, vq_weight, beta,
    #                           ignore_index, max_length, label_smoothing
    kwargs = {}

    # --- VQ-only loss (VQLoss) ---
    if loss_type == "vq":
        # VQLoss only accepts: beta, reduction
        if "beta" in config:
            kwargs["beta"] = config["beta"]
        if "reduction" in config:
            kwargs["reduction"] = config["reduction"]
        return loss_class(**kwargs)

    # --- Losses with reconstruction component ---
    # These accept: ignore_index, label_smoothing
    if "ignore_index" in config:
        kwargs["ignore_index"] = config["ignore_index"]
    if "label_smoothing" in config:
        kwargs["label_smoothing"] = config["label_smoothing"]

    # VQ-related parameters (for vqae, dual_tokenizer_vqae)
    if loss_type in {"vqae", "dual_tokenizer_vqae"}:
        if "beta" in config:
            kwargs["beta"] = config["beta"]
        if "vq_weight" in config:
            kwargs["vq_weight"] = config["vq_weight"]

    # Same tokenizer flag (for reconstruction, vqae)
    if loss_type == "reconstruction":
        kwargs["same_tokenizer"] = True
    elif loss_type == "vqae":
        kwargs["same_tokenizer"] = True

    # Dual tokenizer requirements (for dual_tokenizer_* types)
    if loss_type in _DUAL_TOKENIZER_LOSSES:
        if dec_tokenizer is None:
            raise ValueError(
                f"Loss type '{loss_type}' requires dec_tokenizer parameter!"
            )
        if dec_vocab_size is None:
            raise ValueError(
                f"Loss type '{loss_type}' requires dec_vocab_size parameter!"
            )
        kwargs["dec_tokenizer"] = dec_tokenizer
        kwargs["dec_vocab_size"] = dec_vocab_size
        if "max_length" in config:
            kwargs["max_length"] = config["max_length"]

    return loss_class(**kwargs)


# =============================================================================
# Configuration Validation
# =============================================================================


def validate_loss_config(
    config: Dict[str, Any],
    enc_tokenizer=None,
    dec_tokenizer=None,
    raise_on_error: bool = False,
) -> List[str]:
    """Validate loss configuration against encoder-decoder setup.

    Checks if the configured loss type matches the tokenizer configuration
    and issues warnings for potential mismatches.

    Args:
        config: Full configuration dict (with model and train sections)
        enc_tokenizer: Encoder tokenizer instance
        dec_tokenizer: Decoder tokenizer instance
        raise_on_error: If True, raise ValueError on critical errors

    Returns:
        List of warning messages (empty if no issues)

    Warning Scenarios:
        1. Same tokenizer loss configured but tokenizers are different
        2. Dual tokenizer loss configured but tokenizers are the same
        3. VQ loss configured but no quantizer in model config
        4. Missing required loss parameters

    Example:
        >>> warnings = validate_loss_config(config, bert_tok, gpt2_tok)
        >>> for w in warnings:
        ...     print(f"WARNING: {w}")
    """
    warning_messages = []

    # Extract loss config
    train_cfg = config.get("train", {})
    loss_cfg = train_cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "reconstruction")

    model_cfg = config.get("model", {})
    has_quantizer = model_cfg.get("quantizer") is not None

    # Check tokenizer compatibility if both provided
    if enc_tokenizer is not None and dec_tokenizer is not None:
        compat = validate_tokenizer_compatibility(enc_tokenizer, dec_tokenizer)
        same_tokenizer = compat["same_tokenizer"]

        # Check: Same tokenizer loss with different tokenizers
        if loss_type in _SAME_TOKENIZER_LOSSES and not same_tokenizer:
            msg = (
                f"Loss type '{loss_type}' expects same tokenizer, but encoder and decoder "
                f"have different tokenizers (enc_vocab={compat['enc_vocab_size']}, "
                f"dec_vocab={compat['dec_vocab_size']}). "
                f"Consider using 'dual_tokenizer_{loss_type}' instead."
            )
            warning_messages.append(msg)
            warnings.warn(msg, UserWarning)

        # Check: Dual tokenizer loss with same tokenizers
        if loss_type in _DUAL_TOKENIZER_LOSSES and same_tokenizer:
            msg = (
                f"Loss type '{loss_type}' is for different tokenizers, but encoder and decoder "
                f"appear to have the same tokenizer. Consider using '{loss_type.replace('dual_tokenizer_', '')}' "
                f"for better efficiency."
            )
            warning_messages.append(msg)
            warnings.warn(msg, UserWarning)

    # Check: VQ loss without quantizer
    if loss_type in {"vqae", "dual_tokenizer_vqae", "vq"} and not has_quantizer:
        msg = (
            f"Loss type '{loss_type}' includes VQ loss, but no quantizer is configured "
            f"in model config. VQ loss component will be zero."
        )
        warning_messages.append(msg)
        warnings.warn(msg, UserWarning)

    # Check: Missing VQ parameters
    if loss_type in {"vqae", "dual_tokenizer_vqae"}:
        if "vq_weight" not in loss_cfg:
            msg = "VQ loss configured but 'vq_weight' not specified, using default 1.0"
            warning_messages.append(msg)
        if "beta" not in loss_cfg:
            msg = "VQ loss configured but 'beta' not specified, using default 0.25"
            warning_messages.append(msg)

    # Raise if critical errors and raise_on_error is True
    if raise_on_error and warning_messages:
        critical = [m for m in warning_messages if "expects same tokenizer" in m]
        if critical:
            raise ValueError(critical[0])

    return warning_messages


def infer_loss_type(
    enc_tokenizer=None,
    dec_tokenizer=None,
    has_quantizer: bool = False,
) -> str:
    """Automatically infer appropriate loss type based on configuration.

    Args:
        enc_tokenizer: Encoder tokenizer
        dec_tokenizer: Decoder tokenizer
        has_quantizer: Whether quantizer is present

    Returns:
        Recommended loss type name

    Example:
        >>> loss_type = infer_loss_type(bert_tok, gpt2_tok, has_quantizer=True)
        >>> print(loss_type)  # "dual_tokenizer_vqae"
    """
    # Check tokenizer compatibility
    same_tokenizer = True
    if enc_tokenizer is not None and dec_tokenizer is not None:
        compat = validate_tokenizer_compatibility(enc_tokenizer, dec_tokenizer)
        same_tokenizer = compat["same_tokenizer"]

    # Determine loss type
    if has_quantizer:
        if same_tokenizer:
            return "vqae"
        else:
            return "dual_tokenizer_vqae"
    else:
        if same_tokenizer:
            return "reconstruction"
        else:
            return "dual_tokenizer_reconstruction"


def build_loss_from_config(
    config: Dict[str, Any],
    enc_tokenizer=None,
    dec_tokenizer=None,
    dec_vocab_size: Optional[int] = None,
    validate: bool = True,
) -> Tuple[nn.Module, List[str]]:
    """Build loss function from config with automatic validation.

    This is the main entry point for creating loss functions from config files.

    Args:
        config: Full configuration dict
        enc_tokenizer: Encoder tokenizer (for validation)
        dec_tokenizer: Decoder tokenizer (required for dual_tokenizer losses)
        dec_vocab_size: Decoder vocab size
        validate: Whether to run validation checks

    Returns:
        loss_fn: Configured loss module
        warnings: List of warning messages

    Example:
        >>> config = load_config("configs/uTEST/eqd_train.yml")
        >>> loss_fn, warnings = build_loss_from_config(
        ...     config, enc_tokenizer, dec_tokenizer, dec_vocab_size=50257
        ... )
        >>> for w in warnings:
        ...     print(f"WARNING: {w}")
    """
    # Run validation
    warning_messages = []
    if validate:
        warning_messages = validate_loss_config(config, enc_tokenizer, dec_tokenizer)

    # Extract loss config
    train_cfg = config.get("train", {})
    loss_cfg = train_cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "reconstruction")

    # Build loss
    loss_fn = get_loss(
        loss_type=loss_type,
        config=loss_cfg,
        dec_tokenizer=dec_tokenizer,
        dec_vocab_size=dec_vocab_size,
    )

    return loss_fn, warning_messages
