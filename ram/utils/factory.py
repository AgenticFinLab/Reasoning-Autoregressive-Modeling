"""Factory functions for creating training data structures.

Provides convenience functions for creating TrainingConfig and
ReconstructionSample instances from config dictionaries and decode results.

Functions:
    create_training_config - Create TrainingConfig from YAML config dict
    create_reconstruction_samples - Create ReconstructionSample list from decode result
"""

from typing import Any, Dict, List

from ram.generic import (
    DecoderConfig,
    EncoderConfig,
    QuantizerConfig,
    ReconstructionSample,
    TrainingConfig,
)


def create_training_config(
    config_dict: Dict[str, Any],
    experiment_name: str,
) -> TrainingConfig:
    """Create TrainingConfig from YAML config dictionary.

    Extracts relevant fields from the loaded YAML config and constructs
    a TrainingConfig instance with nested EncoderConfig, DecoderConfig,
    and optional QuantizerConfig.

    Args:
        config_dict: Configuration dictionary loaded from YAML file.
            Expected keys: "train", "model", optional "model_devices".
        experiment_name: Name identifier for the experiment.

    Returns:
        TrainingConfig instance with all nested configurations.

    Example:
        >>> import yaml
        >>> with open("config.yml") as f:
        ...     config_dict = yaml.safe_load(f)
        >>> config = create_training_config(config_dict, "c3_original")
        >>> print(config.batch_size)
        2
    """
    train_cfg = config_dict["training"]
    model_cfg = config_dict["model"]
    env_cfg = config_dict["environment"]
    model_devices = env_cfg["device_map"]

    enc_cfg = model_cfg["encoder"]
    latent_token_len = enc_cfg["latent_token_len"]
    max_length = enc_cfg["max_length"]
    compression_ratio = max_length / latent_token_len

    encoder_device = model_devices["encoder"]["device"]
    decoder_device = model_devices["decoder"]["device"]
    use_pipeline = encoder_device != decoder_device

    return TrainingConfig(
        experiment_name=experiment_name,
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        num_epochs=train_cfg["num_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        gradient_accumulation_steps=train_cfg["gradient"]["accumulation_steps"],
        gradient_clip=train_cfg["gradient"]["max_grad_norm"],
        bf16=train_cfg["bf16"],
        latent_token_len=latent_token_len,
        max_length=max_length,
        compression_ratio=compression_ratio,
        encoder_config=EncoderConfig(**enc_cfg),
        decoder_config=DecoderConfig(**model_cfg["decoder"]),
        quantizer_config=(
            QuantizerConfig(**model_cfg["quantizer"])
            if model_cfg["quantizer"] is not None
            else None
        ),
        use_pipeline=use_pipeline,
        encoder_device=encoder_device,
        decoder_device=decoder_device,
    )


def create_reconstruction_samples(
    decode_result: Dict[str, Any],
) -> List[ReconstructionSample]:
    """Create ReconstructionSample list from decode_logits_to_text result.

    Converts the output from decode_logits_to_text() into a list of
    ReconstructionSample dataclass instances for structured storage.

    Args:
        decode_result: Result dictionary from decode_logits_to_text().
            Expected key: "comparisons" with list of comparison dicts.
            Each comparison should have: "index", "original", "reconstructed".

    Returns:
        List of ReconstructionSample instances.

    Example:
        >>> decode_result = decode_logits_to_text(logits, tokenizer, texts)
        >>> samples = create_reconstruction_samples(decode_result)
        >>> print(samples[0].original[:50])
        "Original input text..."
    """
    samples = []
    comparisons = decode_result["comparisons"]
    for comp in comparisons:
        samples.append(
            ReconstructionSample(
                index=comp["index"],
                original=comp["original"],
                reconstructed=comp["reconstructed"],
            )
        )
    return samples
