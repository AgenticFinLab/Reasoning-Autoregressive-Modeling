"""Generic data structures for training storage and logging.

This module defines the unified data structures for the storage and recording system:

    1. Checkpoint (.pt): Model weights, optimizer states, scheduler states
    2. Training History (.json): Step-by-step training records
    3. Training Config (.json): One-time configuration snapshot

Design Principles:
    - All dataclasses support inheritance for model-specific extensions
    - Dataclasses are grouped logically (Config, Step, Checkpoint, Sample)
    - Utility functions and classes are in ram/utils/ modules

Related Modules:
    - ram/utils/serialization.py: JSON serialization utilities
    - ram/utils/storage.py: TrainingHistory, ReconstructionSampleStore
    - ram/utils/logging.py: TrainingLogger
    - ram/utils/factory.py: Factory functions
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ram.utils.serialization import load_json, save_json, to_json_serializable


# =============================================================================
# Module Exports (Dataclasses Only)
# =============================================================================

__all__ = [
    # Model Configuration
    "ModelConfig",
    "EncoderConfig",
    "DecoderConfig",
    "QuantizerConfig",
    # Training Configuration
    "TrainingConfig",
    # Training Step
    "TrainingStep",
    "TrainingStepWithQuantizer",
    "ReconstructionSample",
    # Checkpoint
    "CheckpointMetadata",
    "CheckpointData",
]


# =============================================================================
# Model Configuration Dataclasses
# =============================================================================


@dataclass
class ModelConfig:
    """Base model configuration for all model types.

    Provides common configuration fields shared by encoder, decoder,
    and quantizer models. Subclasses add specialized fields.

    Attributes:
        model_name: HuggingFace model name or local path.
            Examples: "Qwen/Qwen2.5-1.5B", "./checkpoints/my_model"
        pretrained: Whether to load pretrained weights from HuggingFace.
            Set to False when loading from custom checkpoint.
        freeze: Whether to freeze model weights during training.
            Set to True for frozen encoder in transfer learning.
        device: Target device assignment.
            Options: "auto" (smart selection), "cuda:0", "cuda:1", "cpu"

    Example:
        >>> config = ModelConfig(
        ...     model_name="Qwen/Qwen2.5-1.5B",
        ...     pretrained=True,
        ...     freeze=False,
        ...     device="cuda:0"
        ... )
    """

    model_name: str = ""
    pretrained: bool = True
    freeze: bool = False
    device: str = "auto"


@dataclass
class EncoderConfig(ModelConfig):
    """Encoder model configuration for text-to-latent compression.

    Extends ModelConfig with encoder-specific fields for latent token
    generation and special token handling.

    Architecture:
        Input Text -> Tokenizer -> LLM Encoder -> Latent Tokens [B, N, D]

    Attributes:
        latent_token_len (N): Number of latent tokens to generate.
            Higher values = better reconstruction, lower compression.
            Typical: 32 (40x compression at 1280 chars), 64 (20x)
        max_length (M): Maximum input sequence length.
            Determines compression ratio: M/N.
            Typical: 2048, 8192 for long context models
        use_im_start_end: Whether to use special tokens for latent boundaries.
            C3-style: wraps latent tokens with <img> and </img> tokens.

    Compression Ratio Examples:
        - M=1280, N=32: 40x compression
        - M=1280, N=64: 20x compression
        - M=2048, N=32: 64x compression
    """

    latent_token_len: int = 32
    max_length: int = 2048
    use_im_start_end: bool = True


@dataclass
class DecoderConfig(ModelConfig):
    """Decoder model configuration for latent-to-text reconstruction.

    Extends ModelConfig with decoder-specific fields for latent token
    processing. Must have matching latent_token_len with encoder.

    Architecture:
        Latent Tokens [B, N, D] + Prompt -> LLM Decoder -> Reconstructed Text

    Attributes:
        latent_token_len (N): Number of latent tokens to process.
            MUST match the encoder's latent_token_len for correct operation.

    Example:
        >>> # Encoder and decoder must have matching latent_token_len
        >>> enc_config = EncoderConfig(latent_token_len=32, ...)
        >>> dec_config = DecoderConfig(latent_token_len=32, ...)
    """

    latent_token_len: int = 32


@dataclass
class QuantizerConfig(ModelConfig):
    """Quantizer model configuration for vector quantization (VQ-VAE style).

    Extends ModelConfig with quantizer-specific fields for codebook-based
    discrete latent representation. Used in VAR-style multi-scale quantization.

    Architecture:
        Continuous Latent -> Quantizer -> Discrete Codebook Indices

    Attributes:
        codebook_size: Number of entries in the codebook (vocabulary size).
            Larger = more expressive but higher memory/compute.
            Typical: 8192, 16384
        codebook_dim: Dimension of each codebook entry vector.
            Must match the latent dimension from encoder.
            Typical: 256, 512
        num_codebooks: Number of parallel codebooks for multi-scale.
            VAR-style: multiple codebooks for hierarchical representation.
            Typical: 1 (single), 4 (multi-scale)

    Example:
        >>> # Single codebook quantizer
        >>> quant_config = QuantizerConfig(
        ...     codebook_size=8192,
        ...     codebook_dim=256,
        ...     num_codebooks=1
        ... )
    """

    codebook_size: int = 8192
    codebook_dim: int = 256
    num_codebooks: int = 1


# =============================================================================
# Training Configuration Dataclass
# =============================================================================


@dataclass
class TrainingConfig:
    """Complete training configuration snapshot.

    Stores all hyperparameters and architecture info at training start.
    Saved once per experiment for reproducibility and reference.

    Organization:
        - Basic: experiment_name, created_at
        - Hyperparameters: batch_size, learning_rate, epochs, etc.
        - Architecture: latent_token_len, max_length, compression_ratio
        - Model configs: encoder_config, decoder_config, quantizer_config
        - Device: use_pipeline, encoder_device, decoder_device

    Attributes:
        experiment_name: Unique identifier for this experiment run.
        batch_size: Per-device batch size for training.
        learning_rate: Peak learning rate for optimizer.
        weight_decay: L2 regularization coefficient.
        num_epochs: Total number of training epochs.
        warmup_ratio: Fraction of training for LR warmup (0.01 = 1%).
        gradient_accumulation_steps: Steps to accumulate before update.
        gradient_clip: Maximum gradient norm for clipping.
        bf16: Whether to use BFloat16 mixed precision training.
        latent_token_len: Number of latent tokens (N).
        max_length: Maximum sequence length (M).
        compression_ratio: Compression ratio M/N (computed).
        encoder_config: Encoder model configuration.
        decoder_config: Decoder model configuration.
        quantizer_config: Optional quantizer configuration.
        use_pipeline: Whether using multi-GPU pipeline parallelism.
        encoder_device: Device assignment for encoder.
        decoder_device: Device assignment for decoder.
        created_at: ISO timestamp when config was created.

    Example:
        >>> config = TrainingConfig(
        ...     experiment_name="c3_baseline",
        ...     batch_size=2,
        ...     learning_rate=1e-5,
        ...     num_epochs=5,
        ...     latent_token_len=32,
        ...     max_length=2048,
        ... )
        >>> config.save(Path("logs/train_config.json"))
    """

    experiment_name: str = ""
    batch_size: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    num_epochs: int = 5
    warmup_ratio: float = 0.01
    gradient_accumulation_steps: int = 16
    gradient_clip: float = 1.0
    bf16: bool = True
    latent_token_len: int = 32
    max_length: int = 2048
    compression_ratio: float = 64.0
    encoder_config: Optional[EncoderConfig] = None
    decoder_config: Optional[DecoderConfig] = None
    quantizer_config: Optional[QuantizerConfig] = None
    use_pipeline: bool = False
    encoder_device: str = "cuda:0"
    decoder_device: str = "cuda:0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary.

        Handles nested dataclass instances automatically via
        to_json_serializable recursive processing.
        """
        return to_json_serializable(self)

    def save(self, path: Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Output file path for the configuration.
        """
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load configuration from JSON file.

        Reconstructs nested dataclass instances (EncoderConfig, etc.).

        Args:
            path: Input file path to load from.

        Returns:
            TrainingConfig instance with all nested configs restored.
        """
        data = load_json(path)
        if "encoder_config" in data and data["encoder_config"]:
            data["encoder_config"] = EncoderConfig(**data["encoder_config"])
        if "decoder_config" in data and data["decoder_config"]:
            data["decoder_config"] = DecoderConfig(**data["decoder_config"])
        if "quantizer_config" in data and data["quantizer_config"]:
            data["quantizer_config"] = QuantizerConfig(**data["quantizer_config"])
        return cls(**data)


# =============================================================================
# Training Step Dataclasses
# =============================================================================


@dataclass
class ReconstructionSample:
    """Single reconstruction sample for a training step.

    Captures one input-output pair for visual inspection of model
    reconstruction quality during training.

    Attributes:
        index: Sample index within the batch (0-indexed).
        original: Original input text before encoding.
        reconstructed: Reconstructed text from decoder output.

    Example:
        >>> sample = ReconstructionSample(
        ...     index=0,
        ...     original="The quick brown fox jumps over the lazy dog.",
        ...     reconstructed="The quick brown fox jumped over the lazy dog."
        ... )
    """

    index: int = 0
    original: str = ""
    reconstructed: str = ""


@dataclass
class TrainingStep:
    """Single training step record for history tracking.

    Captures all relevant metrics and metadata for one training step.
    Stored in training_history.json for analysis and visualization.

    Organization:
        - Position: epoch, step_in_epoch, global_step
        - Losses: total_loss, recon_loss, avg_loss
        - Learning rates: lr_encoder, lr_decoder, lr_quantizer
        - Data: reconstruction_samples (optional)
        - Metadata: timestamp

    Attributes:
        epoch: Current epoch number (1-indexed).
        step_in_epoch: Step number within current epoch (1-indexed).
        global_step: Global step counter across all epochs.
        total_loss: Combined loss value for this step.
        recon_loss: Reconstruction loss component.
        avg_loss: Running average loss (for trend tracking).
        lr_encoder: Current encoder learning rate.
        lr_decoder: Current decoder learning rate.
        lr_quantizer: Current quantizer learning rate (if applicable).
        reconstruction_samples: Optional list of reconstruction samples.
        timestamp: ISO format timestamp when step was recorded.

    Example:
        >>> step = TrainingStep(
        ...     epoch=1,
        ...     step_in_epoch=100,
        ...     global_step=100,
        ...     total_loss=0.5,
        ...     recon_loss=0.5,
        ...     avg_loss=0.52,
        ...     lr_encoder=1e-5,
        ...     lr_decoder=1e-5,
        ... )
    """

    epoch: int = 1
    step_in_epoch: int = 1
    global_step: int = 1
    total_loss: float = 0.0
    recon_loss: float = 0.0
    avg_loss: float = 0.0
    lr_encoder: float = 0.0
    lr_decoder: float = 0.0
    lr_quantizer: Optional[float] = None
    reconstruction_samples: List[ReconstructionSample] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Only includes non-None optional fields and non-empty lists.
        """
        d = {
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "global_step": self.global_step,
            "total_loss": self.total_loss,
            "recon_loss": self.recon_loss,
            "avg_loss": self.avg_loss,
            "lr_encoder": self.lr_encoder,
            "lr_decoder": self.lr_decoder,
            "timestamp": self.timestamp,
        }
        if self.lr_quantizer is not None:
            d["lr_quantizer"] = self.lr_quantizer
        if self.reconstruction_samples:
            d["reconstruction_samples"] = [
                asdict(s) for s in self.reconstruction_samples
            ]
        return d


@dataclass
class TrainingStepWithQuantizer(TrainingStep):
    """Training step with quantizer-specific metrics.

    Extends TrainingStep for models using vector quantization (VQ-VAE, VAR).
    Adds codebook health metrics for monitoring quantizer quality.

    Additional Attributes:
        vq_loss: Vector quantization commitment loss.
            Measures how close encoder outputs are to codebook entries.
        codebook_usage: Codebook utilization ratio (0.0 to 1.0).
            Fraction of codebook entries actively used.
            Low values indicate codebook collapse.
        perplexity: Codebook perplexity metric.
            Measures diversity of codebook usage.
            Higher = more uniform usage across codebook.

    Example:
        >>> step = TrainingStepWithQuantizer(
        ...     epoch=1,
        ...     global_step=100,
        ...     vq_loss=0.1,
        ...     codebook_usage=0.8,  # 80% of codebook used
        ...     perplexity=500.0,    # Good diversity
        ... )
    """

    vq_loss: float = 0.0
    codebook_usage: float = 0.0
    perplexity: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary including quantizer metrics."""
        d = super().to_dict()
        d["vq_loss"] = self.vq_loss
        d["codebook_usage"] = self.codebook_usage
        d["perplexity"] = self.perplexity
        return d


# =============================================================================
# Checkpoint Dataclasses
# =============================================================================


@dataclass
class CheckpointMetadata:
    """Metadata for training checkpoint files.

    Stored inside checkpoint .pt files for traceability and resume support.
    Provides quick access to training state without loading full checkpoint.

    Attributes:
        epoch: Epoch number at checkpoint time (1-indexed).
        global_step: Global step at checkpoint time.
        step_in_epoch: Step within epoch at checkpoint time.
        avg_loss: Running average loss at checkpoint.
        timestamp: ISO format timestamp when checkpoint was saved.
        experiment_name: Experiment identifier for reference.

    Example:
        >>> metadata = CheckpointMetadata(
        ...     epoch=5,
        ...     global_step=1000,
        ...     step_in_epoch=200,
        ...     avg_loss=0.35,
        ...     experiment_name="c3_baseline",
        ... )
    """

    epoch: int = 1
    global_step: int = 1
    step_in_epoch: int = 1
    avg_loss: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    experiment_name: str = ""


@dataclass
class CheckpointData:
    """Checkpoint data structure for .pt file storage.

    Contains all model weights, optimizer states, and scheduler states
    for training resumption. Stored via torch.save() for tensor support.

    Storage Format (checkpoint.pt):
        {
            "epoch": 5,
            "global_step": 1000,
            "step_in_epoch": 200,
            "avg_loss": 0.35,
            "timestamp": "2024-01-01T12:00:00",
            "experiment_name": "c3_baseline",
            "encoder_state_dict": {...},
            "decoder_state_dict": {...},
            "encoder_optimizer_state_dict": {...},
            "decoder_optimizer_state_dict": {...},
            "encoder_scheduler_state_dict": {...},
            "decoder_scheduler_state_dict": {...},
            ...extra fields...
        }

    Attributes:
        metadata: Checkpoint metadata with training state info.
        models: Dict mapping model name to state_dict.
            Example: {"encoder": encoder.state_dict(), ...}
        optimizers: Dict mapping optimizer name to state_dict.
            Example: {"encoder": enc_opt.state_dict(), ...}
        schedulers: Dict mapping scheduler name to state_dict.
            Example: {"encoder": enc_sched.state_dict(), ...}
        extra: Additional tensor data (e.g., quantizer codebooks).

    Example:
        >>> # Create from model instances
        >>> ckpt = CheckpointData.from_models(
        ...     models={"encoder": encoder, "decoder": decoder},
        ...     optimizers={"encoder": enc_opt, "decoder": dec_opt},
        ...     schedulers={"encoder": enc_sched, "decoder": dec_sched},
        ...     metadata=CheckpointMetadata(epoch=5, global_step=1000),
        ... )
        >>> ckpt.save(Path("checkpoints/checkpoint_epoch5.pt"))
    """

    metadata: CheckpointMetadata
    models: Dict[str, dict] = field(default_factory=dict)
    optimizers: Dict[str, dict] = field(default_factory=dict)
    schedulers: Dict[str, dict] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to checkpoint dictionary for torch.save.

        Flattens the structure with prefixed keys for compatibility
        with standard PyTorch checkpoint loading conventions.
        """
        result = {
            "epoch": self.metadata.epoch,
            "global_step": self.metadata.global_step,
            "step_in_epoch": self.metadata.step_in_epoch,
            "avg_loss": self.metadata.avg_loss,
            "timestamp": self.metadata.timestamp,
            "experiment_name": self.metadata.experiment_name,
        }
        for name, state_dict in self.models.items():
            result[f"{name}_state_dict"] = state_dict
        for name, state_dict in self.optimizers.items():
            result[f"{name}_optimizer_state_dict"] = state_dict
        for name, state_dict in self.schedulers.items():
            result[f"{name}_scheduler_state_dict"] = state_dict
        for key, value in self.extra.items():
            result[key] = value
        return result

    def save(self, path: Path) -> None:
        """Save checkpoint to .pt file.

        Args:
            path: Output file path for the checkpoint.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def from_models(
        cls,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, Any],
        metadata: CheckpointMetadata,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "CheckpointData":
        """Create CheckpointData from model instances.

        Convenience factory that extracts state_dicts from live model
        and optimizer instances.

        Args:
            models: Dict mapping name to model instance.
            optimizers: Dict mapping name to optimizer instance.
            schedulers: Dict mapping name to scheduler instance.
            metadata: Checkpoint metadata.
            extra: Optional additional tensor data.

        Returns:
            CheckpointData instance ready for saving.
        """
        return cls(
            metadata=metadata,
            models={name: model.state_dict() for name, model in models.items()},
            optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
            schedulers={name: sched.state_dict() for name, sched in schedulers.items()},
            extra=extra or {},
        )
