"""Unified logging utilities for training.

Provides structured logging via Python logging module with file and console output.

Classes:
    TrainingLogger - Unified training logger with structured output methods
"""

import logging
from pathlib import Path
from typing import List, Optional

from ram.generic import ReconstructionSample, TrainingConfig, TrainingStep


class TrainingLogger:
    """Unified training logger using Python logging module.

    Provides structured logging for training progress with both console
    and file output. Includes specialized methods for logging training
    configuration, steps, epochs, and reconstruction samples.

    Log Format:
        2024-01-01 12:00:00 | INFO | Message here

    Attributes:
        logger: Python logger instance.
        log_file: Optional log file path (None if console-only).

    Example:
        >>> logger = TrainingLogger("c3_train", Path("logs/training.log"))
        >>> logger.log_header("Training Started")
        >>> logger.log_config(training_config)
        >>> logger.log_step(step_record)
    """

    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
    ):
        """Initialize training logger.

        Args:
            name: Logger name (usually experiment name).
            log_file: Optional path to log file. If provided, logs are written
                to both console and file.
            level: Logging level (default: INFO).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_format = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

    def info(self, msg: str) -> None:
        """Log INFO level message.

        Args:
            msg: Message to log.
        """
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log WARNING level message.

        Args:
            msg: Message to log.
        """
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log ERROR level message.

        Args:
            msg: Message to log.
        """
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        """Log DEBUG level message.

        Args:
            msg: Message to log.
        """
        self.logger.debug(msg)

    def log_header(self, title: str, width: int = 60) -> None:
        """Log a section header with double-line borders.

        Visual output:
            ============================================================
            Title Here
            ============================================================

        Args:
            title: Header title text.
            width: Total width in characters (default: 60).
        """
        self.logger.info("=" * width)
        self.logger.info(title)
        self.logger.info("=" * width)

    def log_subheader(self, title: str, width: int = 60) -> None:
        """Log a subsection header with single-line borders.

        Visual output:
            ------------------------------------------------------------
            Subsection Title
            ------------------------------------------------------------

        Args:
            title: Subsection title text.
            width: Total width in characters (default: 60).
        """
        self.logger.info("-" * width)
        self.logger.info(title)
        self.logger.info("-" * width)

    def log_config(self, config: TrainingConfig) -> None:
        """Log training configuration summary.

        Outputs key hyperparameters and model architecture info:
            - Experiment name, batch size, learning rate
            - Gradient accumulation, effective batch size
            - Compression ratio, devices (if pipeline mode)

        Args:
            config: TrainingConfig instance to log.
        """
        self.logger.info("Training Configuration:")
        self.logger.info(f"  Experiment: {config.experiment_name}")
        self.logger.info(f"  Batch size: {config.batch_size}")
        self.logger.info(f"  Learning rate: {config.learning_rate}")
        self.logger.info(f"  Epochs: {config.num_epochs}")
        self.logger.info(
            f"  Gradient accumulation: {config.gradient_accumulation_steps}"
        )
        self.logger.info(
            f"  Effective batch size: "
            f"{config.batch_size * config.gradient_accumulation_steps}"
        )
        self.logger.info(f"  Max length (M): {config.max_length}")
        self.logger.info(f"  Latent token len (N): {config.latent_token_len}")
        self.logger.info(f"  Compression ratio: {config.compression_ratio:.1f}x")
        self.logger.info(f"  BFloat16: {config.bf16}")
        self.logger.info(f"  Pipeline mode: {config.use_pipeline}")
        if config.use_pipeline:
            self.logger.info(f"  Encoder device: {config.encoder_device}")
            self.logger.info(f"  Decoder device: {config.decoder_device}")

    def log_step(
        self,
        step: TrainingStep,
        log_interval: int = 10,
    ) -> None:
        """Log training step info.

        Logs step number, losses, and learning rates.
        Only logs when global_step is divisible by log_interval.

        Args:
            step: TrainingStep record to log.
            log_interval: Log every N steps (default: 10).
        """
        if step.global_step % log_interval == 0:
            self.logger.info(
                f"Step {step.global_step}: "
                f"loss={step.total_loss:.4f}, "
                f"avg_loss={step.avg_loss:.4f}, "
                f"lr_enc={step.lr_encoder:.2e}, "
                f"lr_dec={step.lr_decoder:.2e}"
            )

    def log_epoch(
        self,
        epoch: int,
        avg_loss: float,
        total_epochs: int,
    ) -> None:
        """Log epoch completion.

        Args:
            epoch: Completed epoch number (1-indexed).
            avg_loss: Average loss for the epoch.
            total_epochs: Total number of epochs.
        """
        self.logger.info(
            f"Epoch {epoch}/{total_epochs} completed: avg_loss={avg_loss:.4f}"
        )

    def log_checkpoint(self, path: Path) -> None:
        """Log checkpoint saved event.

        Args:
            path: Checkpoint file path.
        """
        self.logger.info(f"Checkpoint saved: {path}")

    def log_reconstruction(
        self,
        samples: List[ReconstructionSample],
        max_display: int = 2,
    ) -> None:
        """Log reconstruction samples for visual inspection.

        Shows truncated original and reconstructed text for comparison.

        Args:
            samples: List of ReconstructionSample instances.
            max_display: Maximum samples to display (default: 2).
        """
        self.logger.info("Reconstruction samples:")
        for i, sample in enumerate(samples[:max_display]):
            self.logger.info(f"  [{i}] Original: {sample.original[:80]}...")
            self.logger.info(f"      Reconstructed: {sample.reconstructed[:80]}...")
