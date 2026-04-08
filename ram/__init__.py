"""TAR: Text Auto-Regressive Modeling

A framework for Chain-of-Thought reasoning inspired by VAR (Visual Autoregressive Modeling).

Key Components:
- TextEncoder: HuggingFace encoder with tokenizer
- TextDecoder: HuggingFace decoder for generation
- MultiScaleQuantizer: Multi-scale residual quantization (VAR innovation)
- TextVQVAE: Complete autoencoder
- Losses: Reconstruction, VQ, and combined losses with tokenizer validation

Data Loading:
    Use lmbase for data loading:
    >>> from lmbase.dataset import registry
    >>> dataset = registry.get("gsm8k", split="train")

    Or use RAM's data loading utilities:
    >>> from ram.data_load import RamDataLoaderRegistry
    >>> loader = RamDataLoaderRegistry(
    ...     data_config={"data_name": "gsm8k"},
    ...     split="train",
    ... )

Usage:
    from ram.models import TextEncoder, build_encoder
    from ram.utils import load_config
    from ram.losses import VQAELoss, validate_tokenizer_compatibility

    config = load_config('configs/uTEST/encoder.yml')
    encoder = build_encoder(config['model']['encoder'])
"""

from .models import (
    TextEncoder,
    TextDecoder,
    MultiScaleQuantizer,
    TextVQVAE,
    build_encoder,
    build_decoder,
    build_quantizer,
    build_text_vqvae,
)
from .utils import load_config, set_seed, count_parameters, get_device
from .losses import (
    ReconstructionLoss,
    DualTokenizerReconstructionLoss,
    VQLoss,
    VQAELoss,
    DualTokenizerVQAELoss,
    validate_tokenizer_compatibility,
)
from .generic import (
    # Config
    TrainingConfig,
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    QuantizerConfig,
    # Step
    TrainingStep,
    TrainingStepWithQuantizer,
    ReconstructionSample,
    # Data Loading
    RamSample,
    RamReconstructSample,
    # Checkpoint
    CheckpointData,
    CheckpointMetadata,
)
from .utils import (
    # Storage
    TrainingHistory,
    ReconstructionSampleStore,
    # Logging
    TrainingLogger,
    # Factory
    create_training_config,
    create_reconstruction_samples,
    # Serialization
    save_json,
    load_json,
    to_json_serializable,
)
from .data_load import (
    # Data Loading
    RamDataLoaderRegistry,
)

__version__ = "0.4.0"
__all__ = [
    # Models
    "TextEncoder",
    "TextDecoder",
    "MultiScaleQuantizer",
    "TextVQVAE",
    "build_encoder",
    "build_decoder",
    "build_quantizer",
    "build_text_vqvae",
    # Utils
    "load_config",
    "set_seed",
    "count_parameters",
    "get_device",
    # Losses
    "ReconstructionLoss",
    "DualTokenizerReconstructionLoss",
    "VQLoss",
    "VQAELoss",
    "DualTokenizerVQAELoss",
    "validate_tokenizer_compatibility",
    # Generic - Config
    "TrainingConfig",
    "ModelConfig",
    "EncoderConfig",
    "DecoderConfig",
    "QuantizerConfig",
    # Generic - Step
    "TrainingStep",
    "TrainingStepWithQuantizer",
    "ReconstructionSample",
    # Generic - Checkpoint
    "CheckpointData",
    "CheckpointMetadata",
    # Generic - Manager
    "TrainingHistory",
    "TrainingLogger",
    "ReconstructionSampleStore",
    # Generic - Utilities
    "create_training_config",
    "create_reconstruction_samples",
    "save_json",
    "load_json",
    "to_json_serializable",
    # Data Loading
    "RamDataLoaderRegistry",
]
