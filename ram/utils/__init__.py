"""TAR Utilities

General utility functions for the TAR framework.

Modules:
    config - YAML configuration loading with !include support
    tools  - Training and environment utilities (seed, device, etc.)
    serialization - JSON serialization utilities
    storage - TrainingHistory and ReconstructionSampleStore
    logging - TrainingLogger for structured training logs
    factory - Factory functions for creating data structures
"""

from .config import load_config
from .tools import (
    set_seed,
    count_parameters,
    get_device,
    setup_environment,
    select_best_gpu,
    get_gpu_info,
    assign_model_devices,
    collate_fn_text,
    decode_logits_to_text,
    find_latest_checkpoint,
    resume_from_checkpoint,
    save_checkpoint,
)
from .serialization import (
    to_json_serializable,
    save_json,
    load_json,
)
from .storage import (
    TrainingHistory,
    ReconstructionSampleStore,
)
from .logging import TrainingLogger
from .factory import (
    create_training_config,
    create_reconstruction_samples,
)

__all__ = [
    # Config
    "load_config",
    # Tools
    "set_seed",
    "count_parameters",
    "get_device",
    "setup_environment",
    "select_best_gpu",
    "get_gpu_info",
    "assign_model_devices",
    "collate_fn_text",
    "decode_logits_to_text",
    "find_latest_checkpoint",
    "resume_from_checkpoint",
    "save_checkpoint",
    # Serialization
    "to_json_serializable",
    "save_json",
    "load_json",
    # Storage
    "TrainingHistory",
    "ReconstructionSampleStore",
    # Logging
    "TrainingLogger",
    # Factory
    "create_training_config",
    "create_reconstruction_samples",
]
