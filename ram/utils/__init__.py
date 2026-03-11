"""TAR Utilities

General utility functions for the TAR framework.

Modules:
    config - YAML configuration loading with !include support
    tools  - Training and environment utilities (seed, device, etc.)
"""

from .config import load_config
from .tools import (
    set_seed,
    count_parameters,
    get_device,
    setup_environment,
)

__all__ = [
    "load_config",
    "set_seed",
    "count_parameters",
    "get_device",
    "setup_environment",
]
