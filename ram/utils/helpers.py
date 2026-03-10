"""
Helper functions for TAR framework.
"""

import os
import random
from typing import Any

import yaml
import torch
import numpy as np


# =============================================================================
# Config Loading with !include Support
# =============================================================================


class IncludeLoader(yaml.SafeLoader):
    """YAML Loader with !include support.

    Supports:
        !include path/to/file.yml        # Include entire file
        !include path/to/file.yml:key    # Include specific key from file
        !include path/to/file.yml:a.b.c  # Include nested key
    """

    def __init__(self, stream):
        self._root = (
            os.path.dirname(stream.name) if hasattr(stream, "name") else os.getcwd()
        )
        super().__init__(stream)


def _include_constructor(loader: IncludeLoader, node: yaml.Node) -> Any:
    """Handle !include directive.

    Args:
        loader: YAML loader instance
        node: YAML node with include path

    Returns:
        Included content (dict, list, or scalar)
    """
    value = loader.construct_scalar(node)

    # Check for key selector: !include file.yml:key
    if ":" in value and not value.startswith("/"):
        # Handle Windows paths (C:\...) vs key selector
        parts = value.rsplit(":", 1)
        if len(parts) == 2 and not parts[0].endswith("\\"):
            filepath, key = parts
        else:
            filepath, key = value, None
    else:
        filepath, key = value, None

    # Resolve relative path
    if not os.path.isabs(filepath):
        filepath = os.path.join(loader._root, filepath)

    # Load included file
    with open(filepath, "r", encoding="utf-8") as f:
        content = yaml.load(f, IncludeLoader)

    # Extract specific key if specified
    if key is not None:
        for k in key.split("."):
            content = content[k]

    return content


IncludeLoader.add_constructor("!include", _include_constructor)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file with !include support.

    Supports:
        !include path/to/file.yml        # Include entire file
        !include path/to/file.yml:key    # Include specific key
        !include path/to/file.yml:a.b.c  # Include nested key

    Example config.yml:
        model:
          encoder: !include encoders/bert.yml
          decoder: !include decoders/gpt2.yml:decoder

    Args:
        config_path: Path to the YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, IncludeLoader)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        int: Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_device() -> torch.device:
    """Get the best available device.

    Returns:
        torch.device: CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
