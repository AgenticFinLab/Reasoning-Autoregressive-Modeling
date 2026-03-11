"""
Training and environment utility tools for TAR framework.

Functions:
    set_seed        - Set random seed for reproducibility
    get_device      - Get compute device (auto/cuda/mps/cpu)
    setup_environment - Combined seed + device setup
    count_parameters  - Count model parameters
"""

import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Sets seed for:
        - Python random
        - NumPy random
        - PyTorch CPU
        - PyTorch CUDA (if available)
        - cuDNN deterministic mode

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


def get_device(device: str = "auto") -> torch.device:
    """Get compute device based on config or auto-detection.

    Args:
        device: Device specification from config.
            - "auto": Auto-detect best available (cuda > mps > cpu)
            - "cuda": Force CUDA (raises error if unavailable)
            - "mps": Force MPS (Apple Silicon)
            - "cpu": Force CPU

    Returns:
        torch.device: Selected compute device.

    Raises:
        RuntimeError: If specified device is unavailable.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    elif device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    else:
        return torch.device(device)


def setup_environment(env_cfg: dict) -> torch.device:
    """Setup training environment from config.

    Handles:
        1. Random seed setting (Python, NumPy, PyTorch, CUDA)
        2. Device selection (auto/cuda/mps/cpu)

    Args:
        env_cfg: Environment config dict with keys:
            - seed (int): Random seed for reproducibility
            - device (str): Device specification ("auto", "cuda", "mps", "cpu")

    Returns:
        torch.device: Selected compute device.

    Example:
        env_cfg = {"seed": 42, "device": "auto"}
        device = setup_environment(env_cfg)
    """
    # Set random seed
    seed = env_cfg.get("seed", 42)
    set_seed(seed)

    # Get device
    device_str = env_cfg.get("device", "auto")
    device = get_device(device_str)

    return device


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
