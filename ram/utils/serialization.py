"""Serialization utilities for JSON and tensor data handling.

Provides functions for converting Python objects to JSON-serializable formats
and handling file I/O with proper serialization.

Functions:
    to_json_serializable - Convert objects to JSON-compatible formats
    save_json - Save data to JSON file with serialization
    load_json - Load data from JSON file
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def to_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format.

    Recursively converts objects to formats compatible with JSON serialization.
    Handles special types like PyTorch tensors, Path objects, and datetimes.

    Type Handling:
        - torch.Tensor: Converted to nested list via .cpu().tolist()
        - Path: Converted to string representation
        - datetime: Converted to ISO format string
        - dict: Recursively converts all values
        - list/tuple: Recursively converts all elements
        - dataclass: Converted via asdict() then processed

    Args:
        obj: Object to convert. Can be any Python object.

    Returns:
        JSON-serializable version of the object.

    Example:
        >>> import torch
        >>> tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> to_json_serializable({"tensor": tensor, "path": Path("/tmp")})
        {'tensor': [1.0, 2.0, 3.0], 'path': '/tmp'}
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    elif hasattr(obj, "__dataclass_fields__"):
        return to_json_serializable(asdict(obj))
    else:
        return obj


def save_json(data: dict, path: Path, indent: int = 2) -> None:
    """Save data to JSON file with serialization handling.

    Automatically converts non-serializable types (tensors, paths, etc.)
    before writing to file. Creates parent directories if needed.

    Args:
        data: Dictionary to save. Will be processed via to_json_serializable().
        path: Output file path. Parent directories created automatically.
        indent: JSON indentation for readability (default: 2).

    Example:
        >>> save_json({"loss": 0.5, "step": 100}, Path("logs/history.json"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = to_json_serializable(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """Load data from JSON file.

    Args:
        path: Input file path to load.

    Returns:
        Loaded dictionary from JSON file.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file contains invalid JSON.

    Example:
        >>> data = load_json(Path("logs/history.json"))
        >>> print(data["step"])
        100
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
