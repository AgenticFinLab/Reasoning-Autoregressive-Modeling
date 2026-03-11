"""
Training and environment utility tools for TAR framework.

Functions:
    set_seed        - Set random seed for reproducibility
    get_device      - Get compute device (auto/cuda/mps/cpu)
    setup_environment - Combined seed + device setup
    count_parameters  - Count model parameters
"""

import random
from typing import List

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


def collate_fn_text(batch) -> List[str]:
    """Extract text from dataset samples.

    Supports common dataset formats with 'question' or 'problem' fields.

    Args:
        batch: List of dataset samples (dicts or other).

    Returns:
        List[str]: Extracted text strings.
    """
    texts = []
    for sample in batch:
        if isinstance(sample, dict):
            if "question" in sample:
                texts.append(sample["question"])
            elif "problem" in sample:
                texts.append(sample["problem"])
            elif "text" in sample:
                texts.append(sample["text"])
            else:
                texts.append(str(sample))
        else:
            texts.append(str(sample))
    return texts


def decode_logits_to_text(
    logits: torch.Tensor,
    tokenizer,
    original_texts: List[str] = None,
) -> dict:
    """Decode logits to text and compare with original.

    Restoration procedure:
        logits [B, L, V=50257] -> argmax(dim=-1) -> pred_ids [B, L]
        pred_ids [B, L] -> tokenizer.decode() -> List[str] texts

        V=50257 is GPT2's vocabulary size:
        - Each position has 50257 logits (one per token)
        - argmax over dim=-1 selects the most likely token ID
        - tokenizer.decode converts token IDs back to text

    Args:
        logits: Decoder output [B, L, V]
        tokenizer: Tokenizer for decoding (e.g., GPT2Tokenizer)
        original_texts: Optional original texts for comparison

    Returns:
        dict with keys:
            - pred_ids: List[List[int]] predicted token IDs (JSON-serializable)
            - pred_texts: List[str] decoded texts
            - original_texts: List[str] original input texts (if provided)
            - comparisons: List[dict] with original/reconstructed pairs
    """
    # logits [B, L, V] -> argmax(dim=-1) -> pred_ids [B, L]
    pred_ids_tensor = logits.argmax(dim=-1)

    # Convert to list for JSON serialization
    pred_ids = pred_ids_tensor.cpu().tolist()

    # pred_ids [B, L] -> tokenizer.decode() -> List[str]
    pred_texts = []
    for i in range(pred_ids_tensor.shape[0]):
        text = tokenizer.decode(pred_ids_tensor[i], skip_special_tokens=True)
        pred_texts.append(text)

    result = {
        "pred_ids": pred_ids,
        "pred_texts": pred_texts,
    }

    # Add original texts and comparisons if provided
    if original_texts is not None:
        result["original_texts"] = original_texts
        comparisons = []
        for i, (orig, pred) in enumerate(zip(original_texts, pred_texts)):
            comparisons.append(
                {
                    "index": i,
                    "original": orig,
                    "reconstructed": pred,
                }
            )
        result["comparisons"] = comparisons

    return result
