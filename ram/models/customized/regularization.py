"""Regularization utilities for deep transformer networks.

Provides stochastic depth (DropPath) for training deep TAR models.
"""

import torch
from torch import nn

__all__ = ["drop_path", "DropPath"]


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample.

    Args:
        x: (B, L, C) input tensor
        drop_prob: Probability of dropping a path
        training: Whether in training mode
        scale_by_keep: Scale output by 1/keep_prob

    Returns:
        (B, L, C) output tensor (same or zeroed for dropped paths)
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample as a module."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"
