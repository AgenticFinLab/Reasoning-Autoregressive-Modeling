"""Sampling utilities for TAR (Text Auto-Regressive) inference.

Provides top-k, top-p (nucleus) sampling and Gumbel softmax for
generating codebook indices during autoregressive text generation.
"""

import torch
from torch.nn import functional as F

__all__ = ["sample_with_top_k_top_p_", "gumbel_softmax_with_rng"]


def sample_with_top_k_top_p_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng=None,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Sample from logits with top-k and/or top-p (nucleus) filtering.

    Args:
        logits_BlV: (B, l, V) logits tensor
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (0 = disabled)
        rng: Optional random generator for reproducibility
        num_samples: Number of samples per position

    Returns:
        (B, l, num_samples) sampled indices from codebook
    """
    B, l, V = logits_BlV.shape

    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)

    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (
            1 - top_p
        )
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(
                sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
            ),
            -torch.inf,
        )

    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng,
    ).view(B, l, num_samples)


def gumbel_softmax_with_rng(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    rng: torch.Generator = None,
) -> torch.Tensor:
    """
    Gumbel softmax with optional custom RNG for reproducibility.

    Args:
        logits: (B, l, V) input logits
        tau: Temperature parameter (lower = sharper)
        hard: If True, use straight-through estimator
        eps: Small constant for numerical stability
        dim: Dimension to apply softmax
        rng: Optional random generator

    Returns:
        (B, l, V) soft one-hot or one-hot (if hard=True)
    """
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_(generator=rng)
        .log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
