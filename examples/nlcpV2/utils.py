"""NLCP V2 Utility Functions.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V2.md

PURPOSE:
    Provide utility functions used across NLCP V2 modules.
    These functions implement common operations with clear documentation
    of their mathematical definitions and dimension transformations.

FUNCTION CATEGORIES:
    - Attention masks: Causal masking for autoregressive generation
    - Loss computation: NTP loss with proper normalization
    - Normalization: RMSNorm for stable training
"""

from typing import Optional

import torch
import torch.nn.functional as F


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask for autoregressive generation.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.3 (Concept Transformer)

    PURPOSE:
        Creates a lower-triangular boolean mask where position i can only
        attend to positions j <= i. This ensures autoregressive property
        during training and inference.

    MATHEMATICAL DEFINITION:
        mask[i, j] = True  if j <= i (can attend)
        mask[i, j] = False if j > i  (cannot attend)

    DIMENSION:
        Input:  seq_len (scalar integer)
        Output: [seq_len, seq_len] boolean tensor

    EXAMPLE:
        seq_len = 4
        Output:
            [[True,  False, False, False],
             [True,  True,  False, False],
             [True,  True,  True,  False],
             [True,  True,  True,  True ]]

    Args:
        seq_len: Sequence length L (number of positions)
        device: Target device for the mask (CPU/GPU)

    Returns:
        [L, L] Causal mask with True for allowed positions
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


def compute_ntp_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    padding_id: int,
) -> torch.Tensor:
    """Compute Next Token Prediction loss.

    DESIGN SOURCE: concept-pyramid-V2.md Section 3.1 (Loss Function)

    PURPOSE:
        Standard cross-entropy loss for next token prediction.
        Computes negative log-likelihood of target tokens given predictions.

    MATHEMATICAL DEFINITION:
        L_NTP = - (1/T) * sum_{t=1}^{T} log P(x_t | x_{<t})

        Where:
            T = number of non-padding tokens
            x_t = target token at position t
            P(x_t | x_{<t}) = softmax(logits)[t, x_t]

    DIMENSION:
        logits:     [B, L, V] - Model output logits
        target_ids: [B, L]    - Target token IDs
        output:     [1]       - Scalar loss value

    EXAMPLE:
        Batch size B = 2, Sequence length L = 3, Vocab size V = 1000
        logits:     [2, 3, 1000] - Raw model outputs
        target_ids: [[5, 23, 88],
                    [12, 5, 0]]   - Target tokens (0 = padding)
        padding_id: 0

        If position [1, 2] has padding_id, it is excluded from loss.

    Args:
        logits: [B, L, V] Model output logits
        target_ids: [B, L] Target token IDs
        padding_id: Padding token ID to ignore in loss computation

    Returns:
        [1] Scalar NTP loss (mean over non-padding tokens)
    """
    batch_size, seq_len, vocab_size = logits.shape

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = target_ids.view(-1)

    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=padding_id,
        reduction="sum",
    )

    mask = (target_ids != padding_id).float()
    num_tokens = mask.sum()

    if num_tokens > 0:
        loss = loss / num_tokens

    return loss


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root Mean Square Layer Normalization.

    DESIGN SOURCE: concept-pyramid-V2.md Section 2.2.1 (Encoder)

    PURPOSE:
        Normalizes input tensor by its root mean square.
        More computationally efficient than LayerNorm (no mean subtraction).
        Used in modern LLMs like Llama and Qwen.

    MATHEMATICAL DEFINITION:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps)

        Where:
            mean(x^2) = (1/D) * sum_{d=1}^{D} x_d^2
            D = hidden dimension
            eps = small constant for numerical stability

    DIMENSION:
        Input:  [..., D] - Tensor of any shape (last dim is hidden dim)
        Output: [..., D] - Normalized tensor with same shape

    EXAMPLE:
        x = [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]  # shape [2, 3]

        For first row [1.0, 2.0, 3.0]:
            mean(x^2) = (1 + 4 + 9) / 3 = 4.67
            rms = sqrt(4.67 + 1e-6) ≈ 2.16
            output = [1.0/2.16, 2.0/2.16, 3.0/2.16] ≈ [0.46, 0.93, 1.39]

    Args:
        x: Input tensor of any shape (last dimension is normalized)
        eps: Small constant for numerical stability (default: 1e-6)

    Returns:
        Normalized tensor with same shape as input
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized
