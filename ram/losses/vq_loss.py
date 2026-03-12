"""Vector Quantization Losses for VQ-VAE.

This module implements VQ-VAE losses including:
- Commitment loss: Encourages encoder outputs to stay close to codebook
- Codebook loss: Encourages codebook to stay close to encoder outputs
- Combined VQ loss with configurable weights

VQ Loss Components:
===================

Standard VQ-VAE Loss (van den Oord et al., 2017):
    L_vq = ||sg[z_e] - e||² + β * ||z_e - sg[e]||²

    Where:
        z_e = encoder output [B, L, D]
        e = quantized (codebook lookup) [B, L, D]
        sg = stop gradient
        β = commitment cost weight (typically 0.25)

    Term 1 (Codebook Loss):
        ||sg[z_e] - e||²
        - Gradient flows to codebook only
        - Moves codebook vectors toward encoder outputs

    Term 2 (Commitment Loss):
        β * ||z_e - sg[e]||²
        - Gradient flows to encoder only
        - Encourages encoder to commit to codebook entries
        - β controls how strongly encoder is regularized

Multi-Scale VQ Loss (VAR-style):
    L_vq = Σ_k L_vq^k

    Where k iterates over scales [1, 2, 4, 8, 16, 32, ...]
    Each scale contributes its own VQ loss.

Flow Diagram:
=============
    z [B, L, D] (encoder output)
         ↓
    ┌────────────────────────────────────┐
    │ For each scale k:                  │
    │   z_down = downsample(z, k)        │  [B, k, D]
    │   indices = codebook_lookup(z_down)│  [B, k]
    │   e = codebook[indices]            │  [B, k, D]
    │   L_k = ||sg[z_down] - e||²        │
    │       + β * ||z_down - sg[e]||²    │
    └────────────────────────────────────┘
         ↓
    L_vq = Σ_k L_k
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLoss(nn.Module):
    """Vector Quantization loss for VQ-VAE.

    Computes the combined commitment and codebook loss for vector quantization.

    Loss Formula:
        L_vq = codebook_loss + β * commitment_loss
             = ||sg[z] - q||² + β * ||z - sg[q]||²

    Where:
        z = input features (encoder output)
        q = quantized features (codebook lookup)
        sg = stop gradient (detach)
        β = commitment cost weight

    Args:
        beta: Commitment cost weight (default: 0.25)
            - Higher β → encoder outputs stay closer to codebook
            - Lower β → more flexibility for encoder
            - Typical range: 0.1 - 1.0
        reduction: Loss reduction ('mean', 'sum', 'none')

    Input shapes:
        z: [B, L, D] or [B, D] - encoder output
        q: [B, L, D] or [B, D] - quantized output (from codebook)

    Output:
        loss: Scalar or tensor depending on reduction
        loss_dict: Dictionary with individual loss components

    Example:
        >>> vq_loss = VQLoss(beta=0.25)
        >>> z = encoder(x)           # [B, L, D]
        >>> q = codebook.lookup(z)   # [B, L, D]
        >>> loss, details = vq_loss(z, q)
        >>> print(f"VQ Loss: {loss.item():.4f}")
        >>> print(f"Commitment: {details['commitment_loss']:.4f}")
        >>> print(f"Codebook: {details['codebook_loss']:.4f}")
    """

    def __init__(
        self,
        beta: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

        # Validation
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

    def forward(
        self,
        z: torch.Tensor,
        q: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute VQ loss.

        Args:
            z: Encoder output [B, L, D] or [B, D]
            q: Quantized output [B, L, D] or [B, D]

        Returns:
            loss: Combined VQ loss
            loss_dict: Dictionary with:
                - commitment_loss: ||z - sg[q]||²
                - codebook_loss: ||sg[z] - q||²
                - total_loss: β * commitment + codebook
        """
        # Validate shapes match
        if z.shape != q.shape:
            raise ValueError(
                f"Shape mismatch: z.shape={z.shape}, q.shape={q.shape}. "
                f"Encoder output and quantized output must have same shape!"
            )

        # Commitment loss: ||z - sg[q]||²
        # Gradient flows to encoder (z), not codebook (q)
        commitment_loss = F.mse_loss(z, q.detach(), reduction=self.reduction)

        # Codebook loss: ||sg[z] - q||²
        # Gradient flows to codebook (q), not encoder (z)
        codebook_loss = F.mse_loss(z.detach(), q, reduction=self.reduction)

        # Combined loss
        total_loss = codebook_loss + self.beta * commitment_loss

        loss_dict = {
            "commitment_loss": (
                commitment_loss.item()
                if commitment_loss.dim() == 0
                else commitment_loss
            ),
            "codebook_loss": (
                codebook_loss.item() if codebook_loss.dim() == 0 else codebook_loss
            ),
            "total_loss": total_loss.item() if total_loss.dim() == 0 else total_loss,
            "beta": self.beta,
        }

        return total_loss, loss_dict


class MultiScaleVQLoss(nn.Module):
    """Multi-scale VQ loss for VAR-style quantization.

    Computes VQ loss across multiple scales, where each scale
    captures different granularity of information.

    Multi-Scale Quantization Flow:
        scale=1:  z [B, L, D] → avg_pool → [B, 1, D] → VQ → loss_1
        scale=2:  z [B, L, D] → avg_pool → [B, 2, D] → VQ → loss_2
        scale=4:  z [B, L, D] → avg_pool → [B, 4, D] → VQ → loss_4
        ...
        Total: L_vq = Σ_k loss_k

    Why Multi-Scale?
        - Scale 1: Captures global/coarse information
        - Scale 2-8: Captures mid-level patterns
        - Scale 16+: Captures fine-grained details
        - Combined: Hierarchical representation

    Args:
        beta: Commitment cost weight
        scale_weights: Optional per-scale loss weights
            - If None, all scales weighted equally
            - If provided, len must match number of scales
        reduction: Loss reduction method

    Input:
        z_per_scale: List of encoder outputs at each scale
            - z_per_scale[k] has shape [B, scale_k, D]
        q_per_scale: List of quantized outputs at each scale
            - q_per_scale[k] has shape [B, scale_k, D]

    Output:
        total_loss: Scalar, sum of all scale losses
        loss_dict: Per-scale loss breakdown

    Example:
        >>> ms_vq_loss = MultiScaleVQLoss(beta=0.25)
        >>> # From quantizer forward pass
        >>> z_scales = [z_1, z_2, z_4, z_8]  # Different scales
        >>> q_scales = [q_1, q_2, q_4, q_8]
        >>> loss, details = ms_vq_loss(z_scales, q_scales)
    """

    def __init__(
        self,
        beta: float = 0.25,
        scale_weights: Optional[List[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.beta = beta
        self.scale_weights = scale_weights
        self.reduction = reduction
        self.vq_loss = VQLoss(beta=beta, reduction=reduction)

    def forward(
        self,
        z_per_scale: List[torch.Tensor],
        q_per_scale: List[torch.Tensor],
        scale_lengths: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute multi-scale VQ loss.

        Args:
            z_per_scale: List of [B, scale_k, D] encoder outputs per scale
            q_per_scale: List of [B, scale_k, D] quantized outputs per scale
            scale_lengths: Optional list of scale sizes for logging

        Returns:
            total_loss: Sum of all scale losses
            loss_dict: Breakdown including per-scale losses
        """
        num_scales = len(z_per_scale)

        if len(q_per_scale) != num_scales:
            raise ValueError(
                f"Number of scales mismatch: z has {num_scales}, q has {len(q_per_scale)}"
            )

        # Validate scale weights if provided
        if self.scale_weights is not None:
            if len(self.scale_weights) != num_scales:
                raise ValueError(
                    f"scale_weights length ({len(self.scale_weights)}) != "
                    f"num_scales ({num_scales})"
                )
            weights = self.scale_weights
        else:
            weights = [1.0] * num_scales

        # Compute loss per scale
        total_loss = 0.0
        per_scale_losses = []

        for k, (z_k, q_k, w_k) in enumerate(zip(z_per_scale, q_per_scale, weights)):
            loss_k, details_k = self.vq_loss(z_k, q_k)
            weighted_loss_k = w_k * loss_k
            total_loss = total_loss + weighted_loss_k

            scale_info = {
                "scale_idx": k,
                "scale_len": scale_lengths[k] if scale_lengths else z_k.shape[1],
                "loss": loss_k.item() if isinstance(loss_k, torch.Tensor) else loss_k,
                "weighted_loss": (
                    weighted_loss_k.item()
                    if isinstance(weighted_loss_k, torch.Tensor)
                    else weighted_loss_k
                ),
                "weight": w_k,
            }
            per_scale_losses.append(scale_info)

        loss_dict = {
            "total_loss": (
                total_loss.item()
                if isinstance(total_loss, torch.Tensor)
                else total_loss
            ),
            "num_scales": num_scales,
            "beta": self.beta,
            "per_scale": per_scale_losses,
        }

        return total_loss, loss_dict


def compute_vq_loss(
    z: torch.Tensor,
    q: torch.Tensor,
    beta: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional API for VQ loss computation.

    This is a simple functional interface for computing VQ loss
    without instantiating a loss class.

    Formula:
        L_vq = ||sg[z] - q||² + β * ||z - sg[q]||²

    Args:
        z: Encoder output [B, L, D]
        q: Quantized output [B, L, D]
        beta: Commitment cost weight

    Returns:
        vq_loss: Combined loss (codebook + β * commitment)
        commitment_loss: ||z - sg[q]||²
        codebook_loss: ||sg[z] - q||²

    Dimension semantics:
        B = batch size
        L = sequence length
        D = codebook dimension (feature dimension)

    Example:
        >>> z = encoder(x)           # [4, 64, 256]
        >>> q = quantize(z)          # [4, 64, 256]
        >>> vq_loss, commit, codebook = compute_vq_loss(z, q, beta=0.25)
    """
    # Commitment loss: encoder → codebook direction
    commitment_loss = F.mse_loss(z, q.detach())

    # Codebook loss: codebook → encoder direction
    codebook_loss = F.mse_loss(z.detach(), q)

    # Combined
    vq_loss = codebook_loss + beta * commitment_loss

    return vq_loss, commitment_loss, codebook_loss


def straight_through_estimator(
    z: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Apply Straight-Through Estimator (STE) for VQ gradient flow.

    The STE allows gradients to flow through the quantization step
    by copying gradients from quantized output to encoder output.

    Formula:
        forward: q (quantized)
        backward: gradient flows to z (encoder output)

    Implementation:
        q_ste = z + (q - z).detach()
              = z + sg[q - z]
              = z - sg[z] + sg[q]

    Why this works:
        - Forward pass: q_ste = q (detached terms cancel)
        - Backward pass: ∂q_ste/∂z = 1 (gradient identity)

    Args:
        z: Encoder output [B, L, D]
        q: Quantized output [B, L, D]

    Returns:
        q_ste: [B, L, D] quantized with gradient passthrough

    Example:
        >>> z = encoder(x)           # [B, L, D], requires_grad=True
        >>> q = codebook.lookup(z)   # [B, L, D], no gradient path
        >>> q_ste = straight_through_estimator(z, q)
        >>> # Now gradients can flow: loss → q_ste → z → encoder
    """
    # STE: copy gradients from q to z
    return z + (q - z).detach()
