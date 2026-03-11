"""Multi-Scale Vector Quantizer (VAR's Core Innovation).

Implements multi-scale residual quantization with f_hat accumulation.
This is the only custom component - encoder/decoder use HuggingFace.

Quantization Basics:
    Vector Quantization maps continuous vectors to discrete codebook entries:
    z [B, L, D] -> find nearest codebook vector -> indices [B, L]

    Each position selects one of V codebook vectors (V = codebook_size).
    This creates a discrete bottleneck for learning compressed representations.

Single-Scale Quantization:
    z [B, L, D] -> codebook_lookup -> quantized [B, L, D]

    Problem: One scale captures one level of detail only.
    Fine details and coarse structure compete for the same codebook.

Multi-Scale Quantization (VAR's Innovation):
    Process at multiple scales, accumulate residuals:

    Scale 1 (coarsest):  z -> down(1)  -> [B, 1, D]  -> quantize -> up(L) -> f_hat
    Scale 2:             residual -> down(2)  -> [B, 2, D]  -> quantize -> up(L) -> f_hat +=
    Scale 4:             residual -> down(4)  -> [B, 4, D]  -> quantize -> up(L) -> f_hat +=
    ...                  ...
    Scale L (finest):    residual -> down(L)  -> [B, L, D]  -> quantize -> up(L) -> f_hat +=

    Each scale captures different granularity:
    - Scale 1: Global structure (1 vector represents entire sequence)
    - Scale 2-8: Coarse patterns
    - Scale 16-L: Fine details

Key Formula:
    f_hat = Σ_k upsample(φ_k(codebook_lookup(downsample(z - f_hat_prev, scale_k))))

Flow Diagram:
    ┌─────────────────────┐
    │ [B, L, D] input z   │
    └────────┬────────────┘
             │ for each scale k ∈ [1, 2, 4, 8, 16, 32]
             ▼
    ┌─────────────────────┐
    │ downsample(z, k)    │ -> [B, k, D]
    │ compute residual    │ -> [B, k, D]  (z_down - f_hat_down)
    │ codebook lookup     │ -> indices [B, k]
    │ apply φ_k           │ -> [B, k, D]
    │ upsample to L       │ -> [B, L, D]
    │ accumulate f_hat    │ -> f_hat += upsampled
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ [B, L, D] f_hat     │ + loss + indices_per_scale
    └─────────────────────┘

Why Multi-Scale?
    1. Hierarchical: Coarse scales capture global, fine scales capture details
    2. Efficient: Fewer codebook entries needed per scale
    3. Generative: Can generate coarse-to-fine (like VAR's next-scale prediction)
"""

from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scale_ops import ScaleOps, AvgPoolScaleOps, build_scale_ops

__all__ = ["MultiScaleQuantizer", "build_quantizer"]


class MultiScaleQuantizer(nn.Module):
    """Multi-Scale Vector Quantizer.

    Args:
        codebook_size: Number of codebook entries V
        codebook_dim: Dimension of codebook vectors C
        scale_lengths: List of sequence lengths [1, 2, 4, ..., L_max]
        beta: Commitment loss weight
        quant_resi: Residual ratio for phi layers
        share_quant_resi: Number of scales sharing phi layer
        scale_ops: ScaleOps instance for down/upsampling (default: AvgPoolScaleOps)

    Input:  [B, L, C] latent features
    Output: [B, L, C] f_hat, loss, indices_per_scale
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        codebook_dim: int = 256,
        scale_lengths: List[int] = [1, 2, 4, 8, 16, 32],
        beta: float = 0.25,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        scale_ops: Optional[ScaleOps] = None,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.scale_lengths = scale_lengths
        self.num_scales = len(scale_lengths)
        self.max_length = max(scale_lengths)
        self.beta = beta
        self.quant_resi = quant_resi

        # Scale operations (pluggable)
        self.scale_ops = scale_ops if scale_ops is not None else AvgPoolScaleOps()

        # Shared codebook: [V, C]
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(
            self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size
        )

        # Phi layers (residual mixing)
        if share_quant_resi > 0 and self.num_scales > share_quant_resi:
            self.num_shared = share_quant_resi
            self.phi_shared = nn.Linear(codebook_dim, codebook_dim)
            self.phi_independent = nn.ModuleList(
                [
                    nn.Linear(codebook_dim, codebook_dim)
                    for _ in range(self.num_scales - share_quant_resi)
                ]
            )
        else:
            self.num_shared = 0
            self.phi_shared = None
            self.phi_independent = nn.ModuleList(
                [nn.Linear(codebook_dim, codebook_dim) for _ in range(self.num_scales)]
            )

    def get_phi(self, scale_idx: int) -> nn.Linear:
        """Get phi layer for given scale index."""
        if self.phi_shared is not None and scale_idx < self.num_shared:
            return self.phi_shared
        idx = scale_idx if self.phi_shared is None else scale_idx - self.num_shared
        return self.phi_independent[idx]

    def quantize(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Quantize at multiple scales with accumulation (following VAR's approach).

        VAR's key insight: maintain f_rest at FULL resolution, update by subtraction.
        - f_rest: residual to be quantized (starts as z, updated each scale)
        - f_hat: accumulated quantized features (starts as zeros)

        Args:
            z: [B, L, C] input features from encoder

        Returns:
            f_hat: [B, L, C] accumulated quantized features
            loss: Scalar VQ loss = β*||f_hat - z||² + ||f_hat - z||²
            indices_per_scale: List of [B, l_k] indices, one per scale

        Dimensions:
            B = batch size
            L = sequence length (maintained at full resolution)
            C = codebook_dim
            l_k = scale_lengths[k] (e.g., 1, 2, 4, 8, 16, 32)
            V = codebook_size

        Flow (for each scale k):
            Step 1: f_rest [B, L, C] -> downsample -> rest_down [B, l_k, C]
            Step 2: rest_down [B, l_k, C] -> codebook lookup -> indices [B, l_k]
            Step 3: indices [B, l_k] -> codebook[indices] -> quantized [B, l_k, C]
            Step 4: quantized [B, l_k, C] -> φ_k -> h_k [B, l_k, C]
            Step 5: h_k [B, l_k, C] -> upsample -> h_k_up [B, L, C]
            Step 6: f_hat += h_k_up, f_rest -= h_k_up

        After loop:
            VQ Loss = β*||f_hat - z||² + ||f_hat - z||²
            STE: f_hat = (f_hat.detach() - z_no_grad) + z

        Reference: third-part/VAR-main/models/quant.py lines 58-98
        """
        B, L, C = z.shape
        device = z.device

        # Initialize f_rest and f_hat at full resolution L
        z_no_grad = z.detach()
        f_rest = z_no_grad.clone()
        f_hat = torch.zeros(B, L, C, device=device, dtype=z.dtype)
        indices_per_scale = []

        # Multi-scale loop
        for k, scale_len in enumerate(self.scale_lengths):
            # Step 1: Downsample f_rest
            rest_down = self.scale_ops.downsample(f_rest, scale_len)

            # Step 2: Find nearest codebook entries
            flat_rest = rest_down.reshape(-1, C)
            distances = (
                flat_rest.pow(2).sum(dim=1, keepdim=True)
                + self.codebook.weight.pow(2).sum(dim=1)
                - 2 * flat_rest @ self.codebook.weight.T
            )
            indices = distances.argmin(dim=1)
            indices_per_scale.append(indices.view(B, scale_len))

            # Step 3-4: Lookup codebook and apply phi
            quantized = self.codebook(indices).view(B, scale_len, C)
            phi = self.get_phi(k)
            h_k = phi(quantized) * self.quant_resi + quantized * (1 - self.quant_resi)

            # Step 5: Upsample to full resolution
            h_k_up = self.scale_ops.upsample(h_k, L)

            # Step 6: Accumulate and update residual
            f_hat = f_hat + h_k_up
            f_rest = f_rest - h_k_up

        # VQ Loss
        commitment_loss = F.mse_loss(f_hat.detach(), z)
        codebook_loss = F.mse_loss(f_hat, z_no_grad)
        vq_loss = commitment_loss * self.beta + codebook_loss

        # Straight-Through Estimator
        f_hat = (f_hat.detach() - z_no_grad) + z

        return f_hat, vq_loss, indices_per_scale

    def decode_indices(
        self,
        indices_per_scale: List[torch.Tensor],
        target_length: int,
    ) -> torch.Tensor:
        """
        Decode indices back to features (reverse of quantize).

        Args:
            indices_per_scale: List of [B, l_k] indices, one per scale
            target_length: Target sequence length L

        Returns:
            f_hat: [B, target_length, C]

        Dimensions:
            B = batch size
            l_k = scale_lengths[k]
            C = codebook_dim

        Flow (for each scale k):
            Step 1: indices [B, l_k] -> codebook[indices] -> quantized [B, l_k, C]
            Step 2: quantized [B, l_k, C] -> φ_k -> h_k [B, l_k, C]
            Step 3: h_k [B, l_k, C] -> upsample -> h_k_up [B, target_length, C]
            Step 4: f_hat += h_k_up
        """
        B = indices_per_scale[0].shape[0]
        device = indices_per_scale[0].device

        # Initialize f_hat
        f_hat = torch.zeros(B, target_length, self.codebook_dim, device=device)

        # Decode each scale and accumulate
        for k, indices in enumerate(indices_per_scale):
            # Step 1: Lookup codebook
            quantized = self.codebook(indices)

            # Step 2: Apply phi layer
            phi = self.get_phi(k)
            h_k = phi(quantized) * self.quant_resi + quantized * (1 - self.quant_resi)

            # Step 3-4: Upsample and accumulate
            h_k_up = self.scale_ops.upsample(h_k, target_length)
            f_hat = f_hat + h_k_up

        return f_hat


def build_quantizer(config: Dict[str, Any]) -> MultiScaleQuantizer:
    """Build quantizer from config.

    Config keys (all required except scale_ops):
        - codebook_size: int
        - codebook_dim: int
        - scale_lengths: list[int]
        - beta: float
        - quant_resi: float
        - share_quant_resi: int
        - scale_ops: dict (optional)
    """
    # Build scale_ops if specified
    scale_ops = None
    if "scale_ops" in config:
        scale_ops = build_scale_ops(config["scale_ops"])

    return MultiScaleQuantizer(
        codebook_size=config["codebook_size"],
        codebook_dim=config["codebook_dim"],
        scale_lengths=config["scale_lengths"],
        beta=config["beta"],
        quant_resi=config["quant_resi"],
        share_quant_resi=config["share_quant_resi"],
        scale_ops=scale_ops,
    )
