"""Multi-Scale Residual Vector Quantizer for Text Sequences.

Design pattern inspired by VAR-main/models/quant.py.
Implementation is original for text next-scaling task.

Core Text-VAR Mechanism (Multi-Scale Feature Accumulation):
=============================================================
This is the key innovation from VAR adapted for text:

1. Start with encoder output f_BCL (B=batch, C=latent_dim, L=max_seq_len)
   Initialize: f_rest = f_BCL.clone(), f_hat = zeros_like(f_BCL)

2. For each scale k from coarsest (length=1) to finest (length=L):
   a) Downsample f_rest to current scale: f_rest -> (B, C, scale_len_k)
   b) Find nearest codebook embedding for each position
   c) Upsample embedding back to max length L: (B, C, scale_len_k) -> (B, C, L)
   d) Apply phi transformation: h_k = phi_k(upsampled_embedding)
   e) ACCUMULATE: f_hat = f_hat + h_k  (Element-wise summation!)
   f) Update residual: f_rest = f_rest - h_k

3. Final f_hat contains accumulated multi-scale features:
   f_hat = sum_k(upsample_to_L(phi_k(codebook_lookup(downsample(f_rest, scale_k)))))

Text-specific design:
- Uses 1D linear interpolation for up/downsampling (not 2D bilinear)
- Scales are sequence lengths: (1, 2, 4, 8, 16, 32) for L=32
- Coarse scales capture global semantics, fine scales capture local details

Tensor dimension conventions:
- B: batch size
- C: latent channel dimension (Cvae, e.g., 32)
- L: maximum sequence length (finest scale)
- pl: patch length at current scale (varies from 1 to L)
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VectorQuantizer2"]


# ============================================================
# Phi (φ) Layers for Residual Blending
# Design pattern inspired by VAR's quant.py
# ============================================================


class Phi(nn.Conv1d):
    """
    Phi (φ) transformation for residual blending at each scale.

    φ(h) = (1 - resi_ratio) * h + resi_ratio * conv(h)

    This learned blending allows the model to control how much of the
    original vs transformed features contribute at each scale.

    Args:
        embed_dim: Latent embedding dimension (Cvae)
        quant_resi: Residual blending ratio (0-1)

    Shape:
        Input:  (B, C, L) - scale features upsampled to max length
        Output: (B, C, L) - blended features for accumulation
    """

    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=ks,
            stride=1,
            padding=ks // 2,
        )
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BCL):
        # h_BCL: (B, C, L) - upsampled scale features
        # Output: (B, C, L) - blended: (1-r)*h + r*conv(h)
        return h_BCL.mul(1 - self.resi_ratio) + super().forward(h_BCL).mul_(
            self.resi_ratio
        )


class PhiShared(nn.Module):
    """Fully shared φ for all scales - one φ layer used across all K scales."""

    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    """Partially shared φ layers - N φ layers shared among K scales."""

    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class PhiNonShared(nn.ModuleList):
    """Non-shared φ layers - one unique φ per scale."""

    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(
            np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()
        )

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


# ============================================================
# VectorQuantizer2: Multi-Scale Residual Vector Quantizer
# Design pattern inspired by VAR's VectorQuantizer2
# ============================================================


class VectorQuantizer2(nn.Module):
    """
    Multi-scale residual vector quantizer for text sequences.

    This is the core component implementing VAR's multi-scale feature accumulation
    for text. Instead of VAR's 2D image patches, we use 1D sequence scales.

    Key Concept (Multi-Scale Feature Summation):
    =============================================
    For text length L=32 with scales (1, 2, 4, 8, 16, 32):

    Scale 1 (len=1):   [global]        ->  upsample to L  ->  f_hat += h_1
    Scale 2 (len=2):   [coarse]        ->  upsample to L  ->  f_hat += h_2
    Scale 3 (len=4):   [paragraph]     ->  upsample to L  ->  f_hat += h_3
    Scale 4 (len=8):   [sentence]      ->  upsample to L  ->  f_hat += h_4
    Scale 5 (len=16):  [phrase]        ->  upsample to L  ->  f_hat += h_5
    Scale 6 (len=32):  [token-level]   ->  (no upsample)  ->  f_hat += h_6

    Final f_hat = h_1 + h_2 + h_3 + h_4 + h_5 + h_6  (all at length L)

    This summation fuses global structure (scale 1) with local details (scale 6).

    Args:
        vocab_size: Codebook size (K) - number of discrete codes
        Cvae: Latent channel dimension (z_channels)
        using_znorm: Whether to use z-normalization for codebook lookup
        beta: Commitment loss weight for VQ training
        v_patch_lens: Sequence lengths at each scale, e.g., (1, 2, 4, 8, 16, 32)
        quant_resi: Residual ratio for φ layers (0-1)
        share_quant_resi: 0=non-shared, 1=fully shared, N=partially shared (N φ layers)
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        Cvae: int = 32,
        using_znorm: bool = False,
        beta: float = 0.25,
        default_qresi_counts: int = 0,
        v_patch_lens: Tuple[int, ...] = (
            1,
            2,
            4,
            8,
            16,
            32,
        ),  # For text: sequence lengths
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.Cvae = Cvae
        self.using_znorm = using_znorm
        self.v_patch_lens: Tuple[int, ...] = v_patch_lens

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # Non-shared: φ_{1 to K} for K scales
            self.quant_resi = PhiNonShared(
                [
                    (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                    for _ in range(default_qresi_counts or len(self.v_patch_lens))
                ]
            )
        elif share_quant_resi == 1:  # Fully shared: single φ for all scales
            self.quant_resi = PhiShared(
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            )
        else:  # Partially shared: share_quant_resi φ layers for K scales
            self.quant_resi = PhiPartiallyShared(
                nn.ModuleList(
                    [
                        (
                            Phi(Cvae, quant_resi)
                            if abs(quant_resi) > 1e-6
                            else nn.Identity()
                        )
                        for _ in range(share_quant_resi)
                    ]
                )
            )

        self.register_buffer(
            "ema_vocab_hit_SV",
            torch.full((len(self.v_patch_lens), self.vocab_size), fill_value=0.0),
        )
        self.record_hit = 0

        self.beta = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

    def extra_repr(self) -> str:
        return f"{self.v_patch_lens}, znorm={self.using_znorm}, beta={self.beta} | S={len(self.v_patch_lens)}, quant_resi={self.quant_resi_ratio}"

    # ===================== forward: Multi-Scale VQ for training =====================
    def forward(
        self, f_BCL: torch.Tensor, ret_usages: bool = False
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        Multi-scale residual quantization with feature accumulation.

        This is the core Text-VAR forward pass implementing:
        f_hat = sum_k(upsample_to_L(phi_k(codebook[nearest(downsample(f, scale_k))])))]

        Args:
            f_BCL: (B, C, L) encoder output
                   B = batch size
                   C = latent dimension (Cvae, e.g., 32)
                   L = max sequence length (e.g., 32)
            ret_usages: Whether to return codebook usage statistics

        Returns:
            f_hat: (B, C, L) accumulated quantized features
                   This is the SUMMATION of all scale contributions
            usages: List of codebook usage percentages per scale
            mean_vq_loss: Average VQ loss across scales

        Algorithm:
            1. Initialize f_rest = f_BCL (residual to encode)
            2. Initialize f_hat = zeros (accumulated features)
            3. For each scale k in [1, 2, 4, 8, 16, 32]:
               a) downsample f_rest to length k: (B, C, L) -> (B, C, k)
               b) find nearest codebook: (B, C, k) -> (B, k) indices
               c) lookup embeddings: (B, k) -> (B, C, k)
               d) upsample to L: (B, C, k) -> (B, C, L)
               e) apply phi: h_k = phi_k(upsampled)
               f) ACCUMULATE: f_hat = f_hat + h_k
               g) update residual: f_rest = f_rest - h_k
            4. Return f_hat = h_1 + h_2 + ... + h_K
        """
        dtype = f_BCL.dtype
        if dtype != torch.float32:
            f_BCL = f_BCL.float()
        B, C, L = f_BCL.shape
        f_no_grad = f_BCL.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(
                self.vocab_size, dtype=torch.float, device=f_BCL.device
            )
            SN = len(self.v_patch_lens)

            for si, pl in enumerate(self.v_patch_lens):  # From small to large
                # Downsample f_rest to current scale length pl
                # Linear interpolation for 1D text (not 2D bilinear like images)
                if si != SN - 1:
                    # Intermediate scales: downsample f_rest from L to pl
                    rest_NC = F.interpolate(
                        f_rest, size=pl, mode="linear", align_corners=False
                    )
                    # Reshape for codebook lookup: (B, C, pl) -> (B*pl, C)
                    rest_NC = rest_NC.permute(0, 2, 1).reshape(-1, C)  # (B*pl, C)
                else:
                    # Final scale (pl == L): no downsampling needed
                    rest_NC = f_rest.permute(0, 2, 1).reshape(-1, C)  # (B*L, C)

                # Find nearest codebook embedding for each position
                # Using L2 distance: argmin_j ||z - e_j||^2
                if self.using_znorm:
                    # Z-normalized lookup: use cosine similarity
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(
                        rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0),
                        dim=1,
                    )
                else:
                    # Standard L2 distance lookup
                    # d[i,j] = ||rest[i] - emb[j]||^2 = ||rest||^2 + ||emb||^2 - 2*rest@emb^T
                    d_no_grad = torch.sum(
                        rest_NC.square(), dim=1, keepdim=True
                    ) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False
                    )
                    d_no_grad.addmm_(
                        rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                    )
                    idx_N = torch.argmin(d_no_grad, dim=1)  # (B*pl,) nearest indices

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()

                # Lookup embeddings and upsample to max length L
                idx_BL = idx_N.view(B, -1)  # (B, pl) index tensor
                h_BCpl = self.embedding(idx_BL).permute(
                    0, 2, 1
                )  # (B, C, pl) embeddings

                if si != SN - 1:
                    # Upsample from scale length pl to max length L
                    # This enables feature accumulation at the same resolution
                    h_BCL = F.interpolate(
                        h_BCpl, size=L, mode="linear", align_corners=False
                    )
                else:
                    # Final scale: already at max length, no upsampling
                    h_BCL = h_BCpl

                # Apply phi transformation for residual blending
                # phi_k(h) = (1-r)*h + r*conv(h) - learned blending
                h_BCL = self.quant_resi[si / (SN - 1)](h_BCL)  # (B, C, L)

                # ACCUMULATE: f_hat = f_hat + h_k (core VAR summation)
                # All scales contribute additively at max resolution L
                f_hat = f_hat + h_BCL  # (B, C, L) accumulated

                # Update residual: what remains to encode at finer scales
                f_rest = f_rest - h_BCL  # (B, C, L) residual

                # Update EMA statistics
                if self.training:
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1

                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BCL).mul_(
                    self.beta
                ) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1.0 / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BCL)  # Straight-through estimator

        margin = (f_BCL.numel() / f_BCL.shape[1]) / self.vocab_size * 0.08
        if ret_usages:
            usages = [
                (self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100
                for si, pl in enumerate(self.v_patch_lens)
            ]
        else:
            usages = None

        return f_hat, usages, mean_vq_loss

    # ===================== embed_to_fhat: Convert embeddings to accumulated f_hat =====================
    def embed_to_fhat(
        self,
        ms_h_BCl: List[torch.Tensor],
        all_to_max_scale: bool = True,
        last_one: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Convert multi-scale embeddings to accumulated f_hat.
        Used during inference after LLM generates codebook indices.

        Args:
            ms_h_BCl: List of embeddings at each scale
                      [(B, C, 1), (B, C, 2), (B, C, 4), ..., (B, C, L)]
            all_to_max_scale: If True, upsample all to max scale L
            last_one: If True, return only final accumulated f_hat

        Returns:
            If last_one: (B, C, L) final accumulated features
            Else: List of (B, C, L) accumulated f_hat at each scale step

        Process:
            f_hat_1 = upsample(h_1)                          # scale 1 only
            f_hat_2 = upsample(h_1) + upsample(h_2)          # scales 1+2
            f_hat_3 = upsample(h_1) + upsample(h_2) + upsample(h_3)  # scales 1+2+3
            ...
            f_hat_K = sum of all upsampled scale embeddings
        """
        ls_f_hat_BCL = []
        B = ms_h_BCl[0].shape[0]
        L = self.v_patch_lens[-1]
        SN = len(self.v_patch_lens)

        if all_to_max_scale:
            f_hat = ms_h_BCl[0].new_zeros(B, self.Cvae, L, dtype=torch.float32)
            for si, pl in enumerate(self.v_patch_lens):
                h_BCl = ms_h_BCl[si]
                if si < len(self.v_patch_lens) - 1:
                    h_BCl = F.interpolate(
                        h_BCl, size=L, mode="linear", align_corners=False
                    )
                h_BCl = self.quant_resi[si / (SN - 1)](h_BCl)
                f_hat.add_(h_BCl)
                if last_one:
                    ls_f_hat_BCL = f_hat
                else:
                    ls_f_hat_BCL.append(f_hat.clone())
        else:
            f_hat = ms_h_BCl[0].new_zeros(
                B, self.Cvae, self.v_patch_lens[0], dtype=torch.float32
            )
            for si, pl in enumerate(self.v_patch_lens):
                f_hat = F.interpolate(
                    f_hat, size=pl, mode="linear", align_corners=False
                )
                h_BCl = self.quant_resi[si / (SN - 1)](ms_h_BCl[si])
                f_hat.add_(h_BCl)
                if last_one:
                    ls_f_hat_BCL = f_hat
                else:
                    ls_f_hat_BCL.append(f_hat)

        return ls_f_hat_BCL

    # ===================== f_to_idxBl_or_fhat: Encode to indices or f_hat =====================
    def f_to_idxBl_or_fhat(
        self,
        f_BCL: torch.Tensor,
        to_fhat: bool,
        v_patch_lens: Optional[Sequence[int]] = None,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:
        """
        Encode input to multi-scale indices or accumulated f_hat at each scale.
        Used for preparing training data or visualization.

        Args:
            f_BCL: (B, C, L) encoder output features
            to_fhat: If True, return accumulated f_hat at each scale
                     If False, return codebook indices at each scale
            v_patch_lens: Custom scale lengths (default: self.v_patch_lens)

        Returns:
            If to_fhat: List of (B, C, L) accumulated f_hat after each scale
            If not to_fhat: List of (B, pl) index tensors for each scale
        """
        B, C, L = f_BCL.shape
        f_no_grad = f_BCL.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []
        patch_lens = list(v_patch_lens or self.v_patch_lens)
        assert patch_lens[-1] == L, f"{patch_lens[-1]=} != {L=}"

        SN = len(patch_lens)
        for si, pl in enumerate(patch_lens):
            # Downsample f_rest
            if si != SN - 1:
                z_NC = F.interpolate(
                    f_rest, size=pl, mode="linear", align_corners=False
                )
                z_NC = z_NC.permute(0, 2, 1).reshape(-1, C)
            else:
                z_NC = f_rest.permute(0, 2, 1).reshape(-1, C)

            # Find nearest embedding
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(
                    z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1
                )
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False
                )
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)

            # Lookup and upsample
            idx_Bl = idx_N.view(B, pl)
            h_BCpl = self.embedding(idx_Bl).permute(0, 2, 1)

            if si != SN - 1:
                h_BCL = F.interpolate(
                    h_BCpl, size=L, mode="linear", align_corners=False
                ).contiguous()
            else:
                h_BCL = h_BCpl.contiguous()

            h_BCL = self.quant_resi[si / (SN - 1)](h_BCL)
            f_hat.add_(h_BCL)
            f_rest.sub_(h_BCL)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, pl))

        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: Prepare TAR training input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert ground-truth indices to TAR training input (teacher forcing).

        For next-scale prediction training, the TAR model receives accumulated
        f_hat from previous scales as conditioning to predict next scale.

        Args:
            gt_ms_idx_Bl: List of ground-truth indices at each scale
                          [(B, 1), (B, 2), (B, 4), ..., (B, L)]

        Returns:
            (B, total_tokens, C) concatenated embeddings for teacher forcing
            where total_tokens = sum(scale_lens[1:]) (excludes first scale)

        Process for scales [1, 2, 4, 8, 16, 32]:
            1. Look up scale 1 indices -> upsample -> f_hat_1
            2. Downsample f_hat_1 to scale 2 length -> input for predicting scale 2
            3. Look up scale 2 indices -> add to f_hat -> f_hat_2
            4. Downsample f_hat_2 to scale 3 length -> input for predicting scale 3
            ... and so on
        """
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        L = self.v_patch_lens[-1]
        SN = len(self.v_patch_lens)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, L, dtype=torch.float32)
        pl_next = self.v_patch_lens[0]

        for si in range(SN - 1):
            h_BCpl = self.embedding(gt_ms_idx_Bl[si]).transpose(1, 2)  # (B, C, pl)
            h_BCL = F.interpolate(h_BCpl, size=L, mode="linear", align_corners=False)
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BCL))

            pl_next = self.v_patch_lens[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=pl_next, mode="linear", align_corners=False)
                .view(B, C, -1)
                .transpose(1, 2)  # (B, pl_next, C)
            )

        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    # ===================== get_next_autoregressive_input: TAR inference =====================
    def get_next_autoregressive_input(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BCl: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Update accumulated f_hat and prepare input for next scale prediction.
        Used during TAR autoregressive inference.

        Args:
            si: Current scale index (0 to SN-1)
            SN: Total number of scales
            f_hat: (B, C, L) current accumulated features
            h_BCl: (B, C, pl) current scale embeddings from prediction

        Returns:
            f_hat: Updated accumulated features with current scale added
            next_input: Downsampled f_hat for next scale prediction
                       Shape (B, C, next_scale_len)

        Process:
            1. Upsample h_BCl from current scale to max length L
            2. Apply phi transformation
            3. Add to f_hat: f_hat = f_hat + phi(upsample(h))
            4. Downsample f_hat to next scale length for conditioning
        """
        L = self.v_patch_lens[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BCl, size=L, mode="linear", align_corners=False)
            )
            f_hat.add_(h)
            return f_hat, F.interpolate(
                f_hat,
                size=self.v_patch_lens[si + 1],
                mode="linear",
                align_corners=False,
            )
        else:
            h = self.quant_resi[si / (SN - 1)](h_BCl)
            f_hat.add_(h)
            return f_hat, f_hat
