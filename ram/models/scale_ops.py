"""Scale Operations for Multi-Scale Quantization.

Provides downsampling and upsampling operations for 1D text sequences.
Extracted as separate module for easy modification, testing, and swapping.

VAR (2D Image) uses:
    - Downsample: F.interpolate(mode='area')
    - Upsample: F.interpolate(mode='bicubic')

TAR (1D Text) uses:
    - Downsample: F.adaptive_avg_pool1d or F.interpolate(mode='linear')
    - Upsample: F.interpolate(mode='linear')
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ScaleOps", "AvgPoolScaleOps", "LinearScaleOps", "build_scale_ops"]


class ScaleOps(nn.Module):
    """Base class for scale operations.

    Subclass this to implement different downsampling/upsampling strategies.
    """

    def downsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Downsample sequence to target length.

        Args:
            x: [B, L, C] input features
            target_len: Target sequence length

        Returns:
            [B, target_len, C] downsampled features
        """
        raise NotImplementedError

    def upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Upsample sequence to target length.

        Args:
            x: [B, l, C] input features
            target_len: Target sequence length

        Returns:
            [B, target_len, C] upsampled features
        """
        raise NotImplementedError


class AvgPoolScaleOps(ScaleOps):
    """Scale operations using adaptive average pooling.

    Downsample: F.adaptive_avg_pool1d (area-preserving)
    Upsample: F.interpolate(mode='linear')

    This is the default for TAR, analogous to VAR's area/bicubic for images.
    """

    def __init__(self, align_corners: bool = False):
        super().__init__()
        self.align_corners = align_corners

    def downsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Downsample using adaptive average pooling.

        Args:
            x: [B, L, C]
            target_len: int

        Returns:
            [B, target_len, C]
        """
        B, L, C = x.shape
        if L == target_len:
            return x
        # [B, L, C] -> [B, C, L] -> pool -> [B, C, target_len] -> [B, target_len, C]
        return F.adaptive_avg_pool1d(x.transpose(1, 2), target_len).transpose(1, 2)

    def upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Upsample using linear interpolation.

        Args:
            x: [B, l, C]
            target_len: int

        Returns:
            [B, target_len, C]
        """
        B, l, C = x.shape
        if l == target_len:
            return x
        # [B, l, C] -> [B, C, l] -> interpolate -> [B, C, target_len] -> [B, target_len, C]
        return F.interpolate(
            x.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=self.align_corners,
        ).transpose(1, 2)


class LinearScaleOps(ScaleOps):
    """Scale operations using linear interpolation for both directions.

    Downsample: F.interpolate(mode='linear')
    Upsample: F.interpolate(mode='linear')

    Alternative to AvgPool - may preserve more structure.
    """

    def __init__(self, align_corners: bool = False):
        super().__init__()
        self.align_corners = align_corners

    def downsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Downsample using linear interpolation.

        Args:
            x: [B, L, C]
            target_len: int

        Returns:
            [B, target_len, C]
        """
        B, L, C = x.shape
        if L == target_len:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=self.align_corners,
        ).transpose(1, 2)

    def upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Upsample using linear interpolation.

        Args:
            x: [B, l, C]
            target_len: int

        Returns:
            [B, target_len, C]
        """
        B, l, C = x.shape
        if l == target_len:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=self.align_corners,
        ).transpose(1, 2)


class LearnedScaleOps(ScaleOps):
    """Learnable scale operations using Conv1d.

    Downsample: Strided Conv1d
    Upsample: TransposedConv1d or Conv1d + interpolate

    More expressive but adds parameters.
    """

    def __init__(self, dim: int, max_scales: int = 6):
        super().__init__()
        self.dim = dim
        # Learnable projections for each scale transition
        self.down_projs = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=3, padding=1) for _ in range(max_scales)]
        )
        self.up_projs = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=3, padding=1) for _ in range(max_scales)]
        )

    def downsample(
        self, x: torch.Tensor, target_len: int, scale_idx: int = 0
    ) -> torch.Tensor:
        """
        Downsample with learned projection.

        Args:
            x: [B, L, C]
            target_len: int
            scale_idx: Which scale (for selecting projection)

        Returns:
            [B, target_len, C]
        """
        B, L, C = x.shape
        if L == target_len:
            return x
        # First interpolate, then apply learned projection
        x_down = F.adaptive_avg_pool1d(x.transpose(1, 2), target_len)
        x_down = self.down_projs[scale_idx](x_down)
        return x_down.transpose(1, 2)

    def upsample(
        self, x: torch.Tensor, target_len: int, scale_idx: int = 0
    ) -> torch.Tensor:
        """
        Upsample with learned projection.

        Args:
            x: [B, l, C]
            target_len: int
            scale_idx: Which scale (for selecting projection)

        Returns:
            [B, target_len, C]
        """
        B, l, C = x.shape
        if l == target_len:
            return x
        # First interpolate, then apply learned projection
        x_up = F.interpolate(
            x.transpose(1, 2), size=target_len, mode="linear", align_corners=False
        )
        x_up = self.up_projs[scale_idx](x_up)
        return x_up.transpose(1, 2)


def build_scale_ops(config: dict) -> ScaleOps:
    """Build scale operations from config.

    Config (all required for each type):
        type: 'avgpool' | 'linear' | 'learned'

        For avgpool/linear:
            align_corners: bool

        For learned:
            dim: int
            max_scales: int
    """
    op_type = config["type"]

    if op_type == "avgpool":
        return AvgPoolScaleOps(align_corners=config["align_corners"])
    elif op_type == "linear":
        return LinearScaleOps(align_corners=config["align_corners"])
    elif op_type == "learned":
        return LearnedScaleOps(
            dim=config["dim"],
            max_scales=config["max_scales"],
        )
    else:
        raise ValueError(f"Unknown scale_ops type: {op_type}")
