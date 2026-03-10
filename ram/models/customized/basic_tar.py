"""Basic building blocks for TAR (Text Auto-Regressive) transformer.

Design pattern inspired by VAR-main/models/basic_var.py.
Implementation is original for text next-scaling task.

Components:
- FFN: Feed-forward network with optional fused MLP
- SelfAttention: Multi-head self-attention with KV caching
- AdaLNSelfAttn: Adaptive LayerNorm + Self-Attention + FFN block
- AdaLNBeforeHead: Adaptive LayerNorm before prediction head

These transformer blocks are used in TAR for next-scale prediction.
The architecture uses Adaptive LayerNorm (AdaLN) conditioning,
which modulates transformer behavior based on scale embeddings.

Tensor dimension conventions:
- B: batch size
- L: sequence length (varies per scale)
- C: embedding dimension
- D: conditioning dimension
- H: number of attention heads
- c: head dimension (C // H)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.regularization import DropPath, drop_path


# This file provides the main transformer blocks used in TAR
__all__ = ["FFN", "SelfAttention", "AdaLNSelfAttn", "AdaLNBeforeHead"]


# ============================================================
# Attempt to import fused operators (optional, for speed)
# These provide optimized implementations when available
# ============================================================
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = (
    flash_attn_func
) = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    pass

try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    pass

try:
    from flash_attn import flash_attn_func
except ImportError:
    pass

try:
    from torch.nn.functional import scaled_dot_product_attention as slow_attn
except ImportError:

    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        """Fallback attention implementation."""
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None:
            attn.add_(attn_mask)
        return (
            F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
            if dropout_p > 0
            else attn.softmax(dim=-1)
        ) @ value


# ============================================================
# FFN (Feed-Forward Network)
# ============================================================
class FFN(nn.Module):
    """
    Feed-forward network with optional fused MLP.

    Structure: Linear -> GELU -> Linear -> Dropout

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (default: in_features)
        out_features: Output dimension (default: in_features)
        drop: Dropout probability
        fused_if_available: Use fused MLP if available

    Shape:
        Input:  (B, L, in_features)
        Output: (B, L, out_features)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0,
        fused_if_available: bool = True,
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_mlp_func is not None:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                    return_residual=False,
                    checkpoint_lvl=0,
                    heuristic=0,
                    process_group=None,
                )
            )
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


# ============================================================
# SelfAttention
# ============================================================
class SelfAttention(nn.Module):
    """
    Multi-head self-attention with optional flash attention and KV caching.

    Supports:
    - Standard attention (torch.nn.functional.scaled_dot_product_attention)
    - Flash attention (when available) for faster training/inference
    - XFormers memory efficient attention (when available)
    - KV caching for efficient autoregressive inference
    - L2 normalized attention (optional)

    Args:
        block_idx: Block index in the transformer
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        attn_drop: Attention dropout probability
        proj_drop: Output projection dropout probability
        attn_l2_norm: Use L2 normalized attention
        flash_if_available: Use flash attention when available

    Shape:
        Input:  (B, L, C) where C = embed_dim
        Output: (B, L, C)

    Attention computation:
        Q, K, V = linear(x)   # (B, L, 3C) -> 3x(B, L, C)
        scores = Q @ K^T / sqrt(head_dim)  # (B, H, L, L)
        attn = softmax(scores + mask) @ V  # (B, H, L, c) -> (B, L, C)
    """

    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_l2_norm: bool = False,
        flash_if_available: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm

        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                requires_grad=True,
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None

        # Only used during inference (KV caching)
        self.caching = False
        self.cached_k = None
        self.cached_v = None

    def kv_caching(self, enable: bool):
        """Enable or disable KV caching for inference."""
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional attention mask.

        Args:
            x: (B, L, C) input features
            attn_bias: (1, 1, L, L) or None attention mask
                       -inf for masked positions, 0 for unmasked
                       None during inference with KV cache

        Returns:
            (B, L, C) output features
        """
        B, L, C = x.shape

        qkv = F.linear(
            input=x,
            weight=self.mat_qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype

        using_flash = (
            self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        )
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)  # q or k or v: BLHc
            dim_cat = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # q or k or v: BHLc
            dim_cat = 2

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
            ).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                attn_bias=(
                    None
                    if attn_bias is None
                    else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
                ),
                p=dropout_p,
                scale=self.scale,
            ).view(B, L, C)
        else:
            oup = (
                slow_attn(
                    query=q,
                    key=k,
                    value=v,
                    scale=self.scale,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                )
                .transpose(1, 2)
                .reshape(B, L, C)
            )

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f"using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}"


# ============================================================
# AdaLNSelfAttn (Adaptive LayerNorm Self-Attention Block)
# ============================================================
class AdaLNSelfAttn(nn.Module):
    """
    Adaptive LayerNorm + Self-Attention + FFN block.

    This is the main transformer block for TAR, using AdaLN for
    scale-conditioned processing. The conditioning produces 6 parameters
    that modulate the block's behavior based on which scale is being predicted.

    AdaLN Mechanism:
        Given conditioning cond_BD (scale or SOS embedding):
        gamma1, gamma2, scale1, scale2, shift1, shift2 = ada_lin(cond_BD)

        For attention: x' = LN(x) * (1 + scale1) + shift1
                       x = x + gamma1 * dropout(attention(x'))

        For FFN:       x'' = LN(x) * (1 + scale2) + shift2
                       x = x + gamma2 * dropout(ffn(x''))

    Args:
        block_idx: Block index in transformer
        embed_dim: Embedding dimension C
        cond_dim: Conditioning dimension D
        shared_aln: If True, use shared parameters across all blocks
        num_heads: Number of attention heads
        mlp_ratio: FFN hidden dimension multiplier
        drop_path: Stochastic depth probability

    Shape:
        x: (B, L, C) input features
        cond_BD: (B, D) or (B, 1, 6, C) conditioning embedding
        Output: (B, L, C)
    """

    def __init__(
        self,
        block_idx: int,
        last_drop_p: float,
        embed_dim: int,
        cond_dim: int,
        shared_aln: bool,
        norm_layer,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        attn_l2_norm: bool = False,
        flash_if_available: bool = False,
        fused_if_available: bool = True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx = block_idx
        self.last_drop_p = last_drop_p
        self.C = embed_dim
        self.D = cond_dim

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = SelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available,
        )
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio),
            drop=drop,
            fused_if_available=fused_if_available,
        )

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(
                torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5
            )
        else:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.fused_add_norm_fn = None

    def forward(
        self, x: torch.Tensor, cond_BD: torch.Tensor, attn_bias: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive modulation.

        Args:
            x: (B, L, C) input features
            cond_BD: (B, D) conditioning embedding for AdaLN
                     or (B, 1, 6, C) pre-computed shared parameters
            attn_bias: (1, 1, L, L) attention mask (None during inference)

        Returns:
            (B, L, C) output features

        Process:
            1. Get AdaLN parameters from conditioning
            2. Apply attention with adaptive modulation
            3. Apply FFN with adaptive modulation
        """
        # Get adaptive parameters from conditioning
        # gamma1/2: output scales, scale1/2: norm scales, shift1/2: norm shifts
        if self.shared_aln:
            # cond_BD is (B, 1, 6, C) from shared_ada_lin
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_gss + cond_BD
            ).unbind(2)
        else:
            # cond_BD is (B, D), compute block-specific parameters
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
            )

        # Attention block with adaptive modulation
        # 1. LayerNorm (no learnable params)
        # 2. Scale and shift: x' = LN(x) * (1 + scale1) + shift1
        # 3. Self-attention
        # 4. Scale output: gamma1 * attn_output
        # 5. Residual + drop path
        x = x + self.drop_path(
            self.attn(
                self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),  # AdaLN modulation
                attn_bias=attn_bias,
            ).mul_(
                gamma1
            )  # Output scaling
        )

        # FFN block with adaptive modulation
        # Same process: LN -> scale+shift -> FFN -> scale -> residual
        x = x + self.drop_path(
            self.ffn(
                self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)  # AdaLN modulation
            ).mul(
                gamma2
            )  # Output scaling
        )

        return x

    def extra_repr(self) -> str:
        return f"shared_aln={self.shared_aln}"


# ============================================================
# AdaLNBeforeHead (Adaptive LayerNorm before prediction head)
# ============================================================
class AdaLNBeforeHead(nn.Module):
    """
    Adaptive LayerNorm applied before the prediction head.

    Provides final normalization with scale conditioning before
    projecting to codebook logits.

    Args:
        C: Embedding dimension
        D: Conditioning dimension
        norm_layer: Normalization layer constructor

    Shape:
        x_BLC: (B, L, C) input features
        cond_BD: (B, D) conditioning embedding
        Output: (B, L, C) normalized and modulated features
    """

    def __init__(self, C: int, D: int, norm_layer):
        """
        Args:
            C: Embedding dimension
            D: Conditioning dimension
        """
        super().__init__()
        self.C = C
        self.D = D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive normalization before head.

        Args:
            x_BLC: (B, L, C) transformer output features
            cond_BD: (B, D) scale/SOS conditioning embedding

        Returns:
            (B, L, C) normalized and modulated features

        Process:
            1. Compute scale, shift from conditioning: (B, D) -> (B, 1, 2, C)
            2. Apply LayerNorm (no learnable params)
            3. Modulate: output = LN(x) * (1 + scale) + shift
        """
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
