"""
Text VAE basic building blocks.

Mirrors: third-part/VAR-main/models/basic_vae.py
- Encoder: Conv1D-based encoder for text sequences
- Decoder: Transformer-based decoder for text reconstruction

Key difference from VAR:
- VAR uses Conv2D for 2D images (H, W)
- We use Conv1D for 1D text sequences (L)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Encoder', 'Decoder']


# ============================================================
# Basic Building Blocks (mirror VAR's basic_vae.py)
# ============================================================

def Normalize1D(in_channels, num_groups=32):
    """GroupNorm for 1D sequences. Mirror: basic_vae.py line 18-19"""
    return nn.GroupNorm(
        num_groups=min(num_groups, in_channels), 
        num_channels=in_channels, 
        eps=1e-6, 
        affine=True
    )


class Downsample1D(nn.Module):
    """
    1D downsampling with strided convolution.
    Mirror: basic_vae.py Downsample2x (lines 31-37)
    
    VAR uses Conv2D with stride=2, we use Conv1D with stride=2.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        # x: (B, C, L)
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    1D upsampling with interpolation + conv.
    Mirror: basic_vae.py Upsample2x (lines 22-28)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # x: (B, C, L)
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class ResnetBlock1D(nn.Module):
    """
    1D ResNet block.
    Mirror: basic_vae.py ResnetBlock (lines 40-60)
    
    Structure: norm1 → silu → conv1 → norm2 → dropout → silu → conv2 + shortcut
    """
    def __init__(self, *, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = Normalize1D(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize1D(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x):
        # x: (B, C, L)
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


class AttnBlock1D(nn.Module):
    """
    1D self-attention block.
    Mirror: basic_vae.py AttnBlock (lines 63-92)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels
        
        self.norm = Normalize1D(in_channels)
        self.qkv = nn.Conv1d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # x: (B, C, L)
        qkv = self.qkv(self.norm(x))
        B, _, L = qkv.shape
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, L).unbind(1)
        
        # Compute attention: (B, L, L)
        q = q.permute(0, 2, 1).contiguous()  # (B, L, C)
        k = k.contiguous()  # (B, C, L)
        w = torch.bmm(q, k).mul_(self.w_ratio)  # (B, L, L)
        w = F.softmax(w, dim=2)
        
        # Attend to values
        v = v.contiguous()  # (B, C, L)
        w = w.permute(0, 2, 1).contiguous()  # (B, L, L)
        h = torch.bmm(v, w)  # (B, C, L)
        
        return x + self.proj_out(h)


def make_attn(in_channels, using_sa=True):
    """Mirror: basic_vae.py lines 95-96"""
    return AttnBlock1D(in_channels) if using_sa else nn.Identity()


# ============================================================
# Encoder (Mirror: basic_vae.py lines 99-160)
# ============================================================

class Encoder(nn.Module):
    """
    1D Encoder for text sequences.
    Mirror: basic_vae.py Encoder (lines 99-160)
    
    Input: (B, in_channels, L) - embedded text
    Output: (B, z_channels, L // downsample_ratio)
    
    For text: in_channels = embed_dim (e.g., 768)
              z_channels = latent_dim (e.g., 32)
              downsample_ratio = 2^(num_resolutions-1) (e.g., 16)
    """
    def __init__(
        self, *,
        ch=128,                    # Base channel count
        ch_mult=(1, 2, 4, 8),      # Channel multipliers per resolution
        num_res_blocks=2,          # ResNet blocks per resolution
        dropout=0.0,
        in_channels=768,           # Input: embedding dimension
        z_channels=32,             # Output: latent dimension
        using_sa=True,             # Use self-attention at lowest resolution
        using_mid_sa=True,         # Use self-attention in middle block
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # Input projection: embed_dim → ch
        self.conv_in = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        
        # Downsampling blocks
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                # Self-attention only at lowest resolution
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1D(block_in)
            self.down.append(down)
        
        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # Output projection: block_in → z_channels
        self.norm_out = Normalize1D(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, L) or (B, L, in_channels)
            
        Returns:
            (B, z_channels, L // downsample_ratio)
        """
        # Handle (B, L, D) input format
        if x.dim() == 3 and x.size(2) == self.in_channels:
            x = x.transpose(1, 2)  # (B, L, D) → (B, D, L)
        
        # Downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        
        # Middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        
        # Output
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h  # (B, z_channels, L // downsample_ratio)


# ============================================================
# Decoder (Mirror: basic_vae.py lines 163-226)
# ============================================================

class Decoder(nn.Module):
    """
    1D Decoder for text reconstruction.
    Mirror: basic_vae.py Decoder (lines 163-226)
    
    For text, we use Conv1D + final projection to vocab.
    
    Input: (B, z_channels, L_latent) - latent features
    Output: (B, vocab_size, L) - token logits
    """
    def __init__(
        self, *,
        ch=128,                    # Base channel count
        ch_mult=(1, 2, 4, 8),      # Channel multipliers per resolution
        num_res_blocks=2,          # ResNet blocks per resolution
        dropout=0.0,
        z_channels=32,             # Input: latent dimension
        out_channels=768,          # Output dimension before vocab projection
        vocab_size=32000,          # Vocabulary size for final projection
        using_sa=True,             # Use self-attention at lowest resolution
        using_mid_sa=True,         # Use self-attention in middle block
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.vocab_size = vocab_size
        
        # Compute block_in at lowest resolution
        block_in = ch * ch_mult[self.num_resolutions - 1]
        
        # Input projection: z_channels → block_in
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1D(block_in)
            self.up.insert(0, up)
        
        # Output projection
        self.norm_out = Normalize1D(block_in)
        self.conv_out = nn.Conv1d(block_in, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Vocabulary projection
        self.vocab_proj = nn.Linear(out_channels, vocab_size)
    
    def forward(self, z):
        """
        Args:
            z: (B, z_channels, L_latent) - latent features (accumulated f_hat)
            
        Returns:
            (B, L, vocab_size) - token logits
        """
        # Input projection + middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        
        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # Output
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))  # (B, out_channels, L)
        h = h.transpose(1, 2)  # (B, L, out_channels)
        logits = self.vocab_proj(h)  # (B, L, vocab_size)
        
        return logits
