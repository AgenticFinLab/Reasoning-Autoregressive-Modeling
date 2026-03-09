"""
Vector Quantizer for Text sequences.

Mirrors: third-part/VAR-main/models/quant.py
- VectorQuantizer2: Multi-scale residual quantization

Key VAR mechanism (quant.py lines 52-104):
1. Start with encoder output f_BChw and f_rest = f.clone()
2. For each scale (small to large):
   - Downsample f_rest to current scale
   - Find nearest codebook embedding
   - Upsample embedding back to max resolution
   - Apply φ transformation
   - Accumulate: f_hat = f_hat + h
   - Update residual: f_rest = f_rest - h
3. Final f_hat contains accumulated multi-scale features

For text (1D):
- v_patch_nums becomes sequence length patches
- F.interpolate with mode='linear' for 1D
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VectorQuantizer2']


# ============================================================
# Phi layers (Mirror: quant.py lines 199-243)
# ============================================================

class Phi(nn.Conv1d):
    """
    Phi (φ) transformation for residual blending.
    Mirror: quant.py Phi (lines 199-206)
    
    φ(h) = (1 - resi_ratio) * h + resi_ratio * conv(h)
    """
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BCL):
        return h_BCL.mul(1 - self.resi_ratio) + super().forward(h_BCL).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    """Fully shared φ for all scales. Mirror: quant.py lines 209-215"""
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    """Partially shared φ layers. Mirror: quant.py lines 218-229"""
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    """Non-shared φ layers (one per scale). Mirror: quant.py lines 232-243"""
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


# ============================================================
# VectorQuantizer2 (Mirror: quant.py lines 15-196)
# ============================================================

class VectorQuantizer2(nn.Module):
    """
    Multi-scale residual vector quantizer.
    Mirror: quant.py VectorQuantizer2 (lines 15-196)
    
    Key difference: Operates on 1D sequences instead of 2D images.
    - VAR: (B, C, H, W) with v_patch_nums = (1, 2, 3, 4, ..., 16)
    - Text: (B, C, L) with v_patch_lens = (1, 2, 4, 8, ..., L)
    
    Args:
        vocab_size: Codebook size (K)
        Cvae: Latent channel dimension (z_channels)
        using_znorm: Whether to use z-normalization
        beta: Commitment loss weight
        v_patch_lens: Sequence length at each scale (e.g., (1, 2, 4, 8, 16, 32))
        quant_resi: Residual ratio for φ layers
        share_quant_resi: 0=non-shared, 1=fully shared, N=partially shared (N φ layers)
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        Cvae: int = 32,
        using_znorm: bool = False,
        beta: float = 0.25,
        default_qresi_counts: int = 0,
        v_patch_lens: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),  # For text: sequence lengths
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
            self.quant_resi = PhiNonShared([
                (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                for _ in range(default_qresi_counts or len(self.v_patch_lens))
            ])
        elif share_quant_resi == 1:  # Fully shared: single φ for all scales
            self.quant_resi = PhiShared(
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            )
        else:  # Partially shared: share_quant_resi φ layers for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([
                (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                for _ in range(share_quant_resi)
            ]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_lens), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_lens}, znorm={self.using_znorm}, beta={self.beta} | S={len(self.v_patch_lens)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== forward: used in VQVAE training =====================
    def forward(self, f_BCL: torch.Tensor, ret_usages: bool = False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        Multi-scale residual quantization.
        Mirror: quant.py forward (lines 52-104)
        
        Args:
            f_BCL: (B, C, L) encoder output
            ret_usages: Whether to return codebook usage statistics
            
        Returns:
            f_hat: (B, C, L) accumulated quantized features
            usages: List of codebook usage percentages per scale
            mean_vq_loss: Average VQ loss across scales
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
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BCL.device)
            SN = len(self.v_patch_lens)
            
            for si, pl in enumerate(self.v_patch_lens):  # From small to large
                # Downsample f_rest to current scale
                if si != SN - 1:
                    rest_NC = F.interpolate(f_rest, size=pl, mode='linear', align_corners=False)
                    rest_NC = rest_NC.permute(0, 2, 1).reshape(-1, C)  # (B*pl, C)
                else:
                    rest_NC = f_rest.permute(0, 2, 1).reshape(-1, C)  # (B*L, C)
                
                # Find nearest embedding
                if self.using_znorm:
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + \
                                torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                
                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                
                # Lookup and upsample
                idx_BL = idx_N.view(B, -1)  # (B, pl) or (B, L)
                h_BCpl = self.embedding(idx_BL).permute(0, 2, 1)  # (B, C, pl)
                
                if si != SN - 1:
                    # Upsample to original length L
                    h_BCL = F.interpolate(h_BCpl, size=L, mode='linear', align_corners=False)
                else:
                    h_BCL = h_BCpl
                
                # Apply φ transformation
                h_BCL = self.quant_resi[si / (SN - 1)](h_BCL)
                
                # Accumulate and update residual
                f_hat = f_hat + h_BCL
                f_rest = f_rest - h_BCL
                
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
                mean_vq_loss += F.mse_loss(f_hat.data, f_BCL).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1.0 / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BCL)  # Straight-through estimator
        
        margin = (f_BCL.numel() / f_BCL.shape[1]) / self.vocab_size * 0.08
        if ret_usages:
            usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 
                      for si, pl in enumerate(self.v_patch_lens)]
        else:
            usages = None
        
        return f_hat, usages, mean_vq_loss
    
    # ===================== embed_to_fhat: for inference =====================
    def embed_to_fhat(
        self, 
        ms_h_BCl: List[torch.Tensor], 
        all_to_max_scale: bool = True, 
        last_one: bool = False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Convert multi-scale embeddings to f_hat.
        Mirror: quant.py embed_to_fhat (lines 107-133)
        
        Args:
            ms_h_BCl: List of embeddings at each scale [(B, C, l1), (B, C, l2), ...]
            all_to_max_scale: Whether to upsample all to max scale
            last_one: Whether to return only the final f_hat
            
        Returns:
            List of accumulated f_hat at each scale, or final f_hat if last_one=True
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
                    h_BCl = F.interpolate(h_BCl, size=L, mode='linear', align_corners=False)
                h_BCl = self.quant_resi[si / (SN - 1)](h_BCl)
                f_hat.add_(h_BCl)
                if last_one:
                    ls_f_hat_BCL = f_hat
                else:
                    ls_f_hat_BCL.append(f_hat.clone())
        else:
            f_hat = ms_h_BCl[0].new_zeros(B, self.Cvae, self.v_patch_lens[0], dtype=torch.float32)
            for si, pl in enumerate(self.v_patch_lens):
                f_hat = F.interpolate(f_hat, size=pl, mode='linear', align_corners=False)
                h_BCl = self.quant_resi[si / (SN - 1)](ms_h_BCl[si])
                f_hat.add_(h_BCl)
                if last_one:
                    ls_f_hat_BCL = f_hat
                else:
                    ls_f_hat_BCL.append(f_hat)
        
        return ls_f_hat_BCL
    
    # ===================== f_to_idxBl_or_fhat: encode to indices or f_hat =====================
    def f_to_idxBl_or_fhat(
        self, 
        f_BCL: torch.Tensor, 
        to_fhat: bool, 
        v_patch_lens: Optional[Sequence[int]] = None
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:
        """
        Convert encoder output to indices or f_hat at each scale.
        Mirror: quant.py f_to_idxBl_or_fhat (lines 135-166)
        
        Args:
            f_BCL: (B, C, L) encoder output
            to_fhat: If True, return f_hat; if False, return indices
            v_patch_lens: Custom patch lengths (default: self.v_patch_lens)
            
        Returns:
            List of f_hat tensors or index tensors at each scale
        """
        B, C, L = f_BCL.shape
        f_no_grad = f_BCL.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        patch_lens = list(v_patch_lens or self.v_patch_lens)
        assert patch_lens[-1] == L, f'{patch_lens[-1]=} != {L=}'
        
        SN = len(patch_lens)
        for si, pl in enumerate(patch_lens):
            # Downsample f_rest
            if si != SN - 1:
                z_NC = F.interpolate(f_rest, size=pl, mode='linear', align_corners=False)
                z_NC = z_NC.permute(0, 2, 1).reshape(-1, C)
            else:
                z_NC = f_rest.permute(0, 2, 1).reshape(-1, C)
            
            # Find nearest embedding
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + \
                            torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            # Lookup and upsample
            idx_Bl = idx_N.view(B, pl)
            h_BCpl = self.embedding(idx_Bl).permute(0, 2, 1)
            
            if si != SN - 1:
                h_BCL = F.interpolate(h_BCpl, size=L, mode='linear', align_corners=False).contiguous()
            else:
                h_BCL = h_BCpl.contiguous()
            
            h_BCL = self.quant_resi[si / (SN - 1)](h_BCL)
            f_hat.add_(h_BCL)
            f_rest.sub_(h_BCL)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, pl))
        
        return f_hat_or_idx_Bl
    
    # ===================== idxBl_to_var_input: for VAR training =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert ground-truth indices to VAR input (teacher forcing).
        Mirror: quant.py idxBl_to_var_input (lines 169-184)
        
        Args:
            gt_ms_idx_Bl: List of ground-truth indices at each scale
            
        Returns:
            (B, total_tokens, C) concatenated embeddings for all scales except last
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
            h_BCL = F.interpolate(h_BCpl, size=L, mode='linear', align_corners=False)
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BCL))
            
            pl_next = self.v_patch_lens[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=pl_next, mode='linear', align_corners=False)
                .view(B, C, -1).transpose(1, 2)  # (B, pl_next, C)
            )
        
        return torch.cat(next_scales, dim=1) if len(next_scales) else None
    
    # ===================== get_next_autoregressive_input: for VAR inference =====================
    def get_next_autoregressive_input(
        self, 
        si: int, 
        SN: int, 
        f_hat: torch.Tensor, 
        h_BCl: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Get next step input for autoregressive generation.
        Mirror: quant.py get_next_autoregressive_input (lines 187-196)
        
        Args:
            si: Current scale index
            SN: Total number of scales
            f_hat: Current accumulated features (B, C, L)
            h_BCl: Current scale embeddings (B, C, pl)
            
        Returns:
            Updated f_hat, downsampled f_hat for next scale
        """
        L = self.v_patch_lens[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](F.interpolate(h_BCl, size=L, mode='linear', align_corners=False))
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=self.v_patch_lens[si + 1], mode='linear', align_corners=False)
        else:
            h = self.quant_resi[si / (SN - 1)](h_BCl)
            f_hat.add_(h)
            return f_hat, f_hat
