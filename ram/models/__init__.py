"""TAR (Text Auto-Regressive) Models.

Modular structure:
- encoder.py: TextEncoder (HuggingFace)
- decoder.py: TextDecoder (HuggingFace)
- quantizer.py: MultiScaleQuantizer (custom, VAR innovation)
- scale_ops.py: ScaleOps (downsample/upsample operations)
- text_vqvae.py: TextVQVAE (combines all)

Architecture:
    [B, L] -> Encoder -> [B, L, D] -> Quantizer -> [B, L, C] -> Decoder -> [B, L, vocab]
"""

from .encoder import TextEncoder, build_encoder
from .decoder import TextDecoder, build_decoder
from .scale_ops import ScaleOps, AvgPoolScaleOps, LinearScaleOps, build_scale_ops
from .quantizer import MultiScaleQuantizer, build_quantizer
from .text_vqvae import TextVQVAE, build_text_vqvae

__all__ = [
    # encoder.py
    "TextEncoder",
    "build_encoder",
    # decoder.py
    "TextDecoder",
    "build_decoder",
    # scale_ops.py
    "ScaleOps",
    "AvgPoolScaleOps",
    "LinearScaleOps",
    "build_scale_ops",
    # quantizer.py
    "MultiScaleQuantizer",
    "build_quantizer",
    # text_vqvae.py
    "TextVQVAE",
    "build_text_vqvae",
]
