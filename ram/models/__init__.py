"""TAR (Text Auto-Regressive) Models.

Modular structure:
- encoder.py: TextEncoder (HuggingFace), C3Encoder (Cascade Compression)
- decoder.py: TextDecoder (HuggingFace), C3Decoder (Cascade Reconstruction)
- quantizer.py: MultiScaleQuantizer (custom, VAR innovation)
- scale_ops.py: ScaleOps (downsample/upsample operations)
- text_vqvae.py: TextVQVAE (combines all)

Architecture:
    [B, L] -> Encoder -> [B, L, D] -> Quantizer -> [B, L, C] -> Decoder -> [B, L, vocab]

C3 Cascade Architecture (arXiv:2511.15244):
    [B, M] text -> C3Encoder -> [B, N, D] latent -> C3Decoder -> [B, M'] reconstructed
    (M >> N, compression ratio M/N = 20x~40x)
"""

from .encoder import TextEncoder, build_encoder, C3Encoder, build_c3_encoder
from .decoder import TextDecoder, build_decoder, C3Decoder, build_c3_decoder
from .scale_ops import ScaleOps, AvgPoolScaleOps, LinearScaleOps, build_scale_ops
from .quantizer import MultiScaleQuantizer, build_quantizer
from .text_vqvae import TextVQVAE, build_text_vqvae

__all__ = [
    # encoder.py
    "TextEncoder",
    "build_encoder",
    "C3Encoder",
    "build_c3_encoder",
    # decoder.py
    "TextDecoder",
    "build_decoder",
    "C3Decoder",
    "build_c3_decoder",
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
