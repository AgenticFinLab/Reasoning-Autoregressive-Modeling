"""
Text-VAR Models

Mirrors the structure of third-part/VAR-main/models/:
- basic_vae.py: Encoder, Decoder (1D Conv versions)
- quant.py: VectorQuantizer2 (multi-scale residual quantization)
- vqvae.py: VQVAE (complete autoencoder)

File mapping:
    VAR-main/models/basic_vae.py  →  ram/models/basic_vae.py
    VAR-main/models/quant.py      →  ram/models/quant.py
    VAR-main/models/vqvae.py      →  ram/models/vqvae.py
"""

from .basic_vae import Encoder, Decoder
from .quant import VectorQuantizer2
from .vqvae import VQVAE

__all__ = [
    'Encoder',
    'Decoder', 
    'VectorQuantizer2',
    'VQVAE',
]
