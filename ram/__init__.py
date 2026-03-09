"""Text-VAR: Multi-Scale Autoregressive Modeling for Text

A framework for Chain-of-Thought reasoning inspired by VAR (Visual Autoregressive Modeling).
Structure mirrors third-part/VAR-main/models/.

Key Components:
- Encoder: 1D Conv encoder (mirrors basic_vae.py)
- Decoder: 1D Conv decoder with vocab projection (mirrors basic_vae.py)
- VectorQuantizer2: Multi-scale residual quantization (mirrors quant.py)
- VQVAE: Complete autoencoder (mirrors vqvae.py)

Data Loading:
    Use lmbase for data loading:
    >>> from lmbase.dataset.registry import get
    >>> config = {"data_name": "math", "data_path": "./data/math"}
    >>> dataset = get(config, split="train")
    
Usage:
    from ram.models import VQVAE
    from ram.utils import load_config
    
    config = load_config('configs/text_var/default.yml')
    model = VQVAE(**config['model'])
"""

from .models import Encoder, Decoder, VectorQuantizer2, VQVAE
from .utils import load_config, set_seed, count_parameters

__version__ = "0.2.0"
__all__ = [
    "Encoder",
    "Decoder",
    "VectorQuantizer2",
    "VQVAE",
    "load_config",
    "set_seed",
    "count_parameters",
]
