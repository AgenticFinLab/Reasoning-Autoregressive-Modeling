"""TAR: Text Auto-Regressive Modeling

A framework for Chain-of-Thought reasoning inspired by VAR (Visual Autoregressive Modeling).

Key Components:
- TextEncoder: HuggingFace encoder with tokenizer
- TextDecoder: HuggingFace decoder for generation
- MultiScaleQuantizer: Multi-scale residual quantization (VAR innovation)
- TextVQVAE: Complete autoencoder

Data Loading:
    Use lmbase for data loading:
    >>> from lmbase.dataset import registry
    >>> dataset = registry.get("gsm8k", split="train")

Usage:
    from ram.models import TextEncoder, build_encoder
    from ram.utils import load_config

    config = load_config('configs/uTEST/encoder.yml')
    encoder = build_encoder(config['model']['encoder'])
"""

from .models import (
    TextEncoder,
    TextDecoder,
    MultiScaleQuantizer,
    TextVQVAE,
    build_encoder,
    build_decoder,
    build_quantizer,
    build_text_vqvae,
)
from .utils import load_config, set_seed, count_parameters, get_device

__version__ = "0.3.0"
__all__ = [
    "TextEncoder",
    "TextDecoder",
    "MultiScaleQuantizer",
    "TextVQVAE",
    "build_encoder",
    "build_decoder",
    "build_quantizer",
    "build_text_vqvae",
    "load_config",
    "set_seed",
    "count_parameters",
    "get_device",
]
