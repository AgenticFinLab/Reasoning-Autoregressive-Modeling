"""Custom implementations (legacy).

These are custom Conv1d-based implementations, kept for reference.
Main code now uses HuggingFace models in encoder.py/decoder.py.

Files:
- basic_vae.py: Custom Conv1d encoder/decoder
- basic_tar.py: Custom transformer blocks (AdaLN, FFN)
- quant.py: VectorQuantizer2 (VAR-style multi-scale)
- vqvae.py: VQVAE with custom encoder/decoder
- tar.py: TAR model with custom blocks
- sampling.py: Top-k/top-p sampling utilities
- regularization.py: DropPath (stochastic depth)
"""
