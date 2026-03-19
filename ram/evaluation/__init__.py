"""Text Reconstruction Evaluation.

This module provides evaluation metrics for text reconstruction tasks,
following the official C3 (Context Cascade Compression) evaluation approach.

Metrics:
    - Token Precision: correct_tokens / total_tokens (main metric from C3 paper)
    - Character Precision: correct_chars / total_chars
    - Edit Distance Ratio: 1 - (edit_distance / max_length)
    - BLEU Score: n-gram overlap (from Fox benchmark)

Reference:
    - Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
    - Official code: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
    - Benchmark: Fox (https://github.com/ucaslcl/Fox)

Flow:
    Original Text → Encoder → Latent Tokens → Decoder → Reconstructed Text
                              ↓
                    Compare with original → Compute metrics
"""

from .text_reconstruction import (
    compute_token_precision,
    compute_char_precision,
    compute_edit_distance,
    compute_edit_distance_ratio,
    compute_bleu_score,
    evaluate_reconstruction,
    evaluate_batch,
    TextReconstructionEvaluator,
    C3ReconstructionEvaluator,  # Backward compatibility alias
)

__all__ = [
    "compute_token_precision",
    "compute_char_precision",
    "compute_edit_distance",
    "compute_edit_distance_ratio",
    "compute_bleu_score",
    "evaluate_reconstruction",
    "evaluate_batch",
    "TextReconstructionEvaluator",
    "C3ReconstructionEvaluator",
]
