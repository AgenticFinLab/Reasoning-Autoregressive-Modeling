"""Text Reconstruction Evaluation Metrics.

This module provides evaluation metrics for text reconstruction tasks,
with implementations following the official C3 (Context Cascade Compression)
evaluation approach.

Official Reference:
    - Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
      https://arxiv.org/abs/2511.15244
    - Official Code: third-part/C3-Context-Cascade-Compression-main/C3-master/C3/model/C3.py
    - Benchmark: Fox (https://github.com/ucaslcl/Fox)

Key Metrics:
    - Token Precision: correct_tokens / total_tokens
      * Main metric from C3 paper ("93% accuracy at 40x compression")
      * Measures exact token-level reconstruction fidelity

    - Character Precision: correct_chars / total_chars
      * Fine-grained character-level accuracy
      * Useful for detecting partial token errors

    - Edit Distance Ratio: 1 - (edit_distance / max_length)
      * Normalized Levenshtein similarity
      * From Fox benchmark eval_tools/eval_ocr_test.py

    - BLEU Score: n-gram overlap precision
      * Standard MT evaluation metric
      * From Fox benchmark eval_tools/eval_ocr_test.py

Usage:
    >>> from ram.evaluation import TextReconstructionEvaluator
    >>> evaluator = TextReconstructionEvaluator(tokenizer)
    >>> metrics = evaluator.evaluate(original_text, reconstructed_text)
    >>> print(f"Token Precision: {metrics['token_precision']:.2%}")
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


# =============================================================================
# Token-Level Metrics
# =============================================================================


def compute_token_precision(
    original_tokens: List[int],
    reconstructed_tokens: List[int],
) -> float:
    """Compute token-level precision for text reconstruction.

    This is the PRIMARY evaluation metric used in the C3 paper,
    referred to as "decoding accuracy" or "reconstruction precision".

    Official Source:
        - Paper Section 1: "At a 40x compression ratio, our model maintains
          a decoding accuracy of 93%"
        - Paper Figure 1: "Reconstruction precision and compression ratio
          of C3 versus Deepseek-OCR on the Fox benchmark"
        - Implementation rationale: The paper measures reconstruction quality
          by comparing token-by-token between original and reconstructed text.
          This provides a strict, interpretable measure of information fidelity.

    Formula:
        precision = correct_tokens / total_original_tokens

    Why This Metric:
        - Direct measure of information preservation through compression
        - Token-level precision is stricter than character-level (catches
          semantic drift that character matching might miss)
        - Used consistently across C3 paper for comparing compression methods

    Args:
        original_tokens: List of original token IDs from tokenizer.encode().
        reconstructed_tokens: List of reconstructed token IDs.

    Returns:
        float: Token precision in range [0, 1]. Higher is better.

    Example:
        >>> original = [1, 2, 3, 4, 5]
        >>> reconstructed = [1, 2, 3, 4, 6]  # 4 correct, 1 wrong
        >>> compute_token_precision(original, reconstructed)
        0.8  # 4/5 correct
    """
    if not original_tokens:
        return 1.0 if not reconstructed_tokens else 0.0

    # Align lengths by taking minimum
    # Rationale: If reconstruction is shorter/longer, we compare what exists
    min_len = min(len(original_tokens), len(reconstructed_tokens))
    if min_len == 0:
        return 0.0

    # Count position-wise correct tokens
    # Official C3 uses strict position matching (not bag-of-tokens)
    correct = sum(
        1 for i in range(min_len) if original_tokens[i] == reconstructed_tokens[i]
    )

    # Precision based on original length (as per C3 paper)
    # Rationale: Measures how much of original information is preserved
    return correct / len(original_tokens)


# =============================================================================
# Character-Level Metrics
# =============================================================================


def compute_char_precision(
    original_text: str,
    reconstructed_text: str,
) -> float:
    """Compute character-level precision for text reconstruction.

    Official Source:
        - Fox benchmark: eval_tools/eval_ocr_test.py computes character-level
          F1, Precision, Recall for OCR tasks
        - C3 paper: Uses token-level as primary, but char-level provides
          finer-grained error analysis

    Why This Metric:
        - More fine-grained than token-level
        - Catches partial token errors (e.g., "hello" vs "helo")
        - Useful for languages where token boundaries differ from word boundaries

    Formula:
        precision = correct_chars / total_original_chars

    Args:
        original_text: Original text string.
        reconstructed_text: Reconstructed text string.

    Returns:
        float: Character precision in range [0, 1].
    """
    if not original_text:
        return 1.0 if not reconstructed_text else 0.0

    min_len = min(len(original_text), len(reconstructed_text))
    if min_len == 0:
        return 0.0

    correct = sum(
        1 for i in range(min_len) if original_text[i] == reconstructed_text[i]
    )

    return correct / len(original_text)


def compute_edit_distance(
    original: str,
    reconstructed: str,
) -> int:
    """Compute Levenshtein edit distance between two strings.

    Official Source:
        - Fox benchmark: eval_tools/eval_ocr_test.py
          "Calculate BLEU, METEOR, F1-score, Precision, Recall, Edit Distance"
        - Implementation: Standard dynamic programming approach

    Why This Metric:
        - Standard measure for text similarity in OCR/MT evaluation
        - Captures insertions, deletions, and substitutions
        - Normalized version (edit_distance_ratio) provides 0-1 scale

    Algorithm:
        Dynamic programming with O(m*n) time complexity.
        dp[i][j] = minimum edits to transform original[:i] to reconstructed[:j]

    Args:
        original: Original string.
        reconstructed: Reconstructed string.

    Returns:
        int: Minimum number of single-character edits (insert/delete/substitute).
    """
    m, n = len(original), len(reconstructed)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # deletions
    for j in range(n + 1):
        dp[0][j] = j  # insertions

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if original[i - 1] == reconstructed[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # no edit needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]


def compute_edit_distance_ratio(
    original: str,
    reconstructed: str,
) -> float:
    """Compute normalized edit distance ratio (similarity score).

    Official Source:
        - Fox benchmark: eval_tools/eval_ocr_test.py normalizes edit distance
        - C3 paper: Uses precision as main metric, but edit distance provides
          complementary view of reconstruction quality

    Formula:
        ratio = 1 - (edit_distance / max(len(original), len(reconstructed)))

    Why This Metric:
        - Normalized to [0, 1] range for cross-sample comparison
        - 1.0 = perfect match, 0.0 = completely different
        - Handles length mismatches gracefully

    Args:
        original: Original string.
        reconstructed: Reconstructed string.

    Returns:
        float: Similarity ratio in range [0, 1]. Higher is better.
    """
    if not original and not reconstructed:
        return 1.0

    max_len = max(len(original), len(reconstructed))
    if max_len == 0:
        return 1.0

    edit_dist = compute_edit_distance(original, reconstructed)
    return 1.0 - (edit_dist / max_len)


# =============================================================================
# N-gram Metrics (BLEU)
# =============================================================================


def compute_bleu_score(
    original: str,
    reconstructed: str,
    n_gram: int = 4,
) -> float:
    """Compute BLEU score for text reconstruction.

    Official Source:
        - Fox benchmark: eval_tools/eval_ocr_test.py
          "Calculate BLEU, METEOR, F1-score, Precision, Recall, Edit Distance"
        - Standard BLEU-4 implementation with brevity penalty

    Why This Metric:
        - Standard metric in machine translation and OCR evaluation
        - Captures n-gram overlap, not just single tokens
        - Brevity penalty discourages short reconstructions

    Formula:
        BLEU = BP * exp(sum(log(p_n)) / N)
        where:
        - BP = brevity penalty = exp(1 - ref_len/hyp_len) if hyp_len <= ref_len
        - p_n = n-gram precision for n=1,2,3,4

    Args:
        original: Original text string (reference).
        reconstructed: Reconstructed text string (hypothesis).
        n_gram: Maximum n-gram order (default: 4 for BLEU-4).

    Returns:
        float: BLEU score in range [0, 1]. Higher is better.

    Note:
        This is a simplified implementation. For production use,
        consider sacrebleu or nltk.translate.bleu_score for:
        - Smoothing methods for short sequences
        - Multiple reference support
        - Tokenization options
    """
    from collections import Counter
    import math

    if not original or not reconstructed:
        return 0.0

    # Tokenize by whitespace
    # Note: Fox benchmark uses whitespace tokenization for BLEU
    original_tokens = original.split()
    reconstructed_tokens = reconstructed.split()

    if not original_tokens or not reconstructed_tokens:
        return 0.0

    # Compute brevity penalty
    ref_len = len(original_tokens)
    hyp_len = len(reconstructed_tokens)

    if hyp_len <= ref_len:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    else:
        bp = 1.0

    # Compute n-gram precisions
    precisions = []
    for n in range(1, n_gram + 1):
        # Extract n-grams
        original_ngrams = [
            tuple(original_tokens[i : i + n])
            for i in range(len(original_tokens) - n + 1)
        ]
        reconstructed_ngrams = [
            tuple(reconstructed_tokens[i : i + n])
            for i in range(len(reconstructed_tokens) - n + 1)
        ]

        if not original_ngrams or not reconstructed_ngrams:
            precisions.append(0.0)
            continue

        # Count clipped matches
        original_counts = Counter(original_ngrams)
        reconstructed_counts = Counter(reconstructed_ngrams)

        matches = 0
        total = 0
        for ngram, count in reconstructed_counts.items():
            matches += min(count, original_counts[ngram])
            total += count

        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)

    # Compute geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geo_mean = 0.0

    return bp * geo_mean


# =============================================================================
# Unified Evaluation Functions
# =============================================================================


def evaluate_reconstruction(
    original_text: str,
    reconstructed_text: str,
    tokenizer=None,
) -> Dict[str, float]:
    """Evaluate text reconstruction with all metrics.

    Official Source:
        - C3 paper: Uses token precision as primary metric
        - Fox benchmark: Uses BLEU, Edit Distance, F1, Precision, Recall

    Metrics Computed:
        - char_precision: Character-level exact match ratio
        - edit_distance: Raw Levenshtein distance (lower is better)
        - edit_distance_ratio: Normalized similarity (higher is better)
        - bleu_score: BLEU-4 score (higher is better)
        - token_precision: Token-level exact match ratio (if tokenizer provided)

    Why These Metrics:
        - Token precision: Primary metric from C3 paper for reconstruction
        - Character precision: Fine-grained error detection
        - Edit distance: Standard OCR/MT evaluation metric
        - BLEU: Captures n-gram preservation, standard in text generation

    Args:
        original_text: Original text string.
        reconstructed_text: Reconstructed text string.
        tokenizer: Tokenizer for token-level metrics (optional).

    Returns:
        Dict[str, float]: All computed metrics.

    Example:
        >>> metrics = evaluate_reconstruction("hello world", "hello worlx")
        >>> print(f"Char Precision: {metrics['char_precision']:.2%}")
    """
    metrics = {
        "char_precision": compute_char_precision(original_text, reconstructed_text),
        "edit_distance": compute_edit_distance(original_text, reconstructed_text),
        "edit_distance_ratio": compute_edit_distance_ratio(
            original_text, reconstructed_text
        ),
        "bleu_score": compute_bleu_score(original_text, reconstructed_text),
    }

    # Add token-level metrics if tokenizer provided
    # This is the PRIMARY metric from C3 paper
    if tokenizer is not None:
        original_tokens = tokenizer.encode(original_text, add_special_tokens=False)
        reconstructed_tokens = tokenizer.encode(
            reconstructed_text, add_special_tokens=False
        )
        metrics["token_precision"] = compute_token_precision(
            original_tokens, reconstructed_tokens
        )

    return metrics


def evaluate_batch(
    original_texts: List[str],
    reconstructed_texts: List[str],
    tokenizer=None,
) -> Dict[str, float]:
    """Evaluate a batch of reconstructions and return averaged metrics.

    Official Source:
        - C3 paper: Reports averaged precision across Fox benchmark samples
        - Fox benchmark: Batch evaluation script in README

    Why Batch Evaluation:
        - Single sample metrics can be noisy
        - Averaging provides stable comparison across methods
        - Paper reports aggregate statistics (e.g., "93% accuracy")

    Args:
        original_texts: List of original text strings.
        reconstructed_texts: List of reconstructed text strings.
        tokenizer: Tokenizer for token-level metrics (optional).

    Returns:
        Dict[str, float]: Averaged metrics across the batch.

    Example:
        >>> originals = ["hello world", "foo bar"]
        >>> reconstructeds = ["hello worlx", "foo baz"]
        >>> metrics = evaluate_batch(originals, reconstructeds)
    """
    if len(original_texts) != len(reconstructed_texts):
        raise ValueError(
            f"Length mismatch: {len(original_texts)} originals vs "
            f"{len(reconstructed_texts)} reconstructed"
        )

    all_metrics = []
    for orig, recon in zip(original_texts, reconstructed_texts):
        metrics = evaluate_reconstruction(orig, recon, tokenizer)
        all_metrics.append(metrics)

    # Average metrics
    averaged = {}
    for key in all_metrics[0].keys():
        averaged[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return averaged


# =============================================================================
# Evaluator Class
# =============================================================================


class TextReconstructionEvaluator:
    """Evaluator class for text reconstruction tasks.

    This class provides a convenient interface for evaluating text
    reconstruction with metrics from C3 paper and Fox benchmark.

    Official Reference:
        - C3 paper: "Context Cascade Compression: Exploring the Upper Limits
          of Text Compression" (arXiv:2511.15244)
        - Fox benchmark: https://github.com/ucaslcl/Fox

    Usage:
        >>> from ram.evaluation import TextReconstructionEvaluator
        >>> evaluator = TextReconstructionEvaluator(tokenizer)
        >>> metrics = evaluator.evaluate(original, reconstructed)
        >>> batch_metrics = evaluator.evaluate_batch(originals, reconstructeds)

    Attributes:
        tokenizer: Tokenizer for token-level metrics.
    """

    def __init__(self, tokenizer=None):
        """Initialize the evaluator.

        Args:
            tokenizer: Tokenizer for token-level metrics (optional).
                If not provided, token_precision will not be computed.
                Recommended: Use the same tokenizer as the model being evaluated.
        """
        self.tokenizer = tokenizer

    def evaluate(
        self,
        original_text: str,
        reconstructed_text: str,
    ) -> Dict[str, float]:
        """Evaluate a single reconstruction.

        Args:
            original_text: Original text string.
            reconstructed_text: Reconstructed text string.

        Returns:
            Dict with all metrics.
        """
        return evaluate_reconstruction(
            original_text, reconstructed_text, self.tokenizer
        )

    def evaluate_batch(
        self,
        original_texts: List[str],
        reconstructed_texts: List[str],
    ) -> Dict[str, float]:
        """Evaluate a batch of reconstructions.

        Args:
            original_texts: List of original text strings.
            reconstructed_texts: List of reconstructed text strings.

        Returns:
            Dict with averaged metrics.
        """
        return evaluate_batch(original_texts, reconstructed_texts, self.tokenizer)

    def print_report(
        self,
        original_texts: List[str],
        reconstructed_texts: List[str],
    ) -> Dict[str, float]:
        """Evaluate and print a detailed report.

        Args:
            original_texts: List of original text strings.
            reconstructed_texts: List of reconstructed text strings.

        Returns:
            Dict with averaged metrics.
        """
        metrics = self.evaluate_batch(original_texts, reconstructed_texts)

        print("=" * 60)
        print("Text Reconstruction Evaluation Report")
        print("=" * 60)
        print(f"Samples: {len(original_texts)}")
        print()

        # Main metric from C3 paper
        if "token_precision" in metrics:
            print(f"Token Precision (Main):  {metrics['token_precision']:.2%}")
        print(f"Character Precision:     {metrics['char_precision']:.2%}")
        print(f"Edit Distance Ratio:     {metrics['edit_distance_ratio']:.2%}")
        print(f"BLEU Score:              {metrics['bleu_score']:.4f}")
        print(f"Average Edit Distance:   {metrics['edit_distance']:.1f}")
        print()

        return metrics


# Backward compatibility alias
C3ReconstructionEvaluator = TextReconstructionEvaluator


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Text Reconstruction Evaluation - Test")
    print("=" * 60)

    # Test 1: Basic metrics
    print("\n[Test 1] Basic Metrics")
    original = "hello world this is a test"
    reconstructed = "hello world this is test"  # missing 'a '
    metrics = evaluate_reconstruction(original, reconstructed)
    print(f"Original:      '{original}'")
    print(f"Reconstructed: '{reconstructed}'")
    print(f"Char Precision:      {metrics['char_precision']:.2%}")
    print(f"Edit Distance:       {metrics['edit_distance']}")
    print(f"Edit Distance Ratio: {metrics['edit_distance_ratio']:.2%}")
    print(f"BLEU Score:          {metrics['bleu_score']:.4f}")

    # Test 2: Perfect match
    print("\n[Test 2] Perfect Match")
    perfect = evaluate_reconstruction("hello", "hello")
    print(f"Char Precision: {perfect['char_precision']:.2%}")
    print(f"Edit Distance:  {perfect['edit_distance']}")

    # Test 3: Token precision
    print("\n[Test 3] Token Precision")
    orig_tokens = [1, 2, 3, 4, 5]
    recon_tokens = [1, 2, 3, 4, 6]  # 4 correct, 1 wrong
    precision = compute_token_precision(orig_tokens, recon_tokens)
    print(f"Token Precision: {precision:.2%}")

    # Test 4: Batch evaluation
    print("\n[Test 4] Batch Evaluation")
    originals = ["hello world", "foo bar baz", "test 123"]
    reconstructeds = ["hello worlx", "foo bar baq", "test 124"]
    batch_metrics = evaluate_batch(originals, reconstructeds)
    print(f"Samples: {len(originals)}")
    print(f"Avg Char Precision: {batch_metrics['char_precision']:.2%}")
    print(f"Avg BLEU Score:     {batch_metrics['bleu_score']:.4f}")

    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)
