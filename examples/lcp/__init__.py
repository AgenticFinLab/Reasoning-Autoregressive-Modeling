"""NLCP V4: Implicit Reasoning via Hierarchical Concept Compression.

USAGE:
    from lcp import ConceptPyramidBuilder

    # Pass raw config dict (loaded from YAML)
    builder = ConceptPyramidBuilder(config_dict)  # Encoder created internally
    # Using BuilderInput: forward() handles ALL tokenization internally
    from lcp.data_loader import BuilderInput
    batch_input = BuilderInput(
        questions=["What is 2+2?"],
        cot_answers=["Let me think... 2+2=4"],
        solutions=["4"],
    )
    pyramid = builder(batch_input)  # Training: PyramidOutput
    # pyramid.concepts: [C_0, ..., C_{K-1}]
    # pyramid.level_outputs: List[LevelOutput]
    # pyramid.reconstructed_hidden: for recon loss

DESIGN SOURCE:
    Reference: hybrid-analysis.md
    - Section 2: Architecture
    - Section 3: Training
    - Section 4: Inference

    Inspired by: docs/VAR.md
    - VAR separates VQ-VAE (extraction) and Transformer (generation)
    - NLCP V4 follows same principle: Builder (extraction) + Predictor (generation)

MODULE STRUCTURE:
    - concept_builder: Phase 1 — ConceptPyramidBuilder (training only)
    - concept_predictor: Phase 2 — ConceptPredictor (next-level prediction)
    - losses: Loss functions (reconstruction, ordering, residual, reasoning)
    - eval_builder: Builder evaluation loop, logging, and standalone CLI
    - eval_predictor: Predictor evaluation loop, logging, and standalone CLI
    - data_loader: DataLoader for Builder training
    - train_builder: Training script for ConceptPyramidBuilder
    - builder_training_analysis: Post-training analysis and visualization
"""

from lcp.concept_builder import (
    ConceptPyramidBuilder,
    EncoderOutput,
    LevelOutput,
    PyramidOutput,
)
from lcp.concept_predictor import ConceptPredictor
from lcp.concept_predictor_parallel import ConceptPredictorParallel
from lcp.data_loader import BuilderInput, NLCPV4DataLoader
from lcp.eval_builder import evaluate_builder
from lcp.eval_predictor import evaluate_predictor
from lcp.losses import compute_builder_loss

__all__ = [
    # Concept Pyramid Builder (Phase 1: training only)
    "ConceptPyramidBuilder",
    # Concept Predictor (Phase 2: next-level prediction)
    "ConceptPredictor",
    "ConceptPredictorParallel",
    # Builder input / output dataclasses
    "BuilderInput",
    "EncoderOutput",
    "LevelOutput",
    "PyramidOutput",
    # DataLoader
    "NLCPV4DataLoader",
    # Evaluation / loss computation
    "compute_builder_loss",
    "evaluate_builder",
    "evaluate_predictor",
]
