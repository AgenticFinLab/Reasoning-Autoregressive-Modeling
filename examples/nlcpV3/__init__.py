"""NLCP V3: Implicit Reasoning via Hierarchical Concept Compression.

USAGE:
    from nlcpV3 import NLCPV3Config, NLCPV3Model

    # RECOMMENDED: Concept Pyramid Builder (Phase 1 of two-phase architecture)
    from nlcpV3 import ConceptPyramidBuilder
    config = NLCPV3Config(
        encoder_model_name="Qwen/Qwen2.5-0.5B",
        encoder_freeze=False,  # Set True to freeze encoder
        ...
    )
    builder = ConceptPyramidBuilder(config)  # Encoder created internally
    enc_out = builder.encode_cot(cot_input_ids, attention_mask=cot_mask)
    pyramid = builder(enc_out.hidden_states)  # Training: PyramidOutput
    # pyramid.concepts: [C_0, ..., C_{K-1}]
    # pyramid.level_outputs: List[LevelOutput]
    # pyramid.reconstructed_hidden: for recon loss

    # Individual extractors (for standalone use / ablation studies)
    from nlcpV3 import (
        ResidualAttentivePoolingConceptGenerator,
        PositionConstrainedConceptGenerator,
        HardOrderedMaskConceptGenerator,
        RecursiveOrderedConceptGenerator,
        OrderConstrainedTrainingConceptGenerator,
        RobustOrderedConceptGenerator,
        AutoregressiveConceptGenerator,
    )
    extractor = ResidualAttentivePoolingConceptGenerator(config, encoder_hidden_dim)
    concepts, aux = extractor(H_cot)

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2: Architecture
    - Section 3: Training
    - Section 4: Inference

    Inspired by: docs/VAR.md
    - VAR separates VQ-VAE (extraction) and Transformer (generation)
    - NLCP V3 follows same principle: Builder (extraction) + Predictor (generation)

MODULE STRUCTURE:
    - config: NLCPV3Config dataclass for all hyperparameters
    - encoder: Qwen2.5-based encoder for Q+CoT (train) / Q (inference)
    - concept_hybrid_builder: Phase 1 — ConceptPyramidBuilder (training only)
    - concept_generator: Individual extractors (standalone, ablation studies)
    - concept_transformer: VAR-style transformer with level-level causality
    - token_decoder: Decodes concepts directly to solution (NOT CoT!)
    - model: NLCPV3Model integrating all components

KEY DIFFERENCE FROM V2:
    V2: Concepts → Decoder → CoT tokens → extract answer
    V3: Concepts → Decoder → Solution (direct, no CoT!)
"""

from nlcpV3.config import NLCPV3Config
from nlcpV3.encoder import NLCPV3Encoder
from nlcpV3.concept_generator import (
    BaseConceptGenerator,
    # Basic training extractors
    ResidualAttentivePoolingConceptGenerator,
    PositionConstrainedConceptGenerator,
    HardOrderedMaskConceptGenerator,
    RecursiveOrderedConceptGenerator,
    OrderConstrainedTrainingConceptGenerator,
    RobustOrderedConceptGenerator,
    # Advanced causal training extractors
    MonotonicSoftAssignmentConceptGenerator,
    CausalSequentialRefinementConceptGenerator,
    ContinuousCausalKernelConceptGenerator,
    AutoregressiveSoftBoundaryConceptGenerator,
    CausalSoftPoolingConceptGenerator,
    # Inference generator
    AutoregressiveConceptGenerator,
)
from nlcpV3.concept_hybrid_builder import (
    ConceptPyramidBuilder,
    EncoderOutput,
    LevelOutput,
    PyramidOutput,
    SingleLevelOutput,
)
from nlcpV3.concept_transformer import ConceptTransformer
from nlcpV3.token_decoder import SolutionDecoder
from nlcpV3.model import NLCPV3Model

__all__ = [
    # Core components
    "NLCPV3Config",
    "NLCPV3Encoder",
    # Base class
    "BaseConceptGenerator",
    # Basic training extractors (standalone use)
    "ResidualAttentivePoolingConceptGenerator",
    "PositionConstrainedConceptGenerator",
    "HardOrderedMaskConceptGenerator",
    "RecursiveOrderedConceptGenerator",
    "OrderConstrainedTrainingConceptGenerator",
    "RobustOrderedConceptGenerator",
    # Advanced causal training extractors
    "MonotonicSoftAssignmentConceptGenerator",
    "CausalSequentialRefinementConceptGenerator",
    "ContinuousCausalKernelConceptGenerator",
    "AutoregressiveSoftBoundaryConceptGenerator",
    "CausalSoftPoolingConceptGenerator",
    # Inference generator
    "AutoregressiveConceptGenerator",
    # Concept Pyramid Builder (Phase 1: training only)
    "ConceptPyramidBuilder",
    # Builder output dataclasses
    "EncoderOutput",
    "LevelOutput",
    "PyramidOutput",
    "SingleLevelOutput",
    # Other components
    "ConceptTransformer",
    "SolutionDecoder",
    "NLCPV3Model",
]
