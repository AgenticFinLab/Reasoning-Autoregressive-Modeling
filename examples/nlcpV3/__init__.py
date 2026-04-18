"""NLCP V3: Implicit Reasoning via Hierarchical Concept Compression.

USAGE:
    from nlcpV3 import NLCPV3Config, NLCPV3Model

    # Individual extractors (for standalone use)
    from nlcpV3 import (
        ResidualAttentivePoolingConceptGenerator,
        PositionConstrainedConceptGenerator,
        HardOrderedMaskConceptGenerator,
        RecursiveOrderedConceptGenerator,
        OrderConstrainedTrainingConceptGenerator,
        RobustOrderedConceptGenerator,
        AutoregressiveConceptGenerator,
    )
    config = NLCPV3Config(...)
    extractor = ResidualAttentivePoolingConceptGenerator(config, encoder_hidden_dim)
    concepts, aux = extractor(H_cot)

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2: Architecture
    - Section 3: Training
    - Section 4: Inference

    Inspired by: docs/VAR.md
    - VAR uses same codebook/φ for both training (VQ-VAE) and inference
    - NLCP V3 unifies concept extraction and generation in one module

MODULE STRUCTURE:
    - config: NLCPV3Config dataclass for all hyperparameters
    - encoder: Qwen2.5-based encoder for Q+CoT (train) / Q (inference)
    - concept_generator: Training & inference methods
        * Individual extractors (standalone, trainable classes)
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
    # Other components
    "ConceptTransformer",
    "SolutionDecoder",
    "NLCPV3Model",
]
