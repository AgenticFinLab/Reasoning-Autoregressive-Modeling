"""NLCP V3: Implicit Reasoning via Hierarchical Concept Compression.

USAGE:
    from nlcpV3 import NLCPV3Config, NLCPV3Model

    config = NLCPV3Config()
    model = NLCPV3Model(config)

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2: Architecture
    - Section 3: Training
    - Section 4: Inference

PURPOSE:
    NLCP V3 shifts the paradigm from explicit CoT generation to implicit
    reasoning via hierarchical concept compression. During training, CoT
    is compressed into hierarchical concepts; during inference, concepts
    are generated directly from Q and decoded to solution (NO CoT generation).

MODULE STRUCTURE:
    - config: NLCPV3Config dataclass for all hyperparameters
    - encoder: Qwen2.5-based encoder for Q+CoT (train) / Q (inference)
    - attentive_pooling: Training-only concept extraction from CoT
    - ordered_concept_extractors: 5 schemes for ordered concept extraction
        * PositionConstrainedExtractor (Scheme 1)
        * HardOrderedMaskExtractor (Scheme 2)
        * RecursiveOrderedExtractor (Scheme 3)
        * OrderConstrainedTraining (Scheme 4)
        * RobustOrderedExtractor (Recommended combination)
    - concept_generator: Inference-only concept generation from Q
    - concept_transformer: VAR-style transformer with level-level causality
    - token_decoder: Decodes concepts directly to solution (NOT CoT!)
    - model: NLCPV3Model integrating all components

KEY DIFFERENCE FROM V2:
    V2: Concepts → Decoder → CoT tokens → extract answer
    V3: Concepts → Decoder → Solution (direct, no CoT!)
"""

from nlcpV3.config import NLCPV3Config
from nlcpV3.encoder import NLCPV3Encoder
from nlcpV3.attentive_pooling import ResidualAttentivePooling
from nlcpV3.ordered_concept_extractors import (
    PositionConstrainedExtractor,
    HardOrderedMaskExtractor,
    RecursiveOrderedExtractor,
    OrderConstrainedTraining,
    RobustOrderedExtractor,
    visualize_concept_attention,
    check_concept_ordering,
)
from nlcpV3.concept_generator import ConceptGenerator
from nlcpV3.concept_transformer import ConceptTransformer
from nlcpV3.token_decoder import SolutionDecoder
from nlcpV3.model import NLCPV3Model

__all__ = [
    "NLCPV3Config",
    "NLCPV3Encoder",
    "ResidualAttentivePooling",
    "PositionConstrainedExtractor",
    "HardOrderedMaskExtractor",
    "RecursiveOrderedExtractor",
    "OrderConstrainedTraining",
    "RobustOrderedExtractor",
    "visualize_concept_attention",
    "check_concept_ordering",
    "ConceptGenerator",
    "ConceptTransformer",
    "SolutionDecoder",
    "NLCPV3Model",
]
