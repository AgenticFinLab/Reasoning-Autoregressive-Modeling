"""NLCP (Next-Level Concept Pyramid) Implementation.

This package implements the NLCP architecture for hierarchical text reasoning.

DESIGN SOURCE:
    - concept-pyramid.md - Complete Architecture Specification
    - concept-pyramid-critic.md - Critical analysis and recommended solutions

COMPONENT VARIANTS (from concept-pyramid-critic.md):
    Expansion Predictors:
        - ExpansionPredictor: Original floor() based (non-differentiable)
        - GumbelSoftmaxExpansionPredictor: Gumbel-Softmax relaxation (Solution 1A, RECOMMENDED)
        - REINFORCEExpansionPredictor: Policy gradient method (Solution 1B)
        - SoftExpansionPredictor: Continuous expansion (Solution 1C)

    Depth Gates:
        - DepthGate: Original non-causal version
        - CausalDepthGate: Causal pooling (Solution 3B, RECOMMENDED)

    Cross-Level Attention:
        - CrossLevelCausalAttention: Original rigid parent-child mapping
        - RelaxedCrossLevelAttention: Flexible multi-parent (Solution 4A, RECOMMENDED)
        - HybridCrossLevelAttention: Local + Global gating (Solution 4B)

    Consistency Losses:
        - CrossScaleConsistencyLoss: Original strict L2
        - DirectionalConsistencyLoss: Relaxed hinge loss (Solution 2A, RECOMMENDED)
        - ResidualConsistencyLoss: Learnable refinement (Solution 2B)
        - MutualInformationConsistencyLoss: InfoNCE based (Solution 2C)
"""

from examples.nlcp.base import (
    NLCPModelConfig,
    NLCPTrainingConfig,
    NLCPInferenceConfig,
    LevelState,
    NLCPOutput,
)
from examples.nlcp.model import NLCPModel, build_nlcp_model
from examples.nlcp.modules import (
    # Original components
    DepthGate,
    ExpansionPredictor,
    CrossLevelCausalAttention,
    NextLevelGenerator,
    TokenDecoder,
    LightweightEncoder,
    RMSNorm,
    # Critic.md solution implementations
    GumbelSoftmaxExpansionPredictor,  # Solution 1A (RECOMMENDED)
    REINFORCEExpansionPredictor,  # Solution 1B
    SoftExpansionPredictor,  # Solution 1C
    CausalDepthGate,  # Solution 3B (RECOMMENDED)
    RelaxedCrossLevelAttention,  # Solution 4A (RECOMMENDED)
    HybridCrossLevelAttention,  # Solution 4B
    # HuggingFace-based Encoder (DLCM-aligned)
    HFCausalEncoder,
)
from examples.nlcp.losses import (
    # Original losses
    NextTokenPredictionLoss,
    CrossScaleConsistencyLoss,
    ExpansionRateRegularization,
    FinalTokenAlignmentLoss,
    NLCPLossComputer,
    # Critic.md solution implementations
    DirectionalConsistencyLoss,  # Solution 2A (RECOMMENDED)
    ResidualConsistencyLoss,  # Solution 2B
    MutualInformationConsistencyLoss,  # Solution 2C
)
from examples.nlcp.inference import NLCPInference, build_inference_engine

__all__ = [
    # Configs
    "NLCPModelConfig",
    "NLCPTrainingConfig",
    "NLCPInferenceConfig",
    # State containers
    "LevelState",
    "NLCPOutput",
    # Model
    "NLCPModel",
    "build_nlcp_model",
    # Original Modules
    "DepthGate",
    "ExpansionPredictor",
    "CrossLevelCausalAttention",
    "NextLevelGenerator",
    "TokenDecoder",
    "LightweightEncoder",
    "RMSNorm",
    # Critic.md Solution Modules (RECOMMENDED)
    "GumbelSoftmaxExpansionPredictor",
    "REINFORCEExpansionPredictor",
    "SoftExpansionPredictor",
    "CausalDepthGate",
    "RelaxedCrossLevelAttention",
    "HybridCrossLevelAttention",
    # HuggingFace-based Encoder (DLCM-aligned)
    "HFCausalEncoder",
    # Original Losses
    "NextTokenPredictionLoss",
    "CrossScaleConsistencyLoss",
    "ExpansionRateRegularization",
    "FinalTokenAlignmentLoss",
    "NLCPLossComputer",
    # Critic.md Solution Losses (RECOMMENDED)
    "DirectionalConsistencyLoss",
    "ResidualConsistencyLoss",
    "MutualInformationConsistencyLoss",
    # Inference
    "NLCPInference",
    "build_inference_engine",
]
