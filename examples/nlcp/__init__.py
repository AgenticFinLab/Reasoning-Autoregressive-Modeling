"""NLCP (Next-Level Concept Pyramid) Implementation.

This package implements the NLCP architecture for hierarchical text reasoning.
Reference: concept-pyramid.md - Complete Architecture Specification
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
    DepthGate,
    ExpansionPredictor,
    CrossLevelCausalAttention,
    NextLevelGenerator,
    TokenDecoder,
    LightweightEncoder,
    RMSNorm,
)
from examples.nlcp.losses import (
    NextTokenPredictionLoss,
    CrossScaleConsistencyLoss,
    ExpansionRateRegularization,
    FinalTokenAlignmentLoss,
    NLCPLossComputer,
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
    # Modules
    "DepthGate",
    "ExpansionPredictor",
    "CrossLevelCausalAttention",
    "NextLevelGenerator",
    "TokenDecoder",
    "LightweightEncoder",
    "RMSNorm",
    # Losses
    "NextTokenPredictionLoss",
    "CrossScaleConsistencyLoss",
    "ExpansionRateRegularization",
    "FinalTokenAlignmentLoss",
    "NLCPLossComputer",
    # Inference
    "NLCPInference",
    "build_inference_engine",
]
