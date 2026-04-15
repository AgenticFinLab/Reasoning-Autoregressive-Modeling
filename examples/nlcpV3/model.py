"""NLCP V3 Model: Complete Model Integrating All Components.

USAGE:
    from nlcpV3 import NLCPV3Config, NLCPV3Model

    config = NLCPV3Config(...)
    model = NLCPV3Model(config)

    # Training
    output = model.forward_training(
        q_cot_input_ids, q_cot_attention_mask,
        solution_input_ids, solution_attention_mask
    )

    # Inference
    solution = model.forward_inference(q_input_ids, q_attention_mask)

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2: Architecture
    - Section 3: Training
    - Section 4: Inference

PURPOSE:
    NLCP V3 Model integrates all components for implicit reasoning:

    Training Path:
        Q+CoT → Encoder → ConceptGenerator (training mode) → Concepts
              → ConceptTransformer → SolutionDecoder → Solution

    Inference Path:
        Q → Encoder → ConceptGenerator (inference mode) → Concepts
          → ConceptTransformer → SolutionDecoder → Solution

    Key Difference from V2:
    - V2: Decoder outputs CoT tokens
    - V3: Decoder outputs Solution tokens directly (no CoT!)

    Key Design (inspired by VAR):
    - ConceptGenerator shares parameters between training and inference
    - Training: Extracts concepts from CoT using various strategies
    - Inference: Generates concepts from Q using shared queries
"""

import torch
import torch.nn as nn
from typing import Optional

from nlcpV3.config import NLCPV3Config
from nlcpV3.encoder import NLCPV3Encoder
from nlcpV3.concept_generator import (
    ConceptGenerator,
    ResidualAttentivePoolingConceptGenerator,
)
from nlcpV3.concept_transformer import ConceptTransformer
from nlcpV3.token_decoder import SolutionDecoder


class NLCPV3Model(nn.Module):
    """NLCP V3 Model for implicit reasoning via concept compression.

    PURPOSE:
        Complete model for V3 architecture. Handles both training
        (with CoT) and inference (without CoT) modes using unified
        ConceptGenerator that shares parameters between modes.

    ATTRIBUTES:
        config: NLCPV3Config instance
        encoder: Text encoder (Qwen2.5-based)
        concept_generator: Unified concept extraction/generation
        concept_transformer: Concept refinement with level-level causality
        solution_decoder: Direct solution decoder (key difference from V2!)

    DIMENSION FLOW:
        Constructor:
            config → initializes all components

        Training:
            Q+CoT [B, L] → Solution logits [B, L_solution, V]

        Inference:
            Q [B, L'] → Solution tokens [B, L_solution]
    """

    def __init__(self, config: NLCPV3Config):
        """Initialize NLCP V3 Model.

        Args:
            config: NLCPV3Config with all hyperparameters
        """
        super().__init__()
        self.config = config

        # Encoder (shared for training and inference)
        self.encoder = NLCPV3Encoder(config)
        encoder_hidden_dim = self.encoder.get_encoder_hidden_dim()

        # Unified Concept Generator (training & inference)
        # Shares concept_queries between training extraction and inference generation
        self.concept_generator = ConceptGenerator(config, encoder_hidden_dim)

        # Concept transformer (shared)
        self.concept_transformer = ConceptTransformer(config)

        # Solution decoder (decodes to solution, NOT CoT!)
        self.solution_decoder = SolutionDecoder(config)

    def forward_training(
        self,
        q_cot_input_ids: torch.Tensor,
        q_cot_attention_mask: torch.Tensor,
        solution_input_ids: torch.Tensor,
        solution_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        PURPOSE:
            Training forward pass with Q+CoT and Solution.
            Uses AttentivePooling to extract concepts from CoT.

        DIMENSION FLOW:
            Input:
                q_cot_input_ids: [B, L] - Q+CoT token indices
                q_cot_attention_mask: [B, L] - Q+CoT attention mask
                solution_input_ids: [B, L_solution] - Solution token indices
                solution_attention_mask: [B, L_solution] - Solution mask

            Process:
                1. Encode Q+CoT: H [B, L, D_encoder]
                2. Extract concepts: [C_0, ..., C_K] via AttentivePooling
                3. Refine concepts: [C'_0, ..., C'_K] via ConceptTransformer
                4. Decode to solution: logits [B, L_solution, V]

            Output:
                Dictionary with logits and auxiliary outputs

        Args:
            q_cot_input_ids: Q+CoT token indices [B, L]
            q_cot_attention_mask: Q+CoT attention mask [B, L]
            solution_input_ids: Solution token indices [B, L_solution]
            solution_attention_mask: Solution attention mask [B, L_solution]

        Returns:
            outputs: Dictionary with 'logits', 'concepts', etc.
        """
        # Step 1: Encode Q+CoT
        H = self.encoder.forward_training(
            q_cot_input_ids, q_cot_attention_mask
        )  # [B, L, D_encoder]

        # Step 2: Extract concepts from CoT (training mode)
        concepts, aux = self.concept_generator.forward_training(
            H, mode="residual_pooling"
        )
        # concepts = [C_0, C_1, ..., C_K]
        # aux contains H_hat, H_rest (like VAR's f_hat, f_rest)

        # Step 3: Refine concepts
        refined_concepts = self.concept_transformer(concepts)
        # refined_concepts = [C'_0, C'_1, ..., C'_K]

        # Step 4: Decode to solution (NOT CoT!)
        logits = self.solution_decoder(
            refined_concepts, solution_input_ids, solution_attention_mask
        )  # [B, L_solution, vocab_size]

        return {
            "logits": logits,
            "concepts": concepts,
            "refined_concepts": refined_concepts,
            "H_hat": aux.get("H_hat"),
            "H_rest": aux.get("H_rest"),
        }

    def forward_inference(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        max_solution_length: int = 128,
        eos_token_id: int = 0,
    ) -> torch.Tensor:
        """Forward pass for inference.

        PURPOSE:
            Inference forward pass with Q only (no CoT!).
            Uses ConceptGenerator (inference mode) to generate concepts from Q.

        DIMENSION FLOW:
            Input:
                q_input_ids: [B, L'] - Q token indices (L' < L, no CoT!)
                q_attention_mask: [B, L'] - Q attention mask

            Process:
                1. Encode Q: H [B, L', D_encoder]
                2. Generate concepts: [C_0, ..., C_K] via ConceptGenerator
                3. Refine concepts: [C'_0, ..., C'_K] via ConceptTransformer
                4. Generate solution autoregressively

            Output:
                Generated solution tokens [B, L_generated]

        Args:
            q_input_ids: Q token indices [B, L']
            q_attention_mask: Q attention mask [B, L']
            max_solution_length: Maximum solution length
            eos_token_id: End-of-sequence token ID

        Returns:
            solution: Generated solution tokens [B, L_generated]
        """
        # Step 1: Encode Q (no CoT!)
        H = self.encoder.forward_inference(
            q_input_ids, q_attention_mask
        )  # [B, L', D_encoder]

        # Step 2: Generate concepts from Q (inference mode)
        concepts = self.concept_generator.inference_generator(H)
        # concepts = [C_0, C_1, ..., C_K]

        # Step 3: Refine concepts
        refined_concepts = self.concept_transformer(concepts)
        # refined_concepts = [C'_0, C'_1, ..., C'_K]

        # Step 4: Generate solution autoregressively
        solution = self.solution_decoder.generate(
            refined_concepts, max_length=max_solution_length, eos_token_id=eos_token_id
        )  # [B, L_generated]

        return solution

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        solution_targets: torch.Tensor,
        solution_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute training loss.

        PURPOSE:
            Compute cross-entropy loss for solution prediction.

        Args:
            outputs: Output dictionary from forward_training
            solution_targets: Target solution tokens [B, L_solution]
            solution_mask: Mask for solution tokens [B, L_solution]

        Returns:
            loss: Scalar loss tensor
        """
        logits = outputs["logits"]  # [B, L_solution, vocab_size]

        # Reshape for cross-entropy
        logits_flat = logits.view(-1, logits.shape[-1])  # [B*L, V]
        targets_flat = solution_targets.view(-1)  # [B*L]

        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Apply mask if provided
        if solution_mask is not None:
            mask_flat = solution_mask.view(-1).float()
            loss = (loss * mask_flat).sum() / mask_flat.sum()
        else:
            loss = loss.mean()

        return loss

    def get_concept_shapes(self) -> list[tuple[int, int]]:
        """Get concept shapes for each level.

        PURPOSE:
            Utility for debugging and buffer allocation.

        Returns:
            List of (L_k, D) tuples for each level
        """
        return self.attentive_pooling.get_concept_shapes()
