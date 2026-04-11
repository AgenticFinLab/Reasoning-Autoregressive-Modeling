"""NLCP (Next-Level Concept Pyramid) Inference Pipeline.

This module implements the inference pipeline for NLCP.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V1.md
    - Section 5: Inference Pipeline and Causal Guarantees
    - Section 5.1: Blocking Generation Algorithm
    - Section 5.2: Inference Flow Step-by-Step
    - Section 5.3: Causal Strictness Proof
    - Section 5.4: Inference Optimization Strategies

    Additional reference: docs/concept-pyramid-critic.md (solutions for V1 issues)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

from examples.nlcp.base import NLCPInferenceConfig
from examples.nlcp.model import NLCPModel


@dataclass
class GenerationState:
    """State container for generation process.

    Reference: concept-pyramid.md Section 5.1
    Blocking generation algorithm state tracking.

    Attributes:
        current_level: Current pyramid level
        hidden_states: Current level hidden states
        kv_cache_self: Self-attention KV cache for current level
        cross_kv: Cross-level K/V from parent level
        expand_mask: Expansion mask used to reach this level
    """

    current_level: int
    hidden_states: torch.Tensor
    kv_cache_self: List[List[torch.Tensor]]
    cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]]
    expand_mask: Optional[torch.Tensor]


class NLCPInference:
    """NLCP Inference Engine.

    Reference: concept-pyramid.md Section 5.1
    Blocking Generation Algorithm:

        H = encoder(Q_ids)  # [1, L₀, d]
        depth = 0
        kv_cache_self = []  # Same-level Self-Attn KV Cache

        while depth < max_depth:
            # 1. Depth gate
            p_cont = depth_gate(H, kv_cache_self)
            if p_cont < τ or H.shape[1] > L_max:
                break

            # 2. Predict expansion rate
            expand_mask = expansion_predictor(H).argmax(dim=-1)
            L_next = expand_mask.sum().item()

            # 3. Construct cross-level K/V (DLCM Concept Replication)
            K_rep = repeat_interleave(H @ W_K, expand_mask, dim=1)
            V_rep = repeat_interleave(H @ W_V, expand_mask, dim=1)

            # 4. Next-Level conditional AR generation
            H = ar_generate_level(
                length=L_next,
                K_cross=K_rep, V_cross=V_rep,
                kv_cache_self=kv_cache_self
            )
            depth += 1

        # 5. Token decode
        logits = (H @ W_unemb.T) / s_μP
        return autoregressive_decode(logits)

    Attributes:
        model: NLCP model instance
        config: Inference configuration
    """

    def __init__(self, model: NLCPModel, config: NLCPInferenceConfig):
        self.model = model
        self.config = config

    def generate_pyramid(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[GenerationState]]:
        """Generate pyramid levels until termination.

        Reference: concept-pyramid.md Section 5.1
        Steps 1-4 of blocking generation algorithm.

        Dimension Flow:
            input_ids: [B, L_q] question tokens
                ↓
            [Encoder]
            H_0: [B, L_0, D]
                ↓
            [Depth Gate] p_cont > τ?
                ↓ Yes
            [Expansion Predictor] → L_1
                ↓
            [Next-Level Generator]
            H_1: [B, L_1, D]
                ↓ ... (loop)
            H_K: [B, L_K, D]

        Args:
            input_ids: [B, L_q] input token IDs

        Returns:
            final_hidden: [B, L_K, D] final level hidden states
            states: List of GenerationState for each level
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Step 1: Encode to Level 0
        # Reference: "H = encoder(Q_ids)  # [1, L₀, d]"
        H = self.model.encoder(input_ids)
        H = self.model.l0_proj(H)
        H = self.model.l0_norm(H)

        # Add level embedding
        level_emb = self.model.level_embedding(
            torch.zeros(batch_size, H.size(1), dtype=torch.long, device=device)
        )
        H = H + level_emb

        # Initialize state tracking
        states: List[GenerationState] = []
        states.append(
            GenerationState(
                current_level=0,
                hidden_states=H,
                kv_cache_self=[],
                cross_kv=None,
                expand_mask=None,
            )
        )

        depth = 0

        # Step 2-4: Dynamic pyramid expansion loop
        while depth < self.config.max_depth:
            # Step 2.1: Depth gate decision
            # Reference: "p_cont = depth_gate(H, kv_cache_self)"
            p_cont = self.model.depth_gate(H)

            # Reference: "if p_cont < τ or H.shape[1] > L_max: break"
            if p_cont.mean() < self.config.depth_threshold:
                break

            model_config = self.model.config
            if H.size(1) >= model_config.l_max:
                break

            # Step 2.2: Predict expansion rates
            # Reference: "expand_mask = expansion_predictor(H).argmax(dim=-1)"
            expand_mask, _ = self.model.expansion_predictor(
                H, temperature=self.config.temperature
            )
            L_next = expand_mask.sum(dim=-1).max().item()

            if L_next <= 0:
                break

            # Step 2.3: Initialize next level
            # Reference: Next-Level Generator input preparation
            H_next = torch.zeros(
                batch_size,
                L_next,
                model_config.hidden_dim,
                device=device,
                dtype=H.dtype,
            )

            # Add level embedding for next level
            level_k_emb = self.model.level_embedding(
                torch.full(
                    (batch_size, L_next), depth + 1, dtype=torch.long, device=device
                )
            )
            H_next = H_next + level_k_emb

            # Step 2.4: Next-Level Generator
            # Reference: "H = ar_generate_level(length=L_next, K_cross=K_rep, V_cross=V_rep, ...)"
            H_next, kv_cache = self.model.level_generators[depth](
                hidden_states=H_next,
                coarse_hidden_states=H,
                expand_mask=expand_mask,
                kv_cache=None,
                use_cache=False,
            )

            # Store state
            states.append(
                GenerationState(
                    current_level=depth + 1,
                    hidden_states=H_next,
                    kv_cache_self=kv_cache if kv_cache else [],
                    cross_kv=None,
                    expand_mask=expand_mask,
                )
            )

            # Update for next iteration
            H = H_next
            depth += 1

        return H, states

    def generate_tokens(
        self,
        final_hidden: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Autoregressive token generation from final hidden states.

        Reference: concept-pyramid.md Section 5.1 Step 5
        "logits = (H @ W_unemb.T) / s_μP
         return autoregressive_decode(logits)"

        Dimension Flow:
            H_K: [B, L_K, D] final hidden states
                ↓
            Token decode: [B, L_K, V] logits
                ↓
            AR sample: [B, 1] next token
                ↓
            Append and loop
                ↓
            Output: [B, L_K + max_new_tokens]

        Args:
            final_hidden: [B, L_K, D] final level hidden states
            max_new_tokens: Maximum tokens to generate

        Returns:
            generated_tokens: [B, L_K + max_new_tokens] generated token sequence
        """
        batch_size = final_hidden.size(0)
        device = final_hidden.device

        # Token decode
        logits = self.model.token_decoder(final_hidden)

        # Initialize with initial positions from hidden states
        generated_tokens = torch.zeros(
            batch_size,
            final_hidden.size(1) + max_new_tokens,
            dtype=torch.long,
            device=device,
        )

        # Fill in initial positions (placeholder, would need proper projection)
        generated_tokens[:, : final_hidden.size(1)] = 0

        # Autoregressive generation
        for i in range(max_new_tokens):
            # Get logits for last position
            current_logits = logits[:, -1, :] / self.config.temperature

            # Top-k filtering
            if self.config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    current_logits, self.config.top_k
                )
                current_logits = torch.full_like(current_logits, float("-inf"))
                current_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Top-p (nucleus) filtering
            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    current_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                current_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(current_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Store generated token
            generated_tokens[:, final_hidden.size(1) + i] = next_token.squeeze(-1)

            # For proper AR generation, would need to update hidden states
            # This is simplified - full implementation would re-encode and re-run

        return generated_tokens

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Full generation pipeline.

        Reference: concept-pyramid.md Section 5.1
        Complete blocking generation algorithm.

        This is the main entry point for inference, combining
        pyramid generation and token generation.

        Dimension Flow:
            input_ids: [B, L_q] question tokens
                ↓
            generate_pyramid() → H_K: [B, L_K, D]
                ↓
            generate_tokens() → output_ids: [B, L_K + max_new_tokens]

        Args:
            input_ids: [B, L_q] input token IDs
            max_new_tokens: Maximum tokens to generate

        Returns:
            generated_tokens: [B, ...] generated token sequence
        """
        # Generate pyramid levels
        final_hidden, states = self.generate_pyramid(input_ids)

        # Early exit check
        # Reference: Section 5.3 "If Depth Gate score is low, terminate early"
        if self.config.early_exit and len(states) < 2:
            # Early exit - minimal processing
            pass

        # Generate tokens from final hidden states
        generated_tokens = self.generate_tokens(final_hidden, max_new_tokens)

        return generated_tokens


def build_inference_engine(
    model: NLCPModel,
    config: NLCPInferenceConfig,
) -> NLCPInference:
    """Build NLCP inference engine.

    Reference: concept-pyramid.md Section 5
    Inference Pipeline and Causal Guarantees.

    Args:
        model: Trained NLCP model
        config: Inference configuration

    Returns:
        NLCPInference instance
    """
    return NLCPInference(model, config)
