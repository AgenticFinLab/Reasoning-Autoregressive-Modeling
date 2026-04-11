"""NLCP (Next-Level Concept Pyramid) Main Model.

This module implements the complete NLCP architecture.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V1.md
    - Section 2.1: High-Level Data Flow (Encoder -> Pyramid -> Token Decoder)
    - Section 2.2: Module Tasks and Connection Logic Table
    - Section 3: Core Mechanisms Detailed Design
    - Section 5: Inference Pipeline and Causal Guarantees

    Additional reference: docs/concept-pyramid-critic.md (solutions for V1 issues)

ARCHITECTURE DATA FLOW (Section 2.1):
    Input: Question Q (Token IDs)
       ↓ [HFCausalEncoder]
    H_0 ∈ R^{L_0 × d}          (Level 0: Global Intent / Problem Abstraction)
       ↓ [Depth Gate] p_cont^(0) > tau ? --No--> Terminate
       ↓ Yes
    [Expansion Predictor] lambda_0 → L_1
       ↓ [Next-Level Generator (Causal Cross-Attn + Self-Attn)]
    H_1 ∈ R^{L_1 × d}          (Level 1: Logical Skeleton / High-Level Steps)
       ↓ [Depth Gate] p_cont^(1) > tau ? --No--> Terminate
       ↓ Yes
    [Expansion Predictor] lambda_1 → L_2
       ↓ [Next-Level Generator]
    H_2 ∈ R^{L_2 × d}          (Level 2: Intermediate Reasoning / Constraints)
       ↓ ... (dynamic loop to Level K)
       ↓ Terminate Condition Met
    [Token Projection Head] → Logits ∈ R^{L_out × V} → Autoregressive Decoding

MODULE TASKS (Section 2.2 Table):
    Encoder:       x ∈ [1, L_q] → H_0 ∈ [1, L_0, d]
    Depth Gate:    H_k ∈ [1, L_k, d] → p_cont ∈ [0,1]
    Expansion:     H_k → expand_mask ∈ [1, L_k] → L_{k+1} = Sum(lambda)
    Generator:     H_k, Q → H_{k+1} ∈ [1, L_{k+1}, d]
    Token Decoder: H_K → Logits ∈ [1, L_K, V]

KEY INSIGHT FROM V1 (Section 1.2-1.4):
    NLCP differs from VAR and DLCM in how it ensures "layer-wise approximation to CoT":

    - VAR: Uses f_rest (residual) to tell model "what to encode" at each scale
           Guarantee: Each scale has explicit supervision via residual decomposition

    - DLCM: Concept = Token Pool, naturally contains reconstruction information
            Guarantee: Concepts are directly extracted from ground truth CoT

    - NLCP: Uses implicit learning via gradient backprop + consistency constraints
            Guarantee: Gradient flow from final layer shapes intermediate layers

KNOWN IMPLEMENTATION GAPS (from concept-pyramid-critic.md):
    1. ExpansionPredictor uses non-differentiable floor()
    2. DepthGate uses full attention (not causal)
    3. CrossLevelAttention has rigid parent-child mapping
    4. ConsistencyLoss uses strict L2 (not relaxed)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from examples.nlcp.base import (
    NLCPModelConfig,
    LevelState,
    NLCPOutput,
)
from examples.nlcp.modules import (
    # Original components
    DepthGate,
    ExpansionPredictor,
    CrossLevelCausalAttention,
    NextLevelGenerator,
    TokenDecoder,
    RMSNorm,
    # Critic.md solution components
    GumbelSoftmaxExpansionPredictor,
    REINFORCEExpansionPredictor,
    SoftExpansionPredictor,
    CausalDepthGate,
    RelaxedCrossLevelAttention,
    HybridCrossLevelAttention,
    # HuggingFace-based Encoder (DLCM-aligned)
    HFCausalEncoder,
)
from examples.nlcp.losses import NLCPLossComputer


class NLCPModel(nn.Module):
    """Next-Level Concept Pyramid Model.

    DESIGN SOURCE - concept-pyramid.md Section 2.1:
        High-level data flow:
            Input: Question Q (Token IDs)
               ↓ [HFCausalEncoder]
            H₀ ∈ ℝ^{L₀ × d}          (Level 0: Global Intent / Problem Abstraction)
               ↓ [Depth Gate] p_cont^(0) > τ ? ──No──→ Terminate
               ↓ Yes
            [Expansion Predictor] λ₀ → L₁
               ↓ [Next-Level Generator (Causal Cross-Attn + Self-Attn)]
            H₁ ∈ ℝ^{L₁ × d}          (Level 1: Logical Skeleton / High-Level Steps)
               ↓ ... (dynamic loop to Level K)
               ↓ Terminate Condition Met
            [Token Projection Head] → Logits

    MODULE TASKS - concept-pyramid.md Section 2.2 Table:
        Module          Input                    Output                    Function
        ─────────────────────────────────────────────────────────────────────────────
        Encoder         x ∈ [1, L_q]            H_0 ∈ [1, L_0, d]         Problem abstraction
        Depth Gate      H_k ∈ [1, L_k, d]       p_cont ∈ [0,1]            Continue/terminate decision
        Expansion       H_k ∈ [1, L_k, d]       expand_mask ∈ [1, L_k]    Per-position expansion rates
        Generator       H_k, Q                  H_{k+1} ∈ [1, L_{k+1}, d] Next level generation
        Token Decoder   H_K ∈ [1, L_K, d]       Logits ∈ [1, L_K, V]      Vocabulary projection

    COMPARISON WITH VAR AND DLCM - concept-pyramid.md Section 1.3:
        VAR (Visual Autoregressive):
            - Fixed pyramid structure (8×8 → 16×16 → 32×32)
            - Image generation focus
            - Deterministic expansion (always 4×)

        DLCM (Dynamic Large Concept Model):
            - Semantic compression (text → latent → text)
            - Dynamic latent mapping
            - No hierarchical generation

        NLCP (This work):
            - Dynamic pyramid depth K (per sample)
            - Dynamic expansion rates λ_k (per position)
            - Hierarchical autoregressive generation
            - Cross-level consistency supervision

    CRITICAL IMPLEMENTATION GAPS (from concept-pyramid-critic.md):

        ISSUE 1 - Expansion Predictor Gradient Flow (Problem 1):
            Location: self.expansion_predictor() calls in forward()
            Problem: floor() operation is non-differentiable
            Impact: Model cannot learn optimal expansion rates directly
            Current workaround: L_depth regularization provides indirect signal

        ISSUE 3 - Depth Gate Causality (Problem 3):
            Location: self.depth_gate() calls in forward()
            Problem: Full attention pooling during training
            Impact: Train/test mismatch in depth decisions
            Current workaround: None (significant gap!)

        ISSUE 4 - Rigid Cross-Level Mapping (Problem 4):
            Location: level_generator (CrossLevelCausalAttention)
            Problem: repeat_interleave enforces strict 1-to-many parent-child
            Impact: Cannot access multi-parent context
            Current workaround: None

    FORWARD PASS ALGORITHM:
        1. Encode input to Level 0 (H_0)
           - Uses HFCausalEncoder (HuggingFace pretrained model)
           - Projects to hidden_dim via l0_proj
           - Applies RMSNorm for stability

        2. Dynamic Pyramid Expansion Loop:
           while current_level < max_depth:
               a. Compute continuation probability p_cont = depth_gate(H_k)
               b. Check termination: if p_cont < τ OR L_k >= L_max: break
               c. Predict expansion: expand_mask, lambda_k = expansion_predictor(H_k)
               d. Generate next level: H_{k+1} = generator(H_k, expand_mask)
               e. Store level state for loss computation
               f. Increment level counter

        3. Final Projection:
           - Apply token_decoder to deepest level H_K
           - Get logits ∈ [B, L_K, V]

        4. Loss Computation (if training):
           - NTP loss at each level
           - Consistency loss between adjacent levels
           - Depth regularization for expansion rates
           - Cross-entropy alignment

    Attributes:
        config: NLCPModelConfig with all hyperparameters
        encoder: HFCausalEncoder for initial token encoding (HuggingFace-based)
        depth_gate: DepthGate for dynamic depth control (see critic Problem 3)
        expansion_predictor: ExpansionPredictor for λ_k prediction (see critic Problem 1)
        level_generators: List of NextLevelGenerator for each level (see critic Problem 4)
        token_decoder: TokenDecoder for vocabulary projection
        loss_computer: NLCPLossComputer for multi-objective training
        level_embedding: Learnable embeddings for each level
        l0_proj: Projection for Level 0
        l0_norm: RMSNorm for Level 0
    """

    def __init__(
        self,
        config: NLCPModelConfig,
        padding_id: int,
        num_encoder_layers: int,
        num_generator_layers: int,
        use_info_nce: bool,
        info_nce_weight: float,
    ):
        super().__init__()
        self.config = config

        # HuggingFace-based Causal Encoder
        # Reference: Section 2.1 "Input: Question Q (Token IDs) ↓ [Encoder]"
        # Reference: Section 3.1 "Reuse HuggingFace pretrained model weights"
        self.encoder = HFCausalEncoder(
            model_name=config.encoder_model_name,
            num_layers=config.encoder_num_layers,
            l0_length=config.l0_length,
            freeze_encoder=config.encoder_freeze,
        )

        # Dynamic Depth Gate
        # Reference: Section 3.2 "Replaces fixed level count, achieves true pyramid structure"
        # Component selection based on config (critic.md Solution 3B)
        if config.depth_gate_type == "causal":
            self.depth_gate = CausalDepthGate(
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        else:  # "standard"
            self.depth_gate = DepthGate(
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )

        # Content-Adaptive Expansion Predictor
        # Reference: Section 3.3 "Fine level length is not preset, but determined by coarse level semantic density"
        # Component selection based on config (critic.md Solutions 1A-1C)
        if config.expansion_predictor_type == "gumbel":
            self.expansion_predictor = GumbelSoftmaxExpansionPredictor(
                hidden_dim=config.hidden_dim,
                expansion_min=config.expansion_min,
                expansion_max=config.expansion_max,
                dropout=config.dropout,
            )
        elif config.expansion_predictor_type == "reinforce":
            self.expansion_predictor = REINFORCEExpansionPredictor(
                hidden_dim=config.hidden_dim,
                expansion_min=config.expansion_min,
                expansion_max=config.expansion_max,
                dropout=config.dropout,
            )
        elif config.expansion_predictor_type == "soft":
            self.expansion_predictor = SoftExpansionPredictor(
                hidden_dim=config.hidden_dim,
                expansion_min=config.expansion_min,
                expansion_max=config.expansion_max,
                dropout=config.dropout,
            )
        else:  # "floor" (original)
            self.expansion_predictor = ExpansionPredictor(
                hidden_dim=config.hidden_dim,
                expansion_min=config.expansion_min,
                expansion_max=config.expansion_max,
                dropout=config.dropout,
            )

        # Next-Level Generators (one per possible transition)
        # Reference: Section 3.4 "Fine level generation is not coarse upsampling,
        # but strictly conditional autoregressive process on coarse level"
        # Component selection based on config (critic.md Solutions 4A-4B)
        self.level_generators = nn.ModuleList(
            [
                NextLevelGenerator(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_layers=num_generator_layers,
                    dropout=config.dropout,
                    cross_attn_type=config.cross_attention_type,
                )
                for _ in range(config.max_depth)
            ]
        )

        # Token Decoder
        # Reference: Section 2.2 "Latent space → discrete vocabulary mapping"
        self.token_decoder = TokenDecoder(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            muP_scale=1.0,  # Will be set from training config
        )

        # Level embeddings to distinguish pyramid levels
        self.level_embedding = nn.Embedding(config.max_depth + 1, config.hidden_dim)

        # Initial level 0 projection
        self.l0_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.l0_norm = RMSNorm(config.hidden_dim)

        # Loss computer
        # Component selection for consistency loss (critic.md Solutions 2A-2C)
        self.loss_computer = NLCPLossComputer(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            padding_id=padding_id,
            lambda_consist=0.1,  # Will be overridden by training config
            lambda_depth=0.05,
            lambda_ce=1.0,
            target_ratio=4.0,
            use_info_nce=use_info_nce,
            info_nce_weight=info_nce_weight,
            consistency_loss_type=config.consistency_loss_type,
            directional_epsilon=0.5,  # Will be overridden by training config
            mi_temperature=0.07,  # Will be overridden by training config
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        padding_id: int,
        compute_loss: bool = True,
    ) -> NLCPOutput:
        """Forward pass through NLCP model.

        Dimension Flow (from Section 2.1):
            input_ids: [B, L_q] question token IDs
                ↓
            [Encoder]
            H_0: [B, L_0, D] Level 0 hidden states
                ↓
            [Depth Gate] p_cont > τ?
                ↓ Yes
            [Expansion Predictor] λ_0 → expand_mask_0 → L_1
                ↓
            [Next-Level Generator]
            H_1: [B, L_1, D] Level 1 hidden states
                ↓ ... (loop until termination)
            H_K: [B, L_K, D] Final level
                ↓
            [Token Decoder]
            logits: [B, L_K, V] vocabulary logits

        Args:
            input_ids: [B, L_q] input token IDs
            target_ids: [B, L_target] target token IDs for loss computation
            padding_id: Padding token ID
            compute_loss: Whether to compute training losses

        Returns:
            NLCPOutput containing logits, level states, and losses
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        level_states: List[LevelState] = []

        # Step 1: Encode input to Level 0
        # Reference: Section 2.1 "[Lightweight Encoder] H₀ ∈ ℝ^{L₀ × d}"
        H_0 = self.encoder(input_ids)  # [B, L_0, D]
        H_0 = self.l0_proj(H_0)
        H_0 = self.l0_norm(H_0)

        # Add level embedding
        level_0_emb = self.level_embedding(
            torch.zeros(batch_size, H_0.size(1), dtype=torch.long, device=device)
        )
        H_0 = H_0 + level_0_emb

        # Store Level 0 state
        level_states.append(
            LevelState(
                hidden_states=H_0,
                length=H_0.size(1),
                expand_mask=None,
                depth_gate_prob=1.0,  # Always start with Level 0
                kv_cache_self=None,
            )
        )

        # Step 2: Dynamic pyramid expansion
        # Reference: Section 5.1 "Blocking generation algorithm"
        current_hidden = H_0
        current_level = 0

        while current_level < self.config.max_depth:
            # Step 2.1: Depth Gate decision
            # Reference: Section 3.2 "If p_cont^(k) < τ or L_k >= L_max, terminate expansion"
            p_cont = self.depth_gate(current_hidden)

            # Check termination conditions
            should_continue = (
                p_cont.mean() >= self.config.depth_gate_threshold
                and current_hidden.size(1) < self.config.l_max
            )

            if not should_continue:
                break

            # Step 2.2: Predict expansion rates
            # Reference: Section 3.3 "λ_k = Softplus(MLP(H_k))"
            expand_mask, lambda_k = self.expansion_predictor(current_hidden)

            # Compute next level length
            L_next = expand_mask.sum(dim=-1).max().item()
            if L_next <= 0:
                break

            # Step 2.3: Initialize next level hidden states
            # Typically initialized as zeros or learned embeddings
            H_next = torch.zeros(
                batch_size,
                L_next,
                self.config.hidden_dim,
                device=device,
                dtype=current_hidden.dtype,
            )

            # Add level embedding
            level_k_emb = self.level_embedding(
                torch.full(
                    (batch_size, L_next),
                    current_level + 1,
                    dtype=torch.long,
                    device=device,
                )
            )
            H_next = H_next + level_k_emb

            # Step 2.4: Next-Level Generator
            # Reference: Section 3.4 "P(H_{k+1} | H_{<=k}, Q)"
            H_next, _ = self.level_generators[current_level](
                hidden_states=H_next,
                coarse_hidden_states=current_hidden,
                expand_mask=expand_mask,
            )

            # Store level state
            level_states.append(
                LevelState(
                    hidden_states=H_next,
                    length=H_next.size(1),
                    expand_mask=expand_mask,
                    depth_gate_prob=p_cont.mean().item(),
                    kv_cache_self=None,
                )
            )

            # Update for next iteration
            current_hidden = H_next
            current_level += 1

        # Step 3: Token decoding
        # Reference: Section 2.2 "Token Decoder: H_K → Logits ∈ ℝ^{L_K × V}"
        logits = self.token_decoder(current_hidden)

        # Step 4: Compute losses if requested
        total_loss = 0.0
        ntp_loss = 0.0
        consist_loss = 0.0
        depth_loss = 0.0
        ce_loss = 0.0

        if compute_loss:
            total_loss, loss_dict = self.loss_computer(
                level_states=level_states,
                logits=logits,
                target_ids=target_ids,
                padding_id=padding_id,
            )
            ntp_loss = loss_dict["ntp_loss"]
            consist_loss = loss_dict["consist_loss"]
            depth_loss = loss_dict["depth_loss"]
            ce_loss = loss_dict["ce_loss"]

        return NLCPOutput(
            logits=logits,
            level_states=level_states,
            total_loss=total_loss,
            ntp_loss=ntp_loss,
            consist_loss=consist_loss,
            depth_loss=depth_loss,
            ce_loss=ce_loss,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        depth_threshold: float,
    ) -> torch.Tensor:
        """Autoregressive generation with dynamic pyramid.

        Reference: concept-pyramid.md Section 5.1
        Blocking generation algorithm:
            1. Encode to H_0
            2. Loop: depth gate, expansion predictor, next-level generator
            3. Token decode and autoregressive decode

        Dimension Flow:
            input_ids: [B, L_q]
                ↓
            Pyramid forward → H_K: [B, L_K, D]
                ↓
            Token decode → logits: [B, L_K, V]
                ↓
            AR generate tokens

        Args:
            input_ids: [B, L_q] input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            depth_threshold: Depth gate threshold τ

        Returns:
            generated_ids: [B, L_q + max_new_tokens] generated token sequence
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Run pyramid forward pass
        output = self.forward(
            input_ids=input_ids,
            target_ids=input_ids,  # Dummy target, loss not computed
            padding_id=0,
            compute_loss=False,
        )

        # Get final level hidden states
        final_hidden = output.level_states[-1].hidden_states

        # Get initial logits
        logits = self.token_decoder(final_hidden)

        # Start with input_ids
        generated_ids = input_ids.clone()

        # Autoregressive generation
        for _ in range(max_new_tokens):
            # Get last token logits
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                )
                next_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Forward pass for next position (simplified, not full pyramid)
            # In full implementation, would re-run pyramid with updated context
            final_hidden = output.level_states[-1].hidden_states
            logits = self.token_decoder(final_hidden)

        return generated_ids


def build_nlcp_model(
    config: NLCPModelConfig,
    padding_id: int,
    num_encoder_layers: int,
    num_generator_layers: int,
    use_info_nce: bool,
    info_nce_weight: float,
) -> NLCPModel:
    """Build NLCP model from configuration.

    Reference: concept-pyramid.md Section 2
    Architecture Overview

    Args:
        config: Model configuration
        padding_id: Padding token ID
        num_encoder_layers: Number of encoder layers
        num_generator_layers: Number of generator layers per level
        use_info_nce: Whether to use InfoNCE in consistency loss
        info_nce_weight: Weight for InfoNCE term

    Returns:
        NLCPModel instance
    """
    return NLCPModel(
        config=config,
        padding_id=padding_id,
        num_encoder_layers=num_encoder_layers,
        num_generator_layers=num_generator_layers,
        use_info_nce=use_info_nce,
        info_nce_weight=info_nce_weight,
    )
