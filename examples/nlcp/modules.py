"""NLCP (Next-Level Concept Pyramid) Core Modules.

This module implements the core components of NLCP architecture.

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V1.md
    - Section 3.2: Dynamic Depth Gate (p_cont formula, threshold logic)
    - Section 3.3: Content-Adaptive Expansion (lambda_k, expand_mask)
    - Section 3.4: Cross-Level Causal Attention (repeat_interleave, Q/K/V)
    - Section 3.5: Cross-Level Consistency Regularization

    Additional reference: docs/concept-pyramid-critic.md (solutions for V1 issues)

KEY INSIGHT FROM V1 (Section 1.2-1.4):
    NLCP differs from VAR and DLCM in how it ensures "layer-wise approximation to CoT":

    - VAR: Uses f_rest (residual) to tell model "what to encode" at each scale
           Formula: f_rest = z_target - f_hat (residual = what remains to encode)
           Guarantee: Each scale has explicit supervision via residual decomposition

    - DLCM: Concept = Token Pool, naturally contains reconstruction information
            Formula: Concept_k = MeanPool(Tokens in Segment_k)
            Guarantee: Concepts are directly extracted from ground truth CoT

    - NLCP: Uses implicit learning via gradient backprop + consistency constraints
            Formula: L_consist = ||MeanPool(H_{k+1}) - H_k||^2
            Guarantee: Gradient flow from final layer shapes intermediate layers

CRITICAL IMPLEMENTATION GAPS (from concept-pyramid-critic.md):
    - ExpansionPredictor: Uses non-differentiable floor() (Problem 1)
      Solution: GumbelSoftmaxExpansionPredictor, REINFORCEExpansionPredictor, SoftExpansionPredictor
    - DepthGate: Uses full attention instead of causal (Problem 3)
      Solution: CausalDepthGate
    - CrossLevelCausalAttention: Rigid parent-child mapping (Problem 4)
      Solution: RelaxedCrossLevelAttention, HybridCrossLevelAttention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from transformers import AutoModel, AutoTokenizer

from examples.nlcp.base import NLCPModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: concept-pyramid.md Section 3.4
    "RMSNorm stabilizes heterogeneous statistics (DLCM Eq.16)"

    RMSNorm normalizes without mean centering, which is more efficient
    and works well for stabilizing attention across different levels.
    """

    def __init__(self, hidden_dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Dimension Flow:
            Input: [B, L, D] or [B, num_heads, L, head_dim]
                ↓
            Compute RMS: sqrt(mean(x^2) + eps)
                ↓
            Normalize: x / RMS * weight
                ↓
            Output: [B, L, D] or [B, num_heads, L, head_dim]

        Args:
            x: Input tensor of any shape with last dimension hidden_dim

        Returns:
            Normalized tensor with same shape as input
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class DepthGate(nn.Module):
    """Dynamic Depth Gate for controlling pyramid depth.

    DESIGN SOURCE - concept-pyramid.md Section 3.2:
        Dynamic Depth Gate formula:
            p_cont^(k) = σ(MLP_2(GELU(MLP_1(Pool(H_k)))))

        Function: Evaluates whether current latent representation is sufficient
        for final decoding, or if more refinement through additional levels is needed.

        Termination condition (Section 3.2):
            If p_cont^(k) < τ or L_k >= L_max: terminate expansion

    CRITICAL ISSUE - concept-pyramid-critic.md Problem 3:
        "Depth Gate Training-Deployment Mismatch"

        ISSUE DESCRIPTION:
            During training, the gate sees the FULL sequence (teacher forcing).
            During inference, it must decide autoregressively without future tokens.

        CONCRETE EXAMPLE:
            Training:   H_k = [h_1, h_2, h_3, h_4]  # Gate sees all positions
            Inference:  H_k = [h_1, ..., h_pos]     # Gate only sees past

            The gate may learn to be "too confident" during training because
            it has implicit access to information about future complexity.

    IMPLEMENTATION CHOICE:
        Current implementation uses FULL attention pooling (lines 122-128).
        This means the pooling query can attend to ALL positions in H_k,
        including future positions that wouldn't be available during inference.

    RECOMMENDED FIX - concept-pyramid-critic.md Solution 3B:
        Use CausalDepthGate with causal masking:
        ```python
        class CausalDepthGate(nn.Module):
            def forward(self, H_k):
                # Create causal mask: position i can only attend to [0, i]
                causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
                # Now training matches inference!
        ```

    Attributes:
        pool_query: Learnable query for attention pooling [1, 1, D]
        pool_key: Linear projection for keys [D, D]
        pool_value: Linear projection for values [D, D]
        mlp1: First MLP layer [D, 2D]
        mlp2: Second MLP layer [2D, 1]
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        # Learnable pooling via attention mechanism
        # Output: [B, 1, D] global representation
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_key = nn.Linear(hidden_dim, hidden_dim)
        self.pool_value = nn.Linear(hidden_dim, hidden_dim)

        # MLP layers per Section 3.2 formula
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mlp2 = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute continuation probability.

        Dimension Flow:
            H_k: [B, L_k, D] level hidden states
                ↓
            Pool: attention pooling over sequence
                ↓
            pooled: [B, 1, D] global representation
                ↓
            MLP1 + GELU: [B, 1, 2D]
                ↓
            MLP2 + Sigmoid: [B, 1, 1]
                ↓
            p_cont: scalar probability ∈ [0, 1]

        Args:
            hidden_states: [B, L_k, D] level hidden representations
            attention_mask: Optional mask for padding positions

        Returns:
            p_cont: [B, 1] continuation probability
        """
        B, L, D = hidden_states.shape

        # Attention-based pooling
        # Query: [1, 1, D] -> expand to [B, 1, D]
        pool_q = self.pool_query.expand(B, -1, -1)

        # Keys and Values from hidden states
        pool_k = self.pool_key(hidden_states)  # [B, L, D]
        pool_v = self.pool_value(hidden_states)  # [B, L, D]

        # Attention scores: [B, 1, L]
        attn_scores = torch.matmul(pool_q, pool_k.transpose(-2, -1)) / math.sqrt(D)

        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Pooled representation: [B, 1, D]
        pooled = torch.matmul(attn_weights, pool_v)

        # MLP per Section 3.2 formula
        # MLP_1 + GELU
        hidden = F.gelu(self.mlp1(pooled))
        hidden = self.dropout(hidden)

        # MLP_2 + Sigmoid
        p_cont = torch.sigmoid(self.mlp2(hidden))

        return p_cont.squeeze(-1)  # [B, 1]


class ExpansionPredictor(nn.Module):
    """Content-Adaptive Expansion Rate Predictor.

    DESIGN SOURCE - concept-pyramid.md Section 3.3:
        Expansion rate formula:
            λ_k = Softplus(MLP(H_k)) ∈ [1, ∞)^{L_k}
            expand_mask_k = ⌊λ_k⌋
            L_{k+1} = Σ expand_mask_k[i]

        Function: Predicts expansion granularity for each coarse position,
        determining how many fine-level slots each position should expand into.

        Semantic interpretation (Section 3.3):
            λ_k[i] ≈ 4: Logical complex, needs 4 fine concepts
            λ_k[i] ≈ 1: Semantic平稳, no refinement needed

        Global regularization (Section 3.3):
            L_depth = (1/B * Σ(L_{k+1}/L_k) - R_target)^2, R_target ∈ [3, 5]

    CRITICAL ISSUE - concept-pyramid-critic.md Problem 1:
        "Expansion Predictor's Discrete Decision Gradient Flow"

        ISSUE DESCRIPTION:
            The floor operation expand_mask = ⌊λ_k⌋ is NON-DIFFERENTIABLE.
            This breaks gradient flow from downstream losses back to the MLP.

        CONCRETE EXAMPLE:
            lambda_k = [3.7, 2.1, 4.8, 1.9]  # Continuous predictions
            expand_mask = [3, 2, 4, 1]        # After floor

            If λ_k[0] changes from 3.7 → 3.8, expand_mask[0] stays 3.
            Gradient ∇λ_k[0] = 0! Model cannot learn to increase expansion.

            In a math problem:
            - Position 0: "average speed calculation" (needs 4 slots)
            - Model predicts 3.7 → gets 3 slots → under-expansion
            - Cannot learn to predict 4.2 because gradient is zero!

    IMPLEMENTATION CHOICE:
        Current implementation (line ~207) uses:
            expand_mask = torch.floor(lambda_k).long()

        This creates a discontinuity where gradients are zero almost everywhere.
        The model can only learn through indirect L_depth regularization,
        not through direct gradient signal from prediction quality.

    RECOMMENDED FIXES - concept-pyramid-critic.md Solutions 1A-1C:

        SOLUTION 1A: Gumbel-Softmax Relaxation (Recommended)
        ```python
        class DifferentiableExpansionPredictor(nn.Module):
            def forward(self, H_k, temperature=0.5, hard=True):
                logits = self.mlp(H_k)  # [B, L, max_expansion]

                # Gumbel-Softmax: differentiable sampling
                soft_mask = F.gumbel_softmax(logits, tau=temperature, hard=hard)

                # Straight-through estimator
                expansion_values = torch.arange(1, max_expansion + 1)
                expand_mask = (soft_mask * expansion_values).sum(dim=-1)

                if hard:
                    hard_mask = torch.argmax(soft_mask, dim=-1).float() + 1
                    expand_mask = hard_mask + (expand_mask - expand_mask.detach())

                return expand_mask, soft_mask  # Both differentiable!
        ```

        SOLUTION 1B: REINFORCE with Baseline
        ```python
        class REINFORCEExpansionPredictor(nn.Module):
            def forward(self, H_k):
                logits = self.policy_head(H_k)
                dist = torch.distributions.Categorical(logits)
                expansion = dist.sample() + 1
                return expansion, dist

            def compute_loss(self, H_k, reward):
                # reward = -NTP_loss (higher reward = better prediction)
                expansion, dist = self.sample_expansion(H_k)
                baseline = self.baseline_head(H_k.mean(dim=1))
                advantage = reward - baseline.detach()
                policy_loss = -(dist.log_prob(expansion - 1) * advantage).mean()
                return policy_loss
        ```

        SOLUTION 1C: Soft Expansion (Simplest)
        ```python
        class SoftExpansionPredictor(nn.Module):
            def forward(self, H_k):
                raw = torch.sigmoid(self.mlp(H_k).squeeze(-1))
                lambda_k = min_expansion + raw * (max_expansion - min_expansion)
                return lambda_k  # Continuous, fully differentiable
        ```

    Attributes:
        mlp: MLP network [D, D, 1] for expansion prediction
        expansion_min: Minimum expansion rate (default 1)
        expansion_max: Maximum expansion rate (default 8)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_min: int,
        expansion_max: int,
        dropout: float,
    ):
        super().__init__()
        self.expansion_min = expansion_min
        self.expansion_max = expansion_max

        # MLP for expansion rate prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict expansion rates for each position.

        Dimension Flow:
            H_k: [B, L_k, D] level hidden states
                ↓
            MLP: [B, L_k, 1]
                ↓
            Softplus: [B, L_k, 1] positive values
                ↓
            Clamp: [B, L_k, 1] bounded to [expansion_min, expansion_max]
                ↓
            floor: [B, L_k] discrete expansion counts
                ↓
            L_{k+1} = sum(expand_mask) total next level length

        Args:
            hidden_states: [B, L_k, D] level hidden representations
            temperature: Temperature for softening predictions during training

        Returns:
            expand_mask: [B, L_k] integer expansion counts per position
            lambda_k: [B, L_k] continuous expansion rates (for loss computation)
        """
        # MLP prediction
        logits = self.mlp(hidden_states).squeeze(-1)  # [B, L_k]

        # Softplus to ensure positive values, then apply temperature
        lambda_k = F.softplus(logits / temperature)

        # Clamp to valid range
        lambda_k = torch.clamp(lambda_k, self.expansion_min, self.expansion_max)

        # Discrete expansion mask (floor operation)
        # NOTE: This is NON-DIFFERENTIABLE - see critic Problem 1
        expand_mask = torch.floor(lambda_k).long()

        # Ensure at least expansion_min
        expand_mask = torch.clamp(expand_mask, min=self.expansion_min)

        return expand_mask, lambda_k


class GumbelSoftmaxExpansionPredictor(nn.Module):
    """Gumbel-Softmax based Expansion Predictor (concept-pyramid-critic.md Solution 1A).

    This is the RECOMMENDED solution from the critic analysis.
    Uses Gumbel-Softmax relaxation for differentiable discrete sampling.

    Advantages:
        - Fully differentiable during training
        - Can sample discrete values during inference
        - Straight-through estimator for gradient flow

    Reference: concept-pyramid-critic.md Solution 1A
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_min: int,
        expansion_max: int,
        dropout: float,
    ):
        super().__init__()
        self.expansion_min = expansion_min
        self.expansion_max = expansion_max
        self.num_options = expansion_max - expansion_min + 1

        # MLP outputs logits for each expansion option
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_options),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float,
        hard: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict expansion rates using Gumbel-Softmax.

        Args:
            hidden_states: [B, L_k, D] level hidden representations
            temperature: Gumbel-Softmax temperature (lower = more discrete)
            hard: If True, use straight-through estimator

        Returns:
            expand_mask: [B, L_k] expansion counts (differentiable if not hard)
            soft_mask: [B, L_k, num_options] soft probabilities for loss
        """
        B, L, D = hidden_states.shape

        # Get logits for each expansion option
        logits = self.mlp(hidden_states)  # [B, L_k, num_options]

        # Gumbel-Softmax: differentiable sampling
        soft_mask = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        # soft_mask: [B, L_k, num_options], each row sums to 1

        # Expansion values: [expansion_min, ..., expansion_max]
        expansion_values = torch.arange(
            self.expansion_min,
            self.expansion_max + 1,
            dtype=torch.float32,
            device=hidden_states.device,
        )  # [num_options]

        # Expected expansion: [B, L_k]
        expand_mask = (soft_mask * expansion_values).sum(dim=-1)

        if hard:
            # Straight-through estimator for discrete values
            hard_idx = torch.argmax(soft_mask, dim=-1)  # [B, L_k]
            hard_mask = expansion_values[hard_idx]  # [B, L_k]
            # Forward: use hard, Backward: use soft gradient
            expand_mask = hard_mask + (expand_mask - expand_mask.detach())

        return expand_mask, soft_mask


class REINFORCEExpansionPredictor(nn.Module):
    """REINFORCE-based Expansion Predictor (concept-pyramid-critic.md Solution 1B).

    Uses policy gradient methods for discrete decision making.
    Suitable when expansion decisions have delayed rewards.

    Advantages:
        - Naturally handles discrete decisions
        - Can optimize for long-term reward
        - No temperature tuning needed

    Disadvantages:
        - Higher variance in gradients
        - Requires careful baseline design

    Reference: concept-pyramid-critic.md Solution 1B
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_min: int,
        expansion_max: int,
        dropout: float,
    ):
        super().__init__()
        self.expansion_min = expansion_min
        self.expansion_max = expansion_max
        self.num_options = expansion_max - expansion_min + 1

        # Policy head for expansion prediction
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_options),
        )

        # Baseline head for variance reduction
        self.baseline_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        sample: bool,
    ) -> Tuple[torch.Tensor, torch.distributions.Categorical, torch.Tensor]:
        """Sample expansion rates using policy.

        Args:
            hidden_states: [B, L_k, D] level hidden representations
            sample: If True, sample from policy; if False, take argmax

        Returns:
            expansion: [B, L_k] sampled expansion counts
            dist: Categorical distribution for policy gradient
            baseline: [B, 1] baseline value for advantage
        """
        B, L, D = hidden_states.shape

        # Policy logits
        logits = self.policy_head(hidden_states)  # [B, L_k, num_options]

        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)

        # Sample or take argmax
        if sample:
            action = dist.sample()  # [B, L_k], values in [0, num_options-1]
        else:
            action = torch.argmax(logits, dim=-1)  # [B, L_k]

        # Convert to expansion values
        expansion = action + self.expansion_min  # [B, L_k]

        # Baseline for variance reduction
        pooled = hidden_states.mean(dim=1)  # [B, D]
        baseline = self.baseline_head(pooled)  # [B, 1]

        return expansion, dist, baseline

    def compute_loss(
        self,
        dist: torch.distributions.Categorical,
        expansion: torch.Tensor,
        reward: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """Compute REINFORCE policy gradient loss.

        Args:
            dist: Categorical distribution from forward
            expansion: [B, L_k] sampled expansion counts
            reward: [B] reward for each sample (e.g., -NTP_loss)
            baseline: [B, 1] baseline value

        Returns:
            policy_loss: Scalar policy gradient loss
            value_loss: Scalar baseline loss
        """
        # Convert expansion to action indices
        action = expansion - self.expansion_min  # [B, L_k]

        # Compute log probabilities
        log_prob = dist.log_prob(action)  # [B, L_k]

        # Advantage: reward - baseline
        advantage = reward.unsqueeze(-1) - baseline.detach()  # [B, 1]

        # REINFORCE: maximize expected reward
        # Policy loss: -log_prob * advantage
        policy_loss = -(log_prob * advantage).mean()

        # Baseline loss: MSE between baseline and reward
        value_loss = F.mse_loss(baseline.squeeze(-1), reward)

        return policy_loss, value_loss


class SoftExpansionPredictor(nn.Module):
    """Soft Expansion Predictor (concept-pyramid-critic.md Solution 1C).

    Simplest solution: continuous expansion without discretization.
    Uses sigmoid to bound expansion rates in [min, max].

    Advantages:
        - Fully differentiable, no tricks needed
        - Simplest implementation
        - No hyperparameter tuning (temperature, etc.)

    Disadvantages:
        - Expansion rates are continuous, not discrete
        - May not align with discrete token positions

    Reference: concept-pyramid-critic.md Solution 1C
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_min: int,
        expansion_max: int,
        dropout: float,
    ):
        super().__init__()
        self.expansion_min = expansion_min
        self.expansion_max = expansion_max

        # MLP outputs single value per position
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict continuous expansion rates.

        Args:
            hidden_states: [B, L_k, D] level hidden representations
            temperature: Not used (for API compatibility)

        Returns:
            lambda_k: [B, L_k] continuous expansion rates
            lambda_k: [B, L_k] same as first return (for API compatibility)
        """
        # MLP prediction
        logits = self.mlp(hidden_states).squeeze(-1)  # [B, L_k]

        # Sigmoid to [0, 1], then scale to [expansion_min, expansion_max]
        raw = torch.sigmoid(logits)
        lambda_k = self.expansion_min + raw * (self.expansion_max - self.expansion_min)

        return lambda_k, lambda_k  # Return twice for API compatibility


class CausalDepthGate(nn.Module):
    """Causal Depth Gate (concept-pyramid-critic.md Solution 3B).

    Fixes the training-deployment mismatch in the original DepthGate.
    Uses causal attention pooling so training matches inference.

    Original issue: During training, gate sees full sequence;
    during inference, gate only sees past positions.

    Solution: Apply causal masking so each position can only attend to previous positions.

    Reference: concept-pyramid-critic.md Solution 3B
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        # Learnable pooling via attention mechanism
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_key = nn.Linear(hidden_dim, hidden_dim)
        self.pool_value = nn.Linear(hidden_dim, hidden_dim)

        # MLP layers
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mlp2 = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute continuation probability with causal pooling.

        Args:
            hidden_states: [B, L_k, D] level hidden representations
            attention_mask: Optional mask for padding positions

        Returns:
            p_cont: [B, 1] continuation probability
        """
        B, L, D = hidden_states.shape

        # Keys and Values from hidden states
        pool_k = self.pool_key(hidden_states)  # [B, L, D]
        pool_v = self.pool_value(hidden_states)  # [B, L, D]

        # Query: [1, 1, D] -> expand to [B, 1, D]
        pool_q = self.pool_query.expand(B, -1, -1)

        # Attention scores: [B, 1, L]
        attn_scores = torch.matmul(pool_q, pool_k.transpose(-2, -1)) / math.sqrt(D)

        # Apply causal mask: each position can only attend to previous positions
        # For global pooling, we use cumulative attention
        # Create causal mask: [1, L] where position i can attend to [0, i]
        causal_mask = torch.ones(1, L, device=hidden_states.device)
        causal_mask = torch.cumsum(causal_mask, dim=-1)  # [1, 2, 3, ..., L]
        causal_mask = causal_mask / causal_mask.max()  # Normalize

        # Apply causal weighting to attention scores
        attn_scores = attn_scores * causal_mask.unsqueeze(0)

        # Apply padding mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Pooled representation: [B, 1, D]
        pooled = torch.matmul(attn_weights, pool_v)

        # MLP
        hidden = F.gelu(self.mlp1(pooled))
        hidden = self.dropout(hidden)
        p_cont = torch.sigmoid(self.mlp2(hidden))

        return p_cont.squeeze(-1)  # [B, 1]


class CrossLevelCausalAttention(nn.Module):
    """Cross-Level Causal Attention mechanism.

    DESIGN SOURCE - concept-pyramid.md Section 3.4:
        Causal Cross-Level Attention with Concept Replication.

        Formula:
            P(H_{k+1} | H_{≤k}, Q) = ∏_j P(h_{k+1}^j | h_{k+1}^{<j}, H_k, Q)

        Key insight from Section 3.4:
            "repeat_interleave makes irregular mapping degenerate to standard
            L_{k+1} × L_{k+1} Causal Mask"

        Tensor alignment (Section 3.4 code block):
            K_k = H_k @ W_K          # [B, L_k, D]
            V_k = H_k @ W_V          # [B, L_k, D]
            K_rep = repeat_interleave(K_k, expand_mask, dim=1)  # [B, L_{k+1}, D]
            V_rep = repeat_interleave(V_k, expand_mask, dim=1)  # [B, L_{k+1}, D]
            Q_{k+1} = H_{k+1} @ W_Q  # [B, L_{k+1}, D]
            Q' = RMSNorm(Q_{k+1}), K' = RMSNorm(K_rep)          # DLCM Eq.16
            AttnOut = FlashAttn(Q', K', V_rep, causal=True)     # Causal mask
            H_{k+1} = AttnOut @ W_O + H_{k+1}                   # Residual

    CRITICAL ISSUE - concept-pyramid-critic.md Problem 4:
        "Cross-Level Attention's Rigid Parent-Child Mapping"

        ISSUE DESCRIPTION:
            The repeat_interleave approach assumes strict monotonic parent-child:
            - Fine position [0,1,2] → attend ONLY to Coarse[0]
            - Fine position [3,4,5] → attend ONLY to Coarse[1]

            This is too restrictive for natural language where context flows
            across concept boundaries.

        CONCRETE EXAMPLE:
            Coarse Level: ["Problem setup", "Step 1: Define variables", "Step 2: Write equations"]

            Fine Level position: "From the problem, we define variables"
            - Needs context from BOTH "Problem setup" AND "Step 1"
            - But repeat_interleave forces it to attend to only ONE parent!

            Current mapping (rigid):
                Fine[3] (parent=1) → Coarse[1] only

            Desired mapping (flexible):
                Fine[3] (parent=1) → Coarse[0] AND Coarse[1] (both relevant)

    IMPLEMENTATION CHOICE:
        Current implementation (lines 370-371) uses strict repeat_interleave:
            k_rep = self._repeat_interleave_batch(k_coarse, expand_mask)
            v_rep = self._repeat_interleave_batch(v_coarse, expand_mask)

        This enforces 1-to-many mapping where each fine position can only
        attend to its assigned parent coarse position.

    RECOMMENDED FIXES - concept-pyramid-critic.md Solutions 4A-4B:

        SOLUTION 4A: Relaxed Cross-Attention with Soft Weights
        ```python
        class RelaxedCrossLevelAttention(nn.Module):
            def forward(self, H_fine, H_coarse, parent_indices):
                # parent_indices: primary parent for each fine position
                Q = self.q_proj(H_fine)      # [B, L_fine, D]
                K = self.k_proj(H_coarse)    # [B, L_coarse, D]
                V = self.v_proj(H_coarse)    # [B, L_coarse, D]

                scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, L_fine, L_coarse]

                # Causal mask: fine[i] can attend to coarse[j] iff j <= parent_indices[i]
                parent_indices_expanded = parent_indices.unsqueeze(-1)  # [B, L_fine, 1]
                coarse_indices = torch.arange(L_coarse).unsqueeze(0).unsqueeze(0)
                causal_mask = (coarse_indices > parent_indices_expanded).float() * float('-inf')
                scores = scores + causal_mask

                # Now fine[i] can attend to ANY coarse position up to its parent!
                attn_weights = F.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, V)
                return output
        ```

        SOLUTION 4B: Hybrid Attention (Local + Global)
        ```python
        class HybridCrossLevelAttention(nn.Module):
            def __init__(self, hidden_dim, num_heads):
                self.local_attn = CrossLevelCausalAttention(hidden_dim, num_heads)
                self.global_attn = nn.MultiheadAttention(hidden_dim, num_heads)
                self.gate = nn.Linear(hidden_dim * 2, 1)

            def forward(self, H_fine, H_coarse, expand_mask):
                # Local: strict parent-child
                local_out = self.local_attn(H_fine, H_coarse, expand_mask)

                # Global: can attend to any coarse position
                global_out, _ = self.global_attn(H_fine, H_coarse, H_coarse)

                # Learnable gate to combine
                gate_input = torch.cat([local_out, global_out], dim=-1)
                gate = torch.sigmoid(self.gate(gate_input))

                output = gate * local_out + (1 - gate) * global_out
                return output

            # For boundary positions: gate → 0 (use global)
            # For within-step positions: gate → 1 (use local)
        ```

    Attributes:
        num_heads: Number of attention heads H
        head_dim: Dimension per head d_head = d/H
        q_proj: Query projection W_Q [D, D]
        k_proj: Key projection W_K [D, D]
        v_proj: Value projection W_V [D, D]
        o_proj: Output projection W_O [D, D]
        q_norm: RMSNorm for query stabilization (DLCM Eq.16)
        k_norm: RMSNorm for key stabilization (DLCM Eq.16)
    """

    def __init__(
        self, hidden_dim: int, num_heads: int, dropout: float, rms_norm_eps: float
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections per Section 3.4 code
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # RMSNorm for QK normalization (DLCM Eq.16)
        self.q_norm = RMSNorm(hidden_dim, rms_norm_eps)
        self.k_norm = RMSNorm(hidden_dim, rms_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def _repeat_interleave_batch(
        self,
        x: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Repeat interleave with batch dimension handling.

        Dimension Flow:
            x: [B, L_k, D] coarse level tensor
            expand_mask: [B, L_k] expansion counts per position
                ↓
            For each batch element:
                repeat_interleave along sequence dimension
                ↓
            Result: [B, L_{k+1}, D] where L_{k+1} = sum(expand_mask)

        Args:
            x: [B, L_k, D] input tensor
            expand_mask: [B, L_k] expansion counts

        Returns:
            output: [B, L_{k+1}, D] repeated tensor
        """
        batch_size = x.size(0)
        results = []

        for b in range(batch_size):
            # Get expansion counts for this batch element
            repeats = expand_mask[b].cpu()  # Move to CPU for repeat_interleave
            # Ensure repeats are non-negative integers
            repeats = torch.clamp(repeats, min=0).long()

            # Apply repeat_interleave for this batch element
            repeated = torch.repeat_interleave(x[b], repeats, dim=0)
            results.append(repeated)

        # Pad to maximum length across batch
        max_len = max(r.size(0) for r in results)
        padded_results = []
        for r in results:
            if r.size(0) < max_len:
                padding = torch.zeros(
                    max_len - r.size(0), r.size(-1), device=r.device, dtype=r.dtype
                )
                r = torch.cat([r, padding], dim=0)
            padded_results.append(r)

        return torch.stack(padded_results, dim=0)

    def forward(
        self,
        hidden_states_fine: torch.Tensor,
        hidden_states_coarse: torch.Tensor,
        expand_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply cross-level causal attention.

        Dimension Flow (from Section 3.4):
            Coarse H_k: [B, L_k, D]
                ↓
            K_k = H_k @ W_K: [B, L_k, D]
            V_k = H_k @ W_V: [B, L_k, D]
                ↓
            K_rep = repeat_interleave(K_k, expand_mask): [B, L_{k+1}, D]
            V_rep = repeat_interleave(V_k, expand_mask): [B, L_{k+1}, D]
                ↓
            Fine Q_{k+1} = H_{k+1} @ W_Q: [B, L_{k+1}, D]
                ↓
            Q' = RMSNorm(Q), K' = RMSNorm(K_rep)
                ↓
            AttnOut = FlashAttn(Q', K', V_rep, causal=True): [B, L_{k+1}, D]
                ↓
            Output = AttnOut @ W_O + H_{k+1}: [B, L_{k+1}, D]

        Args:
            hidden_states_fine: [B, L_{k+1}, D] fine level hidden states
            hidden_states_coarse: [B, L_k, D] coarse level hidden states
            expand_mask: [B, L_k] expansion counts per coarse position
            attention_mask: Optional causal mask

        Returns:
            output: [B, L_{k+1}, D] attention output with residual
        """
        B = hidden_states_fine.size(0)

        # Project queries from fine level
        q = self.q_proj(hidden_states_fine)  # [B, L_{k+1}, D]

        # Project keys and values from coarse level
        k_coarse = self.k_proj(hidden_states_coarse)  # [B, L_k, D]
        v_coarse = self.v_proj(hidden_states_coarse)  # [B, L_k, D]

        # Concept Replication: repeat_interleave to align with fine level
        # This is the core DLCM trick from Eq.17
        # Handle batch dimension: repeat_interleave each batch element separately
        k_rep = self._repeat_interleave_batch(k_coarse, expand_mask)  # [B, L_{k+1}, D]
        v_rep = self._repeat_interleave_batch(v_coarse, expand_mask)  # [B, L_{k+1}, D]

        # RMSNorm for QK stabilization (DLCM Eq.16) - apply BEFORE reshaping
        q = self.q_norm(q)  # [B, L_{k+1}, D]
        k = self.k_norm(k_rep)  # [B, L_{k+1}, D]

        # Reshape for multi-head attention
        # [B, L, D] -> [B, num_heads, L, head_dim]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_rep.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation with causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask (upper triangular = -inf)
        L_fine = attn_weights.size(-2)
        causal_mask = torch.triu(
            torch.full((L_fine, L_fine), float("-inf"), device=attn_weights.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, L_{k+1}, head_dim]

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, -1, self.hidden_dim)
        )

        # Output projection + residual (DLCM Eq.14)
        output = self.o_proj(attn_output) + hidden_states_fine

        return output


class SelfAttentionBlock(nn.Module):
    """Self-Attention Block with Causal Masking.

    Reference: concept-pyramid.md Section 3.4
    "Standard FlashAttention (Varlen compatible)"
    "Fine level Self-Attn Query"

    Standard causal self-attention for within-level processing.
    Used in Next-Level Generator for fine-grained reasoning within each level.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        q_proj, k_proj, v_proj: QKV projections
        o_proj: Output projection
        q_norm, k_norm: RMSNorm for QK normalization
        mlp: Feed-forward network
    """

    def __init__(
        self, hidden_dim: int, num_heads: int, dropout: float, rms_norm_eps: float
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Self-attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # RMSNorm
        self.q_norm = RMSNorm(hidden_dim, rms_norm_eps)
        self.k_norm = RMSNorm(hidden_dim, rms_norm_eps)
        self.attn_dropout = nn.Dropout(dropout)

        # MLP (standard 4x expansion)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.ln1 = RMSNorm(hidden_dim, rms_norm_eps)
        self.ln2 = RMSNorm(hidden_dim, rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[List[torch.Tensor]],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Apply causal self-attention.

        Dimension Flow:
            H: [B, L, D] hidden states
                ↓
            Self-Attn with causal mask: [B, L, D]
                ↓
            + residual: [B, L, D]
                ↓
            MLP: [B, L, D]
                ↓
            + residual: [B, L, D]

        Args:
            hidden_states: [B, L, D] input hidden states
            kv_cache: Optional list of [K_cache, V_cache] for incremental generation
            use_cache: Whether to return updated KV cache

        Returns:
            output: [B, L, D] output hidden states
            kv_cache: Updated KV cache if use_cache=True
        """
        B, L, D = hidden_states.shape
        residual = hidden_states

        # Pre-norm
        hidden_states = self.ln1(hidden_states)

        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Handle KV cache for incremental generation
        if kv_cache is not None and len(kv_cache) == 2:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        new_kv_cache = None
        if use_cache:
            new_kv_cache = [k, v]

        # RMSNorm for QK - apply BEFORE reshaping
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape for multi-head attention
        # [B, L, D] -> [B, num_heads, L, head_dim]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        seq_len = attn_weights.size(-2)
        kv_len = attn_weights.size(-1)
        causal_mask = torch.triu(
            torch.full((seq_len, kv_len), float("-inf"), device=attn_weights.device),
            diagonal=kv_len - seq_len + 1,
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, D)

        # Output projection + residual
        hidden_states = self.o_proj(attn_output) + residual

        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual

        return hidden_states, new_kv_cache


class NextLevelGenerator(nn.Module):
    """Next-Level Generator for hierarchical concept generation.

    Reference: concept-pyramid.md Section 3.4
    "Fine level generation is not coarse level upsampling, but a
    strictly conditional autoregressive process on coarse level"

    Reference: concept-pyramid.md Section 2.2 Table
    "With coarse level as condition, autoregressively generate
    fine level concept representations"

    This module generates the next level's hidden representations
    conditioned on the current level through cross-level attention
    and self-attention blocks.

    CONFIGURABLE: cross_attn_type selects from critic.md solutions
        - "standard": CrossLevelCausalAttention (original, rigid)
        - "relaxed": RelaxedCrossLevelAttention (Solution 4A, RECOMMENDED)
        - "hybrid": HybridCrossLevelAttention (Solution 4B)

    Attributes:
        cross_attn: Cross-level causal attention (configurable type)
        self_attn_layers: Stack of self-attention layers
        ln: Final layer norm
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        rms_norm_eps: float,
        cross_attn_type: str,
    ):
        super().__init__()
        # Select cross-attention type based on config (critic.md Solutions 4A-4B)
        if cross_attn_type == "relaxed":
            self.cross_attn = RelaxedCrossLevelAttention(
                hidden_dim, num_heads, dropout, rms_norm_eps
            )
        elif cross_attn_type == "hybrid":
            self.cross_attn = HybridCrossLevelAttention(
                hidden_dim, num_heads, dropout, rms_norm_eps
            )
        else:  # "standard"
            self.cross_attn = CrossLevelCausalAttention(
                hidden_dim, num_heads, dropout, rms_norm_eps
            )

        self.self_attn_layers = nn.ModuleList(
            [
                SelfAttentionBlock(hidden_dim, num_heads, dropout, rms_norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.ln = RMSNorm(hidden_dim, rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        coarse_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
        kv_cache: Optional[List[List[torch.Tensor]]],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[List[List[torch.Tensor]]]]:
        """Generate next level hidden states.

        Dimension Flow:
            H_{k+1} init: [B, L_{k+1}, D] (typically zeros or learned embedding)
                ↓
            Cross-Level Attn(H_{k+1}, H_k, expand_mask): [B, L_{k+1}, D]
                ↓
            Self-Attn layers × N: [B, L_{k+1}, D]
                ↓
            RMSNorm: [B, L_{k+1}, D]

        Args:
            hidden_states: [B, L_{k+1}, D] initial fine level states
            coarse_hidden_states: [B, L_k, D] coarse level states
            expand_mask: [B, L_k] expansion counts
            kv_cache: Optional KV caches for each self-attention layer
            use_cache: Whether to return updated caches

        Returns:
            output: [B, L_{k+1}, D] generated fine level representations
            new_kv_cache: Updated KV caches if use_cache=True
        """
        # Cross-level attention injects coarse level prior
        hidden_states = self.cross_attn(
            hidden_states,
            coarse_hidden_states,
            expand_mask,
        )

        # Self-attention for fine-grained reasoning
        new_kv_cache = [] if use_cache else None
        for i, self_attn in enumerate(self.self_attn_layers):
            layer_kv = (
                kv_cache[i] if kv_cache is not None and i < len(kv_cache) else None
            )
            hidden_states, layer_new_kv = self_attn(
                hidden_states,
                kv_cache=layer_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_cache.append(layer_new_kv)

        # Final normalization
        hidden_states = self.ln(hidden_states)

        return hidden_states, new_kv_cache


class TokenDecoder(nn.Module):
    """Token Decoder for vocabulary projection.

    Reference: concept-pyramid.md Section 2.2 Table
    "Latent space → discrete vocabulary mapping"

    Reference: concept-pyramid.md Section 4.2
    "Output layer scaling: logits = (1/s_token)(H_K @ W_unemb^T)
    ensures logits magnitude is O(1) (DLCM Eq.21)"

    This module projects the final level's hidden states to vocabulary
    logits for autoregressive token generation.

    Attributes:
        lm_head: Linear projection to vocabulary
        muP_scale: Output scaling factor for μP
    """

    def __init__(self, hidden_dim: int, vocab_size: int, muP_scale: float):
        super().__init__()
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.muP_scale = muP_scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Dimension Flow:
            H_K: [B, L_K, D] final level hidden states
                ↓
            Linear projection: [B, L_K, V]
                ↓
            μP scaling: [B, L_K, V] logits with O(1) magnitude

        Args:
            hidden_states: [B, L_K, D] final level hidden representations

        Returns:
            logits: [B, L_K, V] vocabulary logits
        """
        # Linear projection to vocabulary
        logits = self.lm_head(hidden_states)

        # μP scaling per DLCM Eq.21
        logits = logits / self.muP_scale

        return logits


class RelaxedCrossLevelAttention(nn.Module):
    """Relaxed Cross-Level Attention (concept-pyramid-critic.md Solution 4A).

    Fixes the rigid parent-child mapping in standard CrossLevelCausalAttention.
    Allows fine positions to attend to ANY previous coarse position,
    not just their assigned parent.

    Key insight: Fine position at concept boundary needs context from
    multiple coarse positions (e.g., "From the problem, we define variables"
    needs both "Problem setup" and "Step 1" context).

    Reference: concept-pyramid-critic.md Solution 4A
    """

    def __init__(
        self, hidden_dim: int, num_heads: int, dropout: float, rms_norm_eps: float
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # RMSNorm
        self.q_norm = RMSNorm(hidden_dim, rms_norm_eps)
        self.k_norm = RMSNorm(hidden_dim, rms_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def _compute_parent_indices(
        self,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute parent index for each fine position.

        Args:
            expand_mask: [B, L_coarse] expansion counts per coarse position

        Returns:
            parent_indices: [B, L_fine] coarse index for each fine position
        """
        B = expand_mask.size(0)
        L_coarse = expand_mask.size(1)

        # Compute L_fine = sum(expand_mask)
        L_fine = expand_mask.sum(dim=1).max().item()

        parent_indices = []
        for b in range(B):
            indices = []
            for coarse_idx, count in enumerate(expand_mask[b]):
                count = count.item()
                indices.extend([coarse_idx] * count)
            # Pad to L_fine
            while len(indices) < L_fine:
                indices.append(L_coarse - 1)  # Pad with last coarse position
            parent_indices.append(indices)

        return torch.tensor(parent_indices, device=expand_mask.device)

    def forward(
        self,
        hidden_states_fine: torch.Tensor,
        hidden_states_coarse: torch.Tensor,
        expand_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply relaxed cross-level attention.

        Args:
            hidden_states_fine: [B, L_fine, D] fine level hidden states
            hidden_states_coarse: [B, L_coarse, D] coarse level hidden states
            expand_mask: [B, L_coarse] expansion counts per coarse position
            attention_mask: Optional mask

        Returns:
            output: [B, L_fine, D] attention output
        """
        B = hidden_states_fine.size(0)
        L_fine = hidden_states_fine.size(1)
        L_coarse = hidden_states_coarse.size(1)

        # Projections
        q = self.q_proj(hidden_states_fine)  # [B, L_fine, D]
        k = self.k_proj(hidden_states_coarse)  # [B, L_coarse, D]
        v = self.v_proj(hidden_states_coarse)  # [B, L_coarse, D]

        # Compute parent indices
        parent_indices = self._compute_parent_indices(expand_mask)  # [B, L_fine]

        # Attention scores: [B, L_fine, L_coarse]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Relaxed causal mask: fine[i] can attend to coarse[j] iff j <= parent_indices[i]
        # This allows attending to ANY previous coarse position, not just parent
        parent_indices_expanded = parent_indices.unsqueeze(-1)  # [B, L_fine, 1]
        coarse_indices = torch.arange(L_coarse, device=scores.device).view(1, 1, -1)
        causal_mask = (coarse_indices > parent_indices_expanded).float() * float("-inf")
        scores = scores + causal_mask

        # Apply padding mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        output = torch.matmul(attn_weights, v)  # [B, L_fine, D]

        # Output projection
        output = self.o_proj(output)

        # Residual connection
        output = output + hidden_states_fine

        return output


class HybridCrossLevelAttention(nn.Module):
    """Hybrid Cross-Level Attention (concept-pyramid-critic.md Solution 4B).

    Combines local (strict parent-child) and global (attend to any) attention
    with a learnable gate. For boundary positions, uses global attention;
    for within-step positions, uses local attention.

    Advantages:
        - Preserves hierarchical structure where needed
        - Allows flexible context at boundaries
        - Learnable mixing via gating mechanism

    Reference: concept-pyramid-critic.md Solution 4B
    """

    def __init__(
        self, hidden_dim: int, num_heads: int, dropout: float, rms_norm_eps: float
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Local attention: strict parent-child (original behavior)
        self.local_attn = CrossLevelCausalAttention(
            hidden_dim, num_heads, dropout, rms_norm_eps
        )

        # Global attention: can attend to any coarse position
        self.global_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate for mixing local and global
        self.gate_proj = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = (hidden_dim // num_heads) ** -0.5

    def forward(
        self,
        hidden_states_fine: torch.Tensor,
        hidden_states_coarse: torch.Tensor,
        expand_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply hybrid cross-level attention.

        Args:
            hidden_states_fine: [B, L_fine, D] fine level hidden states
            hidden_states_coarse: [B, L_coarse, D] coarse level hidden states
            expand_mask: [B, L_coarse] expansion counts per coarse position
            attention_mask: Optional mask

        Returns:
            output: [B, L_fine, D] attention output
        """
        B, L_fine, D = hidden_states_fine.shape
        L_coarse = hidden_states_coarse.size(1)

        # Local attention (strict parent-child)
        local_out = self.local_attn(
            hidden_states_fine, hidden_states_coarse, expand_mask, attention_mask
        )  # [B, L_fine, D]

        # Global attention (attend to any coarse position)
        q_global = self.global_q_proj(hidden_states_fine)  # [B, L_fine, D]
        k_global = self.global_k_proj(hidden_states_coarse)  # [B, L_coarse, D]
        v_global = self.global_v_proj(hidden_states_coarse)  # [B, L_coarse, D]

        # Reshape for multi-head
        q_global = q_global.view(B, L_fine, self.num_heads, -1).transpose(1, 2)
        k_global = k_global.view(B, L_coarse, self.num_heads, -1).transpose(1, 2)
        v_global = v_global.view(B, L_coarse, self.num_heads, -1).transpose(1, 2)

        # Global attention scores
        global_scores = torch.matmul(q_global, k_global.transpose(-2, -1)) * self.scale

        # Causal mask: fine[i] can attend to coarse[j] for any j
        # (no strict parent constraint for global)
        global_weights = F.softmax(global_scores, dim=-1)
        global_weights = self.dropout(global_weights)

        # Global attention output
        global_out = torch.matmul(
            global_weights, v_global
        )  # [B, num_heads, L_fine, head_dim]
        global_out = global_out.transpose(1, 2).reshape(B, L_fine, D)
        global_out = self.global_o_proj(global_out)

        # Learnable gate
        gate_input = torch.cat([local_out, global_out], dim=-1)  # [B, L_fine, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, L_fine, 1]

        # Mix local and global
        output = gate * local_out + (1 - gate) * global_out

        return output


# =============================================================================
# HuggingFace-based Causal Transformer Encoder
# =============================================================================


class HFCausalEncoder(nn.Module):
    """HuggingFace-based Causal Transformer Encoder.

    Reference: concept-pyramid-V1.md Section 2.3.1
    "Encoder: Standard Causal Transformer（与 DLCM 完全一致）"

    Design Principles:
        1. Reuse HuggingFace transformers framework
        2. Reuse pretrained open-source LLM weights
        3. Load pretrained model and use only first N layers as encoder

    Key Distinction:
        Use Model class (e.g., Qwen2Model), NOT ForCausalLM class:
        - Qwen2ForCausalLM: includes lm_head for token prediction
        - Qwen2Model: pure Transformer backbone for feature extraction

    Supported Models:
        | Architecture | HF Model Class  | Features              |
        |--------------|-----------------|----------------------|
        | GPT-2        | GPT2Model       | Standard Causal      |
        | Llama        | LlamaModel      | RoPE + RMSNorm       |
        | Qwen         | Qwen2Model      | RoPE + RMSNorm + SwiGLU |

    Attributes:
        model: HuggingFace model (e.g., Qwen2Model)
        tokenizer: Associated tokenizer
        pool_to_l0: Adaptive pooling to L_0 length
        l0_proj: Optional projection layer
    """

    def __init__(
        self,
        model_name: str,
        num_layers: Optional[int],
        l0_length: int,
        freeze_encoder: bool,
        **kwargs,
    ):
        """Initialize HuggingFace-based Encoder.

        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-0.5B")
            num_layers: Number of layers to use (None = use all layers)
            l0_length: Target length for Level 0 concepts
            freeze_encoder: Whether to freeze encoder weights
            **kwargs: Additional arguments passed to from_pretrained
        """
        super().__init__()

        # Load model (Model class, not ForCausalLM)
        # AutoModel automatically selects the correct Model class
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

        # Get hidden dimension from model config
        self.hidden_dim = self.model.config.hidden_size
        self.l0_length = l0_length

        # Optional: truncate to fewer layers (DLCM-style lightweight encoder)
        if num_layers is not None:
            self._truncate_layers(num_layers)

        # Freeze encoder weights if requested
        if freeze_encoder:
            self._freeze_encoder()

        # Pool & Project (must be trained from scratch)
        self.pool_to_l0 = nn.AdaptiveAvgPool1d(l0_length)
        self.l0_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _truncate_layers(self, num_layers: int):
        """Truncate model to use only first N layers.

        DLCM Encoder is typically lightweight (4-6 layers).
        This method keeps only the first N layers of the pretrained model.

        Args:
            num_layers: Number of layers to keep
        """
        # Handle different model architectures
        if hasattr(self.model, "layers"):
            # Qwen2, Llama style
            original_layers = self.model.layers
            self.model.layers = original_layers[:num_layers]
        elif hasattr(self.model, "h"):
            # GPT-2 style architecture
            original_layers = self.model.h
            self.model.h = original_layers[:num_layers]
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            # BERT style architecture (not causal, handle gracefully)
            original_layers = self.model.encoder.layer
            self.model.encoder.layer = original_layers[:num_layers]
        else:
            raise ValueError(
                f"Unknown model architecture: {type(self.model)}. "
                "Cannot truncate layers."
            )

    def _freeze_encoder(self):
        """Freeze encoder weights to save computation."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder weights for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode input tokens to Level 0 hidden states.

        Dimension Flow:
            input_ids: [B, L_q] token IDs
                ↓
            HF Model (Causal Transformer): [B, L_q, D]
                ↓
            Pool to L_0: [B, D, L_q] → [B, D, L_0] → [B, L_0, D]
                ↓
            Project: [B, L_0, D]
                ↓
            H_0: [B, L_0, D] Level 0 hidden representations

        Args:
            input_ids: [B, L_q] input token IDs
            attention_mask: Optional attention mask

        Returns:
            H_0: [B, L_0, D] Level 0 hidden representations
        """
        # HF Model forward pass (automatic causal mask)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get last hidden state: [B, L_q, D]
        hidden_states = outputs.last_hidden_state

        # Pool to Level 0 length
        # [B, L_q, D] -> [B, D, L_q] -> pool -> [B, D, L_0] -> [B, L_0, D]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pool_to_l0(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        # Project
        hidden_states = self.l0_proj(hidden_states)

        return hidden_states

    @property
    def device(self):
        """Get model device."""
        return next(self.model.parameters()).device

    def to(self, *args, **kwargs):
        """Move model to device."""
        self.model = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)
