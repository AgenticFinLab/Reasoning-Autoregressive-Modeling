"""NLCP (Next-Level Concept Pyramid) Loss Functions.

This module implements all loss functions for NLCP training.

DESIGN SOURCE:
    - concept-pyramid.md Section 4 - Pretraining Strategy and Objective Functions
    - concept-pyramid-critic.md - Critical analysis of loss design

COMPLETE LOSS FUNCTION (Section 4.1):
    L_total = Σ_k L_NTP(H_k | H_{<k}, Q)    (hierarchical autoregressive)
            + λ_1 * L_consist               (cross-scale consistency)
            + λ_2 * L_depth                 (expansion rate regularization)
            + λ_3 * L_CE(Tokens | H_K)      (final alignment)

    Weight initialization (Section 4.1):
        λ_1 = 0.1, λ_2 = 0.05, λ_3 = 1.0
        (cosine decay during training)

CRITICAL ISSUES:
    - CrossScaleConsistencyLoss: Too strict L2 constraint (critic Problem 2)
    - ExpansionRateRegularization: Only batch-level, not per-sample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from examples.nlcp.base import LevelState


class NextTokenPredictionLoss(nn.Module):
    """Next Token Prediction Loss for each level.

    Reference: concept-pyramid.md Section 4.1
    "L_NTP: Each level projects to vocabulary and computes standard cross-entropy
    (can share or use independent LM Head)"

    This loss encourages each level's hidden states to predict tokens,
    providing supervision at every pyramid level.
    """

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        # Shared LM head for all levels
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        padding_id: int,
    ) -> torch.Tensor:
        """Compute NTP loss for a single level.

        Dimension Flow:
            H_k: [B, L_k, D] level hidden states
                ↓
            LM Head: [B, L_k, V] vocabulary logits
                ↓
            CrossEntropy: [B, L_k-1] per-position loss
                ↓
            Mean: scalar loss

        Args:
            hidden_states: [B, L_k, D] hidden states at this level
            target_ids: [B, L_target] target token IDs
            padding_id: Padding token ID to ignore

        Returns:
            loss: Scalar NTP loss for this level
        """
        # Project to vocabulary
        logits = self.lm_head(hidden_states)  # [B, L_k, V]

        # Shift for next token prediction
        # logits predict next token, so we compare with shifted targets
        shift_logits = logits[..., :-1, :].contiguous()  # [B, L_k-1, V]
        shift_labels = target_ids[..., 1:].contiguous()  # [B, L_target-1]

        # Handle length mismatch
        if shift_logits.size(1) > shift_labels.size(1):
            # Truncate logits to match target length
            shift_logits = shift_logits[:, : shift_labels.size(1), :]
        elif shift_logits.size(1) < shift_labels.size(1):
            # Truncate labels to match logits length
            shift_labels = shift_labels[:, : shift_logits.size(1)]

        # Compute cross entropy
        loss = self.loss_fn(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        # Reshape and mask padding
        loss = loss.reshape(shift_labels.shape)
        mask = (shift_labels != padding_id).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

        return loss


class CrossScaleConsistencyLoss(nn.Module):
    """Cross-Scale Consistency Regularization Loss.

    DESIGN SOURCE - concept-pyramid.md Section 3.5:
        Formula:
            L_consist = Σ_k ||MeanPool(H_{k+1}, expand_mask_k) - H_k||_2^2
                        + λ_NCE * L_InfoNCE

        Purpose: "Prevent level degradation or attention dilution, provide strong
        supervision gradient anchor points"

        Physical meaning:
            "Force fine level to preserve coarse level semantics after aggregation,
            avoid 'skip coarse level and directly fit fine level' optimization shortcut"

        MeanPool operation:
            Groups fine positions by their parent coarse position,
            computes mean within each group to match coarse dimensions.

    CRITICAL ISSUE - concept-pyramid-critic.md Problem 2:
        "Consistency Loss Creates Information Bottleneck"

        ISSUE DESCRIPTION:
            The loss forces: MeanPool(H_{k+1}) ≈ H_k

            This creates a FUNDAMENTAL CONTRADICTION:
            - Goal A: Fine layer should "expand" and add new information/detail
            - Goal B: After pooling back, it should equal coarse layer (no new info)

        CONCRETE EXAMPLE:
            Coarse Level (H_k): "Calculate average speed"
            Fine Level (H_{k+1}): "t1 = 120/60, t2 = 180/90, v_avg = (d1+d2)/(t1+t2)"

            After MeanPool(H_{k+1}) should equal H_k per consistency loss.
            But H_{k+1} contains NEW information (the actual formulas) not in H_k!

            Mathematical issue:
                To minimize loss, H_{k+1} must satisfy:
                mean(H_{k+1}[i*4:(i+1)*4]) ≈ H_k[i]

                This means H_{k+1} can ONLY add information that averages to zero!
                Any "new" semantic content must be balanced by opposite content.
                This severely limits expressiveness.

    IMPLEMENTATION CHOICE:
        Current implementation (line ~149) uses strict L2:
            consistency_loss = F.mse_loss(pooled_fine, coarse_hidden_states)

        This forces EXACT equality between pooled fine and coarse.
        The model learns to put new information only in variance, not mean.

    RECOMMENDED FIXES - concept-pyramid-critic.md Solutions 2A-2C:

        SOLUTION 2A: Directional Consistency (Relaxed Constraint)
        ```python
        class DirectionalConsistencyLoss(nn.Module):
            def __init__(self, epsilon=0.5):
                self.epsilon = epsilon  # Allow deviation

            def forward(self, H_fine_pooled, H_coarse):
                distance = torch.norm(H_fine_pooled - H_coarse, dim=-1)

                # Hinge loss: only penalize if distance > epsilon
                loss = torch.clamp(distance - self.epsilon, min=0.0).mean()

                return loss

        # Now fine level can deviate up to epsilon!
        # Allows meaningful new information to be added.
        ```

        SOLUTION 2B: Residual-Based Consistency
        ```python
        class ResidualConsistencyLoss(nn.Module):
            def __init__(self, hidden_dim):
                # Learn how much each position can deviate
                self.delta_proj = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, H_fine_pooled, H_coarse):
                # Learnable refinement vector
                delta_H = torch.tanh(self.delta_proj(H_coarse))

                # Target is coarse + refinement, not just coarse
                target = H_coarse + delta_H

                loss = F.mse_loss(H_fine_pooled, target)
                return loss

        # Example:
        # H_coarse[0] = "average speed concept"
        # delta_H[0] = "specific formula: v_avg = total_distance / total_time"
        # H_fine_pooled[0] should match H_coarse[0] + delta_H[0]
        # Now the fine level can add meaningful semantic content!
        ```

        SOLUTION 2C: Information-Theoretic (Mutual Information)
        ```python
        class MutualInformationConsistency(nn.Module):
            def forward(self, H_fine_pooled, H_coarse):
                # Maximize mutual information between coarse and pooled fine
                # This ensures information is preserved without forcing equality

                H_fine_norm = F.normalize(H_fine_pooled, dim=-1)
                H_coarse_norm = F.normalize(H_coarse, dim=-1)

                sim_matrix = torch.matmul(H_fine_norm, H_coarse_norm.T)

                # InfoNCE loss: positive pairs on diagonal
                labels = torch.arange(H_fine_norm.size(0))
                loss = F.cross_entropy(sim_matrix / 0.07, labels)

                return loss

        # Encodes H_fine_pooled to be "predictable" from H_coarse
        # But doesn't force them to be equal
        # H_fine can contain additional information!
        ```

    Attributes:
        use_info_nce: Whether to add InfoNCE contrastive term
        info_nce_weight: Weight for InfoNCE term λ_NCE
    """

    def __init__(self, use_info_nce: bool, info_nce_weight: float):
        super().__init__()
        self.use_info_nce = use_info_nce
        self.info_nce_weight = info_nce_weight

    def forward(
        self,
        fine_hidden_states: torch.Tensor,
        coarse_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-scale consistency loss.

        Dimension Flow:
            H_{k+1}: [B, L_{k+1}, D] fine level states
            H_k: [B, L_k, D] coarse level states
            expand_mask: [B, L_k] expansion counts
                ↓
            MeanPool by groups: [B, L_{k+1}, D] → [B, L_k, D]
                ↓
            L2 distance: [B, L_k, D]
                ↓
            Sum: scalar loss

        Args:
            fine_hidden_states: [B, L_{k+1}, D] fine level hidden states
            coarse_hidden_states: [B, L_k, D] coarse level hidden states
            expand_mask: [B, L_k] expansion counts per coarse position

        Returns:
            loss: Scalar consistency loss
        """
        B = fine_hidden_states.size(0)
        D = fine_hidden_states.size(-1)

        # MeanPool fine level back to coarse level dimensions
        # Group fine positions by their parent coarse position
        pooled_fine = self._mean_pool_by_expand_mask(
            fine_hidden_states, expand_mask
        )  # [B, L_k, D]

        # L2 consistency loss
        consistency_loss = F.mse_loss(pooled_fine, coarse_hidden_states)

        # Optional InfoNCE term
        if self.use_info_nce:
            info_nce_loss = self._compute_info_nce(
                fine_hidden_states, coarse_hidden_states, expand_mask
            )
            consistency_loss = consistency_loss + self.info_nce_weight * info_nce_loss

        return consistency_loss

    def _mean_pool_by_expand_mask(
        self,
        fine_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool fine hidden states by expansion groups.

        Dimension Flow:
            H_{k+1}: [B, L_{k+1}, D]
            expand_mask: [B, L_k] with sum = L_{k+1}
                ↓
            For each coarse position i with expand_mask[i] slots:
                group = H_{k+1}[start:start+expand_mask[i]]
                pooled[i] = mean(group)
                ↓
            Result: [B, L_k, D]

        Args:
            fine_hidden_states: [B, L_{k+1}, D]
            expand_mask: [B, L_k] expansion counts

        Returns:
            pooled: [B, L_k, D] mean pooled representation
        """
        B, L_fine, D = fine_hidden_states.shape
        L_coarse = expand_mask.size(1)

        # Create output tensor
        pooled = torch.zeros(B, L_coarse, D, device=fine_hidden_states.device)

        for b in range(B):
            start_idx = 0
            for i in range(L_coarse):
                count = expand_mask[b, i].item()
                if count > 0:
                    end_idx = start_idx + count
                    pooled[b, i] = fine_hidden_states[b, start_idx:end_idx].mean(dim=0)
                    start_idx = end_idx

        return pooled

    def _compute_info_nce(
        self,
        fine_hidden_states: torch.Tensor,
        coarse_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        Reference: concept-pyramid.md Section 3.5
        "λ_NCE * L_InfoNCE" term in consistency loss

        This encourages each fine position to be most similar to its
        corresponding coarse parent, rather than other coarse positions.

        Args:
            fine_hidden_states: [B, L_{k+1}, D]
            coarse_hidden_states: [B, L_k, D]
            expand_mask: [B, L_k]

        Returns:
            info_nce_loss: Scalar contrastive loss
        """
        B = fine_hidden_states.size(0)
        temperature = 0.07

        # Normalize for cosine similarity
        fine_norm = F.normalize(fine_hidden_states, dim=-1)
        coarse_norm = F.normalize(coarse_hidden_states, dim=-1)

        total_loss = 0.0
        num_pairs = 0

        for b in range(B):
            # Compute similarity matrix
            sim = torch.matmul(fine_norm[b], coarse_norm[b].T) / temperature

            # Create positive pair labels based on expand_mask
            start_idx = 0
            for i in range(expand_mask.size(1)):
                count = expand_mask[b, i].item()
                if count > 0:
                    # Fine positions [start_idx:start_idx+count] have positive i
                    end_idx = start_idx + count
                    pos_sim = sim[start_idx:end_idx, i]  # Positive similarities
                    neg_sim = sim[start_idx:end_idx, :]  # All similarities

                    # InfoNCE: -log(exp(pos) / sum(exp(all)))
                    loss = -pos_sim + torch.logsumexp(neg_sim, dim=-1)
                    total_loss = total_loss + loss.mean()
                    num_pairs = num_pairs + 1

                    start_idx = end_idx

        return total_loss / max(num_pairs, 1)


class ExpansionRateRegularization(nn.Module):
    """Expansion Rate Regularization Loss.

    Reference: concept-pyramid.md Section 3.3
    "Global regularization: L_depth = (1/B * Σ(L_{k+1}/L_k) - R_target)^2,
    R_target ∈ [3, 5]"

    Reference: concept-pyramid.md Section 1.2 Table
    "Global Parser → Global expansion rate regularization loss,
    prevent level collapse or explosion"

    This loss prevents the expansion rate from collapsing to minimum
    or exploding to maximum, encouraging stable pyramid structure.
    """

    def __init__(self, target_ratio: float):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(
        self,
        coarse_length: int,
        fine_length: int,
    ) -> torch.Tensor:
        """Compute expansion rate regularization loss.

        Dimension Flow:
            L_k: scalar coarse level length
            L_{k+1}: scalar fine level length
                ↓
            R = L_{k+1} / L_k: expansion ratio
                ↓
            L_depth = (R - R_target)^2: scalar loss

        Args:
            coarse_length: L_k, coarse level sequence length
            fine_length: L_{k+1}, fine level sequence length

        Returns:
            loss: Scalar regularization loss
        """
        # Compute expansion ratio
        ratio = fine_length / max(coarse_length, 1)

        # Squared deviation from target
        loss = (ratio - self.target_ratio) ** 2

        return loss


class FinalTokenAlignmentLoss(nn.Module):
    """Final Token Alignment Loss.

    Reference: concept-pyramid.md Section 4.1
    "λ_3 * L_CE(Tokens | H_K): final alignment"

    This is the standard cross-entropy loss for the final level's
    predictions against the target tokens.
    """

    def __init__(self, padding_id: int):
        super().__init__()
        self.padding_id = padding_id
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute final token alignment loss.

        Dimension Flow:
            logits: [B, L_K, V] vocabulary logits
            target_ids: [B, L_target] target token IDs
                ↓
            Shift for next token: logits[:-1], targets[1:]
                ↓
            CrossEntropy: scalar loss

        Args:
            logits: [B, L_K, V] vocabulary logits from final level
            target_ids: [B, L_target] target token IDs

        Returns:
            loss: Scalar alignment loss
        """
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # Handle length mismatch
        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]

        # Compute cross entropy with padding mask
        loss = self.loss_fn(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        # Reshape and mask padding
        loss = loss.reshape(shift_labels.shape)
        mask = (shift_labels != self.padding_id).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

        return loss


class NLCPLossComputer(nn.Module):
    """Combined Loss Computer for NLCP.

    Reference: concept-pyramid.md Section 4.1
    Complete Loss Function formula:

    L_total = Σ_k L_NTP(H_k | H_{<k}, Q)    (hierarchical autoregressive)
            + λ_1 * L_consist               (cross-scale consistency)
            + λ_2 * L_depth                 (expansion rate regularization)
            + λ_3 * L_CE(Tokens | H_K)      (final alignment)

    Attributes:
        ntp_loss: Next token prediction loss
        consist_loss: Cross-scale consistency loss
        depth_loss: Expansion rate regularization
        ce_loss: Final token alignment loss
        lambda_consist: Weight for consistency loss
        lambda_depth: Weight for depth loss
        lambda_ce: Weight for CE loss
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        padding_id: int,
        lambda_consist: float,
        lambda_depth: float,
        lambda_ce: float,
        target_ratio: float,
        use_info_nce: bool,
        info_nce_weight: float,
        consistency_loss_type: str = "standard",
        directional_epsilon: float = 0.5,
        mi_temperature: float = 0.07,
    ):
        super().__init__()
        self.ntp_loss = NextTokenPredictionLoss(vocab_size, hidden_dim)

        # Select consistency loss type based on config (critic.md Solutions 2A-2C)
        if consistency_loss_type == "directional":
            self.consist_loss = DirectionalConsistencyLoss(
                epsilon=directional_epsilon,
                use_info_nce=use_info_nce,
                info_nce_weight=info_nce_weight,
            )
        elif consistency_loss_type == "residual":
            self.consist_loss = ResidualConsistencyLoss(
                use_info_nce=use_info_nce,
                info_nce_weight=info_nce_weight,
            )
        elif consistency_loss_type == "mi":
            self.consist_loss = MutualInformationConsistencyLoss(
                temperature=mi_temperature,
            )
        else:  # "standard"
            self.consist_loss = CrossScaleConsistencyLoss(
                use_info_nce=use_info_nce,
                info_nce_weight=info_nce_weight,
            )

        self.depth_loss = ExpansionRateRegularization(target_ratio)
        self.ce_loss = FinalTokenAlignmentLoss(padding_id)

        self.lambda_consist = lambda_consist
        self.lambda_depth = lambda_depth
        self.lambda_ce = lambda_ce

    def forward(
        self,
        level_states: List[LevelState],
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        padding_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total NLCP loss.

        Dimension Flow:
            level_states: List of LevelState for each level k
            logits: [B, L_K, V] final vocabulary logits
            target_ids: [B, L_target] target token IDs
                ↓
            For each level k:
                L_NTP_k = NTP(H_k, targets)
                L_consist_k = Consistency(H_k, H_{k+1}, expand_mask_k)
                L_depth_k = Depth(L_k, L_{k+1})
                ↓
            L_CE = CE(logits, targets)
                ↓
            L_total = Σ L_NTP_k + λ_1 * Σ L_consist_k + λ_2 * Σ L_depth_k + λ_3 * L_CE

        Args:
            level_states: List of LevelState containing hidden states per level
            logits: [B, L_K, V] final vocabulary logits
            target_ids: [B, L_target] target token IDs
            padding_id: Padding token ID

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        device = logits.device
        batch_size = logits.size(0)

        # Initialize loss accumulators
        total_ntp_loss = torch.tensor(0.0, device=device)
        total_consist_loss = torch.tensor(0.0, device=device)
        total_depth_loss = torch.tensor(0.0, device=device)

        # Sum NTP loss over all levels
        for i, state in enumerate(level_states):
            ntp = self.ntp_loss(state.hidden_states, target_ids, padding_id)
            total_ntp_loss = total_ntp_loss + ntp

        # Sum consistency and depth loss over level transitions
        for i in range(len(level_states) - 1):
            coarse_state = level_states[i]
            fine_state = level_states[i + 1]

            # Skip if expand_mask is None (e.g., Level 0)
            if coarse_state.expand_mask is None:
                continue

            # Consistency loss
            consist = self.consist_loss(
                fine_state.hidden_states,
                coarse_state.hidden_states,
                coarse_state.expand_mask,
            )
            total_consist_loss = total_consist_loss + consist

            # Depth regularization
            depth = self.depth_loss(
                coarse_state.length,
                fine_state.length,
            )
            total_depth_loss = total_depth_loss + depth

        # Final token alignment loss
        ce_loss = self.ce_loss(logits, target_ids)

        # Total loss
        total_loss = (
            total_ntp_loss
            + self.lambda_consist * total_consist_loss
            + self.lambda_depth * total_depth_loss
            + self.lambda_ce * ce_loss
        )

        # Loss dictionary for logging
        loss_dict = {
            "ntp_loss": total_ntp_loss.item(),
            "consist_loss": total_consist_loss.item(),
            "depth_loss": total_depth_loss.item(),
            "ce_loss": ce_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict


class DirectionalConsistencyLoss(nn.Module):
    """Directional Consistency Loss (concept-pyramid-critic.md Solution 2A).

    Relaxed version of CrossScaleConsistencyLoss.
    Instead of forcing exact equality (MSE), only requires coarse and fine
    to be "close enough" within an epsilon margin.

    Solves the information bottleneck problem by allowing fine level
    to deviate from coarse level up to epsilon.

    Advantages:
        - Allows fine level to add new information
        - Only penalizes large deviations
        - Hinge loss: no gradient when within epsilon

    Reference: concept-pyramid-critic.md Solution 2A
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        use_info_nce: bool = False,
        info_nce_weight: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.use_info_nce = use_info_nce
        self.info_nce_weight = info_nce_weight

    def _mean_pool_by_expand_mask(
        self,
        fine_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool fine level back to coarse dimensions."""
        B = fine_hidden_states.size(0)
        L_coarse = expand_mask.size(1)
        D = fine_hidden_states.size(-1)

        pooled = torch.zeros(
            B,
            L_coarse,
            D,
            device=fine_hidden_states.device,
            dtype=fine_hidden_states.dtype,
        )

        for b in range(B):
            start_idx = 0
            for i in range(L_coarse):
                count = int(expand_mask[b, i].item())
                if count > 0:
                    end_idx = start_idx + count
                    pooled[b, i] = fine_hidden_states[b, start_idx:end_idx].mean(dim=0)
                    start_idx = end_idx

        return pooled

    def forward(
        self,
        fine_hidden_states: torch.Tensor,
        coarse_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute directional consistency loss with hinge.

        Args:
            fine_hidden_states: [B, L_{k+1}, D] fine level hidden states
            coarse_hidden_states: [B, L_k, D] coarse level hidden states
            expand_mask: [B, L_k] expansion counts per coarse position

        Returns:
            loss: Scalar directional consistency loss
        """
        # MeanPool fine level back to coarse level dimensions
        pooled_fine = self._mean_pool_by_expand_mask(fine_hidden_states, expand_mask)

        # Compute L2 distance per position
        distance = torch.norm(pooled_fine - coarse_hidden_states, dim=-1)  # [B, L_k]

        # Hinge loss: only penalize if distance > epsilon
        loss = torch.clamp(distance - self.epsilon, min=0.0).mean()

        return loss


class ResidualConsistencyLoss(nn.Module):
    """Residual-Based Consistency Loss (concept-pyramid-critic.md Solution 2B).

    Instead of forcing MeanPool(H_fine) ≈ H_coarse,
    learns a refinement vector: MeanPool(H_fine) ≈ H_coarse + ΔH.

    This allows the fine level to add meaningful semantic content
    beyond what's in the coarse level.

    Example:
        H_coarse[0] = "average speed concept"
        ΔH[0] = "specific formula: v_avg = total_distance / total_time"
        Target = H_coarse[0] + ΔH[0]

    Reference: concept-pyramid-critic.md Solution 2B
    """

    def __init__(
        self, hidden_dim: int, use_info_nce: bool = False, info_nce_weight: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable refinement projection
        self.delta_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.use_info_nce = use_info_nce
        self.info_nce_weight = info_nce_weight

    def _mean_pool_by_expand_mask(
        self,
        fine_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool fine level back to coarse dimensions."""
        B = fine_hidden_states.size(0)
        L_coarse = expand_mask.size(1)
        D = fine_hidden_states.size(-1)

        pooled = torch.zeros(
            B,
            L_coarse,
            D,
            device=fine_hidden_states.device,
            dtype=fine_hidden_states.dtype,
        )

        for b in range(B):
            start_idx = 0
            for i in range(L_coarse):
                count = int(expand_mask[b, i].item())
                if count > 0:
                    end_idx = start_idx + count
                    pooled[b, i] = fine_hidden_states[b, start_idx:end_idx].mean(dim=0)
                    start_idx = end_idx

        return pooled

    def forward(
        self,
        fine_hidden_states: torch.Tensor,
        coarse_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residual-based consistency loss.

        Args:
            fine_hidden_states: [B, L_{k+1}, D] fine level hidden states
            coarse_hidden_states: [B, L_k, D] coarse level hidden states
            expand_mask: [B, L_k] expansion counts per coarse position

        Returns:
            loss: Scalar residual consistency loss
        """
        # MeanPool fine level back to coarse level dimensions
        pooled_fine = self._mean_pool_by_expand_mask(fine_hidden_states, expand_mask)

        # Learnable refinement vector
        delta_H = torch.tanh(self.delta_proj(coarse_hidden_states))

        # Target is coarse + refinement, not just coarse
        target = coarse_hidden_states + delta_H

        # MSE loss to target
        loss = F.mse_loss(pooled_fine, target)

        return loss


class MutualInformationConsistencyLoss(nn.Module):
    """Mutual Information Consistency Loss (concept-pyramid-critic.md Solution 2C).

    Information-theoretic approach: maximize mutual information between
    coarse and pooled fine representations.

    Ensures information is preserved without forcing equality.
    Fine level can contain additional information!

    Uses InfoNCE loss as a lower bound on mutual information.

    Reference: concept-pyramid-critic.md Solution 2C
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _mean_pool_by_expand_mask(
        self,
        fine_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool fine level back to coarse dimensions."""
        B = fine_hidden_states.size(0)
        L_coarse = expand_mask.size(1)
        D = fine_hidden_states.size(-1)

        pooled = torch.zeros(
            B,
            L_coarse,
            D,
            device=fine_hidden_states.device,
            dtype=fine_hidden_states.dtype,
        )

        for b in range(B):
            start_idx = 0
            for i in range(L_coarse):
                count = int(expand_mask[b, i].item())
                if count > 0:
                    end_idx = start_idx + count
                    pooled[b, i] = fine_hidden_states[b, start_idx:end_idx].mean(dim=0)
                    start_idx = end_idx

        return pooled

    def forward(
        self,
        fine_hidden_states: torch.Tensor,
        coarse_hidden_states: torch.Tensor,
        expand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mutual information consistency loss.

        Args:
            fine_hidden_states: [B, L_{k+1}, D] fine level hidden states
            coarse_hidden_states: [B, L_k, D] coarse level hidden states
            expand_mask: [B, L_k] expansion counts per coarse position

        Returns:
            loss: Scalar MI consistency loss (InfoNCE)
        """
        # MeanPool fine level back to coarse level dimensions
        pooled_fine = self._mean_pool_by_expand_mask(fine_hidden_states, expand_mask)

        # Normalize for cosine similarity
        pooled_fine_norm = F.normalize(pooled_fine, dim=-1)  # [B, L_k, D]
        coarse_norm = F.normalize(coarse_hidden_states, dim=-1)  # [B, L_k, D]

        # Compute similarity matrix
        # For each coarse position, compute similarity with all pooled fine positions
        B, L_k, D = pooled_fine_norm.shape

        # Reshape for batch matrix multiplication
        pooled_flat = pooled_fine_norm.reshape(B * L_k, D)  # [B*L_k, D]
        coarse_flat = coarse_norm.reshape(B * L_k, D)  # [B*L_k, D]

        # Similarity matrix: [B*L_k, B*L_k]
        sim_matrix = torch.matmul(pooled_flat, coarse_flat.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B * L_k, device=sim_matrix.device)

        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
