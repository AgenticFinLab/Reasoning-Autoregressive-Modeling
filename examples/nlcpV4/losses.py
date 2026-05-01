"""Loss functions for NLCP V4 ConceptPyramidBuilder.

This module centralises all loss computation logic:
  - Ordering loss: margin-based and Gaussian-target variants
  - Reconstruction loss: masked MSE in encoder space
  - Residual loss: masked L1 in concept space
  - compute_builder_loss: weighted combination of the above three

Used by:
    examples/nlcpV4/eval_builder.py  (evaluation loop)
    examples/nlcpV4/train_builder.py (training loop)
"""

import torch
import torch.nn.functional as F

from nlcpV4.concept_builder import PyramidOutput

# ── Ordering loss implementations ────────────────────────────────────


def _ordering_loss_margin(
    attention_weights: torch.Tensor, margin: float
) -> torch.Tensor:
    """Margin-based ordering loss per hybrid-analysis.md Section 5.1.2.

    L_order = Σ_j ReLU(exp_pos[C_j] - exp_pos[C_{j+1}] + margin)
    where exp_pos[C_j] = Σ_t A_j(t) × t

    Args:
        attention_weights: [B, L_k, L] attention weights A_k
        margin: Minimum expected position gap between adjacent concepts

    Returns:
        Scalar ordering loss
    """
    B, Lk, L = attention_weights.shape
    if Lk <= 1:
        return torch.tensor(0.0, device=attention_weights.device)

    positions = torch.arange(L, device=attention_weights.device, dtype=torch.float32)
    # expected_pos: [B, L_k] — expected CoT position for each concept
    expected_pos = (attention_weights * positions.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    loss = torch.tensor(0.0, device=attention_weights.device)
    for j in range(Lk - 1):
        # Enforce: C_j attends to earlier positions than C_{j+1}
        loss = (
            loss + F.relu(expected_pos[:, j] - expected_pos[:, j + 1] + margin).mean()
        )

    return loss


def _ordering_loss_gaussian(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Gaussian-target ordering loss (original implementation).

    Encourages each concept's attention to match a Gaussian centered at
    its expected segment position. Soft but does not explicitly enforce
    monotonic ordering.

    Args:
        attention_weights: [B, L_k, L] attention weights A_k

    Returns:
        Scalar ordering loss
    """
    B, Lk, L = attention_weights.shape
    if Lk <= 1:
        return torch.tensor(0.0, device=attention_weights.device)

    centers = torch.linspace(0, L - 1, Lk, device=attention_weights.device)
    positions = torch.arange(L, device=attention_weights.device).float()
    sigma = max(L / Lk / 2, 1.0)
    target = torch.exp(
        -((positions.unsqueeze(0) - centers.unsqueeze(1)) ** 2) / (2 * sigma**2)
    )
    target = target / target.sum(dim=1, keepdim=True)
    # Average attention across batch: [L_k, L]
    attn = attention_weights.mean(dim=0)
    return -(target * torch.log(attn + 1e-8)).sum(dim=1).mean()


# ── Builder loss computation ─────────────────────────────────────────


def compute_builder_loss(
    pyramid: PyramidOutput,
    loss_weights: dict,
    ordering_loss_type: str,
) -> tuple[torch.Tensor, dict]:
    """Compute all Builder losses: recon + ordering + residual + reasoning.

    Args:
        pyramid: PyramidOutput from builder.forward(), optionally with
            reasoning_logits/reasoning_target_ids populated when
            batch.has_solution (handled automatically by forward()).
        loss_weights: Dict with recon_loss_weight, ordering_loss_weight,
            residual_loss_weight, reasoning_loss_weight, etc.
        ordering_loss_type: "margin" (design doc spec, mandatory) or
            "gaussian" (original soft target). Can also be "both".

    Returns:
        (total_loss, loss_dict)
    """
    loss_dict = {}
    device = pyramid.projected_hidden.device

    # ── Reconstruction loss ──────────────────────────────────────────
    # MSE between back-projected reconstruction and original CoT encodings:
    #   L_recon = ||back_proj(f_hat_K) - H_CoT||^2
    # This measures how well the pyramid preserves the ORIGINAL encoder
    # information, analogous to VAR's reconstruction against frozen encoder output.
    if pyramid.attention_mask is not None:
        # Expand mask for broadcasting: [B, L] -> [B, L, 1]
        mask = pyramid.attention_mask.unsqueeze(-1)
        recon_diff = (
            pyramid.reconstructed_encoder_hidden - pyramid.encoder_hidden_states
        ) * mask
        # Total valid elements = valid_tokens × D_encoder
        num_valid_elements = mask.sum() * pyramid.encoder_hidden_states.shape[-1]
        recon_loss = (recon_diff**2).sum() / num_valid_elements
    else:
        recon_loss = F.mse_loss(
            pyramid.reconstructed_encoder_hidden, pyramid.encoder_hidden_states
        )
    loss_dict["recon"] = recon_loss.item()

    # ── Ordering loss ────────────────────────────────────────────────
    ordering_loss = torch.tensor(0.0, device=device)
    ordering_margin = loss_weights["ordering_margin"]
    levels_with_ordering = 0

    for lo in pyramid.level_outputs:
        Lk = lo.attention_weights.shape[1]
        if Lk <= 1:
            continue
        levels_with_ordering += 1

        if ordering_loss_type == "margin":
            level_order_loss = _ordering_loss_margin(
                lo.attention_weights, margin=ordering_margin
            )
        elif ordering_loss_type == "gaussian":
            level_order_loss = _ordering_loss_gaussian(lo.attention_weights)
        elif ordering_loss_type == "both":
            level_order_loss = _ordering_loss_margin(
                lo.attention_weights, margin=ordering_margin
            ) + _ordering_loss_gaussian(lo.attention_weights)
        else:
            raise ValueError(f"Unknown ordering_loss_type: {ordering_loss_type}")

        ordering_loss = ordering_loss + level_order_loss

    if levels_with_ordering > 0:
        ordering_loss = ordering_loss / levels_with_ordering
    loss_dict["ordering"] = ordering_loss.item()

    # ── Residual loss ────────────────────────────────────────────────
    # L1 averaged over all valid elements (B, L, D), consistent with
    # the per-element mean convention used by reconstruction loss.
    if pyramid.attention_mask is not None:
        mask = pyramid.attention_mask.unsqueeze(-1)
        # Total valid elements = valid_tokens × D
        num_valid_elements = mask.sum() * pyramid.residual_hidden.shape[-1]
        res_loss = (pyramid.residual_hidden.abs() * mask).sum() / num_valid_elements
    else:
        res_loss = pyramid.residual_hidden.abs().mean()
    loss_dict["residual"] = res_loss.item()

    # ── Total loss ───────────────────────────────────────────────────
    residual_weight = loss_weights["residual_loss_weight"]
    total_loss = (
        loss_weights["recon_loss_weight"] * recon_loss
        + loss_weights["ordering_loss_weight"] * ordering_loss
        + residual_weight * res_loss
    )
    loss_dict["total"] = total_loss.item()

    # ── Reasoning loss (NTP: Q + concepts → solution) ──────────────────
    # If prepare_reasoning() was called, pyramid carries logits + target IDs.
    # Cross-entropy is computed here to keep ALL loss logic in losses.py.
    if pyramid.reasoning_logits is not None:
        reasoning_loss = F.cross_entropy(
            pyramid.reasoning_logits.reshape(-1, pyramid.reasoning_logits.shape[-1]),
            pyramid.reasoning_target_ids.reshape(-1),
            # Ignore padding tokens in cross-entropy
            ignore_index=-100,
        )
        loss_dict["reasoning"] = reasoning_loss.item()
        total_loss = total_loss + loss_weights["reasoning_loss_weight"] * reasoning_loss
        loss_dict["total"] = total_loss.item()

    return total_loss, loss_dict
