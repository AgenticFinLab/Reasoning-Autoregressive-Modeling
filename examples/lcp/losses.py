"""Loss functions for lcp (ConceptPyramidBuilder + ConceptPredictor).

This module centralises ALL loss computation logic:

Builder losses (Stage 1):
  - Ordering loss:       margin-based and Gaussian-target variants
  - Reconstruction loss: masked MSE in encoder space
  - Residual loss:       masked L1 in concept space
  - Reasoning loss:      NTP cross-entropy on solution tokens
  - compute_builder_loss: weighted combination of the four losses above

Predictor losses (Stage 2):
  - Concept reconstruction loss: MSE / cosine between predicted and GT
    concepts at every pyramid level (teacher-forcing target).
  - Reasoning loss:       NTP cross-entropy on solution tokens computed
    from [Q, back_proj(predicted_concepts), S].
  - compute_predictor_loss: weighted combination of the two above.

Used by:
    examples/lcp/eval_builder.py     (evaluation loop)
    examples/lcp/train_builder.py    (training loop)
    examples/lcp/train_predictor.py  (predictor training, when added)
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from lcp.concept_builder import PyramidOutput
from lcp.concept_predictor import PredictorOutput

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

    # ── Reasoning loss (NTP: [Q, concepts, S] → predict solution) ─────
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


# ── Predictor loss computation ───────────────────────────────────────
#
# Stage 2 (ConceptPredictor) has two loss components, aligned with
# examples/lcp/concept_predictor.py outputs:
#
#   (1) Concept reconstruction loss — per-level MSE (or cosine) between
#       predicted concepts and ground-truth concepts from the frozen
#       builder. This is analogous to VAR's next-scale token prediction
#       loss, but operating directly in concept space (continuous).
#
#   (2) Reasoning loss — NTP cross-entropy on solution tokens computed
#       by feeding [Q, back_proj(predicted_concepts), S] through the
#       reason_model. Validates that predicted concepts retain enough
#       information to regenerate the solution.
#
# The predictor's forward() already populates PredictorOutput with all
# tensors needed here; this module only computes scalars.


def _concept_loss_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element MSE between predicted and target concepts.

    Args:
        predicted: [B, L_k, D] predicted concept vectors.
        target:    [B, L_k, D] ground-truth concept vectors.

    Returns:
        Scalar MSE averaged over all elements.
    """
    return F.mse_loss(predicted, target)


def _concept_loss_cosine(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity averaged over B * L_k positions.

    Provides a scale-invariant alternative to MSE for concept alignment.

    Args:
        predicted: [B, L_k, D] predicted concept vectors.
        target:    [B, L_k, D] ground-truth concept vectors.

    Returns:
        Scalar cosine distance averaged over B * L_k.
    """
    # cos_sim: [B, L_k]
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    return (1.0 - cos_sim).mean()


def _weighted_level_mean(
    per_level: List[torch.Tensor], level_weights: List[float]
) -> torch.Tensor:
    """Weighted mean across pyramid levels: Σ w_k * L_k / Σ w_k.

    The reduction runs in float32 regardless of the per-level dtype so
    that bf16 / fp16 underflow on tiny ``level_weights`` cannot collapse
    ``Σ w_k`` to zero and produce a silent NaN / Inf. The final scalar
    is cast back to the per-level dtype so the calling autograd graph
    sees the same precision it provided.
    """
    src_dtype = per_level[0].dtype
    device = per_level[0].device
    weights_f32 = torch.tensor(level_weights, device=device, dtype=torch.float32)
    weight_sum = weights_f32.sum()
    if weight_sum.item() <= 0.0:
        raise ValueError(
            "_weighted_level_mean: sum of level_weights must be > 0 "
            f"(got {level_weights})."
        )
    # Promote per-level scalars to float32 for the weighted reduction.
    stacked_f32 = torch.stack([t.to(torch.float32) for t in per_level])
    result_f32 = (stacked_f32 * weights_f32).sum() / weight_sum
    return result_f32.to(src_dtype)


# ── Loss-weights schema validators (fail-fast at process startup) ─────────────────────
#
# These run AT THE TOP of train_*.py / eval_*.py CLIs. Their job is to
# turn schema mismatches (an old YAML resumed against new code, or a
# typo in a new YAML) into a single readable error BEFORE a long
# training run hits the first forward pass and crashes mid-step.
# Both validators enforce the project rule "no defaults in code, all
# knobs in config" — every key listed below must appear in YAML.

_PREDICTOR_REQUIRED_LOSS_KEYS: Tuple[str, ...] = (
    "concept_loss_type",
    "concept_loss_weight",
    "concept_level_weights",
    "reasoning_loss_weight",
    "canvas_loss_weight",
    "canvas_level_weights",
)

_BUILDER_REQUIRED_LOSS_KEYS: Tuple[str, ...] = (
    "recon_loss_weight",
    "ordering_loss_weight",
    "ordering_margin",
    "residual_loss_weight",
    "reasoning_loss_weight",
)


def validate_predictor_loss_weights(loss_weights: dict, num_levels: int) -> None:
    """Fail-fast schema check for ``training.loss_weights`` (predictor).

    Verifies every key in :data:`_PREDICTOR_REQUIRED_LOSS_KEYS` is
    present, that ``concept_loss_type`` is a known value, and that the
    two per-level weight lists have length ``num_levels`` with
    non-negative entries.

    Raises:
        ValueError: on any schema violation, with a message that names
            the offending key and points the user at the YAML.
    """
    if not isinstance(loss_weights, dict):
        raise ValueError(
            f"loss_weights must be a dict, got {type(loss_weights).__name__}."
        )
    missing = [k for k in _PREDICTOR_REQUIRED_LOSS_KEYS if k not in loss_weights]
    if missing:
        raise ValueError(
            f"training.loss_weights missing required keys for predictor: "
            f"{missing}. Add them to the YAML config."
        )
    if loss_weights["concept_loss_type"] not in ("mse", "cosine"):
        raise ValueError(
            "training.loss_weights.concept_loss_type must be 'mse' or "
            f"'cosine', got {loss_weights['concept_loss_type']!r}."
        )
    for key in ("concept_level_weights", "canvas_level_weights"):
        seq = loss_weights[key]
        if not isinstance(seq, (list, tuple)):
            raise ValueError(
                f"training.loss_weights.{key} must be a list, got "
                f"{type(seq).__name__}."
            )
        if len(seq) != num_levels:
            raise ValueError(
                f"training.loss_weights.{key} length {len(seq)} does not "
                f"match model.pyramid.num_levels={num_levels}."
            )
        for i, v in enumerate(seq):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                raise ValueError(
                    f"training.loss_weights.{key}[{i}]={v!r} must be a "
                    f"non-negative number."
                )


def validate_builder_loss_weights(loss_weights: dict) -> None:
    """Fail-fast schema check for ``training.loss_weights`` (builder).

    The builder schema is flat — every weight is a single scalar — so
    presence is the only thing to check. ``reasoning_loss_weight`` is
    listed as required because every shipped builder YAML declares it;
    the value is ignored for solution-less batches.
    """
    if not isinstance(loss_weights, dict):
        raise ValueError(
            f"loss_weights must be a dict, got {type(loss_weights).__name__}."
        )
    missing = [k for k in _BUILDER_REQUIRED_LOSS_KEYS if k not in loss_weights]
    if missing:
        raise ValueError(
            f"training.loss_weights missing required keys for builder: "
            f"{missing}. Add them to the YAML config."
        )


def compute_predictor_concept_loss(
    predicted_concepts: List[torch.Tensor],
    gt_concepts: List[torch.Tensor],
    concept_loss_type: str,
    level_weights: List[float],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Concept reconstruction loss aggregated across pyramid levels.

    Final scalar is a weighted mean: Σ w_k * L_k / Σ w_k.

    Args:
        predicted_concepts: list of K tensors, each [B, L_k, D].
        gt_concepts:        list of K tensors, each [B, L_k, D].
        concept_loss_type:  "mse" or "cosine". REQUIRED (no default).
        level_weights:      list of K floats. REQUIRED (no default).
                            Use [1.0]*K for a uniform mean.
    """
    if len(predicted_concepts) != len(gt_concepts):
        raise ValueError(
            f"predicted_concepts has {len(predicted_concepts)} levels but "
            f"gt_concepts has {len(gt_concepts)}."
        )
    if len(level_weights) != len(predicted_concepts):
        raise ValueError(
            f"concept_level_weights length {len(level_weights)} does not "
            f"match num_levels {len(predicted_concepts)}."
        )
    if concept_loss_type == "mse":
        level_fn = _concept_loss_mse
    elif concept_loss_type == "cosine":
        level_fn = _concept_loss_cosine
    else:
        raise ValueError(f"Unknown concept_loss_type: {concept_loss_type}")

    per_level_losses: List[torch.Tensor] = []
    for predicted, target in zip(predicted_concepts, gt_concepts):
        # Detach target to be safe; GT is from the frozen builder anyway.
        per_level_losses.append(level_fn(predicted, target.detach()))

    total_loss = _weighted_level_mean(per_level_losses, level_weights)
    return total_loss, per_level_losses


def compute_predictor_loss(
    output: PredictorOutput,
    loss_weights: dict,
) -> Tuple[torch.Tensor, dict]:
    """Compute total ConceptPredictor loss = concept + reasoning + canvas.

    PRINCIPLE:
        total = concept_loss_weight   * weighted_mean_k(concept_loss_k)
              + reasoning_loss_weight * reasoning_loss
              + canvas_loss_weight    * weighted_mean_k(canvas_loss_k)

        Each component is computed ONLY when the corresponding output
        tensors are present, so the same function serves training and
        evaluation. Every knob below is read directly from
        ``loss_weights`` (KeyError on missing key); this function
        performs NO defaulting.

    Args:
        output: PredictorOutput from ConceptPredictor.forward().
        loss_weights: dict with the following REQUIRED keys (only the
            keys for components actually computed are looked up):
                "concept_loss_type"      : "mse" or "cosine"
                "concept_loss_weight"    : float
                "concept_level_weights"  : list of K floats
                "reasoning_loss_weight"  : float
                "canvas_loss_weight"     : float
                "canvas_level_weights"   : list of K floats

    Returns:
        (total_loss, loss_dict) — loss_dict records the scalar value of
        each computed component plus per-level breakdowns, for logging.
    """
    loss_dict: dict = {}

    # ── Concept reconstruction loss ─────────────────────────────────────────
    total_loss = None
    if output.gt_concepts is not None and len(output.predicted_concepts) > 0:
        concept_loss, per_level = compute_predictor_concept_loss(
            output.predicted_concepts,
            output.gt_concepts,
            concept_loss_type=loss_weights["concept_loss_type"],
            level_weights=loss_weights["concept_level_weights"],
        )
        loss_dict["concept"] = concept_loss.item()
        loss_dict["concept_per_level"] = [ll.item() for ll in per_level]
        # concept_loss_weight is REQUIRED in config when this branch runs.
        total_loss = loss_weights["concept_loss_weight"] * concept_loss

    # ── Reasoning (NTP) loss ─────────────────────────────────────────────────
    if output.reasoning_logits is not None and output.reasoning_target_ids is not None:
        reasoning_loss = F.cross_entropy(
            output.reasoning_logits.reshape(-1, output.reasoning_logits.shape[-1]),
            output.reasoning_target_ids.reshape(-1),
            ignore_index=-100,
        )
        loss_dict["reasoning"] = reasoning_loss.item()
        # reasoning_loss_weight is REQUIRED in config when this branch runs.
        weighted_reasoning = loss_weights["reasoning_loss_weight"] * reasoning_loss
        total_loss = (
            weighted_reasoning
            if total_loss is None
            else total_loss + weighted_reasoning
        )

    # ── Canvas (soft_boundaries prediction) loss ──────────────────────────
    if (
        output.pred_soft_boundaries is not None
        and output.gt_soft_boundaries is not None
    ):
        canvas_per_level: List[torch.Tensor] = []
        for pred_sb_k, gt_sb_k in zip(
            output.pred_soft_boundaries, output.gt_soft_boundaries
        ):
            canvas_per_level.append(F.mse_loss(pred_sb_k, gt_sb_k))
        canvas_level_weights = loss_weights["canvas_level_weights"]
        if len(canvas_level_weights) != len(canvas_per_level):
            raise ValueError(
                f"canvas_level_weights length {len(canvas_level_weights)} "
                f"does not match num_levels {len(canvas_per_level)}."
            )
        canvas_loss = _weighted_level_mean(canvas_per_level, canvas_level_weights)
        loss_dict["canvas"] = canvas_loss.item()
        loss_dict["canvas_per_level"] = [cl.item() for cl in canvas_per_level]
        # canvas_loss_weight is REQUIRED in config when this branch runs.
        weighted_canvas = loss_weights["canvas_loss_weight"] * canvas_loss
        total_loss = (
            weighted_canvas if total_loss is None else total_loss + weighted_canvas
        )

    if total_loss is None:
        raise ValueError(
            "compute_predictor_loss: no loss components available. Provide "
            "either gt_concepts (for concept loss) or reasoning tensors "
            "(for reasoning loss)."
        )

    loss_dict["total"] = total_loss.item()
    return total_loss, loss_dict
