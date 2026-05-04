"""Train NLCP V4 ConceptPredictor (Stage 2).

Usage:
    # Basic: train with the config's own log paths (relative to CWD by default).
    python3 examples/nlcpV4/train_predictor.py \
        -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml

    # Redirect ALL relative outputs (save_folder/checkpoint_path/log_path)
    # under a storage root — typical on a shared server where the
    # project-local EXPERIMENT/ tree is not writable.  Relative Builder
    # checkpoint paths (model.builder.checkpoint_path) are ALSO resolved
    # under this same root so Stage 2 can find the Stage 1 artefacts the
    # previous builder run wrote.
    python3 examples/nlcpV4/train_predictor.py \
        -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml \
        -s /Data/<proj>

    # Resume from the latest checkpoint under log.checkpoint_path.
    python3 examples/nlcpV4/train_predictor.py \
        -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml --resume

    # Resume AND pin the SwanLab run explicitly (rare).
    python3 examples/nlcpV4/train_predictor.py \
        -c configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml \
        --resume --swanlab-id 5hjp09vuqh402irzz9j9h

Stage-2 contract (vs Stage-1 / train_builder.py):
  - A FROZEN ConceptPyramidBuilder produces gt_concepts (and nothing else)
    per batch.  We build a solution-free BuilderInput before calling
    builder.forward so the builder's own reasoning path is SKIPPED —
    otherwise in `use_shared_model=True` mode we would run reason_model
    twice per step (once in the builder for a result we throw away,
    once in the predictor for the actual gradient).
  - ConceptPredictor.forward runs ONE unified teacher-forced pass over
    [Q, C_gt, S] and returns a PredictorOutput with both readouts.
    `compute_predictor_loss` owns all loss math.
  - Evaluation is inlined (`evaluate_predictor` below) because the
    predictor's output schema differs from the builder's
    (concept+reasoning only, no recon / ordering / residual).
  - Config inheritance: predictor configs do NOT re-declare
    `model.pyramid`.  This script reads the paired builder YAML
    (`model.builder.config_path`) and injects its `model.pyramid` block
    into the predictor config before constructing ConceptPredictor.

Arguments:
    -s / --storage-root   Prefix prepended to RELATIVE log paths in the
                          YAML (save_folder/checkpoint_path/log_path) AS
                          WELL AS the relative Builder checkpoint path
                          (model.builder.checkpoint_path).  Absolute
                          values are preserved.  Default is "./".
    -c / --config         Path to a predictor YAML config.
    --resume              Auto-discover the latest predictor checkpoint
                          under log.checkpoint_path and resume.  Pure
                          CLI concern — no YAML field.
    --swanlab-id          SwanLab run id to resume.  Only consulted with
                          --resume.  Precedence: CLI > logs/<exp>/swanlab.json
                          > hard error.
"""

import argparse
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import swanlab
import torch
from dotenv import load_dotenv
from torch.optim import AdamW
from tqdm import tqdm

# Project-root path injection must precede local imports so that the
# ``nlcpV4``, ``lmbase``, and ``ram`` packages resolve when this script
# is executed directly (``python3 examples/nlcpV4/train_predictor.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from nlcpV4 import _resume_io
from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.concept_predictor import ConceptPredictor
from nlcpV4.data_loader import BuilderInput, NLCPV4DataLoader
from nlcpV4.eval_builder import log_terminal_entry
from nlcpV4.losses import compute_predictor_loss
from ram.utils import apply_storage_root, load_config, print_storage_paths

# =============================================================================
# Environment / seeding
# =============================================================================


def _seed_single_device(seed: int, device: str) -> None:
    """Seed RNGs for CPU + the chosen CUDA device only.

    Rationale identical to train_builder.py._seed_single_device — see
    that module for the full explanation of why manual_seed_all is
    avoided on shared multi-GPU hosts.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device.startswith("cuda:"):
        dev_idx = int(device.split(":")[1])
        with torch.cuda.device(dev_idx):
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Config inheritance + fail-fast sanity checks
# =============================================================================


def _resolve_config_path(raw: str, base: Path) -> Path:
    """Resolve a YAML path: absolute stays, relative joins ``base``."""
    p = Path(raw)
    return p if p.is_absolute() else base / p


def _inherit_pyramid_from_builder(predictor_config: dict, builder_config: dict) -> None:
    """Copy ``model.pyramid`` from the builder config into the predictor config.

    The predictor YAML intentionally does NOT re-declare ``model.pyramid``
    (geometry drift between Stage 1 and Stage 2 would silently corrupt
    training).  This helper is the single place the inheritance
    happens.  Fail-fast if the predictor config already has a
    ``model.pyramid`` block — that would defeat the inheritance guarantee.
    """
    if "model" not in predictor_config:
        raise ValueError("Predictor config missing top-level 'model' block.")
    if "pyramid" in predictor_config["model"]:
        raise ValueError(
            "Predictor config must NOT declare 'model.pyramid' directly; "
            "it is inherited from the builder config pointed to by "
            "'model.builder.config_path'.  Remove the pyramid block "
            "and let the trainer inject it."
        )
    if "model" not in builder_config or "pyramid" not in builder_config["model"]:
        raise ValueError(
            "Builder config does not expose 'model.pyramid'; predictor "
            "inheritance is broken.  Check the paired builder YAML."
        )
    predictor_config["model"]["pyramid"] = builder_config["model"]["pyramid"]


def _fail_fast_shared_sanity(config: dict) -> None:
    """Enforce the ``use_shared_model`` ⇒ ``lora is None`` constraint.

    Attaching LoRA to the shared reason_model would mutate the
    Builder's own forward pass during training, poisoning its
    gt_concepts output.  The constraint is documented in both the
    shared and independent YAML templates; we fail fast here so a
    mis-edited YAML can never reach the optimizer.
    """
    use_shared = config["model"]["predictor"]["use_shared_model"]
    lora = config["training"]["predictor"]["lora"]
    if use_shared and lora is not None:
        raise ValueError(
            "use_shared_model=True requires training.predictor.lora=null. "
            "LoRA on the shared reason_model would mutate the Builder's "
            "backbone mid-training and invalidate gt_concepts."
        )
    if (not use_shared) and lora is None:
        # Independent mode with no LoRA and freeze=True would leave
        # the reason_model completely untrained — a footgun, not a
        # valid config.  Surface it immediately.
        freeze = config["training"]["predictor"]["freeze"]
        if freeze:
            raise ValueError(
                "use_shared_model=False + lora=null + freeze=True leaves "
                "the predictor's reason_model untrained.  Either attach "
                "a LoRA block, set freeze=False, or switch to "
                "use_shared_model=True."
            )


def _resolve_builder_checkpoint_path(raw: str, storage_root: str) -> Path:
    """Mirror ``apply_storage_root`` semantics for the builder checkpoint,
    with prefix-based glob fallback.

    ``apply_storage_root`` only rewrites paths under ``config['log']``.
    The builder checkpoint (``model.builder.checkpoint_path``) lives
    outside that subtree but follows the exact same convention — if
    relative, it is resolved under the CLI storage root so that a
    Stage-2 run launched with ``-s /Data/<proj>`` can locate Stage-1
    artefacts written by the paired Stage-1 run (also launched with
    the same ``-s``).

    Glob fallback:
        Builder training saves best checkpoints with epoch/step suffixes,
        e.g. ``checkpoint_best_eval-epoch9-step18500.pt``.  The YAML
        config carries only the *stem prefix* for brevity:
            ``checkpoint_best_eval.pt``
        If the literal path does not exist, this function strips the
        ``.pt`` suffix, globs ``<stem>*.pt`` in the parent directory,
        and returns the match with the highest step number.  This lets
        the config remain stable across training runs (no manual
        epoch/step editing after each Stage-1 completion).
    """
    p = Path(raw)
    if p.is_absolute():
        resolved = p
    else:
        resolved = Path(storage_root) / p

    # Fast path: exact file exists.
    if resolved.is_file():
        return resolved

    # Glob fallback: treat filename as a prefix.
    parent = resolved.parent
    stem_prefix = resolved.stem  # e.g. "checkpoint_best_eval"
    if parent.is_dir():
        # Match files like checkpoint_best_eval-epoch9-step18500.pt
        candidates = sorted(
            parent.glob(f"{stem_prefix}*.pt"),
            key=lambda f: _extract_step(f.name),
            reverse=True,
        )
        if candidates:
            return candidates[0]

    # No match — return the literal path so downstream FileNotFoundError
    # still reports the expected location.
    return resolved


def _extract_step(filename: str) -> int:
    """Extract the step number from a checkpoint filename for sorting.

    Handles patterns like:
        checkpoint_best_eval-epoch9-step18500.pt  → 18500
        checkpoint_best-epoch5-step10000.pt       → 10000
    Falls back to 0 if no step found.
    """
    m = re.search(r"-step(\d+)", filename)
    return int(m.group(1)) if m else 0


# =============================================================================
# Builder loading (Stage 1 frozen dependency)
# =============================================================================


def _load_frozen_builder(
    builder_config: dict,
    checkpoint_path: Path,
    strict: bool,
    device: str,
    logger: logging.Logger,
) -> ConceptPyramidBuilder:
    """Construct the Builder, load its checkpoint, freeze everything.

    The builder is used in Stage-2 training ONLY to produce gt_concepts
    per batch; its parameters never receive gradients.  We therefore:
      * ``requires_grad = False`` on every parameter,
      * ``builder.eval()`` so dropout / LayerNorm are deterministic,
      * ``builder(batch)`` is always called inside ``torch.no_grad()``
        in the training loop below.

    Args:
        builder_config: The PARSED builder YAML (already loaded).
        checkpoint_path: Resolved absolute path to the builder
            checkpoint.  Storage-root prefix already applied.
        strict: When True, refuse to load a checkpoint with
            missing/unexpected keys — protects against silent schema
            drift between Stage-1 training and Stage-2 consumption.
        device: Target device (``"cuda:0"`` / ``"cpu"``).
        logger: Trainer logger for startup messages.

    Returns:
        The loaded, frozen, eval-mode builder module.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Builder checkpoint not found: {checkpoint_path}.  "
            f"Finish Stage-1 training first or point "
            f"model.builder.checkpoint_path at the right file."
        )
    builder = ConceptPyramidBuilder(builder_config)
    builder.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    missing, unexpected = builder.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict builder load failed.\n"
            f"  missing keys    : {missing}\n"
            f"  unexpected keys : {unexpected}\n"
            f"  checkpoint      : {checkpoint_path}"
        )
    if missing or unexpected:
        logger.warning(
            "Builder loaded with strict=False | missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    for p in builder.parameters():
        p.requires_grad = False
    builder.eval()
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_step = ckpt.get("step", "?")
    logger.info(
        "Builder loaded (epoch=%s step=%s) from %s",
        ckpt_epoch,
        ckpt_step,
        checkpoint_path,
    )
    return builder


# =============================================================================
# Predictor model summary (mirrors train_builder._log_model_summary in shape)
# =============================================================================


def _count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _count_trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _log_predictor_summary(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    config: dict,
    logger: logging.Logger,
) -> None:
    """Emit a detailed trainable-surface summary for the predictor.

    The report lists WHICH modules are owned vs shared vs frozen, and
    accumulates their parameter counts so the operator can sanity-check
    the trainable footprint before a long run begins.
    """
    pyramid_cfg = config["model"]["pyramid"]
    pred_cfg = config["model"]["predictor"]
    train_pred_cfg = config["training"]["predictor"]
    loss_weights = config["training"]["loss_weights"]

    use_shared = pred_cfg["use_shared_model"]

    total_params = sum(p.numel() for p in predictor.parameters())
    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    level_lengths = pyramid_cfg["level_lengths"]
    D = pyramid_cfg["hidden_dim"]
    D_enc = predictor.reason_model_hidden_dim
    total_C = sum(level_lengths)

    lvl_emb = _count(predictor.level_embeddings)
    pos_emb = _count(predictor.position_embeddings)
    head = _count(predictor.concept_head)
    back_proj = _count(predictor.back_proj)
    back_proj_train = _count_trainable(predictor.back_proj)
    reason_total = _count(predictor.reason_model)
    reason_train = _count_trainable(predictor.reason_model)

    mode_label = (
        "SHARED (tied to Builder, frozen)"
        if use_shared
        else "INDEPENDENT (own reason_model)"
    )

    lines = [
        "",
        "=" * 72,
        "  PREDICTOR ARCHITECTURE SUMMARY",
        "=" * 72,
        "",
        "  Mode                : %s" % mode_label,
        "  use_shared_model    : %s" % use_shared,
        "",
        "  Reason Model",
        "  ├─ encoder_dim      : %d" % D_enc,
        "  ├─ vocab_size       : %d" % predictor.reason_model.config.vocab_size,
        "  ├─ num_layers       : %d" % predictor.reason_model.config.num_hidden_layers,
        "  ├─ freeze           : %s"
        % ("(forced) True" if use_shared else train_pred_cfg["freeze"]),
        "  ├─ lora             : %s" % (train_pred_cfg["lora"] or "None"),
        "  ├─ params           : %s (trainable: %s)"
        % (f"{reason_total:,}", f"{reason_train:,}"),
        "",
        "  Pyramid Geometry (inherited from builder)",
        "  ├─ hidden_dim (D)   : %d" % D,
        "  ├─ num_levels (K)   : %d" % pyramid_cfg["num_levels"],
        "  ├─ level_lengths    : %s  (total_C: %d)" % (level_lengths, total_C),
        "  ├─ max_seq_len      : %d" % pyramid_cfg["max_seq_len"],
        "",
        "  Predictor-owned Modules   Shape                 Params",
        "  " + "-" * 68,
        "  level_embeddings         : [%d, %d]              %s"
        % (pyramid_cfg["num_levels"], D_enc, f"{lvl_emb:,}"),
        "  position_embeddings      : [%d, %d]             %s"
        % (max(level_lengths), D_enc, f"{pos_emb:,}"),
        "  concept_head             : [%d→%d→%d] MLP       %s"
        % (D_enc, D_enc, D, f"{head:,}"),
        "  back_proj                : [%d, %d]              %s  (trainable: %s, %s)"
        % (
            D,
            D_enc,
            f"{back_proj:,}",
            f"{back_proj_train:,}",
            "shared" if use_shared else "owned",
        ),
        "",
        "  Loss Weights",
        "  ├─ concept            : %s" % loss_weights["concept_loss_weight"],
        "  ├─ reasoning          : %s" % loss_weights["reasoning_loss_weight"],
        "",
        "  Parameter Summary",
        "  ├─ total              : %s" % f"{total_params:,}",
        "  ├─ trainable          : %s  (%.2f%%)"
        % (f"{trainable_params:,}", 100.0 * trainable_params / max(1, total_params)),
        "  └─ frozen             : %s" % f"{frozen_params:,}",
        "=" * 72,
        "",
    ]
    for line in lines:
        logger.info(line)


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for the predictor trainer."""
    parser = argparse.ArgumentParser(description="Train ConceptPredictor (NLCP V4)")
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix to prepend to every RELATIVE path in the config: "
            "log.save_folder / log.checkpoint_path / log.log_path AND "
            "model.builder.checkpoint_path.  Absolute paths pass through "
            "verbatim.  Default is './' — never an implicit project root."
        ),
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to predictor YAML config"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume from the latest predictor checkpoint auto-discovered "
            "under log.checkpoint_path (pattern "
            "'checkpoint*-epoch<N>-step<M>.pt').  Boolean flag — no path "
            "argument.  Resume is a pure CLI concern; YAML configs carry "
            "no 'training.resume' field."
        ),
    )
    parser.add_argument(
        "--swanlab-id",
        type=str,
        default="",
        help=(
            "SwanLab run id to resume.  Only consulted with --resume. "
            "Precedence: CLI > logs/<exp>/swanlab.json > hard error."
        ),
    )
    return parser.parse_args()


# =============================================================================
# Checkpoint I/O (predictor-only state — builder is an external dependency)
# =============================================================================


def save_checkpoint(
    predictor: ConceptPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: Path,
    filename: str,
    best_eval_loss: float = float("inf"),
) -> Path:
    """Save predictor-only state to ``checkpoint_dir / filename``.

    The Builder is intentionally NOT persisted here: it is a frozen
    Stage-1 artefact located via ``model.builder.checkpoint_path`` in
    the config.  Re-saving it would just bloat checkpoints and invite
    accidental divergence between the Stage-1 source of truth and
    embedded copies.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "best_eval_loss": best_eval_loss,
        "model_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    path = checkpoint_dir / filename
    torch.save(checkpoint, path)
    return path


def purge_best_checkpoints(checkpoint_dir: Path, prefix: str) -> None:
    """Remove previous best files matching ``{prefix}-*.pt`` / ``{prefix}.pt``."""
    for old in checkpoint_dir.glob(f"{prefix}-*.pt"):
        try:
            old.unlink()
        except OSError:
            pass
    legacy = checkpoint_dir / f"{prefix}.pt"
    if legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass


def load_checkpoint(
    checkpoint_path: Path,
    predictor: ConceptPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, int, float, float]:
    """Load predictor checkpoint; returns ``(epoch, step, loss, best_eval_loss)``."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    predictor.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    loss_value = checkpoint.get("loss", checkpoint.get("eval_loss", float("inf")))
    return (
        int(checkpoint["epoch"]),
        int(checkpoint["step"]),
        float(loss_value),
        float(checkpoint.get("best_eval_loss", float("inf"))),
    )


# =============================================================================
# One training-step forward (builder → tokens → predictor)
# =============================================================================


def _strip_solutions(batch: BuilderInput) -> BuilderInput:
    """Return a clone of ``batch`` with ``solutions=[]`` so the Builder
    forward skips its own ``_prepare_reasoning`` path.

    The predictor computes reasoning CE itself inside the unified
    teacher-forced forward — letting the Builder run its reasoning
    step too would double the reason_model forwards (wasteful in both
    modes, and in SHARED mode specifically runs the SAME module tree
    twice).
    """
    return BuilderInput(
        questions=list(batch.questions),
        cot_answers=list(batch.cot_answers),
        solutions=[],
        main_ids=list(batch.main_ids),
    )


def _tokenize_qs(
    builder: ConceptPyramidBuilder,
    batch: BuilderInput,
    max_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize questions and solutions with the Builder's tokenizer.

    Using the Builder's tokenizer guarantees that Q/S token ids live in
    the SAME vocabulary the Builder's reason_model (and, in shared
    mode, the Predictor's reason_model) expects.
    """
    tokenizer = builder.tokenizer
    q = tokenizer(
        batch.questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    s = tokenizer(
        batch.solutions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return (
        q["input_ids"].to(device),
        q["attention_mask"].to(device),
        s["input_ids"].to(device),
        s["attention_mask"].to(device),
    )


def _run_predictor_step(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    batch: BuilderInput,
    max_length: int,
    device: str,
):
    """Build gt_concepts (frozen builder), tokenize, call predictor.

    Returns the full PredictorOutput; the caller computes the loss.
    """
    with torch.no_grad():
        pyramid = builder(_strip_solutions(batch))
        gt_concepts = [c.detach() for c in pyramid.concepts]

    q_ids, q_mask, s_ids, s_mask = _tokenize_qs(builder, batch, max_length, device)

    output = predictor(
        question_ids=q_ids,
        question_attention_mask=q_mask,
        gt_concepts=gt_concepts,
        solution_ids=s_ids,
        solution_attention_mask=s_mask,
    )
    return output


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_predictor(
    predictor: ConceptPredictor,
    builder: ConceptPyramidBuilder,
    eval_dataloader: NLCPV4DataLoader,
    loss_weights: dict,
    max_length: int,
    device: str,
    max_batches: int,
) -> tuple[dict, list[str], list[dict]]:
    """Run eval on ``max_batches`` of test data; return averaged losses.

    Returns:
        (averaged_loss_dict, reasoning_texts, samples)
          - averaged_loss_dict keys: total, concept, reasoning (optional),
            concept_per_level (list[float], averaged elementwise).
          - reasoning_texts: teacher-forced argmax decodes.
          - samples: per-row metadata for eval_sample_history.json.
    """
    predictor.eval()
    all_losses: list[dict] = []
    all_texts: list[str] = []
    all_samples: list[dict] = []

    for i, batch in enumerate(eval_dataloader):
        if max_batches > 0 and i >= max_batches:
            break

        output = _run_predictor_step(predictor, builder, batch, max_length, device)
        _, loss_dict = compute_predictor_loss(
            output, loss_weights, concept_loss_type="mse"
        )
        all_losses.append(loss_dict)

        if output.reasoning_texts is not None:
            all_texts.extend(output.reasoning_texts)

        for j in range(batch.batch_size):
            all_samples.append(
                {
                    "batch_idx": i,
                    "pos_in_batch": j,
                    "main_id": batch.main_ids[j],
                    "question": batch.questions[j],
                    "solution": batch.solutions[j] if batch.has_solution else None,
                }
            )

    predictor.train()
    # Keep the shared reason_model (if any) and the (in independent
    # mode) frozen reason_model base in eval — dropout on a frozen
    # backbone is a no-op for gradients but can still add noise to
    # activations flowing into LoRA / concept_head.
    if hasattr(predictor, "reason_model"):
        predictor.reason_model.eval()

    if not all_losses:
        return ({"total": 0.0, "concept": 0.0}, [], [])

    # Average scalar keys across batches.  ``concept_per_level`` is a
    # list and needs elementwise averaging.
    scalar_keys = [k for k in all_losses[0].keys() if k != "concept_per_level"]
    avg = {k: sum(d[k] for d in all_losses) / len(all_losses) for k in scalar_keys}

    if "concept_per_level" in all_losses[0]:
        per_level = list(zip(*[d["concept_per_level"] for d in all_losses]))
        avg["concept_per_level"] = [sum(col) / len(col) for col in per_level]

    return avg, all_texts, all_samples


def log_eval_results_predictor(
    eval_losses: dict,
    loss_weights: dict,
    eval_type: str,
    global_step: int,
    terminal_log_path: Path,
    eval_history: list,
    log_dir: Path,
    swanlab_prefix: str,
    reasoning_texts: list[str],
    eval_samples: list[dict],
    eval_sample_history: list,
    logger: logging.Logger,
) -> None:
    """Console + SwanLab + eval_history + sample-history writer for eval.

    Mirrors ``eval_builder.log_eval_results`` in shape, but for the
    predictor's two-component loss schema (concept + reasoning).
    """
    w_concept = eval_losses["concept"] * loss_weights["concept_loss_weight"]
    ew = {"concept": w_concept}
    reasoning_part = ""
    if "reasoning" in eval_losses:
        ew["reasoning"] = (
            eval_losses["reasoning"] * loss_weights["reasoning_loss_weight"]
        )
        reasoning_part = " reasoning=%.4f/%.4f" % (
            eval_losses["reasoning"],
            ew["reasoning"],
        )

    label = "eval(quick)" if eval_type == "quick" else "eval(full) "
    logger.info(
        "  %s | total=%.4f concept=%.4f/%.4f%s",
        label,
        eval_losses["total"],
        eval_losses["concept"],
        ew["concept"],
        reasoning_part,
    )

    # SwanLab metrics
    metrics = {
        f"{swanlab_prefix}/total_loss": eval_losses["total"],
        f"{swanlab_prefix}/concept_raw": eval_losses["concept"],
        f"{swanlab_prefix}/concept_weighted": ew["concept"],
    }
    if "reasoning" in eval_losses:
        metrics[f"{swanlab_prefix}/reasoning_raw"] = eval_losses["reasoning"]
        metrics[f"{swanlab_prefix}/reasoning_weighted"] = ew["reasoning"]
    if "concept_per_level" in eval_losses:
        for k, v in enumerate(eval_losses["concept_per_level"]):
            metrics[f"{swanlab_prefix}/concept_level{k}"] = v
    swanlab.log(metrics, step=global_step)

    # terminal_output.jsonl row
    term_entry = {
        "step": global_step,
        "eval_type": eval_type,
        **{
            f"eval_{k}": round(v, 6)
            for k, v in eval_losses.items()
            if k != "concept_per_level"
        },
        **{f"eval_{k}_w": round(v, 6) for k, v in ew.items()},
    }
    if "concept_per_level" in eval_losses:
        term_entry["eval_concept_per_level"] = [
            round(v, 6) for v in eval_losses["concept_per_level"]
        ]
    log_terminal_entry(terminal_log_path, term_entry)

    # eval_history.json (crash-safe rewrite per eval)
    eval_history.append(
        {
            "step": global_step,
            "eval_type": eval_type,
            **eval_losses,
            **{f"{k}_w": v for k, v in ew.items()},
        }
    )
    with open(log_dir / "eval_history.json", "w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2, default=str)

    if reasoning_texts:
        entry = {
            "step": global_step,
            "eval_type": eval_type,
            "texts": reasoning_texts,
        }
        with open(log_dir / "eval_reasoning_texts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    import datetime as _dt

    eval_sample_history.append(
        {
            "step": global_step,
            "eval_type": eval_type,
            "timestamp": _dt.datetime.now().isoformat(),
            "num_samples": len(eval_samples),
            "samples": eval_samples,
        }
    )
    with open(log_dir / "eval_sample_history.json", "w", encoding="utf-8") as f:
        json.dump(eval_sample_history, f, indent=2, default=str)


# =============================================================================
# Main training loop
# =============================================================================


def train_predictor(
    config: dict,
    builder_config: dict,
    config_path: Path,
    storage_root: str,
    cli_swanlab_id: str = "",
    resume: bool = False,
):
    """Stage-2 training loop.

    ``config`` is the predictor YAML (with ``model.pyramid`` already
    injected from the builder config).  ``builder_config`` is the
    separately-loaded builder YAML used to construct the frozen
    Builder — kept separate (rather than nested into ``config``) so
    the predictor's config.json dump does not bloat with a copy of
    the builder's entire tree.
    """
    # ── Extract sub-configs ───────────────────────────────────────────
    train_cfg = config["training"]
    data_cfg = config["data"]
    env_cfg = config["environment"]
    log_cfg = config["log"]
    loss_weights = train_cfg["loss_weights"]

    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_ratio = train_cfg["warmup_ratio"]
    gradient_clip = train_cfg["gradient_clip"]
    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]
    checkpoint_clean = log_cfg["checkpoint_clean"]

    checkpoint_dir = Path(log_cfg["checkpoint_path"])
    log_dir = Path(log_cfg["log_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Rotate non-appendable text logs on resume (before basicConfig) ─
    if resume:
        _resume_io.rotate_if_exists(log_dir / "training.log")
        _resume_io.rotate_if_exists(log_dir / "terminal_output.jsonl")

    logging.basicConfig(
        level=getattr(logging, log_cfg["log_level"].upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("train_predictor")
    terminal_log_path = log_dir / "terminal_output.jsonl"

    seed = env_cfg["seed"]
    device = str(get_device("auto"))
    _seed_single_device(seed, device)
    logger.info("Device: %s | seed: %d", device, seed)

    # Load .env so downstream libraries (SwanLab, HF) can pick up tokens.
    dotenv_path = env_cfg["dotenv_path"]
    load_dotenv(dotenv_path)

    # ── Experiment name derivation (identical to train_builder.py) ────
    configs_root = PROJECT_ROOT / "configs" / "nlcpV4"
    try:
        rel_parts = config_path.resolve().relative_to(configs_root).parent.parts
    except ValueError:
        rel_parts = (config_path.parent.name,)
    experiment_name = "-".join([*rel_parts, config_path.stem])

    # ── SwanLab init (resume-aware) ───────────────────────────────────
    swanlab_meta = _resume_io.load_swanlab_meta(log_dir)
    if resume:
        swanlab_id = _resume_io.resolve_swanlab_id(
            cli_swanlab_id, swanlab_meta, log_dir, resume=True
        )
        swanlab.init(
            project="ReasoningAR",
            experiment_name=experiment_name,
            config=config,
            id=swanlab_id,
            resume="must",
        )
        logger.info("SwanLab resumed (id=%s)", swanlab_id)
    else:
        if cli_swanlab_id:
            logger.info(
                "--swanlab-id=%s ignored: --resume is not set; SwanLab "
                "will allocate a new id.",
                cli_swanlab_id,
            )
        swanlab.init(
            project="ReasoningAR", experiment_name=experiment_name, config=config
        )
        run = swanlab.get_run()
        swanlab_id = getattr(run, "id", None) or ""
        if swanlab_id:
            _resume_io.init_swanlab_meta(
                log_dir, "ReasoningAR", experiment_name, swanlab_id
            )
            logger.info(
                "SwanLab initialized (id=%s, recorded to %s)",
                swanlab_id,
                log_dir / _resume_io.SWANLAB_META_FILENAME,
            )
        else:
            logger.warning(
                "SwanLab returned no run id (mode=%s); swanlab.json NOT written.",
                getattr(run, "mode", "<unknown>"),
            )

    # ── Build frozen Builder (Stage-1 dependency) ─────────────────────
    builder_ckpt_raw = config["model"]["builder"]["checkpoint_path"]
    builder_strict = config["model"]["builder"]["strict_load"]
    builder_ckpt_path = _resolve_builder_checkpoint_path(
        builder_ckpt_raw, storage_root
    ).resolve()
    logger.info("Builder checkpoint (resolved): %s", builder_ckpt_path)
    builder = _load_frozen_builder(
        builder_config, builder_ckpt_path, builder_strict, device, logger
    )

    # ── Build predictor ───────────────────────────────────────────────
    predictor = ConceptPredictor(config, builder=builder)
    predictor.to(device)
    # Re-assert eval mode on the shared reason_model after ``.to(device)``
    # (predictor.train() is the module's default state after
    # construction; in SHARED mode the same module is reachable via
    # both ``predictor.reason_model`` and ``builder.reason_model`` so
    # either path can flip its mode).
    if config["model"]["predictor"]["use_shared_model"]:
        predictor.reason_model.eval()
    _log_predictor_summary(predictor, builder, config, logger)

    # ── Optimizer + scheduler ─────────────────────────────────────────
    trainable_params = [p for p in predictor.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(
            "No trainable parameters in predictor — check "
            "use_shared_model / freeze / lora combination."
        )
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # ── Data loaders (train + optional eval) ──────────────────────────
    dataloader = NLCPV4DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=True,
        shuffle=data_cfg["shuffle"],
        drop_last=data_cfg["drop_last"],
        num_workers=env_cfg["dataloader_num_workers"],
    )
    logger.info(
        "Dataset: %s | Batches/epoch: %d | Batch size: %d",
        data_cfg["data_name"],
        len(dataloader),
        batch_size,
    )

    eval_cfg = config["evaluation"]
    eval_interval = eval_cfg["eval_step_interval"]
    eval_enabled = eval_interval > 0
    eval_dataloader = None
    eval_history: list[dict] = []
    eval_sample_history: list[dict] = []
    quick_eval_batches = 0
    full_eval_batches = 0

    if eval_enabled:
        eval_data_cfg = eval_cfg["data"]
        eval_dataloader = NLCPV4DataLoader(
            data_cfg=eval_data_cfg,
            batch_size=batch_size,
            include_solution=True,
            shuffle=True,
            drop_last=False,
            num_workers=env_cfg["dataloader_num_workers"],
        )
        eval_dataset_size = eval_dataloader.dataset_size

        raw_log_ns = eval_data_cfg["log_num_samples"]
        if 0 < raw_log_ns <= 1.0:
            log_eval_samples = int(eval_dataset_size * raw_log_ns)
        else:
            log_eval_samples = int(raw_log_ns)
        quick_eval_batches = max(1, (log_eval_samples + batch_size - 1) // batch_size)

        raw_eval_ns = eval_data_cfg["eval_num_samples"]
        if 0 < raw_eval_ns <= 1.0:
            full_eval_samples = int(eval_dataset_size * raw_eval_ns)
        else:
            full_eval_samples = int(raw_eval_ns)
        full_eval_batches = max(1, (full_eval_samples + batch_size - 1) // batch_size)

        logger.info(
            "Eval: %s (split=%s) | dataset_size=%d | full_eval=%d batches | quick_eval=%d batches",
            eval_data_cfg["data_name"],
            eval_data_cfg["split"],
            eval_dataset_size,
            full_eval_batches,
            quick_eval_batches,
        )

    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(
        "Total steps: %d | Warmup: %d | LR: %s",
        total_steps,
        warmup_steps,
        learning_rate,
    )

    # ── Resume (predictor-only; Builder is external) ──────────────────
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    best_eval_loss = float("inf")
    history: list = []

    if resume:
        resume_path = _resume_io.find_latest_checkpoint(checkpoint_dir)
        logger.info("Auto-discovered resume checkpoint: %s", resume_path)
        start_epoch, global_step, best_loss, best_eval_loss = load_checkpoint(
            resume_path, predictor, optimizer, scheduler
        )
        history = _resume_io.load_history(log_dir / "training_history.json")
        eval_history = _resume_io.load_history(log_dir / "eval_history.json")
        eval_sample_history = _resume_io.load_history(
            log_dir / "eval_sample_history.json"
        )
        if swanlab_meta is not None:
            _resume_io.record_resume_event(
                log_dir, resume_path, start_epoch, global_step
            )
        logger.info(
            "Resumed from epoch %d, step %d, loss=%.4f, best_eval=%.4f "
            "(history rows: train=%d eval=%d eval_samples=%d)",
            start_epoch,
            global_step,
            best_loss,
            best_eval_loss,
            len(history),
            len(eval_history),
            len(eval_sample_history),
        )

    # Persist the resolved predictor + builder configs for auditability.
    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    with open(log_dir / "builder_config.json", "w", encoding="utf-8") as f:
        json.dump(builder_config, f, indent=2, default=str)

    # ── Training loop ─────────────────────────────────────────────────
    predictor.train()
    if config["model"]["predictor"]["use_shared_model"]:
        predictor.reason_model.eval()
    max_length = config["model"]["pyramid"]["max_seq_len"]

    for epoch in range(start_epoch, num_epochs):
        epoch_losses: list[float] = []
        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", miniters=log_interval
        )
        for batch_idx, batch in enumerate(pbar):
            output = _run_predictor_step(predictor, builder, batch, max_length, device)
            total_loss, loss_dict = compute_predictor_loss(
                output, loss_weights, concept_loss_type="mse"
            )

            total_loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_losses.append(loss_dict["total"])
            global_step += 1

            w = {
                "concept": loss_dict["concept"] * loss_weights["concept_loss_weight"],
            }
            if "reasoning" in loss_dict:
                w["reasoning"] = (
                    loss_dict["reasoning"] * loss_weights["reasoning_loss_weight"]
                )

            # ── Logging at log_interval ────────────────────────────
            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total']:.4f}",
                        "concept": f"{loss_dict['concept']:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )
                reasoning_part = ""
                if "reasoning" in loss_dict:
                    reasoning_part = " reasoning=%.4f/%.4f" % (
                        loss_dict["reasoning"],
                        w["reasoning"],
                    )
                logger.info(
                    "Step %5d | total=%.4f concept=%.4f/%.4f%s lr=%.2e",
                    global_step,
                    loss_dict["total"],
                    loss_dict["concept"],
                    w["concept"],
                    reasoning_part,
                    lr,
                )
                term_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "total": round(loss_dict["total"], 6),
                    "concept": round(loss_dict["concept"], 6),
                    "concept_w": round(w["concept"], 6),
                    "lr": lr,
                }
                if "reasoning" in loss_dict:
                    term_entry["reasoning"] = round(loss_dict["reasoning"], 6)
                    term_entry["reasoning_w"] = round(w["reasoning"], 6)
                if "concept_per_level" in loss_dict:
                    term_entry["concept_per_level"] = [
                        round(v, 6) for v in loss_dict["concept_per_level"]
                    ]
                log_terminal_entry(terminal_log_path, term_entry)

                swanlab_metrics = {
                    "train/total_loss": loss_dict["total"],
                    "train/concept_raw": loss_dict["concept"],
                    "train/concept_weighted": w["concept"],
                    "train/lr": lr,
                }
                if "reasoning" in loss_dict:
                    swanlab_metrics["train/reasoning_raw"] = loss_dict["reasoning"]
                    swanlab_metrics["train/reasoning_weighted"] = w["reasoning"]
                if "concept_per_level" in loss_dict:
                    for k, v in enumerate(loss_dict["concept_per_level"]):
                        swanlab_metrics[f"train/concept_level{k}"] = v
                swanlab.log(swanlab_metrics, step=global_step)

                # Quick eval (skip when full eval fires at same step).
                if eval_enabled and not (global_step % eval_interval == 0):
                    eval_losses, reasoning_texts, samples = evaluate_predictor(
                        predictor,
                        builder,
                        eval_dataloader,
                        loss_weights,
                        max_length,
                        device,
                        max_batches=quick_eval_batches,
                    )
                    log_eval_results_predictor(
                        eval_losses,
                        loss_weights,
                        "quick",
                        global_step,
                        terminal_log_path,
                        eval_history,
                        log_dir,
                        "eval_quick",
                        reasoning_texts,
                        samples,
                        eval_sample_history,
                        logger,
                    )

            # ── Checkpoint scheduling (identical policy to builder) ──
            save_regular = False
            save_tag = ""
            if checkpoint_clean:
                if batch_idx == 0:
                    save_regular = True
                    save_tag = "epoch-start"
            else:
                if global_step % checkpoint_interval == 0:
                    save_regular = True

            if save_regular:
                window = epoch_losses[-100:]
                avg_loss = sum(window) / len(window)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    purge_best_checkpoints(checkpoint_dir, "checkpoint_best")
                    best_path = save_checkpoint(
                        predictor,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        avg_loss,
                        checkpoint_dir,
                        filename=(f"checkpoint_best-epoch{epoch}-step{global_step}.pt"),
                        best_eval_loss=best_eval_loss,
                    )
                    logger.info("Best checkpoint: %s", best_path.name)
                tag_part = f"-{save_tag}" if save_tag else ""
                path = save_checkpoint(
                    predictor,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    avg_loss,
                    checkpoint_dir,
                    filename=(
                        f"checkpoint{tag_part}-epoch{epoch}-step{global_step}.pt"
                    ),
                    best_eval_loss=best_eval_loss,
                )
                logger.info("Checkpoint: %s", path.name)

            # ── Full eval at eval_interval ──────────────────────────
            if eval_enabled and global_step % eval_interval == 0:
                eval_losses, reasoning_texts, samples = evaluate_predictor(
                    predictor,
                    builder,
                    eval_dataloader,
                    loss_weights,
                    max_length,
                    device,
                    max_batches=full_eval_batches,
                )
                log_eval_results_predictor(
                    eval_losses,
                    loss_weights,
                    "full",
                    global_step,
                    terminal_log_path,
                    eval_history,
                    log_dir,
                    "eval",
                    reasoning_texts,
                    samples,
                    eval_sample_history,
                    logger,
                )
                if eval_losses["total"] < best_eval_loss:
                    best_eval_loss = eval_losses["total"]
                    purge_best_checkpoints(checkpoint_dir, "checkpoint_best_eval")
                    best_eval_path = save_checkpoint(
                        predictor,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        eval_losses["total"],
                        checkpoint_dir,
                        filename=(
                            f"checkpoint_best_eval-epoch{epoch}-step{global_step}.pt"
                        ),
                        best_eval_loss=best_eval_loss,
                    )
                    logger.info(
                        "Best eval checkpoint: %s (eval_loss=%.4f)",
                        best_eval_path.name,
                        eval_losses["total"],
                    )

            # History row (raw + weighted scalars; skip per-level list
            # to keep the JSON compact — SwanLab already carries it).
            step_record = {
                "step": global_step,
                "epoch": epoch,
                **{k: v for k, v in loss_dict.items() if k != "concept_per_level"},
                **{f"{k}_w": v for k, v in w.items()},
            }
            history.append(step_record)

        # ── Epoch-end reporting + checkpoint ─────────────────────────
        avg_epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        )
        logger.info("Epoch %d avg loss: %.4f", epoch + 1, avg_epoch_loss)
        log_terminal_entry(
            terminal_log_path,
            {"epoch": epoch, "avg_epoch_loss": round(avg_epoch_loss, 6)},
        )
        swanlab.log(
            {"epoch/avg_loss": avg_epoch_loss, "epoch/epoch": epoch + 1},
            step=global_step,
        )

        is_last_epoch = epoch + 1 == num_epochs
        if (not checkpoint_clean) or is_last_epoch:
            tag_part = "-epoch-end" if checkpoint_clean else ""
            path = save_checkpoint(
                predictor,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                avg_epoch_loss,
                checkpoint_dir,
                filename=f"checkpoint{tag_part}-epoch{epoch+1}-step{global_step}.pt",
                best_eval_loss=best_eval_loss,
            )
            logger.info("Epoch checkpoint: %s", path.name)

        with open(log_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        if eval_history:
            with open(log_dir / "eval_history.json", "w", encoding="utf-8") as f:
                json.dump(eval_history, f, indent=2, default=str)
        if eval_sample_history:
            with open(log_dir / "eval_sample_history.json", "w", encoding="utf-8") as f:
                json.dump(eval_sample_history, f, indent=2, default=str)

    logger.info("Training complete!")
    best_file = next(checkpoint_dir.glob("checkpoint_best-*.pt"), None)
    best_eval_file = next(checkpoint_dir.glob("checkpoint_best_eval-*.pt"), None)
    if best_file is not None:
        logger.info("Best train checkpoint: %s", best_file)
    if best_eval_file is not None:
        logger.info("Best eval checkpoint:  %s", best_eval_file)

    swanlab.finish()
    logger.info("SwanLab run finished")


# =============================================================================
# Entry point
# =============================================================================


def main():
    args = parse_args()

    # 1. Resolve predictor config path (relative → PROJECT_ROOT).
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    yaml_config = load_config(str(config_path))
    apply_storage_root(yaml_config, args.storage_root)
    print_storage_paths(yaml_config, args.storage_root)

    # 2. Load the paired builder config and inherit its pyramid block.
    builder_cfg_path_raw = yaml_config["model"]["builder"]["config_path"]
    builder_cfg_path = _resolve_config_path(builder_cfg_path_raw, PROJECT_ROOT)
    if not builder_cfg_path.exists():
        raise FileNotFoundError(
            f"Builder config not found: {builder_cfg_path}.  "
            f"Check model.builder.config_path in the predictor YAML."
        )
    builder_config = load_config(str(builder_cfg_path))
    _inherit_pyramid_from_builder(yaml_config, builder_config)

    # 3. Fail-fast sanity check on shared+LoRA / independent+freeze combos.
    _fail_fast_shared_sanity(yaml_config)

    # 4. Show resolved builder checkpoint alongside the [STORAGE] block.
    resolved_ckpt = _resolve_builder_checkpoint_path(
        yaml_config["model"]["builder"]["checkpoint_path"], args.storage_root
    )
    print(
        f"[STORAGE]   builder_checkpoint = {yaml_config['model']['builder']['checkpoint_path']}"
    )
    print(f"[STORAGE]                        (absolute: {resolved_ckpt.resolve()})")

    # 5. Launch.
    train_predictor(
        yaml_config,
        builder_config=builder_config,
        config_path=config_path,
        storage_root=args.storage_root,
        cli_swanlab_id=args.swanlab_id,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
