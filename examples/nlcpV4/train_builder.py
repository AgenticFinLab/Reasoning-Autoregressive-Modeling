"""Train NLCP V4 ConceptPyramidBuilder.

Usage:
    # Basic: train with the config's own log paths (relative to project root).
    python3 examples/nlcpV4/train_builder.py -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml

    # Nested variant (e.g. AutoWeighted/ subtree) — same CLI, path just changes.
    python3 examples/nlcpV4/train_builder.py -c configs/nlcpV4/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_6level.yml

    # Redirect ALL relative outputs (save_folder/checkpoint_path/log_path)
    # under a storage root — typical on a shared server where the
    # project-local EXPERIMENT/ tree is not writable.
    python3 examples/nlcpV4/train_builder.py -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml -s /Data/<proj>

    # Resume: boolean flag. The checkpoint to load is auto-discovered
    # under ``log.checkpoint_path`` (the latest epoch/step file).
    python3 examples/nlcpV4/train_builder.py -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml --resume

    # Resume AND pin the SwanLab run explicitly (rare — normally the
    # swanlab_id is auto-recovered from logs/<exp>/swanlab.json).
    python3 examples/nlcpV4/train_builder.py -c configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml --resume --swanlab-id 5hjp09vuqh402irzz9j9h

Arguments:
    -s / --storage-root   Prefix prepended to RELATIVE log paths in the
                          YAML (save_folder/checkpoint_path/log_path).
                          Absolute YAML paths are preserved. Default is
                          ``./`` (current working directory) — NEVER an
                          implicit project root. The resolved absolute
                          paths are printed as a ``[STORAGE]`` block at
                          startup so there is zero ambiguity about
                          where checkpoints / logs will be written.
                          Listed FIRST because it controls every output
                          path this script writes.
    -c / --config         Path to a YAML training config.
    --resume              Boolean flag (store_true). When set, the
                          trainer auto-discovers the latest checkpoint
                          under ``log.checkpoint_path`` (pattern
                          ``checkpoint*-epoch<N>-step<M>.pt``, picked
                          by max ``(step, epoch)``) and restores:
                            - model / optimizer / scheduler state,
                            - ``start_epoch`` / ``global_step`` /
                              ``best_loss`` / ``best_eval_loss``,
                          then rotates ``training.log`` and
                          ``terminal_output.jsonl`` to
                          ``<stem>.partN<ext>`` (N = next free) so the
                          new run's logs never clobber the old chunks,
                          and loads ``training_history.json`` /
                          ``eval_history.json`` /
                          ``eval_sample_history.json`` so epoch-end
                          rewrites append instead of wiping history.
                          If ``log.checkpoint_path`` is missing or
                          contains no matching file, the trainer
                          hard-errors (``CheckpointNotFoundError``)
                          rather than silently fresh-starting.
                          Resume is a pure CLI concern — the YAML
                          config has NO ``training.resume`` field.
                          Resume is always explicit at invocation
                          time and never ambient in a checked-in
                          config file.
    --swanlab-id          Optional SwanLab run id to resume. Only
                          consulted when ``--resume`` is set. Precedence:
                            1. ``--swanlab-id`` (this CLI flag), then
                            2. ``logs/<exp>/swanlab.json`` on disk,
                               otherwise
                            3. HARD ERROR — we refuse to start a
                               disconnected SwanLab run.
                          ``logs/<exp>/swanlab.json`` is written
                          automatically at the start of every fresh
                          run and updated on every resume.

V4 contract (vs V3):
  - ``ConceptPyramidBuilder.forward(batch: BuilderInput) -> PyramidOutput``
    handles encoding, pyramid construction, and (when the batch has
    solutions) reasoning preparation in a single call. No separate
    ``encode_cot`` / ``compute_reasoning_loss`` plumbing in the trainer.
  - ``compute_builder_loss`` in ``nlcpV4.losses`` owns ALL loss math
    (recon + ordering + residual + reasoning) — it reads the reasoning
    logits/targets directly from ``PyramidOutput``.
  - ``evaluate_builder`` / ``log_eval_results`` / ``log_terminal_entry``
    live in ``nlcpV4.eval_builder`` and are reused here, so the trainer
    contains training-loop logic only.
"""

import argparse
import json
import logging
import math
import random
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
# is executed directly (``python3 examples/nlcpV4/train_builder.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from nlcpV4 import _resume_io
from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.data_loader import NLCPV4DataLoader
from nlcpV4.eval_builder import (
    evaluate_builder,
    log_eval_results,
    log_terminal_entry,
)
from nlcpV4.losses import compute_builder_loss
from ram.utils import apply_storage_root, load_config, print_storage_paths


def _seed_single_device(seed: int, device: str) -> None:
    """Seed RNGs for CPU + the chosen CUDA device only.

    Why not ``ram.utils.setup_environment`` (which calls
    ``torch.cuda.manual_seed_all``)?

      ``manual_seed_all`` seeds RNG state on EVERY visible CUDA device.
      To do that, PyTorch must create a full CUDA context on each GPU
      (~300-500 MB each). On a shared cluster, any one of those GPUs
      might be too tight for a new context — the failure is queued as
      an ASYNC error and surfaces later as a misleading "OOM on your
      chosen GPU" when ``builder.to(device)`` runs. Since we only ever
      allocate tensors on ONE device in this script, seeding any other
      device is both wasteful and risky.

    This helper seeds only the chosen device and leaves other GPUs
    untouched, so no spurious context-init failures can be parked.
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


def _log_model_summary(builder: ConceptPyramidBuilder, config: dict, logger):
    """Log a detailed model architecture summary table."""
    reason_cfg = config["model"]["reason_model"]
    pyramid_cfg = config["model"]["pyramid"]
    train_rm_cfg = config["training"]["reason_model"]
    loss_weights = config["training"]["loss_weights"]

    total_params = sum(p.numel() for p in builder.parameters())
    trainable_params = sum(p.numel() for p in builder.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Per-module param counts
    def _count(module):
        return sum(p.numel() for p in module.parameters())

    def _count_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    reason_total = _count(builder.reason_model)
    reason_train = _count_trainable(builder.reason_model)
    proj_params = _count(builder.input_proj) + _count(builder.input_proj_norm)
    query_params = sum(q.numel() for q in builder.concept_queries)
    level_proj_params = _count(builder.level_projs)
    back_proj_params = _count(builder.back_proj)
    temp_params = builder.temperature.numel()

    level_lengths = pyramid_cfg["level_lengths"]
    D = pyramid_cfg["hidden_dim"]
    D_enc = builder.reason_model_hidden_dim

    lines = [
        "",
        "=" * 72,
        "  MODEL ARCHITECTURE SUMMARY",
        "=" * 72,
        "",
        "  Reason Model",
        "  ├─ name              : %s" % reason_cfg["reason_model_name"],
        "  ├─ encoder_dim       : %d" % D_enc,
        "  ├─ vocab_size        : %d" % builder.reason_model.config.vocab_size,
        "  ├─ num_layers        : %d" % builder.reason_model.config.num_hidden_layers,
        "  ├─ freeze            : %s" % train_rm_cfg["freeze"],
        "  ├─ lora              : %s" % (train_rm_cfg["lora"] or "None"),
        "  ├─ params            : %s (trainable: %s)"
        % (f"{reason_total:,}", f"{reason_train:,}"),
        "  ",
        "  Pyramid",
        "  ├─ hidden_dim (D)    : %d" % D,
        "  ├─ num_levels (K)    : %d" % pyramid_cfg["num_levels"],
        "  ├─ level_lengths     : %s  (total: %d)"
        % (level_lengths, sum(level_lengths)),
        "  ├─ max_seq_len       : %d" % pyramid_cfg["max_seq_len"],
        "  ",
        "  Modules                     Shape                 Params",
        "  " + "-" * 68,
        "  input_proj               : [%d, %d] + LN        %s"
        % (D_enc, D, f"{proj_params:,}"),
        "  concept_queries          : %d levels             %s"
        % (len(level_lengths), f"{query_params:,}"),
    ]
    for k, L_k in enumerate(level_lengths):
        lines.append("    level %d               : [%d, %d]" % (k, L_k, D))
    lines += [
        "  temperature              : [1]                   %d" % temp_params,
        "  level_projs              : %d × [%d, %d]        %s"
        % (len(level_lengths), D, D, f"{level_proj_params:,}"),
        "  back_proj                : [%d, %d]              %s"
        % (D, D_enc, f"{back_proj_params:,}"),
        "  ",
        "  Loss Weights",
        "  ├─ recon               : %s" % loss_weights["recon_loss_weight"],
        "  ├─ ordering            : %s" % loss_weights["ordering_loss_weight"],
        "  ├─ residual            : %s" % loss_weights["residual_loss_weight"],
        "  ├─ reasoning           : %s" % loss_weights["reasoning_loss_weight"],
        "  ├─ ordering_margin     : %s" % loss_weights["ordering_margin"],
        "  ",
        "  Parameter Summary",
        "  ├─ total               : %s" % f"{total_params:,}",
        "  ├─ trainable           : %s  (%.2f%%)"
        % (f"{trainable_params:,}", 100.0 * trainable_params / total_params),
        "  └─ frozen              : %s" % f"{frozen_params:,}",
        "=" * 72,
        "",
    ]
    for line in lines:
        logger.info(line)


def parse_args():
    """Parse command-line arguments for the trainer.

    Returns:
        argparse.Namespace with ``config`` (YAML path) and ``resume``
        (optional checkpoint path) fields.
    """
    parser = argparse.ArgumentParser(description="Train ConceptPyramidBuilder")
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix to prepend to every relative output path in "
            "config.log (save_folder / checkpoint_path / log_path). "
            "Absolute paths in YAML are preserved. Default is './' so "
            "paths resolve relative to the CURRENT WORKING DIRECTORY "
            "you launched the command from (no silent project-root "
            "fallback). Pass -s /Data/<proj> on servers where outputs "
            "should land under a dedicated storage root. The resolved "
            "paths are always printed at startup so you can verify "
            "where data is written."
        ),
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume training from the latest checkpoint auto-discovered "
            "under log.checkpoint_path (pattern "
            "'checkpoint*-epoch<N>-step<M>.pt', picked by max step). "
            "Boolean flag — no path argument. Resume is a pure CLI "
            "concern; YAML configs no longer carry a 'training.resume' "
            "field so running with or without --resume is the ONLY "
            "way to select between resume and fresh-start."
        ),
    )
    parser.add_argument(
        "--swanlab-id",
        type=str,
        default="",
        help=(
            "SwanLab run id to resume. Only consulted when --resume is "
            "set. Precedence: CLI --swanlab-id  >  logs/<exp>/swanlab.json "
            " >  hard error. A fresh (non-resume) run ignores this flag "
            "and lets SwanLab allocate a new id, then records it in "
            "logs/<exp>/swanlab.json for later resumes."
        ),
    )
    return parser.parse_args()


def save_checkpoint(
    builder: ConceptPyramidBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: Path,
    filename: str,
    best_eval_loss: float = float("inf"),
) -> Path:
    """Save model/optimizer/scheduler state to ``checkpoint_dir / filename``.

    ``best_eval_loss`` is the running minimum eval loss at the moment
    of save; persisting it is what allows resume to preserve the
    "best eval" bar across process restarts. Old checkpoints written
    before this field existed load fine — ``load_checkpoint`` falls
    back to ``float('inf')`` via ``.get``.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "best_eval_loss": best_eval_loss,
        "model_state_dict": builder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    path = checkpoint_dir / filename
    torch.save(checkpoint, path)
    return path


def purge_best_checkpoints(checkpoint_dir: Path, prefix: str) -> None:
    """Remove previous best checkpoints matching ``{prefix}-*.pt`` or legacy ``{prefix}.pt``.

    Used to preserve the "exactly one best file" invariant when the filename carries
    epoch/step tags, so each new best replaces the old one on disk.
    """
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
    builder: ConceptPyramidBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, int, float, float]:
    """Load a checkpoint and return ``(epoch, step, loss, best_eval_loss)``.

    Schema tolerance:
      * ``best_eval_loss`` is a post-hoc addition; checkpoints produced
        before the field existed load without error and surface
        ``float('inf')`` for it.
      * ``loss`` is the running train-loss anchor used solely to seed
        ``best_loss`` after resume. Legacy best-eval checkpoints
        (commit ``1a510ef``) wrote only ``eval_loss`` and no ``loss``;
        we accept either key, and ultimately fall back to ``inf`` so
        the next batch trivially becomes the new best-train baseline.
        This keeps resume robust to historical schema drift without
        masking real corruption (state-dict keys are still strict).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    builder.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # Defensive .get for ``loss``: legacy ``checkpoint_best_eval-*.pt``
    # files wrote ``eval_loss`` instead. Either is acceptable as a
    # best_loss seed; absent both, ``inf`` is the correct neutral.
    loss_value = checkpoint.get("loss", checkpoint.get("eval_loss", float("inf")))
    return (
        int(checkpoint["epoch"]),
        int(checkpoint["step"]),
        float(loss_value),
        float(checkpoint.get("best_eval_loss", float("inf"))),
    )


def train_builder(
    config: dict,
    config_path: Path,
    cli_swanlab_id: str = "",
    resume: bool = False,
):
    """Main training loop.

    ``resume`` is forwarded verbatim from ``--resume`` at the CLI —
    it is the SINGLE source of truth. The YAML config intentionally
    has no ``training.resume`` field; resume is always an explicit
    command-line choice made at invocation time.

    ``cli_swanlab_id`` is the value forwarded from ``--swanlab-id``.
    Only consulted when ``resume`` is True; otherwise the id is
    allocated by SwanLab and persisted to
    ``logs/<exp>/swanlab.json`` for subsequent resumes.
    """
    # Extract sub-configs
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
    # ``resume`` is supplied by the caller (driven by the --resume CLI
    # flag in main()); intentionally NOT read from config. The YAML
    # has no ``training.resume`` field — resume is a pure CLI concern.
    ordering_loss_type = train_cfg["ordering_loss_type"]

    checkpoint_dir = Path(log_cfg["checkpoint_path"])
    log_dir = Path(log_cfg["log_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume pre-flight: rotate un-appendable text logs ─────────
    # Must happen BEFORE ``logging.basicConfig`` below because
    # FileHandler opens training.log for write on construction; if
    # we rotate afterwards the new run would have already clobbered
    # the file. terminal_output.jsonl is opened append-mode lazily
    # in eval_builder.log_terminal_entry, so rotation timing is less
    # critical there, but we keep the two files in lockstep for
    # operator sanity.
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
    logger = logging.getLogger("train_builder")
    terminal_log_path = log_dir / "terminal_output.jsonl"

    seed = env_cfg["seed"]
    # Pick the device FIRST (NVML-only probe, no CUDA context init on
    # other GPUs), then seed ONLY that device. Avoids
    # manual_seed_all's multi-GPU context init — see
    # _seed_single_device() above for why that matters.
    device = str(get_device("auto"))
    _seed_single_device(seed, device)
    logger.info("Device: %s | seed: %d", device, seed)

    # Load .env so downstream libraries (e.g. SwanLab, HuggingFace)
    # can read their auth tokens from environment variables.
    dotenv_path = env_cfg["dotenv_path"]
    load_dotenv(dotenv_path)

    # Derive experiment name from the config file's location under
    # ``configs/nlcpV4/``. All path segments between that root and the
    # file (dataset, and any nested variant such as ``AutoWeighted/``)
    # are joined by ``-`` with the filename stem so the name is unique
    # and self-describing regardless of directory depth:
    #   configs/nlcpV4/GSM8K/train_builder_Qwen2.5-0.5B_6level.yml
    #     -> "GSM8K-train_builder_Qwen2.5-0.5B_6level"
    #   configs/nlcpV4/GSM8K/AutoWeighted/train_builder_Qwen2.5-0.5B_6level.yml
    #     -> "GSM8K-AutoWeighted-train_builder_Qwen2.5-0.5B_6level"
    # Fail-fast: if the config lives outside configs/nlcpV4/, fall back
    # to the legacy single-parent form so out-of-tree configs still run.
    configs_root = PROJECT_ROOT / "configs" / "nlcpV4"
    try:
        rel_parts = config_path.resolve().relative_to(configs_root).parent.parts
    except ValueError:
        rel_parts = (config_path.parent.name,)
    experiment_name = "-".join([*rel_parts, config_path.stem])

    # ── SwanLab init (resume-aware) ───────────────────────────────
    # Resolve the run id BEFORE calling swanlab.init because the
    # resume and fresh-start code paths take different arguments.
    # Precedence: --swanlab-id  >  on-disk swanlab.json  >  hard
    # error (on resume) or fresh allocation (on non-resume).
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
        # Capture the allocated id and persist it so future resumes
        # can find it without the operator having to remember.
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
            # Disabled/offline mode returns id=None. Log a warning
            # but do not fail — the operator may intentionally be
            # running offline, and resume from such a run is
            # already impossible regardless of this file.
            logger.warning(
                "SwanLab returned no run id (mode=%s); swanlab.json "
                "NOT written. Future --resume from this run will "
                "require an explicit --swanlab-id.",
                getattr(run, "mode", "<unknown>"),
            )

    builder = ConceptPyramidBuilder(config)
    builder.to(device)
    _log_model_summary(builder, config, logger)

    trainable_params = [p for p in builder.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

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

    # ── Evaluation setup ─────────────────────────────────────────
    eval_cfg = config["evaluation"]
    eval_interval = eval_cfg["eval_step_interval"]
    eval_enabled = eval_interval > 0
    eval_dataloader = None
    # Reasoning text recording flags (default: teacher-forced ON, generation OFF)
    tf_reasoning = eval_cfg.get("teacher_force_reasoning", True)
    gen_reasoning = eval_cfg.get("generation_reasoning", False)
    gen_max_tokens = eval_cfg.get("generation_max_tokens", 256)
    # eval_history holds per-invocation loss rows (eval_history.json);
    # eval_sample_history holds the matching sample lists
    # (eval_sample_history.json). Both are written crash-safely after
    # every eval call so either file can be cross-referenced offline.
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

        # Resolve log_num_samples: (0,1] = proportion, >1 = exact count
        raw_log_ns = eval_data_cfg["log_num_samples"]
        if 0 < raw_log_ns <= 1.0:
            log_eval_samples = int(eval_dataset_size * raw_log_ns)
        else:
            log_eval_samples = int(raw_log_ns)
        quick_eval_batches = max(1, (log_eval_samples + batch_size - 1) // batch_size)

        # Resolve eval_num_samples: (0,1] = proportion, >1 = exact count
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

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    best_eval_loss = float("inf")
    history: list = []

    if resume:
        # No CLI path: auto-discover the latest checkpoint under the
        # configured checkpoint_dir. Fail loudly if nothing is found —
        # silent fresh-starts on --resume would lose all continuity.
        resume_path = _resume_io.find_latest_checkpoint(checkpoint_dir)
        logger.info("Auto-discovered resume checkpoint: %s", resume_path)
        start_epoch, global_step, best_loss, best_eval_loss = load_checkpoint(
            resume_path, builder, optimizer, scheduler
        )
        # Load-and-extend the history lists so epoch-end rewrites
        # below append to, rather than clobber, the previous run.
        history = _resume_io.load_history(log_dir / "training_history.json")
        eval_history = _resume_io.load_history(log_dir / "eval_history.json")
        eval_sample_history = _resume_io.load_history(
            log_dir / "eval_sample_history.json"
        )
        # Record the resume event in swanlab.json (counts + last
        # checkpoint + epoch/step). Only safe to call when the
        # file actually exists — otherwise the operator passed
        # --swanlab-id manually and we have nothing to update.
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

    config_save_path = log_dir / "config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    builder.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        # tqdm refresh is throttled to once per ``log_step_interval``
        # iterations (and at most once per second). This matters when
        # stdout is tee'd to a log file — tqdm's default per-iteration
        # ``\r`` refresh becomes a new LINE in the log file rather than
        # an in-place update, flooding it with "per-step" progress
        # rows. With ``miniters=log_interval`` the log file gets at
        # most one progress row per logging interval, matching
        # ``logger.info``'s own cadence.
        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", miniters=log_interval
        )
        for batch_idx, batch in enumerate(pbar):
            # V4 API: single forward pass handles encode + pyramid +
            # reasoning preparation. ``compute_builder_loss`` then
            # reads ``pyramid.reasoning_logits/reasoning_target_ids``
            # to add the reasoning term when the batch has solutions.
            pyramid = builder(batch)
            total_loss, loss_dict = compute_builder_loss(
                pyramid,
                loss_weights,
                ordering_loss_type=ordering_loss_type,
            )

            total_loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_losses.append(loss_dict["total"])
            global_step += 1

            # Compute weighted individual losses
            w = {
                "recon": loss_dict["recon"] * loss_weights["recon_loss_weight"],
                "ordering": loss_dict["ordering"]
                * loss_weights["ordering_loss_weight"],
                "residual": loss_dict["residual"]
                * loss_weights["residual_loss_weight"],
            }
            if "reasoning" in loss_dict:
                w["reasoning"] = (
                    loss_dict["reasoning"] * loss_weights["reasoning_loss_weight"]
                )

            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total']:.4f}",
                        "recon": f"{loss_dict['recon']:.4f}",
                        "order": f"{loss_dict['ordering']:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )
                # Console: raw/weighted for each component
                reasoning_part = ""
                if "reasoning" in loss_dict:
                    reasoning_part = " reasoning=%.4f/%.4f" % (
                        loss_dict["reasoning"],
                        w["reasoning"],
                    )
                logger.info(
                    "Step %5d | total=%.4f recon=%.4f/%.4f ordering=%.4f/%.4f"
                    " residual=%.4f/%.4f%s lr=%.2e",
                    global_step,
                    loss_dict["total"],
                    loss_dict["recon"],
                    w["recon"],
                    loss_dict["ordering"],
                    w["ordering"],
                    loss_dict["residual"],
                    w["residual"],
                    reasoning_part,
                    lr,
                )
                # terminal_output.jsonl: raw + weighted
                terminal_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "total": round(loss_dict["total"], 6),
                    "recon": round(loss_dict["recon"], 6),
                    "recon_w": round(w["recon"], 6),
                    "ordering": round(loss_dict["ordering"], 6),
                    "ordering_w": round(w["ordering"], 6),
                    "residual": round(loss_dict["residual"], 6),
                    "residual_w": round(w["residual"], 6),
                    "lr": lr,
                }
                if "reasoning" in loss_dict:
                    terminal_entry["reasoning"] = round(loss_dict["reasoning"], 6)
                    terminal_entry["reasoning_w"] = round(w["reasoning"], 6)
                log_terminal_entry(terminal_log_path, terminal_entry)

                # SwanLab: raw + weighted as separate metrics
                swanlab_metrics = {
                    "train/total_loss": loss_dict["total"],
                    "train/recon_raw": loss_dict["recon"],
                    "train/recon_weighted": w["recon"],
                    "train/ordering_raw": loss_dict["ordering"],
                    "train/ordering_weighted": w["ordering"],
                    "train/residual_raw": loss_dict["residual"],
                    "train/residual_weighted": w["residual"],
                    "train/lr": lr,
                }
                if "reasoning" in loss_dict:
                    swanlab_metrics["train/reasoning_raw"] = loss_dict["reasoning"]
                    swanlab_metrics["train/reasoning_weighted"] = w["reasoning"]
                swanlab.log(swanlab_metrics, step=global_step)

                # ── Quick eval (skip when full eval fires at same step) ──
                if eval_enabled and not (global_step % eval_interval == 0):
                    eval_losses, reasoning_texts_dict, samples = evaluate_builder(
                        builder,
                        eval_dataloader,
                        loss_weights,
                        ordering_loss_type,
                        max_batches=quick_eval_batches,
                        teacher_force_reasoning=tf_reasoning,
                        generation_reasoning=gen_reasoning,
                        generation_max_tokens=gen_max_tokens,
                    )
                    log_eval_results(
                        eval_losses,
                        loss_weights,
                        "quick",
                        global_step,
                        terminal_log_path,
                        eval_history,
                        log_dir,
                        "eval_quick",
                        reasoning_texts_dict,
                        samples,
                        eval_sample_history,
                    )

            # ── Checkpoint scheduling ──────────────────────────
            #   checkpoint_clean=True  : save ONLY at epoch-start
            #                            (batch_idx==0). The final
            #                            epoch-end checkpoint is saved
            #                            once, after the last epoch
            #                            finishes (see the epoch-end
            #                            block below).
            #                            checkpoint_step_interval is
            #                            ignored.
            #   checkpoint_clean=False : save per checkpoint_step_interval (legacy).
            # Best checkpoints are always tracked (overwrite-by-purge).
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
                # ``epoch_losses`` is always non-empty here because
                # ``epoch_losses.append(...)`` runs above on every
                # iteration before this branch. Average over the last
                # 100 steps to smooth noisy per-step loss for best-
                # checkpoint tracking.
                window = epoch_losses[-100:]
                avg_loss = sum(window) / len(window)
                # Track best: overwrite any previous best file.
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    purge_best_checkpoints(checkpoint_dir, "checkpoint_best")
                    best_path = save_checkpoint(
                        builder,
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
                    builder,
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

            # ── Full eval at eval_interval ──────────────────────
            if eval_enabled and global_step % eval_interval == 0:
                eval_losses, reasoning_texts_dict, samples = evaluate_builder(
                    builder,
                    eval_dataloader,
                    loss_weights,
                    ordering_loss_type,
                    max_batches=full_eval_batches,
                    teacher_force_reasoning=tf_reasoning,
                    generation_reasoning=gen_reasoning,
                    generation_max_tokens=gen_max_tokens,
                )
                log_eval_results(
                    eval_losses,
                    loss_weights,
                    "full",
                    global_step,
                    terminal_log_path,
                    eval_history,
                    log_dir,
                    "eval",
                    reasoning_texts_dict,
                    samples,
                    eval_sample_history,
                )
                # Best eval checkpoint (always tracked; overwrite-by-purge).
                if eval_losses["total"] < best_eval_loss:
                    best_eval_loss = eval_losses["total"]
                    purge_best_checkpoints(checkpoint_dir, "checkpoint_best_eval")
                    best_eval_path = save_checkpoint(
                        builder,
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

            # Record step history with raw + weighted losses
            step_record = {"step": global_step, "epoch": epoch, **loss_dict}
            step_record.update({f"{k}_w": v for k, v in w.items()})
            history.append(step_record)

        avg_epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        )
        num_steps_epoch = len(epoch_losses)
        logger.info(
            "Epoch %d avg loss: %.4f",
            epoch + 1,
            avg_epoch_loss,
        )
        log_terminal_entry(
            terminal_log_path,
            {
                "epoch": epoch,
                "avg_epoch_loss": round(avg_epoch_loss, 6),
            },
        )

        # ── SwanLab epoch-level logging ──────────────────────────
        swanlab.log(
            {
                "epoch/avg_loss": avg_epoch_loss,
                "epoch/epoch": epoch + 1,
            },
            step=global_step,
        )

        # Epoch-end checkpoint policy:
        #   legacy mode (checkpoint_clean=False): save at every epoch boundary.
        #   clean mode  (checkpoint_clean=True):  save ONLY after the last
        #       epoch. Earlier epoch boundaries are covered by the next
        #       epoch's ``epoch-start`` save, so saving them here would
        #       just duplicate files.
        is_last_epoch = epoch + 1 == num_epochs
        if (not checkpoint_clean) or is_last_epoch:
            tag_part = "-epoch-end" if checkpoint_clean else ""
            path = save_checkpoint(
                builder,
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
    # By construction there is at most one best-train and one best-eval file
    # (purge_best_checkpoints removes the old one before every new best save).
    best_file = next(checkpoint_dir.glob("checkpoint_best-*.pt"), None)
    best_eval_file = next(checkpoint_dir.glob("checkpoint_best_eval-*.pt"), None)
    if best_file is not None:
        logger.info("Best train checkpoint: %s", best_file)
    if best_eval_file is not None:
        logger.info("Best eval checkpoint:  %s", best_eval_file)

    # ── Finish SwanLab run ────────────────────────────────────────
    swanlab.finish()
    logger.info("SwanLab run finished")


def main():
    """Entry point: parse CLI args, load YAML config, launch training."""
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    yaml_config = load_config(str(config_path))
    apply_storage_root(yaml_config, args.storage_root)
    # Make the resolved output locations impossible to miss. This is
    # intentionally printed BEFORE any logger setup so it shows up even
    # when a later step crashes (e.g. OOM during model init).
    print_storage_paths(yaml_config, args.storage_root)

    # Forward --resume directly to train_builder(). Resume is a pure
    # CLI concern — the YAML has no ``training.resume`` field, so
    # ``args.resume`` is the single source of truth for this choice.
    train_builder(
        yaml_config,
        config_path=config_path,
        cli_swanlab_id=args.swanlab_id,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
