"""Predictor-specialized launcher built on top of ``run_experiments.py``.

Why a separate entry point:
    ``run_experiments.py -m predictor`` already fans out predictor
    trainings into tmux sessions (CLI, conda pre-flight, tmux launch,
    per-session logs, resume+swanlab forwarding, etc. are all generic
    across builder and predictor). The ONE piece that breaks is
    ``estimate_experiment_memory_mb`` — it expects builder schema
    (``cfg.model.reason_model`` / ``cfg.training.reason_model`` /
    ``cfg.model.pyramid``) which predictor YAMLs intentionally do NOT
    carry: predictor configs inherit ``model.pyramid`` from the paired
    builder config at runtime, and declare their own knobs under
    ``model.predictor`` / ``training.predictor``.

    So the default memory estimator ``KeyError``s on every predictor
    YAML and the launcher aborts before any tmux session is started.

    This wrapper fixes it without touching the original file:
      1. Import every reusable helper from ``run_experiments``.
      2. Define ONE predictor-aware replacement:
         ``estimate_predictor_memory_mb`` — loads the paired builder
         YAML to recover backbone/pyramid geometry, then accounts
         for SHARED (1× backbone in VRAM) vs INDEPENDENT (2× backbone:
         frozen builder + trainable predictor with LoRA).
      3. Monkey-patch that replacement onto the imported module.
      4. Inject ``-m predictor`` into ``sys.argv`` and delegate to
         ``run_experiments.main()``.

CLI contract:
    Identical to ``run_experiments.py`` EXCEPT:
      * ``-m/--module`` MUST NOT be passed (hardcoded to 'predictor');
        the wrapper rejects it with a clear error.
      * ``-d/--dataset`` and ``-e/--experiments`` point at predictor
        YAMLs, e.g.
            configs/nlcpV4/GSM8K/train_predictor_Qwen2.5-0.5B_3level_shared.yml

    Every other flag (``-s``, ``--resume``, ``--swanlab-ids``,
    ``--one-per-gpu``, ``--wait-for-gpu``, ``--gpus``,
    ``--mem-per-exp-mb``, ``--mem-safety-factor``,
    ``--gpu-memory-fraction``, ``--gpu-idle-mem-fraction``,
    ``--warmup-seconds``, ``--launch-stagger``, ``--poll-interval``,
    ``--conda-env``, ``--python-version``, ``--log-dir``,
    ``--keep-alive``, ``--kill-existing``, ``--dry-run``) behaves
    exactly as documented in ``run_experiments.py`` — no reinvention.

Usage:
    # A/B-launch shared + independent variants on any idle GPU:
    python3 examples/RunResults/run_predictor_experiments.py \
        -d GSM8K \
        -e train_predictor_Qwen2.5-0.5B_3level_shared.yml \
           train_predictor_Qwen2.5-0.5B_3level_independent.yml \
        --one-per-gpu

    # Full 84-cell grid, memory-packed, on a big machine:
    python3 examples/RunResults/run_predictor_experiments.py \
        -d GSM8K -e train_predictor_*_shared.yml train_predictor_*_independent.yml \
        -s /Data/ReasoningNLCP --wait-for-gpu

    # Preview the plan (incl. estimated memory + GPU assignments):
    python3 examples/RunResults/run_predictor_experiments.py \
        -d GSM8K -e train_predictor_Qwen3-8B_8level_independent.yml \
        --dry-run

    # Resume a batch:
    python3 examples/RunResults/run_predictor_experiments.py \
        -d GSM8K -e train_predictor_Qwen2.5-0.5B_3level_shared.yml \
        -s /Data/ReasoningNLCP --resume --swanlab-ids abc123

Memory estimation model (SHARED vs INDEPENDENT):
    Let P = backbone params, b = bytes/elem (2 for bf16/fp16, 4 for fp32).
      SHARED      : weights = 1 × P × b            (tied, frozen)
                    optimizer = 0 (backbone not trained; only heads +
                                   back_proj alias + lvl/pos embeds)
      INDEPENDENT : weights = 2 × P × b            (builder frozen +
                                                   predictor trainable)
                    optimizer = 0.2 × P × b         (LoRA/frozen base
                                                   means only adapter
                                                   grads/state; ≈20%)
    Activation term matches ``run_experiments.py``:
      batch_size × max_seq_len × hidden_dim × b × 30
    Total is then multiplied by ``--mem-safety-factor`` (default 1.5).
    Override the whole heuristic with ``--mem-per-exp-mb`` when you
    have measured numbers.
"""

import sys
from pathlib import Path

import yaml

# Ensure the sibling ``run_experiments`` module is importable when this
# script is invoked directly (``python3 examples/RunResults/...``). The
# script's own directory is already on sys.path[0] by default, but make
# it explicit so behaviour is stable regardless of launch method.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import run_experiments as _re  # noqa: E402
from run_experiments import PROJECT_ROOT, parse_model_params_b  # noqa: E402

# =====================================================================
# Predictor-aware memory estimator (the only real override)
# =====================================================================


def estimate_predictor_memory_mb(
    config_path: Path,
    safety_factor: float,
) -> tuple[int, dict]:
    """Estimate GPU memory (MiB) for a predictor training.

    Predictor YAMLs intentionally lack ``model.reason_model`` and
    ``model.pyramid``. Those live in the paired BUILDER config pointed
    to by ``model.builder.config_path``. This function follows that
    pointer to recover the backbone + pyramid geometry, then layers
    predictor-specific adjustments on top:

      * SHARED mode (``use_shared_model=True``):
          - reason_model + back_proj are ALIASES of the frozen builder.
          - Only 1× backbone lives in VRAM.
          - Backbone is never trained → no Adam state for it.
          - Trainable footprint is the 3 small heads
            (level_embeddings, position_embeddings, concept_head),
            negligible relative to the backbone.

      * INDEPENDENT mode (``use_shared_model=False``):
          - Builder carries its own frozen reason_model (for gt_concepts).
          - Predictor holds a SECOND reason_model (LoRA-trainable).
          - So 2× backbone in VRAM.
          - Optimizer state is sized to the LoRA path (~0.2×).

    Returns ``(total_mb_int, breakdown_dict)``.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        pred_cfg = yaml.safe_load(f)

    # ── Follow the pointer to the paired builder config ─────────────
    builder_cfg_path_raw = pred_cfg["model"]["builder"]["config_path"]
    builder_cfg_path = Path(builder_cfg_path_raw)
    if not builder_cfg_path.is_absolute():
        builder_cfg_path = PROJECT_ROOT / builder_cfg_path
    if not builder_cfg_path.is_file():
        raise FileNotFoundError(
            f"Paired builder config not found for {config_path.name}: "
            f"{builder_cfg_path} (check model.builder.config_path)."
        )
    with open(builder_cfg_path, "r", encoding="utf-8") as f:
        builder_cfg = yaml.safe_load(f)

    builder_reason = builder_cfg["model"]["reason_model"]
    pyramid_cfg = builder_cfg["model"]["pyramid"]
    pred_block = pred_cfg["model"]["predictor"]
    train_pred = pred_cfg["training"]["predictor"]

    # ── Which backbone ends up on the GPU ───────────────────────────
    use_shared = bool(pred_block["use_shared_model"])
    if use_shared:
        model_name = builder_reason["reason_model_name"]
        backbone_copies = 1  # tied to builder's frozen backbone
        # SHARED always runs the backbone frozen (see predictor YAML
        # rationale comments). No Adam state for the backbone.
        lora_or_frozen = True
        backbone_trains = False
    else:
        model_name = pred_block.get("predictor_model_name")
        if not model_name:
            raise ValueError(
                f"INDEPENDENT predictor config {config_path.name} has "
                f"predictor_model_name=null — required when "
                f"use_shared_model=False."
            )
        # Frozen builder (for gt_concepts) + trainable predictor backbone.
        backbone_copies = 2
        lora_or_frozen = (train_pred["lora"] is not None) or bool(train_pred["freeze"])
        backbone_trains = True

    params_b = parse_model_params_b(model_name)
    params = params_b * 1e9

    dtype = builder_reason["torch_dtype"]
    bytes_per_elem = 2 if dtype in ("bfloat16", "float16") else 4

    # Raw backbone weights in VRAM.
    weights_mb = params * bytes_per_elem * backbone_copies / 1e6

    # Optimizer state is applied to AT MOST one backbone copy (the
    # trainable one). SHARED mode has no backbone to train, so 0.
    if not backbone_trains:
        optimizer_mb = 0.0
    else:
        one_backbone_mb = params * bytes_per_elem / 1e6
        # LoRA / frozen base ≈ 0.2×; full FT ≈ 3× (grads + m + v for Adam).
        optimizer_mb = one_backbone_mb * (0.2 if lora_or_frozen else 3.0)

    # Activation footprint — same rule of thumb as the builder path.
    # The predictor's unified forward runs ONE reason_model pass over
    # [Q, C_gt, S] each step; ``max_seq_len`` already covers that.
    batch_size = int(pred_cfg["training"]["batch_size"])
    max_seq_len = int(pyramid_cfg["max_seq_len"])
    hidden_dim = int(pyramid_cfg["hidden_dim"])
    activation_mb = batch_size * max_seq_len * hidden_dim * bytes_per_elem * 30 / 1e6

    base_mb = weights_mb + optimizer_mb + activation_mb
    total_mb = int(base_mb * safety_factor)

    breakdown = {
        "mode": "shared" if use_shared else "independent",
        "model": model_name,
        "params_B": params_b,
        "dtype": dtype,
        "backbone_copies": backbone_copies,
        "backbone_trains": backbone_trains,
        "lora_or_frozen": lora_or_frozen,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "hidden_dim": hidden_dim,
        "weights_mb": int(weights_mb),
        "optimizer_mb": int(optimizer_mb),
        "activation_mb": int(activation_mb),
        "safety_factor": safety_factor,
        "total_mb": total_mb,
        "builder_config": (
            str(builder_cfg_path.relative_to(PROJECT_ROOT))
            if builder_cfg_path.is_relative_to(PROJECT_ROOT)
            else str(builder_cfg_path)
        ),
    }
    return total_mb, breakdown


# =====================================================================
# Entry point — monkey-patch + delegate
# =====================================================================


def _reject_module_flag(argv: list[str]) -> None:
    """Hard-error if the user tries to pass ``-m``/``--module``.

    This wrapper is hardcoded to ``-m predictor``. Accepting a
    conflicting value would make the CLI lie about its behaviour.
    Users who want to pick the module at runtime should invoke
    ``run_experiments.py`` directly.
    """
    for token in argv:
        if token == "-m" or token == "--module" or token.startswith("--module="):
            print(
                "[ERROR] run_predictor_experiments.py does NOT accept "
                "-m/--module — it's hardcoded to 'predictor'. To pick "
                "the module at the CLI, use run_experiments.py instead.",
                file=sys.stderr,
            )
            sys.exit(2)


def main() -> int:
    """Wrapper entry point.

    Validates CLI, installs the predictor-aware memory estimator,
    injects ``-m predictor`` into ``sys.argv``, and delegates to
    ``run_experiments.main()``.
    """
    _reject_module_flag(sys.argv[1:])

    # Hot-swap the memory estimator. All of run_experiments' scheduling
    # code calls this symbol through the module, so replacing the
    # attribute is sufficient to redirect every code path.
    _re.estimate_experiment_memory_mb = estimate_predictor_memory_mb

    # Inject ``-m predictor`` so the shared parser is satisfied without
    # requiring the caller to type it.
    sys.argv[1:1] = ["-m", "predictor"]

    print(
        "[PREDICTOR] run_predictor_experiments.py — specialised launcher "
        "for Stage-2 training (reads backbone/pyramid from the paired "
        "builder config for memory estimation)."
    )
    return _re.main()


if __name__ == "__main__":
    sys.exit(main())
