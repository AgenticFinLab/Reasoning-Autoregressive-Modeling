"""Predictor-specialised twin of ``loss_prepare.py``.

Why a separate entry point:
    ``loss_prepare.py`` already handles EVERY non-model concern —
    dataset iteration, group-shared batch sampling, per-config skip /
    retry, dataset-scoped ``Loss_prepare.json``, OOM-aware GPU hopping,
    ``_stats`` / ``_aggregate`` for mean/std/min/max summaries.  The
    one thing it does NOT handle is the PREDICTOR model: its
    ``run_predictor_n_batches`` is a ``NotImplementedError`` stub, and
    predictor training requires an extra step that the builder path
    does not have — loading the paired FROZEN builder and feeding its
    concepts as ground truth.

    This script reuses every helper in ``loss_prepare`` via a plain
    import and adds exactly ONE predictor-aware runner.  Nothing in
    ``loss_prepare.py`` is modified.

Module hardcoded to ``predictor``:
    ``-m`` is not exposed; the runner rejects it if passed.  For
    builder inspection, use ``loss_prepare.py`` directly.

``-M``/``--mode`` filter (shared vs independent):
    A predictor YAML declares its mode via
    ``model.predictor.use_shared_model`` — SHARED (``True``) and
    INDEPENDENT (``False``) variants live side-by-side in the same
    dataset directory.  Their compute + memory profiles differ so
    much (1x vs 2x backbone, no-opt vs LoRA-opt) that mixing their
    results in a single JSON makes downstream weight-tuning harder.
    This script therefore:
      * Accepts ``-M/--mode {shared,independent,both}`` (default:
        ``both``).  When ``both``, each config is routed to its own
        mode's output file based on the YAML it came from.
      * Writes MODE-SCOPED output files (one per mode):
          <storage_root>/EXPERIMENT/nlcpV4/predictor/
              <dataset>_Loss_prepare_shared.json
              <dataset>_Loss_prepare_independent.json
        so each JSON contains EXACTLY one mode's configs and can be
        loaded / diffed / plotted independently.

Differences from ``loss_prepare.run_builder_n_batches``:
    * Builder config is read from ``model.builder.config_path`` in the
      predictor YAML and parsed at runtime.
    * The pyramid block (``model.pyramid``) is INHERITED into the
      predictor config (fail-fast if predictor already declares it),
      mirroring ``_inherit_pyramid_from_builder`` in
      ``train_predictor.py`` so geometry cannot drift.
    * The builder checkpoint path is resolved with the launcher's
      storage-root (same convention as ``_resolve_builder_checkpoint_path``).
    * The shared/independent constraint (``use_shared_model=True`` ⇒
      ``lora is None``; ``use_shared_model=False`` ⇒ not
      ``freeze=True & lora=None``) is enforced up front.
    * Forward pass follows ``_run_predictor_step``:
        builder(_strip_solutions(batch)) → gt_concepts
            → tokenize(Q, S) → predictor(...)  → PredictorOutput
    * Loss uses ``compute_predictor_loss`` (no ``ordering_loss_type``).
      Loss components are ``concept`` and ``reasoning``.
      ``concept_per_level`` is captured as a list (not a scalar) and
      stored separately so the aggregate helper is not broken.

Usage:
    # BOTH modes (default) — one forward per config, each routed to
    # its own mode-scoped output file:
    python3 examples/RunResults/loss_predictor_prepare.py -d GSM8K

    # Only shared-backbone variants:
    python3 examples/RunResults/loss_predictor_prepare.py -d GSM8K -M shared

    # Only independent (+LoRA) variants, with averaged stats:
    python3 examples/RunResults/loss_predictor_prepare.py -d GSM8K -M independent -n 5

    # Mixed batch sizes + non-default storage root:
    python3 examples/RunResults/loss_predictor_prepare.py -d GSM8K -n 5 -f -s /Data/<proj>

Output files (one per mode):
    <storage_root>/EXPERIMENT/nlcpV4/predictor/<dataset>_Loss_prepare_shared.json
    <storage_root>/EXPERIMENT/nlcpV4/predictor/<dataset>_Loss_prepare_independent.json

Re-run tip:
    Identical semantics to ``loss_prepare.py`` — successful configs
    are persisted immediately and ``[SKIP]``-ed on subsequent runs,
    so re-running the same command picks up where the last pass left
    off.  Transient OOMs on shared GPUs therefore do not waste work.
"""

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import torch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path injection: reuse loss_prepare's PROJECT_ROOT sys.path setup by
# importing it first.  Import side-effects (sys.path.insert for package
# resolution) are therefore shared — we do NOT duplicate them here.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import loss_prepare as _lp  # noqa: E402
from loss_prepare import (  # noqa: E402
    PROJECT_ROOT,
    _aggregate,
    _pick_device_fresh,
    collect_batches,
)

# Local imports (nlcpV4 resolves via loss_prepare's sys.path setup).
from nlcpV4.concept_predictor import ConceptPredictor  # noqa: E402
from nlcpV4.losses import compute_predictor_loss  # noqa: E402

# Reuse train_predictor's exact helpers so this script never drifts
# from the trainer's build/step semantics.
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "nlcpV4"))
from train_predictor import (  # noqa: E402
    _fail_fast_shared_sanity,
    _inherit_pyramid_from_builder,
    _load_frozen_builder,
    _resolve_builder_checkpoint_path,
    _run_predictor_step,
)

from ram.utils import load_config, setup_environment  # noqa: E402
from lmbase.utils.env_tools import get_device  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("loss_predictor_prepare")

# Hardcoded — this script is predictor-only (mirror run_predictor_experiments.py).
MODULE = "predictor"

# Valid values for -M/--mode.  "both" is a meta-value that expands to
# ["shared", "independent"] at runtime (it is NEVER used as a file
# suffix — per-config routing always picks the concrete mode).
VALID_MODES = ("shared", "independent", "both")
CONCRETE_MODES = ("shared", "independent")

# Mode-scoped output filename template.  The mode suffix is mandatory —
# there is no un-scoped variant because shared/independent results must
# never be co-mingled (documented in module docstring).
OUT_FILENAME_TEMPLATE = "{dataset}_Loss_prepare_{mode}.json"


# =====================================================================
# Mode-scoped output-path helpers
# =====================================================================


def _pred_loss_path(dataset: str, storage_root: str, mode: str) -> Path:
    """Return the absolute path of the mode-scoped Loss_prepare JSON.

    Layout mirrors ``loss_prepare.loss_prepare_path`` but appends the
    mode to the filename so SHARED and INDEPENDENT results land in
    separate files and can be analysed independently.
    """
    if mode not in CONCRETE_MODES:
        raise ValueError(
            f"_pred_loss_path got mode={mode!r}; expected one of "
            f"{CONCRETE_MODES} (the 'both' meta-value must be expanded "
            f"into concrete modes by the caller)."
        )
    base = Path(storage_root)
    filename = OUT_FILENAME_TEMPLATE.format(dataset=dataset, mode=mode)
    return base / "EXPERIMENT" / "nlcpV4" / "predictor" / filename


def _print_pred_paths(dataset: str, storage_root: str, modes: list[str]) -> None:
    """Print a ``[STORAGE]`` banner covering every mode the run will touch."""
    print(f"[STORAGE] storage_root = {storage_root!r} (cwd={Path.cwd().resolve()})")
    for m in modes:
        rel = _pred_loss_path(dataset, storage_root, m)
        abs_path = rel.expanduser()
        if not abs_path.is_absolute():
            abs_path = (Path.cwd() / abs_path).resolve()
        print(f"[STORAGE]   [{m:<11}] {rel.name} = {rel}")
        print(f"[STORAGE]              {'':<11}  (absolute: {abs_path})")


def _load_pred(dataset: str, storage_root: str, mode: str) -> dict:
    """Load the mode-scoped Loss_prepare JSON (empty dict if absent)."""
    path = _pred_loss_path(dataset, storage_root, mode)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_pred(dataset: str, storage_root: str, mode: str, data: dict) -> None:
    """Persist the mode-scoped Loss_prepare JSON (creates parent dirs)."""
    path = _pred_loss_path(dataset, storage_root, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _mode_of(config: dict) -> str:
    """Classify a predictor config as 'shared' or 'independent'."""
    return (
        "shared" if config["model"]["predictor"]["use_shared_model"] else "independent"
    )


# =====================================================================
# CLI
# =====================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.  Same flags as loss_prepare except -m."""
    parser = argparse.ArgumentParser(
        description=(
            "Predictor-specialised loss inspection runner.  Module is "
            "hardcoded to 'predictor' — use loss_prepare.py for builder."
        )
    )
    parser.add_argument(
        "-s",
        "--storage-root",
        type=str,
        default="./",
        help=(
            "Prefix used to compute the Loss_prepare.json location: "
            "<storage_root>/EXPERIMENT/nlcpV4/predictor/"
            "<dataset>_Loss_prepare.json.  Also used to resolve relative "
            "'model.builder.checkpoint_path' values inside each predictor "
            "config (mirrors train_predictor.py's -s semantics).  "
            "Default is './' (current working directory)."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., GSM8K). Resolves configs/nlcpV4/{dataset}/",
    )
    parser.add_argument(
        "-n",
        "--num-batches",
        type=int,
        default=1,
        help=(
            "Number of batches to sample (default: 1). Each config runs "
            "N forward passes; per-batch losses plus mean/std/min/max "
            "aggregates are recorded."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help=(
            "When configs have mixed batch sizes, force the run by "
            "grouping configs by batch_size and sampling batches once "
            "per group.  Without -f, a mixed run aborts with an error."
        ),
    )
    parser.add_argument(
        "-M",
        "--mode",
        type=str,
        default="both",
        choices=sorted(VALID_MODES),
        help=(
            "Filter configs by predictor mode.  'shared' processes only "
            "configs with model.predictor.use_shared_model=true, "
            "'independent' only those with use_shared_model=false, "
            "'both' (default) processes all configs and ROUTES each to "
            "its mode-scoped output file.  SHARED / INDEPENDENT results "
            "are NEVER co-mingled — each mode has its own "
            "<dataset>_Loss_prepare_<mode>.json."
        ),
    )
    return parser.parse_args()


def _reject_module_flag(argv: list[str]) -> None:
    """Reject ``-m``/``--module``; this script is predictor-hardcoded."""
    for token in argv:
        if token in ("-m", "--module") or token.startswith("--module="):
            print(
                "[ERROR] loss_predictor_prepare.py does NOT accept "
                "-m/--module — it's hardcoded to 'predictor'.  Use "
                "loss_prepare.py directly for builder inspection.",
                file=sys.stderr,
            )
            sys.exit(2)


# =====================================================================
# Predictor batch runner
# =====================================================================


def run_predictor_n_batches(
    config: dict,
    device: str,
    batches: list,
    storage_root: str,
) -> dict:
    """Build predictor (+ frozen builder) and run N loss-inspection forwards.

    Mirrors ``loss_prepare.run_builder_n_batches`` in shape (the
    per_batch / stats output is identical up to the component keys) but
    adds the extra plumbing that predictor forwards require:

      1. Load & inherit the paired BUILDER config's pyramid block into
         ``config["model"]["pyramid"]`` (fail-fast on drift).
      2. Enforce the shared/independent + LoRA/freeze constraint.
      3. Resolve and load the FROZEN builder checkpoint under
         ``storage_root`` (same convention as the trainer).
      4. Instantiate ``ConceptPredictor`` with ``builder=`` wired up so
         SHARED mode correctly aliases ``reason_model`` / ``back_proj``.
      5. Kill autograd on every parameter and run each batch through
         ``_run_predictor_step`` under ``torch.no_grad()``.
      6. Compute ``compute_predictor_loss`` per batch; record scalar
         components AND per-level concept losses for logging.

    Memory hygiene matches the builder path: ``empty_cache`` between
    batches, explicit ``del`` of intermediates, no autograd state.
    """
    # --- Work on a deep copy so mutations (pyramid inheritance) never
    # leak into the caller's config dict or the shared group batches.
    config = deepcopy(config)

    # 1. Load the paired builder config.
    builder_cfg_raw = config["model"]["builder"]["config_path"]
    builder_cfg_path = Path(builder_cfg_raw)
    if not builder_cfg_path.is_absolute():
        builder_cfg_path = PROJECT_ROOT / builder_cfg_path
    if not builder_cfg_path.is_file():
        raise FileNotFoundError(
            f"Paired builder config not found: {builder_cfg_path} "
            f"(check model.builder.config_path)."
        )
    builder_config = load_config(str(builder_cfg_path))

    # 2. Inherit pyramid geometry from the builder (fail-fast on drift).
    _inherit_pyramid_from_builder(config, builder_config)

    # 3. Enforce shared/LoRA/freeze invariants.
    _fail_fast_shared_sanity(config)

    # 4. Resolve the frozen-builder checkpoint under storage_root.
    builder_ckpt_raw = config["model"]["builder"]["checkpoint_path"]
    builder_strict = config["model"]["builder"]["strict_load"]
    builder_ckpt_path = _resolve_builder_checkpoint_path(
        builder_ckpt_raw, storage_root
    ).resolve()

    # 5. Instantiate the FROZEN builder (eval mode, no grads).
    builder = _load_frozen_builder(
        builder_config, builder_ckpt_path, builder_strict, device, logger
    )

    # 6. Instantiate the predictor.  SHARED mode ties reason_model +
    # back_proj to the builder; INDEPENDENT mode allocates a fresh copy.
    predictor = ConceptPredictor(config, builder=builder).to(device)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)
    # Re-assert eval on the (possibly shared) reason_model after
    # eval()+requires_grad flips — predictor.train() default has
    # already been overridden by the line above, but do it explicitly
    # to document the invariant (matches train_predictor.py lines
    # 911-912 rationale).
    if config["model"]["predictor"]["use_shared_model"]:
        predictor.reason_model.eval()

    train_cfg = config["training"]
    loss_weights = train_cfg["loss_weights"]
    batch_size = int(train_cfg["batch_size"])
    max_length = int(config["model"]["pyramid"]["max_seq_len"])

    # Predictor weight key map differs from builder — only two
    # components (concept + reasoning) and no ordering / residual.
    weight_key_map = {
        "concept": "concept_loss_weight",
        "reasoning": "reasoning_loss_weight",
    }

    per_batch: list = []
    weights: dict = {}
    with torch.no_grad():
        for batch in batches:
            output = _run_predictor_step(predictor, builder, batch, max_length, device)
            _total, loss_dict = compute_predictor_loss(output, loss_weights)
            # ``concept_per_level`` is a list; strip it from the scalar
            # dict and keep it under a separate key for logging.
            raw = {
                k: float(v)
                for k, v in loss_dict.items()
                if k not in ("total", "concept_per_level")
            }
            concept_per_level = loss_dict.get("concept_per_level")
            if not weights:
                weights = {
                    k: float(loss_weights[wk])
                    for k, wk in weight_key_map.items()
                    if k in raw and wk in loss_weights
                }
            weighted = {k: raw[k] * weights[k] for k in weights if k in raw}
            record = {
                "raw": {k: round(v, 6) for k, v in raw.items()},
                "weighted": {k: round(v, 6) for k, v in weighted.items()},
                "total_weighted": round(float(loss_dict["total"]), 6),
            }
            if concept_per_level is not None:
                record["concept_per_level"] = [
                    round(float(v), 6) for v in concept_per_level
                ]
            per_batch.append(record)
            # Drop per-batch tensors before the next forward so the
            # CUDA allocator can recycle their slots instead of growing.
            del output, _total, loss_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = {
        "batch_size": batch_size,
        "num_batches": len(per_batch),
        "mode": (
            "shared"
            if config["model"]["predictor"]["use_shared_model"]
            else "independent"
        ),
        "weights": weights,
        "per_batch": per_batch,
        "stats": _aggregate(per_batch),
    }

    # Release GPU memory before the next config starts.
    del predictor, builder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# =====================================================================
# Entry point (structure mirrors loss_prepare.main, specialised for predictor)
# =====================================================================


def main() -> int:
    """Iterate every ``train_predictor_*.yml`` under the dataset dir.

    Skeleton is intentionally close to ``loss_prepare.main`` so the two
    behave identically w.r.t. argument parsing, group sampling, skip /
    retry semantics, and JSON persistence.  The only per-config hook
    that changes is the runner function (``run_predictor_n_batches``
    instead of ``run_builder_n_batches``).
    """
    _reject_module_flag(sys.argv[1:])
    args = parse_args()
    dataset: str = args.dataset
    num_batches: int = max(1, int(args.num_batches))
    force: bool = bool(args.force)
    storage_root: str = args.storage_root
    mode_filter: str = args.mode

    # Expand the 'both' meta-value.  Downstream logic always deals with
    # CONCRETE modes only — file routing, counters, skip checks etc.
    target_modes = list(CONCRETE_MODES) if mode_filter == "both" else [mode_filter]

    # Print every mode-scoped JSON path up front so a misconfigured
    # ``-s`` / ``-M`` is caught before any compute happens.
    _print_pred_paths(dataset, storage_root, target_modes)

    configs_dir = PROJECT_ROOT / "configs" / "nlcpV4" / dataset
    if not configs_dir.is_dir():
        logger.error("Configs dir not found: %s", configs_dir)
        return 1

    prefix = f"train_{MODULE}_"
    yml_files = sorted(configs_dir.glob(f"{prefix}*.yml"))
    if not yml_files:
        logger.error("No YAML configs matching '%s*.yml' under %s", prefix, configs_dir)
        return 1

    load_dotenv(PROJECT_ROOT / ".env")

    # Pre-load every config, classify by mode, and filter against
    # ``target_modes``.  A config whose mode is not in the filter is
    # silently dropped (its output file — if any — is left untouched).
    loaded: list = []  # list of (Path, cfg, mode)
    n_filtered = 0
    for p in yml_files:
        try:
            cfg = load_config(str(p))
        except Exception as e:
            logger.error("Failed to load %s: %s", p.name, e)
            continue
        try:
            mode = _mode_of(cfg)
        except KeyError as e:
            logger.error(
                "Config %s missing model.predictor.use_shared_model: %s",
                p.name,
                e,
            )
            continue
        if mode not in target_modes:
            n_filtered += 1
            continue
        loaded.append((p, cfg, mode))
    if not loaded:
        logger.error(
            "No configs match mode filter %r (filtered %d).",
            mode_filter,
            n_filtered,
        )
        return 1

    # Group by training.batch_size so batch sampling can be reused.
    # Mode diversity WITHIN a group is fine — the per-config loop
    # routes each result to its own file based on its stored mode.
    groups: dict = {}
    for p, cfg, mode in loaded:
        bs = int(cfg["training"]["batch_size"])
        groups.setdefault(bs, []).append((p, cfg, mode))

    bs_list = sorted(groups.keys())
    if len(bs_list) > 1 and not force:
        logger.error(
            "Configs under '%s' have mixed batch sizes: %s. "
            "Pass -f/--force to group by batch_size and proceed.",
            configs_dir,
            bs_list,
        )
        return 1

    first_seed = int(loaded[0][1]["environment"]["seed"])
    setup_environment({"seed": first_seed, "device": "auto"})
    initial_device = str(get_device("auto"))

    # Per-mode config counts (visibility for the operator).
    per_mode_count = {m: sum(1 for _, _, mm in loaded if mm == m) for m in target_modes}
    logger.info(
        "Initial device=%s | module=%s | dataset=%s | mode_filter=%s "
        "| per_mode=%s | num_batches=%d | batch_size_groups=%s "
        "| %d config(s) (filtered out: %d)",
        initial_device,
        MODULE,
        dataset,
        mode_filter,
        per_mode_count,
        num_batches,
        bs_list,
        len(loaded),
        n_filtered,
    )

    # Per-mode counters so the final summary reports shared and
    # independent separately (matching the separated output files).
    n_skip = {m: 0 for m in CONCRETE_MODES}
    n_run = {m: 0 for m in CONCRETE_MODES}
    n_fail = {m: 0 for m in CONCRETE_MODES}

    for bs in bs_list:
        group = groups[bs]
        logger.info("---- Group batch_size=%d | %d config(s) ----", bs, len(group))

        group_first_cfg = group[0][1]
        try:
            shared_batches = collect_batches(group_first_cfg["data"], bs, num_batches)
        except Exception as e:
            logger.error(
                "Failed to sample batches for group bs=%d: %s",
                bs,
                e,
                exc_info=True,
            )
            for _, _, mode in group:
                n_fail[mode] += 1
            continue

        if not shared_batches:
            logger.error(
                "No batches produced for group bs=%d; skipping its %d config(s).",
                bs,
                len(group),
            )
            for _, _, mode in group:
                n_fail[mode] += 1
            continue

        for config_path, config, mode in group:
            key = f"{dataset}/{config_path.stem}"
            # Route both the skip-check AND the save to the MODE-scoped
            # file so SHARED and INDEPENDENT stores never overlap.
            store = _load_pred(dataset, storage_root, mode)

            if key in store:
                logger.info("[SKIP] %s (mode=%s, key already present)", key, mode)
                n_skip[mode] += 1
                continue

            excluded: set[int] = set()
            result = None
            last_exc: Exception | None = None
            max_attempts = torch.cuda.device_count() if torch.cuda.is_available() else 1

            for attempt in range(1, max_attempts + 1):
                try:
                    current_device = _pick_device_fresh(exclude=excluded)
                except Exception as e:
                    last_exc = e
                    break

                logger.info(
                    "[RUN ] %s (mode=%s, device=%s, attempt=%d/%d, bs=%d, n=%d)",
                    key,
                    mode,
                    current_device,
                    attempt,
                    max_attempts,
                    bs,
                    len(shared_batches),
                )

                seed = int(config["environment"]["seed"])
                torch.manual_seed(seed)
                if torch.cuda.is_available() and current_device.startswith("cuda:"):
                    dev_idx = int(current_device.split(":")[1])
                    with torch.cuda.device(dev_idx):
                        torch.cuda.manual_seed(seed)

                try:
                    result = run_predictor_n_batches(
                        config, current_device, shared_batches, storage_root
                    )
                    last_exc = None
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    msg = str(e).lower()
                    is_oom = "out of memory" in msg or isinstance(
                        e, torch.cuda.OutOfMemoryError
                    )
                    if not is_oom:
                        last_exc = e
                        break
                    logger.warning(
                        "[RETRY] %s OOM on %s (attempt %d/%d); trying another GPU.",
                        key,
                        current_device,
                        attempt,
                        max_attempts,
                    )
                    if current_device.startswith("cuda:"):
                        excluded.add(int(current_device.split(":")[1]))
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    last_exc = e
                    continue
                except Exception as e:
                    last_exc = e
                    break

            if result is None:
                logger.error(
                    "Failed on %s (mode=%s, tried %d GPU(s)): %s",
                    key,
                    mode,
                    len(excluded) or 1,
                    last_exc,
                    exc_info=last_exc is not None,
                )
                n_fail[mode] += 1
                continue

            # Re-read the MODE-scoped store just before write to
            # preserve concurrent edits (another terminal running the
            # same command on the same dataset but a different mode
            # writes to a different file, so cross-mode concurrency is
            # safe; within a mode, this re-read guards overlap).
            store = _load_pred(dataset, storage_root, mode)
            store[key] = result
            _save_pred(dataset, storage_root, mode, store)

            total_stats = result["stats"]["total_weighted"]
            weighted_summary = ", ".join(
                f"{k}={v['mean']:.4f}\u00b1{v['std']:.4f}"
                for k, v in result["stats"]["weighted"].items()
            )
            logger.info(
                "[SAVE] %s | mode=%s | bs=%d | total=%.4f\u00b1%.4f (n=%d) | %s",
                key,
                mode,
                bs,
                total_stats["mean"],
                total_stats["std"],
                result["num_batches"],
                weighted_summary,
            )
            n_run[mode] += 1

    logger.info(
        "Done. dataset=%s mode_filter=%s bs_groups=%s "
        "| shared: run=%d skip=%d fail=%d "
        "| independent: run=%d skip=%d fail=%d "
        "| total_loaded=%d (filtered_out=%d)",
        dataset,
        mode_filter,
        bs_list,
        n_run["shared"],
        n_skip["shared"],
        n_fail["shared"],
        n_run["independent"],
        n_skip["independent"],
        n_fail["independent"],
        len(loaded),
        n_filtered,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
