"""Prepare loss values across all configs in a dataset for weight tuning.

Purpose:
    Iterate over every matching YAML config in configs/nlcpV4/{dataset}/
    and run a short forward pass so researchers can inspect raw / weighted
    loss components (per-batch and mean/std/min/max aggregates) and decide
    how to adjust loss_weights.

Behaviour:
    1. Select configs whose filename starts with `train_{module}_`, where
       module is supplied via `-m builder` or `-m predictor`.
    2. Check EXPERIMENT/nlcpV4/{module}/Loss_prepare.json for the config key
       "{dataset}/{config_stem}". If present -> [SKIP].
    3. Otherwise build the model, reuse the group-shared batches, run a
       forward pass with torch.no_grad() for each batch, record per-batch
       raw + weighted + total losses plus mean/std/min/max aggregates,
       then free the model and continue.

Efficiency:
    * Device probe runs exactly once at startup.
    * All YAML configs are expected to share a single `training.batch_size`;
      if they do, batches are sampled ONCE and reused for every config.
    * If batch sizes differ, pass `-f/--force` to group configs by
      batch_size; batches are sampled once per group and reused within
      the group. Without `-f` the run aborts with a clear message.

Re-run tip (IMPORTANT):
    A single invocation may not succeed for every config — e.g., a
    shared-cluster GPU race causes a transient OOM, or all GPUs are
    momentarily tight. Successfully-finished configs are persisted to
    `Loss_prepare.json` right away, and already-present keys are
    `[SKIP]`-ed on subsequent runs. So if some configs failed, JUST
    RE-RUN THE SAME COMMAND one or more times — each pass will pick up
    where the last one stopped and only retry the configs that are
    still missing from the result file, until every config has been
    recorded.

Usage:
    python3 examples/RunResults/loss_prepare.py -m builder -d GSM8K
    python3 examples/RunResults/loss_prepare.py -m builder -d GSM8K -n 5
    python3 examples/RunResults/loss_prepare.py -m builder -d GSM8K -n 5 -f
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

# Project-root path injection must precede local imports so that the
# ``nlcpV4``, ``lmbase``, and ``ram`` packages resolve when this script
# is executed directly (``python3 examples/RunResults/loss_prepare.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from nlcpV4.concept_builder import ConceptPyramidBuilder
from nlcpV4.data_loader import NLCPV4DataLoader
from nlcpV4.losses import compute_builder_loss
from ram.utils import load_config, setup_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("loss_prepare")

OUT_FILENAME = "Loss_prepare.json"
VALID_MODULES = ("builder", "predictor")


def parse_args():
    """Parse CLI arguments for the loss_prepare runner."""
    parser = argparse.ArgumentParser(
        description="Prepare loss inspection across a dataset's configs"
    )
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        required=True,
        choices=sorted(VALID_MODULES),
        help="Module name: 'builder' or 'predictor'. Only configs whose "
        "filename starts with 'train_{module}_' will be processed.",
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
        help="Number of batches to sample (default: 1). Each config runs "
        "N forward passes; per-batch losses plus mean/std/min/max "
        "aggregates are recorded.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="When configs have mixed batch sizes, force the run by "
        "grouping configs by batch_size and sampling batches once "
        "per group. Without -f, a mixed run aborts with an error.",
    )
    return parser.parse_args()


def loss_prepare_path(module: str) -> Path:
    """Return the absolute path of ``Loss_prepare.json`` for a given module."""
    return PROJECT_ROOT / "EXPERIMENT" / "nlcpV4" / module / OUT_FILENAME


def load_loss_prepare(module: str) -> dict:
    """Load the persisted Loss_prepare.json store for ``module`` (empty if absent)."""
    path = loss_prepare_path(module)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_loss_prepare(module: str, data: dict) -> None:
    """Persist the Loss_prepare.json store for ``module`` (creates parents)."""
    path = loss_prepare_path(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ─── Device probing ─────────────────────────────────────────────────────────


def _pick_device_fresh(exclude: set[int] | None = None) -> str:
    """Re-probe the freest GPU; call this BEFORE each config run.

    A single upfront ``get_device('auto')`` would pin us to one GPU for
    the entire batch, which is bad when:

      * the previous config's ``del builder + empty_cache()`` hasn't
        yet reflected in CUDA's free-memory counters, or
      * another process is holding memory on what was initially the
        freest GPU, or
      * allocator fragmentation on one GPU pushes us to a fresher one.

    Args:
        exclude: Set of GPU indices to skip (e.g., GPUs that just OOM'd
            on this config). Pass None for no exclusions.

    Notes:
      * We intentionally do NOT call ``torch.cuda.synchronize()`` here.
        ``mem_get_info`` queries the driver directly, independent of
        PyTorch's allocator / pending ops; and ``synchronize`` would
        trigger CUDA-context initialization on the current device even
        when we haven't allocated anything yet.
      * ``empty_cache()`` is wrapped in try/except so a transient
        allocator hiccup never kills a whole batch run.
      * We only flush when PyTorch has actually allocated on a device
        already (``memory_allocated() > 0``); before the first config,
        there is nothing to flush.
    """
    if not torch.cuda.is_available():
        return "cpu"

    exclude = exclude or set()

    # Flush allocator cache so mem_get_info reflects actually-free VRAM.
    # The flush itself is best-effort: if the CUDA runtime is in a bad
    # state the run should still continue, so we swallow any exception
    # here and log a warning instead of propagating.
    try:
        if any(
            torch.cuda.memory_allocated(i) > 0 for i in range(torch.cuda.device_count())
        ):
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning("empty_cache() failed (non-fatal): %s", e)

    # Inline GPU probing with exclude-support (can't use get_device()
    # here because select_best_gpu has no exclude parameter).
    candidates: list[tuple[int, float]] = []
    for i in range(torch.cuda.device_count()):
        if i in exclude:
            continue
        try:
            free_mb = torch.cuda.mem_get_info(i)[0] / (1024**2)
            candidates.append((i, free_mb))
        except Exception:
            # Device might be unreachable or context-init failed.
            continue

    if not candidates:
        raise RuntimeError(
            f"All {torch.cuda.device_count()} GPU(s) exhausted or excluded: {exclude}"
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_idx, best_free = candidates[0]

    # Mimic select_best_gpu's informative log when running fresh
    if not exclude:
        logger.debug(
            "GPU re-probe: %s -> cuda:%d (%.1f MB free)",
            ",".join(f"cuda:{i}={m:.0f}MB" for i, m in candidates),
            best_idx,
            best_free,
        )
    else:
        logger.info(
            "GPU fallback (excluded=%s): cuda:%d (%.1f MB free)",
            sorted(exclude),
            best_idx,
            best_free,
        )

    return f"cuda:{best_idx}"


# ─── Batch sampling ──────────────────────────────────────────────────────────


def collect_batches(data_cfg: dict, batch_size: int, num_batches: int) -> list:
    """Load ``num_batches`` batches from ``data_cfg`` (called once per group)."""
    dataloader = NLCPV4DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=True,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    it = iter(dataloader)
    batches = []
    for _ in range(num_batches):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    return batches


# ─── Statistics ──────────────────────────────────────────────────────────────


def _stats(values: list) -> dict:
    """Return mean/std/min/max for a list of floats."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(values) == 1:
        v = float(values[0])
        return {"mean": round(v, 6), "std": 0.0, "min": round(v, 6), "max": round(v, 6)}
    return {
        "mean": round(float(statistics.mean(values)), 6),
        # ``statistics.stdev`` returns the sample standard deviation
        # (Bessel-corrected); this is the convention for per-batch
        # aggregates where the n batches are a sample of the full run.
        "std": round(float(statistics.stdev(values)), 6),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def _aggregate(per_batch: list) -> dict:
    """Compute per-component mean/std/min/max across per_batch records."""
    if not per_batch:
        return {"raw": {}, "weighted": {}, "total_weighted": _stats([])}
    raw_components = list(per_batch[0]["raw"].keys())
    weighted_components = list(per_batch[0]["weighted"].keys())
    raw_stats = {
        c: _stats([b["raw"][c] for b in per_batch if c in b["raw"]])
        for c in raw_components
    }
    weighted_stats = {
        c: _stats([b["weighted"][c] for b in per_batch if c in b["weighted"]])
        for c in weighted_components
    }
    total_stats = _stats([b["total_weighted"] for b in per_batch])
    return {
        "raw": raw_stats,
        "weighted": weighted_stats,
        "total_weighted": total_stats,
    }


# ─── Per-module runners ──────────────────────────────────────────────────────


def run_builder_n_batches(config: dict, device: str, batches: list) -> dict:
    """Build model and compute raw + weighted losses for each batch.

    Memory hygiene: with ``-n > 1`` this function is the main GPU-memory
    hot spot, so a few defensive tweaks are applied to prevent OOM on
    models that barely fit at ``n=1``:

    * ``requires_grad_(False)`` on every parameter — kills autograd
      bookkeeping independently of ``inference_mode``.
    * ``torch.inference_mode()`` instead of ``torch.no_grad()`` —
      stricter, no version counters, no grad tracking metadata.
    * Explicit ``del pyramid, total_loss, loss_dict`` at the end of
      each batch so the CUDA caching allocator can reuse those slots
      on the NEXT iteration instead of fragmenting.
    * ``torch.cuda.empty_cache()`` per batch — forces fragmented
      allocations back into the global pool between forward passes.
    """
    train_cfg = config["training"]
    loss_weights = train_cfg["loss_weights"]
    ordering_loss_type = train_cfg["ordering_loss_type"]
    batch_size = train_cfg["batch_size"]

    builder = ConceptPyramidBuilder(config).to(device)
    builder.eval()
    # Disable autograd bookkeeping on every parameter. Even under
    # no_grad/inference_mode, leaving requires_grad=True keeps some
    # lazily-allocated state; turning it off is cheap insurance.
    for p in builder.parameters():
        p.requires_grad_(False)

    weight_key_map = {
        "recon": "recon_loss_weight",
        "ordering": "ordering_loss_weight",
        "residual": "residual_loss_weight",
        "reasoning": "reasoning_loss_weight",
    }

    per_batch = []
    weights: dict = {}
    with torch.inference_mode():
        for batch in batches:
            pyramid = builder(batch)
            total_loss, loss_dict = compute_builder_loss(
                pyramid,
                loss_weights,
                ordering_loss_type=ordering_loss_type,
            )
            raw = {k: float(v) for k, v in loss_dict.items() if k != "total"}
            if not weights:
                weights = {
                    k: float(loss_weights[wk])
                    for k, wk in weight_key_map.items()
                    if k in raw and wk in loss_weights
                }
            weighted = {k: raw[k] * weights[k] for k in weights}
            per_batch.append(
                {
                    "raw": {k: round(v, 6) for k, v in raw.items()},
                    "weighted": {k: round(v, 6) for k, v in weighted.items()},
                    "total_weighted": round(float(loss_dict["total"]), 6),
                }
            )
            # Drop ALL GPU tensor refs before the next forward so the
            # CUDA allocator can reuse the same slots instead of
            # allocating fresh ones (which causes fragmentation).
            del pyramid, total_loss, loss_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = {
        "batch_size": int(batch_size),
        "num_batches": len(per_batch),
        "weights": weights,
        "per_batch": per_batch,
        "stats": _aggregate(per_batch),
    }

    # Release memory before next config.
    del builder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_predictor_n_batches(config: dict, device: str, batches: list) -> dict:
    """Placeholder: predictor training is not yet integrated."""
    raise NotImplementedError(
        "predictor loss_prepare is not implemented yet "
        "(no train_predictor.py / predictor loss in losses.py)."
    )


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    """CLI entry point: iterate all matching configs and prepare loss stats."""
    args = parse_args()
    module: str = args.module
    dataset: str = args.dataset
    num_batches: int = max(1, int(args.num_batches))
    force: bool = bool(args.force)

    configs_dir = PROJECT_ROOT / "configs" / "nlcpV4" / dataset
    if not configs_dir.is_dir():
        logger.error("Configs dir not found: %s", configs_dir)
        sys.exit(1)

    # Only process configs matching the requested module, identified by
    # the conventional `train_{module}_*.yml` filename prefix.
    prefix = f"train_{module}_"
    yml_files = sorted(configs_dir.glob(f"{prefix}*.yml"))
    if not yml_files:
        logger.error("No YAML configs matching '%s*.yml' under %s", prefix, configs_dir)
        sys.exit(1)

    # Load .env once (HF_TOKEN etc.)
    load_dotenv(PROJECT_ROOT / ".env")

    # ---- Pre-load all configs and group by batch_size ----
    # ``loaded`` collects successful (path, config) pairs for later grouping.
    loaded: list = []
    for p in yml_files:
        try:
            loaded.append((p, load_config(str(p))))
        except Exception as e:
            logger.error("Failed to load %s: %s", p.name, e)
    if not loaded:
        logger.error("No configs successfully loaded.")
        sys.exit(1)

    # Map from batch_size to the list of (path, config) sharing that size.
    groups: dict = {}
    for p, cfg in loaded:
        bs = int(cfg["training"]["batch_size"])
        groups.setdefault(bs, []).append((p, cfg))

    bs_list = sorted(groups.keys())
    if len(bs_list) > 1 and not force:
        logger.error(
            "Configs under '%s' have mixed batch sizes: %s. "
            "Pass -f/--force to group by batch_size and proceed.",
            configs_dir,
            bs_list,
        )
        sys.exit(1)

    # ---- Device + seeding (standard setup_environment path) ----
    # Initial probe is only for setup_environment (which wants *a*
    # device). The actual device used for each config is re-probed
    # just before that config runs (see _pick_device_fresh() below).
    first_seed = int(loaded[0][1]["environment"]["seed"])
    setup_environment({"seed": first_seed, "device": "auto"})
    initial_device = str(get_device("auto"))

    logger.info(
        "Initial device=%s | module=%s | dataset=%s | num_batches=%d | batch_size_groups=%s | %d config(s)",
        initial_device,
        module,
        dataset,
        num_batches,
        bs_list,
        len(loaded),
    )

    n_skip = 0
    n_run = 0
    n_fail = 0

    # ---- Process each batch_size group ----
    for bs in bs_list:
        group = groups[bs]
        logger.info("---- Group batch_size=%d | %d config(s) ----", bs, len(group))

        # Sample batches ONCE per group, using the first config's data_cfg.
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
            n_fail += len(group)
            continue

        if not shared_batches:
            logger.error(
                "No batches produced for group bs=%d; skipping its %d config(s).",
                bs,
                len(group),
            )
            n_fail += len(group)
            continue

        for config_path, config in group:
            key = f"{dataset}/{config_path.stem}"
            store = load_loss_prepare(module)

            if key in store:
                logger.info("[SKIP] %s (key already present)", key)
                n_skip += 1
                continue

            # Re-probe the freest GPU before every config run: previous
            # runs may not yet have released memory, or another process
            # may have grabbed it. This lets us hop to a fresher GPU
            # whenever one is available. If a GPU OOMs partway through,
            # we mark it as excluded and retry on the next-freest one.
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
                    "[RUN ] %s (device=%s, attempt=%d/%d, bs=%d, n=%d)",
                    key,
                    current_device,
                    attempt,
                    max_attempts,
                    bs,
                    len(shared_batches),
                )

                # Per-config seed — seed ONLY the chosen device, not all
                # devices. ``manual_seed_all`` touches every GPU which
                # triggers full-context init on each one and can queue
                # deferred OOM errors from tight GPUs. The chosen device
                # is enough because we only allocate tensors there.
                seed = int(config["environment"]["seed"])
                torch.manual_seed(seed)
                if torch.cuda.is_available() and current_device.startswith("cuda:"):
                    dev_idx = int(current_device.split(":")[1])
                    with torch.cuda.device(dev_idx):
                        torch.cuda.manual_seed(seed)

                try:
                    if module == "builder":
                        result = run_builder_n_batches(
                            config, current_device, shared_batches
                        )
                    else:
                        result = run_predictor_n_batches(
                            config, current_device, shared_batches
                        )
                    last_exc = None
                    # Successful run — stop retrying on other GPUs.
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
                    # Mark this device as exhausted for the remainder of
                    # this config's retries and flush its allocator.
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
                    # Non-OOM failure: no retry.
                    last_exc = e
                    break

            if result is None:
                logger.error(
                    "Failed on %s (tried %d GPU(s)): %s",
                    key,
                    len(excluded) or 1,
                    last_exc,
                    exc_info=last_exc is not None,
                )
                n_fail += 1
                continue

            # Re-read store right before write to preserve concurrent edits.
            store = load_loss_prepare(module)
            store[key] = result
            save_loss_prepare(module, store)

            total_stats = result["stats"]["total_weighted"]
            weighted_summary = ", ".join(
                f"{k}={v['mean']:.4f}\u00b1{v['std']:.4f}"
                for k, v in result["stats"]["weighted"].items()
            )
            logger.info(
                "[SAVE] %s | bs=%d | total=%.4f\u00b1%.4f (n=%d) | %s",
                key,
                bs,
                total_stats["mean"],
                total_stats["std"],
                result["num_batches"],
                weighted_summary,
            )
            n_run += 1

    logger.info(
        "Done. module=%s run=%d skip=%d fail=%d total=%d bs_groups=%s",
        module,
        n_run,
        n_skip,
        n_fail,
        len(loaded),
        bs_list,
    )


if __name__ == "__main__":
    main()
