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

# Ensure project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from lmbase.utils.env_tools import get_device
from ram.utils import load_config, setup_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("loss_prepare")

OUT_FILENAME = "Loss_prepare.json"
VALID_MODULES = ("builder", "predictor")


def parse_args():
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
    return PROJECT_ROOT / "EXPERIMENT" / "nlcpV4" / module / OUT_FILENAME


def load_loss_prepare(module: str) -> dict:
    path = loss_prepare_path(module)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_loss_prepare(module: str, data: dict) -> None:
    path = loss_prepare_path(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ─── Device probing ─────────────────────────────────────────────────────────


def _pick_device_fresh() -> str:
    """Re-probe the freest GPU; call this BEFORE each config run.

    A single upfront ``get_device('auto')`` would pin us to one GPU for
    the entire batch, which is bad when:

      * the previous config's ``del builder + empty_cache()`` hasn't
        yet reflected in CUDA's free-memory counters, or
      * another process is holding memory on what was initially the
        freest GPU, or
      * allocator fragmentation on one GPU pushes us to a fresher one.

    We flush PyTorch's caching allocator so freed blocks return to the
    driver, then let ``get_device('auto')`` (which uses
    ``torch.cuda.mem_get_info``) pick the GPU with the most free memory.

    Notes:
      * We intentionally do NOT call ``torch.cuda.synchronize()`` here.
        ``mem_get_info`` queries the driver directly, independent of
        PyTorch's allocator / pending ops; and ``synchronize`` would
        trigger CUDA-context initialization on the current device even
        when we haven't allocated anything yet (first-iteration crash).
      * ``empty_cache()`` is wrapped in try/except so a transient
        allocator hiccup never kills a whole batch run.
      * We only flush when PyTorch has actually allocated on a device
        already (``memory_allocated() > 0``); before the first config,
        there is nothing to flush.
    """
    if torch.cuda.is_available():
        try:
            # Only meaningful once we've allocated something; the check
            # also avoids touching un-initialized CUDA contexts.
            if any(
                torch.cuda.memory_allocated(i) > 0
                for i in range(torch.cuda.device_count())
            ):
                torch.cuda.empty_cache()
        except Exception as e:  # defensive: never let flush kill the run
            logger.warning("empty_cache() failed (non-fatal): %s", e)
    return str(get_device("auto"))


# ─── Batch sampling ──────────────────────────────────────────────────────────


def collect_batches(data_cfg: dict, batch_size: int, num_batches: int) -> list:
    """Load ``num_batches`` batches from ``data_cfg`` (called once per group)."""
    # Lazy import of heavy deps.
    from nlcpV4.data_loader import NLCPV4DataLoader

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
        "std": round(float(statistics.stdev(values)), 6),  # sample std
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
    # Lazy import of heavy deps.
    from nlcpV4.concept_builder import ConceptPyramidBuilder
    from nlcpV4.losses import compute_builder_loss

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
    loaded: list = []  # list[(path, config)]
    for p in yml_files:
        try:
            loaded.append((p, load_config(str(p))))
        except Exception as e:
            logger.error("Failed to load %s: %s", p.name, e)
    if not loaded:
        logger.error("No configs successfully loaded.")
        sys.exit(1)

    groups: dict = {}  # batch_size -> list[(path, config)]
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
            # whenever one is available.
            current_device = _pick_device_fresh()

            logger.info(
                "[RUN ] %s (device=%s, bs=%d, n=%d)",
                key,
                current_device,
                bs,
                len(shared_batches),
            )

            # Cheap per-config re-seed without re-probing the device.
            seed = int(config["environment"]["seed"])
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            try:
                if module == "builder":
                    result = run_builder_n_batches(
                        config, current_device, shared_batches
                    )
                else:
                    result = run_predictor_n_batches(
                        config, current_device, shared_batches
                    )
            except Exception as e:
                logger.error("Failed on %s: %s", key, e, exc_info=True)
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
