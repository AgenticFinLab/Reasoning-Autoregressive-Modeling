"""Prepare loss values across all configs in a dataset for weight tuning.

Purpose:
    Iterate over every matching YAML config in configs/nlcpV4/{dataset}/
    and run a single-batch forward pass so researchers can inspect raw /
    weighted loss components and decide how to adjust loss_weights.

Behaviour:
    1. Select configs whose filename starts with `train_{module}_`, where
       module is supplied via `-m builder` or `-m predictor`.
    2. Check EXPERIMENT/nlcpV4/{module}/Loss_prepare.json for the config key
       "{dataset}/{config_stem}". If present -> [SKIP].
    3. Otherwise build the model, fetch ONE batch, run a forward pass with
       torch.no_grad(), record raw + weighted + total losses, persist to
       Loss_prepare.json, then free the model and continue.

Usage:
    python3 examples/RunResults/loss_prepare.py -m builder -d GSM8K
    python3 examples/RunResults/loss_prepare.py -m predictor -d GSM8K
"""

import argparse
import json
import logging
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


def run_builder_one_batch(config: dict, device: str) -> dict:
    """Build model, fetch one batch, compute raw + weighted losses."""
    # Lazy import of heavy deps so this script is cheap when all keys are cached.
    from nlcpV4.concept_builder import ConceptPyramidBuilder
    from nlcpV4.data_loader import NLCPV4DataLoader
    from nlcpV4.losses import compute_builder_loss

    data_cfg = config["data"]
    train_cfg = config["training"]
    loss_weights = train_cfg["loss_weights"]
    ordering_loss_type = train_cfg["ordering_loss_type"]
    batch_size = train_cfg["batch_size"]

    builder = ConceptPyramidBuilder(config).to(device)
    builder.eval()

    dataloader = NLCPV4DataLoader(
        data_cfg=data_cfg,
        batch_size=batch_size,
        include_solution=True,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    batch = next(iter(dataloader))

    with torch.no_grad():
        pyramid = builder(batch)
        _, loss_dict = compute_builder_loss(
            pyramid,
            loss_weights,
            ordering_loss_type=ordering_loss_type,
        )

    # Raw per-component losses (exclude the aggregate "total")
    raw = {k: float(v) for k, v in loss_dict.items() if k != "total"}

    # Map from loss component -> weight key in config
    weight_key_map = {
        "recon": "recon_loss_weight",
        "ordering": "ordering_loss_weight",
        "residual": "residual_loss_weight",
        "reasoning": "reasoning_loss_weight",
    }
    weights = {
        k: float(loss_weights[wk])
        for k, wk in weight_key_map.items()
        if k in raw and wk in loss_weights
    }
    weighted = {k: raw[k] * weights[k] for k in weights}

    result = {
        "batch_size": int(batch_size),
        "raw": {k: round(v, 6) for k, v in raw.items()},
        "weights": weights,
        "weighted": {k: round(v, 6) for k, v in weighted.items()},
        "total_weighted": round(float(loss_dict["total"]), 6),
    }

    # Release memory before next config.
    del builder, dataloader, batch, pyramid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_predictor_one_batch(config: dict, device: str) -> dict:
    """Placeholder: predictor training is not yet integrated."""
    raise NotImplementedError(
        "predictor loss_prepare is not implemented yet "
        "(no train_predictor.py / predictor loss in losses.py)."
    )


def main():
    args = parse_args()
    module: str = args.module
    dataset: str = args.dataset

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

    device = str(get_device("auto"))
    logger.info(
        "Device=%s | module=%s | dataset=%s | %d config(s) found",
        device,
        module,
        dataset,
        len(yml_files),
    )

    n_skip = 0
    n_run = 0
    n_fail = 0

    for config_path in yml_files:
        try:
            config = load_config(str(config_path))
        except Exception as e:
            logger.error("Failed to load %s: %s", config_path.name, e)
            n_fail += 1
            continue

        key = f"{dataset}/{config_path.stem}"
        store = load_loss_prepare(module)

        if key in store:
            logger.info("[SKIP] %s (key already present)", key)
            n_skip += 1
            continue

        logger.info("[RUN ] %s (module=%s)", key, module)
        setup_environment({"seed": config["environment"]["seed"], "device": "auto"})

        try:
            if module == "builder":
                result = run_builder_one_batch(config, device)
            else:
                result = run_predictor_one_batch(config, device)
        except Exception as e:
            logger.error("Failed on %s: %s", key, e, exc_info=True)
            n_fail += 1
            continue

        # Re-read store right before write to preserve concurrent edits.
        store = load_loss_prepare(module)
        store[key] = result
        save_loss_prepare(module, store)

        weighted_summary = ", ".join(
            f"{k}={v:.4f}" for k, v in result["weighted"].items()
        )
        logger.info(
            "[SAVE] %s | total=%.4f | %s",
            key,
            result["total_weighted"],
            weighted_summary,
        )
        n_run += 1

    logger.info(
        "Done. module=%s run=%d skip=%d fail=%d total=%d",
        module,
        n_run,
        n_skip,
        n_fail,
        len(yml_files),
    )


if __name__ == "__main__":
    main()
