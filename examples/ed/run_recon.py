"""ED Reconstruction Evaluation Script.

Usage:
    python examples/ed/run_recon.py -c configs/ed/config.yaml
    python examples/ed/run_recon.py -c configs/ed/config.yaml -n 50
    python examples/ed/run_recon.py -c configs/ed/config.yaml --final-only

This script loads a trained ED model from checkpoints and generates
reconstruction samples for visualization and analysis. All settings (model,
dataset, paths) are read from the config file.

It saves:
    - Original input text
    - Encoder hidden representations
    - Decoder reconstruction output
    - Evaluation metrics (token_precision, char_precision, BLEU, etc.)
"""

import argparse
from pathlib import Path

import torch
from model import EDModel
from ram import RamDataLoaderRegistry
from ram.utils import (
    find_all_checkpoints,
    load_config,
    load_trained_model_for_recon,
    run_reconstruction_evaluation,
    save_reconstruction_results,
)


def run_reconstruction(
    model: EDModel,
    dataset,
    num_samples: int,
    save_path: Path,
    device: str,
    batch_size: int,
    block_size: int,
) -> None:
    """Run reconstruction on dataset samples and save results.

    Uses shared reconstruction utilities from ram.utils.reconstruction.

    Args:
        model: Trained ED model
        dataset: Dataset with RamSample objects
        num_samples: Number of samples to process
        save_path: Directory to save reconstruction results
        device: Device to run inference on
        batch_size: Number of samples to process in each batch
        block_size: Block size for storage manager
    """
    print(f"Processing {min(num_samples, len(dataset))} samples...")

    # Run reconstruction using shared utility
    all_results = run_reconstruction_evaluation(
        model=model,
        dataset=dataset,
        tokenizer=model.dec_tokenizer,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    # Use shared save utility
    model_info = {
        "encoder": model.encoder_model_name,
        "decoder": model.decoder_model_name,
        "hidden_dim": model.hidden_dim,
        "vocab_size": model.vocab_size,
    }
    metadata_path = save_reconstruction_results(
        results=all_results,
        save_path=save_path,
        block_size=block_size,
        model_info=model_info,
    )

    print(f"\nReconstruction complete!")
    print(f"  Results saved to: {save_path}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"  Total samples processed: {len(all_results)}")


def main():
    """Main entry point for reconstruction script."""
    parser = argparse.ArgumentParser(
        description="ED Reconstruction Evaluation",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/ed/config.yaml)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=True,
        help="Number of samples to reconstruct",
    )
    parser.add_argument(
        "-f",
        "--final-only",
        action="store_true",
        help="Only use checkpoint_final.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to run on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for reconstruction",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        required=True,
        help="Block size for storage manager",
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Extract all configs (same as train_ed.py)
    model_cfg = config["model"]
    data_cfg = config["data"]
    log_cfg = config["log"]

    # Determine paths from config
    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    base_save_path = output_dir / "recon_results"

    # Find all checkpoints
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    ckpt_files = find_all_checkpoints(checkpoint_dir, final_only=args.final_only)

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    print(f"Found {len(ckpt_files)} checkpoint(s):")
    for ckpt in ckpt_files:
        print(f"  - {ckpt.name}")

    # Setup dataset (from config)
    print(f"\nLoading dataset: {data_cfg['data_name']}")
    # Use RamDataLoaderRegistry to get the underlying dataset
    dataloader = RamDataLoaderRegistry(
        {
            "data_name": data_cfg["data_name"],
            "data_dir": data_cfg["data_dir"],
            "split": data_cfg["split"],
            "batch_size": data_cfg["batch_size"],
            "num_workers": data_cfg["num_workers"],
            "shuffle": data_cfg["shuffle"],
            "drop_last": data_cfg["drop_last"],
        }
    )
    # Get the wrapped dataset from dataloader
    dataset = dataloader.dataset
    print(f"Dataset loaded: {len(dataset)} samples")

    # Run reconstruction for each checkpoint
    for ckpt_path in ckpt_files:
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: {ckpt_path.name}")
        print(f"{'='*60}")

        # Create save path for this checkpoint
        save_path = base_save_path / ckpt_path.stem

        # Load model using shared utility
        model = load_trained_model_for_recon(
            checkpoint_path=ckpt_path,
            model_class=EDModel,
            model_cfg=model_cfg,
            device=args.device,
        )

        # Run reconstruction
        run_reconstruction(
            model=model,
            dataset=dataset,
            num_samples=args.num_samples,
            save_path=save_path,
            device=args.device,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )

        # Clean up to free memory
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"All checkpoints processed!")
    print(f"Results saved to: {base_save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
