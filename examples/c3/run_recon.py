"""C3 Reconstruction Visualization Script.

Usage:
    python examples/c3/run_recon.py -c configs/c3/config.yaml
    python examples/c3/run_recon.py -c configs/c3/config.yaml -n 50

This script loads a trained C3 model from the latest checkpoint and generates
reconstruction samples for visualization and analysis. All settings (model,
dataset, paths) are read from the config file.

It saves:
    - Original input text
    - Encoder latent representations
    - Decoder reconstruction output
    - Attention patterns (if available)
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from lmbase.dataset import registry
from lmbase.utils.tools import BlockBasedStoreManager
from model import C3Model
from ram import create_reconstruction_samples
from ram.evaluation import evaluate_reconstruction
from ram.utils import collate_fn_text, decode_logits_to_text, load_config


def load_trained_model(
    model_path: str, model_cfg: dict, device: str = "cuda"
) -> C3Model:
    """Load a trained C3 model from DeepSpeed ZeRO-2 checkpoint.

    DeepSpeed ZeRO-2 saves checkpoints as:
        - mp_rank_00_model_states.pt: Full model weights (use this for inference)
        - bf16_zero_pp_rank_*_optim_states.pt: Sharded optimizer states (training only)

    For inference/visualization, only model_states.pt is needed.

    Args:
        model_path: Path to the model checkpoint directory (e.g., global_step_870/)
        model_cfg: Model configuration dict
        device: Device to load the model on

    Returns:
        Loaded C3Model instance
    """
    model_path = Path(model_path)

    # Build model from config
    model = C3Model(model_cfg)
    model = model.to(device)

    # Load model weights from DeepSpeed checkpoint
    # For ZeRO-2: mp_rank_00_model_states.pt contains full model weights
    model_states_path = model_path / "mp_rank_00_model_states.pt"

    if not model_states_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_states_path}\n"
            f"Expected DeepSpeed ZeRO-2 checkpoint files:\n"
            f"  - mp_rank_00_model_states.pt (model weights)\n"
            f"  - bf16_zero_pp_rank_*_optim_states.pt (optimizer states, not needed for inference)"
        )

    print(f"Loading model weights from: {model_states_path}")
    state_dict = torch.load(model_states_path, map_location=device)

    # DeepSpeed wraps model in 'module' when saving
    if "module" in state_dict:
        state_dict = state_dict["module"]

    # Load weights (non-strict to handle any mismatches)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Latent tokens: {model.latent_token_len}")
    print(f"  Max length: {model.max_length}")

    return model


def run_reconstruction(
    model: C3Model,
    dataset,
    num_samples: int,
    save_path: Path,
    device: str = "cuda",
) -> None:
    """Run reconstruction on dataset samples and save results.

    Processes samples one by one for simplicity and clarity.

    Args:
        model: Trained C3 model
        dataset: Dataset with text samples
        num_samples: Number of samples to process
        save_path: Directory to save reconstruction results
        device: Device to run inference on
    """
    # Setup block-based storage for results
    save_path.mkdir(parents=True, exist_ok=True)
    store_manager = BlockBasedStoreManager(
        folder=str(save_path),
        block_size=50,
    )

    # Get model config for dimensions
    N = model.latent_token_len
    M = model.max_length
    tokenizer = model.tokenizer

    # Limit samples to dataset size
    num_samples = min(num_samples, len(dataset))

    print(f"Processing {num_samples} samples...")

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Reconstructing"):
            # Get single sample and extract text
            # Dataset returns dict, collate_fn_text extracts the text field
            raw_sample = dataset[idx]
            text = collate_fn_text([raw_sample])[0]

            # Forward pass through model
            # Model expects list of texts
            logits, _ = model(
                context_texts=[text],
                target_texts=[text],
                compute_loss=False,
            )

            # Get attention mask for decoding
            encoded = tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=M,
            )
            attention_mask = encoded["attention_mask"].to(device)

            # Decode reconstruction results
            # logits: [1, N+L, V] -> skip N latent tokens -> [1, L, V]
            text_logits = logits[:, N:, :]
            decode_result = decode_logits_to_text(
                text_logits, tokenizer, [text], attention_mask
            )

            # Create reconstruction samples
            recon_samples = create_reconstruction_samples(decode_result)

            # Get reconstructed text
            reconstructed_text = recon_samples[0].reconstructed if recon_samples else ""

            # Evaluate reconstruction quality
            metrics = evaluate_reconstruction(
                original_text=text,
                reconstructed_text=reconstructed_text,
                tokenizer=tokenizer,
            )

            # Build sample record and save immediately
            # No accumulation in memory - saves space
            sample_record = {
                "sample_id": idx,
                "original_text": text,
                "reconstructed_text": reconstructed_text,
                # In reconstruction tasks, target is the original input
                "target_text": text,
                # Evaluation metrics
                "metrics": metrics,
                # Latent representation info
                "latent_tokens_shape": [N, model.encoder_hidden_dim],
                # Store full reconstruction data
                "full_data": {
                    "input_ids": encoded["input_ids"][0].cpu().tolist(),
                    "attention_mask": encoded["attention_mask"][0].cpu().tolist(),
                    # Predicted token IDs from decode_result dict
                    "pred_ids": decode_result["pred_ids"][0],
                    "pred_text": decode_result["pred_texts"][0],
                },
            }

            # Save immediately to avoid memory accumulation
            store_manager.save(f"sample_{idx}", sample_record)

    # Save metadata
    metadata = {
        "total_samples": num_samples,
        "model_config": {
            "latent_token_len": N,
            "max_length": M,
            "encoder_hidden_dim": model.encoder_hidden_dim,
            "decoder_hidden_dim": model.decoder_hidden_dim,
            "vocab_size": model.vocab_size,
        },
        "save_path": str(save_path),
    }

    metadata_path = save_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nReconstruction complete!")
    print(f"  Results saved to: {save_path}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"  Total samples processed: {num_samples}")


def find_all_checkpoints(checkpoint_dir: Path, final_only: bool = False) -> list[Path]:
    """Find all checkpoint directories with model states.

    Args:
        checkpoint_dir: Base checkpoint directory
        final_only: If True, only return checkpoints/final/

    Returns:
        List of checkpoint directories containing mp_rank_00_model_states.pt
    """
    if not checkpoint_dir.exists():
        return []

    checkpoints = []

    if final_only:
        # Only use final checkpoint
        final_dir = checkpoint_dir / "final"
        if final_dir.exists() and (final_dir / "mp_rank_00_model_states.pt").exists():
            checkpoints.append(final_dir)
        return checkpoints

    # Find all subdirectories with model states
    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir():
            continue
        model_file = subdir / "mp_rank_00_model_states.pt"
        if model_file.exists():
            checkpoints.append(subdir)

    # Sort by name for consistent ordering
    return sorted(checkpoints)


def main():
    """Main entry point for reconstruction script."""
    parser = argparse.ArgumentParser(
        description="C3 Reconstruction Visualization",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/c3/config.yaml)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to reconstruct (default: 100)",
    )
    parser.add_argument(
        "-f",
        "--final-only",
        action="store_true",
        help="Only use checkpoints/final/ (default: use all checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Extract all configs (same as train_c3.py)
    model_cfg = config["model"]
    data_cfg = config["data"]
    log_cfg = config["log"]

    # Determine paths from config
    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    base_save_path = output_dir / "recon_results"

    # Find all checkpoints
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    ckpt_dirs = find_all_checkpoints(checkpoint_dir, final_only=args.final_only)

    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    print(f"Found {len(ckpt_dirs)} checkpoint(s):")
    for ckpt in ckpt_dirs:
        print(f"  - {ckpt.name}")

    # Setup dataset (from config)
    print(f"\nLoading dataset: {data_cfg['data_name']}")
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    print(f"Dataset loaded: {len(dataset)} samples")

    # Run reconstruction for each checkpoint
    for ckpt_path in ckpt_dirs:
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: {ckpt_path.name}")
        print(f"{'='*60}")

        # Create save path for this checkpoint
        save_path = base_save_path / ckpt_path.name

        # Load model
        model = load_trained_model(ckpt_path, model_cfg, args.device)

        # Run reconstruction
        run_reconstruction(
            model=model,
            dataset=dataset,
            num_samples=args.num_samples,
            save_path=save_path,
            device=args.device,
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
