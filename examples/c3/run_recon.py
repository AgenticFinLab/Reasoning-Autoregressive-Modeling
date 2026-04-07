"""C3 Reconstruction Visualization Script.

Usage:
    python examples/c3/run_recon.py \
        -m EXPERIMENT/c3/checkpoints/epoch_1 \
        -d gsm8k \
        -n 100 \
        -p EXPERIMENT/c3/recon_results

This script loads a trained C3 model and generates reconstruction samples
for visualization and analysis. It saves:
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmbase.dataset import registry
from lmbase.utils.tools import BlockBasedStoreManager
from model import C3Model
from ram import create_reconstruction_samples
from ram.utils import collate_fn_text, decode_logits_to_text, load_config


def load_trained_model(model_path: str, device: str = "cuda") -> C3Model:
    """Load a trained C3 model from checkpoint.

    Args:
        model_path: Path to the model checkpoint directory
        device: Device to load the model on

    Returns:
        Loaded C3Model instance
    """
    # Load model config from checkpoint directory
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        # Fallback: look for config in parent directory
        config_path = Path(model_path).parent.parent / "logs" / "train_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Extract model config (handle both direct and nested config)
    if "model" in config:
        model_cfg = config["model"]
    else:
        model_cfg = config

    # Build and load model
    model = C3Model(model_cfg)
    model = model.to(device)

    # Load model weights
    # DeepSpeed checkpoint structure: mp_rank_00_model_states.pt
    model_states_path = Path(model_path) / "mp_rank_00_model_states.pt"
    if model_states_path.exists():
        state_dict = torch.load(model_states_path, map_location=device)
        if "module" in state_dict:
            state_dict = state_dict["module"]
        model.load_state_dict(state_dict, strict=False)
    else:
        # Try standard PyTorch checkpoint
        ckpt_path = Path(model_path) / "pytorch_model.bin"
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, model_cfg


def run_reconstruction(
    model: C3Model,
    dataloader: DataLoader,
    num_samples: int,
    save_path: Path,
    device: str = "cuda",
) -> None:
    """Run reconstruction on dataset samples and save results.

    Args:
        model: Trained C3 model
        dataloader: DataLoader with text samples
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

    processed = 0
    sample_records = []

    print(f"Processing {num_samples} samples...")

    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="Reconstructing"):
            if processed >= num_samples:
                break

            # Move batch to device
            batch_size = len(batch_texts)

            # Forward pass through model
            logits, _ = model(
                context_texts=batch_texts,
                target_texts=batch_texts,
                compute_loss=False,
            )

            # Get attention mask for decoding
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=M,
            )
            attention_mask = encoded["attention_mask"].to(device)

            # Decode reconstruction results
            # logits: [B, N+L, V] -> skip N latent tokens -> [B, L, V]
            text_logits = logits[:, N:, :]
            decode_result = decode_logits_to_text(
                text_logits, tokenizer, batch_texts, attention_mask
            )

            # Create reconstruction samples
            recon_samples = create_reconstruction_samples(decode_result)

            # Store each sample in the batch
            for i in range(batch_size):
                if processed >= num_samples:
                    break

                sample_record = {
                    "sample_id": processed,
                    "original_text": batch_texts[i],
                    "reconstructed_text": (
                        recon_samples[i].reconstructed_text
                        if i < len(recon_samples)
                        else ""
                    ),
                    "target_text": (
                        recon_samples[i].target_text
                        if i < len(recon_samples)
                        else batch_texts[i]
                    ),
                    # Latent representation (encoder output)
                    "latent_tokens_shape": [N, model.encoder_hidden_dim],
                    # Store full reconstruction data
                    "full_data": {
                        "input_ids": encoded["input_ids"][i].cpu().tolist(),
                        "attention_mask": encoded["attention_mask"][i].cpu().tolist(),
                        "decoded_tokens": (
                            decode_result.decoded_tokens[i]
                            if i < len(decode_result.decoded_tokens)
                            else []
                        ),
                        "token_confidences": (
                            decode_result.token_confidences[i]
                            if i < len(decode_result.token_confidences)
                            else []
                        ),
                    },
                }

                sample_records.append(sample_record)
                processed += 1

    # Save all samples using BlockBasedStoreManager
    result_file = store_manager.save(sample_records, prefix="reconstruction")

    # Save metadata
    metadata = {
        "total_samples": processed,
        "model_config": {
            "latent_token_len": N,
            "max_length": M,
            "encoder_hidden_dim": model.encoder_hidden_dim,
            "decoder_hidden_dim": model.decoder_hidden_dim,
            "vocab_size": model.vocab_size,
        },
        "result_file": result_file,
    }

    metadata_path = save_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nReconstruction complete!")
    print(f"  Results saved to: {result_file}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"  Total samples processed: {processed}")


def main():
    """Main entry point for reconstruction script."""
    parser = argparse.ArgumentParser(
        description="C3 Reconstruction Visualization",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., gsm8k, math, mmmu)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to reconstruct (default: 100)",
    )
    parser.add_argument(
        "-p",
        "--save-path",
        type=str,
        required=True,
        help="Directory to save reconstruction results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset (e.g., 'algebra' for MATH dataset)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_path}")
    model, model_cfg = load_trained_model(args.model_path, args.device)
    print("Model loaded successfully!")

    # Setup dataset
    print(f"Loading dataset: {args.dataset}")
    data_cfg = {"data_name": args.dataset, "split": args.split}
    if args.subset is not None:
        data_cfg["subset"] = args.subset
    dataset = registry.get(data_cfg, split=args.split)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_text,
        drop_last=False,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Run reconstruction
    save_path = Path(args.save_path)
    run_reconstruction(
        model=model,
        dataloader=dataloader,
        num_samples=args.num_samples,
        save_path=save_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
