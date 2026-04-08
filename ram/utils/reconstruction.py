"""Reconstruction utilities for RAM framework.

Shared functions for text reconstruction evaluation and result saving.
Used across all example methods (C3, ED, EQD, etc.) for consistent
reconstruction handling.
"""

from pathlib import Path
from typing import Any, Dict, List

import torch
from lmbase.utils.tools import BlockBasedStoreManager
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ram.evaluation import evaluate_reconstruction
from ram.generic import RamReconstructSample, RamSample


def run_reconstruction_evaluation(
    model: torch.nn.Module,
    samples: List[RamSample],
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    forward_fn=None,
) -> List[RamReconstructSample]:
    """Run reconstruction evaluation on dataset samples.

    Processes samples in batches and returns all reconstruction results.

    Args:
        model: The model to evaluate
        samples: List of RamSample objects to process
        tokenizer: Tokenizer for decoding
        batch_size: Number of samples per batch
        forward_fn: Optional callable to handle model forward pass.
                   If None, uses model(texts=target_texts, compute_loss=False).
                   Signature: forward_fn(model, target_texts) -> logits

    Returns:
        List of RamReconstructSample with all reconstruction results
    """
    num_samples = len(samples)
    all_results = []

    with torch.no_grad():
        for start_idx in tqdm(range(0, num_samples, batch_size), desc="Reconstructing"):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = samples[start_idx:end_idx]

            # Extract target texts for this batch
            target_texts = [s.target_text for s in batch_samples]

            # Forward pass
            if forward_fn is not None:
                logits = forward_fn(model, target_texts)
            else:
                output = model(texts=target_texts, compute_loss=False)
                # Handle both (logits, loss) and (logits, loss, vq_loss) returns
                logits = output[0] if isinstance(output, tuple) else output

            # Get predictions for all samples in batch
            pred_ids_tensor = torch.argmax(logits, dim=-1)
            pred_ids_list = pred_ids_tensor.cpu().tolist()

            # Build results for each sample in batch
            for i, sample in enumerate(batch_samples):
                pred_ids = pred_ids_list[i]
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                target_text = target_texts[i]

                # Compute metrics
                metrics = evaluate_reconstruction(
                    original_text=target_text,
                    reconstructed_text=pred_text,
                    tokenizer=tokenizer,
                )

                all_results.append(
                    RamReconstructSample(
                        sample_id=(
                            sample.sample_id
                            if sample.sample_id is not None
                            else start_idx + i
                        ),
                        original_text=target_text,
                        reconstructed_text=pred_text,
                        pred_ids=pred_ids,
                        metrics=metrics,
                    )
                )

    return all_results


def save_reconstruction_results(
    results: List[RamReconstructSample],
    save_path: Path,
    block_size: int,
    model_info: Dict[str, Any],
) -> Path:
    """Save reconstruction results to block-based storage.

    Args:
        results: List of RamReconstructSample from run_reconstruction_evaluation
        save_path: Directory to save results
        block_size: Block size for storage manager
        model_info: Model configuration info

    Returns:
        Path to saved metadata file
    """
    save_path.mkdir(parents=True, exist_ok=True)

    # Setup block-based storage
    store_manager = BlockBasedStoreManager(
        folder=str(save_path),
        block_size=block_size,
    )

    # Save each sample
    for result in results:
        sample_record = {
            "sample_id": result.sample_id,
            "original_text": result.original_text,
            "reconstructed_text": result.reconstructed_text,
            "metrics": result.metrics,
            "pred_ids": result.pred_ids,
        }
        store_manager.save(f"sample_{result.sample_id}", sample_record)

    # Save metadata
    metadata = {
        "total_samples": len(results),
        "model_info": model_info,
        "save_path": str(save_path),
    }

    metadata_path = save_path / "metadata.json"
    import json

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata_path


def load_trained_model_for_recon(
    checkpoint_path: Path,
    model_class: type,
    model_cfg: Dict[str, Any],
    device: str,
) -> torch.nn.Module:
    """Load a trained model from checkpoint for reconstruction.

    Supports both regular PyTorch checkpoints and DeepSpeed checkpoints.

    Args:
        checkpoint_path: Path to the checkpoint file or directory
        model_class: Model class to instantiate (e.g., EDModel, C3QwenModel)
        model_cfg: Model configuration dict
        device: Device to load the model on

    Returns:
        Loaded model instance
    """
    # Build model from config
    model = model_class(model_cfg)
    model = model.to(device)

    # Handle both file and directory paths
    if checkpoint_path.is_dir():
        # Try different checkpoint naming conventions
        possible_paths = [
            checkpoint_path / "checkpoint_final.pt",
            checkpoint_path / "mp_rank_00_model_states.pt",
        ]
        ckpt_file = None
        for path in possible_paths:
            if path.exists():
                ckpt_file = path
                break
        if ckpt_file is None:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_path}. "
                f"Tried: {[p.name for p in possible_paths]}"
            )
    else:
        ckpt_file = checkpoint_path

    # Load checkpoint
    state_dict = torch.load(ckpt_file, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "module" in state_dict:
        state_dict = state_dict["module"]

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def find_all_checkpoints(checkpoint_dir: Path, final_only: bool) -> List[Path]:
    """Find all checkpoint files in the checkpoint directory.

    Searches both the root checkpoint directory and all subdirectories
    for .pt checkpoint files.

    Args:
        checkpoint_dir: Base checkpoint directory
        final_only: If True, only return checkpoint_final.pt

    Returns:
        List of checkpoint paths
    """
    if not checkpoint_dir.exists():
        return []

    checkpoints = []

    if final_only:
        # Only use final checkpoint
        final_ckpt = checkpoint_dir / "final" / "mp_rank_00_model_states.pt"
        if final_ckpt.exists():
            checkpoints.append(final_ckpt)
        return checkpoints

    # Find all mp_rank_00_model_states.pt files in subdirectories
    # This is the DeepSpeed ZeRO-2 checkpoint format
    for ckpt_file in checkpoint_dir.rglob("mp_rank_00_model_states.pt"):
        checkpoints.append(ckpt_file)

    # Sort by path for consistent ordering
    return sorted(checkpoints)
