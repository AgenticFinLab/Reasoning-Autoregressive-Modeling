#!/bin/bash
# =============================================================================
# SLURM Script: EQD Scales Training (DeepSpeed ZeRO-2)
# =============================================================================
# Task: Train encoder-quantizer-decoder with DeepSpeed for multi-GPU distributed training
# Pipeline: texts -> Encoder -> [B, L, D] -> Quantizer (multi-scale) -> Decoder -> [B, L, V]
#
# Usage:
#   sbatch train_eqd_scales_ds.slurm
# =============================================================================

#SBATCH --job-name=eqd_scales_ds
#SBATCH --output=EXPERIMENT/eqd_scales_ds/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/eqd_scales_ds/logs/slurm_%j.err

# --- Resource Configuration ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create directories
mkdir -p EXPERIMENT/eqd_scales_ds/logs
mkdir -p EXPERIMENT/eqd_scales_ds/checkpoints

# Navigate to project root
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Activate conda environment (modify as needed)
# source ~/.bashrc
# conda activate LMSim

# =============================================================================
# Run Training (DeepSpeed)
# =============================================================================
echo ""
echo "Starting EQD Scales Training (DeepSpeed)..."
echo "Config: configs/eqd_scales/config.yaml"
echo "GPUs: 4"
echo ""

PYTHONPATH=. torchrun --nproc_per_node=4 examples/eqd_scales/train_eqd_scales_ds.py \
    --config configs/eqd_scales/config.yaml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
