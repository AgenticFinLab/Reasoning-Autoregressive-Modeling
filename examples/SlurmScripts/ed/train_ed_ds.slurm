#!/bin/bash
# =============================================================================
# SLURM Script: Encoder-Decoder Training (DeepSpeed ZeRO-2)
# =============================================================================
# Task: Train encoder-decoder model with DeepSpeed for multi-GPU distributed training
# Pipeline: texts -> Encoder (BERT) -> [B, L, D] -> Decoder (GPT2) -> [B, L, V] -> loss
#
# Usage:
#   sbatch train_ed_ds.slurm
# =============================================================================

#SBATCH --job-name=ed_train_ds
#SBATCH --output=EXPERIMENT/ed_ds/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/ed_ds/logs/slurm_%j.err

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
mkdir -p EXPERIMENT/ed_ds/logs
mkdir -p EXPERIMENT/ed_ds/checkpoints

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
echo "Starting Encoder-Decoder Training (DeepSpeed)..."
echo "Config: configs/ed/config.yaml"
echo "GPUs: 4"
echo ""

PYTHONPATH=. torchrun --nproc_per_node=4 examples/ed/train_ed_ds.py \
    --config configs/ed/config.yaml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
