#!/bin/bash
# =============================================================================
# SLURM Script: C3 Training (DeepSpeed ZeRO-2)
# =============================================================================
# Task: Train C3 model with DeepSpeed for multi-GPU distributed training
# Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
# Pipeline: texts -> Encoder -> [B, N, D] latent -> Decoder -> [B, L, V] -> loss
#
# Usage:
#   sbatch train_c3_ds.slurm
# =============================================================================

#SBATCH --job-name=c3_train_ds
#SBATCH --output=EXPERIMENT/c3_ds/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/c3_ds/logs/slurm_%j.err

# --- Resource Configuration ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

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
mkdir -p EXPERIMENT/c3_ds/logs
mkdir -p EXPERIMENT/c3_ds/checkpoints

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
echo "Starting C3 Training (DeepSpeed ZeRO-2)..."
echo "Config: configs/c3/config.yaml"
echo "DeepSpeed: configs/c3/zero2.json"
echo "GPUs: 4"
echo ""

PYTHONPATH=. torchrun --nproc_per_node=4 examples/c3/train_c3_ds.py \
    --config configs/c3/config.yaml \
    --deepspeed configs/c3/zero2.json

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
