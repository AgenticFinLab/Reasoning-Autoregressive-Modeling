#!/bin/bash
# =============================================================================
# SLURM Script: C3 Training (Pipeline Parallel)
# =============================================================================
# Task: Train C3 model for text compression/reconstruction
# Paper: "Context Cascade Compression: Exploring the Upper Limits of Text Compression"
# Pipeline: texts -> Encoder -> [B, N, D] latent -> Decoder -> [B, L, V] -> loss
#
# Usage:
#   sbatch train_c3.slurm
# =============================================================================

#SBATCH --job-name=c3_train
#SBATCH --output=EXPERIMENT/c3/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/c3/logs/slurm_%j.err

# --- Resource Configuration ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
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
mkdir -p EXPERIMENT/c3/logs
mkdir -p EXPERIMENT/c3/checkpoints

# Navigate to project root
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Activate conda environment (modify as needed)
# source ~/.bashrc
# conda activate LMSim

# =============================================================================
# Run Training
# =============================================================================
echo ""
echo "Starting C3 Training (Pipeline Parallel)..."
echo "Config: configs/c3/config.yaml"
echo "Pipeline: Encoder (Qwen2.5-0.5B) -> Latent [B,32,D] -> Decoder (Qwen2.5-1.5B)"
echo ""

PYTHONPATH=. python examples/c3/train_c3.py \
    -c configs/c3/config.yaml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
