#!/bin/bash
# =============================================================================
# SLURM Script: Encoder-Decoder Training (Pipeline Parallel)
# =============================================================================
# Task: Train encoder-decoder model for text reconstruction (no quantizer)
# Pipeline: texts -> Encoder (BERT) -> [B, L, D] -> Decoder (GPT2) -> [B, L, V] -> loss
#
# Usage:
#   sbatch train_ed.slurm
# =============================================================================

#SBATCH --job-name=ed_train
#SBATCH --output=EXPERIMENT/ed/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/ed/logs/slurm_%j.err

# --- Resource Configuration ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
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
mkdir -p EXPERIMENT/ed/logs
mkdir -p EXPERIMENT/ed/checkpoints

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
echo "Starting Encoder-Decoder Training..."
echo "Config: configs/ed/config.yaml"
echo "Pipeline: Encoder (BERT) -> Decoder (GPT2)"
echo ""

PYTHONPATH=. python examples/ed/train_ed.py \
    -c configs/ed/config.yaml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
