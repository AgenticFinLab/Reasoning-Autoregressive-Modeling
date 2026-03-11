#!/bin/bash
# =============================================================================
# SLURM Script: Encoder-Decoder Training Test
# =============================================================================
# Task: Train encoder-decoder for text reconstruction
# Pipeline: texts -> Encoder -> [B, L, D] -> Decoder -> [B, L, V] -> loss
#
# Usage:
#   sbatch run_ed_train.slurm
# =============================================================================

#SBATCH --job-name=ed_train
#SBATCH --output=EXPERIMENT/uTEST-ed_train/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/uTEST-ed_train/logs/slurm_%j.err

# --- Resource Configuration ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

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
echo "Config: configs/uTEST/ed_train.yml"
echo ""

PYTHONPATH=. python examples/uTEST/test_ed_train.py \
    -c configs/uTEST/ed_train.yml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
