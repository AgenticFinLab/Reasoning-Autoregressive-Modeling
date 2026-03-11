#!/bin/bash
# =============================================================================
# SLURM Script: Encoder-Quantizer-Decoder Training Test (VQ-AE)
# =============================================================================
# Task: Train VQ-AE for text reconstruction with multi-scale quantization
# Pipeline: texts -> Encoder -> [B, L, D] -> Quantizer -> [B, L, D] -> Decoder -> [B, L, V]
#
# Usage:
#   sbatch run_eqd_train.slurm
# =============================================================================

#SBATCH --job-name=eqd_train
#SBATCH --output=EXPERIMENT/uTEST-eqd_train/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/uTEST-eqd_train/logs/slurm_%j.err

# --- Resource Configuration ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
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
echo "Starting Encoder-Quantizer-Decoder (VQ-AE) Training..."
echo "Config: configs/uTEST/eqd_train.yml"
echo "Pipeline: Encoder -> Quantizer (7 scales) -> Decoder"
echo ""

PYTHONPATH=. python examples/uTEST/test_eqd_train.py \
    -c configs/uTEST/eqd_train.yml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
