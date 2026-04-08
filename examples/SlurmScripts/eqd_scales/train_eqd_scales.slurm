#!/bin/bash
# =============================================================================
# SLURM Script: EQD Scales Training (Pipeline Parallel)
# =============================================================================
# Task: Train encoder-quantizer-decoder with multi-scale quantization
# Pipeline: texts -> Encoder -> [B, L, D] -> Quantizer (multi-scale) -> Decoder -> [B, L, V]
#
# Usage:
#   sbatch train_eqd_scales.slurm
# =============================================================================

#SBATCH --job-name=eqd_scales
#SBATCH --output=EXPERIMENT/eqd_scales/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/eqd_scales/logs/slurm_%j.err

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
mkdir -p EXPERIMENT/eqd_scales/logs
mkdir -p EXPERIMENT/eqd_scales/checkpoints

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
echo "Starting EQD Scales Training..."
echo "Config: configs/eqd_scales/config.yaml"
echo ""

PYTHONPATH=. python examples/eqd_scales/train_eqd_scales.py \
    -c configs/eqd_scales/config.yaml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
