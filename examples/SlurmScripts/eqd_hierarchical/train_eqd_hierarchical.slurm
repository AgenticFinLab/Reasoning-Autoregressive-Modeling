#!/bin/bash
# =============================================================================
# SLURM Script: EQD Hierarchical Training (Pipeline Parallel)
# =============================================================================
# Task: Train encoder-quantizer-decoder with hierarchical multi-scale quantization
# Pipeline: texts -> Encoder -> [B, L, D] -> Quantizer (multi-scale) -> Decoder -> [B, L, V]
#
# Usage:
#   sbatch train_eqd_hierarchical.slurm
# =============================================================================

#SBATCH --job-name=eqd_hier
#SBATCH --output=EXPERIMENT/eqd_hierarchical/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/eqd_hierarchical/logs/slurm_%j.err

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
mkdir -p EXPERIMENT/eqd_hierarchical/logs
mkdir -p EXPERIMENT/eqd_hierarchical/checkpoints

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
echo "Starting EQD Hierarchical Training..."
echo "Config: configs/eqd_hierarchical/config.yaml"
echo "Quantizer: Hierarchical multi-scale [1, 4, 16, 64, 256]"
echo ""

PYTHONPATH=. python examples/eqd_hierarchical/train_eqd_hierarchical.py \
    -c configs/eqd_hierarchical/config.yaml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
