#!/bin/bash
# =============================================================================
# SLURM Script: Token-Level Quantization Training (PreExp)
# =============================================================================
# Task: Train VQ-AE with standard token-level VQ-VAE (1:1 mapping)
# Pipeline: texts -> Encoder -> [B, L, D] -> TokenLevelQuantizer -> Decoder
#
# Key Feature:
#   - Each token gets its own discrete code (1:1 mapping)
#   - No multi-scale hierarchy
#   - Standard VQ-VAE for text
#
# Usage:
#   sbatch run_eqd_token_level.slurm
# =============================================================================

#SBATCH --job-name=PreExp-eqd_token_level
#SBATCH --output=EXPERIMENT/PreExp-eqd_token_level/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/PreExp-eqd_token_level/logs/slurm_%j.err

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
echo "Starting Token-Level Quantization Training..."
echo "Config: configs/PreExp/eqd_token_level.yml"
echo "Pipeline: Encoder -> TokenLevelQuantizer (1:1) -> Decoder"
echo ""

PYTHONPATH=. python examples/PreExp/eqd_token_level.py \
    -c configs/PreExp/eqd_token_level.yml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
