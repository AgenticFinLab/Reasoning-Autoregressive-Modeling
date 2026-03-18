#!/bin/bash
# =============================================================================
# SLURM Script: Multi-Scale Quantization Training (PreExp)
# =============================================================================
# Task: Train VQ-AE with VAR-style multi-scale quantization
# Pipeline: texts -> Encoder -> [B, L, D] -> Quantizer (multi-scale) -> Decoder
#
# Scale Configuration (set in config):
#   - VAR Original: [1, 2, 4, 8, 16, 32] = 63 codes
#   - More Codes: [32, 64, 128, 256, 512] = 992 codes
#
# Usage:
#   sbatch run_eqd_scales.slurm
# =============================================================================

#SBATCH --job-name=PreExp-eqd_scales
#SBATCH --output=EXPERIMENT/PreExp-eqd_scales/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/PreExp-eqd_scales/logs/slurm_%j.err

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
echo "Starting Multi-Scale Quantization Training..."
echo "Config: configs/PreExp/eqd_scales.yml"
echo "Pipeline: Encoder -> Quantizer (multi-scale) -> Decoder"
echo ""

PYTHONPATH=. python examples/PreExp/eqd_scales.py \
    -c configs/PreExp/eqd_scales.yml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
