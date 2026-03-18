#!/bin/bash
# =============================================================================
# SLURM Script: Hierarchical Text Quantization Training (PreExp)
# =============================================================================
# Task: Train VQ-AE with linguistically-motivated hierarchical quantization
# Pipeline: texts -> Encoder -> [B, L, D] -> HierarchicalQuantizer -> Decoder
#
# Hierarchy Structure:
#   Scale 0: [1]   -> sentence embedding (topic)
#   Scale 1: [4]   -> clause embeddings (structure)
#   Scale 2: [16]  -> phrase embeddings (phrasing)
#   Scale 3: [64]  -> word embeddings (words)
#   Scale 4: [256] -> subword embeddings (tokens)
#   Total: 341 codes
#
# Usage:
#   sbatch run_eqd_hierarchical.slurm
# =============================================================================

#SBATCH --job-name=PreExp-eqd_hierarchical
#SBATCH --output=EXPERIMENT/PreExp-eqd_hierarchical/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/PreExp-eqd_hierarchical/logs/slurm_%j.err

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
echo "Starting Hierarchical Text Quantization Training..."
echo "Config: configs/PreExp/eqd_hierarchical.yml"
echo "Pipeline: Encoder -> HierarchicalQuantizer -> Decoder"
echo "Hierarchy: sentence -> clause -> phrase -> word -> subword"
echo ""

PYTHONPATH=. python examples/PreExp/eqd_hierarchical.py \
    -c configs/PreExp/eqd_hierarchical.yml

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
