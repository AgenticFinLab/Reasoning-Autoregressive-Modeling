#!/bin/bash
# =============================================================================
# SLURM Script: C3 Reconstruction Inference
# =============================================================================
# Task: Run reconstruction inference on trained C3 model
# Pipeline: texts -> Encoder -> [B, N, D] latent -> Decoder -> reconstructed texts
#
# Usage:
#   sbatch run_recon.slurm
# =============================================================================

#SBATCH --job-name=c3_recon
#SBATCH --output=EXPERIMENT/c3_recon/logs/slurm_%j.out
#SBATCH --error=EXPERIMENT/c3_recon/logs/slurm_%j.err

# --- Resource Configuration ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

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
mkdir -p EXPERIMENT/c3_recon/logs
mkdir -p EXPERIMENT/c3_recon/recon_results

# Navigate to project root
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Activate conda environment (modify as needed)
# source ~/.bashrc
# conda activate LMSim

# =============================================================================
# Run Reconstruction
# =============================================================================
echo ""
echo "Starting C3 Reconstruction Inference..."
echo "Config: configs/c3/config.yaml"
echo ""

# Modify checkpoint path as needed
CHECKPOINT_PATH="EXPERIMENT/c3_ds/checkpoints/checkpoint_final.pt"

PYTHONPATH=. python examples/c3/run_recon.py \
    --config configs/c3/config.yaml \
    --checkpoint_path $CHECKPOINT_PATH \
    --num_samples 100 \
    --output_dir EXPERIMENT/c3_recon/recon_results

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed."
echo "=========================================="
