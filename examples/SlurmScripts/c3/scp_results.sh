#!/bin/bash
# =============================================================================
# SCP Script: Download C3 Experiment Results to Local Machine
# =============================================================================
# Usage:
#   ./scp_results.sh [experiment_name]
#
# Setup (add to ~/.bashrc or ~/.zshrc):
#   export RAM_SCP_SERVER="user@host"
#   export RAM_SCP_REMOTE_BASE="/path/to/project"
#
# Examples:
#   ./scp_results.sh c3          # Download c3 training results
#   ./scp_results.sh c3_ds       # Download c3_ds (DeepSpeed) results
#   ./scp_results.sh c3_recon    # Download reconstruction results
#   ./scp_results.sh all         # Download all c3 experiments
# =============================================================================

# Read from environment variables
SERVER="${RAM_SCP_SERVER:?Error: RAM_SCP_SERVER not set}"
REMOTE_BASE="${RAM_SCP_REMOTE_BASE:?Error: RAM_SCP_REMOTE_BASE not set}"
LOCAL_BASE="./EXPERIMENT"

EXPERIMENT=${1:-c3}

# =============================================================================
# Download Functions
# =============================================================================

download_logs() {
    local exp=$1
    echo "Downloading logs for $exp..."
    mkdir -p $LOCAL_BASE/$exp/logs
    scp -r $SERVER:$REMOTE_BASE/EXPERIMENT/$exp/logs/* $LOCAL_BASE/$exp/logs/
}

download_checkpoints() {
    local exp=$1
    echo "Downloading checkpoints for $exp..."
    mkdir -p $LOCAL_BASE/$exp/checkpoints
    scp -r $SERVER:$REMOTE_BASE/EXPERIMENT/$exp/checkpoints/* $LOCAL_BASE/$exp/checkpoints/
}

download_recon_results() {
    local exp=$1
    echo "Downloading reconstruction results for $exp..."
    mkdir -p $LOCAL_BASE/$exp/recon_results
    scp -r $SERVER:$REMOTE_BASE/EXPERIMENT/$exp/recon_results/* $LOCAL_BASE/$exp/recon_results/
}

download_all() {
    local exp=$1
    download_logs $exp
    download_checkpoints $exp
    # Only download recon_results if it exists
    ssh $SERVER "test -d $REMOTE_BASE/EXPERIMENT/$exp/recon_results" && download_recon_results $exp
}

# =============================================================================
# Main
# =============================================================================

case $EXPERIMENT in
    c3)
        download_all "c3"
        ;;
    c3_ds)
        download_all "c3_ds"
        ;;
    c3_recon)
        download_all "c3_recon"
        ;;
    all)
        download_all "c3"
        download_all "c3_ds"
        download_all "c3_recon"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [c3|c3_ds|c3_recon|all]"
        exit 1
        ;;
esac

echo ""
echo "Download complete!"
echo "Results saved to: $LOCAL_BASE/$EXPERIMENT"
