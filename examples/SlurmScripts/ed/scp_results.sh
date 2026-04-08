#!/bin/bash
# =============================================================================
# SCP Script: Download ED Experiment Results to Local Machine
# =============================================================================
# Server: sjia@10.123.4.30
# Remote Path: /home/sjia/projects/Reasoning-Autoregressive-Modeling/
#
# Usage:
#   ./scp_results.sh [experiment_name]
#
# Examples:
#   ./scp_results.sh ed        # Download ed training results
#   ./scp_results.sh ed_ds     # Download ed_ds (DeepSpeed) results
#   ./scp_results.sh all       # Download all ed experiments
# =============================================================================

SERVER="sjia@10.123.4.30"
REMOTE_BASE="/home/sjia/projects/Reasoning-Autoregressive-Modeling"
LOCAL_BASE="./EXPERIMENT"

EXPERIMENT=${1:-ed}

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

download_all() {
    local exp=$1
    download_logs $exp
    download_checkpoints $exp
}

# =============================================================================
# Main
# =============================================================================

case $EXPERIMENT in
    ed)
        download_all "ed"
        ;;
    ed_ds)
        download_all "ed_ds"
        ;;
    all)
        download_all "ed"
        download_all "ed_ds"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [ed|ed_ds|all]"
        exit 1
        ;;
esac

echo ""
echo "Download complete!"
echo "Results saved to: $LOCAL_BASE/$EXPERIMENT"
