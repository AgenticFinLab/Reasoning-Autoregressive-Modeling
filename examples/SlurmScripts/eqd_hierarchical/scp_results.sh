#!/bin/bash
# =============================================================================
# SCP Script: Download EQD Hierarchical Experiment Results to Local Machine
# =============================================================================
# Server: sjia@10.123.4.30
# Remote Path: /home/sjia/projects/Reasoning-Autoregressive-Modeling/
#
# Usage:
#   ./scp_results.sh [experiment_name]
#
# Examples:
#   ./scp_results.sh eqd_hierarchical        # Download training results
#   ./scp_results.sh eqd_hierarchical_ds     # Download DeepSpeed results
#   ./scp_results.sh all                     # Download all experiments
# =============================================================================

SERVER="sjia@10.123.4.30"
REMOTE_BASE="/home/sjia/projects/Reasoning-Autoregressive-Modeling"
LOCAL_BASE="./EXPERIMENT"

EXPERIMENT=${1:-eqd_hierarchical}

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
    eqd_hierarchical)
        download_all "eqd_hierarchical"
        ;;
    eqd_hierarchical_ds)
        download_all "eqd_hierarchical_ds"
        ;;
    all)
        download_all "eqd_hierarchical"
        download_all "eqd_hierarchical_ds"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [eqd_hierarchical|eqd_hierarchical_ds|all]"
        exit 1
        ;;
esac

echo ""
echo "Download complete!"
echo "Results saved to: $LOCAL_BASE/$EXPERIMENT"
