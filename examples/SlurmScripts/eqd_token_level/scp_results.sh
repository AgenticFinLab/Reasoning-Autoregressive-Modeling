#!/bin/bash
# =============================================================================
# SCP Script: Download EQD Token Level Experiment Results to Local Machine
# =============================================================================
# Server: sjia@10.123.4.30
# Remote Path: /home/sjia/projects/Reasoning-Autoregressive-Modeling/
#
# Usage:
#   ./scp_results.sh [experiment_name]
#
# Examples:
#   ./scp_results.sh eqd_token_level        # Download training results
#   ./scp_results.sh eqd_token_level_ds     # Download DeepSpeed results
#   ./scp_results.sh all                    # Download all experiments
# =============================================================================

SERVER="sjia@10.123.4.30"
REMOTE_BASE="/home/sjia/projects/Reasoning-Autoregressive-Modeling"
LOCAL_BASE="./EXPERIMENT"

EXPERIMENT=${1:-eqd_token_level}

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
    eqd_token_level)
        download_all "eqd_token_level"
        ;;
    eqd_token_level_ds)
        download_all "eqd_token_level_ds"
        ;;
    all)
        download_all "eqd_token_level"
        download_all "eqd_token_level_ds"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [eqd_token_level|eqd_token_level_ds|all]"
        exit 1
        ;;
esac

echo ""
echo "Download complete!"
echo "Results saved to: $LOCAL_BASE/$EXPERIMENT"
