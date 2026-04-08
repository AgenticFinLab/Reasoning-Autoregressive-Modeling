#!/bin/bash
# =============================================================================
# SCP Script: Download Experiment Results to Local Machine
# =============================================================================
# Usage:
#   ./scp_results.sh -e <experiment> -t <type> [-p <local_base>]
#
# Options:
#   -e  Experiment name (required): c3, c3_ds, c3_recon, etc.
#   -t  Download type (required): logs, checkpoints, recon_results, all
#   -p  Local base folder (optional): default is EXPERIMENT
#
# Path structure:
#   Remote: $PROJECT/$EXPERIMENT_DIR/$EXPERIMENT/
#   Local:  $LOCAL_BASE/$EXPERIMENT/
#
# Setup (add to ~/.bashrc or ~/.zshrc):
#   export AgenticFin_Sjia_SERVER="user@host"
#   export AgenticFin_Sjia_BASE="/path/to/projects"
#
# Examples:
#   ./scp_results.sh -e c3 -t all                       # Download to ./EXPERIMENT/c3/
#   ./scp_results.sh -e c3_ds -t logs                   # Download logs to ./EXPERIMENT/c3_ds/
#   ./scp_results.sh -e c3_recon -t recon_results       # Download to ./EXPERIMENT/c3_recon/
#   ./scp_results.sh -e c3_ds -t all -p ./results       # Download to ./results/c3_ds/
# =============================================================================

# Read from environment variables
SERVER="${AgenticFin_Sjia_SERVER:?Error: AgenticFin_Sjia_SERVER not set}"
REMOTE_BASE="${AgenticFin_Sjia_BASE:?Error: AgenticFin_Sjia_BASE not set}"

# Project name (set per project)
PROJECT_NAME="Reasoning-Autoregressive-Modeling"
REMOTE_PROJECT="$REMOTE_BASE/$PROJECT_NAME"

# Experiment folder name (used for both remote and local)
EXPERIMENT_DIR="EXPERIMENT"

# =============================================================================
# Parse Arguments
# =============================================================================
EXPERIMENT=""
DOWNLOAD_TYPE=""
LOCAL_BASE=""

while getopts "e:t:p:h" opt; do
    case $opt in
        e) EXPERIMENT="$OPTARG" ;;
        t) DOWNLOAD_TYPE="$OPTARG" ;;
        p) LOCAL_BASE="$OPTARG" ;;
        h)
            echo "Usage: $0 -e <experiment> -t <type> [-p <local_base>]"
            echo ""
            echo "Options:"
            echo "  -e  Experiment name (required): c3, c3_ds, c3_recon, etc."
            echo "  -t  Download type (required): logs, checkpoints, recon_results, all"
            echo "  -p  Local base folder (optional): default is EXPERIMENT"
            echo ""
            echo "Path structure:"
            echo "  Remote: \$PROJECT/\$EXPERIMENT_DIR/\$EXPERIMENT/"
            echo "  Local:  \$LOCAL_BASE/\$EXPERIMENT/"
            echo ""
            echo "Examples:"
            echo "  $0 -e c3 -t all                      # Download to ./EXPERIMENT/c3/"
            echo "  $0 -e c3_ds -t logs                  # Download logs to ./EXPERIMENT/c3_ds/"
            echo "  $0 -e c3_recon -t recon_results      # Download to ./EXPERIMENT/c3_recon/"
            echo "  $0 -e c3_ds -t all -p ./results      # Download to ./results/c3_ds/"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$EXPERIMENT" ]; then
    echo "Error: Experiment name (-e) is required"
    exit 1
fi

if [ -z "$DOWNLOAD_TYPE" ]; then
    echo "Error: Download type (-t) is required"
    exit 1
fi

# Set default local base if not specified
if [ -z "$LOCAL_BASE" ]; then
    LOCAL_BASE="$EXPERIMENT_DIR"
fi

# Construct full local path: LOCAL_BASE/EXPERIMENT
LOCAL_PATH="$LOCAL_BASE/$EXPERIMENT"

# =============================================================================
# Download Functions
# =============================================================================

download_logs() {
    local exp=$1
    local dest=$2
    echo "Downloading logs for $exp..."
    mkdir -p "$dest/logs"
    scp -r $SERVER:$REMOTE_PROJECT/$EXPERIMENT_DIR/$exp/logs/* "$dest/logs/"
}

download_checkpoints() {
    local exp=$1
    local dest=$2
    echo "Downloading checkpoints for $exp..."
    mkdir -p "$dest/checkpoints"
    scp -r $SERVER:$REMOTE_PROJECT/$EXPERIMENT_DIR/$exp/checkpoints/* "$dest/checkpoints/"
}

download_recon_results() {
    local exp=$1
    local dest=$2
    echo "Downloading reconstruction results for $exp..."
    mkdir -p "$dest/recon_results"
    scp -r $SERVER:$REMOTE_PROJECT/$EXPERIMENT_DIR/$exp/recon_results/* "$dest/recon_results/"
}

download_all() {
    local exp=$1
    local dest=$2
    download_logs "$exp" "$dest"
    download_checkpoints "$exp" "$dest"
    # Only download recon_results if it exists on remote
    ssh $SERVER "test -d $REMOTE_PROJECT/$EXPERIMENT_DIR/$exp/recon_results" && download_recon_results "$exp" "$dest"
}

# =============================================================================
# Execute Download
# =============================================================================

case $DOWNLOAD_TYPE in
    logs)
        download_logs "$EXPERIMENT" "$LOCAL_PATH"
        ;;
    checkpoints)
        download_checkpoints "$EXPERIMENT" "$LOCAL_PATH"
        ;;
    recon_results)
        download_recon_results "$EXPERIMENT" "$LOCAL_PATH"
        ;;
    all)
        download_all "$EXPERIMENT" "$LOCAL_PATH"
        ;;
    *)
        echo "Error: Unknown download type: $DOWNLOAD_TYPE"
        echo "Valid types: logs, checkpoints, recon_results, all"
        exit 1
        ;;
esac

echo ""
echo "Download complete!"
echo "Results saved to: $LOCAL_PATH"
