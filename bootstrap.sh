#!/bin/bash

# Echo-TTS Runpod Serverless Bootstrap Script
# This script sets up the environment for Echo-TTS serverless on Runpod

set -e  # Exit on any error

echo "=== Echo-TTS Runpod Bootstrap Starting ==="

# Configuration
# Code is shipped in the image at /opt/echo-tts-remote. Persistent data/caches live on /runpod-volume.
INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/echo-tts}"
SRC_DIR="${SRC_DIR:-/opt/echo-tts-remote}"
MODELS_DIR="${MODELS_DIR:-$INSTALL_DIR/models}"
AUDIO_VOICES_DIR="${AUDIO_VOICES_DIR:-$INSTALL_DIR/audio_voices}"
OUTPUT_AUDIO_DIR="${OUTPUT_AUDIO_DIR:-$INSTALL_DIR/output_audio}"

# Logging
LOG_FILE="$INSTALL_DIR/bootstrap.log"
mkdir -p "$INSTALL_DIR"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "Log file: $LOG_FILE"
echo "Install directory: $INSTALL_DIR"
echo "Source directory: $SRC_DIR"

# Function to print with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Ensure required directories exist
log "Ensuring required directories exist..."
mkdir -p "$AUDIO_VOICES_DIR" "$OUTPUT_AUDIO_DIR" "$MODELS_DIR"

# Make sure the shipped source is importable
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

# Start handler (runpod serverless mode)
log "Starting RunPod handler..."
log "Container ready for requests"
exec python "$SRC_DIR/handler.py"
