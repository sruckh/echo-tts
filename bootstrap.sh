#!/bin/bash

# Echo-TTS Runpod Serverless Bootstrap Script
# This script sets up the environment for Echo-TTS serverless on Runpod

set -e  # Exit on any error

echo "=== Echo-TTS Runpod Bootstrap Starting ==="

# Configuration
INSTALL_DIR="/runpod-volume/echo-tts"
VENV_DIR="$INSTALL_DIR/venv"
REMOTE_REPO_URL="https://github.com/jordandare/echo-tts.git"
REMOTE_DIR="$INSTALL_DIR/echo-tts-remote"
MODELS_DIR="$INSTALL_DIR/models"
FLAG_FILE="$INSTALL_DIR/.installed_flag"
MODEL_WEIGHTS_FILENAME="pytorch_model.safetensors"
PCA_STATE_FILENAME="pca_state.safetensors"
LOCAL_SRC_DIR="/opt/echo-tts"
SRC="$REMOTE_DIR"
AUDIO_VOICES_DIR="$INSTALL_DIR/audio_voices"
OUTPUT_AUDIO_DIR="$INSTALL_DIR/output_audio"

# Logging
LOG_FILE="$INSTALL_DIR/bootstrap.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "Log file: $LOG_FILE"
echo "Install directory: $INSTALL_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if already installed
if [ -f "$FLAG_FILE" ]; then
    log "Echo-TTS already installed. Skipping bootstrap."
    log "To force reinstall, delete: $FLAG_FILE"
    exit 0
fi

log "Starting fresh installation..."

# Create install directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone remote repository
log "Cloning Echo-TTS repository from $REMOTE_REPO_URL..."
git clone "$REMOTE_REPO_URL" "$REMOTE_DIR"
cp "$LOCAL_SRC_DIR/handler.py" "$REMOTE_DIR/handler.py"
mkdir -p "$AUDIO_VOICES_DIR"
mkdir -p "$OUTPUT_AUDIO_DIR"

# Remove gradio from requirements.txt
log "Removing gradio from requirements.txt..."
if [ -f "$REMOTE_DIR/requirements.txt" ]; then
    sed -i '/gradio/d' "$REMOTE_DIR/requirements.txt"
    log "Updated requirements.txt (removed gradio)"
else
    log "Warning: requirements.txt not found in remote repo"
fi

# Create virtual environment
log "Creating Python virtual environment..."
python3.12 -m venv "$VENV_DIR"

# Activate virtual environment
log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
log "Installing Python dependencies..."
if [ -f "$REMOTE_DIR/requirements.txt" ]; then
    # Add huggingface-hub to requirements
    echo "huggingface-hub" >> "$REMOTE_DIR/requirements.txt"

    # Install requirements
    pip install -r "$REMOTE_DIR/requirements.txt"
    log "Dependencies installed successfully"
else
    log "Installing minimal dependencies..."
    pip install torch torchaudio huggingface-hub safetensors numpy einops fastapi uvicorn
fi

# Install additional serverless dependencies
log "Installing serverless-specific dependencies..."
pip install runpod fastapi uvicorn[standard] pydantic python-multipart tqdm boto3

# Pre-download models using inference helpers (will also cache weights)
log "Downloading Echo-TTS models via inference helpers..."
mkdir -p "$MODELS_DIR"
export HF_HOME="$MODELS_DIR/hf-cache"
export HF_HUB_CACHE="$MODELS_DIR/hf-cache"
PYTHONPATH="$REMOTE_DIR:$PYTHONPATH" python3 - << 'PY'
import os
from inference import load_model_from_hf, load_fish_ae_from_hf, load_pca_state_from_hf

hf_token = os.environ.get("HF_TOKEN")

# Force CPU downloads to avoid GPU requirement during bootstrap
print("Downloading Echo-TTS base model...")
load_model_from_hf(device="cpu", dtype=None, compile=False, delete_blockwise_modules=False, token=hf_token)

print("Downloading Fish Speech S1-DAC autoencoder...")
load_fish_ae_from_hf(device="cpu", dtype=None, compile=False, token=hf_token)

print("Downloading PCA state...")
load_pca_state_from_hf(device="cpu", token=hf_token)

print("Model downloads completed")
PY

# Optional warmup (non-fatal if it fails)
log "Running optional handler warmup..."
python "$SRC/handler.py" --warmup || true

# Start handler (runpod serverless mode)
log "Starting RunPod handler..."
exec python "$SRC/handler.py" --rp_serve_api
