# coding=utf-8
# Echo-TTS RunPod Serverless Configuration
# SPDX-License-Identifier: MIT

"""
Configuration module for Echo-TTS RunPod Serverless.

Centralizes all environment variables, model paths, and configuration constants.
Following Fish Audio pattern for consistency across TTS services.
"""

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# =============================================================================
# HuggingFace Configuration
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

# =============================================================================
# S3 Configuration (required for production audio output storage)
# =============================================================================

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

# =============================================================================
# RunPod Volume Structure
# =============================================================================

RUNPOD_VOLUME = "/runpod-volume"
ECHO_TTS_DIR = f"{RUNPOD_VOLUME}/echo-tts"
AUDIO_VOICES_DIR = Path(os.environ.get("AUDIO_VOICES_DIR", f"{ECHO_TTS_DIR}/audio_voices"))
OUTPUT_AUDIO_DIR = Path(os.environ.get("OUTPUT_AUDIO_DIR", f"{ECHO_TTS_DIR}/output_audio"))

# =============================================================================
# Audio Configuration
# =============================================================================

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}

# Chunking defaults
DEFAULT_MAX_CHARS_PER_CHUNK = 350
DEFAULT_TARGET_DURATION_SECONDS = 150.0

# Crossfade settings for chunk transitions
DEFAULT_CROSSFADE_MS = 100
DEFAULT_SILENCE_THRESHOLD = 0.01
DEFAULT_MIN_SILENCE_SAMPLES = 2400  # ~50ms at 48kHz

# =============================================================================
# Model Configuration
# =============================================================================

# Device configuration
DEVICE = "cuda" if os.environ.get("DEVICE") != "cpu" else "cpu"

# =============================================================================
# LinaCodec Configuration
# =============================================================================

LINACODEC_AVAILABLE = False
try:
    from linacodec.codec import LinaCodec
    LINACODEC_AVAILABLE = True
    log.info("LinaCodec is available for streaming")
except ImportError:
    log.warning("LinaCodec not available. Streaming in linacodec_tokens format will fail.")

# =============================================================================
# Generation Parameter Defaults
# =============================================================================

DEFAULT_NUM_STEPS = 32
DEFAULT_CFG_SCALE_TEXT = 2.5
DEFAULT_CFG_SCALE_SPEAKER = 10.0
DEFAULT_CFG_MIN_T = 0.5
DEFAULT_CFG_MAX_T = 1.0
DEFAULT_SEQUENCE_LENGTH = 640

# =============================================================================
# Parameter Validation Ranges
# =============================================================================

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

MIN_CFG_SCALE = 0.0
MAX_CFG_SCALE = 20.0

MIN_NUM_STEPS = 1
MAX_NUM_STEPS = 100

MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 1000

# =============================================================================
# File Cleanup Configuration
# =============================================================================

CLEANUP_DAYS = 2  # Delete output files older than this many days

# =============================================================================
# Config Class (for runtime validation and logging)
# =============================================================================

class Config:
    """
    Configuration validation and storage.

    Validates required environment variables and creates necessary directories.
    """

    def __init__(self):
        self.validation_errors = []

        # Basic hardware detection
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            log.info(f"GPU detected: {self.gpu_name} with {self.gpu_memory:.1f}GB memory")

        # Required environment variables
        self.HF_TOKEN = HF_TOKEN
        if not self.HF_TOKEN:
            self.validation_errors.append("HF_TOKEN is required but not set")

        # S3 Configuration (required for production)
        self.S3_ENDPOINT_URL = S3_ENDPOINT_URL
        self.S3_ACCESS_KEY_ID = S3_ACCESS_KEY_ID
        self.S3_SECRET_ACCESS_KEY = S3_SECRET_ACCESS_KEY
        self.S3_BUCKET_NAME = S3_BUCKET_NAME
        self.S3_REGION = S3_REGION

        # Check if S3 is properly configured
        s3_missing = [
            var for var in ["S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
            if not getattr(self, var)
        ]
        if s3_missing:
            self.validation_errors.append(f"S3 configuration missing: {', '.join(s3_missing)}")

        # Directory configuration
        self.AUDIO_VOICES_DIR = AUDIO_VOICES_DIR
        self.OUTPUT_AUDIO_DIR = OUTPUT_AUDIO_DIR

        # Ensure directories exist
        try:
            self.AUDIO_VOICES_DIR.mkdir(parents=True, exist_ok=True)
            self.OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            log.info(f"Audio directories: {self.AUDIO_VOICES_DIR}, {self.OUTPUT_AUDIO_DIR}")
        except Exception as e:
            self.validation_errors.append(f"Failed to create directories: {e}")

        # Additional configuration
        self.AUDIO_EXTS = AUDIO_EXTS

        # Log all environment variables (without sensitive data)
        log.info(f"Device: {self.device}")
        log.debug(f"AUDIO_VOICES_DIR: {self.AUDIO_VOICES_DIR}")
        log.debug(f"OUTPUT_AUDIO_DIR: {'SET' if self.OUTPUT_AUDIO_DIR else 'NOT SET'}")
        log.debug(f"S3_ENDPOINT_URL: {'SET' if self.S3_ENDPOINT_URL else 'NOT SET'}")
        log.info(f"S3_BUCKET_NAME: {'SET' if self.S3_BUCKET_NAME else 'NOT SET'}")
        log.info(f"HF_TOKEN: {'SET' if self.HF_TOKEN else 'NOT SET'}")

        # Check audio files in voices directory
        try:
            audio_files = list(self.AUDIO_VOICES_DIR.glob("*"))
            audio_files = [f for f in audio_files if f.suffix.lower() in self.AUDIO_EXTS]
            log.debug(f"Found {len(audio_files)} audio files")
            for f in audio_files[:5]:  # Log first 5
                log.debug(f"  - {f.name}")
            if len(audio_files) > 5:
                log.debug(f"  ... and {len(audio_files) - 5} more")
        except Exception as e:
            log.warning(f"Could not scan audio directory: {e}")

    def validate(self) -> bool:
        """Return True if configuration is valid."""
        if self.validation_errors:
            log.error("Configuration validation failed:")
            for error in self.validation_errors:
                log.error(f"  - {error}")
            return False
        return True


# Global configuration instance (initialized at module load)
config = Config()
