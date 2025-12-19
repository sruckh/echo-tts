"""
RunPod serverless handler for Echo-TTS.

Accepts:
- text (str): text to synthesize. WhisperD-style tokens like [S1] are optional; normalization is handled upstream.
- speaker_voice (str | None): filename inside AUDIO_VOICES_DIR to use as reference audio. Supported suffixes:
  .wav, .mp3, .m4a, .ogg, .flac, .webm, .aac, .opus
- parameters (dict, optional): sampler options (num_steps, cfg_scale_text, cfg_scale_speaker,
  cfg_min_t, cfg_max_t, truncation_factor, rescale_k, rescale_sigma, speaker_kv_scale,
  speaker_kv_max_layers, speaker_kv_min_t, sequence_length, seed).
- session_id (str, optional): used to name the output file (otherwise a UUID is generated).

Output: uploads compressed audio (OGG) to S3-compatible storage and returns filename, presigned URL, and metadata.
"""

import argparse
import os
import re
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4
import tempfile

import runpod
import torch
import torchaudio
import boto3

from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    sample_pipeline,
    sample_euler_cfg_independent_guidances,
)

# Initialize RunPod structured logger
log = runpod.RunPodLogger()

_WHITESPACE_RE = re.compile(r"\s+")


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """
    Split input text into <= max_chars character chunks, preferring sentence/clause/word boundaries.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    normalized = _WHITESPACE_RE.sub(" ", (text or "")).strip()
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [normalized]

    sentence_enders = {".", "!", "?"}
    clause_enders = {",", ";", ":"}
    closers = {'"', "'", ")", "]", "}", "”", "’"}

    chunks: list[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        window = remaining[: max_chars + 1]
        candidate_sentence: int | None = None
        candidate_clause: int | None = None
        candidate_space: int | None = None

        for i in range(1, len(window)):
            if not window[i].isspace():
                continue

            candidate_space = i
            prev = window[i - 1]
            prev2 = window[i - 2] if i >= 2 else ""

            if prev in sentence_enders or (prev in closers and prev2 in sentence_enders):
                candidate_sentence = i
            elif prev in clause_enders or (prev in closers and prev2 in clause_enders):
                candidate_clause = i

        split_at = candidate_sentence or candidate_clause or candidate_space or max_chars
        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)

        remaining = remaining[split_at:].strip()

    return chunks


def chunk_text_for_audio(text: str, max_chars: int = 300, target_duration_seconds: float = 10.0) -> list[str]:
    """Chunk text considering both character limits and estimated audio duration.

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        target_duration_seconds: Target duration per chunk in seconds

    Returns:
        List of text chunks optimized for audio generation
    """
    # Rough estimate: ~12 characters per second of speech
    target_chars = min(max_chars, int(target_duration_seconds * 12))

    chunks = chunk_text(text, max_chars=target_chars)

    # Ensure last chunk isn't too short (minimum 2 seconds)
    if len(chunks) > 1 and len(chunks[-1]) < 24:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()

    return chunks


def crossfade_chunks(audio_chunks: list[torch.Tensor], overlap_samples: int = 4410) -> torch.Tensor:
    """Apply cross-fading between audio chunks for smoother transitions.

    Args:
        audio_chunks: List of audio tensors to concatenate
        overlap_samples: Number of samples to overlap (default: 100ms at 44.1kHz)

    Returns:
        Crossfaded audio tensor
    """

    if len(audio_chunks) <= 1:
        return torch.cat(audio_chunks, dim=-1) if audio_chunks else torch.tensor([])

    result = audio_chunks[0]

    for i in range(1, len(audio_chunks)):
        # Ensure minimum length for crossfade
        chunk_length = audio_chunks[i].shape[-1]
        prev_length = result.shape[-1]

        # Adjust overlap if chunks are too short
        actual_overlap = min(overlap_samples, chunk_length // 4, prev_length // 4)

        if actual_overlap > 0:
            # Create fade windows
            fade_out = torch.linspace(1, 0, actual_overlap, device=audio_chunks[i].device)
            fade_in = torch.linspace(0, 1, actual_overlap, device=audio_chunks[i].device)

            # Match dimensions
            fade_out = fade_out.view(1, -1) if audio_chunks[i].dim() == 2 else fade_out
            fade_in = fade_in.view(1, -1) if audio_chunks[i].dim() == 2 else fade_in

            # Apply crossfade
            # Get overlap from result BEFORE trimming
            prev_chunk_end = result[..., -actual_overlap:] * fade_out
            result = result[..., :-actual_overlap]
            curr_chunk_start = audio_chunks[i][..., :actual_overlap] * fade_in
            crossfaded = prev_chunk_end + curr_chunk_start

            result = torch.cat([result, crossfaded, audio_chunks[i][..., actual_overlap:]], dim=-1)
        else:
            # No overlap, just concatenate
            result = torch.cat([result, audio_chunks[i]], dim=-1)

    return result


def normalize_chunk_boundaries(audio_chunks: list[torch.Tensor],
                             sample_rate: int = 44100,
                             silence_threshold: float = 0.01,
                             min_silence_samples: int = 22050) -> torch.Tensor:
    """Normalize silences at chunk boundaries to reduce artifacts.

    Args:
        audio_chunks: List of audio tensors
        sample_rate: Audio sample rate
        silence_threshold: Energy threshold for silence detection
        min_silence_samples: Minimum silence between chunks (0.5 seconds)

    Returns:
        Normalized audio tensor
    """

    if not audio_chunks:
        return torch.tensor([])

    if len(audio_chunks) == 1:
        return audio_chunks[0]

    normalized_chunks = []

    for i, chunk in enumerate(audio_chunks):
        # Ensure chunk is 2D for consistency
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)

        # Find trailing silence
        if i < len(audio_chunks) - 1:  # Not the last chunk
            # Calculate energy in the last part
            tail_samples = min(chunk.shape[-1], min_silence_samples * 2)
            tail_energy = torch.abs(chunk[..., -tail_samples:])

            # Find actual end of speech
            tail_energy_flat = tail_energy.flatten()
            trailing_silence = 0

            # Check from the end for silence
            for j in range(len(tail_energy_flat) - 1, -1, -1):
                if tail_energy_flat[j] < silence_threshold:
                    trailing_silence += 1
                else:
                    break

            # Normalize trailing silence
            if trailing_silence > min_silence_samples:
                # Remove excess silence
                chunk = chunk[..., :-(trailing_silence - min_silence_samples)]
            elif trailing_silence < min_silence_samples and trailing_silence > 0:
                # Extend to minimum silence
                additional_silence = min_silence_samples - trailing_silence
                # Create silence with same dimensionality as chunk
                silence = torch.zeros(*chunk.shape[:-1], additional_silence, device=chunk.device)
                chunk = torch.cat([chunk, silence], dim=-1)
            elif trailing_silence == 0:
                # Add minimum silence if none found
                # Create silence with same dimensionality as chunk
                silence = torch.zeros(*chunk.shape[:-1], min_silence_samples, device=chunk.device)
                chunk = torch.cat([chunk, silence], dim=-1)

        normalized_chunks.append(chunk)

    # Crossfade for smoother transitions
    result = crossfade_chunks(normalized_chunks)
    return result


# Environment Configuration and Validation
class Config:
    """Configuration validation and storage"""
    def __init__(self):
        self.validation_errors = []

        # Basic hardware detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            log.info(f"GPU detected: {self.gpu_name} with {self.gpu_memory:.1f}GB memory")

        # Required environment variables
        self.HF_TOKEN = os.environ.get("HF_TOKEN")
        if not self.HF_TOKEN:
            self.validation_errors.append("HF_TOKEN is required but not set")

        # S3 Configuration (required for production)
        self.S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
        self.S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
        self.S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
        self.S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
        self.S3_REGION = os.environ.get("S3_REGION", "us-east-1")

        # Check if S3 is properly configured
        s3_missing = [var for var in ["S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
                      if not getattr(self, var)]
        if s3_missing:
            self.validation_errors.append(f"S3 configuration missing: {', '.join(s3_missing)}")

        # Directory configuration
        self.AUDIO_VOICES_DIR = Path(os.environ.get("AUDIO_VOICES_DIR", "/runpod-volume/echo-tts/audio_voices"))
        self.OUTPUT_AUDIO_DIR = Path(os.environ.get("OUTPUT_AUDIO_DIR", "/runpod-volume/echo-tts/output_audio"))

        # Ensure directories exist
        try:
            self.AUDIO_VOICES_DIR.mkdir(parents=True, exist_ok=True)
            self.OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            log.info(f"Audio directories: {self.AUDIO_VOICES_DIR}, {self.OUTPUT_AUDIO_DIR}")
        except Exception as e:
            self.validation_errors.append(f"Failed to create directories: {e}")

        # Additional configuration
        self.AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}

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
        """Return True if configuration is valid"""
        if self.validation_errors:
            log.error("Configuration validation failed:")
            for error in self.validation_errors:
                log.error(f"  - {error}")
            return False
        return True

# Global configuration instance
config = Config()
_MODELS: Dict[str, object] = {}


def _load_models(device: str = None, request_id: Optional[str] = None) -> Tuple[object, object, object]:
    """Lazy-load and cache model, autoencoder, and PCA state."""
    # Use config device if not specified
    if device is None:
        device = config.device

    log.debug(f"_load_models called. Current cache: {list(_MODELS.keys())}", request_id=request_id)

    if _MODELS:
        log.info("Models already cached, reusing", request_id=request_id)
        return _MODELS["model"], _MODELS["fish_ae"], _MODELS["pca_state"]

    # Validate HF token before attempting to load models
    if not config.HF_TOKEN:
        error_msg = "HF_TOKEN is required to load models from HuggingFace"
        log.error(error_msg, request_id=request_id)
        raise RuntimeError(error_msg)

    log.info(f"Starting model loading on device: {device}", request_id=request_id)
    start_time = time.time()

    try:
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else None
        log.debug(f"Using dtype: {torch_dtype}", request_id=request_id)

        # Check available memory
        if device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            log.debug(f"GPU memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total", request_id=request_id)

        # Load main model
        log.info("Loading EchoDiT model from HuggingFace...", request_id=request_id)
        model_start = time.time()

        try:
            model = load_model_from_hf(
                device=device,
                dtype=torch_dtype,
                compile=False,
                delete_blockwise_modules=False,
                token=config.HF_TOKEN
            )
            model_time = time.time() - model_start
            log.info(f"EchoDiT model loaded in {model_time:.2f}s", request_id=request_id)
        except Exception as model_error:
            error_msg = f"Failed to load EchoDiT model: {str(model_error)}"
            log.error(error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Load autoencoder
        log.info("Loading Fish Speech S1-DAC autoencoder from HuggingFace...", request_id=request_id)
        ae_start = time.time()

        try:
            fish_ae = load_fish_ae_from_hf(
                device=device,
                dtype=torch_dtype,
                compile=False,
                token=config.HF_TOKEN
            )
            ae_time = time.time() - ae_start
            log.info(f"Autoencoder loaded in {ae_time:.2f}s", request_id=request_id)
        except Exception as ae_error:
            error_msg = f"Failed to load autoencoder: {str(ae_error)}"
            log.error(error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Load PCA state
        log.info("Loading PCA state from HuggingFace...", request_id=request_id)
        pca_start = time.time()

        try:
            pca_state = load_pca_state_from_hf(device=device, token=config.HF_TOKEN)
            pca_time = time.time() - pca_start
            log.info(f"PCA state loaded in {pca_time:.2f}s", request_id=request_id)
        except Exception as pca_error:
            error_msg = f"Failed to load PCA state: {str(pca_error)}"
            log.error(error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Cache models
        _MODELS.update({"model": model, "fish_ae": fish_ae, "pca_state": pca_state})

        total_time = time.time() - start_time
        log.info(f"All models loaded successfully in {total_time:.2f}s", request_id=request_id)

        # Log memory usage after loading
        if device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            log.debug(f"GPU memory after loading: {allocated:.1f}GB allocated, {cached:.1f}GB cached", request_id=request_id)

        return model, fish_ae, pca_state

    except Exception as e:
        error_trace = traceback.format_exc()
        log.error(f"Model loading failed: {str(e)}", request_id=request_id)
        log.error(f"Traceback:\n{error_trace}", request_id=request_id)
        raise


def _build_sample_fn(params: Dict, request_id: Optional[str] = None) -> callable:
    """Create sampler partial with defaults and overrides."""
    log.debug(f"Building sampler with params: {params}", request_id=request_id)
    return partial(
        sample_euler_cfg_independent_guidances,
        num_steps=params.get("num_steps", 40),
        cfg_scale_text=params.get("cfg_scale_text", 3.0),
        cfg_scale_speaker=params.get("cfg_scale_speaker", 8.0),
        cfg_min_t=params.get("cfg_min_t", 0.5),
        cfg_max_t=params.get("cfg_max_t", 1.0),
        truncation_factor=params.get("truncation_factor"),
        rescale_k=params.get("rescale_k"),
        rescale_sigma=params.get("rescale_sigma"),
        speaker_kv_scale=params.get("speaker_kv_scale"),
        speaker_kv_max_layers=params.get("speaker_kv_max_layers"),
        speaker_kv_min_t=params.get("speaker_kv_min_t"),
        sequence_length=params.get("sequence_length", 640),
    )


def _get_s3_client():
    """Create and return S3 client with enhanced error handling"""
    log.debug("Creating S3 client...")

    # Check S3 configuration
    if not all([config.S3_ENDPOINT_URL, config.S3_ACCESS_KEY_ID, config.S3_SECRET_ACCESS_KEY, config.S3_BUCKET_NAME]):
        missing = []
        if not config.S3_ENDPOINT_URL:
            missing.append("S3_ENDPOINT_URL")
        if not config.S3_ACCESS_KEY_ID:
            missing.append("S3_ACCESS_KEY_ID")
        if not config.S3_SECRET_ACCESS_KEY:
            missing.append("S3_SECRET_ACCESS_KEY")
        if not config.S3_BUCKET_NAME:
            missing.append("S3_BUCKET_NAME")

        error_msg = f"Missing S3 configuration: {', '.join(missing)}"
        log.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        client = boto3.client(
            "s3",
            endpoint_url=config.S3_ENDPOINT_URL,
            region_name=config.S3_REGION,
            aws_access_key_id=config.S3_ACCESS_KEY_ID,
            aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        )
        log.info(f"S3 client created for endpoint: {config.S3_ENDPOINT_URL}")
        return client
    except Exception as e:
        error_msg = f"Failed to create S3 client: {str(e)}"
        log.error(error_msg)
        raise RuntimeError(error_msg)


def _save_and_upload_audio(audio_tensor: torch.Tensor, sample_rate: int, session_id: str, request_id: Optional[str] = None) -> Dict[str, str]:
    """Save audio to OGG (Vorbis) and upload to S3-compatible storage."""
    log.info(f"Saving and uploading audio for session: {session_id}", request_id=request_id)

    # Ensure output directory exists
    config.OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{session_id}.ogg"
    key = f"{filename}"

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        log.debug(f"Saving audio to temporary file: {tmp_path}", request_id=request_id)

        # Validate audio tensor
        if audio_tensor is None:
            raise RuntimeError("Audio tensor is None")
        if len(audio_tensor.shape) < 2:
            raise RuntimeError(f"Invalid audio tensor shape: {audio_tensor.shape}")

        log.debug(f"Audio tensor shape: {audio_tensor.shape}, sample_rate: {sample_rate}", request_id=request_id)

        try:
            torchaudio.save(tmp_path, audio_tensor, sample_rate)
            log.debug(f"Audio saved successfully to: {tmp_path}", request_id=request_id)
        except Exception as save_error:
            error_msg = f"Failed to save audio file: {str(save_error)}"
            log.error(error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Read the saved file
        try:
            with open(tmp_path, "rb") as f:
                data = f.read()
        except Exception as read_error:
            error_msg = f"Failed to read saved audio file: {str(read_error)}"
            log.error(error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        file_size_mb = len(data) / (1024 * 1024)
        log.info(f"Audio file size: {file_size_mb:.2f}MB", request_id=request_id)

        if file_size_mb > 50:  # Warn if file is very large
            log.warning(f"Large audio file: {file_size_mb:.2f}MB", request_id=request_id)

        # Upload to S3
        log.debug("Uploading to S3...", request_id=request_id)
        try:
            s3 = _get_s3_client()
            s3.put_object(
                Bucket=config.S3_BUCKET_NAME,
                Key=key,
                Body=data,
                ContentType="audio/ogg",
            )
            log.debug(f"Successfully uploaded to S3: {key}", request_id=request_id)
        except Exception as upload_error:
            error_msg = f"Failed to upload to S3: {str(upload_error)}"
            log.error(error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Generate presigned URL
        try:
            presigned_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": config.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=3600,
            )
            log.debug("Presigned URL generated successfully", request_id=request_id)
        except Exception as url_error:
            error_msg = f"Failed to generate presigned URL: {str(url_error)}"
            log.error(error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        log.info(f"Audio uploaded successfully: {key}", request_id=request_id)
        return {"filename": filename, "url": presigned_url, "key": key}

    except Exception as e:
        error_msg = f"Failed to save/upload audio: {str(e)}"
        log.error(error_msg, request_id=request_id)
        raise

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
            log.debug(f"Cleaned up temporary file: {tmp_path}", request_id=request_id)
        except OSError as cleanup_error:
            log.warning(f"Failed to clean up temporary file {tmp_path}: {cleanup_error}", request_id=request_id)


def health_check(request_id: Optional[str] = None) -> Dict:
    """Comprehensive health check for the TTS service"""
    log.info("Performing health check...", request_id=request_id)

    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }

    # Check basic configuration
    config_valid = config.validate()
    health_status["checks"]["configuration"] = {
        "status": "pass" if config_valid else "fail",
        "details": f"Validation errors: {len(config.validation_errors)}" if not config_valid else "All good"
    }

    # Check models
    models_loaded = bool(_MODELS)
    health_status["checks"]["models"] = {
        "status": "pass" if models_loaded else "fail",
        "details": f"Loaded models: {list(_MODELS.keys())}"
    }

    # Check GPU/CUDA
    gpu_available = torch.cuda.is_available()
    health_status["checks"]["hardware"] = {
        "status": "pass" if gpu_available else "warn",
        "details": f"CUDA available: {gpu_available}, Device: {config.device}"
    }
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        health_status["checks"]["hardware"]["details"] += f", GPU: {gpu_name}, Memory: {gpu_memory_allocated:.1f}/{gpu_memory_total:.1f}GB"

    # Check S3 configuration
    s3_configured = all([config.S3_ENDPOINT_URL, config.S3_ACCESS_KEY_ID, config.S3_SECRET_ACCESS_KEY, config.S3_BUCKET_NAME])
    health_status["checks"]["s3"] = {
        "status": "pass" if s3_configured else "fail",
        "details": f"S3 configured: {s3_configured}"
    }

    # Check directories
    audio_dir_exists = config.AUDIO_VOICES_DIR.exists()
    output_dir_exists = config.OUTPUT_AUDIO_DIR.exists()
    health_status["checks"]["directories"] = {
        "status": "pass" if (audio_dir_exists and output_dir_exists) else "fail",
        "details": f"Audio dir: {audio_dir_exists}, Output dir: {output_dir_exists}"
    }

    # Check audio files
    try:
        audio_files = list(config.AUDIO_VOICES_DIR.glob("*"))
        audio_files = [f for f in audio_files if f.suffix.lower() in config.AUDIO_EXTS]
        health_status["checks"]["audio_files"] = {
            "status": "pass" if audio_files else "warn",
            "details": f"Found {len(audio_files)} audio files"
        }
    except Exception as e:
        health_status["checks"]["audio_files"] = {
            "status": "fail",
            "details": f"Error checking audio files: {e}"
        }

    # Overall status
    all_pass = all(check["status"] == "pass" for check in health_status["checks"].values())
    health_status["status"] = "healthy" if all_pass else "unhealthy"

    log.info(f"Health check completed: {health_status['status']}", request_id=request_id)
    return health_status


def _synthesize(job_input: Dict, job_id: Optional[str] = None) -> Dict:
    """Simplified synthesis function following Lotus pattern."""

    # Handle health check requests
    if job_input.get("action") == "health_check":
        return health_check()

    # Validate text input
    text = job_input.get("text")
    if not text or not isinstance(text, str):
        return {"error": "Missing or invalid 'text' field (expected string)"}

    if len(text.strip()) == 0:
        return {"error": "Text cannot be empty"}

    if len(text) > 2000:  # Reasonable limit for TTS
        return {"error": f"Text too long: {len(text)} characters (max 2000)"}

    speaker_voice_name = job_input.get("speaker_voice")
    parameters = job_input.get("parameters", {})
    seed = parameters.get("seed", job_input.get("seed", 0))

    try:
        # Load models
        model, fish_ae, pca_state = _load_models()
        sample_fn = _build_sample_fn(parameters)

        # Optional speaker conditioning
        speaker_audio = None
        if speaker_voice_name:
            candidate_path = (config.AUDIO_VOICES_DIR / speaker_voice_name).resolve()
            if not str(candidate_path).startswith(str(config.AUDIO_VOICES_DIR.resolve())):
                return {"error": "Invalid speaker_voice path"}
            if not candidate_path.exists():
                return {"error": f"speaker_voice '{speaker_voice_name}' not found"}
            if candidate_path.suffix.lower() not in config.AUDIO_EXTS:
                return {"error": f"Unsupported speaker_voice extension: {candidate_path.suffix}"}

            speaker_audio = load_audio(str(candidate_path)).to(model.device)

        # Generate audio with improved chunking strategy
        # Default to chunking for long prompts; allow explicit 0 to disable.
        max_chars_per_chunk_raw = parameters.get("max_chars_per_chunk", 300)
        enable_crossfade = parameters.get("enable_crossfade", True)
        normalize_boundaries = parameters.get("normalize_boundaries", True)
        target_duration = parameters.get("target_duration_seconds", 10.0)

        try:
            max_chars_per_chunk = int(max_chars_per_chunk_raw)
        except Exception:
            max_chars_per_chunk = 300


        # Use improved chunking strategy
        if max_chars_per_chunk and max_chars_per_chunk > 0:
            # Use audio-aware chunking
            text_chunks = chunk_text_for_audio(text, max_chars=max_chars_per_chunk,
                                             target_duration_seconds=target_duration)
        else:
            text_chunks = [text]

        if not text_chunks:
            return {"error": "Text is empty after normalization"}

        audio_chunks: list[torch.Tensor] = []
        for idx, chunk in enumerate(text_chunks):
            # Use deterministic seed progression for better continuity
            chunk_seed = seed + (idx * 1000)  # Larger spacing for more variation
            audio_chunk, _ = sample_pipeline(
                model=model,
                fish_ae=fish_ae,
                pca_state=pca_state,
                sample_fn=sample_fn,
                text_prompt=chunk,
                speaker_audio=speaker_audio,
                rng_seed=chunk_seed,
            )
            audio_chunks.append(audio_chunk)


        # Apply audio processing for smoother concatenation
        if normalize_boundaries and len(audio_chunks) > 1:
            audio_out = normalize_chunk_boundaries(audio_chunks, sample_rate=44100)
        elif enable_crossfade and len(audio_chunks) > 1:
            audio_out = crossfade_chunks(audio_chunks)
        else:
            audio_out = torch.cat(audio_chunks, dim=-1)

        # Validate output
        if audio_out is None or len(audio_out) == 0:
            return {"error": "No audio generated"}

        duration_seconds = len(audio_out[0]) / 44_100
        session_id = job_input.get("session_id") or str(uuid4())
        upload_meta = _save_and_upload_audio(audio_out[0].cpu(), 44_100, session_id)

        result = {
            "status": "completed",
            "filename": upload_meta["filename"],
            "url": upload_meta["url"],
            "s3_key": upload_meta["key"],
            "metadata": {
                "sample_rate": 44_100,
                "duration": duration_seconds,
                "seed": seed,
                "device": config.device,
            },
        }
        return result

    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


def handler(job: Dict) -> Dict:
    """RunPod serverless handler simplified like Lotus."""
    log.debug(f"Handler received job: {job.get('id')}")
    try:
        result = _synthesize(job.get("input", {}), job.get("id"))
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        log.error(f"Handler failed: {str(e)}")
        log.error(f"Traceback: {error_trace}")
        return {"error": str(e), "error_type": type(e).__name__}


def main() -> None:
    """Simplified main entry point following Lotus pattern"""
    parser = argparse.ArgumentParser(description="RunPod handler for Echo-TTS")
    parser.add_argument("--warmup", action="store_true", help="Load models to warm cache; exits after.")
    args, _ = parser.parse_known_args()

    # Simple startup logging
    print(f"=== Echo-TTS RunPod Handler Starting ===")
    print(f"Device: {config.device}")
    print(f"Working directory: {os.getcwd()}")

    # Warmup models if requested
    if args.warmup:
        print("=== Starting Model Warmup ===")
        try:
            # Validate configuration before warming up
            if not config.validate():
                print("ERROR: Configuration validation failed")
                for error in config.validation_errors:
                    print(f"  - {error}")
                sys.exit(1)

            print("Loading models...")
            model, fish_ae, pca_state = _load_models()
            print("Warmup completed successfully")
            print(f"Models loaded: {list(_MODELS.keys())}")
        except Exception as e:
            print(f"Warmup failed: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
        return

    # Validate configuration before starting
    if not config.validate():
        print("WARNING: Configuration has validation errors:")
        for error in config.validation_errors:
            print(f"  - {error}")
        print("Starting anyway...")

    # Start the RunPod serverless worker (simple like Lotus)
    print("Starting RunPod serverless worker...")
    print("Handler ready to receive requests")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
