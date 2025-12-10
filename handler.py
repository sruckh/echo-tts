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

# Enhanced logging with immediate output
def log_with_flush(level: str, message: str, request_id: Optional[str] = None):
    """Log message and immediately flush to ensure visibility"""
    if level == "info":
        log.info(message, request_id=request_id)
    elif level == "error":
        log.error(message, request_id=request_id)
    elif level == "debug":
        log.debug(message, request_id=request_id)
    elif level == "warning":
        log.warning(message, request_id=request_id)
    # Force flush for critical messages
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

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
            log_with_flush("info", f"GPU detected: {self.gpu_name} with {self.gpu_memory:.1f}GB memory")

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
            log_with_flush("info", f"Audio directories: {self.AUDIO_VOICES_DIR}, {self.OUTPUT_AUDIO_DIR}")
        except Exception as e:
            self.validation_errors.append(f"Failed to create directories: {e}")

        # Additional configuration
        self.AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}

        # Log all environment variables (without sensitive data)
        log_with_flush("info", f"Device: {self.device}")
        log_with_flush("info", f"AUDIO_VOICES_DIR: {self.AUDIO_VOICES_DIR}")
        log_with_flush("info", f"OUTPUT_AUDIO_DIR: {self.OUTPUT_AUDIO_DIR}")
        log_with_flush("info", f"S3_ENDPOINT_URL: {'SET' if self.S3_ENDPOINT_URL else 'NOT SET'}")
        log_with_flush("info", f"S3_BUCKET_NAME: {'SET' if self.S3_BUCKET_NAME else 'NOT SET'}")
        log_with_flush("info", f"HF_TOKEN: {'SET' if self.HF_TOKEN else 'NOT SET'}")

        # Check audio files in voices directory
        try:
            audio_files = list(self.AUDIO_VOICES_DIR.glob("*"))
            audio_files = [f for f in audio_files if f.suffix.lower() in self.AUDIO_EXTS]
            log_with_flush("info", f"Found {len(audio_files)} audio files in voices directory")
            for f in audio_files[:5]:  # Log first 5
                log_with_flush("info", f"  - {f.name}")
            if len(audio_files) > 5:
                log_with_flush("info", f"  ... and {len(audio_files) - 5} more")
        except Exception as e:
            log_with_flush("warning", f"Could not scan audio directory: {e}")

    def validate(self) -> bool:
        """Return True if configuration is valid"""
        if self.validation_errors:
            log_with_flush("error", "Configuration validation failed:")
            for error in self.validation_errors:
                log_with_flush("error", f"  - {error}")
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

    log_with_flush("debug", f"_load_models called. Current cache: {list(_MODELS.keys())}", request_id=request_id)

    if _MODELS:
        log_with_flush("info", "Models already cached, reusing", request_id=request_id)
        return _MODELS["model"], _MODELS["fish_ae"], _MODELS["pca_state"]

    # Validate HF token before attempting to load models
    if not config.HF_TOKEN:
        error_msg = "HF_TOKEN is required to load models from HuggingFace"
        log_with_flush("error", error_msg, request_id=request_id)
        raise RuntimeError(error_msg)

    log_with_flush("info", f"Starting model loading on device: {device}", request_id=request_id)
    start_time = time.time()

    try:
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else None
        log_with_flush("info", f"Using dtype: {torch_dtype}", request_id=request_id)

        # Check available memory
        if device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            log_with_flush("info", f"GPU memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total", request_id=request_id)

        # Load main model
        log_with_flush("info", "Loading EchoDiT model from HuggingFace...", request_id=request_id)
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
            log_with_flush("info", f"EchoDiT model loaded in {model_time:.2f}s", request_id=request_id)
        except Exception as model_error:
            error_msg = f"Failed to load EchoDiT model: {str(model_error)}"
            log_with_flush("error", error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Load autoencoder
        log_with_flush("info", "Loading Fish Speech S1-DAC autoencoder from HuggingFace...", request_id=request_id)
        ae_start = time.time()

        try:
            fish_ae = load_fish_ae_from_hf(
                device=device,
                dtype=torch_dtype,
                compile=False,
                token=config.HF_TOKEN
            )
            ae_time = time.time() - ae_start
            log_with_flush("info", f"Autoencoder loaded in {ae_time:.2f}s", request_id=request_id)
        except Exception as ae_error:
            error_msg = f"Failed to load autoencoder: {str(ae_error)}"
            log_with_flush("error", error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Load PCA state
        log_with_flush("info", "Loading PCA state from HuggingFace...", request_id=request_id)
        pca_start = time.time()

        try:
            pca_state = load_pca_state_from_hf(device=device, token=config.HF_TOKEN)
            pca_time = time.time() - pca_start
            log_with_flush("info", f"PCA state loaded in {pca_time:.2f}s", request_id=request_id)
        except Exception as pca_error:
            error_msg = f"Failed to load PCA state: {str(pca_error)}"
            log_with_flush("error", error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Cache models
        _MODELS.update({"model": model, "fish_ae": fish_ae, "pca_state": pca_state})

        total_time = time.time() - start_time
        log_with_flush("info", f"All models loaded successfully in {total_time:.2f}s", request_id=request_id)

        # Log memory usage after loading
        if device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            log_with_flush("info", f"GPU memory after loading: {allocated:.1f}GB allocated, {cached:.1f}GB cached", request_id=request_id)

        return model, fish_ae, pca_state

    except Exception as e:
        error_trace = traceback.format_exc()
        log_with_flush("error", f"Model loading failed: {str(e)}", request_id=request_id)
        log_with_flush("error", f"Traceback:\n{error_trace}", request_id=request_id)
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
    log_with_flush("debug", "Creating S3 client...")

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
        log_with_flush("error", error_msg)
        raise RuntimeError(error_msg)

    try:
        client = boto3.client(
            "s3",
            endpoint_url=config.S3_ENDPOINT_URL,
            region_name=config.S3_REGION,
            aws_access_key_id=config.S3_ACCESS_KEY_ID,
            aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        )
        log_with_flush("info", f"S3 client created for endpoint: {config.S3_ENDPOINT_URL}")
        return client
    except Exception as e:
        error_msg = f"Failed to create S3 client: {str(e)}"
        log_with_flush("error", error_msg)
        raise RuntimeError(error_msg)


def _save_and_upload_audio(audio_tensor: torch.Tensor, sample_rate: int, session_id: str, request_id: Optional[str] = None) -> Dict[str, str]:
    """Save audio to OGG (Vorbis) and upload to S3-compatible storage."""
    log_with_flush("info", f"Saving and uploading audio for session: {session_id}", request_id=request_id)

    # Ensure output directory exists
    config.OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{session_id}.ogg"
    key = f"{filename}"

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        log_with_flush("debug", f"Saving audio to temporary file: {tmp_path}", request_id=request_id)

        # Validate audio tensor
        if audio_tensor is None:
            raise RuntimeError("Audio tensor is None")
        if len(audio_tensor.shape) < 2:
            raise RuntimeError(f"Invalid audio tensor shape: {audio_tensor.shape}")

        log_with_flush("debug", f"Audio tensor shape: {audio_tensor.shape}, sample_rate: {sample_rate}", request_id=request_id)

        try:
            torchaudio.save(tmp_path, audio_tensor, sample_rate, format="ogg")
            log_with_flush("debug", f"Audio saved successfully to: {tmp_path}", request_id=request_id)
        except Exception as save_error:
            error_msg = f"Failed to save audio file: {str(save_error)}"
            log_with_flush("error", error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Read the saved file
        try:
            with open(tmp_path, "rb") as f:
                data = f.read()
        except Exception as read_error:
            error_msg = f"Failed to read saved audio file: {str(read_error)}"
            log_with_flush("error", error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        file_size_mb = len(data) / (1024 * 1024)
        log_with_flush("info", f"Audio file size: {file_size_mb:.2f}MB", request_id=request_id)

        if file_size_mb > 50:  # Warn if file is very large
            log_with_flush("warning", f"Large audio file: {file_size_mb:.2f}MB", request_id=request_id)

        # Upload to S3
        log_with_flush("debug", "Uploading to S3...", request_id=request_id)
        try:
            s3 = _get_s3_client()
            s3.put_object(
                Bucket=config.S3_BUCKET_NAME,
                Key=key,
                Body=data,
                ContentType="audio/ogg",
            )
            log_with_flush("debug", f"Successfully uploaded to S3: {key}", request_id=request_id)
        except Exception as upload_error:
            error_msg = f"Failed to upload to S3: {str(upload_error)}"
            log_with_flush("error", error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        # Generate presigned URL
        try:
            presigned_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": config.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=3600,
            )
            log_with_flush("debug", "Presigned URL generated successfully", request_id=request_id)
        except Exception as url_error:
            error_msg = f"Failed to generate presigned URL: {str(url_error)}"
            log_with_flush("error", error_msg, request_id=request_id)
            raise RuntimeError(error_msg)

        log_with_flush("info", f"Audio uploaded successfully: {key}", request_id=request_id)
        return {"filename": filename, "url": presigned_url, "key": key}

    except Exception as e:
        error_msg = f"Failed to save/upload audio: {str(e)}"
        log_with_flush("error", error_msg, request_id=request_id)
        raise

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
            log_with_flush("debug", f"Cleaned up temporary file: {tmp_path}", request_id=request_id)
        except OSError as cleanup_error:
            log_with_flush("warning", f"Failed to clean up temporary file {tmp_path}: {cleanup_error}", request_id=request_id)


def health_check(request_id: Optional[str] = None) -> Dict:
    """Comprehensive health check for the TTS service"""
    log_with_flush("info", "Performing health check...", request_id=request_id)

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

    log_with_flush("info", f"Health check completed: {health_status['status']}", request_id=request_id)
    return health_status


def _synthesize(job_input: Dict, job_id: Optional[str] = None) -> Dict:
    """Core synthesis path shared by serverless handler and CLI warmup."""
    request_id = job_id

    log_with_flush("info", f"Starting synthesis. Job ID: {job_id}", request_id=request_id)
    log_with_flush("debug", f"Input keys: {list(job_input.keys())}", request_id=request_id)

    # Handle health check requests
    if job_input.get("action") == "health_check":
        log_with_flush("info", "Health check requested", request_id=request_id)
        return health_check(request_id=request_id)

    # Validate configuration first
    if not config.validate():
        error_msg = f"Configuration validation failed: {'; '.join(config.validation_errors)}"
        log_with_flush("error", error_msg, request_id=request_id)
        return {"error": error_msg}

    # Validate text input
    text = job_input.get("text")
    if not text or not isinstance(text, str):
        error_msg = "Missing or invalid 'text' field (expected string)"
        log_with_flush("error", error_msg, request_id=request_id)
        return {"error": error_msg}

    if len(text.strip()) == 0:
        error_msg = "Text cannot be empty"
        log_with_flush("error", error_msg, request_id=request_id)
        return {"error": error_msg}

    if len(text) > 2000:  # Reasonable limit for TTS
        error_msg = f"Text too long: {len(text)} characters (max 2000)"
        log_with_flush("error", error_msg, request_id=request_id)
        return {"error": error_msg}

    log_with_flush("info", f"Text length: {len(text)} characters", request_id=request_id)

    speaker_voice_name = job_input.get("speaker_voice")
    parameters = job_input.get("parameters", {})
    seed = parameters.get("seed", job_input.get("seed", 0))

    log_with_flush("info", f"Parameters: seed={seed}, speaker_voice={speaker_voice_name}", request_id=request_id)

    try:
        # Progress: 0-30% - Load models
        if job_id:
            try:
                runpod.serverless.progress_update(job_id, 10, "Loading models...")
            except Exception as progress_error:
                log_with_flush("warning", f"Failed to update progress: {progress_error}", request_id=request_id)

        log_with_flush("info", "Loading models...", request_id=request_id)
        model, fish_ae, pca_state = _load_models(request_id=request_id)
        sample_fn = _build_sample_fn(parameters, request_id=request_id)

        # Progress: 30-40% - Load speaker audio if provided
        if job_id:
            try:
                runpod.serverless.progress_update(job_id, 30, "Processing speaker audio...")
            except Exception as progress_error:
                log_with_flush("warning", f"Failed to update progress: {progress_error}", request_id=request_id)

        # Optional speaker conditioning
        speaker_audio: Optional[torch.Tensor] = None
        if speaker_voice_name:
            log_with_flush("info", f"Loading speaker voice: {speaker_voice_name}", request_id=request_id)
            try:
                candidate_path = (config.AUDIO_VOICES_DIR / speaker_voice_name).resolve()
                if not str(candidate_path).startswith(str(config.AUDIO_VOICES_DIR.resolve())):
                    error_msg = "Invalid speaker_voice path (path traversal attempt)"
                    log_with_flush("error", error_msg, request_id=request_id)
                    return {"error": error_msg}
                if not candidate_path.exists():
                    error_msg = f"speaker_voice '{speaker_voice_name}' not found"
                    log_with_flush("error", error_msg, request_id=request_id)
                    return {"error": error_msg}
                if candidate_path.suffix.lower() not in config.AUDIO_EXTS:
                    error_msg = f"Unsupported speaker_voice extension: {candidate_path.suffix}"
                    log_with_flush("error", error_msg, request_id=request_id)
                    return {"error": error_msg}

                speaker_audio = load_audio(str(candidate_path)).to(model.device)
                log_with_flush("info", f"Speaker audio loaded: shape={speaker_audio.shape}", request_id=request_id)
            except Exception as e:
                error_trace = traceback.format_exc()
                log_with_flush("error", f"Failed to load speaker_voice: {str(e)}", request_id=request_id)
                log_with_flush("debug", f"Traceback:\n{error_trace}", request_id=request_id)
                return {"error": f"Failed to load speaker_voice: {e}"}
        else:
            log_with_flush("info", "No speaker voice specified, using default", request_id=request_id)

        # Progress: 40-90% - Run generation
        if job_id:
            try:
                runpod.serverless.progress_update(job_id, 40, "Generating audio...")
            except Exception as progress_error:
                log_with_flush("warning", f"Failed to update progress: {progress_error}", request_id=request_id)

        log_with_flush("info", "Starting audio generation...", request_id=request_id)
        gen_start = time.time()

        try:
            audio_out, _ = sample_pipeline(
                model=model,
                fish_ae=fish_ae,
                pca_state=pca_state,
                sample_fn=sample_fn,
                text_prompt=text,
                speaker_audio=speaker_audio,
                rng_seed=seed,
            )
            gen_time = time.time() - gen_start
            log_with_flush("info", f"Audio generation completed in {gen_time:.2f}s", request_id=request_id)
        except Exception as gen_error:
            error_trace = traceback.format_exc()
            log_with_flush("error", f"Audio generation failed: {str(gen_error)}", request_id=request_id)
            log_with_flush("error", f"Traceback:\n{error_trace}", request_id=request_id)
            return {"error": f"Audio generation failed: {gen_error}"}

        # Progress: 90-100% - Upload
        if job_id:
            try:
                runpod.serverless.progress_update(job_id, 90, "Uploading audio...")
            except Exception as progress_error:
                log_with_flush("warning", f"Failed to update progress: {progress_error}", request_id=request_id)

        # Validate output
        if audio_out is None or len(audio_out) == 0:
            error_msg = "No audio generated"
            log_with_flush("error", error_msg, request_id=request_id)
            return {"error": error_msg}

        duration_seconds = len(audio_out[0]) / 44_100
        log_with_flush("info", f"Generated audio duration: {duration_seconds:.2f}s", request_id=request_id)

        if duration_seconds < 0.1:
            log_with_flush("warning", f"Very short audio generated: {duration_seconds:.2f}s", request_id=request_id)

        session_id = job_input.get("session_id") or str(uuid4())
        upload_meta = _save_and_upload_audio(audio_out[0].cpu(), 44_100, session_id, request_id=request_id)

        result = {
            "status": "completed",
            "filename": upload_meta["filename"],
            "url": upload_meta["url"],
            "s3_key": upload_meta["key"],
            "metadata": {
                "sample_rate": 44_100,
                "duration": duration_seconds,
                "seed": seed,
                "generation_time": gen_time,
                "device": config.device,
            },
        }

        log_with_flush("info", f"Synthesis completed successfully. File: {upload_meta['filename']}", request_id=request_id)
        return result

    except Exception as e:
        error_trace = traceback.format_exc()
        log_with_flush("error", f"Synthesis failed: {str(e)}", request_id=request_id)
        log_with_flush("error", f"Error type: {type(e).__name__}", request_id=request_id)
        log_with_flush("error", f"Traceback:\n{error_trace}", request_id=request_id)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


def handler(job: Dict) -> Dict:
    """RunPod serverless handler with enhanced logging and error handling."""
    job_id = job.get("id")
    log_with_flush("info", f"=== HANDLER CALLED === Job ID: {job_id}", request_id=job_id)

    # Log incoming job structure (without sensitive data)
    try:
        job_input = job.get("input", {})
        safe_input = {k: v for k, v in job_input.items() if k != "token"}
        log_with_flush("info", f"Job input: {safe_input}", request_id=job_id)
    except Exception as input_error:
        log_with_flush("error", f"Failed to parse job input: {input_error}", request_id=job_id)
        return {"error": f"Invalid job input: {input_error}"}

    # Enhanced startup logging
    if job_id:
        log_with_flush("info", f"Processing job {job_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}", request_id=job_id)

    try:
        # Validate basic job structure
        if not isinstance(job, dict):
            error_msg = f"Invalid job type: {type(job)}"
            log_with_flush("error", error_msg, request_id=job_id)
            return {"error": error_msg}

        if "input" not in job:
            error_msg = "Job missing 'input' field"
            log_with_flush("error", error_msg, request_id=job_id)
            return {"error": error_msg}

        # Log handler state
        log_with_flush("debug", f"Models cached: {list(_MODELS.keys())}", request_id=job_id)
        log_with_flush("debug", f"Device: {config.device}", request_id=job_id)

        # Process the job
        result = _synthesize(job_input, job_id)

        log_with_flush("debug", f"Synthesis result type: {type(result)}", request_id=job_id)
        log_with_flush("debug", f"Synthesis result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}", request_id=job_id)

        # Wrap result in proper RunPod format
        if isinstance(result, dict) and "error" in result:
            # Return errors directly (RunPod expects {"error": "message"} format)
            log_with_flush("error", f"Job failed with error: {result.get('error')}", request_id=job_id)
            if "error_type" in result:
                log_with_flush("error", f"Error type: {result.get('error_type')}", request_id=job_id)
            return result
        else:
            # Wrap successful results in "output" key
            log_with_flush("info", "Job completed successfully", request_id=job_id)
            if isinstance(result, dict):
                log_with_flush("debug", f"Returning output with keys: {list(result.keys())}", request_id=job_id)
            return {"output": result}

    except Exception as e:
        # Catch any unhandled exceptions
        error_trace = traceback.format_exc()
        log_with_flush("error", f"=== UNHANDLED EXCEPTION IN HANDLER ===", request_id=job_id)
        log_with_flush("error", f"Exception: {str(e)}", request_id=job_id)
        log_with_flush("error", f"Exception type: {type(e).__name__}", request_id=job_id)
        log_with_flush("error", f"Traceback:\n{error_trace}", request_id=job_id)

        return {
            "error": f"Unhandled exception: {str(e)}",
            "error_type": type(e).__name__,
            "traceback": error_trace
        }

    finally:
        log_with_flush("debug", f"Handler function completed for job {job_id}", request_id=job_id)


def main() -> None:
    """Main entry point with comprehensive startup diagnostics"""
    parser = argparse.ArgumentParser(description="RunPod handler for Echo-TTS")
    parser.add_argument("--warmup", action="store_true", help="Load models to warm cache; exits after.")
    parser.add_argument("--rp_serve_api", action="store_true", help="Start RunPod serverless API.")
    parser.add_argument("--test_env", action="store_true", help="Test environment configuration and exit.")
    parser.add_argument("--test_health", action="store_true", help="Run health check and exit.")
    args = parser.parse_args()

    # Enhanced startup logging
    log_with_flush("info", "=== Echo-TTS RunPod Handler Starting ===")
    log_with_flush("info", f"Python version: {os.sys.version}")
    log_with_flush("info", f"PyTorch version: {torch.__version__}")
    log_with_flush("info", f"RunPod version: {runpod.__version__}")
    log_with_flush("info", f"Working directory: {os.getcwd()}")
    log_with_flush("info", f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Test environment configuration
    if args.test_env:
        log_with_flush("info", "=== Testing Environment Configuration ===")
        config_valid = config.validate()
        log_with_flush("info", f"Configuration valid: {config_valid}")

        if config.validation_errors:
            log_with_flush("error", "Configuration errors:")
            for error in config.validation_errors:
                log_with_flush("error", f"  - {error}")

        # Test S3 connection if configured
        if all([config.S3_ENDPOINT_URL, config.S3_ACCESS_KEY_ID, config.S3_SECRET_ACCESS_KEY, config.S3_BUCKET_NAME]):
            try:
                s3_client = _get_s3_client()
                log_with_flush("info", "S3 client created successfully")

                # Test bucket access (list objects)
                response = s3_client.list_objects_v2(Bucket=config.S3_BUCKET_NAME, MaxKeys=1)
                log_with_flush("info", f"S3 bucket access confirmed: {config.S3_BUCKET_NAME}")
            except Exception as s3_error:
                log_with_flush("error", f"S3 connection failed: {s3_error}")
        else:
            log_with_flush("warning", "S3 not fully configured")

        return

    # Test health check
    if args.test_health:
        log_with_flush("info", "=== Running Health Check ===")
        health_result = health_check()
        print(f"\nHealth Status: {health_result['status']}")
        for check_name, check_result in health_result['checks'].items():
            print(f"  {check_name}: {check_result['status']} - {check_result['details']}")
        return

    # Warmup models
    if args.warmup:
        log_with_flush("info", "=== Starting Model Warmup ===")

        # Validate configuration before warming up
        if not config.validate():
            log_with_flush("error", "Configuration validation failed - cannot warm up models")
            return

        try:
            log_with_flush("info", "Loading models...")
            model, fish_ae, pca_state = _load_models()
            log_with_flush("info", "Warmup completed successfully")
            log_with_flush("info", f"Models loaded: {list(_MODELS.keys())}")
        except Exception as e:
            error_trace = traceback.format_exc()
            log_with_flush("error", f"Warmup failed: {str(e)}")
            log_with_flush("error", f"Traceback:\n{error_trace}")
            raise
        return

    # Start RunPod serverless API
    if args.rp_serve_api:
        log_with_flush("info", "=== Starting RunPod Serverless API ===")

        # Final configuration check
        log_with_flush("info", "Performing final configuration check...")
        config_valid = config.validate()
        if not config_valid:
            log_with_flush("warning", "Configuration has validation errors, but starting anyway...")
            for error in config.validation_errors:
                log_with_flush("warning", f"  - {error}")

        # Log final status
        log_with_flush("info", f"Device: {config.device}")
        log_with_flush("info", f"Models cached: {len(_MODELS)}")
        log_with_flush("info", f"Audio directory: {config.AUDIO_VOICES_DIR}")
        log_with_flush("info", f"Output directory: {config.OUTPUT_AUDIO_DIR}")
        log_with_flush("info", f"S3 configured: {all([config.S3_ENDPOINT_URL, config.S3_ACCESS_KEY_ID, config.S3_SECRET_ACCESS_KEY, config.S3_BUCKET_NAME])}")

        try:
            log_with_flush("info", "Starting RunPod serverless worker...")
            log_with_flush("info", "Handler ready to receive requests")

            # Start the RunPod serverless worker with explicit configuration
            worker_config = {
                "handler": handler,
                "refresh_worker": False  # Keep worker alive between requests
            }

            log_with_flush("info", f"Worker config: {worker_config}")

            # Start the RunPod serverless worker
            log_with_flush("info", "Calling runpod.serverless.start()...")
            runpod.serverless.start(worker_config)
            log_with_flush("info", "runpod.serverless.start() returned (this should not happen)")

        except KeyboardInterrupt:
            log_with_flush("info", "Server stopped by user")
        except Exception as server_error:
            error_trace = traceback.format_exc()
            log_with_flush("error", f"Failed to start server: {str(server_error)}")
            log_with_flush("error", f"Traceback:\n{error_trace}")
            raise
        return

    parser.error("No action specified. Use --warmup, --rp_serve_api, --test_env, or --test_health.")


if __name__ == "__main__":
    main()
