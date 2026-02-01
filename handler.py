# coding=utf-8
# Echo-TTS RunPod Serverless Handler
# SPDX-License-Identifier: MIT

"""
RunPod Serverless Handler for Echo-TTS

Following Fish Audio pattern for consistency across TTS services.

Accepts:
- text (str): text to synthesize
- speaker_voice (str, optional): filename in AUDIO_VOICES_DIR for speaker reference
- stream (bool, optional): enable streaming mode (default: false)
- output_format (str, optional): "pcm_16" for decoded audio, "linacodec_tokens" for tokens
- parameters (dict, optional): generation options (num_steps, cfg_scale_text, etc.)

Batch mode returns:
{
    "status": "completed",
    "url": str,  # S3 presigned URL
    "filename": str,
    "s3_key": str,
    "metadata": {...}
}

Streaming mode yields:
{
    "status": "streaming",
    "chunk": int,
    "format": str,
    "audio_chunk" or "tokens": str/list,
    ...
}
{
    "status": "complete",
    ...
}
"""

import os
import sys
import argparse
import base64
import subprocess
import tempfile
import time
import traceback
from typing import Dict, Any, Optional, Generator
from uuid import uuid4

import runpod
import torch
import torchaudio
import boto3

import config as config_module
from config import config  # Config instance for validation and runtime settings
from serverless_engine import get_inference_engine, LINACODEC_AVAILABLE

# Initialize RunPod structured logger
log = runpod.RunPodLogger()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cleanup_old_files(directory: str, days: int = 2) -> None:
    """Delete files older than specified days from directory."""
    try:
        from pathlib import Path

        output_dir = Path(directory)
        if not output_dir.exists():
            return

        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)

        deleted_count = 0
        for file_path in output_dir.glob('*'):
            if file_path.is_file():
                file_age = file_path.stat().st_mtime
                if file_age < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1

        if deleted_count > 0:
            log.info(f"Cleaned up {deleted_count} files older than {days} days from {directory}")
    except Exception as e:
        log.error(f"Cleanup failed: {e}")


def get_s3_client():
    """Create and return S3 client with enhanced error handling."""
    log.debug("Creating S3 client...")

    missing = []
    if not config.S3_ENDPOINT_URL:
        missing.append("S3_ENDPOINT_URL")
    if not config.S3_ACCESS_KEY_ID:
        missing.append("S3_ACCESS_KEY_ID")
    if not config.S3_SECRET_ACCESS_KEY:
        missing.append("S3_SECRET_ACCESS_KEY")
    if not config.S3_BUCKET_NAME:
        missing.append("S3_BUCKET_NAME")

    if missing:
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


def encode_to_opus(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """
    Encode audio tensor to Opus format in OGG container.

    Args:
        audio_tensor: Audio tensor (float32)
        sample_rate: Sample rate of audio

    Returns:
        Opus-encoded audio bytes
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_ogg:
        tmp_ogg_path = tmp_ogg.name

    try:
        # Validate audio tensor
        if audio_tensor is None:
            raise RuntimeError("Audio tensor is None")
        if len(audio_tensor.shape) < 2:
            raise RuntimeError(f"Invalid audio tensor shape: {audio_tensor.shape}")

        # Save as WAV first
        torchaudio.save(tmp_wav_path, audio_tensor, sample_rate)

        # Convert to OGG Opus at 24kHz with 128k bitrate
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", tmp_wav_path,
            "-ar", "24000",
            "-c:a", "libopus",
            "-b:a", "128k",
            "-vbr", "on",
            "-compression_level", "10",
            tmp_ogg_path
        ]
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Read the converted Opus file
        with open(tmp_ogg_path, "rb") as f:
            data = f.read()

        file_size_mb = len(data) / (1024 * 1024)
        log.info(f"Audio encoded to Opus: {file_size_mb:.2f}MB")

        return data

    finally:
        # Clean up temporary files
        for tmp_file in [tmp_wav_path, tmp_ogg_path]:
            try:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
            except OSError:
                pass


def upload_to_s3(audio_bytes: bytes, filename: str) -> str:
    """
    Upload audio to S3 and return presigned URL.

    Args:
        audio_bytes: Audio data to upload
        filename: Filename to use in S3

    Returns:
        Presigned URL for the uploaded file
    """
    s3 = get_s3_client()
    key = filename

    try:
        s3.put_object(
            Bucket=config.S3_BUCKET_NAME,
            Key=key,
            Body=audio_bytes,
            ContentType="audio/ogg; codecs=opus",
        )
        log.info(f"Successfully uploaded to S3: {key}")

        # Generate presigned URL
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": config.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=3600,
        )
        return presigned_url

    except Exception as e:
        error_msg = f"Failed to upload to S3: {str(e)}"
        log.error(error_msg)
        raise RuntimeError(error_msg)


def health_check() -> Dict:
    """Comprehensive health check for the TTS service."""
    log.info("Performing health check...")

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

    # Check LinaCodec
    health_status["checks"]["linacodec"] = {
        "status": "pass" if LINACODEC_AVAILABLE else "warn",
        "details": f"LinaCodec available: {LINACODEC_AVAILABLE}"
    }

    # Overall status
    all_pass = all(check["status"] == "pass" for check in health_status["checks"].values())
    health_status["status"] = "healthy" if all_pass else "unhealthy"

    log.info(f"Health check completed: {health_status['status']}")
    return health_status


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

def extract_and_validate_params(job_input: Dict) -> tuple:
    """
    Extract and validate parameters from job input.

    Returns:
        tuple: (params_dict, error_dict) - error_dict is None if validation passes
    """
    log.debug(f"Validating job input: {list(job_input.keys())}")

    # Extract required parameters
    text = job_input.get("text")
    if not text:
        log.error("Validation failed: Missing 'text' parameter")
        return None, {"error": "Missing 'text' parameter"}

    if not isinstance(text, str):
        return None, {"error": "Invalid 'text' parameter (expected string)"}

    if len(text.strip()) == 0:
        return None, {"error": "Text cannot be empty"}

    if len(text) > 4000:
        return None, {"error": f"Text too long: {len(text)} characters (max 4000)"}

    # Optional parameters
    speaker_voice = job_input.get("speaker_voice")
    stream = job_input.get("stream", False)
    output_format = job_input.get("output_format", "linacodec_tokens")
    parameters = job_input.get("parameters", {})

    # Validate speaker_voice if provided
    if speaker_voice:
        candidate_path = (config.AUDIO_VOICES_DIR / speaker_voice).resolve()
        if not str(candidate_path).startswith(str(config.AUDIO_VOICES_DIR.resolve())):
            return None, {"error": "Invalid speaker_voice path"}
        if not candidate_path.exists():
            return None, {"error": f"speaker_voice '{speaker_voice}' not found"}
        if candidate_path.suffix.lower() not in config.AUDIO_EXTS:
            return None, {"error": f"Unsupported speaker_voice extension: {candidate_path.suffix}"}

    # Validate output_format
    valid_formats = ["pcm_16", "linacodec_tokens"]
    if output_format not in valid_formats:
        return None, {"error": f"Invalid output_format: {output_format}. Must be one of {valid_formats}"}

    # Validate streaming with LinaCodec
    if stream and output_format == "linacodec_tokens" and not LINACODEC_AVAILABLE:
        return None, {"error": "LinaCodec streaming requires LinaCodec to be installed"}

    params = {
        "text": text,
        "speaker_voice": speaker_voice,
        "stream": stream,
        "output_format": output_format,
        "parameters": parameters,
    }

    return params, None


# =============================================================================
# HANDLER FUNCTIONS
# =============================================================================

def handler_batch(job_input: Dict) -> Dict:
    """
    Batch mode handler - generates complete audio and uploads to S3.

    Args:
        job_input: Job input dictionary

    Returns:
        Result dictionary with S3 URL and metadata
    """
    # Clean up old output files
    cleanup_old_files(str(config.OUTPUT_AUDIO_DIR), days=config_module.CLEANUP_DAYS)

    # Extract and validate parameters
    params, error = extract_and_validate_params(job_input)
    if error:
        log.error(f"Parameter validation failed: {error}")
        return error

    text = params["text"]
    speaker_voice = params["speaker_voice"]
    parameters = params["parameters"]

    try:
        # Get inference engine
        inference_engine = get_inference_engine()

        # Generate speech
        log.info(f"Generating speech for {len(text)} characters")
        audio_out, sample_rate = inference_engine.generate_speech(
            text=text,
            speaker_voice=speaker_voice,
            parameters=parameters,
        )

        if audio_out is None or len(audio_out) == 0:
            return {"error": "No audio generated"}

        # Duration calculation
        duration_seconds = len(audio_out[0]) / sample_rate
        session_id = job_input.get("session_id") or str(uuid4())

        # Encode to Opus and upload to S3
        audio_bytes = encode_to_opus(audio_out[0].cpu(), sample_rate)
        filename = f"{session_id}.ogg"
        s3_url = upload_to_s3(audio_bytes, filename)

        return {
            "status": "completed",
            "filename": filename,
            "url": s3_url,
            "s3_key": filename,
            "metadata": {
                "sample_rate": 24000,  # Opus output is resampled to 24kHz
                "codec": "opus",
                "bitrate": "128k",
                "duration": duration_seconds,
                "seed": parameters.get("seed", 0),
                "device": config.device,
            },
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        log.error(f"Batch mode failed: {str(e)}")
        log.error(f"Traceback: {error_trace}")
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


def handler_stream(job_input: Dict) -> Generator[Dict, None, None]:
    """
    Streaming mode handler - yields audio chunks as they're generated.

    Args:
        job_input: Job input dictionary

    Yields:
        Dictionaries with streaming chunk data
    """
    # Extract and validate parameters
    params, error = extract_and_validate_params(job_input)
    if error:
        yield error
        return

    text = params["text"]
    speaker_voice = params["speaker_voice"]
    output_format = params["output_format"]
    parameters = params["parameters"]

    try:
        # Get inference engine
        inference_engine = get_inference_engine()

        if output_format == 'pcm_16':
            # Stream decoded audio chunks
            log.info(f"Starting decoded audio stream (pcm_16)")
            yield from inference_engine.generate_audio_stream_decoded(
                text=text,
                speaker_voice=speaker_voice,
                parameters=parameters,
            )
        else:
            # Stream LinaCodec tokens
            log.info(f"Starting LinaCodec token stream")
            yield from inference_engine.generate_linacodec_token_stream(
                text=text,
                speaker_voice=speaker_voice,
                parameters=parameters,
            )

    except Exception as e:
        error_trace = traceback.format_exc()
        log.error(f"Streaming mode failed: {str(e)}")
        log.error(f"Traceback: {error_trace}")
        yield {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


def handler(job: Dict):
    """
    Main RunPod serverless handler.

    Routes to batch or streaming mode based on input parameters.

    Args:
        job: RunPod job dictionary with 'input' key

    Yields:
        Result dictionary(es)
    """
    job_id = job.get('id')
    input_data = job.get('input', {})

    log.debug(f"[{job_id}] Handler called")

    # Handle health check
    if input_data.get("action") == "health_check":
        yield health_check()
        return

    # Extract streaming parameters first
    stream = input_data.get('stream', False)
    output_format = input_data.get('output_format', 'linacodec_tokens')

    # For streaming mode, use generator
    if stream:
        log.info(f"[{job_id}] Streaming mode: format={output_format}")
        yield from handler_stream(input_data)
        return

    # For batch mode, yield result (handler must be generator for RunPod compatibility)
    log.info(f"[{job_id}] Batch mode - input keys: {list(input_data.keys())}")
    result = handler_batch(input_data)
    log.info(f"[{job_id}] Batch mode result status: {result.get('status', result.get('error', 'unknown'))}")
    yield result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main entry point for RunPod serverless worker."""
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
            if not config.validate():
                print("ERROR: Configuration validation failed")
                for error in config.validation_errors:
                    print(f"  - {error}")
                sys.exit(1)

            print("Loading models...")
            inference_engine = get_inference_engine()
            inference_engine._load_core_models()
            print("Warmup completed successfully")
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

    # Start the RunPod serverless worker
    print("Starting RunPod serverless worker...")
    print("Handler ready to receive requests")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
