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

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
AUDIO_VOICES_DIR = Path(os.environ.get("AUDIO_VOICES_DIR", "/runpod-volume/echo-tts/audio_voices"))
OUTPUT_AUDIO_DIR = Path(os.environ.get("OUTPUT_AUDIO_DIR", "/runpod-volume/echo-tts/output_audio"))
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
HF_TOKEN = os.environ.get("HF_TOKEN")
_MODELS: Dict[str, object] = {}


def _load_models(device: str = DEFAULT_DEVICE, request_id: Optional[str] = None) -> Tuple[object, object, object]:
    """Lazy-load and cache model, autoencoder, and PCA state."""
    if _MODELS:
        log.info("Models already cached, reusing", request_id=request_id)
        return _MODELS["model"], _MODELS["fish_ae"], _MODELS["pca_state"]

    log.info(f"Starting model loading on device: {device}", request_id=request_id)
    start_time = time.time()

    try:
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else None
        log.info(f"Using dtype: {torch_dtype}", request_id=request_id)

        # Load main model
        log.info("Loading EchoDiT model from HuggingFace...", request_id=request_id)
        model_start = time.time()
        model = load_model_from_hf(
            device=device,
            dtype=torch_dtype,
            compile=False,
            delete_blockwise_modules=False,
            token=HF_TOKEN
        )
        model_time = time.time() - model_start
        log.info(f"EchoDiT model loaded in {model_time:.2f}s", request_id=request_id)

        # Load autoencoder
        log.info("Loading Fish Speech S1-DAC autoencoder from HuggingFace...", request_id=request_id)
        ae_start = time.time()
        fish_ae = load_fish_ae_from_hf(
            device=device,
            dtype=torch_dtype,
            compile=False,
            token=HF_TOKEN
        )
        ae_time = time.time() - ae_start
        log.info(f"Autoencoder loaded in {ae_time:.2f}s", request_id=request_id)

        # Load PCA state
        log.info("Loading PCA state from HuggingFace...", request_id=request_id)
        pca_start = time.time()
        pca_state = load_pca_state_from_hf(device=device, token=HF_TOKEN)
        pca_time = time.time() - pca_start
        log.info(f"PCA state loaded in {pca_time:.2f}s", request_id=request_id)

        # Cache models
        _MODELS.update({"model": model, "fish_ae": fish_ae, "pca_state": pca_state})

        total_time = time.time() - start_time
        log.info(f"All models loaded successfully in {total_time:.2f}s", request_id=request_id)

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
    if not all([S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
        raise RuntimeError("Missing S3 configuration. Set S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME.")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    )


def _save_and_upload_audio(audio_tensor: torch.Tensor, sample_rate: int, session_id: str, request_id: Optional[str] = None) -> Dict[str, str]:
    """Save audio to OGG (Vorbis) and upload to S3-compatible storage."""
    log.info(f"Saving and uploading audio for session: {session_id}", request_id=request_id)
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{session_id}.ogg"
    key = f"{filename}"

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        log.debug(f"Saving audio to temporary file: {tmp_path}", request_id=request_id)
        torchaudio.save(tmp_path, audio_tensor, sample_rate, format="ogg")

        with open(tmp_path, "rb") as f:
            data = f.read()

        file_size_mb = len(data) / (1024 * 1024)
        log.info(f"Audio file size: {file_size_mb:.2f}MB", request_id=request_id)

        log.debug("Uploading to S3...", request_id=request_id)
        s3 = _get_s3_client()
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=data,
            ContentType="audio/ogg",
        )

        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": key},
            ExpiresIn=3600,
        )

        log.info(f"Audio uploaded successfully: {key}", request_id=request_id)
        return {"filename": filename, "url": presigned_url, "key": key}

    except Exception as e:
        log.error(f"Failed to save/upload audio: {str(e)}", request_id=request_id)
        raise

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _synthesize(job_input: Dict, job_id: Optional[str] = None) -> Dict:
    """Core synthesis path shared by serverless handler and CLI warmup."""
    request_id = job_id

    log.info("Starting synthesis", request_id=request_id)
    log.debug(f"Input keys: {list(job_input.keys())}", request_id=request_id)

    # Validate text input
    text = job_input.get("text")
    if not text or not isinstance(text, str):
        log.error("Missing or invalid 'text' field", request_id=request_id)
        return {"error": "Missing or invalid 'text' (expected string)."}

    log.info(f"Text length: {len(text)} characters", request_id=request_id)

    speaker_voice_name = job_input.get("speaker_voice")
    parameters = job_input.get("parameters", {})
    seed = parameters.get("seed", job_input.get("seed", 0))

    log.info(f"Parameters: seed={seed}, speaker_voice={speaker_voice_name}", request_id=request_id)

    try:
        # Progress: 0-30% - Load models
        if job_id:
            runpod.serverless.progress_update(job_id, 10, "Loading models...")

        log.info("Loading models...", request_id=request_id)
        model, fish_ae, pca_state = _load_models(request_id=request_id)
        sample_fn = _build_sample_fn(parameters, request_id=request_id)

        # Progress: 30-40% - Load speaker audio if provided
        if job_id:
            runpod.serverless.progress_update(job_id, 30, "Processing speaker audio...")

        # Optional speaker conditioning
        speaker_audio: Optional[torch.Tensor] = None
        if speaker_voice_name:
            log.info(f"Loading speaker voice: {speaker_voice_name}", request_id=request_id)
            try:
                candidate_path = (AUDIO_VOICES_DIR / speaker_voice_name).resolve()
                if not str(candidate_path).startswith(str(AUDIO_VOICES_DIR.resolve())):
                    log.error("Invalid speaker_voice path (path traversal attempt)", request_id=request_id)
                    return {"error": "Invalid speaker_voice path."}
                if not candidate_path.exists():
                    log.error(f"Speaker voice file not found: {speaker_voice_name}", request_id=request_id)
                    return {"error": f"speaker_voice '{speaker_voice_name}' not found."}
                if candidate_path.suffix.lower() not in AUDIO_EXTS:
                    log.error(f"Unsupported speaker_voice extension: {candidate_path.suffix}", request_id=request_id)
                    return {"error": f"Unsupported speaker_voice extension: {candidate_path.suffix}"}

                speaker_audio = load_audio(str(candidate_path)).to(model.device)
                log.info(f"Speaker audio loaded: shape={speaker_audio.shape}", request_id=request_id)
            except Exception as e:
                error_trace = traceback.format_exc()
                log.error(f"Failed to load speaker_voice: {str(e)}", request_id=request_id)
                log.debug(f"Traceback:\n{error_trace}", request_id=request_id)
                return {"error": f"Failed to load speaker_voice: {e}"}
        else:
            log.info("No speaker voice specified, using default", request_id=request_id)

        # Progress: 40-90% - Run generation
        if job_id:
            runpod.serverless.progress_update(job_id, 40, "Generating audio...")

        log.info("Starting audio generation...", request_id=request_id)
        gen_start = time.time()

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
        log.info(f"Audio generation completed in {gen_time:.2f}s", request_id=request_id)

        # Progress: 90-100% - Upload
        if job_id:
            runpod.serverless.progress_update(job_id, 90, "Uploading audio...")

        duration_seconds = len(audio_out[0]) / 44_100
        log.info(f"Generated audio duration: {duration_seconds:.2f}s", request_id=request_id)

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
            },
        }

        log.info("Synthesis completed successfully", request_id=request_id)
        return result

    except Exception as e:
        error_trace = traceback.format_exc()
        log.error(f"Synthesis failed: {str(e)}", request_id=request_id)
        log.error(f"Traceback:\n{error_trace}", request_id=request_id)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


def handler(job: Dict) -> Dict:
    """RunPod serverless handler."""
    job_id = job.get("id")
    log.info(f"Received job", request_id=job_id)

    try:
        job_input = job.get("input", {})
        log.debug(f"Job input keys: {list(job_input.keys())}", request_id=job_id)

        result = _synthesize(job_input, job_id)

        # Wrap result in proper RunPod format
        if "error" in result:
            # Return errors directly (RunPod expects {"error": "message"} format)
            log.error(f"Job failed with error: {result.get('error')}", request_id=job_id)
            return result
        else:
            # Wrap successful results in "output" key
            log.info("Job completed successfully", request_id=job_id)
            return {"output": result}

    except Exception as e:
        # Catch any unhandled exceptions
        error_trace = traceback.format_exc()
        log.error(f"Handler exception: {str(e)}", request_id=job_id)
        log.error(f"Traceback:\n{error_trace}", request_id=job_id)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod handler for Echo-TTS")
    parser.add_argument("--warmup", action="store_true", help="Load models to warm cache; exits after.")
    parser.add_argument("--rp_serve_api", action="store_true", help="Start RunPod serverless API.")
    args = parser.parse_args()

    if args.warmup:
        log.info("Starting warmup: loading models...")
        try:
            _load_models()
            log.info("Warmup completed successfully")
        except Exception as e:
            log.error(f"Warmup failed: {str(e)}")
            raise
        return

    if args.rp_serve_api:
        log.info("Starting RunPod serverless API...")
        runpod.serverless.start({"handler": handler})
        return

    parser.error("No action specified. Use --warmup or --rp_serve_api.")


if __name__ == "__main__":
    main()
