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


def _load_models(device: str = DEFAULT_DEVICE) -> Tuple[object, object, object]:
    """Lazy-load and cache model, autoencoder, and PCA state."""
    if _MODELS:
        return _MODELS["model"], _MODELS["fish_ae"], _MODELS["pca_state"]

    torch_dtype = torch.bfloat16 if device.startswith("cuda") else None
    model = load_model_from_hf(device=device, dtype=torch_dtype, compile=False, delete_blockwise_modules=False, token=HF_TOKEN)
    fish_ae = load_fish_ae_from_hf(device=device, dtype=torch_dtype, compile=False, token=HF_TOKEN)
    pca_state = load_pca_state_from_hf(device=device, token=HF_TOKEN)

    _MODELS.update({"model": model, "fish_ae": fish_ae, "pca_state": pca_state})
    return model, fish_ae, pca_state


def _build_sample_fn(params: Dict) -> callable:
    """Create sampler partial with defaults and overrides."""
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


def _save_and_upload_audio(audio_tensor: torch.Tensor, sample_rate: int, session_id: str) -> Dict[str, str]:
    """Save audio to OGG (Vorbis) and upload to S3-compatible storage."""
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{session_id}.ogg"
    key = f"{filename}"

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torchaudio.save(tmp_path, audio_tensor, sample_rate, format="ogg")
        with open(tmp_path, "rb") as f:
            data = f.read()

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
        return {"filename": filename, "url": presigned_url, "key": key}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _synthesize(job_input: Dict) -> Dict:
    """Core synthesis path shared by serverless handler and CLI warmup."""
    text = job_input.get("text")
    if not text or not isinstance(text, str):
        return {"status": "error", "error": "Missing or invalid 'text' (expected string)."}

    speaker_voice_name = job_input.get("speaker_voice")
    parameters = job_input.get("parameters", {})
    seed = parameters.get("seed", job_input.get("seed", 0))

    # Load models (lazy)
    model, fish_ae, pca_state = _load_models()
    sample_fn = _build_sample_fn(parameters)

    # Optional speaker conditioning
    speaker_audio: Optional[torch.Tensor] = None
    if speaker_voice_name:
        try:
            candidate_path = (AUDIO_VOICES_DIR / speaker_voice_name).resolve()
            if not str(candidate_path).startswith(str(AUDIO_VOICES_DIR.resolve())):
                return {"status": "error", "error": "Invalid speaker_voice path."}
            if not candidate_path.exists():
                return {"status": "error", "error": f"speaker_voice '{speaker_voice_name}' not found."}
            if candidate_path.suffix.lower() not in AUDIO_EXTS:
                return {"status": "error", "error": f"Unsupported speaker_voice extension: {candidate_path.suffix}"}
            speaker_audio = load_audio(str(candidate_path)).to(model.device)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"Failed to load speaker_voice: {e}"}

    # Run generation
    audio_out, _ = sample_pipeline(
        model=model,
        fish_ae=fish_ae,
        pca_state=pca_state,
        sample_fn=sample_fn,
        text_prompt=text,
        speaker_audio=speaker_audio,
        rng_seed=seed,
    )

    duration_seconds = len(audio_out[0]) / 44_100
    session_id = job_input.get("session_id") or str(uuid4())
    upload_meta = _save_and_upload_audio(audio_out[0].cpu(), 44_100, session_id)

    return {
        "status": "completed",
        "filename": upload_meta["filename"],
        "url": upload_meta["url"],
        "s3_key": upload_meta["key"],
        "metadata": {
            "sample_rate": 44_100,
            "duration": duration_seconds,
            "seed": seed,
        },
    }


def handler(job: Dict) -> Dict:
    """RunPod serverless handler."""
    try:
        job_input = job.get("input", {})
        print(f"[handler] received job_id={job.get('id')} keys={list(job_input.keys())}", flush=True)
        return _synthesize(job_input)
    except Exception as e:  # noqa: BLE001
        print(f"[handler] error job_id={job.get('id')}: {e}", flush=True)
        return {"status": "error", "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod handler for Echo-TTS")
    parser.add_argument("--warmup", action="store_true", help="Load models to warm cache; exits after.")
    parser.add_argument("--rp_serve_api", action="store_true", help="Start RunPod serverless API.")
    args = parser.parse_args()

    if args.warmup:
        _load_models()
        return

    if args.rp_serve_api:
        runpod.serverless.start({"handler": handler})
        return

    parser.error("No action specified. Use --warmup or --rp_serve_api.")


if __name__ == "__main__":
    main()
