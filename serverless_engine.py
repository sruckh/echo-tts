# coding=utf-8
# Echo-TTS RunPod Serverless Inference Engine
# SPDX-License-Identifier: MIT

"""
Echo-TTS Inference Engine for RunPod Serverless

Based on EchoDiT model with speaker reference conditioning.
Supports streaming with LinaCodec encoding for efficient transmission.
"""

import gc
import os
import re
import tempfile
import time
import logging
from functools import partial
from typing import Dict, Any, Optional, Tuple, Generator, List, Union
from uuid import uuid4

import torch
import torchaudio
import numpy as np

import config

# Import from upstream echo-tts inference module
from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    sample_pipeline,
    sample_euler_cfg_independent_guidances,
)

log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_WHITESPACE_RE = re.compile(r"\s+")

# LinaCodec Availability
LINACODEC_AVAILABLE = config.LINACODEC_AVAILABLE

# =============================================================================
# TEXT CHUNKING UTILITIES
# =============================================================================

def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Split input text into <= max_chars character chunks, preferring sentence/clause/word boundaries."""
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    normalized = _WHITESPACE_RE.sub(" ", (text or "")).strip()
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [normalized]

    sentence_enders = {".", "!", "?"}
    clause_enders = {",", ";", ":"}
    closers = {'"', "'", ")", "]", "}", "\u201d", "\u2019"}

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

        split_at = candidate_sentence or candidate_clause or candidate_space
        if split_at is None:
            split_at = max_chars

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    return chunks


def chunk_text_for_audio(text: str, max_chars: int = 300, target_duration_seconds: float = 10.0) -> list[str]:
    """Split text into chunks optimized for audio generation."""
    chars_per_second = 2.5
    estimated_chars = max(50, int(target_duration_seconds * chars_per_second))
    max_effective = min(max_chars, estimated_chars) if max_chars > 0 else max_chars
    return chunk_text(text, max_effective)


def crossfade_chunks(audio_chunks: List[torch.Tensor], crossfade_ms: int = 50, sample_rate: int = 44100) -> torch.Tensor:
    """Crossfade audio chunks for smoother transitions."""
    if len(audio_chunks) == 0:
        return torch.tensor([])
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    crossfade_samples = int((crossfade_ms / 1000) * sample_rate)
    result = audio_chunks[0]
    for i, chunk in enumerate(audio_chunks[1:], 1):
        if result.dim() > 1:
            result = result.squeeze()
        if chunk.dim() > 1:
            chunk = chunk.squeeze()

        if len(result) < crossfade_samples or len(chunk) < crossfade_samples:
            result = torch.cat([result, chunk])
            continue

        fade_out = torch.linspace(1, 0, crossfade_samples, device=result.device)
        fade_in = torch.linspace(0, 1, crossfade_samples, device=chunk.device)

        result_tail = result[-crossfade_samples:] * fade_out
        chunk_head = chunk[:crossfade_samples] * fade_in
        crossfaded = result_tail + chunk_head

        result = torch.cat([result[:-crossfade_samples], crossfaded, chunk[crossfade_samples:]])

    return result


def normalize_chunk_boundaries(
    audio_chunks: List[torch.Tensor],
    sample_rate: int = 44100,
    silence_threshold: float = 0.01,
    min_silence_samples: int = 2400,
    crossfade_ms: int = 50
) -> torch.Tensor:
    """Normalize silence at chunk boundaries to reduce artifacts."""
    if not audio_chunks:
        return torch.tensor([])
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    normalized_chunks = []
    for i, chunk in enumerate(audio_chunks):
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)

        if i < len(audio_chunks) - 1:
            tail_samples = min(chunk.shape[-1], min_silence_samples * 2)
            tail_energy = torch.abs(chunk[..., -tail_samples:])
            tail_energy_flat = tail_energy.flatten()
            trailing_silence = 0

            for j in range(len(tail_energy_flat) - 1, -1, -1):
                if tail_energy_flat[j] < silence_threshold:
                    trailing_silence += 1
                else:
                    break

            if trailing_silence > min_silence_samples:
                chunk = chunk[..., :-(trailing_silence - min_silence_samples)]
            elif trailing_silence < min_silence_samples and trailing_silence > 0:
                additional_silence = min_silence_samples - trailing_silence
                silence = torch.zeros(*chunk.shape[:-1], additional_silence, device=chunk.device)
                chunk = torch.cat([chunk, silence], dim=-1)
            elif trailing_silence == 0:
                silence = torch.zeros(*chunk.shape[:-1], min_silence_samples, device=chunk.device)
                chunk = torch.cat([chunk, silence], dim=-1)

        normalized_chunks.append(chunk)

    return crossfade_chunks(normalized_chunks, crossfade_ms=crossfade_ms, sample_rate=sample_rate)


# =============================================================================
# ECHO-TTS INFERENCE ENGINE
# =============================================================================

class EchoTTSInference:
    """Echo-TTS Inference Engine. Supports speaker conditioning, streaming with LinaCodec encoding, and batch mode."""

    def __init__(self, device: str = None, torch_dtype: torch.dtype = None):
        self.device = device or config.DEVICE
        self.torch_dtype = torch_dtype or (torch.bfloat16 if self.device.startswith("cuda") else None)
        self._models: Dict[str, object] = {}
        self._linacodec = None
        log.info(f"EchoTTSInference initialized: device={self.device}, dtype={self.torch_dtype}")

    def _load_linacodec(self):
        """Load LinaCodec encoder/decoder (cached)."""
        if self._linacodec:
            return self._linacodec
        if not LINACODEC_AVAILABLE:
            raise RuntimeError("LinaCodec is not installed")

        log.info("Loading LinaCodec model from network volume...")
        os.environ['HF_HOME'] = '/runpod-volume/huggingface-cache'
        os.environ['TRANSFORMERS_CACHE'] = '/runpod-volume/huggingface-cache'

        from linacodec.codec import LinaCodec
        self._linacodec = LinaCodec()
        log.info("LinaCodec loaded and cached to network volume!")
        return self._linacodec

    def _load_core_models(self):
        """Load and cache core Echo-TTS models."""
        if "model" in self._models and "fish_ae" in self._models and "pca_state" in self._models:
            return self._models["model"], self._models["fish_ae"], self._models["pca_state"]

        if not config.HF_TOKEN:
            raise RuntimeError("HF_TOKEN is required to load models from HuggingFace")

        log.info(f"Loading Echo-TTS models on device: {self.device}")
        start_time = time.time()

        try:
            log.info("Loading EchoDiT model from HuggingFace...")
            model = load_model_from_hf(
                device=self.device,
                dtype=self.torch_dtype,
                compile=False,
                delete_blockwise_modules=False,
                token=config.HF_TOKEN
            )
            log.info(f"EchoDiT model loaded in {time.time() - start_time:.2f}s")

            log.info("Loading Fish Speech S1-DAC autoencoder...")
            ae_start = time.time()
            fish_ae = load_fish_ae_from_hf(
                device=self.device,
                dtype=self.torch_dtype,
                compile=False,
                token=config.HF_TOKEN
            )
            log.info(f"Autoencoder loaded in {time.time() - ae_start:.2f}s")

            log.info("Loading PCA state...")
            pca_start = time.time()
            pca_state = load_pca_state_from_hf(device=self.device, token=config.HF_TOKEN)
            log.info(f"PCA state loaded in {time.time() - pca_start:.2f}s")

            self._models.update({"model": model, "fish_ae": fish_ae, "pca_state": pca_state})

            total_time = time.time() - start_time
            log.info(f"All models loaded successfully in {total_time:.2f}s")
            return model, fish_ae, pca_state

        except Exception as e:
            log.error(f"Model loading failed: {str(e)}")
            raise

    def _build_sample_fn(self, params: Dict) -> callable:
        """Create sampler partial with defaults and overrides."""
        return partial(
            sample_euler_cfg_independent_guidances,
            num_steps=params.get("num_steps", config.DEFAULT_NUM_STEPS),
            cfg_scale_text=params.get("cfg_scale_text", config.DEFAULT_CFG_SCALE_TEXT),
            cfg_scale_speaker=params.get("cfg_scale_speaker", config.DEFAULT_CFG_SCALE_SPEAKER),
            cfg_min_t=params.get("cfg_min_t", config.DEFAULT_CFG_MIN_T),
            cfg_max_t=params.get("cfg_max_t", config.DEFAULT_CFG_MAX_T),
            truncation_factor=params.get("truncation_factor"),
            rescale_k=params.get("rescale_k"),
            rescale_sigma=params.get("rescale_sigma"),
            speaker_kv_scale=params.get("speaker_kv_scale"),
            speaker_kv_max_layers=params.get("speaker_kv_max_layers"),
            speaker_kv_min_t=params.get("speaker_kv_min_t"),
            sequence_length=params.get("sequence_length", config.DEFAULT_SEQUENCE_LENGTH),
        )

    def _load_speaker_audio(self, speaker_voice_name: str) -> Optional[torch.Tensor]:
        """Load speaker reference audio from voice directory."""
        if not speaker_voice_name:
            return None

        candidate_path = (config.AUDIO_VOICES_DIR / speaker_voice_name).resolve()
        if not str(candidate_path).startswith(str(config.AUDIO_VOICES_DIR.resolve())):
            raise ValueError("Invalid speaker_voice path")
        if not candidate_path.exists():
            raise ValueError(f"speaker_voice '{speaker_voice_name}' not found")
        if candidate_path.suffix.lower() not in config.AUDIO_EXTS:
            raise ValueError(f"Unsupported speaker_voice extension: {candidate_path.suffix}")

        return load_audio(str(candidate_path)).to(self.device)

    def generate_speech(self, text: str, speaker_voice: str = None, parameters: Dict = None) -> Tuple[torch.Tensor, int]:
        """Generate speech using Echo-TTS."""
        if parameters is None:
            parameters = {}

        model, fish_ae, pca_state = self._load_core_models()
        sample_fn = self._build_sample_fn(parameters)
        speaker_audio = self._load_speaker_audio(speaker_voice) if speaker_voice else None

        max_chars_per_chunk = int(parameters.get("max_chars_per_chunk", config.DEFAULT_MAX_CHARS_PER_CHUNK))
        enable_crossfade = parameters.get("enable_crossfade", True)
        normalize_boundaries = parameters.get("normalize_boundaries", True)
        crossfade_ms = int(parameters.get("crossfade_ms", config.DEFAULT_CROSSFADE_MS))
        target_duration = parameters.get("target_duration_seconds", config.DEFAULT_TARGET_DURATION_SECONDS)

        if max_chars_per_chunk and max_chars_per_chunk > 0:
            text_chunks = chunk_text_for_audio(text, max_chars=max_chars_per_chunk,
                                             target_duration_seconds=target_duration)
        else:
            text_chunks = [text]

        if not text_chunks:
            raise ValueError("Text is empty after normalization")

        seed = parameters.get("seed", 0)
        audio_chunks = []

        for idx, chunk in enumerate(text_chunks):
            chunk_seed = seed + (idx * 1000)
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

        if normalize_boundaries and len(audio_chunks) > 1:
            audio_out = normalize_chunk_boundaries(
                audio_chunks,
                sample_rate=44100,
                crossfade_ms=crossfade_ms
            )
        elif enable_crossfade and len(audio_chunks) > 1:
            audio_out = crossfade_chunks(audio_chunks, crossfade_ms=crossfade_ms)
        else:
            audio_out = torch.cat(audio_chunks, dim=-1)

        return audio_out, 44100

    def encode_to_linacodec(self, audio: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Encode audio to LinaCodec tokens."""
        if not LINACODEC_AVAILABLE:
            raise RuntimeError("LinaCodec is not available")

        lina = self._load_linacodec()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

            audio = audio.detach().cpu().squeeze()

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 0:
                audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() > 2:
                while audio.dim() > 2:
                    audio = audio[0]
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)

            torchaudio.save(tmp_wav_path, audio, 44100)
            tokens, embedding = lina.encode(tmp_wav_path)
            return tokens, embedding

        finally:
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)

    def generate_audio_stream_decoded(
        self,
        text: str,
        speaker_voice: str = None,
        parameters: Dict = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming audio with base64 encoded PCM chunks."""
        import base64
        import traceback
        if parameters is None:
            parameters = {}

        # Stream tuning knobs
        stream_chunk_seconds = parameters.get("stream_chunk_seconds")
        stream_crossfade_ms = parameters.get("stream_crossfade_ms", parameters.get("crossfade_ms", config.DEFAULT_CROSSFADE_MS))
        stream_tail_ms = parameters.get("stream_tail_ms", 0)
        stream_max_chars = int(parameters.get("stream_max_chars_per_chunk", parameters.get("max_chars_per_chunk", config.DEFAULT_MAX_CHARS_PER_CHUNK)))
        enable_crossfade = parameters.get("enable_crossfade", True)

        try:
            # If no streaming chunk size is provided, default to batch-tuned duration
            if stream_chunk_seconds is None:
                stream_chunk_seconds = parameters.get(
                    "target_duration_seconds",
                    config.DEFAULT_TARGET_DURATION_SECONDS,
                )

            # Build the same core pieces as batch for quality consistency
            model, fish_ae, pca_state = self._load_core_models()
            sample_fn = self._build_sample_fn(parameters)
            speaker_audio = self._load_speaker_audio(speaker_voice) if speaker_voice else None

            # Chunk text for streaming
            text_chunks = chunk_text_for_audio(
                text,
                max_chars=stream_max_chars,
                target_duration_seconds=float(stream_chunk_seconds),
            )

            if not text_chunks:
                yield {"error": "Text is empty after normalization"}
                return

            seed = parameters.get("seed", 0)
            sample_rate = 44100

            tail_samples = int(sample_rate * (float(stream_tail_ms) / 1000.0)) if stream_tail_ms else 0
            crossfade_samples = (
                int(sample_rate * (float(stream_crossfade_ms) / 1000.0))
                if stream_crossfade_ms and enable_crossfade
                else 0
            )

            buffer_audio = None
            chunk_num = 0

            for idx, chunk in enumerate(text_chunks):
                chunk_seed = seed + (idx * 1000)
                audio_chunk, _ = sample_pipeline(
                    model=model,
                    fish_ae=fish_ae,
                    pca_state=pca_state,
                    sample_fn=sample_fn,
                    text_prompt=chunk,
                    speaker_audio=speaker_audio,
                    rng_seed=chunk_seed,
                )

                # Normalize to 1D tensor
                if audio_chunk.dim() == 1:
                    chunk_tensor = audio_chunk
                elif audio_chunk.dim() > 1:
                    chunk_tensor = audio_chunk[0]
                else:
                    chunk_tensor = audio_chunk

                if chunk_tensor.dim() > 1:
                    chunk_tensor = chunk_tensor.reshape(-1)

                if buffer_audio is None:
                    merged = chunk_tensor
                else:
                    if crossfade_samples > 0:
                        cf = min(crossfade_samples, len(buffer_audio), len(chunk_tensor))
                        if cf > 0:
                            fade_out = torch.linspace(1.0, 0.0, cf, device=chunk_tensor.device, dtype=chunk_tensor.dtype)
                            fade_in = 1.0 - fade_out
                            blended = buffer_audio[-cf:] * fade_out + chunk_tensor[:cf] * fade_in
                            merged = torch.cat([buffer_audio[:-cf], blended, chunk_tensor[cf:]], dim=-1)
                        else:
                            merged = torch.cat([buffer_audio, chunk_tensor], dim=-1)
                    else:
                        merged = torch.cat([buffer_audio, chunk_tensor], dim=-1)

                if tail_samples > 0 and len(merged) > tail_samples:
                    emit_tensor = merged[:-tail_samples]
                    buffer_audio = merged[-tail_samples:]
                else:
                    emit_tensor = None
                    buffer_audio = merged

                if emit_tensor is not None and len(emit_tensor) > 0:
                    audio_array = emit_tensor.detach().cpu().numpy()
                    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                        audio_int16 = (audio_array * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio_array.astype(np.int16)

                    audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
                    chunk_num += 1
                    yield {
                        'status': 'streaming',
                        'chunk': chunk_num,
                        'format': 'pcm_16',
                        'audio_chunk': audio_b64,
                        'sample_rate': sample_rate,
                    }

            # Flush remaining buffer
            if buffer_audio is not None and len(buffer_audio) > 0:
                audio_array = buffer_audio.detach().cpu().numpy()
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_array.astype(np.int16)

                audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
                chunk_num += 1
                yield {
                    'status': 'streaming',
                    'chunk': chunk_num,
                    'format': 'pcm_16',
                    'audio_chunk': audio_b64,
                    'sample_rate': sample_rate,
                }

            yield {
                'status': 'complete',
                'format': 'pcm_16',
                'message': 'All chunks streamed',
                'total_chunks': chunk_num,
            }

        except Exception as e:
            error_trace = traceback.format_exc()
            log.error(f"Streaming mode failed: {str(e)}")
            log.error(f"Traceback: {error_trace}")
            yield {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_trace
            }

    def generate_linacodec_token_stream(
        self,
        text: str,
        speaker_voice: str = None,
        parameters: Dict = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming LinaCodec tokens."""
        if not LINACODEC_AVAILABLE:
            raise RuntimeError("LinaCodec streaming requires LinaCodec to be installed")

        if parameters is None:
            parameters = {}

        max_chars_per_chunk = int(parameters.get("max_chars_per_chunk", config.DEFAULT_MAX_CHARS_PER_CHUNK))
        chunk_num = 0
        total_tokens = 0
        start_time = time.time()

        if max_chars_per_chunk and max_chars_per_chunk > 0:
            text_chunks = chunk_text_for_audio(text, max_chars=max_chars_per_chunk,
                                             target_duration_seconds=parameters.get("target_duration_seconds", 10.0))
        else:
            text_chunks = [text]

        for chunk in text_chunks:
            chunk_num += 1
            chunk_start = time.time()

            chunk_params = dict(parameters) if parameters else {}
            chunk_audio, _ = self.generate_speech(chunk, speaker_voice, chunk_params)

            tokens, embedding = self.encode_to_linacodec(chunk_audio)

            encode_time = time.time() - chunk_start

            tokens_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            total_tokens += len(tokens_list)

            log.debug(f"Chunk {chunk_num}: {len(tokens_list)} tokens (encode: {encode_time:.3f}s)")

            yield {
                'status': 'streaming',
                'chunk': chunk_num,
                'format': 'linacodec_tokens',
                'tokens': tokens_list,
                'embedding': embedding_list,
                'sample_rate': 48000,
                'original_sample_rate': 44100,
                'num_tokens': len(tokens_list),
                'encode_time_ms': encode_time * 1000
            }

        elapsed = time.time() - start_time
        log.info(f"Stream complete: {chunk_num} chunks, {total_tokens} total tokens, {elapsed:.2f}s")

        yield {
            'status': 'complete',
            'format': 'linacodec_tokens',
            'message': 'All chunks streamed',
            'total_chunks': chunk_num,
            'total_tokens': total_tokens,
            'elapsed_time_seconds': elapsed
        }


# Singleton instance for RunPod serverless
_inference_engine = None


def get_inference_engine(device: str = None) -> EchoTTSInference:
    """Get or create singleton inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = EchoTTSInference(device=device)
    return _inference_engine
