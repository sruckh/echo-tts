# LINACODEC-BE.md
## Backend Implementation - RunPod Serverless LinaCodec Integration (Tier 3)

### 3-Tier System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 1: FRONTEND                             │
│  React/TypeScript Application (Port 4173)                          │
│  - User interface for TTS generation                               │
│  - Audio playback via Web Audio API                                │
│  - Streaming chunk consumption                                     │
└─────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ SSE (via Tier 2)
                                   │ Audio Chunks (48kHz PCM)
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 2: MIDDLEWARE                           │
│  Express Server (Local) + Cloudflare Workers                       │
│  - LinaCodec token decoding (local only)                          │
│  - Audio chunk streaming via SSE                                   │
│  - Service proxy and routing                                       │
└─────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ HTTP/WebSocket
                                   │ LinaCodec Tokens or Audio
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 3: BACKEND ◄── YOU ARE HERE            │
│  RunPod Serverless (GPU)                                           │
│                                                                     │
│  Responsibilities:                                                 │
│  - TTS audio generation (your TTS model)                           │
│  - LinaCodec token encoding (compression)                         │
│  - Streaming output via runpod.stream()                            │
│  - Model caching on network volume                                 │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  Network Attached Volume (Persistent Storage)                  ││
│  │                                                               ││
│  │  /runpod-volume/.cache/huggingface/                           ││
│  │  ├── YatharthS/LinaCodec/                                    ││
│  │  │   ├── config.json                                         ││
│  │  │   ├── model.safetensors (~200MB)                          ││
│  │  │   └── tokenizer/                                          ││
│  │  └── [Your TTS Model]/                                       ││
│  │                                                               ││
│  └────────────────────────────────────────────────────────────────┘│
│                          ↑                                        │
│                    Persists                                     │
│                 between cold starts                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phase

**This component is implemented in Phase 2** (after Tier 2 middleware is set up)

**Why Phase 2?**
- Tier 2 middleware provides test endpoint
- Can test RunPod independently with `curl`
- Streaming infrastructure proven before adding LinaCodec
- Tier 1 (Frontend) not needed for testing

**Prerequisites:**
- None (runs independently on RunPod)
- Optional: Tier 2 middleware endpoint for integration testing

**Dependencies:**
- RunPod Serverless GPU
- Network volume attached (for model caching)
- LinaCodec installed via requirements.txt

---

## Architecture

### Data Flow (Streaming Mode)

```
Tier 2 Middleware sends request
        ↓
RunPod: handler(event) called
        ↓
Load TTS model (cached in memory)
        ↓
Split text into chunks (500 chars each)
        ↓
For each chunk:
        ↓
    Generate audio (TTS model)
        ↓
    Encode to LinaCodec tokens
        ↓
    Yield tokens via runpod.stream()
        ↓
Tier 2 receives tokens → Decodes to audio → Streams to Tier 1
```

### Streaming vs Batch Modes

| Mode | When Used | Output | Use Case |
|------|-----------|--------|----------|
| **Streaming (LinaCodec tokens)** | `stream=true, output_format='linacodec_tokens'` | LinaCodec tokens | Tier 2 local middleware (EchoTTS) |
| **Streaming (Audio chunks)** | `stream=true, output_format='pcm_16'` | Decoded audio | Tier 2 Cloudflare Worker (Vibe/Chatterbox) |
| **Batch** | `stream=false` | Complete audio file | Original behavior (backward compatible) |

---

## File: `handler.py` (Modify Existing)

Your RunPod Serverless handler with LinaCodec streaming support.

```python
"""
RunPod Serverless Handler with LinaCodec Streaming (Tier 3)

Generates audio in chunks, encodes to LinaCodec tokens,
and streams tokens as they're generated.

Storage:
- LinaCodec model: Auto-downloaded to network volume HF cache
- Your TTS model: Stored on network volume
- Both persist between cold starts

Streaming:
- Phase 2A: Audio chunks (no LinaCodec) - test streaming infrastructure
- Phase 2B: LinaCodec tokens - full implementation
"""

import os
import sys
import json
import base64
import time
import runpod
from runpod.serverless.utils import rp_debug, rp_log
import torch
import numpy as np
from typing import Generator, Dict, Any, Tuple, List

# =============================================================================
# LINACODEC IMPORTS (Phase 2B)
# =============================================================================
try:
    from linacodec.codec import LinaCodec
    LINACODEC_AVAILABLE = True
    rp_log("LinaCodec is available")
except ImportError:
    LINACODEC_AVAILABLE = False
    rp_log("Warning: LinaCodec not available. Install with:")
    rp_log("pip install git+https://github.com/ysharma3501/LinaCodec.git")

# =============================================================================
# GLOBAL STATE (CACHED ACROSS REQUESTS FOR PERFORMANCE)
# =============================================================================
# These are cached in GPU memory between requests on the same pod

TTS_MODEL = None
LINA_CODEC = None
DEVICE = None

# =============================================================================
# MODEL LOADING
# =============================================================================

def get_device():
    """Get the best available device (GPU or CPU)"""
    global DEVICE
    if DEVICE is None:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rp_log(f"[Tier 3] Using device: {DEVICE}")
    return DEVICE

def load_tts_model(model_name: str = "default"):
    """
    Load TTS model (cached globally)

    TODO: Replace with your actual TTS model loading code.
    Examples: XTTS, YourTTS, Coqui TTS, Bark, etc.
    """
    global TTS_MODEL

    if TTS_MODEL is not None:
        rp_log("[Tier 3] Using cached TTS model")
        return TTS_MODEL

    rp_log("[Tier 3] Loading TTS model...")

    # TODO: Replace with your actual TTS model
    # Example for XTTS:
    # from TTS.api import TTS
    # TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

    # Example for HuggingFace model:
    # from transformers import pipeline
    # TTS_MODEL = pipeline("text-to-speech", model=model_name, device=DEVICE)

    # Placeholder - replace with your actual model
    TTS_MODEL = f"MockTTSModel({model_name})"

    rp_log("[Tier 3] TTS model loaded!")
    return TTS_MODEL

def load_linacodec():
    """
    Load LinaCodec encoder/decoder (cached globally)

    Model is auto-downloaded from HuggingFace on first use
    and cached to network volume.
    """
    global LINA_CODEC

    if LINA_CODEC is not None:
        rp_log("[Tier 3] Using cached LinaCodec model")
        return LINA_CODEC

    if not LINACODEC_AVAILABLE:
        raise RuntimeError("LinaCodec is not installed")

    rp_log("[Tier 3] Loading LinaCodec model from network volume...")

    # Ensure cache is on network volume
    os.environ['HF_HOME'] = '/runpod-volume/.cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/runpod-volume/.cache/huggingface'

    # LinaCodec auto-downloads from HuggingFace to cache
    LINA_CODEC = LinaCodec()

    rp_log("[Tier 3] LinaCodec loaded and cached to network volume!")
    return LINA_CODEC

# =============================================================================
# TEXT CHUNKING
# =============================================================================

def split_text_for_streaming(text: str, max_chars: int = 500) -> List[str]:
    """
    Split text into chunks for streaming generation.

    Args:
        text: Input text to split
        max_chars: Maximum characters per chunk (default: 500)

    Returns:
        List of text chunks

    Note: Adjust chunking strategy based on your TTS model's behavior.
    Some models work better with sentence boundaries, others with fixed lengths.
    """
    chunks = []

    # Split into paragraphs first
    paragraphs = text.split('\n\n')

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        # Split paragraph into sentences
        sentences = []
        current = ""
        for char in paragraph:
            current += char
            if char in '.!?;' and len(current) > 50:
                sentences.append(current.strip())
                current = ""
        if current:
            sentences.append(current.strip())

        # Group sentences into chunks
        current_chunk = ""
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks

# =============================================================================
# AUDIO GENERATION
# =============================================================================

def generate_audio_chunk(text: str, voice: str, model: Any) -> np.ndarray:
    """
    Generate audio for a single text chunk.

    Args:
        text: Text to synthesize
        voice: Voice ID/name
        model: TTS model instance

    Returns:
        Audio array (float32, typically 24kHz)

    TODO: Replace with your actual TTS generation code.
    """
    rp_debug(f"[Tier 3] Generating audio for: {text[:50]}...")

    # TODO: Replace with your actual TTS generation
    # Example for XTTS:
    # audio = model.tts(text=text, speaker_wav=voice_path, language="en")

    # Example for API-based model:
    # response = requests.post(api_url, json={text, voice})
    # audio = np.array(response.json()['audio'])

    # Placeholder: generate silence for testing
    duration = len(text) * 0.08  # Rough estimate
    samples = int(duration * 24000)  # 24kHz
    audio = np.random.randn(samples).astype(np.float32) * 0.1  # Low noise for testing

    return audio

def generate_audio_stream(
    text: str,
    voice: str,
    model: Any,
    chunk_size: int = 500
) -> Generator[np.ndarray, None, None]:
    """
    Generate audio in chunks using the TTS model.

    Yields audio chunks as they're generated.

    Args:
        text: Full text to synthesize
        voice: Voice ID/name
        model: TTS model
        chunk_size: Max characters per chunk

    Yields:
        Audio arrays (float32)
    """
    text_chunks = split_text_for_streaming(text, chunk_size)

    rp_log(f"[Tier 3] Split text into {len(text_chunks)} chunks")

    for i, chunk in enumerate(text_chunks):
        rp_debug(f"[Tier 3] Generating chunk {i+1}/{len(text_chunks)}")

        audio = generate_audio_chunk(chunk, voice, model)

        yield audio

# =============================================================================
# LINACODEC ENCODING (Phase 2B)
# =============================================================================

def encode_to_linacodec(audio: np.ndarray) -> Tuple[Any, Any]:
    """
    Encode audio to LinaCodec tokens.

    Args:
        audio: Audio array (float32, any sample rate)

    Returns:
        Tuple of (tokens, global_embedding)

    Note: LinaCodec automatically upsamples to 48kHz during encoding.
    """
    if not LINACODEC_AVAILABLE:
        raise RuntimeError("LinaCodec is not available")

    lina = load_linacodec()

    # Ensure audio is numpy array on CPU
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    # Encode to LinaCodec tokens
    # Returns: (tokens, global_embedding)
    tokens, embedding = lina.encode(audio)

    return tokens, embedding

# =============================================================================
# STREAMING GENERATORS
# =============================================================================

def generate_linacodec_token_stream(
    text: str,
    voice: str,
    model: Any
) -> Generator[Dict[str, Any], None, None]:
    """
    Phase 2B: Generate streaming LinaCodec tokens.

    This is the optimal path for local middleware (EchoTTS):
    - Generate audio chunk
    - Encode to LinaCodec tokens
    - Yield tokens immediately

    Benefits:
    - ~60x payload reduction (12.5 tokens/sec vs full audio)
    - 48kHz output quality
    - Faster transmission to Tier 2

    Yields dictionaries with LinaCodec token data.
    """
    if not LINACODEC_AVAILABLE:
        raise RuntimeError("LinaCodec streaming requires LinaCodec to be installed")

    lina = load_linacodec()
    chunk_num = 0
    total_tokens = 0
    start_time = time.time()

    for audio_chunk in generate_audio_stream(text, voice, model):
        chunk_num += 1
        chunk_start = time.time()

        # Encode to LinaCodec tokens
        tokens, embedding = encode_to_linacodec(audio_chunk)

        encode_time = time.time() - chunk_start

        # Convert to list for JSON serialization
        tokens_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

        total_tokens += len(tokens_list)

        rp_debug(f"[Tier 3] Chunk {chunk_num}: {len(tokens_list)} tokens (encode: {encode_time:.3f}s)")

        # Yield token chunk
        yield {
            'status': 'streaming',
            'chunk': chunk_num,
            'format': 'linacodec_tokens',
            'tokens': tokens_list,
            'embedding': embedding_list,
            'sample_rate': 48000,
            'original_sample_rate': 24000,
            'num_tokens': len(tokens_list),
            'encode_time_ms': encode_time * 1000
        }

    elapsed = time.time() - start_time
    rp_log(f"[Tier 3] Stream complete: {chunk_num} chunks, {total_tokens} total tokens, {elapsed:.2f}s")

    # Final completion signal
    yield {
        'status': 'complete',
        'format': 'linacodec_tokens',
        'message': 'All chunks streamed',
        'total_chunks': chunk_num,
        'total_tokens': total_tokens,
        'elapsed_time_seconds': elapsed
    }

def generate_audio_stream_decoded(
    text: str,
    voice: str,
    model: Any
) -> Generator[Dict[str, Any], None, None]:
    """
    Phase 2A: Generate streaming audio with LinaCodec compression then decode.

    This is the compatibility mode for Cloudflare Workers:
    - Generate audio chunk
    - Encode to LinaCodec tokens (compression)
    - Decode back to audio (quality check + 48kHz upsampling)
    - Yield decoded audio chunk

    This gives us the quality benefits of LinaCodec (48kHz output) while
    outputting standard PCM that doesn't require browser decoding.

    Use for:
    - Cloudflare Worker services (Vibe Voice, Chatterbox)
    - Testing streaming infrastructure before LinaCodec
    """
    if not LINACODEC_AVAILABLE:
        # Fallback: stream raw audio without LinaCodec
        rp_log("[Tier 3] LinaCodec not available, streaming raw audio")

        for audio_chunk in generate_audio_stream(text, voice, model):
            audio_b64 = base64.b64encode(audio_chunk.tobytes()).decode('utf-8')
            yield {
                'status': 'streaming',
                'format': 'pcm_24',
                'audio_chunk': audio_b64,
                'sample_rate': 24000
            }

        yield {
            'status': 'complete',
            'format': 'pcm_24',
            'message': 'All chunks streamed (no LinaCodec)'
        }
        return

    lina = load_linacodec()
    chunk_num = 0
    start_time = time.time()

    for audio_chunk in generate_audio_stream(text, voice, model):
        chunk_num += 1
        chunk_start = time.time()

        # Encode to LinaCodec tokens
        tokens, embedding = encode_to_linacodec(audio_chunk)

        # Decode back to audio (now at 48kHz!)
        decoded_audio = lina.decode(tokens, embedding)

        process_time = time.time() - chunk_start

        # Convert to base64 for transmission
        audio_array = decoded_audio.cpu().numpy() if hasattr(decoded_audio, 'cpu') else decoded_audio
        audio_b64 = base64.b64encode(audio_array.tobytes()).decode('utf-8')

        rp_debug(f"[Tier 3] Chunk {chunk_num}: {len(audio_array)} samples (process: {process_time:.3f}s)")

        yield {
            'status': 'streaming',
            'chunk': chunk_num,
            'format': 'pcm_16',  # 16-bit PCM at 48kHz
            'audio_chunk': audio_b64,
            'sample_rate': 48000,
            'process_time_ms': process_time * 1000
        }

    elapsed = time.time() - start_time
    rp_log(f"[Tier 3] Stream complete: {chunk_num} chunks, {elapsed:.2f}s")

    yield {
        'status': 'complete',
        'format': 'pcm_16',
        'message': 'All chunks streamed',
        'total_chunks': chunk_num,
        'elapsed_time_seconds': elapsed
    }

# =============================================================================
# HANDLERS
# =============================================================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler for RunPod Serverless.

    Supports both streaming and batch modes.

    Args:
        event: RunPod event with 'input' key

    Returns:
        Response dict or generator for streaming
    """
    input_data = event.get('input', {})

    # Extract parameters
    text = input_data.get('text', '')
    voice = input_data.get('voice', 'default')
    service = input_data.get('service', 'default')
    stream = input_data.get('stream', False)
    output_format = input_data.get('output_format', 'linacodec_tokens')

    rp_log(f"[Tier 3] Handler called: service={service}, stream={stream}, format={output_format}")
    rp_log(f"[Tier 3] Text length: {len(text)} chars")

    # Validate input
    if not text:
        return {
            'error': 'No text provided',
            'status': 'error'
        }

    # Load model
    model = load_tts_model(service)

    # Routing based on mode
    if stream:
        if output_format == 'pcm_16':
            # Stream decoded audio chunks (for Cloudflare Workers)
            rp_log("[Tier 3] Streaming decoded audio (pcm_16)")
            return runpod.stream(generate_audio_stream_decoded(text, voice, model))
        else:
            # Stream LinaCodec tokens (for local middleware)
            rp_log("[Tier 3] Streaming LinaCodec tokens")
            return runpod.stream(generate_linacodec_token_stream(text, voice, model))
    else:
        # Batch mode (original behavior)
        return generate_batch_audio(text, voice, model, output_format)

def generate_batch_audio(
    text: str,
    voice: str,
    model: Any,
    output_format: str
) -> Dict[str, Any]:
    """
    Generate complete audio in batch mode (original behavior).

    Returns the full audio file at once.

    Args:
        text: Text to synthesize
        voice: Voice ID/name
        model: TTS model
        output_format: 'linacodec_tokens' or 'pcm_24'

    Returns:
        Dict with complete audio data
    """
    rp_log("[Tier 3] Batch mode: generating complete audio")

    # Generate all audio chunks
    audio_chunks = list(generate_audio_stream(text, voice, model))
    full_audio = np.concatenate(audio_chunks)

    rp_log(f"[Tier 3] Generated {len(full_audio)} samples")

    # Encode based on requested format
    if output_format == 'linacodec_tokens':
        if not LINACODEC_AVAILABLE:
            return {
                'error': 'LinaCodec not available',
                'status': 'error'
            }

        tokens, embedding = encode_to_linacodec(full_audio)

        return {
            'status': 'complete',
            'format': 'linacodec_tokens',
            'tokens': tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens),
            'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            'sample_rate': 48000,
            'duration': len(full_audio) / 24000  # Assuming 24kHz input
        }

    else:
        # Return raw audio (base64 encoded)
        audio_b64 = base64.b64encode(full_audio.tobytes()).decode('utf-8')

        return {
            'status': 'complete',
            'format': 'pcm_24',  # Assuming 24kHz float32
            'audio': audio_b64,
            'sample_rate': 24000,
            'duration': len(full_audio) / 24000
        }

# =============================================================================
# RUNPOD SERVERLESS START
# =============================================================================

if __name__ == '__main__':
    rp_log("[Tier 3] Starting RunPod Serverless handler...")
    runpod.serverless.start({"handler": handler})
```

---

## `requirements.txt` (Update)

Add to your RunPod requirements:

```txt
# Core dependencies
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
runpod>=1.5.0

# LinaCodec (install from GitHub)
git+https://github.com/ysharma3501/LinaCodec.git

# Your TTS model dependencies
# TODO: Add your specific TTS model requirements
# Examples:
# TTS>=0.22.0  # Coqui TTS
# transformers>=4.30.0  # HuggingFace models
# libsndfile==1.2.2  # Audio processing
```

---

## `Dockerfile` (For RunPod Template)

```dockerfile
FROM runpod/base:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Create cache directories on network volume
RUN mkdir -p /runpod-volume/.cache/huggingface

# Set environment
ENV PYTHONPATH=/app
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface

# Health check
EXPOSE 8080

# Run handler (handled by RunPod)
CMD ["python3", "-u", "handler.py"]
```

---

## Testing Strategy

### Phase 2A: Test Audio Chunk Streaming (No LinaCodec)

Test streaming infrastructure without LinaCodec complexity:

```bash
# Test streaming with decoded audio
curl -X POST $RUNPOD_ENDPOINT \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "text": "This is a test of streaming audio chunks from RunPod.",
    "voice": "default",
    "stream": true,
    "output_format": "pcm_16"
  }' \
  --output test_stream.json

# Should return JSON lines:
# {"status":"streaming","chunk":1,"audio_chunk":"base64data...","sample_rate":48000}
# {"status":"streaming","chunk":2,"audio_chunk":"base64data...","sample_rate":48000}
# {"status":"complete",...}
```

### Phase 2B: Test LinaCodec Token Streaming

Test with LinaCodec encoding:

```bash
# Test streaming with LinaCodec tokens
curl -X POST $RUNPOD_ENDPOINT \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "text": "Testing LinaCodec token streaming from RunPod.",
    "voice": "default",
    "stream": true,
    "output_format": "linacodec_tokens"
 }' \
  --output test_tokens.json

# Should return JSON lines:
# {"status":"streaming","chunk":1,"tokens":[...],"embedding":[...],"sample_rate":48000}
# {"status":"streaming","chunk":2,"tokens":[...],"embedding":[...],"sample_rate":48000}
# {"status":"complete",...}
```

### Local Testing (Before RunPod Deployment)

```python
# test_handler.py - Local testing script
import sys
sys.path.append('.')

from handler import generate_audio_stream, encode_to_linacodec
import numpy as np

# Test 1: Audio generation
print("Test 1: Generating audio chunks...")
for i, chunk in enumerate(generate_audio_stream("Hello world", "default", None)):
    print(f"  Chunk {i+1}: {len(chunk)} samples")

# Test 2: LinaCodec encoding
print("\nTest 2: Encoding to LinaCodec...")
audio = np.random.randn(24000).astype(np.float32)  # 1 second
tokens, embedding = encode_to_linacodec(audio)
print(f"  Encoded {len(audio)} samples to {len(tokens)} tokens")

# Test 3: Full stream
print("\nTest 3: Full token stream...")
for chunk in generate_linacodec_token_stream("This is a test.", "default", None):
    if chunk['status'] == 'streaming':
        print(f"  Chunk {chunk['chunk']}: {chunk['num_tokens']} tokens")
    elif chunk['status'] == 'complete':
        print(f"  Complete: {chunk['total_chunks']} chunks, {chunk['total_tokens']} tokens")
```

---

## Performance Tuning

### Chunk Size Analysis

| Text Chunk | Audio Duration | LinaCodec Tokens | Token Size (bytes) | Latency Impact |
|------------|----------------|------------------|-------------------|----------------|
| 100 chars | ~8 sec | ~100 tokens | ~400 | Very low |
| 250 chars | ~20 sec | ~250 tokens | ~1 KB | Low |
| 500 chars | ~40 sec | ~500 tokens | ~2 KB | Medium (recommended) |
| 1000 chars | ~80 sec | ~1000 tokens | ~4 KB | Higher |

**Recommendation**: 500 character chunks balance latency and efficiency.

### Memory Usage

Per-request memory estimates:
- TTS Model: ~500MB-2GB (cached globally)
- LinaCodec Model: ~200MB (cached globally)
- Per-chunk audio buffer: ~1-5MB
- Token buffer: <1MB

**Total**: ~1-4GB GPU memory recommended for optimal performance.

### GPU Utilization

For optimal streaming:

1. **Batch chunks**: Process 2-4 chunks in parallel if GPU allows
2. **Cache embeddings**: Reuse voice embeddings across chunks
3. **Async generation**: Start next chunk while streaming current

---

## Network Volume Setup

### Create Network Volume in RunPod

```bash
# Via RunPod CLI
runpodctl create volume linacodec-models --size 20GB

# Or via Web UI
# Navigate: Volumes → Create Volume
# Name: linacodec-models
# Size: 20GB
# Data center: Same as your pods
```

### Attach Volume to Pod

In your RunPod Serverless template configuration:

```yaml
volumeMounts:
  - mountPath: /runpod-volume
    name: linacodec-models

volumes:
  - name: linacodec-models
    volumeId: vol-xxxxxxxxxxxxx  # Your volume ID
```

### Verify Cache Location

```python
import os

# In handler.py
os.environ['HF_HOME'] = '/runpod-volume/.cache/huggingface'

# Verify
print(f"Cache directory: {os.environ['HF_HOME']}")
# Should show files being cached
import subprocess
result = subprocess.run(['ls', '-la', '/runpod-volume/.cache/huggingface'], capture_output=True)
print(result.stdout.decode())
```

---

## Troubleshooting

### Issue: "LinaCodec not available"

**Solution**:
```bash
# Reinstall in container
pip install --upgrade git+https://github.com/ysharma3501/LinaCodec.git

# Verify installation
python -c "from linacodec.codec import LinaCodec; print('OK')"
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce chunk size: `split_text_for_streaming(text, max_chars=250)`
2. Use CPU fallback: `DEVICE = torch.device('cpu')`
3. Clear cache: `torch.cuda.empty_cache()`

### Issue: Slow token generation

**Debug**:
```python
import time

start = time.time()
tokens, embedding = encode_to_linacodec(audio)
encode_time = time.time() - start
print(f"Encode time: {encode_time:.3f}s for {len(audio)} samples")
```

If encoding is slow, check:
- GPU utilization: `nvidia-smi`
- Model is on GPU: `model.device`
- Batch size is appropriate

### Issue: Stream hangs/timeout

**Add timeout handling**:
```python
import time
timeout = 300  # 5 minutes
start_time = time.time()

for audio_chunk in generate_audio_stream(...):
    if time.time() - start_time > timeout:
        raise TimeoutError("Generation timeout")
```

---

## Monitoring

Add metrics to track streaming performance:

```python
def generate_linacodec_token_stream(...):
    start_time = time.time()
    chunk_times = []
    token_counts = []

    for audio_chunk in generate_audio_stream(...):
        chunk_start = time.time()

        tokens, embedding = encode_to_linacodec(audio_chunk)
        chunk_time = time.time() - chunk_start

        chunk_times.append(chunk_time)
        token_counts.append(len(tokens))

        rp_debug({
            'event': 'chunk_encoded',
            'chunk': chunk_num,
            'tokens': len(tokens),
            'latency_ms': chunk_time * 1000,
            'tokens_per_second': len(tokens) / chunk_time if chunk_time > 0 else 0
        })

        yield {...}

    total_time = time.time() - start_time

    rp_log({
        'event': 'stream_complete',
        'total_chunks': chunk_num,
        'total_time_seconds': total_time,
        'avg_chunk_latency_ms': sum(chunk_times) / len(chunk_times) * 1000 if chunk_times else 0,
        'avg_tokens_per_chunk': sum(token_counts) / len(token_counts) if token_counts else 0,
        'total_tokens': sum(token_counts)
    })
```

---

## Rollback Plan

If streaming causes issues:

1. **Fallback to batch**: Set `stream=false` in requests
2. **Disable LinaCodec**: Use `output_format='pcm_24'` instead
3. **No breaking changes**: Existing batch mode unchanged
4. **Quick rollback**: Remove LinaCodec from requirements

---

## Next Steps After Implementation

1. ✅ Test audio chunk streaming (Phase 2A)
2. ✅ Test LinaCodec token streaming (Phase 2B)
3. ✅ Verify model caching to network volume
4. ✅ Performance testing (GPU utilization, throughput)
5. ✅ Integration testing with Tier 2 middleware
6. ✅ A/B test quality (with/without LinaCodec)
