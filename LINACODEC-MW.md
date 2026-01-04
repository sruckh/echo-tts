# LINACODEC-MW.md
## Middleware Implementation - LinaCodec Decoding & SSE Streaming (Tier 2)

### 3-Tier System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 1: FRONTEND                             │
│  React/TypeScript Application (Port 4173)                          │
│  - User interface for TTS generation                               │
│  - Audio playback via Web Audio API                                │
│  - Streaming chunk consumption                                     │
│  - No LinaCodec decoding (handled here in Tier 2)                  │
└─────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ SSE (Server-Sent Events)
                                   │ Audio Chunks (48kHz PCM)
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 2: MIDDLEWARE ◄── YOU ARE HERE         │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Part A: Express Server (Local Container)                     │ │
│  │  - Port 4173                                                  │ │
│  │  - Handles EchoTTS requests                                  │ │
│  │  - Python subprocess for LinaCodec decoding                   │ │
│  │  - Streams audio chunks via SSE                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Part B: Cloudflare Workers                                   │ │
│  │  - Handles Vibe Voice & Chatterbox requests                  │ │
│  │  - Simple forwarding (no LinaCodec at edge)                  │ │
│  │  - RunPod decodes tokens to audio                            │ │
│  │  - Streams audio chunks via SSE                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Responsibilities:                                                 │
│  - Receive requests from Tier 1 (Frontend)                        │
│  - Proxy to Tier 3 (RunPod Serverless)                           │
│  - LinaCodec decoding (local container only)                     │
│  - Audio chunk streaming via SSE to Tier 1                       │
│  - Service routing and fallback                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTP/WebSocket
                                   │ LinaCodec Tokens (local) or Audio (CF)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 3: BACKEND                              │
│  RunPod Serverless (GPU)                                           │
│  - TTS audio generation                                            │
│  - LinaCodec token encoding                                        │
│  - Streaming output via runpod.stream()                            │
│  - Model caching on network volume                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phase

**This component is implemented in Phase 1** (START HERE - first to implement)

**Why Phase 1?**
- Easiest to test locally with `curl`
- Can mock Tier 3 (RunPod) responses initially
- No Cloudflare deployment needed for Part A
- Foundation for Tier 1 (Frontend) to test against

**Dependencies:**
- None for initial setup
- Tier 3 (RunPod) needed for end-to-end testing (can use existing batch endpoint)

---

## Part A: Local Container Middleware (EchoTTS) - Phase 1

### Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Tier 1        │      │   Tier 2 (YOU)   │      │   Tier 3        │
│  Frontend       │─────▶│  Express Server  │─────▶│  RunPod         │
│                 │ SSE  │                  │ HTTP  │                 │
│  Web Audio      │◀────│  LinaCodec       │◀─────│  TTS Model      │
│  Playback       │Audio │  Decoder         │Tokens│                 │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                │
                                │ Python subprocess
                                ▼
                         LinaCodec.decode()
                           (48kHz output)
```

### Files to Modify

#### 1. Modify `server.js`

Add streaming endpoint after the STT endpoints. This is the **main entry point** for Tier 1 streaming requests.

```javascript
// ============================================================================
// STREAMING TTS ENDPOINT (Phase 1)
// ============================================================================

/**
 * POST /api/tts/stream
 *
 * Tier 2 streaming endpoint for EchoTTS service.
 *
 * Flow:
 * 1. Receive streaming request from Tier 1 (Frontend)
 * 2. Proxy to Tier 3 (RunPod) with stream=true
 * 3. Receive LinaCodec tokens from RunPod
 * 4. Decode tokens using Python subprocess
 * 5. Stream 48kHz audio chunks to Tier 1 via SSE
 *
 * Environment Variables Required:
 * - ECHOTTS_STREAMING_ENDPOINT: RunPod streaming endpoint
 * - RUNPOD_API_KEY: RunPod API key
 * - ENABLE_STREAMING: Set to 'true' to enable (feature flag)
 */

const ENABLE_STREAMING = process.env.ENABLE_STREAMING === 'true';

app.post('/api/tts/stream', async (req, res) => {
  const requestId = Math.random().toString(36).substring(7);
  const startTime = Date.now();

  console.log(`[Tier 2][${requestId}] Streaming TTS request received`);
  console.log(`[Tier 2][${requestId}] Service: ${req.body.service}`);
  console.log(`[Tier 2][${requestId}] Text length: ${req.body.text?.length} chars`);

  // Feature flag check
  if (!ENABLE_STREAMING) {
    console.log(`[Tier 2][${requestId}] Streaming disabled via ENABLE_STREAMING flag`);
    return res.status(503).json({
      error: 'Streaming is disabled',
      message: 'Set ENABLE_STREAMING=true to enable'
    });
  }

  // Set headers for SSE streaming (before any body write)
  res.setHeader('Content-Type', 'audio/octet-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');  // Disable Nginx buffering
  res.flushHeaders();  // Send headers immediately

  try {
    const { text, voice, service } = req.body;

    // Validation
    if (!text || typeof text !== 'string') {
      throw new Error('Invalid or missing text parameter');
    }

    if (!voice) {
      throw new Error('Voice parameter is required');
    }

    if (service !== 'echotts') {
      console.log(`[Tier 2][${requestId}] Wrong service for this endpoint: ${service}`);
      return res.status(400).json({
        error: 'Invalid service',
        message: 'This endpoint handles EchoTTS only'
      });
    }

    // Get RunPod streaming endpoint
    const runpodEndpoint = process.env.ECHOTTS_STREAMING_ENDPOINT;
    const apiKey = process.env.RUNPOD_API_KEY;

    if (!runpodEndpoint) {
      throw new Error('ECHOTTS_STREAMING_ENDPOINT not configured');
    }

    if (!apiKey) {
      throw new Error('RUNPOD_API_KEY not configured');
    }

    console.log(`[Tier 2][${requestId}] Connecting to Tier 3 (RunPod)...`);

    // Connect to Tier 3 (RunPod) streaming endpoint
    const runpodResponse = await fetch(runpodEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        text,
        voice,
        service: 'echotts',
        stream: true,
        output_format: 'linacodec_tokens'  // Request LinaCodec token stream
      })
    });

    if (!runpodResponse.ok) {
      const errorText = await runpodResponse.text();
      throw new Error(`Tier 3 error ${runpodResponse.status}: ${errorText}`);
    }

    console.log(`[Tier 2][${requestId}] Tier 3 connected, receiving token stream...`);

    // Read LinaCodec token stream from Tier 3
    const reader = runpodResponse.body.getReader();
    let totalTokens = 0;
    let totalChunks = 0;
    let totalAudioBytes = 0;

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        const elapsed = Date.now() - startTime;
        console.log(`[Tier 2][${requestId}] Stream complete:`);
        console.log(`  - Total chunks: ${totalChunks}`);
        console.log(`  - Total tokens: ${totalTokens}`);
        console.log(`  - Audio output: ${totalAudioBytes} bytes`);
        console.log(`  - Elapsed time: ${elapsed}ms`);
        break;
      }

      // Parse token chunk (JSON format from Tier 3)
      const chunkText = new TextDecoder().decode(value);

      try {
        const tokenChunk = JSON.parse(chunkText);

        if (tokenChunk.status === 'complete') {
          console.log(`[Tier 2][${requestId}] Tier 3 signaled completion`);
          break;
        }

        totalTokens += tokenChunk.tokens?.length || 0;

        // Decode LinaCodec tokens to 48kHz audio
        const decodeStart = Date.now();
        const audioChunk = await decodeTokensToAudio(tokenChunk.tokens, tokenChunk.embedding);
        const decodeTime = Date.now() - decodeStart;

        totalAudioBytes += audioChunk.length;
        totalChunks++;

        console.log(`[Tier 2][${requestId}] Chunk ${totalChunks}: ${audioChunk.length} bytes (decode: ${decodeTime}ms)`);

        // Stream audio chunk to Tier 1
        res.write(audioChunk);

      } catch (parseError) {
        // Might be multiple JSON objects in one chunk, handle gracefully
        console.warn(`[Tier 2][${requestId}] Parse error, continuing: ${parseError.message}`);
      }
    }

    res.end();

  } catch (error) {
    const elapsed = Date.now() - startTime;
    console.error(`[Tier 2][${requestId}] Error after ${elapsed}ms:`, error);

    if (!res.headersSent) {
      res.status(500).json({
        error: error.message,
        tier: 'middleware',
        requestId
      });
    } else {
      // Can't send error response, stream already started
      console.error(`[Tier 2][${requestId}] Cannot send error response (headers already sent)`);
    }
    res.end();
  }
});

/**
 * Decode LinaCodec tokens to 48kHz PCM audio
 *
 * Uses Python subprocess with LinaCodec library.
 * Python script handles the actual decoding.
 *
 * @param {Array|Object} tokens - LinaCodec tokens
 * @param {Array|Object} embedding - Global embedding vector
 * @returns {Promise<Buffer>} - Audio buffer (16-bit PCM, 48kHz, mono)
 */
async function decodeTokensToAudio(tokens, embedding) {
  return new Promise((resolve, reject) => {
    const { spawn } = require('child_process');

    console.log(`[Tier 2] Spawning Python decoder...`);

    const decoder = spawn('python3', [
      '/app/scripts/decode_linacodec.py'
    ], {
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 10000  // 10 second timeout
    });

    // Prepare input data
    const inputData = JSON.stringify({ tokens, embedding });

    // Send tokens to decoder via stdin
    decoder.stdin.write(inputData);
    decoder.stdin.end();

    // Collect audio output
    const audioChunks = [];
    let errorOutput = '';

    decoder.stdout.on('data', (chunk) => {
      audioChunks.push(chunk);
    });

    decoder.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    decoder.on('close', (code) => {
      if (code !== 0) {
        console.error(`[Tier 2] LinaCodec decode failed (exit ${code}): ${errorOutput}`);
        reject(new Error(`LinaCodec decode error: ${errorOutput}`));
      } else {
        if (errorOutput) {
          console.log(`[Tier 2] Decoder warning: ${errorOutput}`);
        }
        resolve(Buffer.concat(audioChunks));
      }
    });

    decoder.on('error', (error) => {
      console.error(`[Tier 2] Failed to spawn decoder: ${error}`);
      reject(new Error(`Decoder spawn error: ${error.message}`));
    });
  });
}
```

#### 2. Create `scripts/decode_linacodec.py`

New Python script for LinaCodec decoding.

```python
#!/usr/bin/env python3
"""
LinaCodec Token Decoder (Tier 2 Component)

Decodes LinaCodec tokens to 48kHz PCM audio for streaming.

Usage:
  echo '{"tokens":[...],"embedding":[...]}' | python3 decode_linacodec.py > audio.pcm

Input: JSON via stdin with tokens and embedding
Output: 16-bit PCM audio to stdout (48kHz, mono, little-endian)

Environment:
  - Runs in Node.js container subprocess
  - LinaCodec model cached in ~/.cache/huggingface
"""

import sys
import json
import io

# =============================================================================
# LINACODEC IMPORTS
# =============================================================================
try:
    from linacodec.codec import LinaCodec
    LINACODEC_AVAILABLE = True
except ImportError:
    LINACODEC_AVAILABLE = False
    print("Error: LinaCodec not installed.", file=sys.stderr)
    print("Install with: pip install git+https://github.com/ysharma3501/LinaCodec.git", file=sys.stderr)
    sys.exit(1)

# =============================================================================
# GLOBAL STATE (CACHED ACROSS CALLS FOR PERFORMANCE)
# =============================================================================
# Note: Each subprocess invocation starts fresh, but model loads from disk cache
_decoder = None

def get_decoder():
    """Get or create LinaCodec decoder instance (cached in process)"""
    global _decoder
    if _decoder is None:
        print("[Tier 2][Python] Loading LinaCodec model...", file=sys.stderr)
        _decoder = LinaCodec()
        print("[Tier 2][Python] LinaCodec loaded!", file=sys.stderr)
    return _decoder

def decode_tokens(tokens, embedding):
    """
    Decode LinaCodec tokens to audio

    Args:
        tokens: List or array of LinaCodec tokens
        embedding: Global embedding vector

    Returns:
        bytes: 16-bit PCM audio (48kHz, mono, little-endian)
    """
    decoder = get_decoder()

    # Convert to numpy arrays if needed
    import numpy as np

    if isinstance(tokens, list):
        tokens = np.array(tokens)
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Decode tokens to audio (48kHz float32)
    audio = decoder.decode(tokens, embedding)

    # Convert float32 to int16 PCM
    audio_float = audio.cpu().numpy() if hasattr(audio, 'cpu') else audio
    audio_int16 = (audio_float * 32767).astype(np.int16)

    # Return as bytes
    return audio_int16.tobytes()

def main():
    """Main entry point for subprocess"""
    try:
        # Read input from stdin (JSON with tokens and embedding)
        input_data = sys.stdin.read()

        if not input_data.strip():
            print("[Tier 2][Python] Error: No input data", file=sys.stderr)
            sys.exit(1)

        # Parse JSON input
        data = json.loads(input_data)
        tokens = data.get('tokens')
        embedding = data.get('embedding')

        if tokens is None:
            print("[Tier 2][Python] Error: Missing 'tokens' in input", file=sys.stderr)
            sys.exit(1)

        if embedding is None:
            print("[Tier 2][Python] Error: Missing 'embedding' in input", file=sys.stderr)
            sys.exit(1)

        # Decode tokens
        print(f"[Tier 2][Python] Decoding {len(tokens)} tokens...", file=sys.stderr)
        audio_bytes = decode_tokens(tokens, embedding)
        print(f"[Tier 2][Python] Output {len(audio_bytes)} bytes", file=sys.stderr)

        # Write to stdout (binary PCM)
        sys.stdout.buffer.write(audio_bytes)
        sys.stdout.buffer.flush()

    except json.JSONDecodeError as e:
        print(f"[Tier 2][Python] Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Tier 2][Python] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```

#### 3. Update `Dockerfile`

Add LinaCodec and Python dependencies.

```dockerfile
# Stage 1: Builder
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Runtime
FROM node:20-alpine
WORKDIR /app

# Install Python and dependencies for LinaCodec decoding
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy

# Install LinaCodec and dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    git+https://github.com/ysharma3501/LinaCodec.git \
    torch \
    numpy

# Copy built app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
RUN npm ci --production

# Copy server and scripts
COPY server.js ./
COPY scripts ./scripts

# Make Python script executable
RUN chmod +x /app/scripts/*.py

EXPOSE 4173
CMD ["node", "server.js"]
```

#### 4. Update `docker-compose.yml`

Add streaming environment variables.

```yaml
services:
  echo-tts-ui:
    build: .
    container_name: echo-tts-ui
    restart: unless-stopped
    networks:
      - shared_net
    environment:
      # ... existing variables ...

      # Streaming TTS configuration (Phase 1)
      - ENABLE_STREAMING=${ENABLE_STREAMING:-true}
      - ECHOTTS_STREAMING_ENDPOINT=${ECHOTTS_STREAMING_ENDPOINT}
      - RUNPOD_API_KEY=${RUNPOD_API_KEY}
```

#### 5. Update `.env`

```bash
# Streaming TTS (Phase 1)
ENABLE_STREAMING=true
ECHOTTS_STREAMING_ENDPOINT=https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync
RUNPOD_API_KEY=your-runpod-api-key
```

---

## Part B: Cloudflare Workers (Vibe Voice, Chatterbox) - Phase 4

### Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Tier 1        │      │   Tier 2 (YOU)   │      │   Tier 3        │
│  Frontend       │─────▶│  Cloudflare      │─────▶│  RunPod         │
│                 │ SSE  │  Worker          │ HTTP  │                 │
│  Web Audio      │◀────│  (Simple         │◀─────│  Decodes to     │
│  Playback       │Audio │  Forwarder)      │Audio │  Audio          │
└─────────────────┘      └──────────────────┘      └─────────────────┘

Note: No LinaCodec decoding at Cloudflare edge due to CPU/memory limits.
RunPod decodes tokens to audio before sending to worker.
```

### Create `cloudflare-worker-streaming.js`

```javascript
/**
 * Cloudflare Worker for Streaming TTS (Phase 4)
 *
 * Forwards audio chunks from Tier 3 (RunPod) to Tier 1 (Frontend) via SSE.
 * RunPod handles LinaCodec decoding (not done at edge).
 *
 * Deployment:
 *   wrangler deploy
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // Default values, override with environment variables
  RUNPOD_STREAMING_ENDPOINT: 'https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync',
  TIMEOUT_MS: 30000,  // 30 second timeout
  CHUNK_TIMEOUT_MS: 5000  // Per-chunk timeout
};

// =============================================================================
// MAIN HANDLER
// =============================================================================

export default {
  /**
   * Main fetch handler
   */
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return handleCORS();
    }

    // Health check
    if (path === '/health') {
      return jsonResponse({
        status: 'healthy',
        tier: 'middleware-cloudflare',
        timestamp: Date.now()
      });
    }

    // Streaming TTS endpoint
    if (path === '/api/tts/stream' && request.method === 'POST') {
      return handleStreamingTTS(request, env);
    }

    return jsonResponse({ error: 'Not found' }, 404);
  }
};

// =============================================================================
// STREAMING TTS HANDLER
// =============================================================================

async function handleStreamingTTS(request, env) {
  const requestId = crypto.randomUUID();
  const startTime = Date.now();

  console.log(`[Tier 2][CF][${requestId}] Streaming TTS request received`);

  try {
    const { text, voice, service } = await request.json();

    // Validation
    if (!text || text.trim().length === 0) {
      return jsonResponse({ error: 'Text is required' }, 400);
    }

    if (!voice) {
      return jsonResponse({ error: 'Voice is required' }, 400);
    }

    // Get RunPod endpoint
    const runpodEndpoint = env.RUNPOD_STREAMING_ENDPOINT || CONFIG.RUNPOD_STREAMING_ENDPOINT;
    const apiKey = env.RUNPOD_API_KEY;

    if (!runpodEndpoint) {
      return jsonResponse({ error: 'RunPod endpoint not configured' }, 500);
    }

    if (!apiKey) {
      return jsonResponse({ error: 'RunPod API key not configured' }, 500);
    }

    console.log(`[Tier 2][CF][${requestId}] Connecting to Tier 3 (RunPod)...`);

    // Create transform stream for forwarding
    const { readable, writable } = new TransformStream();

    // Start forwarding in background
    const pipePromise = forwardRunPodStream({
      runpodEndpoint,
      apiKey,
      text,
      voice,
      service,
      writable,
      requestId
    });

    // Return streaming response immediately
    return new Response(readable, {
      headers: {
        'Content-Type': 'audio/octet-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
      }
    });

  } catch (error) {
    console.error(`[Tier 2][CF] Error:`, error);
    return jsonResponse({ error: error.message }, 500);
  }
}

/**
 * Forward RunPod stream to client (Tier 1)
 */
async function forwardRunPodStream({ runpodEndpoint, apiKey, text, voice, service, writable, requestId }) {
  const writer = writable.getWriter();

  try {
    // Connect to Tier 3 (RunPod) - request decoded audio (not tokens!)
    const runpodResponse = await fetch(runpodEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        text,
        voice,
        service,
        stream: true,
        output_format: 'pcm_16',  // RunPod decodes to PCM
        sample_rate: 48000
      })
    });

    if (!runpodResponse.ok) {
      const errorText = await runpodResponse.text();
      throw new Error(`Tier 3 error: ${runpodResponse.status} - ${errorText}`);
    }

    console.log(`[Tier 2][CF][${requestId}] Tier 3 connected, forwarding stream...`);

    // Read and forward chunks
    const reader = runpodResponse.body.getReader();
    let totalChunks = 0;
    let totalBytes = 0;

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        const elapsed = Date.now() - startTime;
        console.log(`[Tier 2][CF][${requestId}] Stream complete: ${totalChunks} chunks, ${totalBytes} bytes, ${elapsed}ms`);
        break;
      }

      // Forward chunk to Tier 1
      await writer.write(value);
      totalChunks++;
      totalBytes += value.length;

      console.log(`[Tier 2][CF][${requestId}] Forwarded chunk ${totalChunks}: ${value.length} bytes`);
    }

  } catch (error) {
    console.error(`[Tier 2][CF][${requestId}] Forward error:`, error);

    // Try to send error to client
    try {
      const errorData = new TextEncoder().encode(JSON.stringify({ error: error.message }));
      await writer.write(errorData);
    } catch (e) {
      // Client may have disconnected
    }
  } finally {
    await writer.close();
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function handleCORS() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    }
  });
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    }
  });
}
```

### `wrangler.toml` (Cloudflare Worker Config)

```toml
name = "echo-tts-streaming"
main = "cloudflare-worker-streaming.js"
compatibility_date = "2024-01-01"

[env.production]
routes = [
  { pattern = "https://vibe-voice.example.com/*", zone_name = "example.com" },
  { pattern = "https://chatterbox.example.com/*", zone_name = "example.com" }
]

[vars]
RUNPOD_STREAMING_ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync"

[[env.production.vars]]
RUNPOD_API_KEY = "your-api-key"
```

---

## Testing Strategy

### Phase 1A: Test Python Decoder Independently

```bash
# Enter container
docker exec -it echo-tts-ui sh

# Test LinaCodec import
python3 -c "from linacodec.codec import LinaCodec; print('OK')"

# Test decoder with mock data
echo '{"tokens":[1,2,3],"embedding":[0.1,0.2,0.3]}' | python3 /app/scripts/decode_linacodec.py > test.pcm

# Check output
ls -la test.pcm
```

### Phase 1B: Test Streaming Endpoint with Mock Tier 3

```bash
# Start container
docker-compose up -d

# Test with curl (will fail if Tier 3 not ready, but endpoint exists)
curl -X POST http://localhost:4173/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test",
    "voice": "alloy",
    "service": "echotts"
  }' \
  --output test_audio.pcm

# Expected: Error from RunPod (not configured yet) OR audio if ready
```

### Phase 1C: Test with Real Tier 3 (After Phase 2)

Once RunPod streaming is working:

```bash
curl -X POST http://localhost:4173/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test of the complete Tier 1 → Tier 2 → Tier 3 → Tier 2 → Tier 1 flow.",
    "voice": "alloy",
    "service": "echotts"
  }' \
  --output test_audio.pcm

# Verify audio
ffprobe -f s16le -ar 48000 -ac 1 test_audio.pcm
ffplay -f s16le -ar 48000 -ac 1 test_audio.pcm
```

---

## Troubleshooting

### Issue: "LinaCodec not installed"

**Solution**:
```bash
# Rebuild container with LinaCodec
docker-compose down
docker-compose up -d --build

# Or install manually in container
docker exec -it echo-tts-ui sh
pip3 install git+https://github.com/ysharma3501/LinaCodec.git
```

### Issue: "Python not found"

**Check Dockerfile**:
```dockerfile
RUN apk add --no-cache python3 py3-pip py3-numpy
```

### Issue: "Decode timeout"

**Increase timeout in server.js**:
```javascript
const decoder = spawn('python3', [...], {
  timeout: 20000  // 20 seconds instead of 10
});
```

### Issue: Gaps in audio

**Check buffer sizes** - ensure chunks are large enough (9600 bytes = 0.1 sec @ 48kHz)

---

## Dependencies

| Component | Dependency | Version |
|-----------|-----------|---------|
| Python runtime | python3 | Alpine package |
| LinaCodec | linacodec | GitHub (latest) |
| PyTorch | torch | Latest (CPU only) |
| NumPy | numpy | Latest |
| Node.js | spawn | Built-in |

---

## Rollback Plan

If streaming causes issues:

1. **Feature flag**: Set `ENABLE_STREAMING=false` in `.env`
2. **Automatic fallback**: Frontend falls back to batch TTS on errors
3. **No breaking changes**: Existing batch TTS endpoints unchanged
4. **Quick rollback**:
   ```bash
   # Disable streaming without redeploy
   docker-compose echo-tts-ui exec -e ENABLE_STREAMING=false
   ```

---

## Next Steps After Implementation

1. ✅ Test Python decoder independently
2. ✅ Test streaming endpoint with mock data
3. ✅ Test with real Tier 3 (after Phase 2)
4. ✅ Performance testing (decode latency, throughput)
5. ✅ Integration testing with Tier 1 (after Phase 3)
6. ✅ Deploy Cloudflare Worker (Phase 4)
