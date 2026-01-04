# LinaCodec Streaming - Master Implementation Guide

## Overview

This document provides the **complete implementation order** for integrating LinaCodec streaming into your 3-tier TTS system. The implementation is divided into **phases** that allow incremental testing and validation.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: FRONTEND (React/TypeScript)                            │
│ - User interface                                                 │
│ - Web Audio API playback                                        │
│ - Streaming chunk consumption                                   │
└─────────────────────────────────────────────────────────────────┘
                    ▲ SSE (48kHz PCM audio)
                    │
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2: MIDDLEWARE                                              │
│ - Express Server (local): LinaCodec decode + SSE stream       │
│ - Cloudflare Workers: Simple forwarding (no decode at edge)   │
└─────────────────────────────────────────────────────────────────┘
                    ▲ HTTP/WebSocket
                    │ LinaCodec Tokens or Audio
                    │
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: BACKEND (RunPod Serverless)                            │
│ - TTS audio generation                                         │
│ - LinaCodec token encoding                                     │
│ - Streaming via runpod.stream()                                │
│ - Model caching on network volume                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

| Phase | Component | Primary Document | Testing Method | Estimated Time |
|-------|-----------|-------------------|-----------------|----------------|
| 0 | Preserve Existing Functionality | - | Feature flags | 1 hour |
| 1 | Local Middleware (Express Server) | LINACODEC-MW.md | `curl localhost:4173` | 2-3 days |
| 2 | RunPod Serverless (Backend) | LINACODEC-BE.md | `curl $RUNPOD_ENDPOINT` | 3-5 days |
| 3 | Frontend (React/TypeScript) | LINACODEC-FE.md | Browser testing | 2-3 days |
| 4 | Cloudflare Workers | LINACODEC-MW.md (Part B) | Deploy + test | 2-3 days |
| 5 | Testing & Optimization | All documents | Load testing | 2-3 days |

**Total Estimated Time**: 2-3 weeks

---

## Phase 0: Preserve Existing Functionality (1 hour)

**Goal**: Add feature flags to allow instant rollback.

### 0.1 Add Feature Flag to Express Server

**File**: `server.js`

```javascript
// Add at top of server.js
const ENABLE_STREAMING = process.env.ENABLE_STREAMING === 'true';

app.use((req, res, next) => {
  // Inject feature flag to frontend
  originalJson = res.json;
  res.json = function(data) {
    data.STREAMING_ENABLED = ENABLE_STREAMING;
    return originalJson.call(this, data);
  };
  next();
});
```

### 0.2 Add Feature Flag to Frontend

**File**: `src/config.ts`

```typescript
export const isStreamingEnabled = (): boolean => {
  return window.__ENV__?.STREAMING_ENABLED ?? false;
};
```

### 0.3 Update Environment Variables

**File**: `.env`

```bash
# Feature flags
ENABLE_STREAMING=false  # Start disabled, enable after testing
```

---

## Phase 1: Local Middleware (START HERE!)

**Goal**: Add streaming endpoint to Express server for EchoTTS.

**Why First?**
- Easiest to test locally with `curl`
- Can mock RunPod responses
- Foundation for frontend testing

### 1.1 Create Streaming Endpoint

**File**: `server.js`

Add the `/api/tts/stream` endpoint (see LINACODEC-MW.md for full code).

### 1.2 Create Python Decoder Script

**File**: `scripts/decode_linacodec.py`

Create the LinaCodec decoder script (see LINACODEC-MW.md for full code).

### 1.3 Update Dockerfile

Add Python and LinaCodec dependencies (see LINACODEC-MW.md).

### 1.4 Update docker-compose.yml

Add streaming environment variables (see LINACODEC-MW.md).

### 1.5 Test Python Decoder Independently

```bash
# Build container
docker-compose up -d --build

# Enter container
docker exec -it echo-tts-ui sh

# Test LinaCodec import
python3 -c "from linacodec.codec import LinaCodec; print('OK')"

# Test decoder with mock data
echo '{"tokens":[1,2,3],"embedding":[0.1,0.2,0.3]}' | python3 /app/scripts/decode_linacodec.py > test.pcm

# Check output
ls -la test.pcm
```

**Expected**: File created with ~48 bytes (3 tokens × 16-bit).

### 1.6 Test Streaming Endpoint (Mock Tier 3)

```bash
curl -X POST http://localhost:4173/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test",
    "voice": "alloy",
    "service": "echotts"
  }' \
  --output test_audio.pcm
```

**Expected**: Error from RunPod (not configured yet) OR success if endpoint is working.

### 1.7 Enable Feature Flag

```bash
# In .env
ENABLE_STREAMING=true

# Restart container
docker-compose down
docker-compose up -d
```

---

## Phase 2: RunPod Serverless (Backend)

**Goal**: Add LinaCodec encoding and streaming to RunPod handler.

**Why Second?**
- Can test independently with `curl`
- Tier 1 provides proven test endpoint
- No UI dependencies

### 2.1 Install LinaCodec on RunPod

**File**: `requirements.txt` (in RunPod template)

```txt
git+https://github.com/ysharma3501/LinaCodec.git
```

### 2.2 Modify RunPod Handler

**File**: `handler.py`

Add streaming generators (see LINACODEC-BE.md for full code).

Key functions to add:
- `generate_audio_stream()` - Split text into chunks
- `encode_to_linacodec()` - Encode audio to tokens
- `generate_linacodec_token_stream()` - Stream tokens
- `generate_audio_stream_decoded()` - Stream decoded audio (for Cloudflare)

### 2.3 Configure Network Volume

Ensure your RunPod Serverless template has:

```yaml
volumeMounts:
  - mountPath: /runpod-volume
    name: linacodec-models

volumes:
  - name: linacodec-models
    volumeId: vol-xxxxxxxxxxxxx  # Your volume ID
```

### 2.4 Test Audio Chunk Streaming (Phase 2A)

Test without LinaCodec first to verify streaming infrastructure:

```bash
curl -X POST $RUNPOD_ENDPOINT \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "text": "This is a test of streaming audio chunks.",
    "voice": "default",
    "stream": true,
    "output_format": "pcm_16"
  }' \
  --output test_stream.json
```

**Expected Output**:
```json
{"status":"streaming","chunk":1,"audio_chunk":"base64data...","sample_rate":48000}
{"status":"streaming","chunk":2,"audio_chunk":"base64data...","sample_rate":48000}
{"status":"complete",...}
```

### 2.5 Test LinaCodec Token Streaming (Phase 2B)

Test with LinaCodec encoding:

```bash
curl -X POST $RUNPOD_ENDPOINT \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "text": "Testing LinaCodec token streaming.",
    "voice": "default",
    "stream": true,
    "output_format": "linacodec_tokens"
  }' \
  --output test_tokens.json
```

**Expected Output**:
```json
{"status":"streaming","chunk":1,"tokens":[...],"embedding":[...],"sample_rate":48000}
{"status":"streaming","chunk":2,"tokens":[...],"embedding":[...],"sample_rate":48000}
{"status":"complete",...}
```

### 2.6 Verify Model Caching

```python
# In RunPod pod or via SSH
import os
print(f"Cache dir: {os.environ.get('HF_HOME')}")

# Check LinaCodec is cached
import subprocess
result = subprocess.run(['ls', '-la', '/runpod-volume/.cache/huggingface/hub'], capture_output=True)
print(result.stdout.decode())
```

**Expected**: See `models--YatharthS--LinaCodec/` directory.

---

## Phase 3: Frontend (React/TypeScript)

**Goal**: Add Web Audio API streaming playback.

**Why Third?**
- Needs working Tier 2 endpoint
- Can mock stream for development
- Web Audio API can be tested independently

### 3.1 Create Streaming Hook

**File**: `src/hooks/useStreamingTTS.ts`

Create the streaming TTS hook (see LINACODEC-FE.md for full code).

### 3.2 Modify App Component

**File**: `src/App.tsx`

Add streaming toggle and integrate streaming hook (see LINACODEC-FE.md for full code).

### 3.3 Update Configuration

**File**: `src/config.ts`

Add streaming support to service configuration (see LINACODEC-FE.md).

### 3.4 Test Web Audio API Independently

Create `src/test-streaming.ts`:

```typescript
async function testMockStream() {
  const audioContext = new AudioContext({ sampleRate: 48000 });
  const sampleRate = 48000;
  const frequency = 440;
  const chunkDuration = 0.1;
  const samplesPerChunk = sampleRate * chunkDuration;

  for (let i = 0; i < 10; i++) {
    const buffer = audioContext.createBuffer(1, samplesPerChunk, sampleRate);
    const channel = buffer.getChannelData(0);

    for (let j = 0; j < samplesPerChunk; j++) {
      const t = (i * samplesPerChunk + j) / sampleRate;
      channel[j] = Math.sin(2 * Math.PI * frequency * t) * 0.5;
    }

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start(i * chunkDuration);

    console.log(`Playing chunk ${i + 1}/10`);
    await new Promise(r => setTimeout(r, chunkDuration * 1000));
  }
}

testMockStream();
```

Run with: `npx tsx src/test-streaming.ts`

**Expected**: Hear 10 sine wave chunks playing smoothly.

### 3.5 Test with Real Middleware

```bash
# Start dev server
npm run dev

# Enable streaming toggle
# Enter test text: "Testing streaming from frontend."
# Click "Generate & Play"
```

**Expected**:
- First audio plays within 1 second
- No gaps or clicks between chunks
- Progress indicator updates
- Audio quality is good (48kHz)

---

## Phase 4: Cloudflare Workers

**Goal**: Add streaming worker for Vibe Voice and Chatterbox.

**Why Fourth?**
- Similar to local middleware (already tested)
- Requires deployment (hardest to test)
- Uses RunPod's decoded-audio mode (simpler)

### 4.1 Create Cloudflare Worker

**File**: `cloudflare-worker-streaming.js`

Create the worker code (see LINACODEC-MW.md Part B for full code).

### 4.2 Configure Wrangler

**File**: `wrangler.toml`

```toml
name = "echo-tts-streaming"
main = "cloudflare-worker-streaming.js"
compatibility_date = "2024-01-01"

[vars]
RUNPOD_STREAMING_ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync"

[[env.production.vars]]
RUNPOD_API_KEY = "your-api-key"
```

### 4.3 Deploy Worker

```bash
# Install wrangler
npm install -g wrangler

# Login
wrangler login

# Deploy
wrangler deploy
```

### 4.4 Test Worker

```bash
curl -X POST https://your-worker.workers.dev/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Testing Cloudflare Worker streaming.",
    "voice": "default",
    "service": "vibevoice"
  }' \
  --output test_worker_audio.pcm
```

**Expected**: Audio file downloaded successfully.

### 4.5 Update Service Configuration

Update Vibe Voice and Chatterbox service endpoints to point to Cloudflare Worker.

---

## Phase 5: Testing & Optimization

**Goal**: Validate performance and optimize.

### 5.1 End-to-End Testing

Test all three services:

| Service | Path | Expected Time to First Audio |
|---------|------|-----------------------------|
| EchoTTS | Local middleware | < 1 second |
| Vibe Voice | Cloudflare Worker | < 2 seconds |
| Chatterbox | Cloudflare Worker | < 2 seconds |

### 5.2 Performance Testing

**Metrics to track**:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first audio | < 1 sec | Frontend logs |
| Chunk decode latency | < 10ms | Middleware logs |
| Token encode latency | < 50ms | RunPod logs |
| Cloudflare timeout rate | < 1% | Cloudflare analytics |

### 5.3 Quality A/B Testing

Compare quality with/without LinaCodec:

```bash
# Test with LinaCodec
curl -X POST $ENDPOINT \
  -d '{"text":"Same text for both tests.","stream":true,"output_format":"linacodec_tokens"}' \
  --output with_linacodec.pcm

# Test without LinaCodec (original)
curl -X POST $ENDPOINT \
  -d '{"text":"Same text for both tests.","stream":false}' \
  --output without_linacodec.mp3

# Compare
ffplay with_linacodec.pcm
ffplay without_linacodec.mp3
```

### 5.4 Load Testing

Test with various text lengths:

| Text Length | Chunks | Expected Behavior |
|-------------|--------|-------------------|
| 100 chars | 1 | Single chunk, quick response |
| 500 chars | 2-3 | Multiple chunks, smooth playback |
| 2000 chars | 8-12 | Many chunks, no timeouts |
| 10000 chars | 40-60 | Long streaming, verify no drops |

### 5.5 Rollback Testing

Verify rollback works:

1. **Disable streaming**: `ENABLE_STREAMING=false`
2. **Test**: Falls back to batch TTS
3. **Re-enable**: `ENABLE_STREAMING=true`
4. **Test**: Streaming works again

---

## Success Criteria

### Phase Completion Criteria

| Phase | Criteria | How to Verify |
|-------|----------|----------------|
| 0 | Feature flags work | Toggle streaming on/off |
| 1 | Middleware streams audio | `curl` returns audio file |
| 2 | RunPod streams tokens | `curl` returns JSON lines |
| 3 | Frontend plays audio | Browser plays smoothly |
| 4 | Worker forwards audio | `curl` to worker returns audio |
| 5 | All metrics met | Monitoring dashboards |

### Final Success Metrics

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Time to first audio | Full generation | First chunk | < 1 sec |
| Cloudflare timeouts | Frequent | Rare | < 1% |
| RunPod→Middleware payload | Full audio | Tokens | ~60% reduction |
| Audio quality | 24kHz | 48kHz | Subjective improvement |
| User-perceived latency | High | Low | "Feels faster" |

---

## Troubleshooting Guide

### Common Issues Across Phases

| Issue | Phase | Solution |
|-------|-------|----------|
| "Response body not readable" | 1, 3 | Check streaming headers |
| "LinaCodec not installed" | 1, 2 | Rebuild container/pod |
| Gaps in audio playback | 3 | Check Web Audio API timing |
| Cloudflare timeout | 2, 4 | Use decoded-audio mode |
| Model not cached | 2 | Check network volume mount |

### Rollback Procedures

| Level | Trigger | Action | Time to Rollback |
|-------|--------|--------|-----------------|
| User | Toggle off | Flip switch in UI | Instant |
| Environment | Bug found | Set `ENABLE_STREAMING=false` | 1 min (redeploy) |
| Code | Breaking bug | Revert commit | 5-10 min (git revert) |
| Architecture | Fundamental issue | Disable streaming endpoints | 15-30 min |

---

## File Summary

### Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `scripts/decode_linacodec.py` | 1 | LinaCodec decoder (Tier 2) |
| `src/hooks/useStreamingTTS.ts` | 3 | Streaming TTS hook (Tier 1) |
| `cloudflare-worker-streaming.js` | 4 | Cloudflare Worker (Tier 2) |
| `IMPLEMENTATION-ORDER.md` | 0 | This document |

### Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `server.js` | 1 | Add `/api/tts/stream` endpoint |
| `Dockerfile` | 1 | Add Python and LinaCodec |
| `docker-compose.yml` | 1 | Add streaming env vars |
| `.env` | 0, 1 | Add feature flags and endpoints |
| `src/App.tsx` | 3 | Add streaming toggle and hook |
| `src/config.ts` | 3 | Add streaming support |
| `handler.py` | 2 | Add streaming generators |
| `requirements.txt` | 2 | Add LinaCodec |

---

## Next Steps

Once all phases are complete:

1. **Monitor for 1 week** - Collect metrics and user feedback
2. **Optimize chunk sizes** - Based on real-world usage
3. **Add more services** - Extend to other TTS providers
4. **Document learnings** - Update internal documentation
5. **Consider additional features**:
   - Voice cloning with LinaCodec
   - Real-time voice conversion
   - Audio super-resolution

---

## Questions?

Refer to the detailed implementation documents:

- **LINACODEC-FE.md** - Frontend implementation (Tier 1)
- **LINACODEC-MW.md** - Middleware implementation (Tier 2)
- **LINACODEC-BE.md** - Backend implementation (Tier 3)

Each document includes:
- 3-tier system context
- Detailed code examples
- Testing strategies
- Troubleshooting guides
- Rollback plans
