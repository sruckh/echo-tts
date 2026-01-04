# LINACODEC-FE.md
## Frontend Implementation - Streaming Audio Playback (Tier 1)

### 3-Tier System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 1: FRONTEND                             │
│  React/TypeScript Application (Port 4173)                          │
│                                                                     │
│  Responsibilities:                                                 │
│  - User interface for TTS generation                               │
│  - Audio playback via Web Audio API                                │
│  - Streaming chunk consumption                                     │
│  - No LinaCodec decoding (handled by Tier 2)                       │
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │   User Input    │───▶│  Streaming TTS   │───▶│ Audio Player  │ │
│  │  (text, voice)  │    │     Hook         │    │ (Web Audio)   │ │
│  └─────────────────┘    └──────────────────┘    └───────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ SSE (Server-Sent Events)
                                   │ Audio Chunks (48kHz PCM)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 2: MIDDLEWARE                           │
│  Express Server (Local) + Cloudflare Workers                       │
│                                                                     │
│  Responsibilities:                                                 │
│  - LinaCodec token decoding (local only)                          │
│  - Audio chunk streaming via SSE                                   │
│  - Service proxy and routing                                       │
│                                                                     │
│  EchoTTS: Python decoder │ Vibe Voice/Chatterbox: Forward-only    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTP/WebSocket
                                   │ LinaCodec Tokens or Audio
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 3: BACKEND                              │
│  RunPod Serverless (GPU)                                           │
│                                                                     │
│  Responsibilities:                                                 │
│  - TTS audio generation                                            │
│  - LinaCodec token encoding                                        │
│  - Streaming output via runpod.stream()                            │
│  - Model caching on network volume                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phase

**This component is implemented in Phase 3** (after middleware and RunPod streaming work)

**Prerequisites:**
- Tier 2 (Middleware) `/api/tts/stream` endpoint must be working
- Tier 3 (RunPod) streaming handler must be working

**Why this order?**
- Frontend needs real streaming endpoint to test
- Can mock stream during development
- Web Audio API can be tested independently

---

## Architecture

### Data Flow (Streaming Mode)

```
User clicks "Generate"
        ↓
Frontend: POST /api/tts/stream
        ↓
Middleware: Receives request, proxies to RunPod
        ↓
RunPod: Generates audio → Encodes to tokens → Streams
        ↓
Middleware: Decodes tokens → Streams audio chunks (SSE)
        ↓
Frontend: Receives chunks → Queues in Web Audio API → Plays
        ↓
User hears audio in real-time
```

### Audio Format

| Property | Value |
|----------|-------|
| Format | 16-bit PCM |
| Sample Rate | 48000 Hz |
| Channels | 1 (mono) |
| Byte Order | Little-endian |
| Chunk Size | ~9600 bytes (0.1 sec at 48kHz) |

---

## Files to Create

### 1. `src/hooks/useStreamingTTS.ts`

New hook for handling streaming TTS via fetch with streaming response body.

```typescript
/**
 * useStreamingTTS Hook
 *
 * Handles streaming TTS requests via Server-Sent Events (SSE).
 * Receives audio chunks from Tier 2 (Middleware) and plays via Web Audio API.
 *
 * Note: No LinaCodec decoding happens here - all decoding is done by Tier 2.
 */

interface StreamingTTSOptions {
  text: string;
  voice: string;
  serviceId: string;
  onChunk?: (audioChunk: ArrayBuffer, chunkNumber: number) => void;
  onComplete?: (totalDuration: number) => void;
  onError?: (error: Error) => void;
  onProgress?: (progress: number) => void;
}

interface StreamingTTSState {
  isStreaming: boolean;
  progress: number;
  chunksReceived: number;
  totalDuration: number;
}

export function useStreamingTTS() {
  const [state, setState] = useState<StreamingTTSState>({
    isStreaming: false,
    progress: 0,
    chunksReceived: 0,
    totalDuration: 0
  });

  const generateStreaming = async (options: StreamingTTSOptions): Promise<void> => {
    const { text, voice, serviceId, onChunk, onComplete, onError, onProgress } = options;

    setState(prev => ({ ...prev, isStreaming: true, progress: 0, chunksReceived: 0 }));

    try {
      // Get service configuration from Tier 2 injected env vars
      const services = window.__ENV__?.SERVICES || {};
      const service = services[serviceId];

      if (!service) {
        throw new Error(`Service not found: ${serviceId}`);
      }

      // Build streaming endpoint URL (Tier 2)
      const baseUrl = service.endpoint.replace('/v1/audio/speech', '');
      const streamEndpoint = `${baseUrl}/api/tts/stream`;

      console.log(`[Tier 1] Starting streaming TTS: ${streamEndpoint}`);
      console.log(`[Tier 1] Text length: ${text.length} chars`);

      const response = await fetch(streamEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(service.apiKey && { 'Authorization': `Bearer ${service.apiKey}` })
        },
        body: JSON.stringify({
          text,
          voice,
          service: serviceId,
          stream: true
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      // Create audio context for playback (48kHz to match Tier 2 output)
      const audioContext = new AudioContext({ sampleRate: 48000 });
      const sources: AudioBufferSourceNode[] = [];
      let startTime = 0;
      let totalChunks = 0;
      let totalSamples = 0;

      // Read streaming response from Tier 2
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is not readable');
      }

      // Buffer for accumulating partial data
      let buffer = new Uint8Array(0);
      const bytesPerSample = 2; // 16-bit PCM
      const samplesPerChunk = 4800; // 0.1 sec at 48kHz
      const minChunkSize = samplesPerChunk * bytesPerSample;

      console.log(`[Tier 1] Starting to receive chunks...`);

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          console.log(`[Tier 1] Stream complete: ${totalChunks} chunks received`);
          break;
        }

        // Accumulate data
        const newBuffer = new Uint8Array(buffer.length + value.length);
        newBuffer.set(buffer, 0);
        newBuffer.set(value, buffer.length);
        buffer = newBuffer;

        // Try to decode complete chunks
        while (buffer.length >= minChunkSize) {
          try {
            const samples = buffer.length / bytesPerSample;
            const float32Array = new Float32Array(samples);

            // Convert 16-bit PCM to Float32 for Web Audio API
            for (let i = 0; i < samples; i++) {
              const sample = (buffer[i * 2] | (buffer[i * 2 + 1] << 8));
              float32Array[i] = sample < 32768 ? sample / 32768 : (sample - 65536) / 32768;
            }

            // Create audio buffer
            const audioBuffer = audioContext.createBuffer(1, samples, 48000);
            audioBuffer.copyToChannel(float32Array, 0);

            // Schedule playback
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);

            if (startTime === 0) {
              source.start(0);
              startTime = audioContext.currentTime;
              console.log(`[Tier 1] First audio chunk started playing`);
            } else {
              source.start(startTime);
              startTime += audioBuffer.duration;
            }

            sources.push(source);
            totalChunks++;
            totalSamples += samples;

            // Callbacks
            onChunk?.(audioBuffer, totalChunks);
            onProgress?.(Math.min(totalChunks * 5, 95)); // Rough progress

            // Update state
            setState(prev => ({
              ...prev,
              chunksReceived: totalChunks,
              progress: Math.min(totalChunks * 5, 95)
            }));

            // Remove decoded data from buffer
            const remaining = buffer.length - (samples * bytesPerSample);
            buffer = buffer.slice(samples * bytesPerSample);

          } catch (decodeError) {
            // Not enough data or decode error, wait for more
            break;
          }
        }
      }

      const totalDuration = startTime - audioContext.currentTime;
      console.log(`[Tier 1] Playback complete: ${totalDuration.toFixed(2)}s total`);

      setState(prev => ({
        ...prev,
        isStreaming: false,
        progress: 100,
        totalDuration
      }));

      onComplete?.(totalDuration);

    } catch (error) {
      console.error('[Tier 1] Streaming error:', error);
      setState(prev => ({ ...prev, isStreaming: false }));
      onError?.(error as Error);
    }
  };

  return {
    generateStreaming,
    ...state
  };
}
```

---

## Files to Modify

### 2. Modify `src/App.tsx`

Integrate streaming TTS with fallback to batch mode.

```typescript
/**
 * App.tsx Modifications for Streaming TTS
 *
 * Adds streaming toggle and integrates with streaming hook.
 * Maintains backward compatibility with batch TTS.
 */

// Add import
import { useStreamingTTS } from './hooks/useStreamingTTS';

function App() {
  // ... existing state ...

  // Add streaming preference state
  const [useStreaming, setUseStreaming] = useState(() => {
    // Check localStorage for saved preference
    const saved = localStorage.getItem('tts_streaming');
    return saved ? JSON.parse(saved) : true; // Default to true
  });

  // Initialize streaming hook
  const {
    generateStreaming,
    isStreaming,
    progress: streamingProgress,
    chunksReceived
  } = useStreamingTTS();

  // Save streaming preference
  useEffect(() => {
    localStorage.setItem('tts_streaming', JSON.stringify(useStreaming));
  }, [useStreaming]);

  // Modify handleGenerate to support streaming with fallback
  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    const selectedVoice = voices.find(v => v.id === selectedVoiceId);
    if (!selectedVoice) return;

    setLoading(true);
    setError(null);

    try {
      // Decision: Streaming or Batch?
      const shouldStream = useStreaming && selectedService?.id !== 'alibaba';

      console.log(`[App] Generation mode: ${shouldStream ? 'STREAMING' : 'BATCH'}`);

      if (shouldStream) {
        // Use streaming (Phase 3+)
        await generateStreaming({
          text,
          voice: selectedVoice.id,
          serviceId: selectedService.id,
          onChunk: (chunk, num) => {
            console.log(`[App] Received chunk ${num}`);
          },
          onComplete: (duration) => {
            console.log(`[App] Streaming complete: ${duration.toFixed(2)}s`);
            // Note: Streaming audio is not saved to history in this implementation
            // Could implement concatenation if needed
          },
          onError: (error) => {
            console.error('[App] Streaming failed, falling back to batch:', error);
            // Automatic fallback to batch TTS
            handleBatchTTS(text, selectedVoice, selectedService);
          }
        });

      } else {
        // Use batch TTS (existing implementation)
        await handleBatchTTS(text, selectedVoice, selectedService);
      }

    } catch (error) {
      console.error('[App] TTS error:', error);
      setError(error instanceof Error ? error.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  };

  // Batch TTS handler (existing code, extracted for reuse)
  const handleBatchTTS = async (text: string, voice: any, service: any) => {
    console.log('[App] Using batch TTS mode');

    let blob: Blob;

    if (service.id === 'alibaba') {
      blob = await wsGenerate({ text, voice, service });
    } else {
      blob = await httpGenerate({ text, voice: voice.id, serviceId: service.id });
    }

    if (blob) {
      const id = generateId();
      await addToHistory({
        id,
        text,
        blob,
        voice: voice.id,
        service: service.id
      });
      const url = createUrl(id, blob);
      play(id, url);
    }
  };

  // Add streaming toggle to UI (in the controls section)
  return (
    // ... existing JSX ...
    <Box sx={{ mt: 2 }}>
      <FormControlLabel
        control={
          <Switch
            checked={useStreaming}
            onChange={(e) => setUseStreaming(e.target.checked)}
            disabled={isStreaming || loading}
          />
        }
        label={
          <Box>
            <Typography variant="body2">
              Stream audio (faster)
            </Typography>
            {isStreaming && (
              <Typography variant="caption" color="primary">
                Streaming... ({chunksReceived} chunks)
              </Typography>
            )}
          </Box>
        }
      />
    </Box>
  );
}
```

### 3. Modify `src/config.ts`

Add streaming endpoint support for service configuration.

```typescript
/**
 * config.ts Modifications
 *
 * Adds streaming support to service configuration.
 * Tier 2 (Middleware) injects these at runtime.
 */

export interface TTSService {
  id: string;
  label: string;
  endpoint: string;
  apiKey?: string;
  streamingSupported?: boolean;  // New: indicates if streaming is available
  streamingEndpoint?: string;     // New: explicit streaming endpoint (optional)
}

// Update service initialization
export const getServices = (): TTSService[] => {
  // Priority: Runtime env vars (Tier 2) → Build-time env vars → Fallback
  const runtimeServices = window.__ENV__?.SERVICES;
  const buildTimeServices = import.meta.env.VITE_SERVICES;

  if (runtimeServices) {
    return Object.entries(runtimeServices).map(([id, service]: [string, any]) => ({
      id,
      label: service.label || id,
      endpoint: service.endpoint,
      apiKey: service.apiKey,
      streamingSupported: service.streamingSupported ?? true,  // Default to true
      streamingEndpoint: service.streamingEndpoint  // Optional override
    }));
  }

  if (buildTimeServices) {
    try {
      return JSON.parse(buildTimeServices);
    } catch {
      return [];
    }
  }

  // Fallback for development
  if (import.meta.env.VITE_OPEN_AI_TTS_ENDPOINT) {
    return [{
      id: 'default',
      label: 'Default TTS',
      endpoint: import.meta.env.VITE_OPEN_AI_TTS_ENDPOINT,
      streamingSupported: true
    }];
  }

  return [];
};
```

---

## Testing Strategy

### Phase 3A: Mock Stream Testing (Before Tier 2 is ready)

Test Web Audio API without backend dependency:

```typescript
// test-streaming.ts
async function testMockStream() {
  const audioContext = new AudioContext({ sampleRate: 48000 });

  // Generate mock audio chunks (440Hz sine wave)
  const sampleRate = 48000;
  const frequency = 440;
  const chunkDuration = 0.1; // 100ms
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

### Phase 3B: Integration Testing (After Tier 2 is ready)

Once middleware `/api/tts/stream` is working:

```bash
# 1. Test endpoint directly with curl
curl -X POST http://localhost:4173/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test of the streaming TTS system from Tier 1.",
    "voice": "alloy",
    "service": "echotts"
  }' \
  --output test_audio.pcm

# 2. Verify audio format
ffprobe -f s16le -ar 48000 -ac 1 test_audio.pcm

# 3. Play audio
ffplay -f s16le -ar 48000 -ac 1 test_audio.pcm
```

### Phase 3C: Frontend Integration Testing

1. Start the app: `npm run dev`
2. Enable "Stream audio" toggle
3. Enter test text: "Testing streaming from Tier 1 to Tier 3 and back."
4. Click "Generate & Play"
5. Verify:
   - [ ] First audio plays within 1 second
   - [ ] No gaps or clicks between chunks
   - [ ] Progress indicator updates
   - [ ] Audio quality is good (48kHz)
   - [ ] Fallback to batch works if streaming fails

---

## Browser Compatibility

| Browser | Version | Support | Notes |
|---------|---------|---------|-------|
| Chrome | 90+ | ✅ Full | Native Web Audio API |
| Firefox | 88+ | ✅ Full | Native Web Audio API |
| Safari | 14+ | ✅ Full | Native Web Audio API |
| Edge | 90+ | ✅ Full | Native Web Audio API |
| Mobile Safari | iOS 14+ | ✅ Full | Works on iPhone/iPad |
| Mobile Chrome | Android 8+ | ✅ Full | Works on Android |

---

## Troubleshooting

### Issue: "Response body is not readable"

**Cause**: Streaming endpoint not returning a stream
**Check**:
- Verify Tier 2 middleware is running
- Check that response headers include `Transfer-Encoding: chunked`

### Issue: Gaps or clicks in audio playback

**Cause**: Chunk timing issues in Web Audio API
**Solutions**:
```typescript
// Ensure chunks are scheduled with correct timing
if (startTime === 0) {
  source.start(0);
  startTime = audioContext.currentTime;
} else {
  // Use exact end time of previous chunk
  source.start(startTime);
  startTime += audioBuffer.duration;
}
```

### Issue: Audio plays too fast/slow

**Cause**: Sample rate mismatch
**Fix**:
```typescript
// Tier 1 must match Tier 2 output sample rate
const audioContext = new AudioContext({ sampleRate: 48000 }); // Must be 48000
```

### Issue: Fallback to batch not working

**Check**:
```typescript
// Verify service detection
const shouldStream = useStreaming && selectedService?.id !== 'alibaba';
console.log('Should stream?', shouldStream);
```

---

## Dependencies

No new npm packages required!

Uses native browser APIs:
- `fetch` with streaming response body
- `Web Audio API` (`AudioContext`, `AudioBuffer`, etc.)
- Existing React hooks

---

## Rollback Plan

If streaming causes issues:

1. **User can disable** via "Stream audio" toggle
2. **Automatic fallback** on errors
3. **No breaking changes** - batch TTS still works
4. **Feature flag** via environment variable:

```typescript
const STREAMING_ENABLED = window.__ENV__?.STREAMING_ENABLED ?? true;
```

---

## Next Steps After Implementation

1. ✅ Test with EchoTTS (local middleware)
2. ✅ Test with Vibe Voice (Cloudflare Worker)
3. ✅ Test with Chatterbox (Cloudflare Worker)
4. ✅ Performance testing (long texts, slow connections)
5. ✅ User acceptance testing
6. ✅ Make streaming the default after validation
