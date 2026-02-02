# Echo-TTS Cloudflare Bridge

Cloudflare Worker that translates OpenAI TTS API requests to RunPod Echo-TTS serverless format.

## Features

- OpenAI TTS API compatible endpoint
- Optional Bearer token authentication
- Streaming support (PCM chunks)
- CORS enabled for browser usage
- Voice mapping configurable via env var

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | OpenAI TTS compatible endpoint |
| `/api/tts/stream` | POST | Custom streaming endpoint |
| `/health` | GET | Health check |

## Voice Mapping

Default OpenAI voice mapping (override via `VOICE_MAP` env var):

| OpenAI Voice | Echo-TTS Speaker |
|--------------|------------------|
| `alloy` | Elijah |
| `echo` | Kurt |
| `fable` | Kim |
| `onyx` | Scott |
| `nova` | Dorota |
| `shimmer` | Kim |

If the voice is not in the map, it is passed through as `speaker_voice`.

## Usage

### OpenAI TTS Compatible Request

```bash
curl https://your-worker.workers.dev/v1/audio/speech \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "echo-tts",
    "input": "Hello, world!",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.ogg
```

Note: Echo-TTS batch output is OGG Opus. The worker always returns that audio and sets `X-Actual-Format: ogg-opus`.

### Streaming Request

```bash
curl https://your-worker.workers.dev/api/tts/stream \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "speaker_voice": "Dorota",
    "output_format": "pcm_16"
  }'
```

Streaming always returns raw PCM bytes (content type `audio/pcm`).

### Python Client (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-worker.workers.dev/v1",
    api_key="YOUR_AUTH_TOKEN"
)

response = client.audio.speech.create(
    model="echo-tts",
    text="Hello, world!",
    voice="alloy"
)

response.stream_to_file("speech.ogg")
```

## Environment Variables

Set via `npx wrangler secret put <NAME>`:

| Variable | Description | Required |
|----------|-------------|----------|
| `RUNPOD_URL` | RunPod serverless endpoint URL | Yes |
| `RUNPOD_API_KEY` | RunPod API key | Yes |
| `API_KEY` | Optional authentication token | No |
| `VOICE_MAP` | JSON mapping for OpenAI voice -> speaker_voice | No |
| `DEFAULT_VOICE` | Fallback speaker voice | No |

### Example VOICE_MAP

```json
{
  "alloy": "Dorota",
  "echo": "Elijah"
}
```

## Deployment

```bash
npm install
npx wrangler secret put RUNPOD_URL
npx wrangler secret put RUNPOD_API_KEY
npx wrangler deploy
```

## License

Apache 2.0
