/**
 * Cloudflare Worker: OpenAI TTS API -> RunPod Echo-TTS Bridge
 *
 * Translates OpenAI Text-to-Speech API requests to RunPod Echo-TTS format
 * and returns OpenAI-compatible responses (audio bytes).
 */

const DEFAULT_VOICE_MAP = {
  alloy: 'Elijah',
  echo: 'Kurt',
  fable: 'Kim',
  onyx: 'Scott',
  nova: 'Dorota',
  shimmer: 'Kim'
};

const DEFAULT_VOICES = ['Dorota', 'Elijah', 'Kim', 'Kurt', 'Scott'];

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return handleCORS();
    }

    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'healthy',
        tier: 'middleware-cloudflare-echotts',
        timestamp: Date.now()
      }), {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      });
    }

    if (env.API_KEY) {
      const authHeader = request.headers.get('Authorization');
      if (!authHeader) {
        return openaiError('Missing Authorization header', 'authentication_error', null, 401);
      }

      const token = authHeader.replace(/^Bearer\s+/i, '');
      if (token !== env.API_KEY) {
        return openaiError('Invalid authentication token', 'authentication_error', null, 401);
      }
    }

    if (request.method === 'POST' && url.pathname === '/v1/audio/speech') {
      return handleOpenAITTS(request, env);
    }

    if (request.method === 'POST' && url.pathname === '/api/tts/stream') {
      return handleStreamingTTS(request, env);
    }

    return errorResponse('Not found. Available endpoints: POST /v1/audio/speech, POST /api/tts/stream', 404);
  }
};

async function handleOpenAITTS(request, env) {
  try {
    if (!env.RUNPOD_URL || !env.RUNPOD_API_KEY) {
      console.error('CRITICAL: RunPod configuration missing!');
      return openaiError('Server configuration error', 'server_error', null, 500);
    }

    const openaiRequest = await request.json();
    const input = openaiRequest.input || openaiRequest.text;
    const voice = openaiRequest.voice;
    const response_format = openaiRequest.response_format || 'mp3';
    const stream = Boolean(openaiRequest.stream);
    const parameters = openaiRequest.parameters || {};

    if (!input || !voice) {
      return openaiError('Missing required parameters: input and voice', 'invalid_request_error', null, 400);
    }

    const speakerVoice = resolveSpeakerVoice(voice, openaiRequest, env);

    if (stream) {
      return handleOpenAIStreaming(env, {
        text: input,
        speaker_voice: speakerVoice,
        output_format: 'pcm_16',
        parameters
      }, 'pcm');
    }

    // Batch mode (async /run + poll)
    const runpodRequest = {
      input: {
        text: input,
        speaker_voice: speakerVoice,
        stream: false,
        parameters
      }
    };

    const runUrl = normalizeRunpodUrl(env.RUNPOD_URL, '/run');
    const runResponse = await fetch(runUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${env.RUNPOD_API_KEY}`
      },
      body: JSON.stringify(runpodRequest)
    });

    if (!runResponse.ok) {
      const errorText = await runResponse.text();
      console.error(`RunPod /run failed: ${runResponse.status}. Body: ${errorText}`);
      return openaiError(`RunPod gateway error: ${runResponse.status}`, 'server_error', null, 500);
    }

    const jobData = await runResponse.json();
    return await pollJobStatus(jobData.id, env, response_format);
  } catch (error) {
    console.error('Worker error:', error);
    return openaiError(error.message, 'server_error', null, 500);
  }
}

async function pollJobStatus(jobId, env, response_format) {
  const statusUrl = normalizeRunpodUrl(env.RUNPOD_URL, '') + `/status/${jobId}`;
  const maxAttempts = 40;
  const pollInterval = 3000;

  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, pollInterval));

    const resp = await fetch(statusUrl, {
      headers: { 'Authorization': `Bearer ${env.RUNPOD_API_KEY}` }
    });

    if (!resp.ok) {
      if (i % 5 === 0) console.warn(`Status poll ${i} failed: ${resp.status}`);
      continue;
    }

    const data = await resp.json();

    if (data.status === 'COMPLETED') {
      return handleJobOutput(data.output, response_format);
    }

    if (data.status === 'FAILED') {
      console.error('Job failed on RunPod:', JSON.stringify(data, null, 2));
      return openaiError('Inference failed on backend', 'server_error', null, 500);
    }
  }

  return openaiError('Job timed out after 2 minutes', 'server_error', null, 504);
}

async function handleJobOutput(output, response_format) {
  const result = Array.isArray(output) ? output[output.length - 1] : output;
  if (!result || result.error) {
    console.error('Backend returned logical error:', result?.error || 'Empty output');
    return openaiError(result?.error || 'Backend failed', 'server_error', null, 500);
  }

  let audioBytes;
  if (result.url || result.audio_url) {
    const audioUrl = result.url || result.audio_url;
    const audioResp = await fetch(audioUrl);
    if (!audioResp.ok) return openaiError('Failed to fetch audio from storage', 'server_error', null, 502);
    audioBytes = await audioResp.arrayBuffer();
  } else if (result.audio_base64 || result.audio) {
    audioBytes = base64ToArrayBuffer(result.audio_base64 || result.audio);
  } else {
    return openaiError('No audio data in response', 'server_error', null, 500);
  }

  // Echo-TTS batch output is OGG Opus from S3.
  return new Response(audioBytes, {
    headers: {
      'Content-Type': pickBatchContentType(response_format),
      'Access-Control-Allow-Origin': '*',
      'X-Actual-Format': 'ogg-opus'
    }
  });
}

async function handleOpenAIStreaming(env, params, response_format) {
  const { text, speaker_voice, output_format, parameters } = params;
  const runUrl = normalizeRunpodUrl(env.RUNPOD_URL, '/run');
  const streamBaseUrl = normalizeRunpodUrl(env.RUNPOD_URL, '/stream');

  const runpodRequest = {
    input: {
      text,
      speaker_voice,
      stream: true,
      output_format,
      parameters
    }
  };

  const runResponse = await fetch(runUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${env.RUNPOD_API_KEY}`
    },
    body: JSON.stringify(runpodRequest)
  });

  if (!runResponse.ok) {
    return openaiError(`Stream submit failed: ${runResponse.status}`, 'server_error', null, 500);
  }

  const jobData = await runResponse.json();
  const streamUrl = `${streamBaseUrl}/${jobData.id}`;

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();

  (async () => {
    try {
      let lastPos = 0;
      let isFinished = false;
      const startTime = Date.now();
      const pollInterval = 2000;

      while (!isFinished && (Date.now() - startTime) < 300000) {
        const resp = await fetch(streamUrl, {
          headers: { 'Authorization': `Bearer ${env.RUNPOD_API_KEY}` }
        });
        if (!resp.ok) break;

        const data = await resp.json();
        const stream = (data.stream || []).concat(data.output || []);

        if (stream.length > lastPos) {
          for (const item of stream.slice(lastPos)) {
            const payload = item?.output || item;
            if (payload.audio_chunk) {
              await writer.write(new Uint8Array(base64ToArrayBuffer(payload.audio_chunk)));
            } else if (payload.error) {
              console.error('Stream item error:', payload.error);
            }
          }
          lastPos = stream.length;
        }

        if (data.status === 'COMPLETED' || data.status === 'FAILED') isFinished = true;
        if (!isFinished) await new Promise(r => setTimeout(r, pollInterval));
      }
    } catch (e) {
      console.error('Stream polling error:', e);
    } finally {
      await writer.close();
    }
  })();

  return new Response(readable, {
    headers: {
      'Content-Type': pickStreamContentType(response_format),
      'Transfer-Encoding': 'chunked',
      'Access-Control-Allow-Origin': '*'
    }
  });
}

async function handleStreamingTTS(request, env) {
  try {
    const p = await request.json();
    if (!p.text) return errorResponse('Missing text', 400);

    return handleOpenAIStreaming(env, {
      text: p.text,
      speaker_voice: p.speaker_voice || p.speaker || p.voice || (env.DEFAULT_VOICE || 'Dorota'),
      output_format: p.output_format || 'pcm_16',
      parameters: p.parameters || {}
    }, 'pcm');
  } catch (e) {
    return errorResponse(e.message, 500);
  }
}

function resolveSpeakerVoice(voice, body, env) {
  if (body.speaker_voice || body.speaker) return body.speaker_voice || body.speaker;

  const envMap = parseJson(env.VOICE_MAP);
  const map = envMap || DEFAULT_VOICE_MAP;

  if (map && map[voice]) return map[voice];
  return voice || env.DEFAULT_VOICE || 'Dorota';
}

function pickBatchContentType(response_format) {
  return 'audio/ogg';
}

function pickStreamContentType(response_format) {
  return 'audio/pcm';
}

function normalizeRunpodUrl(baseUrl, suffix) {
  const trimmed = baseUrl.replace(/\/(run|runsync|stream|status)$/, '').replace(/\/$/, '');
  return `${trimmed}${suffix}`;
}

function parseJson(value) {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch (e) {
    console.warn('Failed to parse JSON env var:', e);
    return null;
  }
}

function handleCORS() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    }
  });
}

function openaiError(message, type, param, status) {
  return new Response(
    JSON.stringify({ error: { message, type, param, code: null } }),
    { status, headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' } }
  );
}

function errorResponse(message, status) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }
  });
}

function base64ToArrayBuffer(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}
