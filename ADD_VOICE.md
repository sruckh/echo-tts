# Adding Voices (Dynamic Reference Audio) for `echo-tts` RunPod Serverless

This repository is the RunPod serverless inference portion of a larger 3-part system. It is already working for TTS. “Adding a voice” in this project means **making new reference audio files available to the serverless worker at runtime** (no training step).

`echo-tts` consumes reference audio by filename (or relative path) under `AUDIO_VOICES_DIR` and loads it per-request.

## What a “voice” is in this repo

- A **voice is a reference audio file** (the speaker you want to condition on).
- There is **no enrolled speaker embedding stored** by this worker; it loads the reference audio and conditions the generation.
- The handler enforces:
  - `speaker_voice` must resolve under `AUDIO_VOICES_DIR` (prevents path traversal outside the directory).
  - File must exist.
  - File extension must be one of: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.aac`, `.opus`.
- Reference audio is normalized in `inference.load_audio()`:
  - Converted to mono (channel-mean), resampled to `44_100 Hz`, peak-normalized.
  - `max_duration` is 300 seconds (longer files will be truncated at 5 minutes).

## Where voices live at runtime

- `AUDIO_VOICES_DIR` (default: `/runpod-volume/echo-tts/audio_voices`)
- This should be a **persistent, shared storage path** for your deployment:
  - If you want new voices to appear without redeploying, `AUDIO_VOICES_DIR` must point at a location you can update while the endpoint is running (commonly a RunPod Network Volume mounted at `/runpod-volume`).

`bootstrap.sh` ensures `AUDIO_VOICES_DIR` exists but does not populate it.

## Request contract (how callers select a voice)

Pass the reference filename as `speaker_voice` in the RunPod job input:

```json
{
  "input": {
    "text": "Hello from Echo-TTS.",
    "speaker_voice": "my-voice.ogg",
    "parameters": {
      "num_steps": 40,
      "cfg_scale_text": 3.0,
      "cfg_scale_speaker": 8.0,
      "seed": 1234
    }
  }
}
```

Notes:
- `speaker_voice` is treated as a filesystem path relative to `AUDIO_VOICES_DIR`. Keeping it to a single filename (no subdirectories) is simplest for operations and auditing.
- If `speaker_voice` is missing/`null`, the worker generates with no speaker conditioning (default voice).

## Dynamic voice workflow (no code changes in this repo)

To “add a voice” dynamically, **create a new reference audio file** and **publish it into `AUDIO_VOICES_DIR`**.

### 1) Prepare reference audio (recommended)

Recommended reference properties (quality/consistency, not hard requirements):
- 10–30 seconds of clean speech (30–60 seconds is fine if you want more stability).
- Minimal background noise/music; minimal reverb.
- Single speaker; no overlapping voices.
- Consistent loudness (the loader peak-normalizes, but clipping/noise still hurts).

Format:
- Any supported extension works; `.ogg` is a good default for storage.
- Sample rate/channels don’t matter (the loader resamples and converts to mono).

### 2) Choose a stable filename

The filename is the “voice key” this worker understands.

Recommendations:
- Use a deterministic slug: `voice_id.ogg` (e.g., `dorota.ogg`, `en_us_john_01.ogg`).
- Avoid spaces and special characters (callers must match the name exactly).
- If you need versioning, encode it in the filename: `dorota_v2.ogg`.

### 3) Publish into the mounted voices directory (atomic write)

Because the worker may try to read the file while you’re uploading it, publish using an atomic move:

1. Upload to a temporary name (e.g., `dorota.ogg.tmp`)
2. Rename/move to final (e.g., `dorota.ogg`)

On POSIX filesystems, rename is atomic and prevents partial reads.

### 4) Make it visible to all workers

This is the main operational requirement for “dynamic voices”:
- If you run a single worker with a shared volume: copy once → immediately available.
- If you run multiple workers that do not share the same `AUDIO_VOICES_DIR`: you must replicate the file to each worker’s storage (or switch to a shared volume).

No restart is required: the handler checks existence and loads the file per request.

## Suggested integration point with the rest of the 3-part system

This worker intentionally does not implement voice enrollment, approvals, or voice listing. Those belong upstream (your “voice manager” / API / UI layer).

Recommended upstream responsibilities:
- Accept user uploads / microphone recordings.
- Validate content (duration/type) and enforce access control/quota/approvals.
- Convert/normalize into one of the supported formats (optionally enforce `.ogg`).
- Choose the final `speaker_voice` filename and store metadata in your database:
  - `voice_id`, `display_name`, `speaker_voice` (filename), `owner`, `status`, timestamps.
- Publish the final file to `AUDIO_VOICES_DIR` (shared volume) using an atomic move.

## Deleting or disabling a voice

In this repo, a voice is “gone” when the file is gone:
- Removing the file from `AUDIO_VOICES_DIR` causes requests referencing it to fail with `speaker_voice '<name>' not found`.
- If you need “soft delete” semantics, enforce it upstream by not sending `speaker_voice` values for disabled voices.

## Health check (verifies the worker can see voices)

The handler supports a job action:

```json
{ "input": { "action": "health_check" } }
```

This reports configuration status and includes an `audio_files` check (how many files are visible under `AUDIO_VOICES_DIR`).

## Troubleshooting

- Error: `speaker_voice '<name>' not found`
  - File doesn’t exist under `AUDIO_VOICES_DIR`, name mismatch (case/spacing), or not replicated to this worker.
- Error: `Unsupported speaker_voice extension`
  - Convert to one of the supported suffixes listed above.
- “Voice doesn’t sound right”
  - Use a cleaner reference, avoid background noise, keep it single-speaker; consider increasing `cfg_scale_speaker` slightly.

