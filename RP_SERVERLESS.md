# Runpod Serverless Implementation Plan

This file contains the detailed implementation plan for converting Echo-TTS to a Runpod serverless endpoint.

## Current Status: Handler-based serverless flow implemented

- New `handler.py` provides the RunPod serverless entrypoint with:
  - `text` input plus optional `speaker_voice` filename (loaded from `AUDIO_VOICES_DIR` with Gradio-supported audio extensions).
  - Sampler `parameters` passthrough.
  - Optional `session_id` for deterministic output filenames.
  - Outputs are compressed to OGG, uploaded to S3-compatible storage (e.g., B2), and return `filename`, presigned `url`, `s3_key`, and metadata.
- Dockerfile updated to use `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` and install Python/system deps up front.
- `bootstrap.sh` slimmed to: clone repo, venv deps, model warmup via inference loaders (HF token-aware), set up `audio_voices` and `output_audio`, then start `handler.py` (`--warmup` optional, then `--rp_serve_api`).
- Model downloads and handler loading now honor `HF_TOKEN` for gated weights.
- S3/B2 envs required for output uploads; no secrets hardcoded.

### Remaining considerations
- Ensure env vars are provided at deploy time:
  - `HF_TOKEN`
  - `AUDIO_VOICES_DIR` (default `/runpod-volume/echo-tts/audio_voices`)
  - `OUTPUT_AUDIO_DIR` (default `/runpod-volume/echo-tts/output_audio`)
  - `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`, `S3_REGION` (default `us-east-1`)
- Reference audio must be present in `AUDIO_VOICES_DIR`; no base64 audio in requests.
- Outputs rely on S3-compatible storage; presigned URLs expire (currently 1h).
- Tests/health checks: not added yet; add lightweight health if needed.

### Next steps (optional)
- Add simple health/ready endpoint or warm-start hook if required.
- Add cleanup policy for stale objects or shorten presign TTL if desired.
- Expand logging/metrics cautiously without leaking envs.

### Plan Location
The detailed implementation plan is stored at: `/root/.claude/plans/graceful-wondering-glacier.md`

### Key Points
1. **No Local Installation**: Everything runs in containers
2. **Stateless Design**: No persistent state between requests
3. **Memory Optimization**: 6GB+ GPU memory with dynamic configuration
4. **Cold Start Handling**: 30-60s model loading with caching

### Next Steps
1. Review and approve the implementation plan
2. Begin Phase 1: Core Serverless Worker development
3. Create Docker configuration
4. Deploy to Runpod for testing

---
*This file is tracked in .gitignore and contains sensitive planning details.*
