# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 🛑 CRITICAL: DEPLOYMENT RULES

## ⚠️ ABSOLUTELY NO LOCAL INSTALLATION
**THIS PROJECT IS BEING CONVERTED TO RUNPOD SERVERLESS - DO NOT INSTALL LOCALLY**

- **NEVER** run `pip install -r requirements.txt` on the host machine
- **NEVER** install any dependencies locally
- **NEVER** run the application locally on the host
- **ALL** dependencies are installed during the Docker image build
- **ALL** development and testing happens in a container environment (or on your own dev machine), not on the RunPod host

## 🐳 Docker Command Guidelines
**DEPRECATED**: `docker-compose` (use `docker compose` instead)
**CORRECT**: `docker compose` (space-separated command)

```bash
# ❌ WRONG - Deprecated
docker-compose up

# ✅ CORRECT
docker compose up
```

## 🚀 Runpod Serverless Deployment
This project is being converted from a standalone application to a Runpod serverless endpoint. All development must follow these rules:

1. **Container-only development**: All code changes must be tested in the Docker container
2. **Build-time dependency installation**: Use `Dockerfile` (no runtime `pip install`, no volume venv)
3. **Serverless architecture**: Design for stateless, request-response pattern
4. **No persistent state**: Store all data in temporary container storage
5. **Resource optimization**: Minimize container size and startup time

# 🛑 STOP — Run codemap before ANY task

```bash
codemap .                     # Project structure
codemap --deps                # How files connect
codemap --diff                # What changed vs main
codemap --diff --ref <branch> # Changes vs specific branch
```

## Required Usage

**BEFORE starting any task**, run `codemap .` first.

**ALWAYS run `codemap --deps` when:**
- User asks how something works
- Refactoring or moving code
- Tracing imports or dependencies

**ALWAYS run `codemap --diff` when:**
- Reviewing or summarizing changes
- Before committing code
- User asks what changed
- Use `--ref <branch>` when comparing against something other than main

## Project Overview

Echo-TTS is a multi-speaker text-to-speech model with speaker reference conditioning. The project implements a DiT (Diffusion Transformer) architecture for generating audio conditioned on text prompts and optional speaker reference audio.

### Key Components

- **`model.py`**: Core EchoDiT transformer model implementation with rotary position embeddings, low-rank AdaLN adaptation, and multi-modal conditioning (text, speaker, latent)
- **`handler.py`**: RunPod serverless handler (this repo’s primary code); performs chunked generation and uploads audio to S3
- **Upstream Echo-TTS code**: Cloned during image build and pinned to a specific commit via `Dockerfile` (includes `inference.py`, model code, autoencoder, etc.)

### Model Architecture

The EchoDiT model processes three modalities:
- **Text**: Tokenized and embedded using a separate text encoder
- **Speaker**: Reference audio encoded into patches with dedicated speaker encoder
- **Latent**: Audio latent representations denoised through the diffusion process

The model uses classifier-free guidance (CFG) with independent scales for text and speaker conditioning.

## Common Development Commands

### 🐳 Container Development Only

```bash
# Build the container (pins upstream echo-tts at build time)
docker build -t echo-tts .

# Run the serverless worker locally (requires GPU)
docker run --gpus all echo-tts
```

### Running the Application

**IMPORTANT**: Only run inside the container environment

```bash
# This repo is a RunPod serverless worker image; it starts the worker via CMD.
# Requests are sent via RunPod endpoints (/run, /runsync).
```

### Text chunking (serverless)

- `handler.py` includes enhanced chunking for long prompts with audio boundary processing to reduce artifacts.
- Features audio-aware chunking, cross-fading between chunks, and silence normalization for smooth transitions.
- Control with `parameters.max_chars_per_chunk` (default `300`; set `0` to disable).
- Additional parameters: `enable_crossfade` (default `true`), `normalize_boundaries` (default `true`), `target_duration_seconds` (default `10.0`).

## Key Configuration Files

- **`text_presets.txt`**: Pre-defined text prompt templates for different speaking styles
- **`sampler_presets.json`**: Pre-configured sampling parameters for various generation scenarios
- **`audio_prompts/`**: Directory containing reference audio files with licensing information

## Usage Patterns

### Basic Inference Pipeline

1. Load models from HuggingFace using `load_model_from_hf()`, `load_fish_ae_from_hf()`, `load_pca_state_from_hf()`
2. Load speaker reference audio with `load_audio()` (optional, can be None)
3. Configure sampler with `sample_euler_cfg_independent_guidances()`
4. Generate audio using `sample_pipeline()`
5. Save output with `torchaudio.save()`

### Blockwise Generation

For streaming or longer generations, use `inference_blockwise.py` which provides:
- Chunked generation for memory efficiency
- Audio continuation capabilities
- Causal decoding support

### Speaker Conditioning

- Reference audio up to 5 minutes supported (10+ seconds typical)
- Use "Force Speaker" (KV scaling) for out-of-distribution text
- Scale values: 1.0 (baseline), 1.5 (default when enabled), adjust lower if possible

## Licensing

- Code: MIT License (except `autoencoder.py`: Apache-2.0)
- Model weights: CC-BY-NC-SA-4.0
- Audio outputs: CC-BY-NC-SA-4.0 (due to Fish Speech dependency)
- Audio prompts: See `audio_prompts/LICENSE`
