# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- **`inference.py`**: High-level inference API with model loading functions from HuggingFace, sampling pipelines, and audio processing utilities
- **`inference_blockwise.py`**: Blockwise/continuation generation for streaming and longer audio generation
- **`autoencoder.py`**: Fish Speech S1-DAC autoencoder for audio encoding/decoding (Apache-2.0 licensed)
- **`gradio_app.py`**: Web UI for interactive TTS generation with custom audio/text prompts and sampling parameters

### Model Architecture

The EchoDiT model processes three modalities:
- **Text**: Tokenized and embedded using a separate text encoder
- **Speaker**: Reference audio encoded into patches with dedicated speaker encoder
- **Latent**: Audio latent representations denoised through the diffusion process

The model uses classifier-free guidance (CFG) with independent scales for text and speaker conditioning.

## Common Development Commands

### Setup and Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and CUDA-capable GPU (8GB+ VRAM recommended).

### Running the Application

```bash
# Start Gradio web interface
python gradio_app.py

# Run inference examples (see bottom of inference.py)
python inference.py
```

### Low VRAM Configuration (8GB)

Edit `gradio_app.py`:
```python
FISH_AE_DTYPE = torch.bfloat16  # Instead of float32
DEFAULT_SAMPLE_LATENT_LENGTH = 576  # Reduce from 640
```

## Key Configuration Files

- **`requirements.txt`**: Python dependencies (PyTorch, Gradio, HuggingFace Hub, etc.)
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