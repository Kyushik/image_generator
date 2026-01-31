# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixel art image generation pipeline that:
1. Uses GPT (OpenAI) to generate unique image prompts with Korean subjects
2. Generates pixel art images using Tongyi-MAI/Z-Image-Turbo model
3. Saves prompt-image pairs for HuggingFace dataset upload

## Setup

```bash
pip install -r requirements.txt
```

`.env` file required with API keys:
```
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
```

## Commands

```bash
# Run the full pipeline (generate prompts + images)
python main.py --num_images 10

# Generate prompts only
python generate_prompts.py --num_prompts 10

# Generate images from existing prompts
python generate_images.py --prompts_file data/prompts.jsonl

# Upload dataset to HuggingFace
python upload_to_hf.py --dataset_dir data/dataset --repo_name your-username/pixel-art-dataset
```

## Architecture

- `main.py` - Main pipeline orchestrator
- `generate_prompts.py` - LLM-based prompt generation with subject diversity tracking
- `generate_images.py` - Image generation using Z-Image-Turbo
- `upload_to_hf.py` - HuggingFace dataset upload utility
- `data/` - Generated prompts, images, and dataset files
- `subjects_history.json` - Tracks used subjects to avoid duplicates

## Key Design Decisions

- Subject uniqueness via OpenAI `text-embedding-3-small` + cosine similarity (threshold: 0.85)
- `subjects_history.json` stores subjects with their embeddings for similarity checking
- Image generation resumes from last index if interrupted (checks existing metadata.jsonl)
- Dataset format follows HuggingFace ImageFolder structure with metadata.jsonl
