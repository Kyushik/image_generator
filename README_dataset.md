---
language:
- en
- ko
license: mit
task_categories:
- text-to-image
tags:
- pixel-art
- image-generation
- fine-tuning
size_categories:
- n<1K
---

# Pixel Art Image Dataset

A dataset of pixel art style images with English prompts, designed for text-to-image model fine-tuning.

## Dataset Description

- **Images**: 492 pixel art style images (1024x1024 PNG)
- **Prompts**: English prompts describing each image with pixel art style suffixes
- **Source**: Generated using Tongyi-MAI/Z-Image-Turbo model
- **Subjects**: Korean-inspired creative subjects translated to English prompts

## Dataset Structure

| Field | Type | Description |
|-------|------|-------------|
| image | Image | Pixel art style image (1024x1024) |
| prompt | string | English prompt used to generate the image |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("mks0813/pixel_image_dataset")

# Access an example
example = dataset["train"][0]
image = example["image"]
prompt = example["prompt"]
```

## Prompt Format

Each prompt follows this structure:
- Subject description in English
- Fixed pixel art style suffix: "large, clearly visible pixels, chunky pixel blocks, low resolution look, limited color palette, no smooth gradients, no anti-aliasing, no blur, sharp pixel edges, retro 16-bit game style"

## License

MIT License
