import json
import os
import torch
from pathlib import Path
from diffusers import ZImagePipeline
from dotenv import load_dotenv

load_dotenv()


class ImageGenerator:
    def __init__(self):
        print("Loading model on cuda...")

        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self.pipe.to("cuda")
        print("Model loaded!")

    def generate(self, prompt: str, output_path: str, seed: int = None) -> str:
        """Generate an image from prompt and save it."""
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator
        ).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path


def load_prompts(prompts_file: str) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def generate_images_from_prompts(
    prompts_file: str,
    output_dir: str = "data/dataset",
    start_index: int = 0
):
    """Generate images for all prompts in the file."""
    prompts = load_prompts(prompts_file)
    generator = ImageGenerator()

    metadata = []
    metadata_file = Path(output_dir) / "metadata.jsonl"

    # Load existing metadata if exists
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        start_index = len(metadata)
        print(f"Resuming from index {start_index}")

    for i, item in enumerate(prompts[start_index:], start=start_index):
        print(f"[{i+1}/{len(prompts)}] Generating image...")
        print(f"  Subject: {item['subject']}")

        image_filename = f"image_{i:04d}.png"
        image_path = Path(output_dir) / image_filename

        generator.generate(
            prompt=item["prompt"],
            output_path=str(image_path),
            seed=i
        )

        metadata.append({
            "file_name": image_filename,
            "subject": item["subject"],
            "prompt": item["prompt"]
        })

        # Save metadata after each image (for resume capability)
        with open(metadata_file, "w", encoding="utf-8") as f:
            for m in metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"  Saved: {image_path}")

    print(f"\nDone! Generated {len(metadata)} images in {output_dir}")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, default="data/prompts.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/dataset")
    args = parser.parse_args()

    generate_images_from_prompts(args.prompts_file, args.output_dir)
