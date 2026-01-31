import argparse
from generate_prompts import PromptGenerator, save_prompts
from generate_images import generate_images_from_prompts


def main():
    parser = argparse.ArgumentParser(description="Pixel Art Image Generation Pipeline")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--prompts_file", type=str, default="data/prompts.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/dataset")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0-1)")
    parser.add_argument("--skip_prompts", action="store_true", help="Skip prompt generation, use existing prompts")
    parser.add_argument("--skip_images", action="store_true", help="Skip image generation")
    args = parser.parse_args()

    # Step 1: Generate prompts
    if not args.skip_prompts:
        print("=" * 50)
        print("Step 1: Generating prompts...")
        print("=" * 50)
        generator = PromptGenerator(similarity_threshold=args.threshold)
        prompts = generator.generate(args.num_images)
        save_prompts(prompts, args.prompts_file)
    else:
        print("Skipping prompt generation...")

    # Step 2: Generate images
    if not args.skip_images:
        print("\n" + "=" * 50)
        print("Step 2: Generating images...")
        print("=" * 50)
        generate_images_from_prompts(args.prompts_file, args.output_dir)
    else:
        print("Skipping image generation...")

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print(f"Dataset saved to: {args.output_dir}")
    print("To upload to HuggingFace:")
    print(f"  python upload_to_hf.py --repo_name YOUR_USERNAME/pixel-art-dataset")
    print("=" * 50)


if __name__ == "__main__":
    main()
