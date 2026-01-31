import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()


def upload_dataset(dataset_dir: str, repo_name: str, private: bool = False):
    """Upload dataset to HuggingFace Hub."""
    api = HfApi(token=os.getenv("HF_TOKEN"))

    # Create repo if not exists
    try:
        create_repo(repo_name, repo_type="dataset", private=private, token=os.getenv("HF_TOKEN"))
        print(f"Created new dataset repo: {repo_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Repo {repo_name} already exists, uploading to it...")
        else:
            raise e

    # Upload the entire dataset folder
    api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_name,
        repo_type="dataset",
    )

    print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/dataset")
    parser.add_argument("--repo_name", type=str, required=True, help="e.g., username/pixel-art-dataset")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    upload_dataset(args.dataset_dir, args.repo_name, args.private)
