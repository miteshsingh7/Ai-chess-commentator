"""Upload phi35-adapter-v10 to Hugging Face Hub under Mitex07/phi35-chess-adapter-v10."""

import os
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Mitex07/phi35-chess-adapter-v10"
SOURCE_DIR = Path(".kaggle_output_v10/phi35_adapter_clean")

def main():
    api = HfApi()
    print(f"Ensuring repository {REPO_ID} exists on Hugging Face...")
    api.create_repo(REPO_ID, exist_ok=True, repo_type="model")

    print(f"Uploading files from {SOURCE_DIR} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(SOURCE_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=["dataset-metadata.json", "training_args.bin"],
    )
    print(f"✓ Upload successful! View your model at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
