import os
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Mitex07/phi35-chess-adapter-v10"
SOURCE_DIR = Path(".kaggle_output_v10/phi35_adapter_clean")

api = HfApi()
files = [
    "adapter_config.json",
    "README.md",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "adapter_model.safetensors",
]

for fname in files:
    fpath = SOURCE_DIR / fname
    if not fpath.exists():
        print(f"Skipping missing: {fname}")
        continue
    size_mb = fpath.stat().st_size / (1024 * 1024)
    print(f"Uploading {fname} ({size_mb:.2f} MB)...", flush=True)
    api.upload_file(
        path_or_fileobj=str(fpath),
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"✓ Uploaded {fname}", flush=True)

print(f"\nAll files uploaded to https://huggingface.co/{REPO_ID}")
