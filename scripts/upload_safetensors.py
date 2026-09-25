from huggingface_hub import HfApi

api = HfApi()
print("Starting direct upload of adapter_model.safetensors (35.7 MB)...", flush=True)
url = api.upload_file(
    path_or_fileobj=".kaggle_output_v10/phi35_adapter_clean/adapter_model.safetensors",
    path_in_repo="adapter_model.safetensors",
    repo_id="Mitex07/phi35-chess-adapter-v10",
    repo_type="model",
)
print("✓ Upload finished successfully! URL:", url)
