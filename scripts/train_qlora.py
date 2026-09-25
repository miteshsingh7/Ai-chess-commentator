"""CLI entrypoint for QLoRA fine-tuning."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from chess_commentator.training.config import TrainingConfig
from chess_commentator.training.trainer import run_qlora_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QLoRA fine-tuning on chess commentary dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_phi35.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override base model ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace access token.",
    )
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)
    if args.model_id:
        config.model_id = args.model_id
    if args.output_dir:
        config.output_dir = args.output_dir

    print(f"Starting QLoRA fine-tuning for model: {config.model_id}")
    print(f"Loading data from: {config.train_file}")
    print(f"Target modules: {config.target_modules}")

    try:
        metrics = run_qlora_training(config, token=args.hf_token)
        print(f"Training completed successfully! Metrics: {metrics}")
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
