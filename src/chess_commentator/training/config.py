"""Configuration dataclass for QLoRA training."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict
import yaml


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for QLoRA fine-tuning."""
    model_id: str = "microsoft/Phi-3.5-mini-instruct"
    train_file: str = "data/dataset/train.jsonl"
    val_file: Optional[str] = "data/dataset/val.jsonl"
    output_dir: str = "models/qlora_adapter"
    max_seq_length: int = 512

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # Optimization parameters (tuned for 16GB Kaggle T4)
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"

    # Runtime & Checkpoints
    seed: int = 42
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
