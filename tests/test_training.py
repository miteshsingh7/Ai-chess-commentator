"""Unit tests for Stage 4: Training configurations and reproducibility."""

import tempfile
import torch
from chess_commentator.training.config import TrainingConfig
from chess_commentator.training.model_loader import get_torch_device
from chess_commentator.training.trainer import set_seed


def test_training_config_yaml_serialization():
    config = TrainingConfig(
        model_id="microsoft/Phi-3.5-mini-instruct",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
    )

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tf:
        config.to_yaml(tf.name)
        loaded = TrainingConfig.from_yaml(tf.name)

        assert loaded.model_id == "microsoft/Phi-3.5-mini-instruct"
        assert loaded.per_device_train_batch_size == 4
        assert loaded.gradient_accumulation_steps == 4
        assert loaded.learning_rate == 1e-4
        assert loaded.lora_r == 16


def test_torch_device_resolution():
    device = get_torch_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")


def test_reproducibility_seed():
    set_seed(1234)
    t1 = torch.rand(5)
    set_seed(1234)
    t2 = torch.rand(5)
    assert torch.allclose(t1, t2)
