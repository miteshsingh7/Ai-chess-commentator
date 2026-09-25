"""Model and tokenizer loader configured with 4-bit BitsAndBytes and PEFT LoRA."""

import os
from typing import Optional, Tuple, Any
import torch
from chess_commentator.training.config import TrainingConfig


def get_torch_device() -> torch.device:
    """Device-agnostic PyTorch device resolution."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_quantization_config(compute_dtype: Optional[torch.dtype] = None) -> Any:
    """Create BitsAndBytes 4-bit NormalFloat configuration."""
    from transformers import BitsAndBytesConfig

    if compute_dtype is None:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def get_lora_config(config: TrainingConfig) -> Any:
    """Create PEFT LoraConfig for target linear layers."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )


def load_model_and_tokenizer(
    config: TrainingConfig,
    token: Optional[str] = None,
    for_training: bool = True,
) -> Tuple[Any, Any]:
    """Load base model quantized to 4-bit and corresponding tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import prepare_model_for_kbit_training

    hf_token = token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Only load 4-bit quantization if CUDA is available (e.g. Kaggle GPU)
    if torch.cuda.is_available():
        bnb_config = get_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
        if for_training:
            model = prepare_model_for_kbit_training(model)
    else:
        # Fallback for CPU / MPS testing
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else None,
            torch_dtype=torch.float32,
            token=hf_token,
            trust_remote_code=True,
        )

    return model, tokenizer
