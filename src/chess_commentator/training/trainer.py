"""QLoRA fine-tuning orchestrator using HuggingFace TRL SFTTrainer."""

import os
import random
from typing import Optional, Dict, Any
import numpy as np
import torch
from chess_commentator.training.config import TrainingConfig
from chess_commentator.training.model_loader import (
    load_model_and_tokenizer,
    get_lora_config,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_qlora_training(
    config: TrainingConfig,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end QLoRA training and save adapter checkpoint."""
    from datasets import load_dataset
    from transformers import TrainingArguments

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Load data
    dataset_files = {"train": config.train_file}
    if config.val_file and os.path.exists(config.val_file):
        dataset_files["validation"] = config.val_file
    
    raw_dataset = load_dataset("json", data_files=dataset_files)

    # 2. Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(config, token=token, for_training=True)
    peft_config = get_lora_config(config)
    from peft import get_peft_model
    model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # 3. Tokenize datasets
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )

    tokenized_train = raw_dataset["train"].map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )
    tokenized_val = None
    if "validation" in raw_dataset:
        tokenized_val = raw_dataset["validation"].map(
            tokenize_fn,
            batched=True,
            remove_columns=raw_dataset["validation"].column_names,
        )

    from transformers import DataCollatorForLanguageModeling, Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Configure training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.evaluation_strategy if tokenized_val is not None else "no",
        optim=config.optim if torch.cuda.is_available() else "adamw_torch",
        fp16=config.fp16 or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        bf16=config.bf16 or (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        report_to="none",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # 5. Train and save
    train_result = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = train_result.metrics
    metrics["log_history"] = trainer.state.log_history
    return metrics
