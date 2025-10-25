#!/usr/bin/env python3
"""
Fine-tuning Script for Q&A Agent using Unsloth + AMD MI300X
Optimized for fast, memory-efficient training
"""

import json
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from training.configs.default_config import get_config


def load_dataset(data_path: str) -> Dataset:
    """Load dataset from JSON file"""
    print(f"Loading dataset from {data_path}...")

    with open(data_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")
    return Dataset.from_list(data)


def format_dataset(example):
    """Format examples for instruction tuning"""
    # Alpaca format
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": text}


def setup_model_and_tokenizer(model_config, lora_config):
    """Load model with Unsloth optimization"""
    print("=" * 60)
    print("Setting up model with Unsloth...")
    print("=" * 60)

    # Load model
    print(f"Loading model: {model_config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_name,
        max_seq_length=model_config.max_seq_length,
        dtype=model_config.dtype,
        load_in_4bit=model_config.load_in_4bit,
    )

    print("✓ Model loaded")

    # Add LoRA adapters
    print(f"Adding LoRA adapters (r={lora_config.r})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        target_modules=lora_config.target_modules,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
        use_rslora=lora_config.use_rslora,
        loftq_config=lora_config.loftq_config,
    )

    print("✓ LoRA adapters added")
    print("=" * 60)

    return model, tokenizer


def train(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    training_config,
    data_config,
):
    """Run training with SFTTrainer"""
    print("Setting up trainer...")

    # Create output directory
    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
        learning_rate=training_config.learning_rate,
        fp16=not torch.cuda.is_bf16_supported() if training_config.fp16 is None else training_config.fp16,
        bf16=torch.cuda.is_bf16_supported() if training_config.bf16 is None else training_config.bf16,
        logging_steps=training_config.logging_steps,
        optim=training_config.optim,
        weight_decay=training_config.weight_decay,
        lr_scheduler_type=training_config.lr_scheduler_type,
        seed=training_config.seed,
        output_dir=str(output_dir),
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        evaluation_strategy=training_config.evaluation_strategy if training_config.do_eval else "no",
        eval_steps=training_config.eval_steps if training_config.do_eval else None,
        report_to=training_config.report_to,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if training_config.do_eval else None,
        dataset_text_field=data_config.dataset_text_field,
        max_seq_length=data_config.max_seq_length,
        args=training_args,
    )

    # Print training info
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Train examples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val examples: {len(val_dataset)}")
    print(f"Batch size: {training_config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Max steps: {training_config.max_steps}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    print()

    # GPU stats before training
    if torch.cuda.is_available():
        print("GPU Status:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print()

    # Train
    print("Starting training...")
    print("=" * 60)
    start_time = datetime.now()

    trainer.train()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("=" * 60)
    print(f"✓ Training completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("=" * 60)

    return trainer


def save_model(model, tokenizer, output_dir: str):
    """Save final model"""
    save_path = Path(output_dir) / "final_model"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to {save_path}...")

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    print("✓ Model saved")

    # Also save as safetensors for faster loading
    print("Saving as 16-bit for inference...")
    model.save_pretrained_merged(
        str(save_path / "merged_16bit"),
        tokenizer,
        save_method="merged_16bit",
    )
    print("✓ 16-bit model saved")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Q&A agent with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "fast", "high_quality"],
        help="Training configuration preset",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.json",
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val.json",
        help="Path to validation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override max training steps"
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)
    model_config = config["model"]
    lora_config = config["lora"]
    training_config = config["training"]
    data_config = config["data"]

    # Override configs from args
    if args.train_data:
        data_config.train_data_path = args.train_data
    if args.val_data:
        data_config.val_data_path = args.val_data
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.max_steps:
        training_config.max_steps = args.max_steps

    print("\n" + "=" * 60)
    print("AMD Hackathon - Q&A Agent Fine-tuning")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {model_config.model_name}")
    print("=" * 60)
    print()

    # Load datasets
    train_dataset = load_dataset(data_config.train_data_path)
    val_dataset = load_dataset(data_config.val_data_path) if training_config.do_eval else None

    # Format datasets
    print("Formatting datasets...")
    train_dataset = train_dataset.map(format_dataset)
    if val_dataset:
        val_dataset = val_dataset.map(format_dataset)
    print("✓ Datasets formatted")
    print()

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(model_config, lora_config)

    # Train
    trainer = train(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        training_config,
        data_config,
    )

    # Save
    save_model(model, tokenizer, training_config.output_dir)

    print("\n" + "=" * 60)
    print("Training Complete! 🎉")
    print("=" * 60)
    print(f"Model saved to: {training_config.output_dir}/final_model")
    print("\nNext steps:")
    print("1. Test inference: python inference/generate_qa.py")
    print("2. Evaluate quality: python evaluation/evaluate.py")
    print("3. Iterate on dataset if needed")
    print("=" * 60)


if __name__ == "__main__":
    main()
