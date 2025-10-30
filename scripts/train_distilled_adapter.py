#!/usr/bin/env python3
"""
Train adapter on distillation data with reasoning chains
"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

def load_distillation_data(data_path):
    """Load distillation data"""
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} training examples with reasoning")
    return data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train adapter on distillation data")
    parser.add_argument("--data", required=True, help="Path to distillation data JSON")
    parser.add_argument("--output", required=True, help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("TRAINING ADAPTER ON DISTILLATION DATA")
    print(f"{'='*60}\n")

    # Load distillation data
    distillation_data = load_distillation_data(args.data)

    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(distillation_data)

    print(f"\nLoading DeepSeek-R1-32B base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters (RSLoRA, r=128)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        lora_alpha=256,  # α/√r scaling for RSLoRA
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Training arguments
    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: 2 (effective: 8 with grad accumulation)")
    print(f"  Learning rate: 2e-5")
    print(f"  Output: {args.output}")
    print()

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        fp16=False,
        bf16=True,
        logging_steps=50,
        optim="adamw_torch",  # Use standard PyTorch optimizer (ROCm compatible)
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )

    # Train
    print("Starting training...")
    print(f"{'='*60}\n")

    trainer.train()

    # Save final model
    print(f"\n{'='*60}")
    print("Saving final adapter...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"✓ Training complete!")
    print(f"✓ Adapter saved to: {args.output}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
