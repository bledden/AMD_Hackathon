#!/usr/bin/env python3
"""
Train simple adapter (Q->A, no reasoning) - FAST training for tournament
"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

print(f"\n{'='*60}")
print("TRAINING SIMPLE Q→A ADAPTER (NO REASONING)")
print(f"{'='*60}\n")

# Load simple training data
print("Loading simple training data...")
with open("/home/rocm-user/AMD_Hackathon/data/simple_training_5k.json", 'r') as f:
    training_data = json.load(f)

print(f"Loaded {len(training_data)} training examples")

# Convert to HuggingFace dataset
dataset = Dataset.from_list(training_data)

print(f"\nLoading DeepSeek-R1-32B base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=1024,  # Shorter since no reasoning
    dtype=None,
    load_in_4bit=False,
)

# Add LoRA adapters
print("Adding LoRA adapters (RSLoRA, r=128)...")
model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    lora_alpha=256,
    use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,  # No dropout for faster training
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Training arguments - optimized for speed
print(f"\nTraining configuration:")
print(f"  Epochs: 2")
print(f"  Batch size: 4 (effective: 16 with grad accumulation)")
print(f"  Learning rate: 2e-4 (higher for faster convergence)")
print(f"  Output: /home/rocm-user/AMD_Hackathon/models/simple_adapter_5k")
print()

training_args = TrainingArguments(
    output_dir="/home/rocm-user/AMD_Hackathon/models/simple_adapter_5k",
    per_device_train_batch_size=4,  # Increased
    gradient_accumulation_steps=4,
    warmup_steps=30,
    num_train_epochs=2,
    learning_rate=2e-4,  # Higher learning rate
    fp16=False,
    bf16=True,
    logging_steps=25,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_strategy="epoch",
    save_total_limit=1,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
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
model.save_pretrained("/home/rocm-user/AMD_Hackathon/models/simple_adapter_5k")
tokenizer.save_pretrained("/home/rocm-user/AMD_Hackathon/models/simple_adapter_5k")

print(f"✓ Training complete!")
print(f"✓ Adapter saved to: /home/rocm-user/AMD_Hackathon/models/simple_adapter_5k")
print(f"{'='*60}\n")
