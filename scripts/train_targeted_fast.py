#!/usr/bin/env python3
"""
FAST targeted training - 1 hour max
Focus on general_knowledge (40/54 failures)
"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

print(f"\n{'='*60}")
print("FAST TARGETED TRAINING - GENERAL KNOWLEDGE FOCUS")
print(f"{'='*60}\n")

# Load targeted data
with open("/home/rocm-user/AMD_Hackathon/data/targeted_training.json", 'r') as f:
    all_data = json.load(f)

# Sample 6000 questions (balanced: 80% general_knowledge, 20% other)
gk_questions = [q for q in all_data if q.get('domain') == 'general_knowledge']
other_questions = [q for q in all_data if q.get('domain') != 'general_knowledge']

# Sample evenly
gk_sample = gk_questions[::len(gk_questions)//4800][:4800]  # 80% = 4800
other_sample = other_questions[::len(other_questions)//1200][:1200]  # 20% = 1200

training_data = gk_sample + other_sample

print(f"Training on {len(training_data)} questions")
print(f"  General knowledge: {len(gk_sample)}")
print(f"  Other domains: {len(other_sample)}")

dataset = Dataset.from_list(training_data)

print(f"\nLoading DeepSeek-R1-32B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=768,  # Medium length
    dtype=None,
    load_in_4bit=False,
)

print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # SMALLER rank = faster training
    lora_alpha=128,
    use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print(f"\nTraining configuration:")
print(f"  Epochs: 1 (FAST)")
print(f"  Batch size: 8 (larger = faster)")
print(f"  Learning rate: 5e-5 (moderate)")
print()

training_args = TrainingArguments(
    output_dir="/home/rocm-user/AMD_Hackathon/models/targeted_adapter",
    per_device_train_batch_size=8,  # Larger batch
    gradient_accumulation_steps=2,  # Less accumulation
    warmup_steps=20,
    num_train_epochs=1,  # JUST 1 EPOCH
    learning_rate=5e-5,
    fp16=False,
    bf16=True,
    logging_steps=50,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_strategy="no",  # Don't save checkpoints (faster)
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=768,
    dataset_num_proc=4,
    packing=False,
    args=training_args,
)

print("Starting FAST training...")
print(f"{'='*60}\n")

trainer.train()

print(f"\n{'='*60}")
print("Saving adapter...")
model.save_pretrained("/home/rocm-user/AMD_Hackathon/models/targeted_adapter")
tokenizer.save_pretrained("/home/rocm-user/AMD_Hackathon/models/targeted_adapter")

print(f"✅ Training complete!")
print(f"✅ Adapter saved to: /home/rocm-user/AMD_Hackathon/models/targeted_adapter")
print(f"{'='*60}\n")
