#!/usr/bin/env python3
"""
BALANCED TRAINING - Middle ground between ultra-minimal and aggressive
- 800 questions (not 100, not 6000)
- Moderate learning rate: 1e-5 (between 5e-6 and 5e-5)
- Medium rank: 64
- Batch size 2 with gradient accumulation
- Targeted on weak domains
"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

print(f"\n{'='*60}")
print("BALANCED TRAINING - MIDDLE GROUND APPROACH")
print(f"{'='*60}\n")

# Load targeted training data
with open("/home/rocm-user/AMD_Hackathon/data/targeted_training.json", 'r') as f:
    all_data = json.load(f)

# Take 800 questions, prioritizing general_knowledge
gk_questions = [q for q in all_data if q.get('domain') == 'general_knowledge']
other_questions = [q for q in all_data if q.get('domain') != 'general_knowledge']

# 75% general_knowledge (600), 25% other (200)
import random
random.seed(42)
training_data = random.sample(gk_questions, min(600, len(gk_questions))) + \
                random.sample(other_questions, min(200, len(other_questions)))
random.shuffle(training_data)

print(f"Training on {len(training_data)} questions")
print(f"  General knowledge: {sum(1 for q in training_data if q.get('domain') == 'general_knowledge')}")
print(f"  Other domains: {sum(1 for q in training_data if q.get('domain') != 'general_knowledge')}")
print()

dataset = Dataset.from_list(training_data)

print("Loading DeepSeek-R1-32B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=False,
)

print("Adding BALANCED LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Medium rank
    lora_alpha=64,  # 1:1 scaling
    use_rslora=False,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # All attention
    lora_dropout=0.05,  # Light dropout
    bias="none",
)

print(f"\nBALANCED training settings:")
print(f"  Questions: {len(training_data)}")
print(f"  Epochs: 3")
print(f"  Batch size: 2")
print(f"  Gradient accumulation: 2 (effective batch = 4)")
print(f"  Learning rate: 1e-5 (moderate)")
print(f"  LoRA rank: 64 (medium)")
print()

training_args = TrainingArguments(
    output_dir="/home/rocm-user/AMD_Hackathon/models/balanced_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,  # Effective batch size = 4
    warmup_steps=50,
    num_train_epochs=3,
    learning_rate=1e-5,  # Moderate - between 5e-6 and 5e-5
    fp16=False,
    bf16=True,
    logging_steps=50,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",  # Cosine decay for stability
    seed=42,
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

print("Starting BALANCED training...")
print(f"{'='*60}\n")

trainer.train()

print(f"\n{'='*60}")
print("Saving balanced adapter...")
model.save_pretrained("/home/rocm-user/AMD_Hackathon/models/balanced_adapter")
tokenizer.save_pretrained("/home/rocm-user/AMD_Hackathon/models/balanced_adapter")

print(f"✅ Training complete!")
print(f"✅ Adapter saved")
print(f"{'='*60}\n")

# Immediate sanity check
print("SANITY CHECK:")
print("="*60)

FastLanguageModel.for_inference(model)

test_prompt = """<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
What is 2+2?

A. 3
B. 4
C. 5
D. 6<|im_end|>
<|im_start|>assistant
The answer is """

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=8, temperature=0.1, do_sample=False)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split("The answer is")[-1][:20]

print(f"Test: 2+2=?")
print(f"Output: '{answer}'")
print(f"Token IDs: {outputs[0][-8:].tolist()}")

if '10000000' in answer or answer.strip() == '':
    print("❌ MODE COLLAPSE DETECTED!")
else:
    print("✅ Sanity check passed")

print("="*60)
