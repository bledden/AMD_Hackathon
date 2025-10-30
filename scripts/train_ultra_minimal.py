#!/usr/bin/env python3
"""
ULTRA-MINIMAL TRAINING - Last attempt to avoid mode collapse
- ONLY 100 general_knowledge questions
- ULTRA-LOW learning rate: 5e-6
- 5 epochs for stability
- Batch size 1 for careful learning
"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

print(f"\n{'='*60}")
print("ULTRA-MINIMAL TRAINING - ANTI-MODE-COLLAPSE")
print(f"{'='*60}\n")

# Load ONLY general_knowledge questions
with open("/home/rocm-user/AMD_Hackathon/data/targeted_training.json", 'r') as f:
    all_data = json.load(f)

gk_only = [q for q in all_data if q.get('domain') == 'general_knowledge']

# Take ONLY 100 questions, evenly spaced
step = len(gk_only) // 100
training_data = [gk_only[i] for i in range(0, len(gk_only), step)][:100]

print(f"Training on ONLY {len(training_data)} general_knowledge questions")
print("This forces the model to learn, not collapse\n")

dataset = Dataset.from_list(training_data)

print("Loading DeepSeek-R1-32B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=512,  # Shorter for stability
    dtype=None,
    load_in_4bit=False,
)

print("Adding MINIMAL LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # TINY rank to prevent overfitting
    lora_alpha=32,  # 1:1 scaling
    use_rslora=False,  # Disable RSLoRA
    target_modules=["q_proj", "v_proj"],  # ONLY attention, not MLP
    lora_dropout=0.1,  # Add dropout to prevent collapse
    bias="none",
)

print(f"\nULTRA-CONSERVATIVE training:")
print(f"  Questions: {len(training_data)}")
print(f"  Epochs: 5")
print(f"  Batch size: 1 (slowest, most careful)")
print(f"  Learning rate: 5e-6 (ultra-low)")
print(f"  LoRA rank: 32 (tiny)")
print()

training_args = TrainingArguments(
    output_dir="/home/rocm-user/AMD_Hackathon/models/minimal_adapter",
    per_device_train_batch_size=1,  # Single example at a time
    gradient_accumulation_steps=1,
    warmup_steps=10,
    num_train_epochs=5,  # More epochs, slower learning
    learning_rate=5e-6,  # ULTRA LOW
    fp16=False,
    bf16=True,
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="constant",  # No decay
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

print("Starting ULTRA-CAREFUL training...")
print(f"{'='*60}\n")

trainer.train()

print(f"\n{'='*60}")
print("Saving minimal adapter...")
model.save_pretrained("/home/rocm-user/AMD_Hackathon/models/minimal_adapter")
tokenizer.save_pretrained("/home/rocm-user/AMD_Hackathon/models/minimal_adapter")

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
