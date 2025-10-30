#!/usr/bin/env python3
"""
Debug: Check what TOKENS the adapter is generating
"""

import json
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# Load model with adapter
print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=768,
    dtype=None,
    load_in_4bit=False,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, "/home/rocm-user/AMD_Hackathon/models/targeted_adapter")
FastLanguageModel.for_inference(model)

# Load question
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    q = json.load(f)[0]

choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(q['choices'].items())])

prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
{q['question']}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n" + "="*60)
print("GENERATING WITH TOKEN OUTPUT...")
print("="*60)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        temperature=0.1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True
    )

# Get the generated token IDs
generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
print(f"\nGenerated token IDs: {generated_ids.tolist()}")

# Decode each token individually
print("\nToken-by-token decoding:")
for i, token_id in enumerate(generated_ids):
    token_text = tokenizer.decode([token_id])
    print(f"  Token {i}: ID={token_id.item()}, Text='{token_text}'")

# Full decode
full_text = tokenizer.decode(generated_ids)
print(f"\nFull decoded text: '{full_text}'")

# Now compare to TRAINING data format
print("\n" + "="*60)
print("CHECKING TRAINING FORMAT:")
print("="*60)

with open("/home/rocm-user/AMD_Hackathon/data/targeted_training.json", 'r') as f:
    train_sample = json.load(f)[0]

print("Sample training text (answer part):")
train_text = train_sample['text']
answer_part = train_text.split("The answer is")[-1].split("<|im_end|>")[0]
print(f"'{answer_part}'")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print(f"Expected format: 'The answer is C.<|im_end|>'")
print(f"Adapter generates: '{full_text}'")
print(f"Token IDs: {generated_ids.tolist()}")
print("="*60)
