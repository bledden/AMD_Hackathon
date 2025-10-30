#!/usr/bin/env python3
"""
Try MERGING adapter into base model before inference
"""

import json
import torch
from unsloth import FastLanguageModel

# Load test question
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    question = json.load(f)[0]

print("\n" + "="*60)
print("TESTING MERGED ADAPTER APPROACH")
print("="*60)

# Load base model with adapter path directly
print("Loading model with adapter merged...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/targeted_adapter",  # Load adapter directly as model
    max_seq_length=768,
    dtype=None,
    load_in_4bit=False,
)

FastLanguageModel.for_inference(model)

choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(question['choices'].items())])

prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
{question['question']}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=8, temperature=0.1, do_sample=False)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer_part = response.split("The answer is")[-1][:50]

print(f"\nQuestion: {question['question']}")
print(f"Correct answer: {question['correct_answer']}")
print(f"Model output: '{answer_part}'")
print("="*60)
