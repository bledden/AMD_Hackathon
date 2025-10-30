#!/usr/bin/env python3
"""
Debug what the simple adapter is actually outputting
"""

import json
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# Load model with adapter
print("Loading model with simple adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=False,
)

model = PeftModel.from_pretrained(model, "/home/rocm-user/AMD_Hackathon/models/simple_adapter_5k")
FastLanguageModel.for_inference(model)

# Load test question
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    question = json.load(f)[0]

print("\n" + "="*60)
print("TEST QUESTION:")
print("="*60)
print(f"Question: {question['question']}")
print(f"Correct answer: {question['correct_answer']}")
print()

choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(question['choices'].items())])

prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
{question['question']}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("="*60)
print("GENERATING RESPONSE WITH 4 TOKENS...")
print("="*60)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4,
        temperature=0.1,
        do_sample=False,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nFULL RESPONSE:")
print("="*60)
print(response)
print("="*60)

print("\nJUST THE ANSWER PART:")
print("="*60)
answer_part = response.split("The answer is")[-1] if "The answer is" in response else "NOT FOUND"
print(f"'{answer_part}'")
print("="*60)

# Try with more tokens
print("\n" + "="*60)
print("TRYING WITH 20 TOKENS...")
print("="*60)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.1,
        do_sample=False,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer_part = response.split("The answer is")[-1] if "The answer is" in response else response[-100:]
print(answer_part)
print("="*60)
