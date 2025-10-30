#!/usr/bin/env python3
"""
Test SAME question with baseline vs adapter side-by-side
Debug what's different
"""

import json
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# Load ONE test question
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    question = json.load(f)[0]

print("\n" + "="*60)
print("TEST QUESTION:")
print(question['question'])
print(f"Correct: {question['correct_answer']}")
print("="*60)

choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(question['choices'].items())])

prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
{question['question']}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

print("\n1. TESTING BASELINE (NO ADAPTER):")
print("="*60)

model_base, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=768,
    dtype=None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model_base)

inputs = tokenizer(prompt, return_tensors="pt").to(model_base.device)

with torch.no_grad():
    outputs = model_base.generate(**inputs, max_new_tokens=8, temperature=0.1, do_sample=False)

response_base = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_base.split("The answer is")[-1][:50])

print("\n2. TESTING WITH TARGETED ADAPTER:")
print("="*60)

# CRITICAL: Try loading adapter BEFORE setting inference mode
model_adapter, tokenizer2 = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=768,
    dtype=None,
    load_in_4bit=False,
)

# Load adapter BEFORE inference mode
print("Loading adapter...")
model_adapter = PeftModel.from_pretrained(model_adapter, "/home/rocm-user/AMD_Hackathon/models/targeted_adapter")

# THEN set inference mode
print("Setting inference mode...")
FastLanguageModel.for_inference(model_adapter)

inputs2 = tokenizer2(prompt, return_tensors="pt").to(model_adapter.device)

with torch.no_grad():
    outputs2 = model_adapter.generate(**inputs2, max_new_tokens=8, temperature=0.1, do_sample=False)

response_adapter = tokenizer2.decode(outputs2[0], skip_special_tokens=True)
print(response_adapter.split("The answer is")[-1][:50])

print("\n3. COMPARISON:")
print("="*60)
print(f"Baseline output:  '{response_base.split('The answer is')[-1][:20]}'")
print(f"Adapter output:   '{response_adapter.split('The answer is')[-1][:20]}'")
print("="*60)

# Check if adapter is actually being used
print("\n4. ADAPTER CHECK:")
print("="*60)
print(f"Baseline trainable params: {sum(p.numel() for p in model_base.parameters() if p.requires_grad):,}")
print(f"Adapter trainable params:  {sum(p.numel() for p in model_adapter.parameters() if p.requires_grad):,}")
print("="*60)
