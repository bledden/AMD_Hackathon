#!/usr/bin/env python3
"""
Debug what the adapter is actually outputting
"""

import json
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# Load model with adapter
print("Loading model with adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

model = PeftModel.from_pretrained(model, "/home/rocm-user/AMD_Hackathon/models/distilled_adapter_3k")
FastLanguageModel.for_inference(model)

# Load one test question
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    test_data = json.load(f)

question = test_data[0]

print("\n" + "="*60)
print("TEST QUESTION:")
print("="*60)
print(f"Question: {question['question']}")
print(f"Correct answer: {question['correct_answer']}")
print()

choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(question['choices'].items())])

# Try the reasoning prompt
prompt = f"""<|im_start|>system
You are an expert educator. When answering questions, explain your reasoning step-by-step inside <think> tags, then provide the final answer.<|im_end|>
<|im_start|>user
{question['question']}

{choices_text}

Please think through this carefully and explain your reasoning before answering.<|im_end|>
<|im_start|>assistant
<think>"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("="*60)
print("GENERATING RESPONSE...")
print("="*60)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=False)

print("\nFULL RESPONSE:")
print("="*60)
print(response)
print("="*60)

# Show just the assistant part
if "<|im_start|>assistant" in response:
    assistant_part = response.split("<|im_start|>assistant")[-1]
    print("\nASSISTANT OUTPUT ONLY:")
    print("="*60)
    print(assistant_part)
    print("="*60)

# Try to extract answer
print("\nANSWER EXTRACTION ATTEMPTS:")
print("="*60)

for letter in ['A', 'B', 'C', 'D']:
    if f"answer is {letter}" in response.lower():
        print(f"✓ Found 'answer is {letter}'")
    if f"answer: {letter}" in response.lower():
        print(f"✓ Found 'answer: {letter}'")
    if f"correct answer is {letter}" in response.lower():
        print(f"✓ Found 'correct answer is {letter}'")
    if f"the answer is {letter}" in response.lower():
        print(f"✓ Found 'the answer is {letter}'")

print("="*60)
