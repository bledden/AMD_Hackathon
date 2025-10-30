#!/usr/bin/env python3
"""
QUICK TEST: Qwen2.5-7B-Instruct (our last hope!)
Test on 50 questions for speed + accuracy check
"""

import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

print("Loading Qwen2.5-7B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/rocm-user/AMD_Hackathon/models/qwen2.5_7b_instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("/home/rocm-user/AMD_Hackathon/models/qwen2.5_7b_instruct")

print("Loading test data (50 questions for quick check)...")
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    test_data = json.load(f)[:50]  # ONLY 50 for speed

print(f"\nQUICK TEST: Qwen2.5-7B on {len(test_data)} questions...")
print("="*60)

correct = 0
times = []

for item in tqdm(test_data):
    question = item['question']
    choices = item['choices']
    correct_answer = item['correct_answer']

    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions accurately."},
        {"role": "user", "content": f"{question}\n\n{choices_text}\n\nAnswer with only the letter (A, B, C, or D):"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512).to(model.device)

    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=4, temperature=0.1, do_sample=False)
    elapsed = time.time() - start
    times.append(elapsed)

    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Extract answer
    predicted = None
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response.upper():
            predicted = letter
            break

    if predicted == correct_answer:
        correct += 1

accuracy = (correct / len(test_data)) * 100
avg_time = sum(times) / len(times)
max_time = max(times)

print(f"\n{'='*60}")
print(f"QWEN2.5-7B QUICK TEST:")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
print(f"Avg time: {avg_time:.3f}s")
print(f"Max time: {max_time:.3f}s")
print(f"Passes <6s: {'✅ YES' if max_time < 6.0 else '❌ NO'}")
print("="*60)

if max_time < 6.0 and accuracy >= 60:
    print("\n🎉 QWEN2.5-7B LOOKS GOOD! Deploy this!")
else:
    print(f"\n⚠️  Issues: accuracy={accuracy:.1f}%, max_time={max_time:.3f}s")
