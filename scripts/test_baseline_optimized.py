#!/usr/bin/env python3
"""
Test baseline with timeout protection to ensure <6s compliance
"""

import json
import torch
import time
from unsloth import FastLanguageModel
from tqdm import tqdm

print("Loading DeepSeek-R1-32B baseline...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=768,
    dtype=None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

# Load test data
print("Loading test data...")
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    test_data = json.load(f)[:200]

print(f"\nTesting BASELINE OPTIMIZED with {len(test_data)} questions...")
print("Testing different max_new_tokens settings...")
print("="*60)

# Test with different token limits
for max_tokens in [4, 8, 12]:
    print(f"\nTesting with max_new_tokens={max_tokens}...")

    correct = 0
    times = []
    timeouts = 0

    for item in tqdm(test_data, desc=f"max_tokens={max_tokens}"):
        question = item['question']
        choices = item['choices']
        correct_answer = item['correct_answer']

        choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(choices.items())])

        prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
{question}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - start
        times.append(elapsed)

        if elapsed > 6.0:
            timeouts += 1

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_end = response.split("The answer is")[-1][:30]

        # Extract answer
        predicted = None
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_end:
                predicted = letter
                break

        if predicted == correct_answer:
            correct += 1

    accuracy = (correct / len(test_data)) * 100
    avg_time = sum(times) / len(times)
    max_time = max(times)

    print(f"\n  Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
    print(f"  Avg time: {avg_time:.3f}s")
    print(f"  Max time: {max_time:.3f}s")
    print(f"  Timeouts (>6s): {timeouts}")
    print(f"  Compliance: {'✅ PASS' if max_time < 6.0 else '❌ FAIL'}")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("Pick the setting with highest accuracy that passes <6s requirement")
print("="*60)
