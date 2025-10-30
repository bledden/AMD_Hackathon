#!/usr/bin/env python3
"""
Test baseline DeepSeek-R1-32B for speed compliance
Need: <6s per question for A-Agent
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

print(f"\nTesting BASELINE with {len(test_data)} questions...")
print("="*60)

correct = 0
times = []
slow_questions = []

for idx, item in enumerate(tqdm(test_data)):
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
            max_new_tokens=8,
            temperature=0.1,
            do_sample=False,
        )
    elapsed = time.time() - start
    times.append(elapsed)

    if elapsed > 6.0:
        slow_questions.append((idx, elapsed, question[:80]))

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_end = response.split("The answer is")[-1][:20]

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
min_time = min(times)

print(f"\n{'='*60}")
print(f"BASELINE MODEL PERFORMANCE:")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
print(f"\nSPEED METRICS:")
print(f"  Average time: {avg_time:.3f}s")
print(f"  Min time: {min_time:.3f}s")
print(f"  Max time: {max_time:.3f}s")
print(f"  Median time: {sorted(times)[len(times)//2]:.3f}s")
print(f"\nTOURNAMENT COMPLIANCE:")
print(f"  Requirement: <6.0s per question")
print(f"  Status: {'✅ PASS' if max_time < 6.0 else '❌ FAIL'}")

if slow_questions:
    print(f"\n⚠️  SLOW QUESTIONS ({len(slow_questions)} total):")
    for idx, elapsed, q_text in slow_questions[:5]:
        print(f"  Q{idx}: {elapsed:.3f}s - {q_text}...")

print("="*60)

# Save results
results = {
    'accuracy': accuracy,
    'correct': correct,
    'total': len(test_data),
    'avg_time': avg_time,
    'max_time': max_time,
    'min_time': min_time,
    'passes_6s_requirement': max_time < 6.0,
    'slow_question_count': len(slow_questions)
}

with open('/home/rocm-user/AMD_Hackathon/baseline_speed_test.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to baseline_speed_test.json")
