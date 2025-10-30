#!/usr/bin/env python3
"""
Test baseline with FINAL optimizations:
- max_seq_length=512 (truncate long questions)
- max_new_tokens=4 (only need 1 letter answer)
"""

import json
import torch
import time
from unsloth import FastLanguageModel
from tqdm import tqdm

print("Loading DeepSeek-R1-32B with FINAL OPTIMIZATIONS...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=512,  # REDUCED from 768 to truncate long prompts
    dtype=None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

# Load test data
print("Loading test data...")
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    test_data = json.load(f)[:200]

print(f"\nTesting FINAL BASELINE (max_seq=512, max_tokens=4)...")
print("="*60)

correct = 0
times = []
slow_count = 0
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

    # Truncate prompt to max 512 tokens
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start
    times.append(elapsed)

    if elapsed > 6.0:
        slow_count += 1
        slow_questions.append((idx, elapsed, question[:100]))

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
median_time = sorted(times)[len(times)//2]

print(f"\n{'='*60}")
print(f"FINAL BASELINE RESULTS:")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
print(f"\nSPEED METRICS:")
print(f"  Average: {avg_time:.3f}s")
print(f"  Median:  {median_time:.3f}s")
print(f"  Max:     {max_time:.3f}s")
print(f"  Questions >6s: {slow_count}")
print(f"\nTOURNAMENT COMPLIANCE:")
print(f"  Requirement: <6.0s per question")
print(f"  Status: {'✅ PASS - READY TO SUBMIT!' if max_time < 6.0 else '❌ FAIL'}")

if slow_questions:
    print(f"\n⚠️  SLOW QUESTIONS:")
    for idx, elapsed, q_text in slow_questions:
        print(f"  Q{idx}: {elapsed:.3f}s - {q_text}...")

print("="*60)

results = {
    'accuracy': accuracy,
    'correct': correct,
    'total': len(test_data),
    'avg_time': avg_time,
    'max_time': max_time,
    'passes_6s': max_time < 6.0,
    'slow_count': slow_count,
    'config': 'max_seq_length=512, max_new_tokens=4'
}

with open('/home/rocm-user/AMD_Hackathon/final_baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to final_baseline_results.json")

if max_time < 6.0:
    print("\n" + "="*60)
    print("🎉 BASELINE IS TOURNAMENT-READY!")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Max time: {max_time:.3f}s < 6.0s")
    print("   Ready to submit!")
    print("="*60)
