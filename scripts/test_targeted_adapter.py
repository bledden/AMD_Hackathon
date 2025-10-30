#!/usr/bin/env python3
"""
Test targeted adapter - should fix general_knowledge failures
"""

import json
import torch
import time
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm import tqdm

print("Loading DeepSeek-R1-32B with targeted adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=768,
    dtype=None,
    load_in_4bit=False,
)

print("Loading minimal adapter...")
model = PeftModel.from_pretrained(model, "/home/rocm-user/AMD_Hackathon/models/minimal_adapter")
FastLanguageModel.for_inference(model)

# Load test data
print("Loading test data...")
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    test_data = json.load(f)[:200]

print(f"\nTesting with {len(test_data)} questions...")
print("="*60)

correct = 0
times = []
by_domain = {}

for item in tqdm(test_data):
    question = item['question']
    choices = item['choices']
    correct_answer = item['correct_answer']
    domain = item.get('domain', 'unknown')

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
            max_new_tokens=8,  # Slightly more tokens for robustness
            temperature=0.1,
            do_sample=False,
        )
    elapsed = time.time() - start
    times.append(elapsed)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_end = response.split("The answer is")[-1][:20]

    # Extract answer
    predicted = None
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_end:
            predicted = letter
            break

    is_correct = (predicted == correct_answer)
    if is_correct:
        correct += 1

    # Track by domain
    if domain not in by_domain:
        by_domain[domain] = {'correct': 0, 'total': 0}
    by_domain[domain]['total'] += 1
    if is_correct:
        by_domain[domain]['correct'] += 1

accuracy = (correct / len(test_data)) * 100
avg_time = sum(times) / len(times)
max_time = max(times)

print(f"\n{'='*60}")
print(f"RESULTS WITH MINIMAL ADAPTER:")
print(f"{'='*60}")
print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
print(f"Average time: {avg_time:.3f}s")
print(f"Max time: {max_time:.3f}s")
print(f"Passes 6s requirement: {'✅ YES' if max_time < 6.0 else '❌ NO'}")
print()

# Show performance on key domains
print("PERFORMANCE BY DOMAIN:")
gk_stats = by_domain.get('general_knowledge', {'correct': 0, 'total': 0})
if gk_stats['total'] > 0:
    gk_acc = (gk_stats['correct'] / gk_stats['total']) * 100
    print(f"  general_knowledge: {gk_acc:.1f}% ({gk_stats['correct']}/{gk_stats['total']})")

for domain, stats in sorted(by_domain.items(), key=lambda x: -x[1]['total'])[:5]:
    if domain != 'general_knowledge' and stats['total'] >= 5:
        acc = (stats['correct'] / stats['total']) * 100
        print(f"  {domain}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

print()
print("COMPARISON:")
print(f"Baseline:        73.0% (146/200)")
print(f"Minimal adapter: {accuracy:.1f}% ({correct}/200)")
print(f"Improvement:     {accuracy - 73.0:+.1f}%")
print("="*60)

results = {
    'accuracy': accuracy,
    'correct': correct,
    'total': len(test_data),
    'avg_time': avg_time,
    'max_time': max_time,
    'improvement': accuracy - 73.0,
    'by_domain': {k: {'accuracy': v['correct']/v['total']*100, **v} for k, v in by_domain.items()}
}

with open('/home/rocm-user/AMD_Hackathon/minimal_adapter_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to minimal_adapter_results.json")
