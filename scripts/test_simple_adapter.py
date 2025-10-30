#!/usr/bin/env python3
"""
Test simple adapter (direct Q->A, no reasoning)
"""

import json
import torch
import time
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm import tqdm

# Load model with simple adapter
print("Loading DeepSeek-R1-32B with simple adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=False,
)

print("Loading simple adapter...")
model = PeftModel.from_pretrained(model, "/home/rocm-user/AMD_Hackathon/models/simple_adapter_5k")
FastLanguageModel.for_inference(model)

# Load test data
print("Loading test data...")
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    test_data = json.load(f)[:200]

print(f"\nTesting with {len(test_data)} questions using SIMPLE prompt...")
print("="*60)

correct = 0
times = []
failed_extractions = []

for i, item in enumerate(tqdm(test_data)):
    question = item['question']
    choices = item['choices']
    correct_answer = item['correct_answer']

    choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(choices.items())])

    # Simple prompt matching training format
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
            max_new_tokens=4,  # Just need the letter
            temperature=0.1,
            do_sample=False,
        )
    elapsed = time.time() - start
    times.append(elapsed)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    predicted = None
    response_end = response.split("The answer is")[-1][:10]

    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_end:
            predicted = letter
            break

    if predicted == correct_answer:
        correct += 1
    elif predicted is None:
        failed_extractions.append({
            'question_num': i,
            'response_end': response_end
        })

accuracy = (correct / len(test_data)) * 100
avg_time = sum(times) / len(times)
max_time = max(times)

print(f"\n{'='*60}")
print(f"RESULTS WITH SIMPLE ADAPTER:")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
print(f"Average time: {avg_time:.3f}s")
print(f"Max time: {max_time:.3f}s")
print(f"Passes 6s requirement: {'✅ YES' if max_time < 6.0 else '❌ NO'}")
print(f"Failed extractions: {len(failed_extractions)}")
print(f"{'='*60}")

# Compare to baseline
print("\nCOMPARISON TO BASELINE:")
print(f"Baseline accuracy: 73.0%")
print(f"Simple adapter:    {accuracy:.1f}%")
print(f"Improvement:       {accuracy - 73.0:+.1f}%")
print("="*60)

results = {
    'accuracy': accuracy,
    'correct': correct,
    'total': len(test_data),
    'avg_time': avg_time,
    'max_time': max_time,
    'failed_extractions': len(failed_extractions),
    'improvement_over_baseline': accuracy - 73.0
}

with open('/home/rocm-user/AMD_Hackathon/simple_adapter_test.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to simple_adapter_test.json")
