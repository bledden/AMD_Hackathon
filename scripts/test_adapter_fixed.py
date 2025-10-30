#!/usr/bin/env python3
"""
Test distilled adapter with INCREASED token limit to allow full reasoning
"""

import json
import torch
import time
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm import tqdm

# Load model with adapter
print("Loading DeepSeek-R1-32B with distilled adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, "/home/rocm-user/AMD_Hackathon/models/distilled_adapter_3k")
FastLanguageModel.for_inference(model)

# Load test data
print("Loading test data...")
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json", 'r') as f:
    test_data = json.load(f)[:200]

print(f"\nTesting with {len(test_data)} questions using REASONING prompt (512 tokens)...")
print("="*60)

correct = 0
times = []
failed_extractions = []

for i, item in enumerate(tqdm(test_data)):
    question = item['question']
    choices = item['choices']
    correct_answer = item['correct_answer']

    choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(choices.items())])

    # Use the SAME prompt format as training
    prompt = f"""<|im_start|>system
You are an expert educator. When answering questions, explain your reasoning step-by-step inside <think> tags, then provide the final answer.<|im_end|>
<|im_start|>user
{question}

{choices_text}

Please think through this carefully and explain your reasoning before answering.<|im_end|>
<|im_start|>assistant
<think>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # INCREASED from 256 to allow full reasoning
            temperature=0.1,
            do_sample=False,
        )
    elapsed = time.time() - start
    times.append(elapsed)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer - try multiple patterns
    predicted = None
    response_lower = response.lower()

    # Try various extraction patterns
    for letter in ['A', 'B', 'C', 'D']:
        if predicted:
            break
        patterns = [
            f"answer is {letter}",
            f"answer: {letter}",
            f"correct answer is {letter}",
            f"the answer is {letter}",
            f"so the answer is {letter}",
            f"therefore, the answer is {letter}",
            f"answer would be {letter}",
            f"choose {letter}",
            f"option {letter}",
        ]
        for pattern in patterns:
            if pattern in response_lower:
                predicted = letter
                break

    if predicted == correct_answer:
        correct += 1
    elif predicted is None:
        failed_extractions.append({
            'question_num': i,
            'response_end': response[-200:]  # Last 200 chars
        })

accuracy = (correct / len(test_data)) * 100
avg_time = sum(times) / len(times)
max_time = max(times)

print(f"\n{'='*60}")
print(f"RESULTS WITH REASONING PROMPT (512 tokens):")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
print(f"Average time: {avg_time:.3f}s")
print(f"Max time: {max_time:.3f}s")
print(f"Passes 6s requirement: {'✅ YES' if max_time < 6.0 else '❌ NO'}")
print(f"Failed extractions: {len(failed_extractions)}")
print(f"{'='*60}")

results = {
    'accuracy': accuracy,
    'correct': correct,
    'total': len(test_data),
    'avg_time': avg_time,
    'max_time': max_time,
    'failed_extractions': len(failed_extractions)
}

with open('/home/rocm-user/AMD_Hackathon/adapter_fixed_test.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to adapter_fixed_test.json")

if failed_extractions and len(failed_extractions) <= 5:
    print("\nSample failed extractions:")
    for fail in failed_extractions[:3]:
        print(f"\nQuestion {fail['question_num']}:")
        print(fail['response_end'])
