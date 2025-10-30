#!/usr/bin/env python3
"""
Test distilled adapter accuracy and speed
"""

import json
import torch
import time
from pathlib import Path
from unsloth import FastLanguageModel
from tqdm import tqdm

def load_model(adapter_path=None):
    """Load DeepSeek-R1-32B with optional adapter"""
    print(f"\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )

    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    FastLanguageModel.for_inference(model)
    return model, tokenizer

def answer_question(model, tokenizer, question_data):
    """Answer a multiple choice question"""
    question = question_data['question']
    choices = question_data['choices']
    correct_answer = question_data['correct_answer']

    # Format choices
    choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(choices.items())])

    # Simple prompt (no reasoning needed for answers, just final answer)
    prompt = f"""<|im_start|>system
Answer the multiple choice question. Output only the letter (A, B, C, or D).<|im_end|>
<|im_start|>user
{question}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            temperature=0.1,
            do_sample=False,
        )
    elapsed = time.time() - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer letter
    predicted = None
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response.split("The answer is")[-1][:10]:
            predicted = letter
            break

    is_correct = (predicted == correct_answer)

    return {
        'predicted': predicted,
        'correct': correct_answer,
        'is_correct': is_correct,
        'time': elapsed
    }

def run_test(model, tokenizer, test_data, model_name):
    """Run accuracy test"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}\n")

    results = []
    correct_count = 0
    times = []

    for item in tqdm(test_data, desc=f"Testing {model_name}"):
        result = answer_question(model, tokenizer, item)
        results.append(result)
        if result['is_correct']:
            correct_count += 1
        times.append(result['time'])

    accuracy = (correct_count / len(test_data)) * 100
    avg_time = sum(times) / len(times)
    max_time = max(times)

    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_data)})")
    print(f"Average time: {avg_time:.3f}s")
    print(f"Max time: {max_time:.3f}s")
    print(f"Passes 6s requirement: {'✅ YES' if max_time < 6.0 else '❌ NO'}")
    print(f"{'='*60}\n")

    return {
        'accuracy': accuracy,
        'correct': correct_count,
        'total': len(test_data),
        'avg_time': avg_time,
        'max_time': max_time,
        'results': results
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test distilled adapter")
    parser.add_argument("--adapter", help="Path to adapter directory")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON")
    parser.add_argument("--limit", type=int, default=200, help="Number of test questions")
    args = parser.parse_args()

    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]

    print(f"Testing with {len(test_data)} questions")

    # Test baseline
    print("\n" + "="*60)
    print("TEST 1: BASELINE (No Adapter)")
    print("="*60)
    model_baseline, tokenizer = load_model(adapter_path=None)
    baseline_results = run_test(model_baseline, tokenizer, test_data, "Baseline")

    # Clear model from memory
    del model_baseline
    torch.cuda.empty_cache()

    if args.adapter:
        # Test with adapter
        print("\n" + "="*60)
        print("TEST 2: WITH DISTILLED ADAPTER")
        print("="*60)
        model_adapter, tokenizer = load_model(adapter_path=args.adapter)
        adapter_results = run_test(model_adapter, tokenizer, test_data, "Distilled Adapter")

        # Comparison
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Baseline accuracy: {baseline_results['accuracy']:.1f}%")
        print(f"Adapter accuracy:  {adapter_results['accuracy']:.1f}%")
        print(f"Improvement:       {adapter_results['accuracy'] - baseline_results['accuracy']:+.1f}%")
        print(f"\nBaseline avg time: {baseline_results['avg_time']:.3f}s")
        print(f"Adapter avg time:  {adapter_results['avg_time']:.3f}s")
        print(f"Time difference:   {adapter_results['avg_time'] - baseline_results['avg_time']:+.3f}s")
        print("="*60)

        # Save results
        output = {
            'baseline': baseline_results,
            'adapter': adapter_results,
            'improvement': adapter_results['accuracy'] - baseline_results['accuracy']
        }

        with open('/home/rocm-user/AMD_Hackathon/adapter_test_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("\n✓ Results saved to adapter_test_results.json")

if __name__ == "__main__":
    main()
