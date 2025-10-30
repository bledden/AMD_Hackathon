#!/usr/bin/env python3
"""
Baseline Test: DeepSeek-R1-Distill-Qwen-32B
Test accuracy and speed on sample questions to decide if adapter training is needed
"""

import json
import logging
import time
import random
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_model():
    """Load DeepSeek-R1-32B with speed optimizations"""
    logging.info("Loading DeepSeek-R1-Distill-Qwen-32B...")
    start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/workspace/models/deepseek_r1_qwen32b",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    elapsed = time.time() - start
    logging.info(f"✅ Model loaded in {elapsed:.1f}s")

    return model, tokenizer

def answer_question(model, tokenizer, question_data):
    """Answer a single MCQ question"""
    question = question_data['question']
    choices = question_data['choices']

    # Concise prompt for speed
    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

    prompt = f"""<|im_start|>system
Answer the multiple choice question. Output only the letter (A, B, C, or D).<|im_end|>
<|im_start|>user
{question}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    inference_time = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()

    # Extract answer
    answer = extract_answer(response)

    return answer, inference_time

def extract_answer(response):
    """Extract answer letter"""
    response_upper = response.upper().strip()

    if response_upper and response_upper[0] in ['A', 'B', 'C', 'D']:
        return response_upper[0]

    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_upper:
            return letter

    return "A"

def run_baseline_test(sample_size=200):
    """
    Run baseline accuracy test on sample questions

    Args:
        sample_size: Number of questions to test (200 = good statistical sample)
    """
    logging.info("=" * 80)
    logging.info("🧪 BASELINE TEST: DeepSeek-R1-Distill-Qwen-32B")
    logging.info("=" * 80)

    # Load question pool
    pool_file = Path("/workspace/question_pool.json")
    logging.info(f"Loading question pool from {pool_file}...")

    with open(pool_file) as f:
        all_questions = json.load(f)

    logging.info(f"Loaded {len(all_questions)} total questions")

    # Sample questions
    if len(all_questions) > sample_size:
        test_questions = random.sample(all_questions, sample_size)
    else:
        test_questions = all_questions

    logging.info(f"Testing on {len(test_questions)} questions")
    logging.info("")

    # Load model
    model, tokenizer = load_model()
    logging.info("")

    # Test each question
    results = []
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    inference_times = []

    logging.info(f"Running inference on {len(test_questions)} questions...")
    logging.info("")

    for i, q in enumerate(test_questions, 1):
        predicted, inf_time = answer_question(model, tokenizer, q)
        correct = q['correct_answer']
        is_correct = (predicted == correct)

        category = q.get('category', 'unknown')
        category_stats[category]['total'] += 1
        if is_correct:
            category_stats[category]['correct'] += 1

        results.append({
            'question_id': q['id'],
            'category': category,
            'predicted': predicted,
            'correct': correct,
            'is_correct': is_correct,
            'inference_time': inf_time
        })

        inference_times.append(inf_time)

        if i % 20 == 0:
            current_acc = sum(r['is_correct'] for r in results) / len(results) * 100
            avg_time = sum(inference_times) / len(inference_times)
            logging.info(f"   Progress: {i}/{len(test_questions)} | Accuracy: {current_acc:.1f}% | Avg time: {avg_time:.2f}s")

    logging.info("")
    logging.info("=" * 80)
    logging.info("📊 BASELINE TEST RESULTS")
    logging.info("=" * 80)

    # Overall accuracy
    total_correct = sum(r['is_correct'] for r in results)
    overall_accuracy = total_correct / len(results) * 100

    logging.info(f"\n📈 Overall Performance:")
    logging.info(f"   Total questions: {len(results)}")
    logging.info(f"   Correct: {total_correct}")
    logging.info(f"   Accuracy: {overall_accuracy:.2f}%")

    # Speed stats
    avg_time = sum(inference_times) / len(inference_times)
    max_time = max(inference_times)
    min_time = min(inference_times)

    logging.info(f"\n⚡ Speed Performance:")
    logging.info(f"   Average time: {avg_time:.2f}s")
    logging.info(f"   Min time: {min_time:.2f}s")
    logging.info(f"   Max time: {max_time:.2f}s")
    logging.info(f"   6s requirement: {'✅ PASS' if max_time < 6.0 else '❌ FAIL'}")

    # Category breakdown
    logging.info(f"\n📚 Accuracy by Category:")
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        cat_acc = stats['correct'] / stats['total'] * 100
        logging.info(f"   {category:20s}: {cat_acc:5.1f}% ({stats['correct']}/{stats['total']})")

    # Identify weak categories
    weak_categories = []
    for category, stats in category_stats.items():
        cat_acc = stats['correct'] / stats['total'] * 100
        if cat_acc < 75.0 and stats['total'] >= 5:  # At least 5 questions
            weak_categories.append((category, cat_acc, stats['total']))

    if weak_categories:
        logging.info(f"\n⚠️  Weak Categories (< 75% accuracy):")
        for cat, acc, total in sorted(weak_categories, key=lambda x: x[1]):
            logging.info(f"   {cat:20s}: {acc:5.1f}% ({total} questions)")

    # Save detailed results
    output_file = Path("/workspace/baseline_test_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'overall_accuracy': overall_accuracy,
            'total_questions': len(results),
            'correct': total_correct,
            'speed': {
                'average': avg_time,
                'min': min_time,
                'max': max_time,
                'passes_6s_requirement': max_time < 6.0
            },
            'category_stats': dict(category_stats),
            'weak_categories': weak_categories,
            'detailed_results': results
        }, f, indent=2)

    logging.info(f"\n💾 Detailed results saved to {output_file}")

    # Recommendation
    logging.info("")
    logging.info("=" * 80)
    logging.info("🎯 RECOMMENDATION")
    logging.info("=" * 80)

    if overall_accuracy >= 90.0:
        logging.info("✅ Base model is EXCELLENT (≥90% accuracy)")
        logging.info("   → Use as-is, NO adapter training needed!")
        logging.info("   → Focus on Q-Agent and tournament prep")
    elif overall_accuracy >= 80.0:
        if weak_categories:
            logging.info("⚠️  Base model is GOOD (80-90% accuracy) but has weak areas")
            logging.info(f"   → Consider training adapters on weak categories: {[c[0] for c in weak_categories]}")
            logging.info(f"   → Estimated improvement: +3-5% accuracy")
        else:
            logging.info("✅ Base model is GOOD (80-90% accuracy)")
            logging.info("   → Optional: Train adapters for +2-3% improvement")
            logging.info("   → Or use as-is to save time")
    else:
        logging.info("❌ Base model accuracy is LOW (<80%)")
        logging.info("   → STRONGLY recommend training adapters on all 41,724 questions")
        logging.info("   → Expected improvement: +5-10% accuracy")
        if weak_categories:
            logging.info(f"   → Priority areas: {[c[0] for c in weak_categories[:3]]}")

    if max_time >= 6.0:
        logging.info(f"\n⚠️  WARNING: Max inference time {max_time:.2f}s exceeds 6s limit!")
        logging.info("   → Need further optimization or fallback to 14B model")

    logging.info("=" * 80)

    return overall_accuracy, max_time, weak_categories

if __name__ == "__main__":
    accuracy, max_time, weak_cats = run_baseline_test(sample_size=200)
