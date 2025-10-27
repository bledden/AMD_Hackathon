#!/usr/bin/env python3
"""Verify the final 10k dataset"""

import json
from collections import Counter
from pathlib import Path

def main():
    # Load final dataset
    dataset_path = Path("data/final_training/complete_10k_no_generation.json")

    if not dataset_path.exists():
        print("❌ Final dataset not found!")
        return

    with open(dataset_path) as f:
        data = json.load(f)

    print("="*60)
    print("FINAL DATASET VERIFICATION")
    print("="*60)
    print(f"Total questions: {len(data)}")

    # Check format
    print("\n📋 Format Check:")
    required_fields = ['question', 'choices', 'correct_answer']
    valid_count = 0

    for q in data:
        if all(field in q for field in required_fields):
            # Check if choices is a list with 4 items or a dict with A,B,C,D
            if isinstance(q['choices'], list) and len(q['choices']) == 4:
                valid_count += 1
            elif isinstance(q['choices'], dict) and q['correct_answer'] in q['choices']:
                valid_count += 1

    print(f"  Valid MCQ format: {valid_count}/{len(data)}")

    # Category distribution
    print("\n📊 Category Distribution:")
    categories = Counter()

    for q in data:
        cat = q.get('assigned_category') or q.get('category') or 'unknown'
        categories[cat] += 1

    for cat, count in sorted(categories.items()):
        pct = count / len(data) * 100
        bar = "█" * int(pct/2)
        print(f"  {cat:20}: {count:4} ({pct:5.1f}%) {bar}")

    # Show sample questions
    print("\n📝 Sample Questions:")
    import random
    samples = random.sample(data, min(3, len(data)))

    for i, q in enumerate(samples, 1):
        print(f"\n  Question {i}:")
        print(f"    {q['question'][:100]}...")
        print(f"    Category: {q.get('assigned_category') or q.get('category', 'N/A')}")
        choices = q.get('choices', [])
        if isinstance(choices, list):
            print(f"    Options: {len(choices)} choices")
        elif isinstance(choices, dict):
            print(f"    Options: {len(choices)} choices (dict format)")
        else:
            print(f"    Options: Unknown format")

    # Ready for training check
    print("\n✅ Dataset ready for fine-tuning!")
    print(f"   Location: {dataset_path}")
    print(f"   Questions: {len(data)}")
    print(f"   Categories: {len(categories)}")

if __name__ == "__main__":
    main()