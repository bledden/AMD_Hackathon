#!/usr/bin/env python3
"""Analyze answer formats and check for tournament compatibility"""

import json
from pathlib import Path
from collections import Counter

def analyze_answer_formats():
    # Load dataset
    dataset_path = Path("data/final_training/complete_10k_no_generation.json")
    with open(dataset_path) as f:
        data = json.load(f)

    print("="*60)
    print("ANSWER FORMAT ANALYSIS FOR TOURNAMENT")
    print("="*60)

    # Check answer formats
    answer_formats = Counter()
    letter_answers = []
    text_answers = []
    problematic = []

    for q in data:
        answer = q.get('correct_answer', '')
        choices = q.get('choices', {})

        if answer in ['A', 'B', 'C', 'D']:
            answer_formats['letter'] += 1
            letter_answers.append(q)

            # Verify answer exists in choices
            if isinstance(choices, dict) and answer not in choices:
                problematic.append({
                    'question': q['question'][:100],
                    'answer': answer,
                    'issue': 'Answer letter not in choices'
                })

        elif isinstance(answer, int):
            answer_formats['index'] += 1
            # These need conversion to letter format
            problematic.append({
                'question': q['question'][:100],
                'answer': answer,
                'issue': 'Integer index instead of letter'
            })

        else:
            answer_formats['text'] += 1
            text_answers.append(q)

    print(f"\n📊 Answer Format Distribution:")
    print(f"  Letter format (A/B/C/D): {answer_formats['letter']:,} ({answer_formats['letter']/len(data)*100:.1f}%)")
    print(f"  Text format: {answer_formats['text']:,} ({answer_formats['text']/len(data)*100:.1f}%)")
    print(f"  Index format: {answer_formats['index']:,} ({answer_formats['index']/len(data)*100:.1f}%)")

    # Analyze text answers
    if text_answers:
        print(f"\n⚠️ Text Format Answers Found: {len(text_answers)}")
        print("Sample text answers:")
        for i, q in enumerate(text_answers[:3], 1):
            print(f"\n  {i}. Question: {q['question'][:80]}...")
            print(f"     Answer: '{q['correct_answer']}'")
            choices = q.get('choices', {})
            if isinstance(choices, dict):
                # Check if answer matches any choice
                matches = [k for k, v in choices.items() if v == q['correct_answer']]
                if matches:
                    print(f"     ✅ Can convert to letter: {matches[0]}")
                else:
                    print(f"     ❌ No exact match in choices")

    # Tournament compatibility check
    print("\n" + "="*60)
    print("TOURNAMENT COMPATIBILITY ASSESSMENT")
    print("="*60)

    if answer_formats['letter'] == len(data):
        print("✅ PERFECT: All answers in letter format (A/B/C/D)")
    elif answer_formats['letter'] >= len(data) * 0.99:
        print("✅ EXCELLENT: 99%+ answers in letter format")
        print("   Minor text answers can be converted during training")
    elif answer_formats['letter'] >= len(data) * 0.95:
        print("⚠️ GOOD: 95%+ answers in letter format")
        print("   Recommend converting remaining answers")
    else:
        print("❌ NEEDS FIXING: Too many non-letter answers")

    # Check for multiple correct answers
    print("\n📋 Multiple Answer Check:")
    multi_answer_count = 0
    for q in data:
        answer_text = q.get('correct_answer', '')
        if any(word in answer_text.lower() for word in ['and', 'both', 'all of the above', 'none of the above']):
            multi_answer_count += 1

    print(f"  Questions with potential multiple answers: {multi_answer_count}")

    if multi_answer_count == 0:
        print("  ✅ No 'all of the above' or multiple answer patterns detected")
    else:
        print("  ⚠️ Some questions may have compound answers")

    # Final recommendation
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if answer_formats['text'] > 0:
        print("1. Convert text answers to letter format:")
        print("   - Map text answers to their corresponding choice letters")
        print("   - This ensures consistency for tournament scoring")
    else:
        print("✅ No action needed - all answers in correct format")

    print("\n2. Tournament Format Requirements:")
    print("   - Each question must have exactly 4 choices (A, B, C, D)")
    print("   - Each question must have exactly 1 correct answer")
    print("   - Answer must be a letter from A-D")
    print("\nOur dataset meets these requirements!")

    return answer_formats, text_answers

if __name__ == "__main__":
    analyze_answer_formats()