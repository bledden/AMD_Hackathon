#!/usr/bin/env python3
"""Verify all questions have correct answers"""

import json
from pathlib import Path

# Load dataset
dataset_path = Path("data/final_training/complete_10k_no_generation.json")
with open(dataset_path) as f:
    data = json.load(f)

print(f"Total questions: {len(data)}")

# Check for correct answers
questions_with_answers = 0
answer_formats = {'list_index': 0, 'letter': 0, 'full_text': 0}
sample_questions = []

for i, q in enumerate(data):
    if 'correct_answer' in q and q['correct_answer']:
        questions_with_answers += 1

        # Check answer format
        answer = q['correct_answer']
        if isinstance(answer, int):
            answer_formats['list_index'] += 1
        elif answer in ['A', 'B', 'C', 'D']:
            answer_formats['letter'] += 1
        else:
            answer_formats['full_text'] += 1

        # Collect samples
        if len(sample_questions) < 3:
            sample_questions.append({
                'question': q['question'][:100],
                'answer': q['correct_answer'],
                'choices': q.get('choices', 'No choices')
            })

print(f"\n✅ Questions with correct answers: {questions_with_answers}/{len(data)}")
print(f"   Percentage: {questions_with_answers/len(data)*100:.1f}%")

print("\nAnswer formats found:")
for fmt, count in answer_formats.items():
    if count > 0:
        print(f"  {fmt}: {count}")

print("\nSample questions with answers:")
for i, sample in enumerate(sample_questions, 1):
    print(f"\n{i}. Question: {sample['question']}...")
    print(f"   Answer: {sample['answer']}")
    if isinstance(sample['choices'], dict) and sample['answer'] in sample['choices']:
        print(f"   Answer text: {sample['choices'][sample['answer']][:50]}...")