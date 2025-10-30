#!/usr/bin/env python3
"""
Analyze baseline failures and create targeted training data
"""

import json
from collections import Counter

# Load baseline test results
with open("/home/rocm-user/AMD_Hackathon/adapter_test_results.json") as f:
    results = json.load(f)

baseline_results = results['baseline']['results']

# Load the original test data to get question metadata
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/val_5k.json") as f:
    test_questions = json.load(f)[:200]

# Analyze failures
failed_indices = []
failed_domains = []

for i, (result, question) in enumerate(zip(baseline_results, test_questions)):
    if not result['is_correct']:
        failed_indices.append(i)
        failed_domains.append(question.get('domain', 'unknown'))

print(f"BASELINE FAILURE ANALYSIS")
print("="*60)
print(f"Total failed: {len(failed_indices)}/200 ({len(failed_indices)/2:.1f}%)")
print()

# Domain breakdown
domain_counts = Counter(failed_domains)
print("TOP FAILING DOMAINS:")
for domain, count in domain_counts.most_common(10):
    print(f"  {domain}: {count} failures")

print()
print("="*60)

# Now load ALL training data and filter for these weak domains
print("\nExtracting training data for weak domains...")

with open("/home/rocm-user/AMD_Hackathon/data/curriculum/train_45k.json") as f:
    all_training = json.load(f)

# Get top failing domains (those with 3+ failures)
weak_domains = [domain for domain, count in domain_counts.items() if count >= 3]
print(f"Targeting domains: {weak_domains[:5]}...")

# Extract questions from weak domains
targeted_training = [q for q in all_training if q.get('domain') in weak_domains]

print(f"Found {len(targeted_training)} training questions in weak domains")

# Also add some general questions for balance (20%)
general_sample_size = len(targeted_training) // 4
general_questions = [q for q in all_training if q.get('domain') not in weak_domains][:general_sample_size]

final_training = targeted_training + general_questions
print(f"Final training set: {len(final_training)} questions")

# Format as simple Q->A
simple_training = []
for item in final_training:
    question = item['question']
    choices = item['choices']
    correct_answer = item['correct_answer']

    choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(choices.items())])

    text = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
{question}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is {correct_answer}.<|im_end|>"""

    simple_training.append({
        "text": text,
        "domain": item.get('domain', 'unknown')
    })

# Save
with open("/home/rocm-user/AMD_Hackathon/data/targeted_training.json", 'w') as f:
    json.dump(simple_training, f, indent=2)

print(f"\n✅ Saved {len(simple_training)} targeted training examples")
print(f"✅ Output: /home/rocm-user/AMD_Hackathon/data/targeted_training.json")
print("="*60)
