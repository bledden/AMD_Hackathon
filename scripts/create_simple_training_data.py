#!/usr/bin/env python3
"""
Create simple Q->A training data (NO reasoning, just direct answers)
"""

import json
from pathlib import Path

# Load the original training data
print("Loading training data...")
with open("/home/rocm-user/AMD_Hackathon/data/curriculum/train_45k.json", 'r') as f:
    data = json.load(f)

# Sample 5000 questions evenly
step = len(data) // 5000
sampled_data = [data[i] for i in range(0, len(data), step)][:5000]

print(f"Sampled {len(sampled_data)} questions")

# Convert to simple training format
simple_training = []

for item in sampled_data:
    question = item['question']
    choices = item['choices']
    correct_answer = item['correct_answer']

    # Format choices
    choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(choices.items())])

    # Simple format: Question -> Direct answer (NO REASONING)
    text = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
{question}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is {correct_answer}.<|im_end|>"""

    simple_training.append({
        "text": text,
        "domain": item.get('domain', 'unknown'),
        "id": item.get('id', f'q_{len(simple_training)}')
    })

# Save
output_path = Path("/home/rocm-user/AMD_Hackathon/data/simple_training_5k.json")
with open(output_path, 'w') as f:
    json.dump(simple_training, f, indent=2)

print(f"\n✓ Created {len(simple_training)} simple training examples")
print(f"✓ Saved to: {output_path}")

# Show sample
print("\nSample training example:")
print("="*60)
print(simple_training[0]['text'])
print("="*60)
