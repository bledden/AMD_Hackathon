#!/usr/bin/env python3
"""
Merge 50K existing curriculum dataset + 15K new questions = 65K total
Apply curriculum ordering to the merged dataset
"""

import json
import random

print("Loading 50K curriculum dataset...")
with open('/workspace/data/curriculum/train_45k.json') as f:
    existing_data = json.load(f)

print(f"Loaded {len(existing_data)} questions from existing curriculum dataset")

print("Loading 15K new questions...")
with open('/workspace/data/questions_100k_raw.json') as f:
    new_data = json.load(f)

print(f"Loaded {len(new_data)} new questions")

# Merge datasets
print("Merging datasets...")
merged_data = existing_data + new_data

print(f"Total merged: {len(merged_data)} questions")

# Sort by difficulty (curriculum ordering)
# Existing data already has curriculum_position, new data needs difficulty estimation
for idx, item in enumerate(merged_data):
    if 'curriculum_position' not in item:
        # Estimate difficulty based on category
        difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
        diff_score = difficulty_map.get(item.get('difficulty', 'medium'), 2)
        item['curriculum_position'] = len(existing_data) + idx
        item['difficulty_score'] = diff_score

# Sort by difficulty_score if exists, otherwise by curriculum_position
merged_data.sort(key=lambda x: x.get('difficulty_score', x.get('curriculum_position', 0)))

# Save merged dataset
output_path = '/workspace/data/curriculum/train_65k_merged.json'
with open(output_path, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"✅ Saved {len(merged_data)} questions to {output_path}")

# Category distribution
categories = {}
for q in merged_data:
    cat = q.get('category', q.get('domain', 'unknown'))
    categories[cat] = categories.get(cat, 0) + 1

print("\nCategory Distribution:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
