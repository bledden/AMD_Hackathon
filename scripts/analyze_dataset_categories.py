#!/usr/bin/env python3
"""Analyze MCQ dataset to see what categories exist."""

import json
from collections import Counter

# Load the 65K dataset
with open("/workspace/data/curriculum_65k_dataset.json", "r") as f:
    data = json.load(f)

# Count categories
categories = Counter()
for item in data:
    cat = item.get("category", "unknown")
    categories[cat] += 1

# Show all categories sorted by count
print("=== DATASET CATEGORIES (65K) ===")
for cat, count in categories.most_common():
    pct = count/len(data)*100
    print(f"{cat:40} {count:>6} questions ({pct:.1f}%)")

print(f"\nTotal: {len(data)} questions")

# Identify CS/Programming questions
cs_keywords = ["computer", "programming", "algorithm", "software", "code", "coding"]
cs_questions = 0
for item in data:
    cat = item.get("category", "").lower()
    if any(keyword in cat for keyword in cs_keywords):
        cs_questions += 1

print(f"\n=== CS/PROGRAMMING ANALYSIS ===")
print(f"CS-related questions: {cs_questions} ({cs_questions/len(data)*100:.1f}%)")
