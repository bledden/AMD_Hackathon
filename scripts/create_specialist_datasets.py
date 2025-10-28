#!/usr/bin/env python3
"""
Create Specialist Datasets for Domain-Specific LoRA Training

Novel Approach: Train domain specialists, then merge via TIES/DARE
This solves latency (single merged model) while maintaining expertise
"""

import json
from collections import Counter

# Load merged dataset
print("Loading 65K merged dataset...")
with open('/workspace/data/curriculum/train_65k_merged.json') as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions")

# Define specialist categories
STEM_CATEGORIES = {
    'science', 'clinical_knowledge', 'professional_medicine',
    'high_school_biology', 'nutrition', 'elementary_mathematics',
    'high_school_mathematics', 'high_school_chemistry', 'high_school_physics',
    'college_mathematics', 'college_physics', 'college_chemistry',
    'college_biology', 'anatomy', 'virology', 'medical_genetics'
}

HUMANITIES_CATEGORIES = {
    'professional_law', 'professional_psychology', 'high_school_psychology',
    'moral_scenarios', 'philosophy', 'prehistory', 'moral_disputes',
    'jurisprudence', 'world_religions', 'sociology', 'high_school_government_and_politics',
    'high_school_world_history', 'high_school_us_history', 'high_school_european_history',
    'us_foreign_policy', 'public_relations', 'international_law', 'human_rights'
}

# Split into specialists
stem_questions = []
humanities_questions = []
general_questions = []

for q in data:
    cat = q.get('category', q.get('domain', 'unknown')).lower()

    if any(stem_cat in cat for stem_cat in STEM_CATEGORIES):
        stem_questions.append(q)
    elif any(hum_cat in cat for hum_cat in HUMANITIES_CATEGORIES):
        humanities_questions.append(q)
    else:
        # General knowledge / common sense / unknown
        general_questions.append(q)

print(f"\n📊 Dataset Split:")
print(f"   STEM: {len(stem_questions)} questions")
print(f"   Humanities: {len(humanities_questions)} questions")
print(f"   General: {len(general_questions)} questions")
print(f"   Total: {len(stem_questions) + len(humanities_questions) + len(general_questions)}")

# STEM gets some general knowledge (for context)
stem_with_context = stem_questions + general_questions[:len(general_questions)//2]
print(f"\n✅ STEM + Context: {len(stem_with_context)} questions")

# Humanities gets remaining general knowledge
humanities_with_context = humanities_questions + general_questions[len(general_questions)//2:]
print(f"✅ Humanities + Context: {len(humanities_with_context)} questions")

# Sort both by curriculum order (easy → hard)
stem_with_context.sort(key=lambda x: x.get('difficulty_score', x.get('curriculum_position', 0)))
humanities_with_context.sort(key=lambda x: x.get('difficulty_score', x.get('curriculum_position', 0)))

# Save specialist datasets
stem_path = '/workspace/data/curriculum/train_stem_specialist.json'
with open(stem_path, 'w') as f:
    json.dump(stem_with_context, f, indent=2)
print(f"\n💾 Saved STEM specialist: {stem_path}")

humanities_path = '/workspace/data/curriculum/train_humanities_specialist.json'
with open(humanities_path, 'w') as f:
    json.dump(humanities_with_context, f, indent=2)
print(f"💾 Saved Humanities specialist: {humanities_path}")

# Show category breakdown
print("\n📋 STEM Categories:")
stem_cats = Counter([q.get('category', q.get('domain', 'unknown')) for q in stem_questions])
for cat, count in sorted(stem_cats.items(), key=lambda x: -x[1])[:10]:
    print(f"   {cat}: {count}")

print("\n📋 Humanities Categories:")
hum_cats = Counter([q.get('category', q.get('domain', 'unknown')) for q in humanities_questions])
for cat, count in sorted(hum_cats.items(), key=lambda x: -x[1])[:10]:
    print(f"   {cat}: {count}")

print("\n✅ Specialist datasets created!")
print("\nNext steps:")
print("1. Train STEM specialist (RSLoRA r=128, ~3-4 hours)")
print("2. Train Humanities specialist (DoRA r=64, ~1-2 hours)")
print("3. Merge using TIES-Merging")
print("4. Ensemble with Model #1")
