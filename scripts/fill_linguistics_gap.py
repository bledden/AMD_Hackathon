#!/usr/bin/env python3
"""Find and add linguistics questions from our existing dataset"""

import json
from pathlib import Path
from collections import Counter
import re

def find_linguistics_questions():
    """Find all linguistics questions in our massive dataset"""

    # Load the full dataset
    with open('data/massive_datasets/massive_training_ready.json') as f:
        all_questions = json.load(f)

    # Load current balanced dataset
    with open('data/final_training/complete_10k_no_generation.json') as f:
        current = json.load(f)

    # Create set of current questions
    current_questions = {q['question'] for q in current}

    # Find linguistics questions not yet included
    linguistics_questions = []

    linguistics_keywords = [
        'language', 'linguistic', 'grammar', 'syntax', 'semantics', 'phonetics',
        'morphology', 'dialect', 'translation', 'vocabulary', 'etymology',
        'phonology', 'pragmatics', 'sociolinguistic', 'lexicon', 'noun', 'verb',
        'adjective', 'adverb', 'pronoun', 'preposition', 'conjunction',
        'sentence structure', 'clause', 'phrase', 'tense', 'conjugation',
        'declension', 'inflection', 'derivation', 'compound word', 'prefix',
        'suffix', 'root word', 'cognate', 'loanword', 'pidgin', 'creole',
        'bilingual', 'multilingual', 'accent', 'intonation', 'stress pattern',
        'syllable', 'phoneme', 'morpheme', 'lexeme', 'idiom', 'metaphor',
        'metonymy', 'synecdoche', 'collocation', 'connotation', 'denotation',
        'register', 'discourse', 'speech act', 'conversation analysis'
    ]

    for q in all_questions:
        if q['question'] in current_questions:
            continue

        # Check question text and choices
        text = q.get('question', '').lower()
        choices_text = ' '.join(str(v).lower() for v in q.get('choices', {}).values())
        subject = q.get('subject', '').lower()
        full_text = text + ' ' + choices_text + ' ' + subject

        # Check for linguistics keywords
        is_linguistics = any(keyword in full_text for keyword in linguistics_keywords)

        if is_linguistics:
            linguistics_questions.append(q)

    print(f"Found {len(linguistics_questions)} linguistics questions not in current dataset")

    # Analyze the quality
    print("\nSample linguistics questions found:")
    for i, q in enumerate(linguistics_questions[:5], 1):
        print(f"\n{i}. {q['question'][:100]}...")
        print(f"   Subject: {q.get('subject', 'N/A')}")

    return linguistics_questions

def add_linguistics_to_dataset(target_count=100):
    """Add linguistics questions to improve coverage"""

    linguistics_pool = find_linguistics_questions()

    if len(linguistics_pool) < target_count:
        print(f"\n⚠️ Only {len(linguistics_pool)} linguistics questions available")
        target_count = len(linguistics_pool)

    # Select best linguistics questions
    selected = linguistics_pool[:target_count]

    # Load current dataset
    with open('data/final_training/complete_10k_no_generation.json') as f:
        current = json.load(f)

    print(f"\n📊 Current dataset: {len(current)} questions")
    print(f"➕ Adding: {len(selected)} linguistics questions")

    # Add linguistics questions
    for q in selected:
        q['assigned_category'] = 'linguistics'

    # Combine datasets
    enhanced = current + selected

    # If over 10k, remove some from overrepresented categories
    if len(enhanced) > 10000:
        print(f"\n⚖️ Balancing dataset back to 10,000 questions...")

        # Count categories
        category_counts = Counter()
        for q in enhanced:
            cat = q.get('assigned_category') or q.get('category', 'unknown')
            category_counts[cat] += 1

        # Find most overrepresented categories
        overrepresented = [cat for cat, count in category_counts.most_common()
                          if cat not in ['linguistics', 'unknown']][:3]

        # Remove some from overrepresented categories
        to_remove = len(enhanced) - 10000
        removed_per_cat = to_remove // len(overrepresented) + 1

        final_dataset = []
        removed_counts = {cat: 0 for cat in overrepresented}

        for q in enhanced:
            cat = q.get('assigned_category') or q.get('category', 'unknown')

            if cat in overrepresented and removed_counts[cat] < removed_per_cat:
                removed_counts[cat] += 1
                continue

            final_dataset.append(q)

            if len(final_dataset) == 10000:
                break

        enhanced = final_dataset
        print(f"✅ Balanced back to {len(enhanced)} questions")

    # Save enhanced dataset
    output_path = Path('data/final_training/enhanced_10k_with_linguistics.json')
    with open(output_path, 'w') as f:
        json.dump(enhanced, f, indent=2)

    print(f"\n💾 Enhanced dataset saved to: {output_path}")
    print(f"   Total questions: {len(enhanced)}")

    # Show new distribution
    category_counts = Counter()
    for q in enhanced:
        cat = q.get('assigned_category') or q.get('category', 'unknown')
        category_counts[cat] += 1

    print("\n📊 Updated Category Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / len(enhanced) * 100
        print(f"  {cat:20}: {count:4} ({pct:5.1f}%)")

    return enhanced

if __name__ == "__main__":
    print("="*60)
    print("FILLING LINGUISTICS GAP")
    print("="*60)

    # First find available linguistics questions
    linguistics_questions = find_linguistics_questions()

    if linguistics_questions:
        print(f"\n✅ Can add up to {len(linguistics_questions)} linguistics questions")
        print("\nEnhancing dataset with linguistics questions...")

        # Add linguistics questions (aim for ~200 to get 2% coverage)
        enhanced_dataset = add_linguistics_to_dataset(target_count=200)

        print("\n✅ Dataset enhanced with linguistics coverage!")
    else:
        print("\n❌ No additional linguistics questions found")