#!/usr/bin/env python3
"""
Convert Training Dataset to Question Pool
Use our existing 45K training questions as pre-generated tournament questions
"""

import json
import logging
import random
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_training_datasets():
    """Load all training datasets"""
    datasets = []

    # STEM dataset
    stem_file = Path("/workspace/data/curriculum/train_stem_specialist.json")
    if stem_file.exists():
        with open(stem_file) as f:
            stem_data = json.load(f)
            logging.info(f"Loaded {len(stem_data)} STEM questions")
            datasets.extend([(q, "STEM") for q in stem_data])

    # Humanities dataset
    hum_file = Path("/workspace/data/curriculum/train_humanities_specialist.json")
    if hum_file.exists():
        with open(hum_file) as f:
            hum_data = json.load(f)
            logging.info(f"Loaded {len(hum_data)} Humanities questions")
            datasets.extend([(q, "Humanities") for q in hum_data])

    # Math dataset
    math_file = Path("/workspace/data/curriculum/train_math_distill.json")
    if math_file.exists():
        with open(math_file) as f:
            math_data = json.load(f)
            logging.info(f"Loaded {len(math_data)} Math questions")
            datasets.extend([(q, "Math") for q in math_data])

    logging.info(f"Total questions loaded: {len(datasets)}")
    return datasets

def format_for_tournament(item, category):
    """
    Convert training format to tournament question format

    Tournament format:
    {
        "id": int,
        "question": str,
        "choices": {"A": str, "B": str, "C": str, "D": str},
        "correct_answer": str (A/B/C/D)
    }
    """
    try:
        # Handle different input formats
        if isinstance(item, dict):
            question = item.get('question', '')
            choices = item.get('choices', {})
            correct = item.get('answer', item.get('correct_answer', 'A'))
        else:
            # If it's just a string, skip it
            return None

        # Ensure choices is a dict with A, B, C, D
        if not isinstance(choices, dict):
            return None

        # Ensure we have all 4 choices
        if not all(k in choices for k in ['A', 'B', 'C', 'D']):
            return None

        # Ensure correct answer is A, B, C, or D
        if correct not in ['A', 'B', 'C', 'D']:
            return None

        return {
            "question": question.strip(),
            "choices": {
                "A": str(choices['A']).strip(),
                "B": str(choices['B']).strip(),
                "C": str(choices['C']).strip(),
                "D": str(choices['D']).strip(),
            },
            "correct_answer": correct,
            "category": category
        }

    except Exception as e:
        logging.warning(f"Skipping malformed question: {e}")
        return None

def main():
    logging.info("=" * 80)
    logging.info("🔄 CONVERTING TRAINING DATA TO QUESTION POOL")
    logging.info("=" * 80)

    # Load all training datasets
    raw_data = load_training_datasets()

    # Convert to tournament format
    questions = []
    for item, category in tqdm(raw_data, desc="Converting questions"):
        formatted = format_for_tournament(item, category)
        if formatted:
            questions.append(formatted)

    logging.info(f"Successfully formatted {len(questions)} questions")

    # Shuffle for randomness
    random.shuffle(questions)

    # Assign IDs
    for i, q in enumerate(questions, 1):
        q['id'] = i

    # Save full pool
    output_file = Path("/workspace/question_pool.json")
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=2)

    logging.info(f"✅ Saved {len(questions)} questions to {output_file}")

    # Statistics
    categories = {}
    for q in questions:
        cat = q.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    logging.info("\n📊 Question Pool Statistics:")
    logging.info(f"   Total: {len(questions)}")
    for cat, count in sorted(categories.items()):
        logging.info(f"   {cat}: {count}")

    # Create a smaller subset for testing (1000 questions)
    test_pool = questions[:1000]
    test_file = Path("/workspace/question_pool_test.json")
    with open(test_file, 'w') as f:
        json.dump(test_pool, f, indent=2)

    logging.info(f"\n✅ Also saved test pool ({len(test_pool)} questions) to {test_file}")

    logging.info("=" * 80)
    logging.info("✅ CONVERSION COMPLETE")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()
