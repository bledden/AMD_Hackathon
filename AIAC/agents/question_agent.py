#!/usr/bin/env python3
"""
Tournament Question Agent
Selects pre-generated questions from pool (<10s requirement)
"""

import json
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load question pool once at module import
QUESTION_POOL_FILE = Path("/workspace/question_pool.json")
QUESTION_POOL = []
USED_QUESTION_IDS = set()

def load_question_pool():
    """Load question pool from disk"""
    global QUESTION_POOL

    logging.info(f"Loading question pool from {QUESTION_POOL_FILE}...")

    with open(QUESTION_POOL_FILE) as f:
        QUESTION_POOL = json.load(f)

    logging.info(f"Loaded {len(QUESTION_POOL)} questions")

def get_next_question():
    """
    Get next question from pool (random selection, no repeats)

    Returns:
        dict: {
            "id": int,
            "question": str,
            "choices": {"A": str, "B": str, "C": str, "D": str},
            "correct_answer": str
        }
    """
    # Load pool if not already loaded
    if not QUESTION_POOL:
        load_question_pool()

    # Get unused questions
    available = [q for q in QUESTION_POOL if q['id'] not in USED_QUESTION_IDS]

    if not available:
        # Reset if we've used all questions
        logging.info("All questions used, resetting pool...")
        USED_QUESTION_IDS.clear()
        available = QUESTION_POOL

    # Random selection
    question = random.choice(available)
    USED_QUESTION_IDS.add(question['id'])

    # Return only required fields
    return {
        "id": question['id'],
        "question": question['question'],
        "choices": question['choices'],
        "correct_answer": question['correct_answer']
    }

def main():
    """
    Main entry point for tournament
    Called by: python -m agents.question_agent

    Output: Writes question to outputs/questions.json
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.info("Q-Agent starting...")

    # Get question
    question = get_next_question()

    # Save to outputs/questions.json
    output_dir = Path("/workspace/AIAC/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "questions.json"

    # Append to file (tournament may ask for multiple questions)
    existing_questions = []
    if output_file.exists():
        with open(output_file) as f:
            try:
                existing_questions = json.load(f)
            except:
                existing_questions = []

    existing_questions.append(question)

    with open(output_file, 'w') as f:
        json.dump(existing_questions, f, indent=2)

    logging.info(f"✅ Question saved to {output_file}")
    logging.info(f"   Question ID: {question['id']}")
    logging.info(f"   Question: {question['question'][:80]}...")

if __name__ == "__main__":
    main()
