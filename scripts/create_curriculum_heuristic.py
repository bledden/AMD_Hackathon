#!/usr/bin/env python3
"""
Create curriculum-ordered dataset using heuristic difficulty scoring
Fast approach without requiring additional model inference
Target: Process 50K questions in minutes
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def assess_difficulty_heuristic(question: Dict) -> float:
    """
    Assess difficulty using heuristic rules (no model needed)

    Args:
        question: Question dict

    Returns:
        Difficulty score 1.0-10.0 (1=easiest, 10=hardest)
    """
    score = 5.0  # Base difficulty

    q_text = question.get('question', '')
    choices = question.get('choices', {})
    source = question.get('source', '')
    domain = question.get('domain', '')

    # 1. Question length complexity
    q_words = len(q_text.split())
    if q_words > 50:
        score += 2.0
    elif q_words > 30:
        score += 1.0
    elif q_words < 10:
        score -= 1.0

    # 2. Choice complexity
    if choices:
        avg_choice_len = sum(len(str(c).split()) for c in choices.values()) / len(choices)
        if avg_choice_len > 15:
            score += 1.5
        elif avg_choice_len < 3:
            score -= 1.0

    # 3. Keywords indicating difficulty
    hard_keywords = [
        'analyze', 'synthesize', 'evaluate', 'compare', 'contrast',
        'infer', 'deduce', 'derive', 'prove', 'demonstrate',
        'explain why', 'justify', 'assess', 'critique'
    ]
    easy_keywords = [
        'what', 'when', 'where', 'who', 'which',
        'define', 'list', 'name', 'identify', 'state'
    ]

    q_lower = q_text.lower()
    hard_count = sum(1 for kw in hard_keywords if kw in q_lower)
    easy_count = sum(1 for kw in easy_keywords if kw in q_lower)

    score += hard_count * 0.5
    score -= easy_count * 0.3

    # 4. Source-based difficulty
    source_difficulty = {
        # MMLU subjects (by typical difficulty)
        'mmlu_abstract_algebra': 9.0,
        'mmlu_college_mathematics': 8.5,
        'mmlu_college_physics': 8.5,
        'mmlu_formal_logic': 8.0,
        'mmlu_machine_learning': 8.0,
        'mmlu_professional_law': 7.5,
        'mmlu_philosophy': 7.0,
        'mmlu_high_school_mathematics': 6.5,
        'mmlu_high_school_chemistry': 6.0,
        'mmlu_high_school_physics': 6.0,
        'mmlu_elementary_mathematics': 4.0,
        'mmlu_miscellaneous': 5.0,

        # Other sources
        'commonsense_qa': 6.5,  # Reasoning required
        'trivia_qa': 4.0,        # Fact recall
        'logiqa': 8.0,           # Logic problems
    }

    for src_key, src_difficulty in source_difficulty.items():
        if src_key in source.lower():
            score = (score + src_difficulty) / 2  # Average with heuristic
            break

    # 5. Domain-based difficulty
    domain_difficulty = {
        'abstract_algebra': 9.0,
        'formal_logic': 8.5,
        'philosophy': 7.5,
        'professional_law': 7.5,
        'common_sense_reasoning': 6.5,
        'general_knowledge': 4.0,
    }

    for dom_key, dom_difficulty in domain_difficulty.items():
        if dom_key in domain.lower():
            score = (score * 0.7 + dom_difficulty * 0.3)
            break

    # Clamp to 1-10
    return max(1.0, min(10.0, score))


def create_curriculum(questions: List[Dict]) -> List[Dict]:
    """
    Create curriculum-ordered dataset (easy → hard)

    Args:
        questions: List of question dicts

    Returns:
        Curriculum-ordered list with difficulty metadata
    """
    logging.info("📚 Assessing difficulty for all questions...")

    # Assess difficulty for each question
    questions_with_difficulty = []
    for i, q in enumerate(questions):
        difficulty = assess_difficulty_heuristic(q)
        questions_with_difficulty.append((q, difficulty))

        if (i + 1) % 10000 == 0:
            logging.info(f"   Processed {i+1}/{len(questions)} questions")

    logging.info("📊 Sorting by difficulty (easy → hard)...")

    # Sort by difficulty
    sorted_questions = sorted(questions_with_difficulty, key=lambda x: x[1])

    # Add curriculum metadata
    curriculum = []
    for i, (q, diff) in enumerate(sorted_questions):
        enhanced_q = q.copy()
        enhanced_q['difficulty_score'] = diff
        enhanced_q['curriculum_position'] = i
        enhanced_q['curriculum_percentile'] = (i / len(sorted_questions)) * 100
        curriculum.append(enhanced_q)

    # Statistics
    difficulties = [score for _, score in sorted_questions]
    logging.info(f"\n📊 Curriculum Statistics:")
    logging.info(f"   Total questions: {len(curriculum)}")
    logging.info(f"   Difficulty range: {min(difficulties):.1f} - {max(difficulties):.1f}")
    logging.info(f"   Mean difficulty: {sum(difficulties)/len(difficulties):.1f}")
    logging.info(f"   Easy (1-4): {sum(1 for d in difficulties if d < 4)} ({sum(1 for d in difficulties if d < 4)/len(difficulties)*100:.1f}%)")
    logging.info(f"   Medium (4-7): {sum(1 for d in difficulties if 4 <= d < 7)} ({sum(1 for d in difficulties if 4 <= d < 7)/len(difficulties)*100:.1f}%)")
    logging.info(f"   Hard (7-10): {sum(1 for d in difficulties if d >= 7)} ({sum(1 for d in difficulties if d >= 7)/len(difficulties)*100:.1f}%)")

    return curriculum


def split_train_val(curriculum: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """
    Split into training and validation sets (stratified by difficulty)

    Args:
        curriculum: Curriculum-ordered questions
        val_ratio: Validation set ratio (default 10%)

    Returns:
        (train_set, val_set)
    """
    logging.info(f"\n📂 Splitting into train/val ({(1-val_ratio)*100:.0f}%/{val_ratio*100:.0f}%)...")

    # Stratified split by difficulty tertiles
    easy = [q for q in curriculum if q['difficulty_score'] < 4]
    medium = [q for q in curriculum if 4 <= q['difficulty_score'] < 7]
    hard = [q for q in curriculum if q['difficulty_score'] >= 7]

    # Shuffle each group
    random.shuffle(easy)
    random.shuffle(medium)
    random.shuffle(hard)

    # Split each group
    val_easy = easy[:int(len(easy) * val_ratio)]
    train_easy = easy[int(len(easy) * val_ratio):]

    val_medium = medium[:int(len(medium) * val_ratio)]
    train_medium = medium[int(len(medium) * val_ratio):]

    val_hard = hard[:int(len(hard) * val_ratio)]
    train_hard = hard[int(len(hard) * val_ratio):]

    # Combine in curriculum order
    train_set = train_easy + train_medium + train_hard
    val_set = val_easy + val_medium + val_hard

    logging.info(f"✅ Training set: {len(train_set)} questions")
    logging.info(f"✅ Validation set: {len(val_set)} questions")

    return train_set, val_set


def main():
    """Main pipeline"""
    logging.info("=" * 80)
    logging.info("📚 CURRICULUM LEARNING - HEURISTIC DIFFICULTY ASSESSMENT")
    logging.info("=" * 80)

    # Paths
    input_path = Path('data/comprehensive/full_50k_mcq.json')
    output_train_path = Path('data/curriculum/train_45k.json')
    output_val_path = Path('data/curriculum/val_5k.json')
    output_full_path = Path('data/curriculum/full_50k_ordered.json')

    # Load dataset
    logging.info(f"\n📂 Loading dataset from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    logging.info(f"✅ Loaded {len(questions)} questions")

    # Create curriculum
    curriculum = create_curriculum(questions)

    # Split train/val
    train_set, val_set = split_train_val(curriculum, val_ratio=0.1)

    # Save
    output_train_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n💾 Saving curriculum datasets...")
    with open(output_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_set, f, indent=2, ensure_ascii=False)
    logging.info(f"   ✅ Training: {output_train_path}")

    with open(output_val_path, 'w', encoding='utf-8') as f:
        json.dump(val_set, f, indent=2, ensure_ascii=False)
    logging.info(f"   ✅ Validation: {output_val_path}")

    with open(output_full_path, 'w', encoding='utf-8') as f:
        json.dump(curriculum, f, indent=2, ensure_ascii=False)
    logging.info(f"   ✅ Full ordered: {output_full_path}")

    logging.info("\n" + "=" * 80)
    logging.info("✅ CURRICULUM CREATION COMPLETE")
    logging.info("=" * 80)
    logging.info(f"📊 Ready for LoRA training with replay buffer!")
    logging.info(f"   Next: Train on {len(train_set)} questions (curriculum order)")
    logging.info(f"   Validate on {len(val_set)} questions (difficulty-stratified)")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
