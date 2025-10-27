#!/usr/bin/env python3
"""
FAST MCQ Conversion using heuristic distractor generation
Target: Convert 18,518 questions in ~5-10 minutes
Strategy: Use smart heuristics + existing dataset answers as distractors
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import random
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FastMCQConverter:
    """Fast MCQ converter using heuristic distractors"""

    def __init__(self, all_answers: List[str]):
        """
        Initialize with pool of existing answers for realistic distractors

        Args:
            all_answers: List of all answers in dataset (for sampling)
        """
        self.answer_pool = list(set(all_answers))  # Unique answers
        random.shuffle(self.answer_pool)

        # Group answers by type for better matching
        self.answers_by_type = self._categorize_answers(all_answers)

        logging.info(f"✅ Initialized with {len(self.answer_pool)} unique answers")

    def _categorize_answers(self, answers: List[str]) -> Dict[str, List[str]]:
        """Categorize answers by type"""
        categories = defaultdict(list)

        for ans in set(answers):
            ans_lower = ans.lower()

            # Categorize
            if ans.isdigit():
                categories['number'].append(ans)
            elif len(ans.split()) == 1 and ans[0].isupper() and len(ans) > 2:
                categories['name'].append(ans)
            elif any(word in ans_lower for word in ['yes', 'no', 'true', 'false']):
                categories['boolean'].append(ans)
            elif ans_lower in ['a', 'b', 'c', 'd', 'e']:
                categories['letter'].append(ans)
            elif len(ans) > 100:
                categories['long'].append(ans)
            else:
                categories['general'].append(ans)

        return categories

    def _get_answer_type(self, answer: str) -> str:
        """Detect answer type"""
        ans_lower = answer.lower()

        if answer.isdigit():
            return 'number'
        elif any(word in ans_lower for word in ['yes', 'no', 'true', 'false']):
            return 'boolean'
        elif len(answer.split()) == 1 and answer[0].isupper():
            return 'name'
        elif len(answer) > 100:
            return 'long'
        else:
            return 'general'

    def generate_distractors(self, correct_answer: str, question: str = '', num: int = 3) -> List[str]:
        """
        Generate distractors using heuristics + answer pool sampling

        Args:
            correct_answer: The correct answer
            question: Question text (for context)
            num: Number of distractors needed

        Returns:
            List of distractor strings
        """
        distractors = []
        answer_type = self._get_answer_type(correct_answer)

        # Strategy 1: Sample from similar answer types
        similar_answers = self.answers_by_type.get(answer_type, [])
        if similar_answers:
            candidates = [a for a in similar_answers if a != correct_answer]
            if len(candidates) >= num:
                distractors = random.sample(candidates, num)
                return distractors

        # Strategy 2: Add type-specific distractors
        if answer_type == 'number':
            try:
                val = int(correct_answer)
                distractors = [
                    str(val + random.randint(1, 10)),
                    str(val - random.randint(1, 10)),
                    str(val * 2)
                ]
            except:
                distractors = ['0', '1', '10', '100']

        elif answer_type == 'boolean':
            if 'yes' in correct_answer.lower():
                distractors = ['No', 'Maybe', 'Unknown']
            elif 'no' in correct_answer.lower():
                distractors = ['Yes', 'Maybe', 'Unknown']
            elif 'true' in correct_answer.lower():
                distractors = ['False', 'Uncertain', 'Depends']
            else:
                distractors = ['True', 'Uncertain', 'Depends']

        elif answer_type == 'name':
            # Sample random names from pool
            names = [a for a in self.answers_by_type['name'] if a != correct_answer]
            if len(names) >= num:
                distractors = random.sample(names, num)
            else:
                distractors = names + ['Unknown', 'Not specified']

        # Strategy 3: Sample from general pool
        while len(distractors) < num:
            candidate = random.choice(self.answer_pool)
            if candidate != correct_answer and candidate not in distractors:
                distractors.append(candidate)

        return distractors[:num]

    def convert_to_mcq(self, question_data: Dict) -> Dict:
        """Convert single Q&A to MCQ"""
        # Skip if already MCQ
        if 'choices' in question_data or 'options' in question_data:
            return question_data

        question = question_data.get('question', '')
        correct_answer = question_data.get('answer', '')

        if not question or not correct_answer:
            return None

        # Generate 3 distractors
        distractors = self.generate_distractors(correct_answer, question, num=3)

        # Create all choices
        all_choices = [correct_answer] + distractors
        random.shuffle(all_choices)

        # Find correct answer position
        correct_index = all_choices.index(correct_answer)
        correct_letter = ['A', 'B', 'C', 'D'][correct_index]

        # Create MCQ
        mcq = question_data.copy()
        mcq['choices'] = {
            'A': all_choices[0],
            'B': all_choices[1],
            'C': all_choices[2],
            'D': all_choices[3]
        }
        mcq['correct_answer'] = correct_letter
        mcq['original_answer'] = correct_answer
        mcq['needs_mcq_conversion'] = False
        mcq['mcq_method'] = 'heuristic_fast'

        return mcq


def main():
    """Main conversion pipeline"""
    logging.info("=" * 80)
    logging.info("⚡ FAST MCQ CONVERSION - HEURISTIC DISTRACTORS")
    logging.info("=" * 80)

    # Paths
    input_path = Path('data/comprehensive/full_50k_dataset.json')
    output_path = Path('data/comprehensive/full_50k_mcq.json')

    # Load dataset
    logging.info(f"📂 Loading dataset from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)

    logging.info(f"✅ Loaded {len(all_questions)} questions")

    # Separate MCQ-ready vs needs conversion
    already_mcq = [q for q in all_questions if 'choices' in q or 'options' in q]
    needs_conversion = [q for q in all_questions if q.get('needs_mcq_conversion', False)]

    logging.info(f"📊 Dataset breakdown:")
    logging.info(f"   Already MCQ: {len(already_mcq)} ({len(already_mcq)/len(all_questions)*100:.1f}%)")
    logging.info(f"   Needs conversion: {len(needs_conversion)} ({len(needs_conversion)/len(all_questions)*100:.1f}%)")

    # Extract all answers for distractor pool
    all_answers = [q.get('answer', '') for q in all_questions if 'answer' in q]
    all_answers += [choice for q in already_mcq for choice in q.get('choices', {}).values()]

    logging.info(f"📝 Building distractor pool from {len(all_answers)} answers...")

    # Initialize converter
    converter = FastMCQConverter(all_answers)

    # Convert questions
    logging.info(f"\n⚡ Converting {len(needs_conversion)} questions (FAST MODE)...")
    logging.info(f"   Estimated time: ~{len(needs_conversion)/100:.1f} minutes")

    converted = []
    for q in tqdm(needs_conversion, desc="Converting"):
        mcq = converter.convert_to_mcq(q)
        if mcq:
            converted.append(mcq)

    # Combine
    final_dataset = already_mcq + converted

    logging.info(f"\n📊 Conversion results:")
    logging.info(f"   Total questions: {len(final_dataset)}")
    logging.info(f"   All MCQ format: {sum(1 for q in final_dataset if 'choices' in q)}")
    logging.info(f"   Success rate: {len(converted)/len(needs_conversion)*100:.1f}%")

    # Save
    logging.info(f"\n💾 Saving to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    logging.info("=" * 80)
    logging.info("✅ FAST MCQ CONVERSION COMPLETE")
    logging.info(f"   Output: {output_path}")
    logging.info(f"   Total MCQ questions: {len(final_dataset)}")
    logging.info(f"   Ready for Chain-of-Thought generation!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
