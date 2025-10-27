#!/usr/bin/env python3
"""
Convert open-ended Q&A to MCQ format with AI-generated distractors
Uses DeepSeek-V3.1 GGUF to generate plausible wrong answers
Target: Convert 18,518 questions to MCQ format (~2 hours)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MCQConverter:
    """Convert Q&A to MCQ format with AI-generated distractors"""

    def __init__(self, use_llama_cpp: bool = True):
        """
        Initialize converter

        Args:
            use_llama_cpp: Use DeepSeek GGUF for high-quality distractors
        """
        self.use_llama_cpp = use_llama_cpp
        self.model = None

        if use_llama_cpp:
            try:
                from llama_cpp import Llama
                model_path = "/home/rocm-user/AMD_Hackathon/models/DeepSeek-V3.1-UD-TQ1_0.gguf"

                logging.info(f"🚀 Loading DeepSeek-V3.1 GGUF for distractor generation...")
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    n_ctx=2048,
                    n_batch=512,
                    verbose=False
                )
                logging.info("✅ Model loaded successfully")
            except Exception as e:
                logging.warning(f"⚠️  Failed to load llama.cpp: {e}")
                logging.info("💡 Falling back to heuristic distractor generation")
                self.use_llama_cpp = False

    def generate_distractors_ai(self, question: str, correct_answer: str, num_distractors: int = 3) -> List[str]:
        """
        Generate plausible wrong answers using AI

        Args:
            question: The question text
            correct_answer: The correct answer
            num_distractors: Number of wrong answers to generate

        Returns:
            List of distractor strings
        """
        if not self.model:
            return self.generate_distractors_heuristic(correct_answer, num_distractors)

        prompt = f"""Generate {num_distractors} plausible but incorrect answers for this question. The distractors should be similar in style and length to the correct answer, but clearly wrong.

Question: {question}

Correct Answer: {correct_answer}

Generate {num_distractors} incorrect answers that could plausibly be chosen by someone who doesn't know the right answer. Make them realistic distractors, not obviously wrong.

Distractors (one per line):"""

        try:
            output = self.model(
                prompt,
                max_tokens=200,
                temperature=0.8,
                top_p=0.9,
                stop=["\n\n", "Question:"],
                echo=False
            )

            text = output['choices'][0]['text'].strip()

            # Parse distractors (one per line)
            distractors = [line.strip() for line in text.split('\n') if line.strip()]

            # Clean up numbered/bulleted lists
            cleaned = []
            for d in distractors:
                # Remove leading numbers, bullets, dashes
                d = d.lstrip('0123456789.-•* ')
                if d and d != correct_answer and len(d) > 0:
                    cleaned.append(d)

            # Return exactly num_distractors, pad if needed
            if len(cleaned) >= num_distractors:
                return cleaned[:num_distractors]
            else:
                # Pad with heuristic distractors if AI didn't generate enough
                remaining = num_distractors - len(cleaned)
                cleaned.extend(self.generate_distractors_heuristic(correct_answer, remaining))
                return cleaned[:num_distractors]

        except Exception as e:
            logging.warning(f"AI distractor generation failed: {e}")
            return self.generate_distractors_heuristic(correct_answer, num_distractors)

    def generate_distractors_heuristic(self, correct_answer: str, num_distractors: int = 3) -> List[str]:
        """
        Generate distractors using heuristic rules (fallback)

        Args:
            correct_answer: The correct answer
            num_distractors: Number to generate

        Returns:
            List of heuristic distractors
        """
        distractors = []

        # Strategy 1: Common wrong answers for different types
        common_distractors = {
            'number': ['0', '1', '100', '1000', '50'],
            'person': ['Unknown', 'Not specified', 'Various authors', 'Multiple people'],
            'place': ['United States', 'Europe', 'Asia', 'Unknown location'],
            'date': ['1900', '2000', '1950', 'Unknown'],
            'yes_no': ['Yes', 'No', 'Maybe', 'Depends'],
        }

        # Detect type
        answer_lower = correct_answer.lower()
        if any(word in answer_lower for word in ['yes', 'no', 'true', 'false']):
            pool = common_distractors['yes_no']
        elif correct_answer.isdigit():
            pool = common_distractors['number']
        elif any(word in answer_lower for word in ['mr', 'mrs', 'dr', 'president']):
            pool = common_distractors['person']
        elif len(correct_answer.split()) == 1 and correct_answer[0].isupper():
            pool = common_distractors['place']
        else:
            # Generic distractors
            pool = [
                'None of the above',
                'Not applicable',
                'Unknown',
                'All of the above'
            ]

        # Sample distractors
        for _ in range(num_distractors):
            if pool:
                d = random.choice(pool)
                if d != correct_answer and d not in distractors:
                    distractors.append(d)

        # Pad if needed
        while len(distractors) < num_distractors:
            distractors.append(f"Option {len(distractors) + 1}")

        return distractors[:num_distractors]

    def convert_to_mcq(self, question_data: Dict) -> Dict:
        """
        Convert a single question to MCQ format

        Args:
            question_data: Question dict with 'question' and 'answer'

        Returns:
            Question dict with 'choices' and 'correct_answer' (A/B/C/D)
        """
        # If already MCQ, return as-is
        if 'choices' in question_data or 'options' in question_data:
            return question_data

        # Extract Q&A
        question = question_data.get('question', '')
        correct_answer = question_data.get('answer', '')

        if not question or not correct_answer:
            logging.warning(f"Skipping invalid question: {question[:50]}")
            return None

        # Generate distractors
        distractors = self.generate_distractors_ai(question, correct_answer, num_distractors=3)

        # Create all choices (correct + 3 distractors)
        all_choices = [correct_answer] + distractors

        # Shuffle to randomize correct answer position
        random.shuffle(all_choices)

        # Find correct answer position
        correct_index = all_choices.index(correct_answer)
        correct_letter = ['A', 'B', 'C', 'D'][correct_index]

        # Create choices dict
        choices = {
            'A': all_choices[0],
            'B': all_choices[1],
            'C': all_choices[2],
            'D': all_choices[3]
        }

        # Create MCQ question
        mcq_question = question_data.copy()
        mcq_question['choices'] = choices
        mcq_question['correct_answer'] = correct_letter
        mcq_question['original_answer'] = correct_answer
        mcq_question['needs_mcq_conversion'] = False
        mcq_question['mcq_converted'] = True

        return mcq_question

    def convert_batch(self, questions: List[Dict]) -> List[Dict]:
        """
        Convert a batch of questions to MCQ format

        Args:
            questions: List of question dicts

        Returns:
            List of MCQ-formatted questions
        """
        converted = []

        for q in tqdm(questions, desc="Converting to MCQ"):
            try:
                mcq = self.convert_to_mcq(q)
                if mcq:
                    converted.append(mcq)
            except Exception as e:
                logging.error(f"Error converting question: {e}")
                # Keep original question if conversion fails
                converted.append(q)

        return converted


def main():
    """Main conversion pipeline"""
    logging.info("=" * 80)
    logging.info("📝 MCQ CONVERSION - GENERATE DISTRACTORS FOR OPEN-ENDED Q&A")
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

    # Initialize converter
    converter = MCQConverter(use_llama_cpp=True)

    # Convert questions that need it
    logging.info(f"\n🔄 Converting {len(needs_conversion)} questions to MCQ format...")
    logging.info(f"   Estimated time: ~{len(needs_conversion)/10/60:.1f} minutes")

    converted = converter.convert_batch(needs_conversion)

    # Combine all questions
    final_dataset = already_mcq + converted

    logging.info(f"\n📊 Conversion results:")
    logging.info(f"   Total questions: {len(final_dataset)}")
    logging.info(f"   All MCQ format: {sum(1 for q in final_dataset if 'choices' in q)}")

    # Save
    logging.info(f"\n💾 Saving to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    logging.info("=" * 80)
    logging.info("✅ MCQ CONVERSION COMPLETE")
    logging.info(f"   Output: {output_path}")
    logging.info(f"   Ready for Chain-of-Thought generation!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
