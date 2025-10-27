#!/usr/bin/env python3
"""
Create curriculum-ordered dataset using Phi-4 for difficulty assessment
This uses the recommended strategy: Use Phi-4 to score difficulty for curriculum learning
Target: ~28GB VRAM, 1-2 hours for 50K questions, progressive easy→hard ordering
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Phi4DifficultyAssessor:
    """Assess question difficulty using Phi-4"""

    def __init__(self, model_name: str = "microsoft/phi-4"):
        """
        Initialize Phi-4 for difficulty scoring

        Args:
            model_name: HuggingFace model ID
        """
        logging.info(f"🚀 Loading {model_name} for difficulty assessment...")
        logging.info(f"   Expected VRAM: ~28GB")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            logging.info(f"✅ Phi-4 loaded successfully")
            logging.info(f"   VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        except Exception as e:
            logging.error(f"❌ Failed to load Phi-4: {e}")
            logging.info("💡 Falling back to heuristic difficulty scoring")
            self.model = None
            self.tokenizer = None

    def assess_difficulty(self, question: str, choices: Dict[str, str], correct_answer: str) -> float:
        """
        Assess difficulty of a question on scale 1-10

        Args:
            question: Question text
            choices: Answer choices
            correct_answer: Correct answer letter

        Returns:
            Difficulty score 1.0-10.0 (1=easiest, 10=hardest)
        """
        if self.model is None:
            # Fallback: Heuristic scoring
            return self._heuristic_difficulty(question, choices)

        # Format for Phi-4
        choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

        prompt = f"""Rate the difficulty of this multiple choice question on a scale from 1 (very easy) to 10 (very hard).

Question: {question}

Choices:
{choices_text}

Correct Answer: {correct_answer}

Consider:
- Complexity of concepts
- Amount of reasoning required
- Likelihood of confusion between choices
- Background knowledge needed

Difficulty rating (1-10):"""

        # Generate score
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # Deterministic for consistency
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        rating_text = full_text.split("Difficulty rating (1-10):")[-1].strip()

        # Extract number
        match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
        if match:
            score = float(match.group(1))
            # Clamp to 1-10
            return max(1.0, min(10.0, score))
        else:
            # Fallback to heuristic if parsing fails
            return self._heuristic_difficulty(question, choices)

    def _heuristic_difficulty(self, question: str, choices: Dict[str, str]) -> float:
        """
        Heuristic difficulty based on question/choice complexity
        This runs if Phi-4 isn't available
        """
        score = 5.0  # Base

        # Question length complexity
        q_words = len(question.split())
        if q_words > 50:
            score += 2.0
        elif q_words > 30:
            score += 1.0
        elif q_words < 10:
            score -= 1.0

        # Choice length (longer = more nuanced)
        avg_choice_len = sum(len(c.split()) for c in choices.values()) / len(choices)
        if avg_choice_len > 15:
            score += 1.5
        elif avg_choice_len < 3:
            score -= 1.0

        # Keywords indicating difficulty
        hard_keywords = ['analyze', 'synthesize', 'evaluate', 'compare', 'contrast', 'infer', 'deduce']
        easy_keywords = ['what', 'define', 'list', 'name', 'identify']

        q_lower = question.lower()
        if any(kw in q_lower for kw in hard_keywords):
            score += 1.5
        if any(kw in q_lower for kw in easy_keywords):
            score -= 1.0

        # Clamp
        return max(1.0, min(10.0, score))

    def assess_batch(self, questions: List[Dict], batch_size: int = 8) -> List[Tuple[Dict, float]]:
        """
        Assess difficulty for batch of questions

        Args:
            questions: List of question dicts
            batch_size: Questions per batch

        Returns:
            List of (question, difficulty_score) tuples
        """
        results = []

        for i in tqdm(range(0, len(questions), batch_size), desc="Assessing difficulty"):
            batch = questions[i:i+batch_size]

            for q in batch:
                try:
                    choices = q.get('choices', q.get('options', {}))
                    correct = q.get('correct_answer', 'A')

                    if not choices:
                        logging.warning(f"Skipping question with no choices")
                        continue

                    # Assess
                    difficulty = self.assess_difficulty(
                        q['question'],
                        choices,
                        correct
                    )

                    results.append((q, difficulty))

                except Exception as e:
                    logging.error(f"Error assessing difficulty: {e}")
                    # Default to medium difficulty
                    results.append((q, 5.0))

            # Clear cache
            if i % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results


def create_curriculum(questions_with_scores: List[Tuple[Dict, float]]) -> List[Dict]:
    """
    Sort questions by difficulty to create curriculum

    Args:
        questions_with_scores: List of (question, difficulty) tuples

    Returns:
        Sorted list of questions (easy → hard) with difficulty metadata
    """
    logging.info("📚 Creating curriculum ordering (easy → hard)...")

    # Sort by difficulty
    sorted_qs = sorted(questions_with_scores, key=lambda x: x[1])

    # Add difficulty metadata
    curriculum = []
    for i, (q, diff) in enumerate(sorted_qs):
        enhanced_q = q.copy()
        enhanced_q['difficulty_score'] = diff
        enhanced_q['curriculum_position'] = i
        enhanced_q['curriculum_percentile'] = (i / len(sorted_qs)) * 100
        curriculum.append(enhanced_q)

    # Stats
    difficulties = [score for _, score in sorted_qs]
    logging.info(f"📊 Curriculum Statistics:")
    logging.info(f"   Total questions: {len(curriculum)}")
    logging.info(f"   Difficulty range: {min(difficulties):.1f} - {max(difficulties):.1f}")
    logging.info(f"   Mean difficulty: {sum(difficulties)/len(difficulties):.1f}")
    logging.info(f"   Easy (<4): {sum(1 for d in difficulties if d < 4)}")
    logging.info(f"   Medium (4-7): {sum(1 for d in difficulties if 4 <= d < 7)}")
    logging.info(f"   Hard (7+): {sum(1 for d in difficulties if d >= 7)}")

    return curriculum


def main():
    """Main pipeline"""
    logging.info("=" * 80)
    logging.info("📚 PHI-4 CURRICULUM LEARNING - DIFFICULTY ASSESSMENT")
    logging.info("=" * 80)

    # Paths
    input_path = Path('data/enhanced/cot_enhanced_50k.json')
    output_path = Path('data/enhanced/curriculum_ordered_50k.json')

    # Load CoT-enhanced data
    logging.info(f"📂 Loading CoT-enhanced dataset from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    logging.info(f"✅ Loaded {len(questions)} questions")

    # Initialize assessor
    assessor = Phi4DifficultyAssessor()

    # Assess difficulty
    logging.info(f"🔄 Assessing difficulty for {len(questions)} questions...")
    logging.info(f"   Estimated time: 1-2 hours")
    questions_with_scores = assessor.assess_batch(questions, batch_size=8)

    # Create curriculum
    curriculum = create_curriculum(questions_with_scores)

    # Save
    logging.info(f"💾 Saving curriculum-ordered dataset to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(curriculum, f, indent=2, ensure_ascii=False)

    logging.info("=" * 80)
    logging.info("✅ CURRICULUM CREATION COMPLETE")
    logging.info(f"   Output: {output_path}")
    logging.info(f"   Ready for Qwen3-235B fine-tuning!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
