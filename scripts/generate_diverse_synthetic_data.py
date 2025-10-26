"""
Phase 1B: Diverse Synthetic Data Generation
Generates Q&A pairs across 9 pattern-based categories
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
import random

# Pattern-based categories with specific subtopics
CATEGORIES = {
    "constraint_satisfaction": {
        "name": "Constraint Satisfaction",
        "subtopics": [
            "seating arrangements with constraints",
            "task scheduling and time allocation",
            "resource distribution problems",
            "assignment and matching puzzles"
        ],
        "target": 126  # Adjusted for existing 99
    },
    "relationship_mapping": {
        "name": "Relationship Mapping",
        "subtopics": [
            "family trees and blood relations",
            "organizational hierarchies",
            "network connections and paths",
            "comparative relationships"
        ],
        "target": 93  # Adjusted for existing 126
    },
    "sequential_reasoning": {
        "name": "Sequential Reasoning",
        "subtopics": [
            "number and letter series patterns",
            "pattern completion puzzles",
            "coding-decoding sequences",
            "progressive logic puzzles"
        ],
        "target": 250
    },
    "quantitative_reasoning": {
        "name": "Quantitative Reasoning",
        "subtopics": [
            "speed distance and time problems",
            "work and rate problems",
            "probability and combinations",
            "percentage and ratio calculations"
        ],
        "target": 250
    },
    "logical_deduction": {
        "name": "Logical Deduction",
        "subtopics": [
            "if-then conditional statements",
            "truth-teller and liar puzzles",
            "syllogisms and logical arguments",
            "cause and effect chains"
        ],
        "target": 250
    },
    "spatial_reasoning": {
        "name": "Spatial Reasoning",
        "subtopics": [
            "direction and navigation puzzles",
            "shape rotation and transformation",
            "paper folding and cutting",
            "visual pattern recognition"
        ],
        "target": 250
    },
    "analytical_puzzles": {
        "name": "Analytical Puzzles",
        "subtopics": [
            "data sufficiency problems",
            "statement analysis puzzles",
            "assumption identification",
            "logical consistency checks"
        ],
        "target": 188
    },
    "trick_questions": {
        "name": "Trick Questions",
        "subtopics": [
            "misdirection and red herrings",
            "double meanings and wordplay",
            "common misconception traps",
            "lateral thinking puzzles"
        ],
        "target": 125
    },
    "general_knowledge": {
        "name": "General Knowledge MCQs",
        "subtopics": [
            "basic facts and trivia",
            "common sense reasoning",
            "real-world applications",
            "current affairs and history"
        ],
        "target": 188
    }
}

# Enhanced prompt for diverse question generation
QUESTION_GENERATION_PROMPT = """You are an expert at creating challenging multiple-choice questions for competitive exams.

Generate a difficult {category} question about: {subtopic}

The question should test {reasoning_type} and be challenging but fair.

IMPORTANT: Respond ONLY with valid JSON in exactly this format:

{{
  "question": "The complete question text...",
  "choices": {{
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct_answer": "B",
  "explanation": "Clear explanation of why B is correct and others are wrong...",
  "difficulty": "medium/hard/expert",
  "primary_category": "{category_key}",
  "secondary_categories": ["optional_cross_category1", "optional_cross_category2"]
}}

Requirements:
- Make all 4 choices plausible
- Ensure exactly one correct answer
- Test deep understanding, not memorization
- Output ONLY the JSON, nothing else."""

# Answer validation prompt
ANSWER_PROMPT = """Solve this {category} problem step by step.

Question: {question}
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Think through the problem logically, then provide ONLY the letter (A, B, C, or D) of the correct answer."""


class DiverseTeacherModel:
    """Enhanced teacher model for diverse category generation"""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"✅ {model_name} loaded successfully")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.8) -> str:
        """Generate text with chat template support"""
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def generate_question(self, category_key: str, category_info: Dict) -> Dict:
        """Generate a question for a specific category"""
        subtopic = random.choice(category_info["subtopics"])

        # Determine reasoning type based on category
        reasoning_types = {
            "constraint_satisfaction": "logical constraint solving",
            "relationship_mapping": "relationship analysis",
            "sequential_reasoning": "pattern recognition",
            "quantitative_reasoning": "mathematical reasoning",
            "logical_deduction": "deductive reasoning",
            "spatial_reasoning": "spatial visualization",
            "analytical_puzzles": "analytical thinking",
            "trick_questions": "careful attention to details",
            "general_knowledge": "factual knowledge and reasoning"
        }

        prompt = QUESTION_GENERATION_PROMPT.format(
            category=category_info["name"],
            subtopic=subtopic,
            reasoning_type=reasoning_types.get(category_key, "logical thinking"),
            category_key=category_key
        )

        response = self.generate(prompt, max_tokens=600, temperature=0.85)

        # Extract and parse JSON
        try:
            import re

            # Check for ```json code blocks first (models often wrap JSON in markdown)
            if '```json' in response:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Fallback to regular JSON extraction
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    if start != -1 and end != 0:
                        json_str = response[start:end]
                    else:
                        print(f"⚠️ No JSON found in response")
                        return None
            else:
                # Regular JSON extraction
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = response[start:end]
                else:
                    print(f"⚠️ No JSON found in response")
                    return None

            # Clean up JSON string
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas

            data = json.loads(json_str)

            # Add metadata
            data['subtopic'] = subtopic
            data['teacher_model'] = self.model_name

            return data
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            return None

    def answer_question(self, question: str, choices: Dict[str, str], category: str) -> str:
        """Answer a question with category context"""
        prompt = ANSWER_PROMPT.format(
            category=CATEGORIES.get(category, {"name": "general"})["name"],
            question=question,
            choice_a=choices['A'],
            choice_b=choices['B'],
            choice_c=choices['C'],
            choice_d=choices['D']
        )
        response = self.generate(prompt, max_tokens=20, temperature=0.1)

        # Extract letter
        response = response.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response:
                return letter
        return None


def generate_diverse_dataset(
    models: List[DiverseTeacherModel],
    output_dir: str
):
    """Generate questions across all categories"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_questions = []

    # Calculate questions per model per category
    for category_key, category_info in CATEGORIES.items():
        target = category_info["target"]
        per_model = target // len(models)
        remainder = target % len(models)

        print(f"\n{'='*60}")
        print(f"Generating {target} questions for {category_info['name']}")
        print(f"({per_model} per model, +{remainder} for first model)")
        print(f"{'='*60}\n")

        category_questions = []

        for i, model in enumerate(models):
            # First model gets remainder
            model_target = per_model + (remainder if i == 0 else 0)

            print(f"\n{model.model_name}: Generating {model_target} questions...")

            for j in range(model_target):
                print(f"  Question {j+1}/{model_target}...", end=" ")

                question = model.generate_question(category_key, category_info)

                if question:
                    question['category'] = category_key
                    category_questions.append(question)
                    print("✅")
                else:
                    print("❌ Failed")

        print(f"\nGenerated {len(category_questions)} {category_info['name']} questions")
        all_questions.extend(category_questions)

    # Save raw questions
    raw_file = output_path / "diverse_raw_questions.json"
    with open(raw_file, 'w') as f:
        json.dump(all_questions, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total questions generated: {len(all_questions)}")
    print(f"Saved to: {raw_file}")
    print(f"{'='*60}")

    return all_questions


def validate_diverse_questions(
    questions: List[Dict],
    models: List[DiverseTeacherModel],
    output_dir: str
):
    """Validate using generator-weighted approach"""

    print(f"\n{'='*60}")
    print(f"Validating {len(questions)} Questions")
    print(f"Using Generator-Weighted Logic")
    print(f"{'='*60}\n")

    validated = []
    rejected = []

    for i, q in enumerate(questions):
        print(f"Question {i+1}/{len(questions)}: ", end="")

        generator_answer = q['correct_answer']
        generator_model = q['teacher_model']
        category = q.get('category', 'general')

        # Get answers from OTHER models
        solver_answers = []
        for model in models:
            if model.model_name != generator_model:
                answer = model.answer_question(
                    q['question'],
                    q['choices'],
                    category
                )
                if answer:
                    solver_answers.append(answer)

        if len(solver_answers) < len(models) - 1:
            print(f"⚠️ Incomplete answers")
            rejected.append(q)
            continue

        # Apply validation logic
        all_answers = [generator_answer] + solver_answers

        if len(set(all_answers)) == 1:
            # Unanimous
            q['validation_status'] = "unanimous"
            q['confidence'] = "high"
            print(f"✅ UNANIMOUS: all agree on {generator_answer}")
            validated.append(q)
        elif generator_answer in solver_answers:
            # Generator supported
            q['validation_status'] = "generator_supported"
            q['confidence'] = "high"
            print(f"✅ SUPPORTED: generator + solver(s) agree on {generator_answer}")
            validated.append(q)
        elif len(solver_answers) >= 2 and solver_answers[0] == solver_answers[1]:
            # Solvers override
            q['correct_answer'] = solver_answers[0]
            q['validation_status'] = "solvers_override"
            q['confidence'] = "medium"
            print(f"✅ OVERRIDE: solvers agree on {solver_answers[0]}")
            validated.append(q)
        else:
            # No consensus
            print(f"❌ NO CONSENSUS")
            rejected.append(q)

    # Save results
    output_path = Path(output_dir)

    validated_file = output_path / "diverse_validated_questions.json"
    with open(validated_file, 'w') as f:
        json.dump(validated, f, indent=2)

    rejected_file = output_path / "diverse_rejected_questions.json"
    with open(rejected_file, 'w') as f:
        json.dump(rejected, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"  Total: {len(questions)}")
    print(f"  Validated: {len(validated)} ({len(validated)/len(questions)*100:.1f}%)")
    print(f"  Rejected: {len(rejected)}")

    # Category breakdown
    print(f"\nValidated by Category:")
    from collections import Counter
    cat_counts = Counter([q.get('category', 'unknown') for q in validated])
    for cat, count in sorted(cat_counts.items()):
        cat_name = CATEGORIES.get(cat, {}).get("name", cat)
        target = CATEGORIES.get(cat, {}).get("target", 0)
        # Adjust target for existing questions
        if cat == "constraint_satisfaction":
            target = 200 - 99  # Account for existing 99
        elif cat == "relationship_mapping":
            target = 200 - 126  # Account for existing 126
        print(f"  {cat_name}: {count}/{target} ({count/target*100:.1f}% of target)")

    print(f"{'='*60}")

    return validated


def main():
    parser = argparse.ArgumentParser(description="Generate diverse synthetic Q&A data")
    parser.add_argument("--models", nargs="+", default=[
        "microsoft/phi-4",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-Nemo-Instruct-2407"
    ])
    parser.add_argument("--output-dir", default="data/synthetic_diverse")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading teacher models...")
    models = [DiverseTeacherModel(name) for name in args.models]

    # Generate
    if not args.skip_generation:
        questions = generate_diverse_dataset(models, args.output_dir)
    else:
        raw_file = output_dir / "diverse_raw_questions.json"
        print(f"Loading from {raw_file}")
        with open(raw_file) as f:
            questions = json.load(f)

    # Validate
    if not args.skip_validation:
        validated = validate_diverse_questions(questions, models, args.output_dir)

    print("\n✅ Diverse data generation complete!")


if __name__ == "__main__":
    main()