"""
Phase 1: Ensemble Data Generation
Generates synthetic Q&A pairs from 3 teacher models (Phi-4, Qwen3, Mistral NeMo)
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import argparse
from pathlib import Path

# Topics for question generation
TOPICS = {
    "seating": "seating arrangements",
    "blood": "blood relations"
}

# Prompt template for question generation
QUESTION_GENERATION_PROMPT = """You are an expert at creating challenging logic puzzles. Generate a difficult multiple-choice puzzle about {topic}.

IMPORTANT: Respond ONLY with valid JSON in exactly this format, with no additional text:

{{
  "question": "The puzzle description...",
  "choices": {{
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct_answer": "B",
  "explanation": "Why B is correct..."
}}

Make the puzzle challenging but solvable. Ensure all 4 choices are plausible.
Output ONLY the JSON, nothing else."""

# Prompt template for answering
ANSWER_PROMPT = """Answer this multiple-choice question. Respond with ONLY the letter (A, B, C, or D) of the correct answer.

Question: {question}
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Your answer (letter only):"""


class TeacherModel:
    """Wrapper for teacher model"""

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
        """Generate text from prompt using chat template"""
        # Format as chat message
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to plain prompt
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

    def generate_question(self, topic: str) -> Dict:
        """Generate a logic puzzle question"""
        prompt = QUESTION_GENERATION_PROMPT.format(topic=topic)
        response = self.generate(prompt, max_tokens=512, temperature=0.8)

        # Try to extract JSON
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data
            else:
                print(f"⚠️ No JSON found in response from {self.model_name}")
                return None
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error from {self.model_name}: {e}")
            return None

    def answer_question(self, question: str, choices: Dict[str, str]) -> str:
        """Answer a multiple-choice question"""
        prompt = ANSWER_PROMPT.format(
            question=question,
            choice_a=choices['A'],
            choice_b=choices['B'],
            choice_c=choices['C'],
            choice_d=choices['D']
        )
        response = self.generate(prompt, max_tokens=10, temperature=0.1)

        # Extract just the letter
        response = response.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response:
                return letter

        print(f"⚠️ Could not extract answer letter from {self.model_name}: {response}")
        return None


def generate_from_teachers(
    models: List[TeacherModel],
    n_questions_per_model: int,
    output_file: str
):
    """Generate questions from all teacher models"""

    all_questions = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Generating {n_questions_per_model} questions from {model.model_name}")
        print(f"{'='*60}\n")

        model_questions = []

        for topic_name, topic_full in TOPICS.items():
            n_per_topic = n_questions_per_model // 2
            print(f"Topic: {topic_full} ({n_per_topic} questions)")

            for i in range(n_per_topic):
                print(f"  Generating question {i+1}/{n_per_topic}...", end=" ")

                question_data = model.generate_question(topic_full)

                if question_data:
                    # Add metadata
                    question_data['teacher_model'] = model.model_name
                    question_data['topic_category'] = topic_name
                    model_questions.append(question_data)
                    print("✅")
                else:
                    print("❌ Failed")

        print(f"\nGenerated {len(model_questions)} questions from {model.model_name}")
        all_questions.extend(model_questions)

    # Save raw questions
    with open(output_file, 'w') as f:
        json.dump(all_questions, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total questions generated: {len(all_questions)}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")

    return all_questions


def cross_validate_questions(
    questions: List[Dict],
    models: List[TeacherModel],
    output_file: str
):
    """Cross-validate questions using generator-weighted voting"""

    print(f"\n{'='*60}")
    print(f"Cross-Validating {len(questions)} Questions")
    print(f"Using Generator-Weighted Validation Logic")
    print(f"{'='*60}\n")

    validated_questions = []
    rejected_questions = []

    for i, q in enumerate(questions):
        print(f"Question {i+1}/{len(questions)}: ", end="")

        question_text = q['question']
        choices = q['choices']
        generator_answer = q['correct_answer']
        generator_model = q['teacher_model']

        # Get answers from OTHER models (solvers, not generator)
        solver_answers = []
        solver_models = []
        for model in models:
            if model.model_name != generator_model:
                answer = model.answer_question(question_text, choices)
                if answer:
                    solver_answers.append(answer)
                    solver_models.append(model.model_name)

        if len(solver_answers) < len(models) - 1:
            print(f"⚠️ Not all solvers answered (got {len(solver_answers)}/{len(models)-1})")
            q['validation_status'] = 'incomplete'
            q['solver_answers'] = solver_answers
            rejected_questions.append(q)
            continue

        # Generator-weighted validation logic
        all_answers = [generator_answer] + solver_answers
        from collections import Counter
        vote_counts = Counter(all_answers)

        # Decision tree
        if len(set(all_answers)) == 1:
            # All 3 agree (perfect consensus)
            final_answer = generator_answer
            status = "unanimous"
            confidence = "high"
            reason = f"all agree on {final_answer}"
            accept = True
        elif generator_answer in solver_answers:
            # Generator + at least 1 solver agree (2-1 or better)
            final_answer = generator_answer
            status = "generator_supported"
            confidence = "high"
            reason = f"generator + {solver_answers.count(generator_answer)} solver(s) agree on {final_answer}"
            accept = True
        elif solver_answers[0] == solver_answers[1]:
            # Both solvers agree, differ from generator (2-1 against generator)
            final_answer = solver_answers[0]
            status = "solvers_override"
            confidence = "medium"
            reason = f"both solvers agree on {final_answer}, overriding generator's {generator_answer}"
            accept = True
        else:
            # All 3 disagree (1-1-1 split)
            final_answer = None
            status = "no_consensus"
            confidence = "none"
            reason = f"3-way split: gen={generator_answer}, solvers={solver_answers}"
            accept = False

        # Store metadata
        q['validation_status'] = status
        q['confidence'] = confidence
        q['generator_answer'] = generator_answer
        q['solver_answers'] = solver_answers
        q['solver_models'] = solver_models
        q['final_answer'] = final_answer
        q['vote_distribution'] = dict(vote_counts)

        # Accept or reject
        if accept:
            # Update the correct answer if solvers overrode
            if status == "solvers_override":
                q['correct_answer'] = final_answer
                q['answer_updated'] = True
            print(f"✅ {status.upper()}: {reason}")
            validated_questions.append(q)
        else:
            print(f"❌ REJECTED: {reason}")
            rejected_questions.append(q)

    # Save validated questions
    with open(output_file, 'w') as f:
        json.dump(validated_questions, f, indent=2)

    # Also save rejected questions for analysis
    rejected_file = output_file.replace('validated_', 'rejected_')
    with open(rejected_file, 'w') as f:
        json.dump(rejected_questions, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"  Total questions: {len(questions)}")
    print(f"  Validated (passed): {len(validated_questions)}")
    print(f"  Rejected: {len(rejected_questions)}")
    print(f"  Pass rate: {len(validated_questions)/len(questions)*100:.1f}%")
    print(f"Saved to: {output_file}")
    print(f"Rejected saved to: {rejected_file}")
    print(f"{'='*60}")

    return validated_questions


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A data from teacher models")
    parser.add_argument("--models", nargs="+", default=[
        "microsoft/phi-4",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-Nemo-Instruct-2407"
    ], help="Teacher model names")
    parser.add_argument("--n-per-model", type=int, default=100, help="Questions per model")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, only validate")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_file = output_dir / "raw_questions.json"
    validated_file = output_dir / "validated_questions.json"

    # Load teacher models
    print("Loading teacher models...")
    models = [TeacherModel(model_name) for model_name in args.models]

    # Generate questions
    if not args.skip_generation:
        questions = generate_from_teachers(models, args.n_per_model, str(raw_file))
    else:
        print(f"Loading existing questions from {raw_file}")
        with open(raw_file) as f:
            questions = json.load(f)

    # Cross-validate
    if not args.skip_validation:
        validated = cross_validate_questions(questions, models, str(validated_file))
    else:
        validated = questions

    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    main()
