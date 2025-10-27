#!/usr/bin/env python3
"""
Generate Chain-of-Thought reasoning for questions
Uses Qwen3-30B (fast) to generate reasoning chains
This is the KEY competitive advantage!
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

class CoTGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """Initialize model for CoT generation"""
        print(f"Loading {model_name} for CoT generation...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def generate_reasoning(self, question: str, choices: dict, correct_answer: str) -> str:
        """Generate step-by-step reasoning for a question"""

        # Format choices
        choices_text = "\\n".join([f"{k}. {v}" for k, v in choices.items()])

        prompt = f"""Analyze this multiple choice question and provide step-by-step reasoning to reach the correct answer.

Question: {question}

Choices:
{choices_text}

Provide your reasoning in this format:
1. First, identify what the question is asking
2. Analyze each option systematically
3. Eliminate wrong answers and explain why
4. Conclude with the correct answer and why it's right

Correct Answer: {correct_answer}

Your step-by-step reasoning:"""

        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        reasoning = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return reasoning.strip()

def add_chain_of_thought(questions: list, batch_size: int = 8) -> list:
    """Add Chain-of-Thought reasoning to all questions"""

    print("="*70)
    print("GENERATING CHAIN-OF-THOUGHT REASONING")
    print("="*70)

    generator = CoTGenerator()

    enhanced_questions = []

    print(f"\\nProcessing {len(questions)} questions in batches of {batch_size}...")

    for i in tqdm(range(0, len(questions), batch_size), desc="Generating CoT"):
        batch = questions[i:i+batch_size]

        for q in batch:
            try:
                # Skip if already has reasoning
                if 'reasoning' in q:
                    enhanced_questions.append(q)
                    continue

                # Get choices (handle different formats)
                choices = q.get('choices', q.get('options', {}))

                if not choices or not isinstance(choices, dict):
                    # Can't generate CoT without multiple choices
                    enhanced_questions.append(q)
                    continue

                correct_answer = q.get('correct_answer', '')

                # Generate reasoning
                reasoning = generator.generate_reasoning(
                    q['question'],
                    choices,
                    correct_answer
                )

                # Add reasoning to question
                enhanced_q = q.copy()
                enhanced_q['reasoning'] = reasoning
                enhanced_q['has_cot'] = True

                enhanced_questions.append(enhanced_q)

            except Exception as e:
                print(f"\\nError on question: {str(e)[:100]}")
                # Keep original question if generation fails
                enhanced_questions.append(q)

    # Statistics
    with_cot = sum(1 for q in enhanced_questions if q.get('has_cot', False))

    print(f"\\n✅ Generated CoT reasoning for {with_cot}/{len(questions)} questions")
    print(f"   Success rate: {with_cot/len(questions)*100:.1f}%")

    return enhanced_questions

def create_cot_training_format(questions: list) -> list:
    """Format questions for CoT training"""

    formatted = []

    for q in questions:
        if not q.get('has_cot', False):
            continue

        # Create training example with reasoning
        training_example = {
            'instruction': f"Answer this question with step-by-step reasoning:\\n\\n{q['question']}",
            'input': json.dumps(q.get('choices', {}), indent=2),
            'output': f"{q.get('reasoning', '')}\\n\\nTherefore, the answer is: {q['correct_answer']}",
            'question': q['question'],
            'correct_answer': q['correct_answer']
        }

        formatted.append(training_example)

    return formatted

def main():
    """Main CoT generation pipeline"""

    # Load comprehensive dataset
    dataset_path = Path("data/comprehensive/full_50k_dataset.json")

    if not dataset_path.exists():
        print("❌ Dataset not found! Run download_50k_comprehensive_dataset.py first")
        return

    print("Loading dataset...")
    with open(dataset_path) as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")

    # Generate Chain-of-Thought reasoning
    enhanced_questions = add_chain_of_thought(questions, batch_size=4)

    # Save enhanced dataset
    output_path = Path("data/cot_enhanced/cot_50k_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(enhanced_questions, f, indent=2)

    print(f"\\n✅ Saved enhanced dataset to: {output_path}")

    # Create training-ready format
    print("\\nCreating training-ready format...")
    training_data = create_cot_training_format(enhanced_questions)

    training_path = Path("data/cot_enhanced/cot_training_ready.json")
    with open(training_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"✅ Saved training data to: {training_path}")
    print(f"   Training examples: {len(training_data)}")

    # Show sample
    if training_data:
        print("\\n" + "="*70)
        print("SAMPLE TRAINING EXAMPLE")
        print("="*70)
        sample = training_data[0]
        print(f"\\nInstruction: {sample['instruction'][:200]}...")
        print(f"\\nReasoning: {sample['output'][:300]}...")

    print("\\n" + "="*70)
    print("✅ CHAIN-OF-THOUGHT GENERATION COMPLETE!")
    print("="*70)
    print("\\nReady for curriculum ordering and training!")

if __name__ == "__main__":
    main()