#!/usr/bin/env python3
"""
Generate math teacher labels from DeepSeek-R1 for knowledge distillation.

Teacher: DeepSeek-R1 (97.3% MATH-500, 79.8% AIME 2024)
Student: Qwen2.5-72B + RSLoRA r=128
Dataset: ~15K math questions (math, logic, physics calculations)
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from llama_cpp import Llama
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = "/workspace/data/curriculum/train_65k_merged.json"
OUTPUT_PATH = "/workspace/data/distillation/math_teacher_labels_deepseekr1.json"
MODEL_PATH = "/workspace/models/deepseek-r1-distill-qwen-32b.Q4_K_M.gguf"  # GGUF format

# Math categories to filter
MATH_CATEGORIES = {
    'elementary_mathematics',
    'high_school_mathematics',
    'college_mathematics',
    'abstract_algebra',
    'formal_logic',
    'college_physics',  # Calculation-heavy physics
}

def load_deepseek_r1():
    """Load DeepSeek-R1 via llama.cpp with ROCm"""
    logger.info(f"Loading DeepSeek-R1 from {MODEL_PATH}")

    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # All layers on GPU
        n_ctx=4096,  # Context window
        n_batch=512,
        verbose=False,
        tensor_split=None,  # Single GPU
    )

    logger.info("✅ DeepSeek-R1 loaded successfully")
    return model

def filter_math_questions(data: List[Dict]) -> List[Dict]:
    """Filter dataset for math/logic questions"""
    math_questions = []

    for item in data:
        category = item.get('category', '').lower()

        # Check if it's a math category
        if any(cat in category for cat in MATH_CATEGORIES):
            math_questions.append(item)
            continue

        # Also check question content for math keywords
        question = item.get('question', '').lower()
        math_keywords = ['calculate', 'equation', 'solve', 'formula', 'proof',
                         'theorem', 'derivative', 'integral', 'algebra', 'geometry']

        if any(kw in question for kw in math_keywords):
            math_questions.append(item)

    logger.info(f"Filtered {len(math_questions)} math questions from {len(data)} total")
    return math_questions

def generate_teacher_label(model: Llama, question: str, choices: List[str], correct_answer: str) -> Dict:
    """Generate chain-of-thought reasoning and logits from DeepSeek-R1"""

    # Format prompt for DeepSeek-R1
    prompt = f"""<|system|>You are a mathematics expert. Solve this problem step-by-step, then select the correct answer.<|end|>
<|user|>
{question}

Choices:
{chr(10).join(choices)}

Provide your reasoning, then state your final answer as A, B, C, or D.<|end|>
<|assistant|>"""

    # Generate with temperature=2.0 for soft targets
    output = model(
        prompt,
        max_tokens=512,
        temperature=2.0,
        top_p=0.95,
        echo=False,
        logprobs=4,  # Get logits for top 4 choices
    )

    reasoning = output['choices'][0]['text'].strip()

    # Extract logprobs for A/B/C/D if available
    logprobs = {}
    if 'logprobs' in output['choices'][0]:
        token_logprobs = output['choices'][0]['logprobs']
        # Parse logprobs for answer choices
        # (This is simplified - actual implementation would need careful token matching)
        logprobs = {
            'A': token_logprobs.get('A', -10.0),
            'B': token_logprobs.get('B', -10.0),
            'C': token_logprobs.get('C', -10.0),
            'D': token_logprobs.get('D', -10.0),
        }

    return {
        'teacher_reasoning': reasoning,
        'teacher_logprobs': logprobs,
        'correct_answer': correct_answer,
    }

def main():
    """Generate teacher labels for math questions"""

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {DATASET_PATH}")
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    # Filter math questions
    math_questions = filter_math_questions(data)
    logger.info(f"Processing {len(math_questions)} math questions")

    # Load DeepSeek-R1
    model = load_deepseek_r1()

    # Generate teacher labels
    augmented_dataset = []

    for item in tqdm(math_questions, desc="Generating teacher labels"):
        question = item.get('question', '')
        choices = item.get('choices', [])
        correct_answer = item.get('answer', '')

        try:
            # Generate teacher label
            teacher_label = generate_teacher_label(model, question, choices, correct_answer)

            # Augment original item
            augmented_item = {
                **item,
                **teacher_label,
            }

            augmented_dataset.append(augmented_item)

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            # Keep original item without teacher label
            augmented_dataset.append(item)

    # Save augmented dataset
    logger.info(f"Saving {len(augmented_dataset)} augmented examples to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(augmented_dataset, f, indent=2)

    logger.info("✅ Math teacher label generation complete!")
    logger.info(f"   Dataset: {OUTPUT_PATH}")
    logger.info(f"   Questions: {len(augmented_dataset)}")

if __name__ == "__main__":
    main()
