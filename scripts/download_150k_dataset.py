#!/usr/bin/env python3
"""
Download 100K additional high-quality MCQ questions to reach 150K total
Focus on: STEM, Science, Math, History, General Knowledge
"""

import json
import logging
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

# High-quality MCQ datasets on HuggingFace
DATASETS = [
    ("cais/mmlu", "all", 50000),  # Massive Multitask Language Understanding
    ("allenai/sciq", None, 12000),  # Science questions
    ("Rowan/hellaswag", None, 10000),  # Commonsense reasoning
    ("lighteval/MATH", None, 7500),  # Math problems
    ("lighteval/mmlu-pro", "all", 12000),  # Advanced MMLU
    ("truthful_qa", "multiple_choice", 800),  # Truthfulness
    ("piqa", None, 16000),  # Physical reasoning
    ("winogrande", "winogrande_xl", 9000),  # Commonsense
]

def extract_mcq_format(item, dataset_name):
    """Convert various dataset formats to our MCQ format"""

    if dataset_name == "cais/mmlu":
        return {
            "question": item["question"],
            "choices": {
                "A": item["choices"][0],
                "B": item["choices"][1],
                "C": item["choices"][2],
                "D": item["choices"][3],
            },
            "answer": chr(65 + item["answer"]),  # 0->A, 1->B, etc.
            "category": item.get("subject", "general"),
            "difficulty": "medium"
        }

    elif dataset_name == "allenai/sciq":
        choices = [
            item["correct_answer"],
            item["distractor1"],
            item["distractor2"],
            item["distractor3"]
        ]
        import random
        random.shuffle(choices)
        answer_idx = choices.index(item["correct_answer"])

        return {
            "question": item["question"],
            "choices": {
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
            },
            "answer": chr(65 + answer_idx),
            "category": "science",
            "difficulty": "medium"
        }

    elif dataset_name == "Rowan/hellaswag":
        if len(item["endings"]) < 4:
            return None
        return {
            "question": item["ctx"],
            "choices": {
                "A": item["endings"][0],
                "B": item["endings"][1],
                "C": item["endings"][2],
                "D": item["endings"][3],
            },
            "answer": chr(65 + int(item["label"])),
            "category": "reasoning",
            "difficulty": "hard"
        }

    # Add more dataset handlers as needed
    return None

def download_datasets():
    """Download and merge multiple MCQ datasets"""
    all_questions = []

    for dataset_name, config, max_samples in DATASETS:
        try:
            logging.info(f"Downloading {dataset_name}...")

            if config:
                dataset = load_dataset(dataset_name, config, split="train")
            else:
                dataset = load_dataset(dataset_name, split="train")

            # Limit samples
            dataset = dataset.select(range(min(len(dataset), max_samples)))

            logging.info(f"Processing {len(dataset)} samples from {dataset_name}")

            for item in tqdm(dataset):
                mcq = extract_mcq_format(item, dataset_name)
                if mcq:
                    all_questions.append(mcq)

            logging.info(f"✅ Added {len([q for q in all_questions if q.get('category') == mcq.get('category')])} questions from {dataset_name}")

        except Exception as e:
            logging.error(f"Failed to load {dataset_name}: {e}")
            continue

    logging.info(f"\n{'='*80}")
    logging.info(f"TOTAL QUESTIONS COLLECTED: {len(all_questions)}")
    logging.info(f"{'='*80}")

    # Save
    output_path = "/workspace/data/questions_150k_raw.json"
    with open(output_path, 'w') as f:
        json.dump(all_questions, f, indent=2)

    logging.info(f"✅ Saved to {output_path}")

    return all_questions

if __name__ == "__main__":
    questions = download_datasets()

    # Print statistics
    categories = {}
    for q in questions:
        cat = q.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1

    logging.info("\nCategory Distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {cat}: {count}")
