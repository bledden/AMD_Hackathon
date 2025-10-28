#!/usr/bin/env python3
"""
Download and merge 100-120K high-quality MCQ questions from HuggingFace datasets
Fixed version with working dataset sources
"""

import json
import logging
from datasets import load_dataset
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO)

# High-quality MCQ datasets (confirmed working)
DATASETS = [
    # Working datasets from previous run
    ("allenai/sciq", None, 11679),           # Science Q&A
    ("Rowan/hellaswag", "train", 30000),     # Commonsense reasoning (increased)

    # Additional working datasets
    ("allenai/ai2_arc", "ARC-Challenge", 15000),  # Science reasoning
    ("allenai/ai2_arc", "ARC-Easy", 15000),       # Easier science questions
    ("commonsense_qa", "train", 12000),           # Commonsense QA
    ("openbookqa", "train", 6000),                # Open book QA
    ("social_i_qa", "train", 15000),              # Social interaction QA
    ("cosmos_qa", "train", 10000),                # Reading comprehension
]

def extract_mcq_format(item, dataset_name, subset=None):
    """Convert dataset items to our MCQ format"""
    try:
        if dataset_name == "allenai/sciq":
            return {
                "question": item["question"],
                "choices": {
                    "A": item["distractor1"],
                    "B": item["distractor2"],
                    "C": item["distractor3"],
                    "D": item["correct_answer"]
                },
                "correct_answer": "D",
                "category": "science",
                "difficulty": "medium",
                "source": "sciq"
            }

        elif dataset_name == "Rowan/hellaswag":
            # HellaSwag: complete the sentence
            ctx = item.get("ctx", item.get("context", ""))
            endings = item.get("endings", [])
            if len(endings) < 4:
                return None

            return {
                "question": f"Complete the following: {ctx}",
                "choices": {
                    "A": endings[0],
                    "B": endings[1],
                    "C": endings[2],
                    "D": endings[3] if len(endings) > 3 else endings[0]
                },
                "correct_answer": chr(65 + int(item.get("label", 0))),  # 0->A, 1->B, etc
                "category": "reasoning",
                "difficulty": "hard",
                "source": "hellaswag"
            }

        elif dataset_name == "allenai/ai2_arc":
            choices_data = item.get("choices", {})
            choices_list = choices_data.get("text", [])
            labels = choices_data.get("label", [])

            if len(choices_list) < 4:
                # Pad with dummy choices if less than 4
                while len(choices_list) < 4:
                    choices_list.append("Not applicable")
                    labels.append(f"Z{len(labels)}")

            answer_key = item.get("answerKey", "A")
            # Map answer key to 0-3 index
            if answer_key in labels:
                answer_idx = labels.index(answer_key)
            else:
                answer_idx = ord(answer_key) - ord('A') if answer_key in 'ABCD' else 0

            return {
                "question": item["question"],
                "choices": {
                    "A": choices_list[0],
                    "B": choices_list[1],
                    "C": choices_list[2] if len(choices_list) > 2 else "None",
                    "D": choices_list[3] if len(choices_list) > 3 else "None"
                },
                "correct_answer": chr(65 + answer_idx),
                "category": "science",
                "difficulty": "hard" if subset == "ARC-Challenge" else "easy",
                "source": f"arc_{subset.lower()}"
            }

        elif dataset_name == "commonsense_qa":
            choices_data = item.get("choices", {})
            choices_list = choices_data.get("text", [])
            labels = choices_data.get("label", [])

            if len(choices_list) < 4:
                while len(choices_list) < 4:
                    choices_list.append("Not applicable")

            answer_key = item.get("answerKey", "A")
            if answer_key in labels:
                answer_idx = labels.index(answer_key)
            else:
                answer_idx = 0

            return {
                "question": item["question"],
                "choices": {
                    "A": choices_list[0],
                    "B": choices_list[1],
                    "C": choices_list[2] if len(choices_list) > 2 else "None",
                    "D": choices_list[3] if len(choices_list) > 3 else "None"
                },
                "correct_answer": chr(65 + answer_idx),
                "category": "commonsense",
                "difficulty": "medium",
                "source": "commonsense_qa"
            }

        elif dataset_name in ["openbookqa", "social_i_qa", "cosmos_qa"]:
            # Generic handler for similar formats
            choices_data = item.get("choices", item.get("answers", {}))
            if isinstance(choices_data, dict):
                choices_list = choices_data.get("text", choices_data.get("label", []))
            else:
                choices_list = choices_data

            if len(choices_list) < 4:
                while len(choices_list) < 4:
                    choices_list.append("Not applicable")

            answer_key = str(item.get("answerKey", item.get("label", "0")))
            if answer_key.isdigit():
                answer_idx = int(answer_key)
            elif answer_key in 'ABCD':
                answer_idx = ord(answer_key) - ord('A')
            else:
                answer_idx = 0

            return {
                "question": item.get("question", item.get("context", "")),
                "choices": {
                    "A": str(choices_list[0]),
                    "B": str(choices_list[1]),
                    "C": str(choices_list[2]) if len(choices_list) > 2 else "None",
                    "D": str(choices_list[3]) if len(choices_list) > 3 else "None"
                },
                "correct_answer": chr(65 + min(answer_idx, 3)),
                "category": "general",
                "difficulty": "medium",
                "source": dataset_name.split("/")[-1]
            }

        return None

    except Exception as e:
        logging.debug(f"Failed to convert item from {dataset_name}: {e}")
        return None

def main():
    all_questions = []

    for dataset_name, subset, max_samples in DATASETS:
        try:
            logging.info(f"Downloading {dataset_name}...")

            if subset:
                try:
                    dataset = load_dataset(dataset_name, subset, split="train", trust_remote_code=True)
                except:
                    # Try without split specification
                    dataset = load_dataset(dataset_name, subset, trust_remote_code=True)
                    # Get the largest split
                    split_name = max(dataset.keys(), key=lambda k: len(dataset[k]))
                    dataset = dataset[split_name]
            else:
                try:
                    dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
                except:
                    dataset = load_dataset(dataset_name, trust_remote_code=True)
                    split_name = max(dataset.keys(), key=lambda k: len(dataset[k]))
                    dataset = dataset[split_name]

            # Sample if dataset is larger than max_samples
            if len(dataset) > max_samples:
                indices = random.sample(range(len(dataset)), max_samples)
                dataset = dataset.select(indices)

            logging.info(f"Processing {len(dataset)} samples from {dataset_name}")

            converted = 0
            for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
                mcq = extract_mcq_format(item, dataset_name, subset)
                if mcq:
                    all_questions.append(mcq)
                    converted += 1

            logging.info(f"✅ Added {converted} questions from {dataset_name}")

        except Exception as e:
            logging.error(f"Failed to load {dataset_name}: {e}")
            continue

    # Save results
    output_path = "/workspace/data/questions_100k_raw.json"
    with open(output_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    logging.info(f"\n{'='*80}")
    logging.info(f"TOTAL QUESTIONS COLLECTED: {len(all_questions)}")
    logging.info(f"{'='*80}")
    logging.info(f"✅ Saved to {output_path}")

    # Category distribution
    categories = {}
    for q in all_questions:
        cat = q.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    logging.info("\nCategory Distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logging.info(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
