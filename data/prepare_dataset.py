#!/usr/bin/env python3
"""
Dataset Preparation Pipeline for Q&A Agent
Supports multiple strategies: existing datasets, synthetic generation, or hybrid
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
import argparse

from dataset_config import (
    SELECTED_THEME,
    get_theme_config,
    format_question_generation_prompt,
    format_question_answering_prompt,
    create_alpaca_format_example,
    validate_example,
    TARGET_EXAMPLES,
    TRAIN_RATIO,
)


class DatasetPreparator:
    """Prepare Q&A datasets for fine-tuning"""

    def __init__(self, theme: str = SELECTED_THEME, output_dir: str = "data/processed"):
        self.theme = theme
        self.theme_config = get_theme_config(theme)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.examples = []

    def load_existing_dataset(self, dataset_name: str = "squad"):
        """Load and filter existing dataset"""
        print(f"Loading {dataset_name} dataset...")

        try:
            from datasets import load_dataset

            if dataset_name == "squad":
                dataset = load_dataset("squad_v2", split="train")
                print(f"Loaded {len(dataset)} examples from SQuAD v2")

                # Convert to Q&A format
                for item in dataset:
                    if item["answers"]["text"]:  # Has answer
                        example = {
                            "question": item["question"],
                            "answer": item["answers"]["text"][0],
                            "context": item["context"][:200],  # First 200 chars
                        }
                        if validate_example(example):
                            self.examples.append(example)

            print(f"Filtered to {len(self.examples)} valid examples")

        except ImportError:
            print("Error: 'datasets' library not installed")
            print("Install with: pip install datasets")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def generate_synthetic_examples(self, n_examples: int = 100):
        """Generate synthetic Q&A examples for the theme"""
        print(f"Generating {n_examples} synthetic examples for theme: {self.theme}")
        print("Note: This creates templates. For real synthetic data, use LLM API.")

        topics = self.theme_config["example_topics"]
        difficulties = self.theme_config["difficulty_levels"]

        for i in range(n_examples):
            topic = random.choice(topics)
            difficulty = random.choice(difficulties)

            # Template question - replace with actual LLM generation
            question = f"What is the relationship between {topic} and modern {self.theme}?"
            answer = f"The relationship between {topic} and modern {self.theme} involves..."

            example = {
                "question": question,
                "answer": answer,
                "topic": topic,
                "difficulty": difficulty,
                "type": "synthetic",
            }

            self.examples.append(example)

        print(f"Generated {len(self.examples)} template examples")
        print("TODO: Replace with actual LLM API calls for production")

    def create_manual_examples(self):
        """Create high-quality manual examples for the theme"""
        print("Creating manual examples...")

        # Science theme examples
        if self.theme == "science":
            manual_examples = [
                {
                    "question": "What is the Heisenberg Uncertainty Principle?",
                    "answer": "The Heisenberg Uncertainty Principle states that it is impossible to simultaneously know both the exact position and exact momentum of a particle. This fundamental principle of quantum mechanics reveals that the act of measurement itself affects the system being measured.",
                    "topic": "quantum mechanics",
                    "difficulty": "undergraduate",
                },
                {
                    "question": "How does CRISPR-Cas9 gene editing work?",
                    "answer": "CRISPR-Cas9 is a gene editing tool that uses a guide RNA to direct the Cas9 enzyme to a specific DNA sequence. The enzyme cuts the DNA at that location, allowing scientists to remove, add, or replace genetic material with precision.",
                    "topic": "molecular biology",
                    "difficulty": "undergraduate",
                },
                {
                    "question": "What is the difference between machine learning and deep learning?",
                    "answer": "Machine learning is a broad field where algorithms learn from data to make predictions. Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to learn hierarchical representations of data. Deep learning excels at tasks like image recognition and natural language processing.",
                    "topic": "artificial intelligence",
                    "difficulty": "undergraduate",
                },
            ]

            self.examples.extend(manual_examples)
            print(f"Added {len(manual_examples)} manual examples")

    def format_for_training(self, split_ratio: float = TRAIN_RATIO):
        """Format examples for instruction fine-tuning"""
        print("Formatting examples for training...")

        formatted_examples = []

        for example in self.examples:
            question = example["question"]
            answer = example["answer"]

            # Create question generation example
            topic = example.get("topic", self.theme)
            difficulty = example.get("difficulty", "challenging")

            q_gen_instruction = format_question_generation_prompt(topic, difficulty)
            q_gen_example = create_alpaca_format_example(
                instruction=q_gen_instruction,
                output=question,
            )
            formatted_examples.append(q_gen_example)

            # Create question answering example
            q_ans_instruction = format_question_answering_prompt(question)
            q_ans_example = create_alpaca_format_example(
                instruction=q_ans_instruction,
                output=answer,
            )
            formatted_examples.append(q_ans_example)

        # Shuffle
        random.shuffle(formatted_examples)

        # Split train/val
        split_idx = int(len(formatted_examples) * split_ratio)
        train_data = formatted_examples[:split_idx]
        val_data = formatted_examples[split_idx:]

        print(f"Created {len(formatted_examples)} formatted examples")
        print(f"  Train: {len(train_data)}")
        print(f"  Val: {len(val_data)}")

        return train_data, val_data

    def save_dataset(self, train_data: List[Dict], val_data: List[Dict]):
        """Save dataset to JSON files"""
        train_path = self.output_dir / "train.json"
        val_path = self.output_dir / "val.json"

        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=2)

        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=2)

        print(f"\n✓ Dataset saved:")
        print(f"  Train: {train_path} ({len(train_data)} examples)")
        print(f"  Val: {val_path} ({len(val_data)} examples)")

        # Save metadata
        metadata = {
            "theme": self.theme,
            "theme_description": self.theme_config["description"],
            "total_examples": len(train_data) + len(val_data),
            "train_examples": len(train_data),
            "val_examples": len(val_data),
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata: {metadata_path}")

    def prepare(
        self,
        strategy: str = "manual",
        n_synthetic: int = 100,
        use_existing: Optional[str] = None,
    ):
        """Main preparation pipeline"""
        print("=" * 60)
        print(f"Preparing Q&A Dataset for Theme: {self.theme}")
        print("=" * 60)

        # Strategy 1: Manual examples (recommended for quick start)
        if strategy == "manual" or strategy == "hybrid":
            self.create_manual_examples()

        # Strategy 2: Synthetic generation
        if strategy == "synthetic" or strategy == "hybrid":
            self.generate_synthetic_examples(n_synthetic)

        # Strategy 3: Existing datasets
        if use_existing:
            self.load_existing_dataset(use_existing)

        if not self.examples:
            print("Error: No examples created!")
            return

        # Format and split
        train_data, val_data = self.format_for_training()

        # Save
        self.save_dataset(train_data, val_data)

        print("\n" + "=" * 60)
        print("Dataset preparation complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review the data in data/processed/")
        print("2. Adjust examples if needed")
        print("3. Run training: python training/scripts/train.py")


def main():
    parser = argparse.ArgumentParser(description="Prepare Q&A dataset for fine-tuning")
    parser.add_argument(
        "--theme",
        type=str,
        default=SELECTED_THEME,
        choices=["science", "history", "space", "technology"],
        help="Theme for Q&A agent",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="manual",
        choices=["manual", "synthetic", "existing", "hybrid"],
        help="Dataset creation strategy",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=100,
        help="Number of synthetic examples to generate",
    )
    parser.add_argument(
        "--existing-dataset",
        type=str,
        choices=["squad", "natural_questions", "trivia_qa"],
        help="Use existing dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed", help="Output directory"
    )

    args = parser.parse_args()

    preparator = DatasetPreparator(theme=args.theme, output_dir=args.output_dir)
    preparator.prepare(
        strategy=args.strategy,
        n_synthetic=args.n_synthetic,
        use_existing=args.existing_dataset,
    )


if __name__ == "__main__":
    main()
