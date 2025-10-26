"""
Download existing MCQ datasets as alternative/supplement to generated data
"""

from datasets import load_dataset
import json
import random
from pathlib import Path

def download_and_format_datasets(output_dir="data/external_datasets"):
    """Download MMLU, ARC, and other MCQ datasets"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_questions = []

    # 1. Download ARC (AI2 Reasoning Challenge)
    print("Downloading ARC dataset...")
    try:
        arc_challenge = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train')
        arc_easy = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='train')

        # Format ARC questions
        for item in list(arc_challenge)[:500] + list(arc_easy)[:500]:
            formatted = {
                "question": item['question'],
                "choices": {
                    chr(65+i): choice for i, choice in enumerate(item['choices']['text'][:4])
                },
                "correct_answer": item['answerKey'],
                "source": "ARC",
                "difficulty": "challenge" if item in arc_challenge else "easy"
            }
            all_questions.append(formatted)
        print(f"✅ ARC: {len([q for q in all_questions if q['source'] == 'ARC'])} questions")
    except Exception as e:
        print(f"❌ Error loading ARC: {e}")

    # 2. Download MMLU subset
    print("Downloading MMLU dataset...")
    try:
        # Download a subset of MMLU topics
        topics = ['elementary_mathematics', 'high_school_physics', 'logical_fallacies',
                 'abstract_algebra', 'computer_science']

        for topic in topics:
            try:
                mmlu_topic = load_dataset('cais/mmlu', topic, split='test')

                for item in list(mmlu_topic)[:100]:
                    formatted = {
                        "question": item['question'],
                        "choices": {
                            "A": item['choices'][0],
                            "B": item['choices'][1],
                            "C": item['choices'][2],
                            "D": item['choices'][3] if len(item['choices']) > 3 else "None of the above"
                        },
                        "correct_answer": chr(65 + item['answer']),  # Convert 0-3 to A-D
                        "source": "MMLU",
                        "topic": topic
                    }
                    all_questions.append(formatted)
            except:
                pass

        print(f"✅ MMLU: {len([q for q in all_questions if q['source'] == 'MMLU'])} questions")
    except Exception as e:
        print(f"❌ Error loading MMLU: {e}")

    # 3. Download TruthfulQA (smaller dataset)
    print("Downloading TruthfulQA dataset...")
    try:
        truthful_qa = load_dataset('truthful_qa', 'multiple_choice', split='validation')

        for item in list(truthful_qa)[:200]:
            # TruthfulQA has multiple correct answers, pick the first
            formatted = {
                "question": item['question'],
                "choices": {
                    chr(65+i): choice for i, choice in enumerate(item['mc1_targets']['choices'][:4])
                },
                "correct_answer": chr(65 + item['mc1_targets']['labels'].index(1)),
                "source": "TruthfulQA",
                "category": item.get('category', 'general')
            }
            all_questions.append(formatted)
        print(f"✅ TruthfulQA: {len([q for q in all_questions if q['source'] == 'TruthfulQA'])} questions")
    except Exception as e:
        print(f"❌ Error loading TruthfulQA: {e}")

    # 4. Format for our training pipeline
    print(f"\nTotal questions downloaded: {len(all_questions)}")

    # Shuffle and save
    random.shuffle(all_questions)

    # Save in our format
    output_file = output_path / "external_mcq_questions.json"
    with open(output_file, 'w') as f:
        json.dump(all_questions, f, indent=2)

    print(f"Saved to: {output_file}")

    # Also create a version compatible with our training format
    formatted_for_training = []
    for q in all_questions:
        formatted_for_training.append({
            "question": q["question"],
            "choices": q["choices"],
            "correct_answer": q["correct_answer"],
            "explanation": f"Source: {q['source']}",
            "teacher_model": "external_dataset",
            "validation_status": "external",
            "confidence": "high"
        })

    training_file = output_path / "external_training_ready.json"
    with open(training_file, 'w') as f:
        json.dump(formatted_for_training, f, indent=2)

    print(f"Training-ready format saved to: {training_file}")

    return all_questions

if __name__ == "__main__":
    questions = download_and_format_datasets()

    # Print statistics
    from collections import Counter
    sources = Counter([q['source'] for q in questions])
    print("\n=== Dataset Statistics ===")
    for source, count in sources.items():
        print(f"{source}: {count} questions")