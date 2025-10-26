"""
Download MASSIVE amounts of MCQ data from multiple sources
Goal: 10,000+ high-quality MCQs for comprehensive training
"""

from datasets import load_dataset
import json
import random
from pathlib import Path
from tqdm import tqdm

def download_massive_mcq_datasets(output_dir="data/massive_datasets"):
    """Download as many MCQ datasets as possible"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_questions = []

    # 1. MMLU - ALL topics (57 subjects)
    print("Downloading FULL MMLU dataset (57 subjects)...")
    mmlu_subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics',
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
        'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
        'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
        'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
        'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
        'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]

    for subject in tqdm(mmlu_subjects, desc="MMLU subjects"):
        try:
            dataset = load_dataset('cais/mmlu', subject, split='test')
            for item in dataset:
                formatted = {
                    "question": item['question'],
                    "choices": {
                        "A": item['choices'][0],
                        "B": item['choices'][1],
                        "C": item['choices'][2],
                        "D": item['choices'][3] if len(item['choices']) > 3 else "None"
                    },
                    "correct_answer": chr(65 + item['answer']),
                    "source": "MMLU",
                    "subject": subject
                }
                all_questions.append(formatted)
        except Exception as e:
            print(f"  Failed to load {subject}: {e}")

    print(f"✅ MMLU: {len([q for q in all_questions if q['source'] == 'MMLU'])} questions")

    # 2. ARC - FULL dataset
    print("Downloading FULL ARC dataset...")
    try:
        for split in ['train', 'test', 'validation']:
            for difficulty in ['ARC-Challenge', 'ARC-Easy']:
                try:
                    dataset = load_dataset('allenai/ai2_arc', difficulty, split=split)
                    for item in dataset:
                        formatted = {
                            "question": item['question'],
                            "choices": {
                                chr(65+i): choice for i, choice in enumerate(item['choices']['text'][:4])
                            },
                            "correct_answer": item['answerKey'],
                            "source": "ARC",
                            "difficulty": difficulty
                        }
                        all_questions.append(formatted)
                except:
                    pass
    except Exception as e:
        print(f"  ARC error: {e}")

    print(f"✅ ARC: {len([q for q in all_questions if q['source'] == 'ARC'])} questions")

    # 3. CommonsenseQA
    print("Downloading CommonsenseQA...")
    try:
        csqa = load_dataset('commonsense_qa', split='train')
        for item in list(csqa)[:2000]:
            choices_dict = {}
            correct = None
            for i, choice in enumerate(item['choices']['text'][:4]):
                letter = chr(65+i)
                choices_dict[letter] = choice
                if item['choices']['label'][i] == item['answerKey']:
                    correct = letter

            formatted = {
                "question": item['question'],
                "choices": choices_dict,
                "correct_answer": correct or "A",
                "source": "CommonsenseQA"
            }
            all_questions.append(formatted)
    except Exception as e:
        print(f"  CommonsenseQA error: {e}")

    print(f"✅ CommonsenseQA: {len([q for q in all_questions if q['source'] == 'CommonsenseQA'])} questions")

    # 4. TruthfulQA - FULL
    print("Downloading FULL TruthfulQA...")
    try:
        tqa = load_dataset('truthful_qa', 'multiple_choice', split='validation')
        for item in tqa:
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
    except Exception as e:
        print(f"  TruthfulQA error: {e}")

    print(f"✅ TruthfulQA: {len([q for q in all_questions if q['source'] == 'TruthfulQA'])} questions")

    # 5. RACE (Reading comprehension)
    print("Downloading RACE dataset...")
    try:
        for level in ['middle', 'high']:
            race = load_dataset('race', level, split='train')
            for item in list(race)[:1000]:
                formatted = {
                    "question": f"Context: {item['article'][:500]}... Question: {item['question']}",
                    "choices": {
                        chr(65+i): option for i, option in enumerate(item['options'][:4])
                    },
                    "correct_answer": item['answer'],
                    "source": "RACE",
                    "level": level
                }
                all_questions.append(formatted)
    except Exception as e:
        print(f"  RACE error: {e}")

    print(f"✅ RACE: {len([q for q in all_questions if q['source'] == 'RACE'])} questions")

    # 6. SciQ (Science questions)
    print("Downloading SciQ dataset...")
    try:
        sciq = load_dataset('sciq', split='train')
        for item in list(sciq)[:2000]:
            # SciQ has 3 distractors + 1 correct
            choices = [item['correct_answer'], item['distractor1'],
                      item['distractor2'], item['distractor3']]
            random.shuffle(choices)
            correct_idx = choices.index(item['correct_answer'])

            formatted = {
                "question": item['question'],
                "choices": {chr(65+i): choice for i, choice in enumerate(choices)},
                "correct_answer": chr(65 + correct_idx),
                "source": "SciQ",
                "support": item.get('support', '')[:200]
            }
            all_questions.append(formatted)
    except Exception as e:
        print(f"  SciQ error: {e}")

    print(f"✅ SciQ: {len([q for q in all_questions if q['source'] == 'SciQ'])} questions")

    # Print final statistics
    print(f"\n{'='*60}")
    print(f"TOTAL QUESTIONS DOWNLOADED: {len(all_questions)}")
    print(f"{'='*60}")

    # Save raw dataset
    raw_file = output_path / "massive_mcq_raw.json"
    with open(raw_file, 'w') as f:
        json.dump(all_questions, f, indent=2)
    print(f"Saved raw to: {raw_file}")

    # Format for training
    training_ready = []
    for q in all_questions:
        training_ready.append({
            "question": q["question"],
            "choices": q["choices"],
            "correct_answer": q["correct_answer"],
            "explanation": f"Source: {q['source']}",
            "teacher_model": "external_dataset",
            "validation_status": "external_high_quality",
            "confidence": "high"
        })

    # Shuffle
    random.shuffle(training_ready)

    # Save training-ready
    training_file = output_path / "massive_training_ready.json"
    with open(training_file, 'w') as f:
        json.dump(training_ready, f, indent=2)
    print(f"Training format saved to: {training_file}")

    # Statistics
    from collections import Counter
    sources = Counter([q['source'] for q in all_questions])
    print("\n=== Breakdown by Source ===")
    for source, count in sources.most_common():
        print(f"{source}: {count:,} questions")

    return all_questions

if __name__ == "__main__":
    print("Starting massive dataset download...")
    print("This will download 10,000+ MCQ questions from:")
    print("- MMLU (all 57 subjects)")
    print("- ARC (full dataset)")
    print("- CommonsenseQA")
    print("- TruthfulQA")
    print("- RACE (reading comprehension)")
    print("- SciQ (science)")
    print()

    questions = download_massive_mcq_datasets()