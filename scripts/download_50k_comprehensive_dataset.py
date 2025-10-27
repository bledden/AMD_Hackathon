#!/usr/bin/env python3
"""
Download comprehensive 50K Q&A dataset
Combines: Existing 10K + TriviaQA + Full MMLU + CommonsenseQA + LogiQA
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random

def download_comprehensive_datasets():
    print("="*70)
    print("DOWNLOADING COMPREHENSIVE 50K DATASET")
    print("="*70)

    all_questions = []

    # 1. Load existing 10K dataset
    print("\n1️⃣ Loading existing enhanced dataset...")
    existing_path = Path("data/final_training/enhanced_90_percent_coverage.json")
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        print(f"   ✅ Loaded {len(existing)} existing questions")
        all_questions.extend(existing)
    else:
        print("   ⚠️ Existing dataset not found, starting fresh")

    # 2. Download TriviaQA
    print("\n2️⃣ Downloading TriviaQA (20,000 questions)...")
    try:
        trivia = load_dataset('trivia_qa', 'unfiltered.nocontext', split='train[:20000]')
        trivia_questions = []

        for item in tqdm(trivia, desc="Processing TriviaQA"):
            # TriviaQA format: question + answer (need to generate choices)
            # We'll mark these for MCQ conversion later
            question = {
                'question': item['question'],
                'answer': item['answer']['value'],
                'source': 'trivia_qa',
                'needs_mcq_conversion': True,
                'domain': 'general_knowledge'
            }
            trivia_questions.append(question)

        print(f"   ✅ Downloaded {len(trivia_questions)} TriviaQA questions")
        all_questions.extend(trivia_questions)
    except Exception as e:
        print(f"   ❌ Error downloading TriviaQA: {e}")

    # 3. Download FULL MMLU (all 57 subjects)
    print("\n3️⃣ Downloading Full MMLU (15,000 questions)...")
    try:
        mmlu_subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
            'clinical_knowledge', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_medicine',
            'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics',
            'formal_logic', 'global_facts', 'high_school_biology',
            'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography',
            'high_school_government_and_politics', 'high_school_macroeconomics',
            'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology',
            'high_school_statistics', 'high_school_us_history',
            'high_school_world_history', 'human_aging', 'human_sexuality',
            'international_law', 'jurisprudence', 'logical_fallacies',
            'machine_learning', 'management', 'marketing', 'medical_genetics',
            'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
            'philosophy', 'prehistory', 'professional_accounting',
            'professional_law', 'professional_medicine', 'professional_psychology',
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
            'virology', 'world_religions'
        ]

        mmlu_questions = []
        for subject in tqdm(mmlu_subjects, desc="MMLU subjects"):
            try:
                dataset = load_dataset('cais/mmlu', subject, split='test')
                for item in dataset:
                    question = {
                        'question': item['question'],
                        'choices': {
                            'A': item['choices'][0],
                            'B': item['choices'][1],
                            'C': item['choices'][2],
                            'D': item['choices'][3]
                        },
                        'correct_answer': ['A', 'B', 'C', 'D'][item['answer']],
                        'source': f'mmlu_{subject}',
                        'domain': subject,
                        'verified': True
                    }
                    mmlu_questions.append(question)
            except:
                continue

        print(f"   ✅ Downloaded {len(mmlu_questions)} MMLU questions")
        all_questions.extend(mmlu_questions)
    except Exception as e:
        print(f"   ❌ Error downloading MMLU: {e}")

    # 4. Download CommonsenseQA
    print("\n4️⃣ Downloading CommonsenseQA (12,000 questions)...")
    try:
        csqa = load_dataset('commonsense_qa', split='train')
        csqa_questions = []

        for item in tqdm(csqa, desc="Processing CommonsenseQA"):
            # Find correct answer index
            correct_idx = ord(item['answerKey']) - ord('A')

            question = {
                'question': item['question'],
                'choices': {
                    'A': item['choices']['text'][0],
                    'B': item['choices']['text'][1],
                    'C': item['choices']['text'][2],
                    'D': item['choices']['text'][3] if len(item['choices']['text']) > 3 else 'N/A',
                    'E': item['choices']['text'][4] if len(item['choices']['text']) > 4 else 'N/A'
                },
                'correct_answer': item['answerKey'],
                'source': 'commonsense_qa',
                'domain': 'common_sense_reasoning',
                'verified': True
            }
            csqa_questions.append(question)

        print(f"   ✅ Downloaded {len(csqa_questions)} CommonsenseQA questions")
        all_questions.extend(csqa_questions)
    except Exception as e:
        print(f"   ❌ Error downloading CommonsenseQA: {e}")

    # 5. Download LogiQA (for logic reasoning)
    print("\n5️⃣ Downloading LogiQA (8,000 questions)...")
    try:
        logiqa = load_dataset('lucasmccabe/logiqa', split='train')
        logiqa_questions = []

        for item in tqdm(logiqa, desc="Processing LogiQA"):
            question = {
                'question': item['context'] + ' ' + item['query'],
                'choices': {
                    'A': item['options'][0],
                    'B': item['options'][1],
                    'C': item['options'][2],
                    'D': item['options'][3]
                },
                'correct_answer': ['A', 'B', 'C', 'D'][item['correct_option']],
                'source': 'logiqa',
                'domain': 'logic_reasoning',
                'verified': True
            }
            logiqa_questions.append(question)

        print(f"   ✅ Downloaded {len(logiqa_questions)} LogiQA questions")
        all_questions.extend(logiqa_questions)
    except Exception as e:
        print(f"   ❌ Error downloading LogiQA: {e}")

    # Summary
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"\nTotal questions collected: {len(all_questions)}")

    # Count by source
    from collections import Counter
    sources = Counter(q.get('source', 'unknown') for q in all_questions)
    print("\nBreakdown by source:")
    for source, count in sources.most_common():
        print(f"  {source:30}: {count:6,} questions")

    # Sample to 50K if we have more
    if len(all_questions) > 50000:
        print(f"\n⚙️  Sampling down to 50,000 questions...")
        random.shuffle(all_questions)
        all_questions = all_questions[:50000]

    # Save comprehensive dataset
    output_path = Path("data/comprehensive/full_50k_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_questions, f, indent=2)

    print(f"\n✅ Saved {len(all_questions)} questions to: {output_path}")

    return all_questions

if __name__ == "__main__":
    questions = download_comprehensive_datasets()

    print("\n" + "="*70)
    print("✅ DATASET DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\nReady for Chain-of-Thought generation!")
    print(f"Total questions: {len(questions)}")
    print(f"\nNext step: Generate reasoning chains for these questions")