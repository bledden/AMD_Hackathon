#!/usr/bin/env python3
"""Download verified Q&A datasets for modern domains"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random

def download_modern_domain_datasets():
    """Download verified datasets for modern domains"""

    print("="*70)
    print("DOWNLOADING VERIFIED MODERN DOMAIN DATASETS")
    print("="*70)

    all_questions = []

    # 1. AI/ML Questions - from various ML benchmarks
    print("\n📚 1. AI/ML Domain:")
    try:
        # MMLU has machine_learning subset
        ml_dataset = load_dataset('cais/mmlu', 'machine_learning', split='test')
        ml_questions = []

        for item in ml_dataset:
            question = {
                'question': item['question'],
                'choices': {
                    'A': item['choices'][0],
                    'B': item['choices'][1],
                    'C': item['choices'][2],
                    'D': item['choices'][3]
                },
                'correct_answer': ['A', 'B', 'C', 'D'][item['answer']],
                'domain': 'ai_ml',
                'source': 'mmlu_machine_learning',
                'verified': True
            }
            ml_questions.append(question)

        print(f"  ✅ Found {len(ml_questions)} ML questions from MMLU")
        all_questions.extend(ml_questions)
    except Exception as e:
        print(f"  ❌ Error loading ML dataset: {e}")

    # 2. Cryptocurrency/Blockchain - Check for fintech datasets
    print("\n📚 2. Cryptocurrency Domain:")
    try:
        # Check MMLU's econometrics and finance for crypto questions
        econ_dataset = load_dataset('cais/mmlu', 'econometrics', split='test')
        crypto_questions = []

        crypto_keywords = ['bitcoin', 'blockchain', 'cryptocurrency', 'digital currency',
                          'decentralized', 'crypto', 'ledger', 'mining']

        for item in econ_dataset:
            text = item['question'].lower()
            if any(keyword in text for keyword in crypto_keywords):
                question = {
                    'question': item['question'],
                    'choices': {
                        'A': item['choices'][0],
                        'B': item['choices'][1],
                        'C': item['choices'][2],
                        'D': item['choices'][3]
                    },
                    'correct_answer': ['A', 'B', 'C', 'D'][item['answer']],
                    'domain': 'cryptocurrency',
                    'source': 'mmlu_filtered',
                    'verified': True
                }
                crypto_questions.append(question)

        print(f"  ⚠️  Found {len(crypto_questions)} crypto questions (limited availability)")
        all_questions.extend(crypto_questions)
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # 3. Climate Change - Environmental science datasets
    print("\n📚 3. Climate Change Domain:")
    try:
        # Load environmental science questions
        env_datasets = ['college_biology', 'high_school_biology', 'high_school_geography']
        climate_questions = []

        climate_keywords = ['climate', 'global warming', 'greenhouse', 'carbon', 'emissions',
                           'renewable', 'fossil fuel', 'temperature rise', 'sea level']

        for ds_name in env_datasets:
            dataset = load_dataset('cais/mmlu', ds_name, split='test')
            for item in dataset:
                text = item['question'].lower()
                if any(keyword in text for keyword in climate_keywords):
                    question = {
                        'question': item['question'],
                        'choices': {
                            'A': item['choices'][0],
                            'B': item['choices'][1],
                            'C': item['choices'][2],
                            'D': item['choices'][3]
                        },
                        'correct_answer': ['A', 'B', 'C', 'D'][item['answer']],
                        'domain': 'climate_change',
                        'source': f'mmlu_{ds_name}',
                        'verified': True
                    }
                    climate_questions.append(question)

        print(f"  ✅ Found {len(climate_questions)} climate change questions")
        all_questions.extend(climate_questions)
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # 4. Gaming - Check computer science and entertainment
    print("\n📚 4. Gaming Domain:")
    gaming_keywords = ['video game', 'gaming', 'esports', 'game design', 'playstation',
                      'xbox', 'nintendo', 'rpg', 'fps', 'mmorpg', 'game development']
    gaming_questions = []
    # Limited verified gaming Q&A available in academic datasets
    print(f"  ⚠️  Gaming domain has limited academic coverage")

    # 5. Social Media - Sociology and communications
    print("\n📚 5. Social Media Domain:")
    try:
        sociology_dataset = load_dataset('cais/mmlu', 'sociology', split='test')
        social_questions = []

        social_keywords = ['social media', 'facebook', 'twitter', 'instagram', 'tiktok',
                          'viral', 'influencer', 'online', 'digital communication']

        for item in sociology_dataset:
            text = item['question'].lower()
            if any(keyword in text for keyword in social_keywords):
                question = {
                    'question': item['question'],
                    'choices': {
                        'A': item['choices'][0],
                        'B': item['choices'][1],
                        'C': item['choices'][2],
                        'D': item['choices'][3]
                    },
                    'correct_answer': ['A', 'B', 'C', 'D'][item['answer']],
                    'domain': 'social_media',
                    'source': 'mmlu_sociology',
                    'verified': True
                }
                social_questions.append(question)

        print(f"  ⚠️  Found {len(social_questions)} social media questions")
        all_questions.extend(social_questions)
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # 6. Pandemic/Public Health
    print("\n📚 6. Pandemic Domain:")
    try:
        health_datasets = ['virology', 'college_medicine', 'clinical_knowledge']
        pandemic_questions = []

        pandemic_keywords = ['pandemic', 'covid', 'coronavirus', 'virus', 'vaccine',
                           'epidemic', 'outbreak', 'quarantine', 'transmission']

        for ds_name in health_datasets:
            try:
                dataset = load_dataset('cais/mmlu', ds_name, split='test')
                for item in dataset:
                    text = item['question'].lower()
                    if any(keyword in text for keyword in pandemic_keywords):
                        question = {
                            'question': item['question'],
                            'choices': {
                                'A': item['choices'][0],
                                'B': item['choices'][1],
                                'C': item['choices'][2],
                                'D': item['choices'][3]
                            },
                            'correct_answer': ['A', 'B', 'C', 'D'][item['answer']],
                            'domain': 'pandemic',
                            'source': f'mmlu_{ds_name}',
                            'verified': True
                        }
                        pandemic_questions.append(question)
            except:
                continue

        print(f"  ✅ Found {len(pandemic_questions)} pandemic-related questions")
        all_questions.extend(pandemic_questions)
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # 7. Space Exploration
    print("\n📚 7. Space Exploration Domain:")
    try:
        astronomy_dataset = load_dataset('cais/mmlu', 'astronomy', split='test')
        space_questions = []

        for item in astronomy_dataset:
            # All astronomy questions relate to space
            question = {
                'question': item['question'],
                'choices': {
                    'A': item['choices'][0],
                    'B': item['choices'][1],
                    'C': item['choices'][2],
                    'D': item['choices'][3]
                },
                'correct_answer': ['A', 'B', 'C', 'D'][item['answer']],
                'domain': 'space_exploration',
                'source': 'mmlu_astronomy',
                'verified': True
            }
            space_questions.append(question)

        print(f"  ✅ Found {len(space_questions)} space exploration questions")
        all_questions.extend(space_questions)
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # 8. Mental Health
    print("\n📚 8. Mental Health Domain:")
    try:
        psych_dataset = load_dataset('cais/mmlu', 'clinical_knowledge', split='test')
        mental_questions = []

        mental_keywords = ['mental health', 'depression', 'anxiety', 'therapy', 'psychiatry',
                         'counseling', 'ptsd', 'bipolar', 'schizophrenia', 'psychological']

        for item in psych_dataset:
            text = item['question'].lower()
            if any(keyword in text for keyword in mental_keywords):
                question = {
                    'question': item['question'],
                    'choices': {
                        'A': item['choices'][0],
                        'B': item['choices'][1],
                        'C': item['choices'][2],
                        'D': item['choices'][3]
                    },
                    'correct_answer': ['A', 'B', 'C', 'D'][item['answer']],
                    'domain': 'mental_health',
                    'source': 'mmlu_clinical',
                    'verified': True
                }
                mental_questions.append(question)

        print(f"  ✅ Found {len(mental_questions)} mental health questions")
        all_questions.extend(mental_questions)
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF MODERN DOMAIN COVERAGE")
    print("="*70)

    domain_counts = {}
    for q in all_questions:
        domain = q['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    total = len(all_questions)
    print(f"\nTotal verified questions found: {total}")

    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain:20}: {count:4} questions")

    # Identify gaps
    target_domains = ['ai_ml', 'cryptocurrency', 'climate_change', 'gaming',
                     'social_media', 'pandemic', 'space_exploration', 'mental_health']

    gaps = []
    for domain in target_domains:
        if domain not in domain_counts or domain_counts[domain] < 50:
            current = domain_counts.get(domain, 0)
            needed = 50 - current
            gaps.append((domain, current, needed))

    if gaps:
        print("\n⚠️  Domains needing generation or additional sources:")
        for domain, current, needed in gaps:
            print(f"  {domain:20}: Has {current}, needs {needed} more")

    # Save what we found
    if all_questions:
        output_path = Path('data/modern_domains/verified_modern_questions.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_questions, f, indent=2)

        print(f"\n💾 Saved {len(all_questions)} verified questions to: {output_path}")

    return all_questions, gaps

if __name__ == "__main__":
    questions, gaps = download_modern_domain_datasets()

    if gaps:
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print("\nFor domains with insufficient verified data:")
        print("1. Use multi-model generation with Phi-4, Qwen2.5, and Mistral-Nemo")
        print("2. Apply ensemble validation (2/3 agreement)")
        print("3. Focus generation on specific gaps identified above")
        print("\nThis leverages the multi-model approach you mentioned!")