"""
Create a balanced dataset of 10,000 questions with equal coverage across topics
Then identify and fill gaps with targeted generation
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

def categorize_question(question: Dict) -> str:
    """Categorize a question into broad topic areas"""

    q_text = question.get('question', '').lower()
    subject = question.get('subject', '').lower()
    topic = question.get('topic', '').lower()
    source = question.get('source', '').lower()

    # Define category mappings
    if any(word in subject + topic for word in ['math', 'algebra', 'geometry', 'calculus', 'statistics']):
        return 'mathematics'
    elif any(word in subject + topic for word in ['physics', 'chemistry', 'biology', 'science']):
        return 'science'
    elif any(word in subject + topic for word in ['history', 'geography', 'social', 'political']):
        return 'social_studies'
    elif any(word in subject + topic for word in ['computer', 'programming', 'algorithm', 'code']):
        return 'computer_science'
    elif any(word in subject + topic for word in ['logic', 'reasoning', 'puzzle', 'fallac']):
        return 'logic_reasoning'
    elif any(word in subject + topic for word in ['language', 'literature', 'writing', 'grammar']):
        return 'language_arts'
    elif any(word in subject + topic for word in ['business', 'economics', 'finance', 'management']):
        return 'business_economics'
    elif any(word in subject + topic for word in ['psychology', 'sociology', 'philosophy', 'ethics']):
        return 'humanities'
    elif any(word in subject + topic for word in ['medical', 'medicine', 'health', 'anatomy']):
        return 'medical'
    elif any(word in subject + topic for word in ['law', 'legal', 'jurisprudence']):
        return 'law'
    elif 'seating' in q_text or 'arrangement' in q_text:
        return 'logic_reasoning'
    elif 'blood' in q_text and 'relation' in q_text:
        return 'logic_reasoning'
    elif source == 'commonsenseqa':
        return 'common_sense'
    elif source == 'truthfulqa':
        return 'critical_thinking'
    else:
        return 'general_knowledge'


def create_balanced_dataset(input_file="data/final_training/final_training_full.json",
                          output_dir="data/balanced_10k",
                          target_size=10000):
    """Create a balanced dataset with equal representation across categories"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all questions
    print("Loading dataset...")
    with open(input_file) as f:
        all_questions = json.load(f)

    print(f"Total questions loaded: {len(all_questions):,}")

    # Categorize all questions
    print("\nCategorizing questions...")
    categorized = defaultdict(list)

    for q in all_questions:
        category = categorize_question(q)
        q['assigned_category'] = category
        categorized[category].append(q)

    # Print category distribution
    print("\n" + "="*60)
    print("ORIGINAL DISTRIBUTION")
    print("="*60)
    for cat, questions in sorted(categorized.items()):
        print(f"{cat:20} : {len(questions):6,} questions")

    # Define target categories (12 main categories for balanced coverage)
    target_categories = [
        'mathematics',
        'science',
        'social_studies',
        'computer_science',
        'logic_reasoning',
        'language_arts',
        'business_economics',
        'humanities',
        'medical',
        'law',
        'common_sense',
        'critical_thinking',
        'general_knowledge'
    ]

    # Calculate questions per category for balanced dataset
    questions_per_category = target_size // len(target_categories)
    remainder = target_size % len(target_categories)

    print(f"\nTarget: {questions_per_category} questions per category")
    print(f"Categories: {len(target_categories)}")
    print(f"Total target: {target_size}")

    # Build balanced dataset
    balanced_dataset = []
    gaps_identified = {}

    for i, cat in enumerate(target_categories):
        # Add remainder to first few categories
        target_for_cat = questions_per_category + (1 if i < remainder else 0)

        available = categorized[cat]

        if len(available) >= target_for_cat:
            # We have enough - randomly sample
            selected = random.sample(available, target_for_cat)
            balanced_dataset.extend(selected)
            print(f"✅ {cat:20} : Selected {target_for_cat:4} / {len(available):6} available")
        else:
            # Not enough - take all and note the gap
            balanced_dataset.extend(available)
            gap = target_for_cat - len(available)
            gaps_identified[cat] = gap
            print(f"⚠️  {cat:20} : Only {len(available):4} available (need {gap:4} more)")

    # Shuffle the balanced dataset
    random.shuffle(balanced_dataset)

    # Save balanced dataset
    balanced_file = output_path / "balanced_10k_dataset.json"
    with open(balanced_file, 'w') as f:
        json.dump(balanced_dataset, f, indent=2)

    print(f"\n✅ Balanced dataset saved: {len(balanced_dataset):,} questions")
    print(f"   Location: {balanced_file}")

    # Identify specific gaps to fill
    print("\n" + "="*60)
    print("GAPS TO FILL WITH GENERATION")
    print("="*60)

    total_gap = sum(gaps_identified.values())

    if total_gap > 0:
        print(f"Total questions needed: {total_gap}")
        print("\nBreakdown by category:")
        for cat, gap in sorted(gaps_identified.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat:20} : {gap:4} questions needed")

        # Create generation targets file
        generation_targets = {
            "total_needed": total_gap,
            "categories": {}
        }

        # Define specific prompts for each gap category
        category_prompts = {
            'computer_science': [
                "algorithm complexity and Big O notation",
                "data structures (trees, graphs, hash tables)",
                "programming concepts and debugging",
                "software engineering principles"
            ],
            'logic_reasoning': [
                "formal logic and syllogisms",
                "constraint satisfaction problems",
                "logical paradoxes and puzzles",
                "deductive and inductive reasoning"
            ],
            'critical_thinking': [
                "identifying logical fallacies",
                "evaluating arguments and evidence",
                "distinguishing fact from opinion",
                "analyzing assumptions and biases"
            ],
            'common_sense': [
                "everyday physics and causality",
                "social situations and norms",
                "practical problem solving",
                "temporal and spatial reasoning"
            ]
        }

        for cat, gap in gaps_identified.items():
            generation_targets["categories"][cat] = {
                "needed": gap,
                "suggested_topics": category_prompts.get(cat, ["general " + cat.replace('_', ' ') + " concepts"])
            }

        targets_file = output_path / "generation_targets.json"
        with open(targets_file, 'w') as f:
            json.dump(generation_targets, f, indent=2)

        print(f"\n✅ Generation targets saved to: {targets_file}")
    else:
        print("No gaps identified - dataset is complete!")

    # Save category statistics
    stats = {
        "total_questions": len(balanced_dataset),
        "target_size": target_size,
        "categories": {}
    }

    for cat in target_categories:
        cat_questions = [q for q in balanced_dataset if q.get('assigned_category') == cat]
        stats["categories"][cat] = {
            "count": len(cat_questions),
            "percentage": len(cat_questions) / len(balanced_dataset) * 100
        }

    stats_file = output_path / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("FINAL BALANCED DATASET STATISTICS")
    print("="*60)
    for cat in target_categories:
        count = stats["categories"][cat]["count"]
        pct = stats["categories"][cat]["percentage"]
        print(f"{cat:20} : {count:4} questions ({pct:5.1f}%)")

    return balanced_dataset, gaps_identified


def generate_gap_filling_script(gaps: Dict[str, int], output_dir="data/balanced_10k"):
    """Create a script to generate questions for identified gaps"""

    if not gaps:
        print("No gaps to fill!")
        return

    output_path = Path(output_dir)

    script_content = '''"""
Auto-generated script to fill gaps in balanced dataset
Targets specific categories that need more questions
"""

import json
from pathlib import Path

# Gap filling targets
GAPS_TO_FILL = '''

    script_content += json.dumps(gaps, indent=4)

    script_content += '''

def generate_questions_for_gaps():
    """Generate targeted questions for each gap category"""

    print("Generating questions to fill gaps...")
    print(f"Total needed: {sum(GAPS_TO_FILL.values())}")

    # TODO: Implement generation logic here
    # This would use the same generation approach but with targeted prompts

    for category, count in GAPS_TO_FILL.items():
        print(f"\\nGenerating {count} questions for {category}...")
        # Add generation code here

if __name__ == "__main__":
    generate_questions_for_gaps()
'''

    script_file = output_path / "fill_gaps.py"
    with open(script_file, 'w') as f:
        f.write(script_content)

    print(f"\nGap-filling script created: {script_file}")


if __name__ == "__main__":
    print("Creating balanced 10,000 question dataset...")
    print("This will ensure equal coverage across all topic areas")
    print()

    # Create balanced dataset
    balanced_data, gaps = create_balanced_dataset()

    # Generate gap-filling script if needed
    if gaps:
        generate_gap_filling_script(gaps)

        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Review the balanced dataset (10k questions)")
        print("2. Generate questions to fill identified gaps")
        print("3. Combine for final training dataset")
        print("4. Begin fine-tuning on balanced dataset")
    else:
        print("\n✅ Dataset is perfectly balanced and ready for training!")