"""
Validate and filter external datasets for quality
Also identify gaps where we might want to generate specific questions
"""

import json
import random
from pathlib import Path
from typing import Dict, List

def validate_mcq_question(question: Dict) -> tuple[bool, str]:
    """Validate a single MCQ question for quality"""

    # Check required fields
    if not question.get('question'):
        return False, "Missing question text"

    if not question.get('choices'):
        return False, "Missing choices"

    if not question.get('correct_answer'):
        return False, "Missing correct answer"

    # Check choices format
    choices = question['choices']
    if not isinstance(choices, dict):
        return False, "Choices not in dict format"

    if len(choices) < 4:
        return False, f"Only {len(choices)} choices (need 4)"

    # Check correct answer is valid
    correct = question['correct_answer']
    if correct not in choices:
        return False, f"Correct answer '{correct}' not in choices"

    # Check for duplicate choices
    choice_values = list(choices.values())
    if len(set(choice_values)) != len(choice_values):
        return False, "Duplicate choices found"

    # Check question isn't too short/long
    q_text = question['question']
    if len(q_text) < 10:
        return False, "Question too short"
    if len(q_text) > 5000:
        return False, "Question too long (likely includes full article)"

    # Check for common quality issues
    if q_text.count('?') > 3:
        return False, "Multiple questions in one"

    # Check if all choices are reasonable length
    for choice in choices.values():
        if len(str(choice)) < 1:
            return False, "Empty choice"
        if len(str(choice)) > 500:
            return False, "Choice too long"

    return True, "Valid"


def analyze_dataset_quality(dataset_path: str) -> Dict:
    """Analyze quality of a dataset"""

    with open(dataset_path) as f:
        questions = json.load(f)

    print(f"\nAnalyzing: {dataset_path}")
    print(f"Total questions: {len(questions)}")

    valid_count = 0
    issues = {}
    sample_questions = []

    for i, q in enumerate(questions):
        is_valid, reason = validate_mcq_question(q)

        if is_valid:
            valid_count += 1
            # Save some valid samples
            if len(sample_questions) < 5:
                sample_questions.append(q)
        else:
            if reason not in issues:
                issues[reason] = []
            if len(issues[reason]) < 3:  # Keep examples of each issue
                issues[reason].append(i)

    # Analysis results
    results = {
        'total': len(questions),
        'valid': valid_count,
        'invalid': len(questions) - valid_count,
        'validity_rate': valid_count / len(questions) * 100,
        'issues': issues,
        'samples': sample_questions
    }

    # Print summary
    print(f"Valid: {valid_count}/{len(questions)} ({results['validity_rate']:.1f}%)")

    if issues:
        print("Issues found:")
        for issue, indices in issues.items():
            print(f"  - {issue}: {len(indices)} occurrences")

    return results


def filter_and_combine_datasets(output_dir="data/final_training"):
    """Filter all datasets and combine into final training set"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_valid_questions = []

    # 1. Load and validate our generated questions
    print("\n" + "="*60)
    print("VALIDATING OUR GENERATED QUESTIONS")
    print("="*60)

    if Path("data/synthetic/validated_questions.json").exists():
        with open("data/synthetic/validated_questions.json") as f:
            our_questions = json.load(f)

        for q in our_questions:
            is_valid, _ = validate_mcq_question(q)
            if is_valid:
                q['source'] = 'generated'
                q['quality'] = 'high'  # Cross-validated
                all_valid_questions.append(q)

        print(f"✅ Our generated: {len([q for q in all_valid_questions if q['source'] == 'generated'])} valid questions")

    # 2. Load and validate external datasets
    print("\n" + "="*60)
    print("VALIDATING EXTERNAL DATASETS")
    print("="*60)

    external_files = [
        "data/external_datasets/external_training_ready.json",
        "data/massive_datasets/massive_training_ready.json"
    ]

    for file_path in external_files:
        if Path(file_path).exists():
            results = analyze_dataset_quality(file_path)

            # Load and filter
            with open(file_path) as f:
                questions = json.load(f)

            for q in questions:
                is_valid, _ = validate_mcq_question(q)
                if is_valid:
                    q['quality'] = 'external'
                    all_valid_questions.append(q)

    # 3. Analyze coverage
    print("\n" + "="*60)
    print("COVERAGE ANALYSIS")
    print("="*60)

    # Count by source
    from collections import Counter
    sources = Counter([q.get('source', 'unknown') for q in all_valid_questions])

    print("Questions by source:")
    for source, count in sources.most_common():
        print(f"  {source}: {count:,}")

    # 4. Identify gaps
    print("\n" + "="*60)
    print("IDENTIFYING GAPS")
    print("="*60)

    # Check topic coverage
    topics_covered = set()
    for q in all_valid_questions:
        if 'subject' in q:
            topics_covered.add(q['subject'])
        elif 'category' in q:
            topics_covered.add(q['category'])
        elif 'topic' in q:
            topics_covered.add(q['topic'])

    print(f"Unique topics/subjects covered: {len(topics_covered)}")

    # Identify potential gaps
    important_topics = {
        'logic_puzzles', 'coding', 'current_events', 'practical_reasoning',
        'safety_questions', 'ethics', 'creativity', 'analogies'
    }

    missing_topics = important_topics - topics_covered
    if missing_topics:
        print(f"Potentially missing topics: {missing_topics}")
        print("Consider generating specific questions for these areas")

    # 5. Save final dataset
    print("\n" + "="*60)
    print("SAVING FINAL DATASET")
    print("="*60)

    # Shuffle for good measure
    random.shuffle(all_valid_questions)

    # Save full dataset
    full_path = output_path / "final_training_full.json"
    with open(full_path, 'w') as f:
        json.dump(all_valid_questions, f, indent=2)

    print(f"Total valid questions: {len(all_valid_questions):,}")
    print(f"Saved to: {full_path}")

    # Create a smaller high-quality subset (best of both)
    high_quality = [q for q in all_valid_questions if q.get('quality') == 'high']
    external_sample = random.sample(
        [q for q in all_valid_questions if q.get('quality') == 'external'],
        min(2000, len([q for q in all_valid_questions if q.get('quality') == 'external']))
    )

    balanced_set = high_quality + external_sample
    random.shuffle(balanced_set)

    balanced_path = output_path / "final_training_balanced.json"
    with open(balanced_path, 'w') as f:
        json.dump(balanced_set, f, indent=2)

    print(f"Balanced high-quality set: {len(balanced_set):,} questions")
    print(f"Saved to: {balanced_path}")

    return all_valid_questions


def recommend_generation_targets(valid_questions: List[Dict]):
    """Recommend specific areas where generation would help"""

    print("\n" + "="*60)
    print("GENERATION RECOMMENDATIONS")
    print("="*60)

    # Analyze what we have
    has_logic = any('logic' in str(q).lower() for q in valid_questions[:100])
    has_trick = any('trick' in str(q).lower() for q in valid_questions[:100])
    has_coding = any('code' in str(q).lower() or 'program' in str(q).lower() for q in valid_questions[:100])

    recommendations = []

    if not has_trick:
        recommendations.append("Trick questions (catch opponents off-guard)")

    if not has_coding:
        recommendations.append("Programming/algorithm questions (growing topic)")

    # Always recommend these high-value categories
    recommendations.extend([
        "Adversarial questions (designed to be VERY hard)",
        "Questions with subtle distinctions between choices",
        "Time-sensitive questions that require careful reading"
    ])

    print("Recommended generation targets:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print(f"\nSuggestion: Generate ~50-100 questions in each area")
    print(f"Time estimate: 1-2 hours per category with focused prompts")

    return recommendations


if __name__ == "__main__":
    print("Starting dataset validation and filtering...")
    print()

    # Run validation
    valid_questions = filter_and_combine_datasets()

    # Get recommendations
    recommendations = recommend_generation_targets(valid_questions)

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Download massive datasets if not done (10K+ questions)")
    print("2. Run this validator to filter bad questions")
    print("3. Optionally generate 200-500 specialized questions")
    print("4. Proceed with training on final dataset")