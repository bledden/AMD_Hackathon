"""
Analyze if gaps can be filled from existing surplus questions
"""

import json
from pathlib import Path
from collections import defaultdict
import re

def deep_categorize_question(question: dict) -> list:
    """
    Return ALL categories a question could belong to, not just primary
    """
    q_text = question.get('question', '').lower()
    options_text = ' '.join(str(v).lower() for v in question.get('options', {}).values())
    full_text = q_text + ' ' + options_text

    # Also check metadata
    subject = question.get('subject', '').lower()
    topic = question.get('topic', '').lower()
    meta_text = subject + ' ' + topic

    categories = []

    # Common sense - practical everyday reasoning
    if any(word in full_text for word in [
        'everyday', 'common', 'typical', 'usually', 'normally', 'practical',
        'daily', 'routine', 'household', 'social situation', 'interpersonal',
        'cause and effect', 'why would', 'what would happen', 'likely result',
        'sensible', 'reasonable', 'obvious', 'intuitive', 'street smart'
    ]):
        categories.append('common_sense')

    # Humanities - human culture, behavior, society
    if any(word in full_text for word in [
        'culture', 'society', 'human behavior', 'psychology', 'sociology',
        'anthropology', 'philosophy', 'ethics', 'moral', 'art', 'literature',
        'music', 'religion', 'belief', 'tradition', 'customs', 'values',
        'social norm', 'gender', 'race', 'ethnicity', 'identity', 'emotion',
        'personality', 'motivation', 'relationship', 'community'
    ]):
        categories.append('humanities')

    # Critical thinking - analysis, evaluation, reasoning
    if any(word in full_text for word in [
        'analyze', 'evaluate', 'assess', 'critique', 'judge', 'argument',
        'fallacy', 'bias', 'assumption', 'evidence', 'conclusion', 'premise',
        'valid', 'invalid', 'sound', 'unsound', 'correlation', 'causation',
        'generalization', 'stereotype', 'propaganda', 'misleading', 'flawed'
    ]):
        categories.append('critical_thinking')

    # Logic and reasoning - formal logic, puzzles, patterns
    if any(word in full_text for word in [
        'logic', 'syllogism', 'deductive', 'inductive', 'if then', 'implies',
        'necessary', 'sufficient', 'all', 'some', 'none', 'pattern', 'sequence',
        'puzzle', 'riddle', 'constraint', 'given that', 'therefore', 'thus',
        'follows that', 'contradiction', 'tautology', 'truth table'
    ]):
        categories.append('logic_reasoning')

    # Computer science
    if any(word in full_text for word in [
        'algorithm', 'data structure', 'programming', 'code', 'software',
        'database', 'network', 'operating system', 'compiler', 'complexity',
        'big o', 'binary', 'hexadecimal', 'encryption', 'protocol', 'api',
        'stack', 'queue', 'tree', 'graph', 'hash', 'sort', 'search',
        'recursive', 'iteration', 'loop', 'function', 'object', 'class'
    ]):
        categories.append('computer_science')

    # Check subject metadata for additional hints
    if 'conceptual_physics' in meta_text or 'global_facts' in meta_text:
        categories.append('common_sense')

    if 'philosophy' in meta_text or 'moral' in meta_text or 'world_religions' in meta_text:
        categories.append('humanities')

    if 'logical_fallacies' in meta_text or 'formal_logic' in meta_text:
        if 'critical_thinking' not in categories:
            categories.append('critical_thinking')
        if 'logic_reasoning' not in categories:
            categories.append('logic_reasoning')

    return categories

def main():
    # Load the full validated dataset
    with open('data/massive_datasets/massive_training_ready.json') as f:
        all_questions = json.load(f)

    print(f"Total available questions: {len(all_questions)}")

    # Load current balanced dataset to see what we already included
    with open('data/balanced_10k/balanced_10k_dataset.json') as f:
        included = json.load(f)

    # Create set of included questions for comparison
    included_set = {q['question'] for q in included}

    # Get questions not yet included
    available = [q for q in all_questions if q['question'] not in included_set]
    print(f"Questions not yet included: {len(available)}")

    # Categories we need to fill
    gaps = {
        'common_sense': 554,
        'humanities': 409,
        'critical_thinking': 328,
        'logic_reasoning': 316,
        'computer_science': 92
    }

    print("\n" + "="*60)
    print("SEARCHING FOR GAP-FILLING CANDIDATES")
    print("="*60)

    # Find questions that could fill gaps
    gap_fillers = defaultdict(list)

    for question in available:
        # Get all possible categories for this question
        possible_cats = deep_categorize_question(question)

        # Check if it matches any gap category
        for gap_cat in gaps.keys():
            if gap_cat in possible_cats:
                gap_fillers[gap_cat].append(question)

    print("\nPotential gap fillers found:")
    total_fillable = 0

    for category, needed in gaps.items():
        found = len(gap_fillers[category])
        fillable = min(found, needed)
        total_fillable += fillable

        status = "✅" if found >= needed else "⚠️"
        print(f"{status} {category:20}: Found {found:4} / Need {needed:4}")

        if found > 0:
            # Show a few examples
            print(f"   Example questions:")
            for i, q in enumerate(gap_fillers[category][:2]):
                print(f"   {i+1}. {q['question'][:80]}...")

    print(f"\n📊 Total gaps fillable from existing: {total_fillable} / {sum(gaps.values())}")

    # Create filled dataset
    print("\n" + "="*60)
    print("CREATING FILLED DATASET")
    print("="*60)

    filled_questions = []
    remaining_gaps = {}

    for category, needed in gaps.items():
        available_for_cat = gap_fillers[category]

        if len(available_for_cat) >= needed:
            # We have enough, take what we need
            selected = available_for_cat[:needed]
            filled_questions.extend(selected)
            print(f"✅ {category}: Filled completely with {needed} questions")
        else:
            # Take what we have
            filled_questions.extend(available_for_cat)
            remaining = needed - len(available_for_cat)
            remaining_gaps[category] = remaining
            print(f"⚠️  {category}: Partially filled {len(available_for_cat)}/{needed}, still need {remaining}")

    # Save the gap-filled questions
    output_file = Path("data/balanced_10k/gap_filled_from_existing.json")
    with open(output_file, 'w') as f:
        json.dump(filled_questions, f, indent=2)

    print(f"\n💾 Saved {len(filled_questions)} gap-filling questions to: {output_file}")

    # Update generation targets with remaining gaps
    if remaining_gaps:
        print("\n" + "="*60)
        print("REMAINING GAPS AFTER FILLING FROM EXISTING")
        print("="*60)

        total_remaining = sum(remaining_gaps.values())
        print(f"Total questions still needed: {total_remaining}")

        for cat, need in remaining_gaps.items():
            print(f"  {cat:20}: {need} questions")

        # Save updated targets
        updated_targets = {
            'total_needed': total_remaining,
            'categories': {cat: {'needed': need} for cat, need in remaining_gaps.items()}
        }

        with open('data/balanced_10k/remaining_generation_targets.json', 'w') as f:
            json.dump(updated_targets, f, indent=2)

        print(f"\n💾 Updated generation targets saved")
    else:
        print("\n🎉 ALL GAPS CAN BE FILLED FROM EXISTING QUESTIONS!")
        print("No generation needed!")

    # Combine everything for final dataset
    print("\n" + "="*60)
    print("CREATING FINAL COMBINED DATASET")
    print("="*60)

    # Load original balanced dataset
    with open('data/balanced_10k/balanced_10k_dataset.json') as f:
        balanced = json.load(f)

    # Add gap fillers
    final_dataset = balanced + filled_questions

    # Save final dataset
    final_file = Path("data/final_training/complete_10k_no_generation.json")
    final_file.parent.mkdir(parents=True, exist_ok=True)

    with open(final_file, 'w') as f:
        json.dump(final_dataset, f, indent=2)

    print(f"✅ Final dataset created: {len(final_dataset)} questions")
    print(f"📁 Location: {final_file}")

    # Show final distribution
    from collections import Counter

    # Count using the assigned category or primary detection
    final_counts = Counter()
    for q in final_dataset:
        if 'assigned_category' in q:
            final_counts[q['assigned_category']] += 1
        else:
            # For gap fillers, use first detected category
            cats = deep_categorize_question(q)
            if cats:
                final_counts[cats[0]] += 1
            else:
                final_counts['general_knowledge'] += 1

    print("\n📊 Final Category Distribution:")
    for cat, count in sorted(final_counts.items()):
        print(f"  {cat:20}: {count:4} ({count/len(final_dataset)*100:.1f}%)")

if __name__ == "__main__":
    main()