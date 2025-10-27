#!/usr/bin/env python3
"""
Comprehensive analysis of knowledge gaps in our Q&A dataset
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set

def extract_topics_from_question(q: Dict) -> Set[str]:
    """Extract all possible topics from a question"""
    text = q.get('question', '').lower()
    text += ' ' + ' '.join(str(v).lower() for v in q.get('choices', {}).values())
    text += ' ' + q.get('subject', '').lower()
    text += ' ' + q.get('topic', '').lower()

    topics = set()

    # Comprehensive topic detection
    topic_patterns = {
        # STEM Core
        'physics': ['physics', 'force', 'energy', 'momentum', 'gravity', 'thermodynamic', 'quantum', 'relativity', 'newton', 'electromagnetic', 'wave', 'particle', 'motion', 'velocity', 'acceleration'],
        'chemistry': ['chemical', 'element', 'compound', 'reaction', 'molecule', 'atom', 'periodic table', 'bond', 'acid', 'base', 'oxidation', 'reduction', 'organic', 'inorganic'],
        'biology': ['biology', 'cell', 'dna', 'rna', 'gene', 'evolution', 'species', 'organism', 'ecosystem', 'photosynthesis', 'respiration', 'protein', 'enzyme', 'mutation'],
        'mathematics': ['math', 'equation', 'calculate', 'algebra', 'geometry', 'calculus', 'statistics', 'probability', 'trigonometry', 'derivative', 'integral', 'matrix', 'vector'],
        'computer_science': ['algorithm', 'programming', 'code', 'software', 'database', 'network', 'data structure', 'complexity', 'binary', 'recursion', 'loop', 'function', 'class', 'object'],

        # Applied Sciences
        'engineering': ['engineering', 'design', 'circuit', 'mechanical', 'electrical', 'civil', 'structural', 'material', 'system', 'optimization'],
        'medicine': ['medical', 'disease', 'treatment', 'diagnosis', 'patient', 'symptom', 'drug', 'surgery', 'anatomy', 'physiology', 'pathology'],
        'technology': ['technology', 'innovation', 'digital', 'internet', 'artificial intelligence', 'machine learning', 'robotics', 'automation', 'cyber'],

        # Social Sciences
        'psychology': ['psychology', 'behavior', 'cognitive', 'mental', 'personality', 'emotion', 'memory', 'learning', 'perception', 'consciousness'],
        'sociology': ['sociology', 'society', 'culture', 'social', 'community', 'group', 'institution', 'demographic', 'urbanization'],
        'economics': ['economy', 'market', 'supply', 'demand', 'price', 'inflation', 'gdp', 'trade', 'finance', 'investment', 'budget'],
        'political_science': ['political', 'government', 'democracy', 'policy', 'election', 'constitution', 'law', 'rights', 'power', 'state'],

        # Humanities
        'history': ['history', 'historical', 'ancient', 'medieval', 'modern', 'war', 'revolution', 'empire', 'civilization', 'century'],
        'philosophy': ['philosophy', 'ethics', 'moral', 'logic', 'metaphysics', 'epistemology', 'existential', 'truth', 'knowledge', 'reality'],
        'literature': ['literature', 'novel', 'poetry', 'author', 'literary', 'narrative', 'theme', 'character', 'plot', 'genre'],
        'linguistics': ['language', 'linguistic', 'grammar', 'syntax', 'semantics', 'phonetics', 'morphology', 'dialect', 'translation'],

        # Arts & Culture
        'art': ['art', 'painting', 'sculpture', 'artist', 'artistic', 'museum', 'gallery', 'renaissance', 'modern art', 'abstract'],
        'music': ['music', 'melody', 'rhythm', 'composer', 'instrument', 'symphony', 'jazz', 'classical', 'harmony', 'scale'],
        'film_media': ['film', 'movie', 'cinema', 'director', 'media', 'television', 'broadcast', 'journalism', 'documentary'],

        # Geography & Environment
        'geography': ['geography', 'map', 'continent', 'country', 'city', 'mountain', 'river', 'ocean', 'climate', 'region'],
        'environmental_science': ['environment', 'ecology', 'pollution', 'climate change', 'sustainability', 'conservation', 'renewable', 'biodiversity'],
        'earth_science': ['geology', 'rock', 'mineral', 'earthquake', 'volcano', 'plate tectonics', 'fossil', 'sediment', 'erosion'],
        'astronomy': ['astronomy', 'star', 'planet', 'galaxy', 'universe', 'solar system', 'telescope', 'orbit', 'cosmic', 'space'],

        # Practical Knowledge
        'business': ['business', 'management', 'marketing', 'accounting', 'entrepreneur', 'strategy', 'leadership', 'organization', 'profit'],
        'law': ['law', 'legal', 'court', 'judge', 'attorney', 'crime', 'contract', 'regulation', 'statute', 'jurisdiction'],
        'education': ['education', 'teaching', 'learning', 'curriculum', 'pedagogy', 'assessment', 'school', 'university', 'student'],
        'sports': ['sport', 'athlete', 'game', 'competition', 'olympic', 'football', 'basketball', 'soccer', 'baseball', 'tennis'],

        # Life Skills & Common Sense
        'health_nutrition': ['health', 'nutrition', 'diet', 'exercise', 'fitness', 'vitamin', 'calorie', 'wellness', 'lifestyle'],
        'personal_finance': ['personal finance', 'budget', 'savings', 'credit', 'loan', 'mortgage', 'retirement', 'insurance', 'tax'],
        'cooking_food': ['cooking', 'recipe', 'ingredient', 'cuisine', 'baking', 'culinary', 'meal', 'kitchen', 'restaurant'],
        'everyday_life': ['daily', 'routine', 'household', 'practical', 'common sense', 'everyday', 'typical', 'usual'],

        # Logic & Reasoning
        'formal_logic': ['syllogism', 'premise', 'conclusion', 'deductive', 'inductive', 'fallacy', 'valid', 'sound', 'argument'],
        'critical_thinking': ['analyze', 'evaluate', 'evidence', 'bias', 'assumption', 'correlation', 'causation', 'reasoning'],
        'problem_solving': ['problem', 'solution', 'strategy', 'puzzle', 'riddle', 'challenge', 'approach', 'method'],

        # Current Events & Pop Culture
        'current_events': ['recent', 'current', '2020', '2021', '2022', '2023', '2024', 'news', 'trending', 'viral'],
        'pop_culture': ['celebrity', 'entertainment', 'social media', 'meme', 'trend', 'fashion', 'popular', 'mainstream'],

        # Specialized Areas
        'mythology': ['myth', 'legend', 'folklore', 'deity', 'hero', 'epic', 'fable', 'tale'],
        'religion': ['religion', 'faith', 'belief', 'worship', 'scripture', 'prayer', 'sacred', 'spiritual'],
        'agriculture': ['agriculture', 'farming', 'crop', 'livestock', 'harvest', 'soil', 'irrigation', 'cultivation'],
        'military': ['military', 'army', 'navy', 'warfare', 'strategy', 'tactics', 'weapon', 'defense'],
        'transportation': ['transport', 'vehicle', 'traffic', 'aviation', 'maritime', 'railway', 'highway', 'logistics']
    }

    # Check each topic pattern
    for topic, keywords in topic_patterns.items():
        if any(keyword in text for keyword in keywords):
            topics.add(topic)

    return topics

def analyze_dataset_coverage():
    """Analyze comprehensive topic coverage"""

    # Load the final dataset
    dataset_path = Path("data/final_training/complete_10k_no_generation.json")

    if not dataset_path.exists():
        print("❌ Dataset not found!")
        return

    with open(dataset_path) as f:
        data = json.load(f)

    print("=" * 70)
    print("COMPREHENSIVE KNOWLEDGE GAP ANALYSIS")
    print("=" * 70)
    print(f"Analyzing {len(data)} questions for topic coverage...\n")

    # Extract topics from all questions
    topic_coverage = defaultdict(int)
    questions_by_topic = defaultdict(list)

    for q in data:
        topics = extract_topics_from_question(q)
        if not topics:
            topics = {'uncategorized'}

        for topic in topics:
            topic_coverage[topic] += 1
            questions_by_topic[topic].append(q['question'][:100])

    # Sort topics by coverage
    sorted_topics = sorted(topic_coverage.items(), key=lambda x: x[1], reverse=True)

    # Identify well-covered vs gaps
    total_questions = len(data)
    well_covered = []
    moderate_coverage = []
    gaps = []
    critical_gaps = []

    for topic, count in sorted_topics:
        percentage = (count / total_questions) * 100

        if percentage >= 5.0:
            well_covered.append((topic, count, percentage))
        elif percentage >= 2.0:
            moderate_coverage.append((topic, count, percentage))
        elif percentage >= 0.5:
            gaps.append((topic, count, percentage))
        else:
            critical_gaps.append((topic, count, percentage))

    # Display results
    print("📊 WELL-COVERED TOPICS (≥5% of dataset)")
    print("-" * 60)
    for topic, count, pct in well_covered[:15]:
        bar = "█" * int(pct * 2)
        print(f"  {topic:25}: {count:4} ({pct:5.1f}%) {bar}")

    print("\n📈 MODERATE COVERAGE (2-5% of dataset)")
    print("-" * 60)
    for topic, count, pct in moderate_coverage[:10]:
        bar = "▓" * int(pct * 2)
        print(f"  {topic:25}: {count:4} ({pct:5.1f}%) {bar}")

    print("\n⚠️  GAPS (0.5-2% of dataset)")
    print("-" * 60)
    for topic, count, pct in gaps[:10]:
        bar = "░" * max(1, int(pct * 2))
        print(f"  {topic:25}: {count:4} ({pct:5.1f}%) {bar}")

    print("\n🚨 CRITICAL GAPS (<0.5% of dataset)")
    print("-" * 60)
    for topic, count, pct in critical_gaps[:10]:
        print(f"  {topic:25}: {count:4} ({pct:5.1f}%)")

    # Topics with zero coverage
    all_possible_topics = {
        'physics', 'chemistry', 'biology', 'mathematics', 'computer_science',
        'engineering', 'medicine', 'technology', 'psychology', 'sociology',
        'economics', 'political_science', 'history', 'philosophy', 'literature',
        'linguistics', 'art', 'music', 'film_media', 'geography',
        'environmental_science', 'earth_science', 'astronomy', 'business', 'law',
        'education', 'sports', 'health_nutrition', 'personal_finance', 'cooking_food',
        'everyday_life', 'formal_logic', 'critical_thinking', 'problem_solving',
        'current_events', 'pop_culture', 'mythology', 'religion', 'agriculture',
        'military', 'transportation'
    }

    missing_topics = all_possible_topics - set(topic_coverage.keys())

    if missing_topics:
        print("\n❌ COMPLETELY MISSING TOPICS")
        print("-" * 60)
        for topic in sorted(missing_topics):
            print(f"  • {topic}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("📋 SUMMARY STATISTICS")
    print("-" * 60)
    print(f"  Total unique topics covered: {len(topic_coverage)}")
    print(f"  Topics with >5% coverage: {len(well_covered)}")
    print(f"  Topics with 2-5% coverage: {len(moderate_coverage)}")
    print(f"  Topics with <2% coverage: {len(gaps) + len(critical_gaps)}")
    print(f"  Completely missing topics: {len(missing_topics)}")

    # Average questions per topic
    avg_per_topic = sum(topic_coverage.values()) / len(topic_coverage)
    print(f"  Average questions per topic: {avg_per_topic:.1f}")

    # Coverage concentration
    top_10_coverage = sum(count for _, count in sorted_topics[:10])
    print(f"  Top 10 topics cover: {top_10_coverage/total_questions*100:.1f}% of dataset")

    # Recommendations
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATIONS")
    print("-" * 60)

    critical_to_add = []

    # Check for important missing areas
    important_topics = {
        'current_events': "Recent events and contemporary issues",
        'sports': "Sports knowledge and athletics",
        'pop_culture': "Entertainment and popular culture",
        'cooking_food': "Culinary arts and food preparation",
        'personal_finance': "Financial literacy and money management",
        'mythology': "World mythology and folklore",
        'transportation': "Transportation systems and vehicles",
        'military': "Military history and defense"
    }

    for topic, description in important_topics.items():
        if topic in missing_topics or (topic in topic_coverage and topic_coverage[topic] < 20):
            critical_to_add.append((topic, description))

    if critical_to_add:
        print("Priority topics to add for comprehensive Q&A coverage:\n")
        for topic, desc in critical_to_add:
            current = topic_coverage.get(topic, 0)
            print(f"  • {topic:20}: {desc}")
            print(f"    Current: {current} questions | Recommended: 100+ questions\n")

    # Final verdict
    print("\n" + "=" * 70)
    print("✅ FINAL ASSESSMENT")
    print("-" * 60)

    coverage_score = len(well_covered) + (len(moderate_coverage) * 0.5)
    max_score = len(all_possible_topics)

    print(f"Coverage Score: {coverage_score:.1f} / {max_score} ({coverage_score/max_score*100:.1f}%)")

    if coverage_score / max_score > 0.7:
        print("\n✅ Good coverage for general Q&A tournament")
        print("   Minor gaps can be addressed during training")
    elif coverage_score / max_score > 0.5:
        print("\n⚠️  Moderate coverage - some important gaps exist")
        print("   Consider adding questions for missing topics")
    else:
        print("\n🚨 Significant gaps in coverage")
        print("   Strongly recommend filling critical gaps before training")

    # Save gap analysis
    gap_report = {
        'total_questions': total_questions,
        'topics_covered': len(topic_coverage),
        'well_covered': [(t, c, p) for t, c, p in well_covered],
        'gaps': [(t, c, p) for t, c, p in gaps + critical_gaps],
        'missing': list(missing_topics),
        'recommendations': [{'topic': t, 'description': d} for t, d in critical_to_add]
    }

    with open('data/balanced_10k/gap_analysis_report.json', 'w') as f:
        json.dump(gap_report, f, indent=2)

    print(f"\n📁 Detailed report saved to: data/balanced_10k/gap_analysis_report.json")

if __name__ == "__main__":
    analyze_dataset_coverage()