#!/usr/bin/env python3
"""
Evaluation Framework for Q&A Agent
Test question quality and answer accuracy
"""

import sys
from pathlib import Path
import argparse
import json
from typing import List, Dict
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))


class QAEvaluator:
    """Evaluate Q&A agent performance"""

    def __init__(self):
        self.question_scores = []
        self.answer_scores = []

    def evaluate_question_quality(self, question: str) -> Dict:
        """
        Evaluate question quality based on heuristics

        Criteria:
        - Length (not too short, not too long)
        - Starts with question word
        - Ends with question mark
        - Contains topic-relevant keywords
        - Not too simple (no yes/no unless complex)
        """
        score = {"total": 0, "max": 100, "breakdown": {}}

        # Length check (20 points)
        length = len(question)
        if 30 <= length <= 200:
            length_score = 20
        elif 20 <= length < 30 or 200 < length <= 250:
            length_score = 15
        elif 10 <= length < 20 or 250 < length <= 300:
            length_score = 10
        else:
            length_score = 5
        score["breakdown"]["length"] = length_score
        score["total"] += length_score

        # Question format (20 points)
        question_words = ["what", "why", "how", "when", "where", "who", "which", "explain", "describe"]
        starts_with_q_word = any(question.lower().startswith(w) for w in question_words)
        ends_with_qmark = question.strip().endswith("?")

        format_score = 0
        if starts_with_q_word:
            format_score += 10
        if ends_with_qmark:
            format_score += 10
        score["breakdown"]["format"] = format_score
        score["total"] += format_score

        # Complexity (20 points) - avoid yes/no questions
        yes_no_indicators = ["is it", "are you", "do you", "does it", "can you", "will you"]
        is_yes_no = any(question.lower().startswith(ind) for ind in yes_no_indicators)

        complexity_score = 15 if not is_yes_no else 5
        score["breakdown"]["complexity"] = complexity_score
        score["total"] += complexity_score

        # Specificity (20 points) - has specific terms
        has_specific_terms = len(question.split()) > 5  # More than 5 words
        specificity_score = 20 if has_specific_terms else 10
        score["breakdown"]["specificity"] = specificity_score
        score["total"] += specificity_score

        # Clarity (20 points) - no obvious issues
        clarity_issues = 0
        if "  " in question:  # Double spaces
            clarity_issues += 1
        if question.count("?") > 1:  # Multiple question marks
            clarity_issues += 1
        if len(question.split()) < 3:  # Too few words
            clarity_issues += 1

        clarity_score = 20 - (clarity_issues * 7)
        score["breakdown"]["clarity"] = max(clarity_score, 0)
        score["total"] += clarity_score

        score["percentage"] = (score["total"] / score["max"]) * 100

        return score

    def evaluate_answer_quality(self, answer: str, question: str = "") -> Dict:
        """
        Evaluate answer quality based on heuristics

        Criteria:
        - Length (substantial but concise)
        - Completeness (not truncated)
        - Relevance (relates to question)
        - Clarity (well-structured)
        """
        score = {"total": 0, "max": 100, "breakdown": {}}

        # Length check (25 points)
        length = len(answer)
        if 50 <= length <= 500:
            length_score = 25
        elif 30 <= length < 50 or 500 < length <= 700:
            length_score = 20
        elif 20 <= length < 30 or 700 < length <= 1000:
            length_score = 15
        else:
            length_score = 10
        score["breakdown"]["length"] = length_score
        score["total"] += length_score

        # Completeness (25 points) - not obviously truncated
        truncation_indicators = ["...", " and", " or", " but"]
        ends_badly = any(answer.strip().endswith(ind) for ind in truncation_indicators)
        completeness_score = 15 if ends_badly else 25
        score["breakdown"]["completeness"] = completeness_score
        score["total"] += completeness_score

        # Structure (25 points) - has sentences
        has_period = "." in answer
        sentence_count = answer.count(". ") + (1 if has_period else 0)

        if sentence_count >= 2:
            structure_score = 25
        elif sentence_count == 1:
            structure_score = 20
        else:
            structure_score = 10
        score["breakdown"]["structure"] = structure_score
        score["total"] += structure_score

        # Informativeness (25 points) - not too vague
        vague_phrases = ["it depends", "not sure", "maybe", "possibly", "i don't know"]
        is_vague = any(phrase in answer.lower() for phrase in vague_phrases)

        info_score = 15 if is_vague else 25
        score["breakdown"]["informativeness"] = info_score
        score["total"] += info_score

        score["percentage"] = (score["total"] / score["max"]) * 100

        return score

    def evaluate_qa_pair(self, question: str, answer: str) -> Dict:
        """Evaluate a Q&A pair"""
        q_score = self.evaluate_question_quality(question)
        a_score = self.evaluate_answer_quality(answer, question)

        self.question_scores.append(q_score["percentage"])
        self.answer_scores.append(a_score["percentage"])

        return {
            "question": question,
            "answer": answer,
            "question_score": q_score,
            "answer_score": a_score,
            "overall_score": (q_score["percentage"] + a_score["percentage"]) / 2,
        }

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics"""
        if not self.question_scores or not self.answer_scores:
            return {}

        return {
            "question_quality": {
                "mean": sum(self.question_scores) / len(self.question_scores),
                "min": min(self.question_scores),
                "max": max(self.question_scores),
                "count": len(self.question_scores),
            },
            "answer_quality": {
                "mean": sum(self.answer_scores) / len(self.answer_scores),
                "min": min(self.answer_scores),
                "max": max(self.answer_scores),
                "count": len(self.answer_scores),
            },
            "overall_quality": {
                "mean": (sum(self.question_scores) + sum(self.answer_scores))
                / (len(self.question_scores) + len(self.answer_scores)),
            },
        }


def evaluate_tournament_results(results_path: str):
    """Evaluate tournament results from JSON file"""
    print("=" * 70)
    print("Evaluating Tournament Results")
    print("=" * 70)

    with open(results_path, "r") as f:
        results = json.load(f)

    evaluator = QAEvaluator()

    print(f"\nFound {len(results)} rounds to evaluate\n")

    for i, round_result in enumerate(results, 1):
        print(f"\n--- Round {i} ---")

        # Evaluate my question
        if "my_question" in round_result:
            q = round_result["my_question"]
            q_score = evaluator.evaluate_question_quality(q)
            print(f"My Question: {q}")
            print(f"Question Score: {q_score['percentage']:.1f}%")
            evaluator.question_scores.append(q_score["percentage"])

        # Evaluate my answer
        if "my_answer" in round_result:
            a = round_result["my_answer"]
            a_score = evaluator.evaluate_answer_quality(a)
            print(f"My Answer: {a[:100]}...")
            print(f"Answer Score: {a_score['percentage']:.1f}%")
            evaluator.answer_scores.append(a_score["percentage"])

    # Summary statistics
    summary = evaluator.get_summary_statistics()

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if "question_quality" in summary:
        q_stats = summary["question_quality"]
        print(f"\nQuestion Quality:")
        print(f"  Mean: {q_stats['mean']:.1f}%")
        print(f"  Range: {q_stats['min']:.1f}% - {q_stats['max']:.1f}%")
        print(f"  Count: {q_stats['count']}")

    if "answer_quality" in summary:
        a_stats = summary["answer_quality"]
        print(f"\nAnswer Quality:")
        print(f"  Mean: {a_stats['mean']:.1f}%")
        print(f"  Range: {a_stats['min']:.1f}% - {a_stats['max']:.1f}%")
        print(f"  Count: {a_stats['count']}")

    if "overall_quality" in summary:
        print(f"\nOverall Quality: {summary['overall_quality']['mean']:.1f}%")

    print("=" * 70)

    return summary


def evaluate_generated_questions(questions_file: str):
    """Evaluate a file of generated questions"""
    print("=" * 70)
    print("Evaluating Generated Questions")
    print("=" * 70)

    with open(questions_file, "r") as f:
        questions = [line.strip() for line in f if line.strip()]

    evaluator = QAEvaluator()

    print(f"\nEvaluating {len(questions)} questions...\n")

    for i, question in enumerate(questions, 1):
        score = evaluator.evaluate_question_quality(question)
        evaluator.question_scores.append(score["percentage"])
        print(f"{i}. {question}")
        print(f"   Score: {score['percentage']:.1f}%")
        print()

    # Summary
    if evaluator.question_scores:
        mean_score = sum(evaluator.question_scores) / len(evaluator.question_scores)
        print("=" * 70)
        print(f"Average Question Quality: {mean_score:.1f}%")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q&A agent performance")
    parser.add_argument(
        "--mode",
        type=str,
        default="tournament",
        choices=["tournament", "questions", "answers", "pair"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="evaluation/tournament_results.json",
        help="Input file to evaluate",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to evaluate",
    )
    parser.add_argument(
        "--answer",
        type=str,
        help="Single answer to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for detailed results",
    )

    args = parser.parse_args()

    if args.mode == "tournament":
        summary = evaluate_tournament_results(args.input)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n✓ Detailed results saved to: {args.output}")

    elif args.mode == "questions":
        evaluate_generated_questions(args.input)

    elif args.mode == "pair":
        if not args.question or not args.answer:
            print("Error: --question and --answer required for pair mode")
            return

        evaluator = QAEvaluator()
        result = evaluator.evaluate_qa_pair(args.question, args.answer)

        print("\n" + "=" * 70)
        print("Q&A Pair Evaluation")
        print("=" * 70)
        print(f"\nQuestion: {result['question']}")
        print(f"Question Score: {result['question_score']['percentage']:.1f}%")
        print(f"\nAnswer: {result['answer']}")
        print(f"Answer Score: {result['answer_score']['percentage']:.1f}%")
        print(f"\nOverall Score: {result['overall_score']:.1f}%")
        print("=" * 70)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
