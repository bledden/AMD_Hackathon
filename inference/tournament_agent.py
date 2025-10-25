#!/usr/bin/env python3
"""
Tournament Agent
Optimized for competition: strategic question generation and accurate answering
"""

import sys
from pathlib import Path
import argparse
from typing import Optional, Dict, List
import json

sys.path.append(str(Path(__file__).parent.parent))

import torch
from unsloth import FastLanguageModel


class TournamentAgent:
    """
    Optimized Q&A agent for tournament competition

    Strategy:
    - Generate challenging but fair questions
    - Answer with high accuracy and conciseness
    - Adapt difficulty based on opponent
    """

    def __init__(
        self,
        model_path: str,
        theme: str = "science",
        strategy: str = "balanced",
    ):
        """
        Initialize tournament agent

        Args:
            model_path: Path to fine-tuned model
            theme: Competition theme
            strategy: "aggressive", "balanced", or "defensive"
        """
        self.theme = theme
        self.strategy = strategy

        print(f"Initializing Tournament Agent")
        print(f"  Theme: {theme}")
        print(f"  Strategy: {strategy}")

        # Load model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(self.model)
        print("✓ Agent ready for battle")

        # Strategy parameters
        self._set_strategy_params()

    def _set_strategy_params(self):
        """Set generation parameters based on strategy"""
        if self.strategy == "aggressive":
            # Generate harder questions, more confident answers
            self.question_temp = 0.8
            self.answer_temp = 0.2
            self.difficulty = "very challenging"

        elif self.strategy == "balanced":
            # Balanced approach
            self.question_temp = 0.7
            self.answer_temp = 0.3
            self.difficulty = "challenging"

        elif self.strategy == "defensive":
            # Safer questions, very accurate answers
            self.question_temp = 0.6
            self.answer_temp = 0.1
            self.difficulty = "moderately challenging"

    def generate_question(
        self,
        specific_topic: Optional[str] = None,
        custom_difficulty: Optional[str] = None,
    ) -> str:
        """
        Generate a strategic question for tournament

        Args:
            specific_topic: Override default theme topic
            custom_difficulty: Override strategy difficulty

        Returns:
            Generated question string
        """
        topic = specific_topic or self.theme
        difficulty = custom_difficulty or self.difficulty

        # Craft strategic prompt
        prompt = f"""### Instruction:
Generate a {difficulty} question about {topic} that tests deep understanding rather than simple recall. The question should be clear, specific, and answerable with knowledge of the topic.

### Response:
"""

        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=self.question_temp,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            num_beams=1,
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if "### Response:" in generated:
            question = generated.split("### Response:")[-1].strip()
        else:
            question = generated.strip()

        # Remove any "Question:" prefix
        if question.startswith("Question:"):
            question = question[9:].strip()

        return question

    def answer_question(self, question: str) -> str:
        """
        Answer a question with high accuracy

        Args:
            question: Question to answer

        Returns:
            Answer string
        """
        prompt = f"""### Instruction:
Answer this question accurately and concisely. Provide a clear, direct answer without unnecessary elaboration.

Question: {question}

### Response:
"""

        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=self.answer_temp,
            top_p=0.95,
            do_sample=True if self.answer_temp > 0 else False,
            repetition_penalty=1.1,
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if "### Response:" in generated:
            answer = generated.split("### Response:")[-1].strip()
        else:
            answer = generated.strip()

        # Remove any "Answer:" prefix
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()

        return answer

    def battle_round(
        self,
        round_number: int,
        opponent_question: Optional[str] = None,
    ) -> Dict:
        """
        Execute one tournament battle round

        Args:
            round_number: Current round number
            opponent_question: Question from opponent (if any)

        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 70)
        print(f"TOURNAMENT ROUND {round_number}")
        print("=" * 70)

        results = {
            "round": round_number,
            "theme": self.theme,
            "strategy": self.strategy,
        }

        # Phase 1: Generate my question
        print("\n[QUESTION GENERATION PHASE]")
        my_question = self.generate_question()
        results["my_question"] = my_question
        print(f"→ My Question: {my_question}")

        # Phase 2: Answer opponent's question
        if opponent_question:
            print("\n[QUESTION ANSWERING PHASE]")
            print(f"← Opponent Question: {opponent_question}")
            my_answer = self.answer_question(opponent_question)
            results["opponent_question"] = opponent_question
            results["my_answer"] = my_answer
            print(f"→ My Answer: {my_answer}")

        print("\n" + "=" * 70)
        return results

    def multi_round_tournament(
        self,
        n_rounds: int = 3,
        opponent_questions: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Simulate multiple tournament rounds

        Args:
            n_rounds: Number of rounds to simulate
            opponent_questions: List of opponent questions (optional)

        Returns:
            List of round results
        """
        print("\n" + "=" * 70)
        print(f"MULTI-ROUND TOURNAMENT: {n_rounds} ROUNDS")
        print(f"Agent: {self.theme.upper()} | Strategy: {self.strategy.upper()}")
        print("=" * 70)

        all_results = []

        for i in range(n_rounds):
            opponent_q = None
            if opponent_questions and i < len(opponent_questions):
                opponent_q = opponent_questions[i]

            round_result = self.battle_round(i + 1, opponent_q)
            all_results.append(round_result)

        return all_results

    def save_results(self, results: List[Dict], output_path: str):
        """Save tournament results to file"""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tournament Q&A Agent")
    parser.add_argument(
        "--model-path",
        type=str,
        default="training/outputs/final_model",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="science",
        help="Competition theme",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="balanced",
        choices=["aggressive", "balanced", "defensive"],
        help="Battle strategy",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of tournament rounds to simulate",
    )
    parser.add_argument(
        "--opponent-questions",
        type=str,
        nargs="+",
        help="List of opponent questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/tournament_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Initialize agent
    agent = TournamentAgent(
        model_path=args.model_path,
        theme=args.theme,
        strategy=args.strategy,
    )

    # Run tournament
    results = agent.multi_round_tournament(
        n_rounds=args.rounds,
        opponent_questions=args.opponent_questions,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_results(results, str(output_path))

    # Summary
    print("\n" + "=" * 70)
    print("TOURNAMENT SUMMARY")
    print("=" * 70)
    print(f"Total Rounds: {len(results)}")
    print(f"Questions Generated: {len(results)}")
    print(f"Questions Answered: {sum(1 for r in results if 'my_answer' in r)}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)

    print("\n✓ Tournament simulation complete!")
    print("\nNext steps:")
    print("1. Review results in evaluation/tournament_results.json")
    print("2. Evaluate quality with: python evaluation/evaluate.py")
    print("3. Adjust strategy if needed")


if __name__ == "__main__":
    main()
