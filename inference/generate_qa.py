#!/usr/bin/env python3
"""
Q&A Generation and Answering Script
Generate questions and answer them using fine-tuned model
"""

import sys
from pathlib import Path
import argparse
from typing import Optional, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from unsloth import FastLanguageModel


class QAAgent:
    """Q&A Agent for tournament"""

    def __init__(self, model_path: str, max_seq_length: int = 2048):
        """Initialize agent with fine-tuned model"""
        print(f"Loading model from {model_path}...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # Enable inference mode
        FastLanguageModel.for_inference(self.model)

        print("✓ Model loaded and ready for inference")

    def generate_question(
        self,
        topic: str,
        difficulty: str = "challenging",
        temperature: float = 0.7,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate a question on a given topic"""
        prompt = f"""### Instruction:
Generate a {difficulty} question about {topic}.

### Response:
"""

        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if "### Response:" in generated:
            question = generated.split("### Response:")[-1].strip()
        else:
            question = generated.strip()

        return question

    def answer_question(
        self,
        question: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
    ) -> str:
        """Answer a given question"""
        prompt = f"""### Instruction:
Answer this question accurately and concisely: {question}

### Response:
"""

        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True if temperature > 0 else False,
            repetition_penalty=1.1,
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if "### Response:" in generated:
            answer = generated.split("### Response:")[-1].strip()
        else:
            answer = generated.strip()

        return answer

    def tournament_battle(
        self,
        opponent_question: Optional[str] = None,
        my_topic: str = "science",
    ) -> dict:
        """Execute one round of tournament battle"""
        result = {}

        # Generate my question
        print("\n" + "=" * 60)
        print("Generating my question...")
        my_question = self.generate_question(my_topic)
        result["my_question"] = my_question
        print(f"My Question: {my_question}")

        # Answer opponent's question
        if opponent_question:
            print("\n" + "=" * 60)
            print("Answering opponent's question...")
            my_answer = self.answer_question(opponent_question)
            result["my_answer"] = my_answer
            print(f"Opponent Question: {opponent_question}")
            print(f"My Answer: {my_answer}")

        print("=" * 60)
        return result


def batch_generate_questions(
    agent: QAAgent,
    topics: List[str],
    n_per_topic: int = 3,
    difficulty: str = "challenging",
) -> List[dict]:
    """Generate multiple questions for testing"""
    questions = []

    for topic in topics:
        print(f"\nGenerating {n_per_topic} questions on: {topic}")
        for i in range(n_per_topic):
            question = agent.generate_question(topic, difficulty)
            questions.append({
                "topic": topic,
                "difficulty": difficulty,
                "question": question,
                "number": i + 1,
            })
            print(f"  {i+1}. {question}")

    return questions


def batch_answer_questions(
    agent: QAAgent,
    questions: List[str],
) -> List[dict]:
    """Answer multiple questions for testing"""
    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Q: {question}")
        answer = agent.answer_question(question)
        print(f"   A: {answer}")

        results.append({
            "question": question,
            "answer": answer,
        })

    return results


def interactive_mode(agent: QAAgent):
    """Interactive Q&A mode"""
    print("\n" + "=" * 60)
    print("Interactive Q&A Mode")
    print("=" * 60)
    print("Commands:")
    print("  /q [topic] - Generate a question")
    print("  /a [question] - Answer a question")
    print("  /quit - Exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                break

            if user_input.startswith("/q "):
                topic = user_input[3:].strip()
                if not topic:
                    print("Usage: /q [topic]")
                    continue
                question = agent.generate_question(topic)
                print(f"\nGenerated Question: {question}")

            elif user_input.startswith("/a "):
                question = user_input[3:].strip()
                if not question:
                    print("Usage: /a [question]")
                    continue
                answer = agent.answer_question(question)
                print(f"\nAnswer: {answer}")

            else:
                print("Unknown command. Use /q, /a, or /quit")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="Q&A Agent Inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="training/outputs/final_model",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "generate", "answer", "tournament"],
        help="Inference mode",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="science",
        help="Topic for question generation",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to answer",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="challenging",
        help="Question difficulty",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=3,
        help="Number of questions to generate",
    )

    args = parser.parse_args()

    # Load agent
    agent = QAAgent(args.model_path)

    # Run mode
    if args.mode == "interactive":
        interactive_mode(agent)

    elif args.mode == "generate":
        print("\n" + "=" * 60)
        print(f"Generating {args.n_questions} questions on: {args.topic}")
        print("=" * 60)

        for i in range(args.n_questions):
            question = agent.generate_question(args.topic, args.difficulty)
            print(f"\n{i+1}. {question}")

    elif args.mode == "answer":
        if not args.question:
            print("Error: --question required for answer mode")
            return

        print("\n" + "=" * 60)
        print(f"Question: {args.question}")
        print("=" * 60)

        answer = agent.answer_question(args.question)
        print(f"\nAnswer: {answer}")

    elif args.mode == "tournament":
        print("\n" + "=" * 60)
        print("Tournament Mode - Simulating Battle Round")
        print("=" * 60)

        # Simulate opponent question
        opponent_question = "What is the Heisenberg Uncertainty Principle and why is it important?"

        result = agent.tournament_battle(
            opponent_question=opponent_question,
            my_topic=args.topic,
        )

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
