#!/usr/bin/env python3
"""
Q-Agent: Question Generator for AMD Hackathon Tournament
Generates challenging MCQ questions using TIES-merged ensemble adapter
"""

import json
import logging
import sys
import torch
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class QuestionAgent:
    def __init__(self, model_path: str = "/workspace/models/ties_merged_ensemble_r128"):
        """
        Initialize Question Agent with merged ensemble adapter

        Args:
            model_path: Path to TIES-merged adapter
        """
        logging.info("Loading Q-Agent model...")

        # Load base model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Load merged adapter
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, model_path)

        # Enable inference mode
        FastLanguageModel.for_inference(self.model)

        logging.info("Q-Agent ready")

    def generate_question(self, topic: str = None, difficulty: str = "hard") -> dict:
        """
        Generate a challenging MCQ question

        Args:
            topic: Optional topic hint (ignored for now - model chooses)
            difficulty: Difficulty level hint

        Returns:
            dict: {
                "question": str,
                "choices": {"A": str, "B": str, "C": str, "D": str},
                "correct_answer": str
            }
        """
        # Prompt for question generation
        prompt = f"""<|im_start|>system
You are an expert question writer. Generate a challenging, college-level multiple choice question with 4 answer choices.
Make the question difficult but fair - it should test deep understanding, not just memorization.
Ensure exactly one answer is clearly correct.<|im_end|>
<|im_start|>user
Generate a {difficulty} difficulty multiple choice question. Format your response as:

Question: [question text]

Choices:
A. [choice A]
B. [choice B]
C. [choice C]
D. [choice D]

Correct Answer: [A/B/C/D]<|im_end|>
<|im_start|>assistant
"""

        # Generate
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,  # Higher temp for creativity
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        # Parse response
        return self._parse_question(response)

    def _parse_question(self, response: str) -> dict:
        """
        Parse model response into structured question format

        Handles various formats and edge cases with robust fallbacks
        """
        try:
            lines = [l.strip() for l in response.split('\n') if l.strip()]

            # Find question
            question = ""
            for i, line in enumerate(lines):
                if line.lower().startswith("question:"):
                    question = line.split(":", 1)[1].strip()
                    break
                elif "?" in line and len(line) > 20:
                    # Heuristic: question likely contains "?"
                    question = line
                    break

            if not question:
                question = lines[0] if lines else "What is the correct answer?"

            # Find choices
            choices = {}
            for line in lines:
                for letter in ['A', 'B', 'C', 'D']:
                    if line.startswith(f"{letter}.") or line.startswith(f"{letter})"):
                        choice_text = line.split(".", 1)[1].strip() if "." in line else line.split(")", 1)[1].strip()
                        choices[letter] = choice_text
                        break

            # Ensure all 4 choices
            for letter in ['A', 'B', 'C', 'D']:
                if letter not in choices:
                    choices[letter] = f"Option {letter}"

            # Find correct answer
            correct = "A"  # Default
            for line in lines:
                if "correct answer" in line.lower():
                    for letter in ['A', 'B', 'C', 'D']:
                        if letter in line.upper():
                            correct = letter
                            break
                    break

            return {
                "question": question,
                "choices": choices,
                "correct_answer": correct
            }

        except Exception as e:
            logging.error(f"Parsing error: {e}")
            # Fallback question
            return {
                "question": "What is the primary function of mitochondria in eukaryotic cells?",
                "choices": {
                    "A": "Protein synthesis",
                    "B": "ATP production through cellular respiration",
                    "C": "DNA replication",
                    "D": "Photosynthesis"
                },
                "correct_answer": "B"
            }

def main():
    """
    Main entry point for Q-Agent
    Reads from stdin, writes to stdout (tournament format)
    """
    try:
        # Initialize agent
        agent = QuestionAgent()

        # Read request from stdin (tournament system)
        request = sys.stdin.read().strip()
        logging.info(f"Request: {request}")

        # Generate question
        question_data = agent.generate_question()

        # Output as JSON to stdout
        output = json.dumps(question_data, indent=2)
        print(output)

        logging.info("Question generated successfully")

    except Exception as e:
        logging.error(f"Q-Agent error: {e}")
        # Output fallback question
        fallback = {
            "question": "What is the time complexity of binary search on a sorted array of n elements?",
            "choices": {
                "A": "O(n)",
                "B": "O(log n)",
                "C": "O(n log n)",
                "D": "O(1)"
            },
            "correct_answer": "B"
        }
        print(json.dumps(fallback, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
