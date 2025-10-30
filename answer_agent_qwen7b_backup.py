#!/usr/bin/env python3
"""
BACKUP A-Agent: Uses Qwen2.5-7B-Instruct (smaller, faster)
Deploy if DeepSeek-R1-32B fails speed requirements
"""

import json
import logging
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class AnswerAgent:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize Answer Agent with Qwen2.5-7B-Instruct

        Args:
            model_path: HuggingFace model ID or local path
        """
        logging.info("Loading A-Agent model (Qwen2.5-7B-Instruct)...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logging.info("A-Agent ready (Qwen2.5-7B baseline)")

    def answer_question(self, question_data: dict) -> str:
        """
        Answer a multiple choice question

        Args:
            question_data: {
                "question": str,
                "choices": {"A": str, "B": str, "C": str, "D": str}
            }

        Returns:
            str: Single letter answer (A, B, C, or D)
        """
        question = question_data["question"]
        choices = question_data["choices"]

        # Format prompt for Qwen
        choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions accurately."},
            {"role": "user", "content": f"{question}\n\n{choices_text}\n\nSelect the correct answer (A, B, C, or D):"}
        ]

        # Format with chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and generate
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=512).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=4,  # Only need 1 letter
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode response
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Extract answer letter
        answer = self._extract_answer(response)
        return answer

    def _extract_answer(self, response: str) -> str:
        """
        Extract answer letter (A, B, C, or D) from model response
        """
        response_upper = response.upper().strip()

        # Strategy 1: First letter if it's A, B, C, or D
        if response_upper and response_upper[0] in ['A', 'B', 'C', 'D']:
            return response_upper[0]

        # Strategy 2: Find first occurrence of valid letter
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                return letter

        # Strategy 3: Default fallback
        logging.warning(f"Could not extract answer from: {response}")
        return "A"

def main():
    """
    Main entry point for A-Agent
    """
    try:
        # Initialize agent
        agent = AnswerAgent()

        # Read question from stdin (tournament format)
        question_json = sys.stdin.read().strip()
        question_data = json.loads(question_json)

        logging.info(f"Question: {question_data['question'][:80]}...")

        # Answer question
        answer = agent.answer_question(question_data)

        # Output answer as JSON to stdout
        output = json.dumps({"answer": answer})
        print(output)

        logging.info(f"Answer: {answer}")

    except Exception as e:
        logging.error(f"A-Agent error: {e}")
        # Output fallback answer
        fallback = {"answer": "A"}
        print(json.dumps(fallback))
        sys.exit(1)

if __name__ == "__main__":
    main()
