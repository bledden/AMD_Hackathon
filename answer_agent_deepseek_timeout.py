#!/usr/bin/env python3
"""
A-Agent: DeepSeek-R1-32B with timeout protection
Kills generation after 5.5s to guarantee <6s compliance
"""

import json
import logging
import sys
import torch
import threading
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class GenerationTimeout:
    """Context manager for generation with timeout"""
    def __init__(self, seconds):
        self.seconds = seconds
        self.result = None
        self.error = None

    def run_generation(self, model, inputs, gen_kwargs):
        try:
            self.result = model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            self.error = e

    def generate_with_timeout(self, model, inputs, **gen_kwargs):
        thread = threading.Thread(
            target=self.run_generation,
            args=(model, inputs, gen_kwargs)
        )
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.seconds)

        if thread.is_alive():
            logging.warning(f"Generation timeout after {self.seconds}s - using fallback")
            return None

        if self.error:
            raise self.error

        return self.result

class AnswerAgent:
    def __init__(self, model_path: str = "/workspace/models/deepseek_r1_qwen32b"):
        """Initialize Answer Agent with DeepSeek-R1-32B"""
        logging.info("Loading A-Agent model (DeepSeek-R1-32B with timeout protection)...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(self.model)
        logging.info("A-Agent ready")

    def answer_question(self, question_data: dict) -> str:
        """Answer a multiple choice question with timeout protection"""
        question = question_data["question"]
        choices = question_data["choices"]

        choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

        prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
{question}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

        inputs = self.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512).to("cuda")

        # Generate with 5.5 second timeout
        timeout_handler = GenerationTimeout(seconds=5.5)
        outputs = timeout_handler.generate_with_timeout(
            self.model,
            inputs,
            max_new_tokens=4,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if outputs is None:
            # Timeout occurred - use heuristic fallback
            logging.warning("Using fallback answer due to timeout")
            # Simple heuristic: return most common answer or 'B'
            return "B"

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        answer = self._extract_answer(response)
        return answer

    def _extract_answer(self, response: str) -> str:
        """Extract answer letter from response"""
        response_upper = response.upper().strip()

        if response_upper and response_upper[0] in ['A', 'B', 'C', 'D']:
            return response_upper[0]

        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                return letter

        logging.warning(f"Could not extract answer from: {response}")
        return "B"

def main():
    """Main entry point"""
    try:
        agent = AnswerAgent()
        question_json = sys.stdin.read().strip()
        question_data = json.loads(question_json)

        logging.info(f"Question: {question_data['question'][:80]}...")
        answer = agent.answer_question(question_data)

        output = json.dumps({"answer": answer})
        print(output)
        logging.info(f"Answer: {answer}")

    except Exception as e:
        logging.error(f"A-Agent error: {e}")
        fallback = {"answer": "B"}
        print(json.dumps(fallback))
        sys.exit(1)

if __name__ == "__main__":
    main()
