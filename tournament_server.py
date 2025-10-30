#!/usr/bin/env python3
"""
Tournament Server: Keeps models loaded in memory for fast Q&A
Provides HTTP endpoints for question generation and answering
"""

import json
import logging
import time
import torch
from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

app = Flask(__name__)

# Global model instances (loaded once at startup)
model = None
tokenizer = None

def load_models():
    """Load base model + merged adapter once at startup"""
    global model, tokenizer

    logging.info("=" * 80)
    logging.info("🚀 LOADING TOURNAMENT MODELS")
    logging.info("=" * 80)

    start = time.time()

    # Load base model
    logging.info("📦 Loading Qwen2.5-72B-Instruct base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Load merged adapter
    logging.info("🔀 Loading TIES-merged ensemble adapter...")
    model = PeftModel.from_pretrained(
        model,
        "/workspace/models/ties_merged_ensemble_r128"
    )

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    elapsed = time.time() - start
    logging.info(f"✅ Models loaded in {elapsed:.1f}s")
    logging.info("=" * 80)

@app.route('/generate_question', methods=['POST'])
def generate_question():
    """
    Generate a challenging MCQ question

    POST /generate_question
    Body: {} (empty or with optional params)

    Returns: {
        "question": str,
        "choices": {"A": str, "B": str, "C": str, "D": str},
        "correct_answer": str,
        "generation_time": float
    }
    """
    start = time.time()

    try:
        # Prompt for question generation
        prompt = """<|im_start|>system
You are an expert question writer. Generate a challenging, college-level multiple choice question with 4 answer choices.
Make the question difficult but fair - it should test deep understanding, not just memorization.
Ensure exactly one answer is clearly correct.<|im_end|>
<|im_start|>user
Generate a hard difficulty multiple choice question. Format your response as:

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
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Reduced from 512 for speed
                temperature=0.7,  # Lower temperature for faster convergence
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,  # Greedy-ish for speed
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        # Parse response
        question_data = parse_question(response)

        elapsed = time.time() - start
        question_data['generation_time'] = elapsed

        logging.info(f"✅ Question generated in {elapsed:.2f}s")

        return jsonify(question_data), 200

    except Exception as e:
        logging.error(f"❌ Error generating question: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/answer_question', methods=['POST'])
def answer_question():
    """
    Answer a multiple choice question

    POST /answer_question
    Body: {
        "question": str,
        "choices": {"A": str, "B": str, "C": str, "D": str}
    }

    Returns: {
        "answer": str,
        "answer_time": float
    }
    """
    start = time.time()

    try:
        data = request.get_json()
        question = data['question']
        choices = data['choices']

        # Format prompt
        choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

        prompt = f"""<|im_start|>system
You are a highly knowledgeable assistant that answers multiple choice questions accurately.
Think carefully and select the best answer.<|im_end|>
<|im_start|>user
Question: {question}

Choices:
{choices_text}

Select the correct answer (A, B, C, or D):<|im_end|>
<|im_start|>assistant
The correct answer is """

        # Generate with low temperature for accuracy
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        # Extract answer letter
        answer = extract_answer(response)

        elapsed = time.time() - start

        logging.info(f"✅ Question answered in {elapsed:.2f}s: {answer}")

        return jsonify({
            "answer": answer,
            "answer_time": elapsed
        }), 200

    except Exception as e:
        logging.error(f"❌ Error answering question: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ready", "model_loaded": model is not None}), 200

def parse_question(response: str) -> dict:
    """Parse model response into structured question format"""
    try:
        lines = [l.strip() for l in response.split('\n') if l.strip()]

        # Find question
        question = ""
        for i, line in enumerate(lines):
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
                break
            elif "?" in line and len(line) > 20:
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
        correct = "A"
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

def extract_answer(response: str) -> str:
    """Extract answer letter (A, B, C, or D) from model response"""
    response_upper = response.upper().strip()

    # Strategy 1: First letter if it's A, B, C, or D
    if response_upper and response_upper[0] in ['A', 'B', 'C', 'D']:
        return response_upper[0]

    # Strategy 2: Look for explicit patterns
    for pattern in ["ANSWER IS", "CORRECT ANSWER IS", "SELECT", "CHOOSE"]:
        if pattern in response_upper:
            after_pattern = response_upper.split(pattern)[1].strip()
            for letter in ['A', 'B', 'C', 'D']:
                if letter in after_pattern[:10]:
                    return letter

    # Strategy 3: Find first occurrence of valid letter
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_upper:
            return letter

    # Default fallback
    logging.warning(f"Could not extract answer from: {response}")
    return "A"

if __name__ == '__main__':
    # Load models once at startup
    load_models()

    # Start Flask server
    logging.info("🌐 Starting tournament server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
