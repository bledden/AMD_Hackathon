#!/usr/bin/env python3
"""
Tournament Answer Agent - OPTIMIZED FOR SPEED
Uses DeepSeek-R1-Distill-Qwen-32B with aggressive optimizations for <6s requirement
"""

import json
import logging
import time
import torch
from pathlib import Path
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global model (loaded once at module import)
model = None
tokenizer = None
MODEL_LOADED = False

def load_model_once():
    """Load model once at startup with aggressive speed optimizations"""
    global model, tokenizer, MODEL_LOADED

    if MODEL_LOADED:
        return

    logging.info("Loading DeepSeek-R1-Distill-Qwen-32B with speed optimizations...")
    start = time.time()

    # Load with 4-bit quantization for speed
    model_obj, tokenizer_obj = FastLanguageModel.from_pretrained(
        model_name="/workspace/models/deepseek_r1_qwen32b",
        max_seq_length=512,  # Short context for MCQs
        dtype=None,
        load_in_4bit=True,  # Critical for speed
    )

    # Enable inference mode with all optimizations
    FastLanguageModel.for_inference(model_obj)

    model = model_obj
    tokenizer = tokenizer_obj
    MODEL_LOADED = True

    elapsed = time.time() - start
    logging.info(f"✅ Model loaded in {elapsed:.1f}s")

def answer_question(question_data):
    """
    Answer MCQ with aggressive speed optimizations

    Args:
        question_data: {"question": str, "choices": {"A": str, "B": str, "C": str, "D": str}}

    Returns:
        str: Answer letter (A/B/C/D)
    """
    if not MODEL_LOADED:
        load_model_once()

    question = question_data['question']
    choices = question_data['choices']

    # Ultra-concise prompt - NO extra reasoning, just the answer
    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

    prompt = f"""<|im_start|>system
Answer the multiple choice question. Output only the letter (A, B, C, or D).<|im_end|>
<|im_start|>user
{question}

{choices_text}<|im_end|>
<|im_start|>assistant
The answer is """

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,  # CRITICAL: Only need 1-2 tokens for answer letter!
            temperature=0.1,  # Low temp for accuracy
            do_sample=False,  # Greedy = fastest
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()

    # Extract answer
    answer = extract_answer(response)
    return answer

def extract_answer(response):
    """Extract answer letter - robust extraction"""
    response_upper = response.upper().strip()

    # Direct letter match (most common)
    if response_upper and response_upper[0] in ['A', 'B', 'C', 'D']:
        return response_upper[0]

    # Scan for any valid letter
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_upper:
            return letter

    # Fallback
    logging.warning(f"Could not extract from: {response}")
    return "A"

def main():
    """
    Main entry point - Tournament compliant
    Called by: python -m agents.answer_agent
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.info("🚀 A-Agent starting (DeepSeek-R1-32B optimized)...")

    # Load model
    load_model_once()

    # Read questions
    questions = []

    question_file = Path("/workspace/AIAC/outputs/questions.json")
    if question_file.exists():
        with open(question_file) as f:
            questions_data = json.load(f)
            questions = questions_data if isinstance(questions_data, list) else [questions_data]

    if not questions:
        try:
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                questions_data = json.loads(stdin_data)
                questions = questions_data if isinstance(questions_data, list) else [questions_data]
        except:
            pass

    if not questions:
        logging.error("❌ No questions found!")
        return

    logging.info(f"Processing {len(questions)} question(s)...")

    # Answer each question
    answers = []
    total_inference_time = 0

    for i, q in enumerate(questions, 1):
        start = time.time()
        answer = answer_question(q)
        elapsed = time.time() - start
        total_inference_time += elapsed

        answers.append({
            "question_id": q.get('id', i),
            "answer": answer,
            "answer_time": elapsed
        })

        logging.info(f"   Q{i}: {answer} ({elapsed:.2f}s)")

    # Save outputs
    output_dir = Path("/workspace/AIAC/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "answers.json"
    with open(output_file, 'w') as f:
        json.dump(answers, f, indent=2)

    # Stats
    avg_time = total_inference_time / len(answers)
    max_time = max(a['answer_time'] for a in answers)

    logging.info(f"✅ {len(answers)} answer(s) saved to {output_file}")
    logging.info(f"   Average inference time: {avg_time:.2f}s")
    logging.info(f"   Max inference time: {max_time:.2f}s")

    if max_time > 6.0:
        logging.warning(f"⚠️  WARNING: {max_time:.2f}s exceeds 6s limit!")
    else:
        logging.info(f"✅ All answers within 6s limit!")

if __name__ == "__main__":
    main()
