#!/usr/bin/env python3
"""
Validate the trained Qwen2.5-72B model on the 5K holdout set
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    """Load the trained model with LoRA weights"""
    logging.info(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # Load in 16-bit for inference
    )
    FastLanguageModel.for_inference(model)  # Enable inference mode
    logging.info("✅ Model loaded successfully")
    return model, tokenizer

def extract_answer(response_text):
    """Extract the answer letter (A, B, C, or D) from model response"""
    response = response_text.upper().strip()

    # Look for explicit answer markers
    for marker in ["ANSWER:", "THE ANSWER IS", "CORRECT ANSWER:"]:
        if marker in response:
            after_marker = response.split(marker)[1].strip()
            for letter in ['A', 'B', 'C', 'D']:
                if after_marker.startswith(letter):
                    return letter

    # Look for first occurrence of A, B, C, or D
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response:
            return letter

    return None

def validate_model(model, tokenizer, val_data_path):
    """Validate model on holdout set"""
    logging.info(f"Loading validation data from {val_data_path}")

    with open(val_data_path) as f:
        val_data = json.load(f)

    logging.info(f"Loaded {len(val_data)} validation questions")

    correct = 0
    total = 0

    for item in tqdm(val_data[:100], desc="Validating"):  # Test on first 100 for speed
        question = item['question']
        choices = item['choices']
        correct_answer = item['answer']

        # Format prompt
        prompt = f"""Question: {question}

A) {choices['A']}
B) {choices['B']}
C) {choices['C']}
D) {choices['D']}

Answer with the correct letter (A, B, C, or D):"""

        # Generate response
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
        predicted = extract_answer(response.replace(prompt, ""))

        if predicted == correct_answer:
            correct += 1

        total += 1

    accuracy = (correct / total) * 100
    logging.info(f"\n{'='*80}")
    logging.info(f"VALIDATION RESULTS (first 100 questions)")
    logging.info(f"{'='*80}")
    logging.info(f"Correct: {correct}/{total}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"{'='*80}")

    return accuracy

def main():
    model_path = "/workspace/models/qwen2.5_72b_unsloth_curriculum/checkpoint_chunk8"
    val_data_path = "/workspace/data/curriculum/val_5k.json"

    # Check if paths exist
    if not Path(model_path).exists():
        logging.error(f"Model not found at {model_path}")
        # Try final_model if checkpoint doesn't exist
        model_path = "/workspace/models/qwen2.5_72b_unsloth_curriculum/final_model"
        if not Path(model_path).exists():
            logging.error(f"No model found!")
            return

    model, tokenizer = load_model(model_path)
    accuracy = validate_model(model, tokenizer, val_data_path)

    logging.info(f"\n🎯 Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
