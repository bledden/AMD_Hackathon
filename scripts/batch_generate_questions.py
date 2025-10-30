#!/usr/bin/env python3
"""
Batch Generate Tournament Questions
Pre-generate 10,000+ MCQ questions offline for <10s question selection
"""

import json
import logging
import time
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Topics from tournament specification
TOPICS = [
    "Science", "Math", "History", "Literature", "Geography",
    "Technology", "Medicine", "Philosophy", "Arts", "Engineering",
    "Computer Science", "Physics", "Chemistry", "Biology",
    "Economics", "Psychology", "Sociology", "Political Science"
]

DIFFICULTIES = ["easy", "medium", "hard", "expert"]

def load_model():
    """Load model once for batch generation"""
    logging.info("Loading Qwen2.5-72B + TIES-merged adapter...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = PeftModel.from_pretrained(
        model,
        "/workspace/models/ties_merged_ensemble_r128"
    )

    FastLanguageModel.for_inference(model)

    logging.info("Model loaded successfully")
    return model, tokenizer

def generate_question(model, tokenizer, topic=None, difficulty="hard"):
    """Generate a single MCQ question"""

    topic_hint = f" about {topic}" if topic else ""

    prompt = f"""<|im_start|>system
You are an expert question writer for academic competitions. Generate a challenging, college-level multiple choice question with 4 answer choices.
Make the question difficult but fair - it should test deep understanding, not just memorization.
Ensure exactly one answer is clearly correct.<|im_end|>
<|im_start|>user
Generate a {difficulty} difficulty multiple choice question{topic_hint}. Format your response EXACTLY as:

Question: [clear, specific question text]

Choices:
A. [first choice]
B. [second choice]
C. [third choice]
D. [fourth choice]

Correct Answer: [A/B/C/D]

Be concise but clear.<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()

    return parse_question(response, topic, difficulty)

def parse_question(response, topic, difficulty):
    """Parse model response into structured format"""
    try:
        lines = [l.strip() for l in response.split('\n') if l.strip()]

        # Extract question
        question = ""
        for line in lines:
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
                break
            elif "?" in line and len(line) > 20:
                question = line
                break

        if not question:
            question = lines[0] if lines else "What is the correct answer?"

        # Extract choices
        choices = {}
        for line in lines:
            for letter in ['A', 'B', 'C', 'D']:
                if line.startswith(f"{letter}.") or line.startswith(f"{letter})"):
                    sep = "." if "." in line else ")"
                    choice_text = line.split(sep, 1)[1].strip()
                    choices[letter] = choice_text
                    break

        # Ensure all 4 choices exist
        default_choices = {
            "A": "First option",
            "B": "Second option",
            "C": "Third option",
            "D": "Fourth option"
        }
        for letter in ['A', 'B', 'C', 'D']:
            if letter not in choices:
                choices[letter] = default_choices[letter]

        # Extract correct answer
        correct = "A"
        for line in lines:
            if "correct answer" in line.lower():
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in line.upper():
                        correct = letter
                        break
                break

        return {
            "id": None,  # Will be assigned later
            "question": question,
            "choices": choices,
            "correct_answer": correct,
            "topic": topic,
            "difficulty": difficulty,
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "Qwen2.5-72B + TIES-merged ensemble"
            }
        }

    except Exception as e:
        logging.error(f"Parse error: {e}")
        return None

def generate_batch(model, tokenizer, target_count=10000):
    """Generate large batch of questions"""

    output_file = Path("/workspace/question_pool.json")
    checkpoint_file = Path("/workspace/question_pool_checkpoint.json")

    # Load existing progress if any
    existing_questions = []
    if checkpoint_file.exists():
        logging.info("Loading existing checkpoint...")
        with open(checkpoint_file) as f:
            existing_questions = json.load(f)
        logging.info(f"Resuming from {len(existing_questions)} questions")

    questions = existing_questions
    start_idx = len(questions)

    logging.info(f"Generating {target_count - start_idx} more questions...")
    logging.info(f"Target: {target_count} total questions")

    # Generate questions with progress bar
    pbar = tqdm(total=target_count - start_idx, desc="Generating questions")

    for i in range(start_idx, target_count):
        # Randomize topic and difficulty
        topic = random.choice(TOPICS)
        difficulty = random.choice(DIFFICULTIES)

        try:
            question = generate_question(model, tokenizer, topic, difficulty)

            if question:
                question["id"] = i + 1
                questions.append(question)
                pbar.update(1)

                # Save checkpoint every 100 questions
                if (i + 1) % 100 == 0:
                    with open(checkpoint_file, 'w') as f:
                        json.dump(questions, f, indent=2)
                    logging.info(f"Checkpoint saved: {len(questions)} questions")

                # Brief pause to avoid overheating
                if (i + 1) % 50 == 0:
                    time.sleep(2)

        except Exception as e:
            logging.error(f"Error generating question {i+1}: {e}")
            continue

    pbar.close()

    # Save final output
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=2)

    logging.info(f"✅ Generated {len(questions)} questions")
    logging.info(f"   Saved to: {output_file}")

    # Statistics
    topics_count = {}
    difficulty_count = {}
    for q in questions:
        topics_count[q.get('topic', 'unknown')] = topics_count.get(q.get('topic', 'unknown'), 0) + 1
        difficulty_count[q.get('difficulty', 'unknown')] = difficulty_count.get(q.get('difficulty', 'unknown'), 0) + 1

    logging.info("\n📊 Statistics:")
    logging.info(f"   Topics: {topics_count}")
    logging.info(f"   Difficulty: {difficulty_count}")

    return questions

def main():
    logging.info("=" * 80)
    logging.info("🎯 BATCH QUESTION GENERATION")
    logging.info("=" * 80)

    # Load model
    model, tokenizer = load_model()

    # Generate 10,000 questions
    questions = generate_batch(model, tokenizer, target_count=10000)

    logging.info("=" * 80)
    logging.info("✅ BATCH GENERATION COMPLETE")
    logging.info(f"   Total questions: {len(questions)}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()
