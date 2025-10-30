#!/usr/bin/env python3
"""
Generate distillation data with teacher reasoning from DeepSeek-R1-32B
This creates training data with reasoning chains: Question -> [Reasoning] -> Answer
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from unsloth import FastLanguageModel
import time
from datetime import datetime

def load_model():
    """Load DeepSeek-R1-32B for generating reasoning"""
    print("Loading DeepSeek-R1-Distill-Qwen-32B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b",
        max_seq_length=2048,  # Increased to handle longer reasoning
        dtype=None,
        load_in_4bit=False,  # Disable 4bit for AMD ROCm compatibility
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_reasoning(model, tokenizer, question_data):
    """Generate reasoning chain for a question using DeepSeek-R1"""
    question = question_data['question']
    choices = question_data['choices']
    correct_answer = question_data['correct_answer']

    # Format choices - choices is a dict like {"A": "text1", "B": "text2", ...}
    choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(choices.items())])

    # Prompt that encourages reasoning
    prompt = f"""<|im_start|>system
You are an expert educator. When answering questions, first explain your reasoning step-by-step inside <think> tags, then provide the final answer.<|im_end|>
<|im_start|>user
{question}

{choices_text}

Please think through this carefully and explain your reasoning before answering.<|im_end|>
<|im_start|>assistant
<think>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Reduced for speed (still allows reasoning)
            temperature=0.1,  # Low temperature for consistent, focused reasoning
            do_sample=False,  # Greedy decoding for speed
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract the assistant's response (reasoning + answer)
    if "<|im_start|>assistant" in response:
        assistant_response = response.split("<|im_start|>assistant")[-1]
        assistant_response = assistant_response.replace("<|im_end|>", "").strip()
    else:
        assistant_response = response

    # Ensure the response ends with the correct answer
    if correct_answer not in assistant_response:
        # If model didn't give correct answer, append it
        if "</think>" in assistant_response:
            assistant_response += f"\n\nThe correct answer is {correct_answer}."
        else:
            assistant_response += f"</think>\n\nThe correct answer is {correct_answer}."

    return assistant_response

def load_training_data(category, max_samples=None, difficulty_threshold=None):
    """Load training data for a specific category"""
    # Use the full 45K dataset - no need to split by category
    file_path = Path("/home/rocm-user/AMD_Hackathon/data/curriculum/train_45k.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # If difficulty threshold is set, filter for harder questions
    if difficulty_threshold:
        data = [q for q in data if q.get('difficulty_score', 0) >= difficulty_threshold]
        print(f"Filtered to {len(data)} questions with difficulty >= {difficulty_threshold}")

    # If max_samples is set, take a representative sample
    if max_samples and len(data) > max_samples:
        # Sample evenly across the curriculum
        step = len(data) // max_samples
        data = [data[i] for i in range(0, len(data), step)][:max_samples]
        print(f"Sampled {len(data)} questions evenly across curriculum")

    return data

def create_distillation_dataset(category, model, tokenizer, output_path, limit=None):
    """Create distillation dataset with reasoning for a category"""
    print(f"\n{'='*60}")
    print(f"Generating {category} distillation data")
    print(f"{'='*60}\n")

    # Load training data
    training_data = load_training_data(category, max_samples=limit if limit else 5000)

    distillation_data = []
    failed_count = 0

    start_time = time.time()

    for i, item in enumerate(tqdm(training_data, desc=f"Processing {category}")):
        try:
            # Generate reasoning
            reasoning_response = generate_reasoning(model, tokenizer, item)

            # Format choices for training example
            question = item['question']
            choices = item['choices']
            choices_text = "\n".join([f"{label}. {text}" for label, text in sorted(choices.items())])

            # Create training example with reasoning
            training_text = f"""<|im_start|>system
You are an expert educator. When answering questions, explain your reasoning step-by-step inside <think> tags, then provide the final answer.<|im_end|>
<|im_start|>user
{question}

{choices_text}

Please think through this carefully and explain your reasoning before answering.<|im_end|>
<|im_start|>assistant
{reasoning_response}<|im_end|>"""

            distillation_data.append({
                "text": training_text,
                "id": item.get('id', f"q_{i}"),
                "domain": item.get('domain', 'unknown')
            })

            # Save checkpoint every 1000 questions
            if (i + 1) % 1000 == 0:
                checkpoint_path = output_path.replace('.json', f'_checkpoint_{i+1}.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump(distillation_data, f, indent=2)

                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(training_data) - i - 1) / rate / 3600
                print(f"\n✓ Checkpoint saved: {i+1}/{len(training_data)} questions")
                print(f"  Rate: {rate:.2f} questions/sec")
                print(f"  Estimated time remaining: {remaining:.1f} hours")

        except Exception as e:
            print(f"\n✗ Error processing question {i}: {e}")
            failed_count += 1
            continue

    # Save final dataset
    with open(output_path, 'w') as f:
        json.dump(distillation_data, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ {category} distillation data complete!")
    print(f"  Total questions: {len(distillation_data)}")
    print(f"  Failed: {failed_count}")
    print(f"  Time taken: {elapsed/3600:.2f} hours")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    return distillation_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate distillation data with reasoning")
    parser.add_argument("--limit", type=int, default=5000,
                       help="Number of questions to process (default: 5000 for speed)")
    args = parser.parse_args()

    print(f"Will generate reasoning for {args.limit} questions")
    print(f"Estimated time: {args.limit * 24.5 / 3600:.1f} hours")
    print()

    # Load model once
    model, tokenizer = load_model()

    # Create output directory
    output_dir = Path("/home/rocm-user/AMD_Hackathon/data/distillation")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate distillation data
    output_file = output_dir / f"distillation_{args.limit}q.json"
    create_distillation_dataset("all", model, tokenizer, str(output_file), args.limit)

    print("\n" + "="*60)
    print("✓ ALL DISTILLATION DATA GENERATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
