#!/usr/bin/env python3
"""
Validate fine-tuned model on holdout set
Decide if Stage 2 CoT is needed based on results
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_validation_data(path: Path) -> List[Dict]:
    """Load validation dataset"""
    logging.info(f"📂 Loading validation data from {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    logging.info(f"✅ Loaded {len(data)} validation questions")
    return data


def load_finetuned_model(model_path: str, base_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"):
    """Load fine-tuned LoRA model"""
    logging.info(f"🚀 Loading fine-tuned model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load base model with 4-bit quantization
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base, model_path)

    logging.info(f"✅ Model loaded successfully")
    return model, tokenizer


def format_question_for_inference(question: Dict) -> str:
    """Format question as prompt"""
    q_text = question['question']
    choices = question.get('choices', {})

    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

    prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
Question: {q_text}

Choices:
{choices_text}

Please select the correct answer (A, B, C, or D).<|im_end|>
<|im_start|>assistant
The correct answer is"""

    return prompt


def extract_answer(response: str) -> str:
    """Extract answer letter from model response"""
    # Look for A, B, C, or D
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)
    return "A"  # Default fallback


def evaluate_model(model, tokenizer, val_data: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    Evaluate model on validation set

    Returns:
        (accuracy, detailed_results)
    """
    logging.info("🧪 Evaluating model on validation set...")

    correct = 0
    results = []

    for i, question in enumerate(val_data):
        # Format prompt
        prompt = format_question_for_inference(question)

        # Generate answer
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = extract_answer(response.split("assistant")[-1])
        actual = question.get('correct_answer', 'A')

        is_correct = (predicted == actual)
        if is_correct:
            correct += 1

        results.append({
            'question_id': i,
            'difficulty': question.get('difficulty_score', 5.0),
            'predicted': predicted,
            'actual': actual,
            'correct': is_correct
        })

        if (i + 1) % 100 == 0:
            logging.info(f"   Progress: {i+1}/{len(val_data)} ({correct/(i+1)*100:.1f}% so far)")

    accuracy = correct / len(val_data) * 100
    return accuracy, results


def analyze_results(accuracy: float, results: List[Dict]):
    """
    Analyze validation results and recommend next steps

    Args:
        accuracy: Overall accuracy percentage
        results: Detailed results per question
    """
    logging.info("\n" + "=" * 80)
    logging.info("📊 VALIDATION RESULTS")
    logging.info("=" * 80)

    logging.info(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")

    # Breakdown by difficulty
    easy = [r for r in results if r['difficulty'] < 4]
    medium = [r for r in results if 4 <= r['difficulty'] < 7]
    hard = [r for r in results if r['difficulty'] >= 7]

    easy_acc = sum(1 for r in easy if r['correct']) / len(easy) * 100 if easy else 0
    medium_acc = sum(1 for r in medium if r['correct']) / len(medium) * 100 if medium else 0
    hard_acc = sum(1 for r in hard if r['correct']) / len(hard) * 100 if hard else 0

    logging.info(f"\n📈 Accuracy by Difficulty:")
    logging.info(f"   Easy (1-4):   {easy_acc:.1f}% ({len(easy)} questions)")
    logging.info(f"   Medium (4-7): {medium_acc:.1f}% ({len(medium)} questions)")
    logging.info(f"   Hard (7-10):  {hard_acc:.1f}% ({len(hard)} questions)")

    # Decision tree
    logging.info("\n" + "=" * 80)
    logging.info("🎯 STAGE 2 RECOMMENDATION")
    logging.info("=" * 80)

    if accuracy >= 86.0:
        logging.info("\n✅ EXCELLENT RESULT!")
        logging.info(f"   Accuracy {accuracy:.1f}% meets/exceeds target (86%)")
        logging.info("   ✅ RECOMMENDATION: Deploy immediately, NO CoT needed!")
        logging.info("   🏆 Ready for tournament!")

    elif 85.0 <= accuracy < 86.0:
        logging.info("\n⚠️  GOOD RESULT (slight gap from target)")
        logging.info(f"   Accuracy {accuracy:.1f}% is close to target (86%)")
        logging.info("   💡 RECOMMENDATION: Light CoT on weak areas (2-4 hours)")
        logging.info(f"      Target: Hard questions ({hard_acc:.1f}% → 90%+)")

    elif 84.0 <= accuracy < 85.0:
        logging.info("\n⚠️  MODERATE GAP")
        logging.info(f"   Accuracy {accuracy:.1f}% is below target (86%)")
        logging.info("   💡 RECOMMENDATION: Self-CoT on 3-5K medium/hard questions (12-15 hours)")
        logging.info(f"      This could boost to ~86-87%")

    else:
        logging.info("\n🚨 SIGNIFICANT GAP")
        logging.info(f"   Accuracy {accuracy:.1f}% is significantly below target")
        logging.info("   💡 RECOMMENDATION: Full CoT enhancement needed")
        logging.info(f"      Consider 7K+ CoT questions if time permits (20-25 hours)")

    logging.info("=" * 80)


def main():
    """Main validation pipeline"""
    logging.info("=" * 80)
    logging.info("🧪 MODEL VALIDATION - STAGE 1 PERFORMANCE TEST")
    logging.info("=" * 80)

    # Paths
    val_path = Path('data/curriculum/val_5k.json')
    model_path = 'models/qwen3_235b_lora_curriculum/final_model'

    # Load validation data
    val_data = load_validation_data(val_path)

    # Load model
    model, tokenizer = load_finetuned_model(model_path)

    # Evaluate
    accuracy, results = evaluate_model(model, tokenizer, val_data)

    # Analyze and recommend
    analyze_results(accuracy, results)

    # Save detailed results
    results_path = Path('results/validation_stage1.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'detailed_results': results
        }, f, indent=2)

    logging.info(f"\n💾 Detailed results saved: {results_path}")


if __name__ == "__main__":
    main()
