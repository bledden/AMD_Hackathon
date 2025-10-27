#!/usr/bin/env python3
"""
Generate Chain-of-Thought reasoning using DeepSeek-V3.1-Terminus
This provides high-quality reasoning for training Qwen3-235B
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

class DeepSeekCoTGenerator:
    def __init__(self):
        """Initialize DeepSeek-V3.1-Terminus for CoT generation"""

        model_name = "unsloth/DeepSeek-V3-0324-2.7bit"  # Quantized version that fits

        print("="*70)
        print("LOADING DeepSeek-V3 FOR COT GENERATION")
        print("="*70)
        print(f"\nModel: {model_name}")
        print("Quantization: 2.7-bit (fits in 192GB)")
        print("Purpose: Generate high-quality reasoning chains")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("\nLoading model (this may take 5-10 minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        print("✅ DeepSeek-V3 loaded successfully!")

    def generate_cot(self, question: str, choices: dict, correct_answer: str) -> dict:
        """Generate Chain-of-Thought reasoning for a question"""

        # Format choices
        choices_text = "\\n".join([f"{k}. {v}" for k, v in choices.items()])

        prompt = f"""You are an expert at solving multiple choice questions. Provide step-by-step reasoning to reach the correct answer.

Question: {question}

Choices:
{choices_text}

Think through this step-by-step:
1. What is the question asking?
2. Analyze each option carefully
3. Eliminate incorrect answers with clear reasoning
4. Explain why the correct answer is right

Your detailed reasoning:"""

        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )

        reasoning = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return {
            'reasoning': reasoning.strip(),
            'correct_answer': correct_answer,
            'model': 'DeepSeek-V3.1-Terminus'
        }

def generate_cot_for_dataset(questions: list, output_path: str, batch_size: int = 1):
    """Generate CoT for entire dataset"""

    print("\\n" + "="*70)
    print("GENERATING CHAIN-OF-THOUGHT REASONING")
    print("="*70)

    generator = DeepSeekCoTGenerator()

    enhanced_questions = []
    errors = 0

    print(f"\\nProcessing {len(questions)} questions...")
    print("Estimated time: 6-8 hours\\n")

    start_time = time.time()

    for i, q in enumerate(tqdm(questions, desc="Generating CoT")):
        try:
            # Get choices
            choices = q.get('choices', q.get('options', {}))

            if not choices or not isinstance(choices, dict):
                enhanced_questions.append(q)
                continue

            if len(choices) < 2:
                enhanced_questions.append(q)
                continue

            # Generate CoT
            cot_result = generator.generate_cot(
                q['question'],
                choices,
                q['correct_answer']
            )

            # Add CoT to question
            enhanced_q = q.copy()
            enhanced_q['chain_of_thought'] = cot_result['reasoning']
            enhanced_q['cot_model'] = cot_result['model']
            enhanced_q['has_cot'] = True

            enhanced_questions.append(enhanced_q)

            # Save checkpoint every 500 questions
            if (i + 1) % 500 == 0:
                checkpoint_path = Path(output_path).parent / f"checkpoint_{i+1}.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(enhanced_questions, f, indent=2)

                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(questions) - i - 1) / rate

                print(f"\\n  Checkpoint: {i+1}/{len(questions)}")
                print(f"  Rate: {rate:.2f} questions/sec")
                print(f"  Estimated remaining: {remaining/3600:.1f} hours")

        except Exception as e:
            print(f"\\nError on question {i}: {str(e)[:100]}")
            errors += 1
            enhanced_questions.append(q)

            if errors > 100:
                print("\\n⚠️ Too many errors, stopping...")
                break

    # Final save
    with open(output_path, 'w') as f:
        json.dump(enhanced_questions, f, indent=2)

    success_count = sum(1 for q in enhanced_questions if q.get('has_cot', False))

    print("\\n" + "="*70)
    print("COT GENERATION COMPLETE")
    print("="*70)
    print(f"\\nTotal processed: {len(enhanced_questions)}")
    print(f"With CoT: {success_count}")
    print(f"Success rate: {success_count/len(questions)*100:.1f}%")
    print(f"Errors: {errors}")
    print(f"\\nOutput saved to: {output_path}")

    return enhanced_questions

def main():
    """Main entry point"""

    # Load dataset
    dataset_path = Path("data/comprehensive/full_50k_dataset.json")

    if not dataset_path.exists():
        print("❌ Dataset not found!")
        return

    with open(dataset_path) as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")

    # Generate CoT
    output_path = Path("data/cot_enhanced/deepseek_cot_50k.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enhanced = generate_cot_for_dataset(questions, str(output_path))

    print("\\n✅ Ready for curriculum learning and training!")

if __name__ == "__main__":
    main()
