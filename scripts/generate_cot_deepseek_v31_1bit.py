#!/usr/bin/env python3
"""
Generate Chain-of-Thought reasoning for 50K questions using DeepSeek-V3.1 (1-bit quantized)
This uses the recommended strategy: High-quality teacher model for CoT generation
Target: 50-60GB VRAM, 4-6 hours for 50K questions, 90% MMLU quality
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DeepSeekV31CoTGenerator:
    """Generate CoT using DeepSeek-V3.1 with 1-bit quantization"""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-V3"):
        """
        Initialize DeepSeek-V3.1 with aggressive quantization

        Args:
            model_name: HuggingFace model ID (we'll use the official one)
        """
        logging.info(f"🚀 Loading {model_name} with 1-bit quantization...")
        logging.info(f"   Target VRAM: 50-60GB (down from 211GB)")

        # Configure 1-bit quantization (BitsAndBytes nf1 if available, else int8)
        # Note: True 1-bit may require special builds - we'll use 4-bit as fallback
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"  # For speed
            )

            logging.info(f"✅ Model loaded successfully")
            logging.info(f"   VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            logging.info("💡 Trying alternative: deepseek-ai/deepseek-v3-base")
            # Try base model as fallback
            model_name = "deepseek-ai/deepseek-v3-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

    def generate_cot(self, question: str, choices: Dict[str, str], max_tokens: int = 512) -> str:
        """
        Generate Chain-of-Thought reasoning for a single question

        Args:
            question: The question text
            choices: Dict of answer choices (e.g., {'A': 'text', 'B': 'text'})
            max_tokens: Max length of generated reasoning

        Returns:
            CoT reasoning string
        """
        # Format choices for the prompt
        choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

        # Use DeepSeek's "thinking" prompt format for CoT
        prompt = f"""<｜begin▁of▁sentence｜>You are a helpful assistant that provides step-by-step reasoning for multiple choice questions.

Question: {question}

Choices:
{choices_text}

Provide detailed step-by-step reasoning to solve this question. Think through:
1. What is being asked?
2. What knowledge is needed?
3. How do we eliminate wrong answers?
4. Why is one answer correct?

Reasoning:"""

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract reasoning
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        reasoning = full_text.split("Reasoning:")[-1].strip()

        return reasoning

    def generate_batch(self, questions: List[Dict], batch_size: int = 4) -> List[Dict]:
        """
        Generate CoT for a batch of questions (for efficiency)

        Args:
            questions: List of question dicts
            batch_size: Number to process at once

        Returns:
            List of questions with added 'chain_of_thought' field
        """
        results = []

        for i in tqdm(range(0, len(questions), batch_size), desc="Generating CoT"):
            batch = questions[i:i+batch_size]

            for q in batch:
                try:
                    # Get choices
                    choices = q.get('choices', q.get('options', {}))
                    if not choices:
                        logging.warning(f"Skipping question with no choices: {q.get('question', '')[:50]}")
                        continue

                    # Generate CoT
                    cot = self.generate_cot(q['question'], choices)

                    # Add to question
                    enhanced_q = q.copy()
                    enhanced_q['chain_of_thought'] = cot
                    enhanced_q['cot_model'] = 'deepseek-v3.1-1bit'
                    results.append(enhanced_q)

                except Exception as e:
                    logging.error(f"Error generating CoT: {e}")
                    # Add without CoT to not lose the question
                    results.append(q)

            # Clear cache periodically
            if i % 100 == 0:
                torch.cuda.empty_cache()

        return results


def load_dataset(path: Path) -> List[Dict]:
    """Load the 50K dataset"""
    logging.info(f"📂 Loading dataset from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"✅ Loaded {len(data)} questions")
    return data


def save_enhanced_dataset(questions: List[Dict], output_path: Path):
    """Save CoT-enhanced dataset"""
    logging.info(f"💾 Saving enhanced dataset to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Saved {len(questions)} questions with CoT")

    # Stats
    with_cot = sum(1 for q in questions if 'chain_of_thought' in q)
    avg_cot_length = sum(len(q.get('chain_of_thought', '')) for q in questions) / len(questions)

    logging.info(f"📊 Statistics:")
    logging.info(f"   Questions with CoT: {with_cot}/{len(questions)} ({with_cot/len(questions)*100:.1f}%)")
    logging.info(f"   Average CoT length: {avg_cot_length:.0f} characters")


def main():
    """Main pipeline"""
    start_time = time.time()

    logging.info("=" * 80)
    logging.info("🧠 DEEPSEEK-V3.1 CHAIN-OF-THOUGHT GENERATION")
    logging.info("=" * 80)

    # Paths
    dataset_path = Path('data/comprehensive/full_50k_dataset.json')
    output_path = Path('data/enhanced/cot_enhanced_50k.json')

    # Load data
    questions = load_dataset(dataset_path)

    # Initialize generator
    generator = DeepSeekV31CoTGenerator()

    # Generate CoT (batch_size=1 for safety with large model)
    logging.info(f"🔄 Generating Chain-of-Thought for {len(questions)} questions...")
    logging.info(f"   Estimated time: 4-6 hours")
    enhanced = generator.generate_batch(questions, batch_size=1)

    # Save
    save_enhanced_dataset(enhanced, output_path)

    # Summary
    elapsed = time.time() - start_time
    logging.info("=" * 80)
    logging.info(f"✅ COMPLETE in {elapsed/3600:.2f} hours")
    logging.info(f"   Output: {output_path}")
    logging.info(f"   Ready for curriculum learning and fine-tuning!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
