#!/usr/bin/env python3
"""
Fine-tune Qwen3-235B with LoRA + Replay Buffer + Curriculum Learning
Implements April 2025 research: "Replay to Remember" for catastrophic forgetting mitigation
Target: 86% accuracy with minimal forgetting
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ReplayBuffer:
    """
    Replay buffer for catastrophic forgetting mitigation
    Based on: "Replay to Remember" (April 2025)
    """

    def __init__(self, buffer_size: int = 500):
        """
        Initialize replay buffer

        Args:
            buffer_size: Number of examples to keep in buffer
        """
        self.buffer_size = buffer_size
        self.buffer = []
        logging.info(f"📦 Replay buffer initialized (size: {buffer_size})")

    def add_batch(self, examples: List[Dict]):
        """Add examples to replay buffer with reservoir sampling"""
        for example in examples:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(example)
            else:
                # Reservoir sampling: replace random element
                idx = random.randint(0, len(self.buffer) - 1)
                self.buffer[idx] = example

    def sample(self, n: int) -> List[Dict]:
        """Sample n examples from buffer"""
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)

    def get_replay_dataset(self) -> List[Dict]:
        """Get all buffered examples for replay"""
        return self.buffer.copy()


def format_question_for_training(question: Dict) -> str:
    """
    Format question as instruction-following prompt

    Args:
        question: Question dict with question, choices, correct_answer

    Returns:
        Formatted prompt string
    """
    q_text = question['question']
    choices = question.get('choices', {})
    correct = question.get('correct_answer', 'A')

    # Format choices
    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

    # Instruction format for Qwen3
    prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
Question: {q_text}

Choices:
{choices_text}

Please select the correct answer (A, B, C, or D).<|im_end|>
<|im_start|>assistant
The correct answer is {correct}.<|im_end|>"""

    return prompt


def create_training_dataset(
    questions: List[Dict],
    replay_buffer: ReplayBuffer,
    replay_ratio: float = 0.1
) -> List[Dict]:
    """
    Create training dataset with replay examples mixed in

    Args:
        questions: New questions to train on
        replay_buffer: Buffer containing previous examples
        replay_ratio: Ratio of replay examples to add (0.1 = 10%)

    Returns:
        Combined dataset with replay examples
    """
    # Format new questions
    formatted_new = [
        {'text': format_question_for_training(q)}
        for q in questions
    ]

    # Add to replay buffer
    replay_buffer.add_batch(questions)

    # Sample replay examples
    n_replay = int(len(formatted_new) * replay_ratio)
    if n_replay > 0:
        replay_examples = replay_buffer.sample(n_replay)
        formatted_replay = [
            {'text': format_question_for_training(q)}
            for q in replay_examples
        ]

        # Mix replay examples throughout
        combined = formatted_new + formatted_replay
        random.shuffle(combined)

        logging.info(f"   📊 Training batch: {len(formatted_new)} new + {len(formatted_replay)} replay = {len(combined)} total")
        return combined

    return formatted_new


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"):
    """
    Load Qwen3-235B with 4-bit quantization for LoRA training

    Args:
        model_name: Model identifier

    Returns:
        (model, tokenizer)
    """
    logging.info(f"🚀 Loading {model_name} with 4-bit quantization...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    logging.info(f"✅ Model loaded successfully")
    logging.info(f"   VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    return model, tokenizer


def setup_lora(model, rank: int = 64, alpha: int = 128):
    """
    Configure LoRA for parameter-efficient fine-tuning

    Args:
        model: Base model
        rank: LoRA rank (higher = more parameters, better quality)
        alpha: LoRA alpha scaling factor

    Returns:
        LoRA-enabled model
    """
    logging.info(f"⚙️  Setting up LoRA (rank={rank}, alpha={alpha})...")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    logging.info(f"✅ LoRA configured")
    logging.info(f"   Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
    logging.info(f"   Total params: {total:,}")

    return model


def curriculum_training(
    model,
    tokenizer,
    train_data: List[Dict],
    output_dir: str,
    replay_buffer: ReplayBuffer,
    chunk_size: int = 5000
):
    """
    Train with curriculum learning (easy → hard) + replay buffer

    Args:
        model: LoRA-enabled model
        tokenizer: Tokenizer
        train_data: Curriculum-ordered training data
        output_dir: Output directory for checkpoints
        replay_buffer: Replay buffer for forgetting mitigation
        chunk_size: Questions per curriculum chunk
    """
    logging.info("=" * 80)
    logging.info("🎓 CURRICULUM TRAINING WITH REPLAY BUFFER")
    logging.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split into curriculum chunks
    n_chunks = (len(train_data) + chunk_size - 1) // chunk_size
    logging.info(f"📚 Training in {n_chunks} curriculum chunks ({chunk_size} questions each)")
    logging.info(f"   Strategy: Easy → Medium → Hard with replay throughout")

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(train_data))
        chunk = train_data[start_idx:end_idx]

        # Get difficulty range for this chunk
        difficulties = [q.get('difficulty_score', 5.0) for q in chunk]
        avg_diff = sum(difficulties) / len(difficulties)

        logging.info(f"\n📖 Chunk {chunk_idx + 1}/{n_chunks}")
        logging.info(f"   Questions: {len(chunk)}")
        logging.info(f"   Avg difficulty: {avg_diff:.1f}")

        # Create training dataset with replay
        replay_ratio = 0.1 + (chunk_idx / n_chunks) * 0.1  # Increase replay over time
        train_dataset = create_training_dataset(chunk, replay_buffer, replay_ratio)

        # Convert to HuggingFace dataset
        hf_dataset = Dataset.from_list(train_dataset)

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=2048,
                return_tensors='pt'
            )

        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        # Adaptive learning rate (decrease for harder questions)
        lr = 2e-5 * (1.0 - chunk_idx / n_chunks * 0.5)  # 2e-5 → 1e-5

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path / f"checkpoint_chunk{chunk_idx}"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=lr,
            warmup_steps=50,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        logging.info(f"   🏋️  Training chunk {chunk_idx + 1}...")
        trainer.train()

        # Save chunk checkpoint
        chunk_output = output_path / f"checkpoint_chunk{chunk_idx}"
        model.save_pretrained(chunk_output)
        tokenizer.save_pretrained(chunk_output)
        logging.info(f"   ✅ Checkpoint saved: {chunk_output}")

    # Save final model
    final_output = output_path / "final_model"
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    logging.info("\n" + "=" * 80)
    logging.info("✅ CURRICULUM TRAINING COMPLETE")
    logging.info(f"   Final model: {final_output}")
    logging.info("=" * 80)


def main():
    """Main training pipeline"""
    logging.info("=" * 80)
    logging.info("🎯 QWEN3-235B FINE-TUNING: LoRA + Replay + Curriculum")
    logging.info("=" * 80)

    # Paths
    train_path = Path('data/curriculum/train_45k.json')
    output_dir = 'models/qwen3_235b_lora_curriculum'

    # Load training data
    logging.info(f"\n📂 Loading training data from {train_path}")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    logging.info(f"✅ Loaded {len(train_data)} curriculum-ordered questions")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size=500)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Setup LoRA
    model = setup_lora(model, rank=64, alpha=128)

    # Train with curriculum + replay
    curriculum_training(
        model,
        tokenizer,
        train_data,
        output_dir,
        replay_buffer,
        chunk_size=5000
    )

    logging.info("\n🎉 Training complete! Ready for validation testing.")


if __name__ == "__main__":
    main()
