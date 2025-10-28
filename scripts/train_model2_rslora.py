#!/usr/bin/env python3
"""
MODEL #2: Fine-tune Qwen2.5-72B with RSLoRA (RoRA) + 65K dataset
Optimized for AMD MI300X with 192GB VRAM
Target: 88-92% accuracy (+6.5% gain from RSLoRA vs standard LoRA)

RSLoRA (Rank-Stabilized LoRA / RoRA):
- Enables high-rank adaptation (128) without gradient collapse
- Uses α/√r scaling instead of α/r for stable gradients
- January 2025 research: +6.5% over LoRA, +2.9% over DoRA
- No training speed penalty - just scaling factor change

Configuration:
- Rank: 128 (2× Model #1)
- Alpha: 256 (effective scale: 256/√128 ≈ 22.6)
- Dataset: 65K questions (50K + 15K new)
- Batch size: 8 × 8 gradient accumulation (optimized for MI300X)
"""

import json
import logging
import random
import os
from pathlib import Path
from typing import Dict, List

# Skip torchao import to avoid torch.int1 compatibility issues
os.environ['TRANSFORMERS_NO_TORCHAO'] = '1'

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
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
    Format question as instruction-following prompt for Qwen2.5

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

    # Instruction format for Qwen2.5
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


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
    """
    Load Qwen2.5-72B with Unsloth for ROCm-compatible 4-bit quantization

    Args:
        model_name: Model identifier

    Returns:
        (model, tokenizer)
    """
    logging.info(f"🚀 Loading {model_name} with Unsloth...")
    logging.info(f"   Note: On AMD, Unsloth uses 16-bit instead of 4-bit")
    logging.info(f"   Expected VRAM usage: ~135GB (leaving ~57GB free)")

    # Load with Unsloth - optimized for ROCm
    # Note: Unsloth automatically disables 4-bit on AMD (uses 16-bit instead)
    # This means the model will use ~135GB VRAM instead of ~50GB
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # Auto-detect (will use bf16)
        load_in_4bit=True,  # Ignored on AMD, loads in 16-bit
    )

    vram_used = torch.cuda.memory_allocated() / 1024**3
    logging.info(f"✅ Model loaded successfully with Unsloth")
    logging.info(f"   VRAM: {vram_used:.2f}GB / 192GB")
    logging.info(f"   Free for training: {192 - vram_used:.2f}GB")

    return model, tokenizer


def setup_lora(model, rank: int = 128, alpha: int = 256):
    """
    Configure RSLoRA (RoRA) for parameter-efficient fine-tuning using Unsloth

    RSLoRA uses α/√r scaling instead of α/r for stable high-rank training.

    Args:
        model: Base model
        rank: LoRA rank (128 for Model #2, 2× Model #1)
        alpha: LoRA alpha scaling factor (256 = effective scale ~22.6 with RSLoRA)

    Returns:
        RSLoRA-enabled model
    """
    logging.info(f"⚙️  Setting up RSLoRA (rank={rank}, alpha={alpha})...")
    logging.info(f"   Effective scale: α/√r = {alpha}/{rank**0.5:.2f} ≈ {alpha/(rank**0.5):.2f}")

    # Unsloth's optimized LoRA setup with RSLoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_alpha=alpha,
        lora_dropout=0,  # Set to 0 for Unsloth fast patching
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        use_rslora=True,  # ✅ Enable RSLoRA (α/√r scaling)
        random_state=42,
    )

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    logging.info(f"✅ LoRA configured with Unsloth")
    logging.info(f"   Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
    logging.info(f"   Total params: {total:,}")
    logging.info(f"   Unsloth speedup: ~2x faster training expected")

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
    logging.info("🎓 CURRICULUM TRAINING WITH REPLAY BUFFER (UNSLOTH)")
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

        # Adaptive learning rate (decrease for harder questions)
        lr = 2e-5 * (1.0 - chunk_idx / n_chunks * 0.5)  # 2e-5 → 1e-5

        # Training arguments - can use larger batch size with Qwen2.5-72B
        training_args = TrainingArguments(
            output_dir=str(output_path / f"checkpoint_chunk{chunk_idx}"),
            num_train_epochs=1,
            per_device_train_batch_size=4,  # Increased from 1 (more VRAM available)
            gradient_accumulation_steps=4,  # Reduced from 16 (larger batch size)
            learning_rate=lr,
            warmup_steps=50,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=False,
            bf16=True,  # Use bfloat16
            optim="adamw_torch",  # Standard PyTorch AdamW (no bitsandbytes needed)
            report_to="none",
            gradient_checkpointing=True,  # Save VRAM
        )

        # Unsloth's optimized SFT trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            args=training_args,
            packing=False,  # Don't pack sequences for Q&A
        )

        logging.info(f"   🏋️  Training chunk {chunk_idx + 1}...")

        # Log expected time
        steps_per_chunk = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        est_time_minutes = steps_per_chunk * 0.5  # Estimate 0.5 min per step
        logging.info(f"   ⏱️  Estimated time: {est_time_minutes:.0f} minutes")

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
    logging.info("🎯 MODEL #2: QWEN2.5-72B with RSLoRA (RoRA) + 65K Dataset")
    logging.info("=" * 80)
    logging.info("📊 Expected Results (January 2025 Research):")
    logging.info("   - Model #1 (Standard LoRA r=64): 85-87%")
    logging.info("   - Model #2 (RSLoRA r=128): 88-92% (+6.5% potential gain)")
    logging.info("   - Training time: 10-12 hours (65K dataset)")
    logging.info("   - VRAM usage: ~60GB (rank 128 = 2× Model #1 params)")
    logging.info("\n🔬 RSLoRA Configuration:")
    logging.info("   - Rank: 128 (2× Model #1)")
    logging.info("   - Alpha: 256 (effective scale: α/√r ≈ 22.6)")
    logging.info("   - Scaling: α/√r (RSLoRA) vs α/r (standard LoRA)")
    logging.info("   - Dataset: 65K questions (50K + 15K new)")

    # Paths
    train_path = Path('/workspace/data/curriculum/train_65k_merged.json')
    output_dir = '/workspace/models/model2_rslora_r128'

    # Load training data
    logging.info(f"\n📂 Loading training data from {train_path}")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    logging.info(f"✅ Loaded {len(train_data)} curriculum-ordered questions")

    # Show difficulty distribution
    difficulties = [q.get('difficulty_score', 5.0) for q in train_data]
    easy = sum(1 for d in difficulties if d < 4.5)
    medium = sum(1 for d in difficulties if 4.5 <= d < 7.0)
    hard = sum(1 for d in difficulties if d >= 7.0)
    logging.info(f"   Distribution: Easy {easy} ({easy/len(train_data)*100:.1f}%), Medium {medium} ({medium/len(train_data)*100:.1f}%), Hard {hard} ({hard/len(train_data)*100:.1f}%)")

    # Initialize replay buffer (larger for bigger dataset)
    replay_buffer = ReplayBuffer(buffer_size=1000)

    # Load model and tokenizer with Unsloth
    model, tokenizer = load_model_and_tokenizer()

    # Setup RSLoRA with Unsloth (rank=128, alpha=256, use_rslora=True)
    model = setup_lora(model, rank=128, alpha=256)

    # Train with curriculum + replay
    curriculum_training(
        model,
        tokenizer,
        train_data,
        output_dir,
        replay_buffer,
        chunk_size=6500  # Slightly larger chunks for 65K dataset
    )

    logging.info("\n🎉 Model #2 training complete!")
    logging.info("   RSLoRA (rank 128) training finished")
    logging.info("   Next: Validate and prepare Model #3 (DoRA)")


if __name__ == "__main__":
    main()