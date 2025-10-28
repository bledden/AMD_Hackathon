#!/usr/bin/env python3
"""
HUMANITIES SPECIALIST - RSLoRA r=128

Novel Approach: Domain-specific specialist with high-rank RSLoRA
Deep expertise in Law, Psychology, Philosophy, Social Sciences

This specialist will be MERGED with STEM specialist via TIES-Merging
for zero-latency inference with complementary domain knowledge.

Configuration:
- Rank: 128 (high-rank for complex ethical/legal reasoning)
- Alpha: 256 (RSLoRA α/√r ≈ 22.63 scaling)
- Dataset: 24K humanities + general questions
- Training: ~3-4 hours (curriculum learning)
"""

import json
import logging
import random
import os
from pathlib import Path
from typing import Dict, List

os.environ['TRANSFORMERS_NO_TORCHAO'] = '1'

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s'
)

class ReplayBuffer:
    """Replay buffer for catastrophic forgetting mitigation"""

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer = []
        logging.info(f"📦 Replay buffer initialized (size: {buffer_size})")

    def add_batch(self, examples: List[Dict]):
        for example in examples:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(example)
            else:
                idx = random.randint(0, len(self.buffer) - 1)
                self.buffer[idx] = example

    def sample(self, n: int) -> List[Dict]:
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)


def format_question_for_training(question: Dict) -> str:
    """Format question as instruction-following prompt for Qwen2.5"""
    q_text = question['question']
    choices = question.get('choices', {})
    correct = question.get('correct_answer', 'A')

    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

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
    """Create training dataset with replay examples"""
    formatted_new = [{'text': format_question_for_training(q)} for q in questions]
    replay_buffer.add_batch(questions)

    n_replay = int(len(formatted_new) * replay_ratio)
    if n_replay > 0:
        replay_examples = replay_buffer.sample(n_replay)
        formatted_replay = [{'text': format_question_for_training(q)} for q in replay_examples]
        combined = formatted_new + formatted_replay
        random.shuffle(combined)
        logging.info(f"   📊 Training batch: {len(formatted_new)} new + {len(formatted_replay)} replay = {len(combined)} total")
        return combined

    return formatted_new


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
    """Load Qwen2.5-72B with Unsloth"""
    logging.info(f"🚀 Loading {model_name} with Unsloth...")
    logging.info(f"   Expected VRAM usage: ~135GB")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    vram_used = torch.cuda.memory_allocated() / 1024**3
    logging.info(f"✅ Model loaded successfully")
    logging.info(f"   VRAM: {vram_used:.2f}GB / 192GB")
    logging.info(f"   Free: {192 - vram_used:.2f}GB")

    return model, tokenizer


def setup_rslora(model, rank: int = 128, alpha: int = 256):
    """Configure RSLoRA for parameter-efficient fine-tuning"""
    logging.info(f"⚙️  Setting up RSLoRA (rank={rank}, alpha={alpha})...")
    logging.info(f"   Effective scale: α/√r = {alpha}/{rank**0.5:.2f} ≈ {alpha/(rank**0.5):.2f}")
    logging.info(f"   This enables HIGH-RANK training without gradient collapse")

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=True,  # ✅ RSLoRA α/√r scaling
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    logging.info(f"✅ RSLoRA configured")
    logging.info(f"   Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
    logging.info(f"   Total params: {total:,}")

    return model


def curriculum_training(
    model,
    tokenizer,
    train_data: List[Dict],
    output_dir: str,
    replay_buffer: ReplayBuffer,
    chunk_size: int = 4000
):
    """Train with curriculum learning + replay buffer"""
    logging.info("=" * 80)
    logging.info("🎓 HUMANITIES SPECIALIST CURRICULUM TRAINING (RSLoRA)")
    logging.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(train_data) + chunk_size - 1) // chunk_size
    logging.info(f"📚 Training in {n_chunks} curriculum chunks ({chunk_size} questions each)")

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(train_data))
        chunk = train_data[start_idx:end_idx]

        difficulties = [q.get('difficulty_score', 5.0) for q in chunk]
        avg_diff = sum(difficulties) / len(difficulties)

        logging.info(f"\n📖 Chunk {chunk_idx + 1}/{n_chunks}")
        logging.info(f"   Questions: {len(chunk)}")
        logging.info(f"   Avg difficulty: {avg_diff:.1f}")

        replay_ratio = 0.1 + (chunk_idx / n_chunks) * 0.1
        train_dataset = create_training_dataset(chunk, replay_buffer, replay_ratio)
        hf_dataset = Dataset.from_list(train_dataset)

        base_lr = 2e-5 * (1.0 - chunk_idx / n_chunks * 0.5)

        training_args = TrainingArguments(
            output_dir=str(output_path / f"checkpoint_chunk{chunk_idx}"),
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=base_lr,
            warmup_steps=50,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=False,
            bf16=True,
            optim="adamw_torch",
            report_to="none",
            gradient_checkpointing=True,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            args=training_args,
            packing=False,
        )

        logging.info(f"   🏋️  Training chunk {chunk_idx + 1}...")
        steps = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        est_time = steps * 3 / 60
        logging.info(f"   ⏱️  Estimated time: {est_time:.0f} minutes")

        trainer.train()

        chunk_output = output_path / f"checkpoint_chunk{chunk_idx}"
        model.save_pretrained(chunk_output)
        tokenizer.save_pretrained(chunk_output)
        logging.info(f"   ✅ Checkpoint saved: {chunk_output}")

    final_output = output_path / "final_model"
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    logging.info("\n" + "=" * 80)
    logging.info("✅ HUMANITIES SPECIALIST TRAINING COMPLETE")
    logging.info(f"   Final model: {final_output}")
    logging.info("=" * 80)


def main():
    """Main training pipeline"""
    logging.info("=" * 80)
    logging.info("📚 HUMANITIES SPECIALIST - RSLoRA r=128")
    logging.info("=" * 80)
    logging.info("🎯 Training Focus:")
    logging.info("   - Law (professional law, jurisprudence)")
    logging.info("   - Psychology (clinical, social, developmental)")
    logging.info("   - Philosophy (ethics, moral reasoning)")
    logging.info("   - Social Sciences (sociology, history, politics)")
    logging.info("   - General knowledge (context)")
    logging.info("\n🔬 Configuration:")
    logging.info("   - Method: RSLoRA (Rank-Stabilized LoRA)")
    logging.info("   - Rank: 128 (high-rank for complex ethical/legal reasoning)")
    logging.info("   - Alpha: 256 (α/√r ≈ 22.63 scaling)")
    logging.info("   - Dataset: 23,734 humanities + general questions")
    logging.info("   - Training time: ~3-4 hours")

    train_path = Path('/workspace/data/curriculum/train_humanities_specialist.json')
    output_dir = '/workspace/models/humanities_specialist_rslora_r128'

    logging.info(f"\n📂 Loading Humanities specialist dataset from {train_path}")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    logging.info(f"✅ Loaded {len(train_data)} humanities-focused questions")

    difficulties = [q.get('difficulty_score', 5.0) for q in train_data]
    easy = sum(1 for d in difficulties if d < 4.5)
    medium = sum(1 for d in difficulties if 4.5 <= d < 7.0)
    hard = sum(1 for d in difficulties if d >= 7.0)
    logging.info(f"   Distribution: Easy {easy} ({easy/len(train_data)*100:.1f}%), Medium {medium} ({medium/len(train_data)*100:.1f}%), Hard {hard} ({hard/len(train_data)*100:.1f}%)")

    replay_buffer = ReplayBuffer(buffer_size=1000)
    model, tokenizer = load_model_and_tokenizer()
    model = setup_rslora(model, rank=128, alpha=256)

    curriculum_training(
        model,
        tokenizer,
        train_data,
        output_dir,
        replay_buffer,
        chunk_size=4000
    )

    logging.info("\n🎉 Humanities Specialist training complete!")
    logging.info("   Next: Merge STEM + Humanities specialists via TIES-Merging")
    logging.info("   Then: Create ensemble with Model #1")


if __name__ == "__main__":
    main()
