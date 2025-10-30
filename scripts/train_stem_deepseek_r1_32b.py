#!/usr/bin/env python3
"""
Train STEM Specialist Adapter on DeepSeek-R1-Distill-Qwen-32B
Uses our existing 22K STEM questions for gap filling
"""

import json
import logging
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_stem_questions():
    """Load STEM questions from our question pool"""
    pool_file = Path("/workspace/question_pool.json")

    with open(pool_file) as f:
        all_questions = json.load(f)

    # Filter STEM questions
    stem_questions = [q for q in all_questions if q.get('category') == 'STEM']

    logging.info(f"Loaded {len(stem_questions)} STEM questions")
    return stem_questions

def format_for_training(question):
    """Format MCQ question for training"""
    q = question['question']
    choices = question['choices']
    correct = question['correct_answer']

    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

    # Training format with answer
    text = f"""<|im_start|>system
You are an expert in STEM subjects. Answer multiple choice questions accurately.<|im_end|>
<|im_start|>user
{q}

{choices_text}<|im_end|>
<|im_start|>assistant
The correct answer is {correct}.<|im_end|>"""

    return {"text": text}

def main():
    logging.info("=" * 80)
    logging.info("🎓 TRAINING STEM ADAPTER - DeepSeek-R1-32B")
    logging.info("=" * 80)

    # Load model
    logging.info("Loading DeepSeek-R1-Distill-Qwen-32B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/workspace/models/deepseek_r1_qwen32b",
        max_seq_length=1024,  # Longer context for complex questions
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    logging.info("Adding RSLoRA adapters (r=128, α=256)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=256,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=True,  # RSLoRA for stability
        random_state=42,
    )

    # Load and format dataset
    logging.info("Loading STEM dataset...")
    stem_questions = load_stem_questions()

    # Format for training
    formatted = [format_for_training(q) for q in stem_questions]
    dataset = Dataset.from_list(formatted)

    logging.info(f"Training on {len(dataset)} STEM questions")

    # Training config - optimized for 32B speed
    training_args = TrainingArguments(
        output_dir='/workspace/models/stem_deepseek_r1_32b_r128',
        num_train_epochs=2,  # 2 epochs for 22K questions
        per_device_train_batch_size=2,  # 32B can handle larger batches
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )

    # Train
    logging.info("Starting training...")
    trainer.train()

    # Save final model
    logging.info("Saving final model...")
    model.save_pretrained("/workspace/models/stem_deepseek_r1_32b_r128/final_model")
    tokenizer.save_pretrained("/workspace/models/stem_deepseek_r1_32b_r128/final_model")

    logging.info("=" * 80)
    logging.info("✅ STEM ADAPTER TRAINING COMPLETE")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()
