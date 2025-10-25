"""
Default Training Configuration for AMD MI300X + Unsloth
Fine-tuning Q&A agent with LoRA/QLoRA
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration"""

    # Base model options - choose one
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"  # Recommended
    # model_name: str = "unsloth/mistral-7b-bnb-4bit"
    # model_name: str = "unsloth/Qwen2.5-7B-bnb-4bit"

    # Model parameters
    max_seq_length: int = 2048  # Maximum sequence length
    load_in_4bit: bool = True  # Use 4-bit quantization (saves memory)
    dtype: Optional[str] = None  # Auto-detect (bfloat16 on MI300X)


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration"""

    # LoRA parameters
    r: int = 16  # LoRA rank (8, 16, 32, 64)
    lora_alpha: int = 16  # LoRA alpha (usually same as r)
    lora_dropout: float = 0.0  # Dropout (0 recommended for Unsloth)

    # Target modules for LoRA
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Additional LoRA settings
    bias: str = "none"  # Bias training ("none", "all", "lora_only")
    use_gradient_checkpointing: bool = True  # Saves memory
    random_state: int = 3407  # Random seed
    use_rslora: bool = False  # Rank-stabilized LoRA
    loftq_config: Optional[dict] = None  # LoftQ initialization


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    # Batch sizes
    per_device_train_batch_size: int = 2  # Batch size per GPU
    gradient_accumulation_steps: int = 4  # Effective batch = 2 * 4 = 8

    # Learning rate
    learning_rate: float = 2e-4  # 2e-4 is standard for LoRA
    weight_decay: float = 0.01  # Weight decay for regularization
    warmup_steps: int = 5  # Warmup steps
    lr_scheduler_type: str = "linear"  # LR scheduler

    # Training steps
    max_steps: int = 100  # Maximum training steps
    # Alternative: num_train_epochs: int = 3

    # Optimization
    optim: str = "adamw_8bit"  # 8-bit AdamW optimizer
    fp16: bool = False  # FP16 training (auto-detect)
    bf16: bool = True  # BF16 training (better on MI300X)

    # Logging
    logging_steps: int = 1  # Log every N steps
    save_steps: int = 50  # Save checkpoint every N steps
    eval_steps: int = 50  # Evaluate every N steps

    # Evaluation
    evaluation_strategy: str = "steps"  # "steps" or "epoch"
    do_eval: bool = True  # Run evaluation

    # Output
    output_dir: str = "training/outputs"  # Output directory
    save_total_limit: int = 3  # Keep only last N checkpoints

    # Misc
    seed: int = 3407  # Random seed
    report_to: str = "none"  # Reporting ("tensorboard", "wandb", "none")


@dataclass
class DataConfig:
    """Dataset configuration"""

    train_data_path: str = "data/processed/train.json"
    val_data_path: str = "data/processed/val.json"

    # Data formatting
    dataset_text_field: str = "text"  # After formatting
    max_seq_length: int = 2048  # Must match ModelConfig

    # Alpaca format
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_LORA_CONFIG = LoRAConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()


def get_config(config_name: str = "default"):
    """Get training configuration by name"""
    if config_name == "default":
        return {
            "model": DEFAULT_MODEL_CONFIG,
            "lora": DEFAULT_LORA_CONFIG,
            "training": DEFAULT_TRAINING_CONFIG,
            "data": DEFAULT_DATA_CONFIG,
        }
    elif config_name == "fast":
        # Fast training for testing
        training_config = TrainingConfig(
            max_steps=20,
            save_steps=10,
            eval_steps=10,
        )
        return {
            "model": DEFAULT_MODEL_CONFIG,
            "lora": DEFAULT_LORA_CONFIG,
            "training": training_config,
            "data": DEFAULT_DATA_CONFIG,
        }
    elif config_name == "high_quality":
        # Longer training for better quality
        training_config = TrainingConfig(
            max_steps=200,
            save_steps=50,
            eval_steps=50,
            learning_rate=1e-4,
        )
        lora_config = LoRAConfig(r=32)  # Higher rank
        return {
            "model": DEFAULT_MODEL_CONFIG,
            "lora": lora_config,
            "training": training_config,
            "data": DEFAULT_DATA_CONFIG,
        }
    else:
        raise ValueError(f"Unknown config: {config_name}")


if __name__ == "__main__":
    # Print default configuration
    print("Default Training Configuration")
    print("=" * 60)
    print(f"\nModel: {DEFAULT_MODEL_CONFIG.model_name}")
    print(f"Max Sequence Length: {DEFAULT_MODEL_CONFIG.max_seq_length}")
    print(f"4-bit Quantization: {DEFAULT_MODEL_CONFIG.load_in_4bit}")
    print(f"\nLoRA Rank: {DEFAULT_LORA_CONFIG.r}")
    print(f"LoRA Alpha: {DEFAULT_LORA_CONFIG.lora_alpha}")
    print(f"\nBatch Size: {DEFAULT_TRAINING_CONFIG.per_device_train_batch_size}")
    print(
        f"Gradient Accumulation: {DEFAULT_TRAINING_CONFIG.gradient_accumulation_steps}"
    )
    print(f"Effective Batch Size: {DEFAULT_TRAINING_CONFIG.per_device_train_batch_size * DEFAULT_TRAINING_CONFIG.gradient_accumulation_steps}")
    print(f"Learning Rate: {DEFAULT_TRAINING_CONFIG.learning_rate}")
    print(f"Max Steps: {DEFAULT_TRAINING_CONFIG.max_steps}")
    print("=" * 60)
