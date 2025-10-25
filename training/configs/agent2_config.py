"""
Agent 2: Challenger - Qwen3 8B Configuration

Strategy: Aggressive, creative with synthetic data
Model: Qwen3 8B Instruct (2025)
Performance: Hybrid reasoning, 8x longer context, Dynamic 2.0 quantization
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration for Qwen3 8B"""
    model_name: str = "unsloth/Qwen3-8B-Instruct-bnb-4bit"  # Pre-quantized 4-bit model
    max_seq_length: int = 4096  # Can leverage 8x longer context
    load_in_4bit: bool = True
    dtype: str = "float16"  # Use FP16 for training

    # Qwen3 specific settings
    use_gradient_checkpointing: bool = True
    use_cache: bool = False  # Disable during training

@dataclass
class LoRAConfig:
    """LoRA configuration for Qwen3 8B - Aggressive approach"""
    r: int = 24  # Higher LoRA rank for more expressiveness
    lora_alpha: int = 24  # Matches rank
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Target modules for Qwen3
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Training configuration for Qwen3 8B"""
    output_dir: str = "training/outputs/agent2_qwen3"

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # Standard for 8B model
    gradient_accumulation_steps: int = 2  # Total effective batch = 8

    # Optimizer settings
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    optim: str = "adamw_8bit"

    # Precision and performance
    fp16: bool = True
    bf16: bool = False  # Use FP16 instead
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 3

    # Evaluation
    evaluation_strategy: str = "epoch"
    eval_steps: int = 100

    # Speed optimizations
    group_by_length: bool = True
    dataloader_num_workers: int = 4

    # SFT specific
    max_seq_length: int = 4096
    packing: bool = False  # No packing for cleaner training

# Default configs
MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()

# Agent 2 specific settings
AGENT_NAME = "Challenger"
AGENT_STRATEGY = "Aggressive, synthetic data, creative questions"
EXPECTED_TRAINING_TIME_HOURS = 70
EXPECTED_COST_USD = 140  # 70 hours * $1.99/hr
