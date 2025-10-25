"""
Agent 3: Hybrid - Mistral NeMo 12B Configuration

Strategy: Domain specialist with balanced approach
Model: Mistral NeMo 12B Instruct (2024)
Performance: Superior instruction following, 128K context, fits in 12GB VRAM
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration for Mistral NeMo 12B"""
    model_name: str = "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"  # Pre-quantized 4-bit model
    max_seq_length: int = 4096  # Can leverage 128K context if needed
    load_in_4bit: bool = True
    dtype: str = "float16"  # Use FP16 for training

    # Mistral NeMo specific settings
    use_gradient_checkpointing: bool = True
    use_cache: bool = False  # Disable during training

@dataclass
class LoRAConfig:
    """LoRA configuration for Mistral NeMo 12B - Balanced approach"""
    r: int = 16  # Balanced LoRA rank
    lora_alpha: int = 16  # Matches rank
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Target modules for Mistral NeMo
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Training configuration for Mistral NeMo 12B"""
    output_dir: str = "training/outputs/agent3_nemo"

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 3  # Slightly reduced for 12B model
    gradient_accumulation_steps: int = 3  # Total effective batch = 9

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

# Agent 3 specific settings
AGENT_NAME = "Hybrid"
AGENT_STRATEGY = "Domain specialist, tech/science focus, balanced approach"
EXPECTED_TRAINING_TIME_HOURS = 60
EXPECTED_COST_USD = 120  # 60 hours * $1.99/hr (faster training)
