from dataclasses import dataclass
from typing import Optional, List
import multiprocessing as mp

@dataclass
class TrainingConfig:
    """Configuration for GRPO email training."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-0.6B"
    max_length: int = 4096
    
    # Generation settings
    temperature: float = 0.7  # Reduced from 0.8 for better numerical stability
    top_p: float = 0.9
    num_emails_per_persona: int = 8
    generation_batch_size: int = 16  # Increased from 4 for better GPU utilization
    
    # GRPO training settings
    learning_rate: float = 1e-5
    batch_size: int = 32  # Increased from 16 for faster training
    num_epochs: int = 10
    gradient_accumulation_steps: int = 2  # Reduced from 4 to match larger batch size
    max_grad_norm: float = 1.0
    
    # PPO-style settings for GRPO
    ppo_epochs: int = 2  # Reduced from 4 for faster convergence
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Group relative settings
    group_size: int = 8  # Number of emails per group for relative ranking
    advantage_normalization: bool = True
    reward_scaling: float = 1.0
    
    # Training loop settings
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    warmup_steps: int = 100
    
    # Data settings
    train_personas_file: Optional[str] = None
    val_personas_file: Optional[str] = None
    max_personas: Optional[int] = None
    
    # Paths
    output_dir: str = "./grpo_checkpoints"
    cache_dir: str = "./cache"
    
    # QLoRA settings (auto-disabled on MPS)
    use_qlora: bool = True
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (scaling)
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None  # Auto-detect if None
    lora_bias: str = "none"  # "none", "all", or "lora_only"
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"  # "float16" or "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"  # "fp4" or "nf4"
    
    # Hardware settings
    device: Optional[str] = None  # Auto-detect if None
    mixed_precision: bool = True
    dataloader_num_workers: int = min(8, mp.cpu_count())  # Better CPU utilization
    
    # Scoring settings
    scorer_max_concurrent: int = 20  # Increased from 5 for faster scoring
    scorer_timeout: float = 30.0
    
    # Logging
    wandb_project: Optional[str] = "grpo-email-training"
    wandb_run_name: Optional[str] = None
    log_level: str = "INFO"
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_emails_per_persona % self.group_size != 0:
            raise ValueError(
                f"num_emails_per_persona ({self.num_emails_per_persona}) must be "
                f"divisible by group_size ({self.group_size})"
            )
        
        if self.batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by "
                f"gradient_accumulation_steps ({self.gradient_accumulation_steps})"
            )

@dataclass 
class PersonaConfig:
    """Configuration for persona data loading."""
    
    # Persona fields
    persona_field: str = "persona"
    context_field: Optional[str] = None
    
    # Data processing
    min_persona_length: int = 50
    max_persona_length: int = 2000
    shuffle_personas: bool = True
    
    # Validation split
    val_split: float = 0.1
    random_seed: int = 42