#!/usr/bin/env python3
"""
Main training script for GRPO email model fine-tuning.
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import set_seed

from config import TrainingConfig, PersonaConfig
from email_generator import EmailGenerator
from grpo_trainer import GRPOTrainer
from utils import EmailScorer, load_personas, setup_logging, count_parameters

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train email model with GRPO")
    
    # Data arguments
    parser.add_argument(
        "--personas_file",
        type=str,
        required=True,
        help="Path to JSON file containing personas"
    )
    parser.add_argument(
        "--val_personas_file",
        type=str,
        help="Path to validation personas file (optional)"
    )
    parser.add_argument(
        "--max_personas",
        type=int,
        help="Maximum number of personas to use"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grpo_checkpoints",
        help="Output directory for checkpoints"
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_emails_per_persona",
        type=int,
        default=8,
        help="Number of emails to generate per persona"
    )
    
    # Logging arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training"
    )
    
    # Resuming
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    # Config file
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to JSON config file (overrides other args)"
    )
    
    return parser.parse_args()

def load_config_from_file(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def create_config(args) -> TrainingConfig:
    """Create training configuration from arguments."""
    config_dict = {}
    
    # Load from config file if provided
    if args.config_file:
        config_dict.update(load_config_from_file(args.config_file))
    
    # Override with command line arguments
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and key != 'config_file':
            config_dict[key] = value
    
    return TrainingConfig(**{k: v for k, v in config_dict.items() 
                           if k in TrainingConfig.__dataclass_fields__})

def split_personas(personas: List[str], val_split: float = 0.1, seed: int = 42) -> tuple:
    """Split personas into train and validation sets."""
    if val_split <= 0:
        return personas, []
    
    import random
    random.seed(seed)
    
    shuffled = personas.copy()
    random.shuffle(shuffled)
    
    val_size = int(len(personas) * val_split)
    val_personas = shuffled[:val_size]
    train_personas = shuffled[val_size:]
    
    return train_personas, val_personas

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting GRPO email model training")
    
    # Create configuration
    config = create_config(args)
    
    # Ensure output_dir is not an absolute path starting with /
    if config.output_dir.startswith('/') and not config.output_dir.startswith('/Users/'):
        logger.warning(f"Absolute path detected: {config.output_dir}, converting to relative path")
        config.output_dir = f".{config.output_dir}"
    
    logger.info(f"Training configuration output_dir: {config.output_dir}")
    logger.info(f"Full config: {config}")
    
    # Set random seed
    set_seed(42)
    
    # Load personas
    logger.info(f"Loading personas from {args.personas_file}")
    all_personas = load_personas(args.personas_file, config.max_personas)
    
    # Split or load validation personas
    if args.val_personas_file:
        logger.info(f"Loading validation personas from {args.val_personas_file}")
        train_personas = all_personas
        val_personas = load_personas(args.val_personas_file)
    else:
        logger.info("Splitting personas into train/val sets")
        train_personas, val_personas = split_personas(
            all_personas, val_split=0.1, seed=42
        )
    
    logger.info(f"Training personas: {len(train_personas)}")
    logger.info(f"Validation personas: {len(val_personas)}")
    
    # Initialize components
    logger.info("Initializing email generator...")
    
    # Prepare LoRA config if using QLoRA
    lora_config = None
    if config.use_qlora:
        lora_config = {
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "bias": config.lora_bias,
            "target_modules": config.target_modules
        }
    
    generator = EmailGenerator(
        model_name=config.model_name,
        device=config.device,
        max_length=config.max_length,
        temperature=config.temperature,
        top_p=config.top_p,
        use_qlora=config.use_qlora,
        lora_config=lora_config
    )
    
    logger.info("Initializing email scorer...")
    scorer = EmailScorer(
        max_concurrent=config.scorer_max_concurrent,
        timeout=config.scorer_timeout
    )
    
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(config, generator, scorer)
    
    # Log model info
    param_count = count_parameters(trainer.model)
    logger.info(f"Model has {param_count:,} trainable parameters")
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        from utils import load_checkpoint
        checkpoint_info = load_checkpoint(
            args.resume_from_checkpoint,
            trainer.model,
            trainer.optimizer,
            is_lora=config.use_qlora
        )
        trainer.current_epoch = checkpoint_info['epoch']
        trainer.current_step = checkpoint_info['step']
        if 'best_score' in checkpoint_info:
            trainer.best_score = checkpoint_info['best_score']
    
    # Training loop
    logger.info("Starting training loop...")
    os.makedirs(config.output_dir, exist_ok=True)
    
    best_val_score = float('-inf')
    
    try:
        for epoch in range(trainer.current_epoch, config.num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train epoch
            metrics = trainer.train_epoch(train_personas, val_personas)
            
            # Log metrics
            logger.info("Epoch metrics:")
            
            # Log basic metrics first
            basic_metrics = ['train_loss', 'train_score', 'val_score_mean', 'val_score_std', 'val_score_max', 'val_score_min']
            for key in basic_metrics:
                if key in metrics:
                    logger.info(f"  {key}: {metrics[key]:.4f}")
            
            # Log accuracy metrics if available
            if any(k.startswith('val_accuracy_') for k in metrics.keys()):
                logger.info("  Validation Accuracy Metrics:")
                accuracy_keys = [k for k in metrics.keys() if k.startswith('val_accuracy_')]
                for key in accuracy_keys:
                    logger.info(f"    {key}: {metrics[key]:.3f} ({metrics[key]*100:.1f}%)")
            
            # Log quality distribution if available
            if any(k.startswith('val_pct_') for k in metrics.keys()):
                logger.info("  Email Quality Distribution:")
                quality_keys = ['val_pct_excellent', 'val_pct_good', 'val_pct_maybe', 'val_pct_poor', 'val_pct_spam']
                for key in quality_keys:
                    if key in metrics:
                        logger.info(f"    {key}: {metrics[key]:.3f} ({metrics[key]*100:.1f}%)")
            
            # Log email counts if available
            if 'val_total_emails' in metrics:
                logger.info(f"  Total validation emails: {int(metrics['val_total_emails'])}")
                if 'val_high_quality_count' in metrics:
                    logger.info(f"  High quality emails (≥0.7): {int(metrics['val_high_quality_count'])}")
                if 'val_spam_count' in metrics:
                    logger.info(f"  Low quality emails (<0.3): {int(metrics['val_spam_count'])}")
            
            # Log any remaining metrics
            remaining_keys = [k for k in metrics.keys() if not any(k.startswith(prefix) for prefix in 
                            ['train_', 'val_score_', 'val_accuracy_', 'val_pct_', 'val_total_', 'val_high_quality_', 'val_spam_'])]
            for key in remaining_keys:
                logger.info(f"  {key}: {metrics[key]:.4f}")
            
            # Save checkpoint
            is_best = False
            if 'val_score_mean' in metrics:
                val_score = metrics['val_score_mean']
                if val_score > best_val_score:
                    best_val_score = val_score
                    is_best = True
                    logger.info(f"New best validation score: {val_score:.4f}")
            
            checkpoint_path = trainer.save_checkpoint(
                config.output_dir, 
                is_best=is_best
            )
            
            # Early stopping check
            if 'val_score_mean' in metrics:
                if trainer.should_stop_early(metrics['val_score_mean']):
                    logger.info("Early stopping triggered")
                    break
            
            # Save regular checkpoint every few epochs
            if (epoch + 1) % 2 == 0:
                trainer.save_checkpoint(config.output_dir, is_best=False)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(config.output_dir, is_best=False)
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        logger.info("Training completed")
        
        # Save final model
        final_path = os.path.join(config.output_dir, "final_model")
        
        if config.use_qlora:
            # Save LoRA adapters
            trainer.model.save_pretrained(final_path)
            
            # Also save merged model for easy inference
            from utils import merge_lora_adapters
            merged_path = os.path.join(config.output_dir, "final_model_merged")
            merge_lora_adapters(trainer.model, merged_path)
            trainer.tokenizer.save_pretrained(merged_path)
            logger.info(f"Final LoRA adapters saved to {final_path}")
            logger.info(f"Final merged model saved to {merged_path}")
        else:
            # Full model
            trainer.model.save_pretrained(final_path)
            trainer.tokenizer.save_pretrained(final_path)
            logger.info(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()