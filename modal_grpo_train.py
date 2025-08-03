"""
Train the GRPO email model on Modal GPUs with full restartability 
(resume from checkpoints stored in Modal volume).

Launch locally:
    modal deploy modal_grpo_train.py       # one-off build
    modal run modal_grpo_train.py::train_grpo_email_model \
        --personas_file data/personas/b2b_saas_personas_massive_train.json \
        --val_personas_file data/personas/b2b_saas_personas_massive_eval.json \
        --num_epochs 10 --batch_size 16 --learning_rate 1e-5
"""

from pathlib import Path
import modal
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1. Docker image with ML dependencies
# ──────────────────────────────────────────────────────────────────────────────
ML_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    # Create directory first
    .run_commands("mkdir -p /root/data_gen")
    .pip_install(
        # Core ML dependencies
        "torch>=2.0.0",
        "transformers>=4.51.0", 
        "tokenizers>=0.15.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        # QLoRA dependencies
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        # Training dependencies
        "numpy>=1.24.0",
        "scipy>=1.10.0", 
        "scikit-learn>=1.3.0",
        # Logging and tracking
        "wandb>=0.16.0",
        "tensorboard>=2.15.0",
        # API clients
        "anthropic>=0.8.0",
        # Utilities
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        gpu="H100",
    )
    # Copy GRPO training modules (keep these last)
    .add_local_file("grpo_training/config.py", remote_path="/root/config.py")
    .add_local_file("grpo_training/grpo_trainer.py", remote_path="/root/grpo_trainer.py")
    .add_local_file("grpo_training/email_generator.py", remote_path="/root/email_generator.py")
    .add_local_file("grpo_training/utils.py", remote_path="/root/utils.py")
    .add_local_file("data/personas/b2b_saas_personas_massive_train.json", remote_path="/root/b2b_saas_personas_massive_train.json")
    .add_local_file("data/personas/b2b_saas_personas_massive_eval.json", remote_path="/root/b2b_saas_personas_massive_eval.json")
    # Copy decision maker modules
    .add_local_file("data_gen/decision_maker_score.py", remote_path="/root/data_gen/decision_maker_score.py")
    .add_local_file("data_gen/decision_maker.py", remote_path="/root/data_gen/decision_maker.py")
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Modal app & shared volume
# ──────────────────────────────────────────────────────────────────────────────
app = modal.App(
    "grpo-email-training",
    image=ML_IMAGE,
    secrets=[
        modal.Secret.from_dotenv()
    ],
)

VOL = Path("/vol")
outputs = modal.Volume.from_name("grpo-training-vol", create_if_missing=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Training entry-point
# ──────────────────────────────────────────────────────────────────────────────
@app.function(
    gpu="H100",
    timeout=60 * 60 * 12,                       # 12 hours
    memory=100_000,                             # 100GB RAM
    volumes={VOL: outputs},
    retries=modal.Retries(initial_delay=0.0, max_retries=1),
)
def train_grpo_email_model(
    personas_file: str = "/root/b2b_saas_personas_massive_train.json",
    val_personas_file: str = "/root/b2b_saas_personas_massive_eval.json",
    model_name: str = "Qwen/Qwen3-0.6B", 
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    num_emails_per_persona: int = 8,
    max_personas: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    use_qlora: bool = True,
):
    """
    Train GRPO email model and save checkpoints to Modal volume.
    """
    # ── 0. Setup environment ───────────────────────────────────────────────
    import os
    import logging
    from datetime import datetime
    
    # Setup W&B environment 
    if "WANDB_API_KEY" in os.environ:
        os.environ["WANDB__SERVICE_WAIT"] = "300"

    # ── 1. Heavy imports (inside container) ─────────────────────────────────
    import torch
    import wandb
    from transformers import set_seed
    
    from config import TrainingConfig
    from grpo_trainer import GRPOTrainer
    from email_generator import EmailGenerator
    from utils import EmailScorer, load_personas, setup_logging, count_parameters

    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting GRPO email model training on Modal")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 2. Checkpoint helpers ──────────────────────────────────────────────
    CKPT_DIR = VOL / "checkpoints"
    
    if wandb_run_name:
        RUN_NAME = wandb_run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_NAME = f"grpo_{model_name.replace('/', '_')}_ep{num_epochs}_bs{batch_size}_lr{learning_rate:.0e}_{timestamp}"
    
    RUN_DIR = CKPT_DIR / RUN_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    def latest_ckpt() -> Path | None:
        """Find the latest checkpoint."""
        ckpts = sorted(RUN_DIR.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        return ckpts[-1] if ckpts else None

    def save_ckpt(path, *, trainer, epoch, step):
        """Save training checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_score': getattr(trainer, 'best_score', float('-inf')),
            'config': trainer.config.__dict__,
        }
        
        if hasattr(trainer, 'scheduler') and trainer.scheduler:
            checkpoint_data['scheduler_state_dict'] = trainer.scheduler.state_dict()
            
        torch.save(checkpoint_data, path)
        logger.info(f"   ↳ Saved checkpoint: {path}")

    def load_ckpt(path, trainer):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(trainer, 'scheduler') and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        trainer.current_epoch = checkpoint['epoch']
        trainer.current_step = checkpoint['step']
        trainer.best_score = checkpoint.get('best_score', float('-inf'))
        
        logger.info(f"   ↳ Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        return checkpoint['epoch'], checkpoint['step']

    # ── 3. Load persona data ───────────────────────────────────────────────
    logger.info(f"Loading personas from {personas_file}")
    
    # Check if personas file exists in volume first, otherwise use local path
    vol_personas_path = VOL / personas_file
    if vol_personas_path.exists():
        personas_path = str(vol_personas_path)
    else:
        # Copy from local to volume
        import shutil
        vol_personas_path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(personas_file):
            shutil.copy2(personas_file, vol_personas_path)
            personas_path = str(vol_personas_path)
        else:
            raise FileNotFoundError(f"Personas file not found: {personas_file}")
    
    all_personas = load_personas(personas_path, max_personas)
    
    # Load validation personas from separate file
    vol_val_path = VOL / val_personas_file
    if not vol_val_path.exists() and os.path.exists(val_personas_file):
        vol_val_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(val_personas_file, vol_val_path)
    
    val_personas = load_personas(str(vol_val_path))
    train_personas = all_personas  # Use all training personas from the training file
    
    logger.info(f"Training personas: {len(train_personas)}")
    logger.info(f"Validation personas: {len(val_personas)}")

    # ── 4. Create training configuration ───────────────────────────────────
    config = TrainingConfig(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_emails_per_persona=num_emails_per_persona,
        use_qlora=use_qlora,
        output_dir=str(RUN_DIR),
        wandb_project="grpo-email-training-modal",
        wandb_run_name=RUN_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── 5. Initialize components ───────────────────────────────────────────
    logger.info("Initializing email generator...")
    
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

    # ── 6. Initialize W&B ──────────────────────────────────────────────────
    run = wandb.init(
        project=config.wandb_project,
        name=RUN_NAME,
        config=config.__dict__,
        resume="allow",
    )

    # ── 7. Resume from checkpoint if specified ─────────────────────────────
    start_epoch = 0
    if resume_checkpoint:
        ckpt_path = RUN_DIR / resume_checkpoint if not resume_checkpoint.startswith('/') else Path(resume_checkpoint)
        if ckpt_path.exists():
            logger.info(f"🔁 Resuming from checkpoint: {ckpt_path}")
            start_epoch, _ = load_ckpt(ckpt_path, trainer)
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")
    else:
        # Check for latest checkpoint
        latest = latest_ckpt()
        if latest:
            logger.info(f"🔁 Resuming from latest checkpoint: {latest}")
            start_epoch, _ = load_ckpt(latest, trainer)

    # ── 8. Training loop ───────────────────────────────────────────────────
    logger.info("Starting training loop...")
    best_val_score = getattr(trainer, 'best_score', float('-inf'))
    
    try:
        set_seed(42)
        
        for epoch in range(start_epoch, config.num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train epoch
            metrics = trainer.train_epoch(train_personas, val_personas)
            
            # Log metrics
            logger.info("Epoch metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
                    
            # Log to wandb
            wandb.log({"epoch": epoch + 1, **metrics})
            
            # Save checkpoint
            is_best = False
            if 'val_score_mean' in metrics:
                val_score = metrics['val_score_mean']
                if val_score > best_val_score:
                    best_val_score = val_score
                    trainer.best_score = best_val_score
                    is_best = True
                    logger.info(f"🎉 New best validation score: {val_score:.4f}")
            
            # Save checkpoint
            ckpt_name = f"checkpoint_step_{trainer.current_step}.pt"
            if is_best:
                ckpt_name = f"best_{ckpt_name}"
                
            save_ckpt(
                RUN_DIR / ckpt_name,
                trainer=trainer,
                epoch=epoch + 1,
                step=trainer.current_step
            )
            
            # Early stopping check
            if hasattr(trainer, 'should_stop_early') and 'val_score_mean' in metrics:
                if trainer.should_stop_early(metrics['val_score_mean']):
                    logger.info("⏹️ Early stopping triggered")
                    break
    
    except KeyboardInterrupt:
        logger.info("⏸️ Training interrupted by user")
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        raise
    finally:
        # Save final checkpoint
        final_ckpt_path = RUN_DIR / "final_checkpoint.pt"
        save_ckpt(
            final_ckpt_path,
            trainer=trainer,
            epoch=trainer.current_epoch,
            step=trainer.current_step
        )
        
        # Save final model
        if config.use_qlora:
            final_model_path = RUN_DIR / "final_model"
            trainer.model.save_pretrained(final_model_path)
            trainer.tokenizer.save_pretrained(final_model_path)
            logger.info(f"💾 Final LoRA model saved to {final_model_path}")
        else:
            final_model_path = RUN_DIR / "final_model"
            trainer.model.save_pretrained(final_model_path)
            trainer.tokenizer.save_pretrained(final_model_path)
            logger.info(f"💾 Final model saved to {final_model_path}")
        
        run.finish()
        logger.info(f"🏁 Training completed! Best validation score: {best_val_score:.4f}")

    # ── 9. Commit volume (Modal keeps snapshots) ───────────────────────────
    outputs.commit()
    logger.info("💾 Checkpoints and models saved to Modal volume")
    
    return {
        "best_score": best_val_score,
        "final_epoch": trainer.current_epoch,
        "total_steps": trainer.current_step,
        "model_path": str(final_model_path),
        "run_name": RUN_NAME
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Helper function to list checkpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.function(volumes={VOL: outputs})
def list_checkpoints(run_name: Optional[str] = None):
    """List available checkpoints in the volume."""
    ckpt_dir = VOL / "checkpoints"
    
    if run_name:
        run_dir = ckpt_dir / run_name
        if run_dir.exists():
            checkpoints = list(run_dir.glob("*.pt"))
            print(f"Checkpoints for run '{run_name}':")
            for ckpt in sorted(checkpoints):
                print(f"  {ckpt.name}")
        else:
            print(f"Run '{run_name}' not found")
    else:
        if ckpt_dir.exists():
            runs = [d.name for d in ckpt_dir.iterdir() if d.is_dir()]
            print("Available training runs:")
            for run in sorted(runs):
                print(f"  {run}")
        else:
            print("No checkpoints found")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Helper function to download checkpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.function(volumes={VOL: outputs})
def download_checkpoint(run_name: str, checkpoint_name: str = "best_checkpoint_step_*.pt"):
    """Download a specific checkpoint from Modal volume."""
    import glob
    
    run_dir = VOL / "checkpoints" / run_name
    
    if not run_dir.exists():
        print(f"Run '{run_name}' not found")
        return
        
    # Find matching checkpoints
    pattern = str(run_dir / checkpoint_name)
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"No checkpoints matching '{checkpoint_name}' found in run '{run_name}'")
        return
        
    # Return the latest/best checkpoint
    checkpoint_path = Path(sorted(matching_files)[-1])  
    
    print(f"Downloading checkpoint: {checkpoint_path.name}")
    
    # In a real implementation, this would copy the file to a download location
    # For now, just return the path
    return str(checkpoint_path)