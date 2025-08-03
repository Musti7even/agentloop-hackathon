import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass
import wandb

from email_generator import EmailGenerator
from utils import EmailScorer, ValidationEmailScorer, compute_group_advantages
from config import TrainingConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingBatch:
    """Data structure for a training batch."""
    personas: List[str]
    emails: List[List[str]]
    scores: List[List[float]]
    advantages: List[List[float]]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

class EmailDataset(Dataset):
    """Dataset for email training data."""
    
    def __init__(
        self,
        personas: List[str],
        emails: List[List[str]],
        advantages: List[List[float]],
        tokenizer,
        max_length: int = 512
    ):
        self.personas = personas
        self.emails = emails
        self.advantages = advantages
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Flatten data for indexing
        self.flat_data = []
        for p_idx, (persona, persona_emails, persona_advantages) in enumerate(
            zip(personas, emails, advantages)
        ):
            for e_idx, (email, advantage) in enumerate(zip(persona_emails, persona_advantages)):
                self.flat_data.append({
                    'persona_idx': p_idx,
                    'email_idx': e_idx,
                    'persona': persona,
                    'email': email,
                    'advantage': advantage
                })
    
    def __len__(self):
        return len(self.flat_data)
    
    def __getitem__(self, idx):
        item = self.flat_data[idx]
        
        # Create prompt
        prompt = f"Write a professional cold outreach email for the following persona:\n\nPersona: {item['persona']}\n\nEmail:\nSubject: "
        full_text = prompt + item['email']
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (only compute loss on generated part)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        labels = encoding['input_ids'].clone()
        # Mask prompt tokens
        prompt_length = prompt_encoding['input_ids'].shape[1]
        labels[:, :prompt_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'advantage': torch.tensor(item['advantage'], dtype=torch.float32),
            'persona_idx': item['persona_idx'],
            'email_idx': item['email_idx']
        }

class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for email generation.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        generator: EmailGenerator,
        scorer: EmailScorer
    ):
        self.config = config
        self.generator = generator
        self.scorer = scorer
        self.validation_scorer = ValidationEmailScorer(max_concurrent=5)
        
        # Initialize model and optimizer
        self.model = generator.model
        self.tokenizer = generator.tokenizer
        self.use_qlora = generator.use_qlora
        
        # Make model trainable
        self.model.train()
        
        # For QLoRA, only LoRA parameters are trainable by default
        if not self.use_qlora:
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Setup optimizer - only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Log parameter counts
        if self.use_qlora:
            trainable_count = sum(p.numel() for p in trainable_params)
            total_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"QLoRA: Training {trainable_count:,} / {total_count:,} parameters ({100 * trainable_count / total_count:.2f}%)")
        else:
            logger.info(f"Full fine-tuning: Training {sum(p.numel() for p in trainable_params):,} parameters")
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_score = float('-inf')
        self.patience_counter = 0
        
        # Initialize wandb if configured
        if config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
    
    def train_epoch(
        self,
        personas: List[str],
        val_personas: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            personas: Training personas
            val_personas: Validation personas
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting epoch {self.current_epoch + 1}")
        epoch_metrics = {'train_loss': 0.0, 'train_score': 0.0}
        
        # Generate emails for training
        logger.info("Generating training emails...")
        train_emails = self.generator.generate_emails(
            personas,
            num_emails_per_persona=self.config.num_emails_per_persona,
            batch_size=self.config.generation_batch_size
        )
        
        # Log reduced training email summary (debug level only)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Generated training emails summary:")
            for i, (persona, persona_emails) in enumerate(zip(personas, train_emails)):
                logger.debug(f"Persona {i+1}: {len(persona_emails)} emails, first: {persona_emails[0][:100]}...")
        else:
            logger.info(f"Generated {sum(len(emails) for emails in train_emails)} training emails for {len(personas)} personas")
        
        # Score emails
        logger.info("Scoring training emails...")
        train_scores = self.scorer.score_emails_batch(personas, train_emails)
        
        # Extract numeric scores
        numeric_scores = []
        for persona_results in train_scores:
            persona_scores = [result['score'] for result in persona_results]
            numeric_scores.append(persona_scores)
        
        # Compute advantages
        advantages = compute_group_advantages(
            numeric_scores,
            self.config.group_size,
            self.config.advantage_normalization
        )
        
        # Create dataset and dataloader
        dataset = EmailDataset(
            personas, train_emails, advantages,
            self.tokenizer, self.config.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size // self.config.gradient_accumulation_steps,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers
        )
        
        # Setup learning rate scheduler
        total_steps = len(dataloader) * self.config.ppo_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop with PPO-style updates
        total_loss = 0.0
        total_score = 0.0
        num_batches = 0
        
        for ppo_epoch in range(self.config.ppo_epochs):
            logger.debug(f"PPO epoch {ppo_epoch + 1}/{self.config.ppo_epochs}")  # Reduced to debug level
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                advantages = batch['advantage'].to(self.model.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Compute policy loss with advantages
                loss = outputs.loss
                
                # Weight loss by advantages (GRPO core idea)
                # Positive advantages increase loss (encourage), negative decrease (discourage)
                advantage_weights = torch.exp(advantages * self.config.reward_scaling)
                weighted_loss = loss * advantage_weights.mean()
                
                # Apply gradient accumulation
                weighted_loss = weighted_loss / self.config.gradient_accumulation_steps
                weighted_loss.backward()
                
                total_loss += weighted_loss.item()
                total_score += advantages.mean().item()
                num_batches += 1
                
                # Gradient step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Calculate gradient norm before clipping for monitoring
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Monitor for gradient explosion or NaN
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.error(f"Invalid gradient norm detected: {grad_norm}. Skipping optimizer step.")
                        self.optimizer.zero_grad()
                        continue
                    
                    if grad_norm > self.config.max_grad_norm * 2:
                        logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")
                    
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.current_step += 1
                    
                    # Reduced logging frequency
                    if self.current_step % (self.config.logging_steps * 2) == 0:  # Log half as often
                        avg_loss = total_loss / num_batches
                        avg_score = total_score / num_batches
                        
                        logger.info(
                            f"Step {self.current_step}: loss={avg_loss:.4f}, "
                            f"avg_advantage={avg_score:.4f}"
                        )
                        
                        if self.config.wandb_project:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/avg_advantage': avg_score,
                                'train/learning_rate': scheduler.get_last_lr()[0],
                                'step': self.current_step
                            })
        
        # Compute epoch metrics
        epoch_metrics['train_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_metrics['train_score'] = total_score / num_batches if num_batches > 0 else 0.0
        
        # Validation
        if val_personas:
            val_metrics = self.evaluate(val_personas)
            epoch_metrics.update(val_metrics)
        
        self.current_epoch += 1
        return epoch_metrics
    
    def evaluate(self, personas: List[str]) -> Dict[str, float]:
        """
        Evaluate on validation personas.
        
        Args:
            personas: Validation personas
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info("Running validation...")
        self.model.eval()
        
        with torch.no_grad():
            # Generate validation emails
            val_emails = self.generator.generate_emails(
                personas,
                num_emails_per_persona=self.config.num_emails_per_persona // 2,
                batch_size=self.config.generation_batch_size
            )
            
            # Log reduced validation email summary (debug level only)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Generated validation emails summary:")
                for i, (persona, persona_emails) in enumerate(zip(personas, val_emails)):
                    logger.debug(f"Val Persona {i+1}: {len(persona_emails)} emails, first: {persona_emails[0][:100]}...")
            else:
                logger.info(f"Generated {sum(len(emails) for emails in val_emails)} validation emails for {len(personas)} personas")
            
            # Score validation emails using both scorers
            val_scores = self.scorer.score_emails_batch(personas, val_emails)
            val_binary_scores = self.validation_scorer.score_emails_batch(personas, val_emails)
            
            # Compute score-based metrics
            all_scores = []
            for persona_results in val_scores:
                all_scores.extend([result['score'] for result in persona_results])
            
            # Compute binary decision metrics
            all_decisions = []
            for persona_results in val_binary_scores:
                all_decisions.extend([result['decision'] for result in persona_results])
            
            # Score distribution metrics
            val_metrics = {
                'val_score_mean': np.mean(all_scores),
                'val_score_std': np.std(all_scores),
                'val_score_max': np.max(all_scores),
                'val_score_min': np.min(all_scores)
            }
            
            # Binary accuracy metrics
            decisions_array = np.array(all_decisions)
            val_metrics.update({
                'val_binary_accuracy': np.mean(decisions_array),  # Main accuracy metric: % of emails that would get responses
                'val_response_rate': np.mean(decisions_array),    # Same as accuracy but clearer naming
                'val_response_count': np.sum(decisions_array),    # Number of emails that would get responses
                'val_no_response_count': np.sum(~decisions_array) # Number of emails that would not get responses
            })
            
            # Accuracy metrics based on score thresholds (for comparison)
            scores_array = np.array(all_scores)
            
            # Define quality thresholds based on decision_maker_score.py ranges
            high_quality_threshold = 0.7  # "Likely to respond" threshold
            medium_quality_threshold = 0.5  # "Might respond" threshold
            low_quality_threshold = 0.3   # "Unlikely to respond" threshold
            
            # Calculate threshold-based accuracy metrics
            val_metrics.update({
                'val_threshold_accuracy_high': np.mean(scores_array >= high_quality_threshold),
                'val_threshold_accuracy_medium': np.mean(scores_array >= medium_quality_threshold),
                'val_threshold_accuracy_low': np.mean(scores_array >= low_quality_threshold),
                'val_threshold_reject_spam': np.mean(scores_array < low_quality_threshold),
                
                # Distribution by quality bands
                'val_pct_excellent': np.mean(scores_array >= 0.9),  # Definitely will respond
                'val_pct_good': np.mean((scores_array >= 0.7) & (scores_array < 0.9)),  # Likely to respond
                'val_pct_maybe': np.mean((scores_array >= 0.5) & (scores_array < 0.7)),  # Might respond
                'val_pct_poor': np.mean((scores_array >= 0.3) & (scores_array < 0.5)),  # Unlikely to respond
                'val_pct_spam': np.mean(scores_array < 0.3),  # Definitely won't respond
                
                # Email count metrics
                'val_total_emails': len(all_scores),
                'val_high_quality_count': np.sum(scores_array >= high_quality_threshold),
                'val_spam_count': np.sum(scores_array < low_quality_threshold)
            })
        
        self.model.train()
        
        # Log validation metrics
        if self.config.wandb_project:
            wandb.log({f'val/{k.replace("val_", "")}': v for k, v in val_metrics.items()})
        
        return val_metrics
    
    def should_stop_early(self, val_score: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_score: Current validation score
            
        Returns:
            True if should stop early
        """
        if val_score > self.best_score + self.config.min_delta:
            self.best_score = val_score
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def save_checkpoint(self, checkpoint_dir: str, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            is_best: Whether this is the best checkpoint
        """
        import os
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        suffix = "_best" if is_best else ""
        
        if self.use_qlora:
            # Save LoRA adapters separately
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                lora_path = os.path.join(checkpoint_dir, f"lora_adapters{suffix}")
                self.model.save_pretrained(lora_path)
                logger.info(f"Saved LoRA adapters to {lora_path}")
                
                # Also save training state
                state_dict = {
                    'epoch': self.current_epoch,
                    'step': self.current_step,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_score': self.best_score
                }
                state_path = os.path.join(checkpoint_dir, f"training_state{suffix}.pt")
                torch.save(state_dict, state_path)
                logger.info(f"Saved training state to {state_path}")
                
                return lora_path
        else:
            # Full model checkpoint
            from utils import save_checkpoint
            checkpoint_path = save_checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                self.current_step,
                0.0,  # We don't track single loss value
                checkpoint_dir,
                f"grpo_checkpoint{suffix}"
            )
            return checkpoint_path