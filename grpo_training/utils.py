import json
import logging
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import time
from asyncio import Queue, create_task, gather
from functools import lru_cache
import threading

# Add parent directory to path to import decision makers
sys.path.append(str(Path(__file__).parent.parent))
from data_gen.decision_maker_score import decide_response
from data_gen.decision_maker import decide_response as decide_binary_response

logger = logging.getLogger(__name__)

class EmailScorer:
    """
    Email scorer using the decision_maker_score module.
    Handles concurrent scoring with rate limiting and caching.
    """
    
    def __init__(self, max_concurrent: int = 20, timeout: float = 30.0):  # Increased from 10 to 20
        """
        Initialize email scorer.
        
        Args:
            max_concurrent: Maximum concurrent scoring requests
            timeout: Timeout for individual scoring requests
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._score_cache = {}  # Simple cache for repeated persona-email pairs
        self._cache_lock = threading.Lock()
        
    def _create_cache_key(self, persona: str, email: str) -> str:
        """Create a cache key for persona-email pairs."""
        # Use hash of both to create a reasonable cache key
        return f"{hash(persona)}_{hash(email)}"
    
    def score_email(self, persona: str, email: str) -> Dict[str, Any]:
        """
        Score a single email with caching.
        
        Args:
            persona: Persona description
            email: Email content
            
        Returns:
            Dictionary with score and reasoning
        """
        # Check cache first
        cache_key = self._create_cache_key(persona, email)
        with self._cache_lock:
            if cache_key in self._score_cache:
                return self._score_cache[cache_key].copy()
        
        try:
            result = decide_response(persona, email)
            score_result = {
                'score': float(result['score']),
                'reasoning': result['reasoning'],
                'success': True,
                'error': None
            }
            
            # Cache the result
            with self._cache_lock:
                if len(self._score_cache) < 1000:  # Limit cache size
                    self._score_cache[cache_key] = score_result.copy()
            
            return score_result
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            error_result = {
                'score': 0.0,
                'reasoning': f"Scoring failed: {str(e)}",
                'success': False,
                'error': str(e)
            }
            return error_result
    
    def score_emails_batch(
        self, 
        personas: List[str], 
        emails: List[List[str]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Score multiple emails for multiple personas concurrently with improved efficiency.
        
        Args:
            personas: List of persona descriptions
            emails: List of lists of emails (one list per persona)
            
        Returns:
            List of lists of scoring results
        """
        # Pre-initialize results structure for better performance
        results = []
        for persona_emails in emails:
            results.append([{} for _ in persona_emails])
        
        # Create mapping for tasks
        task_mapping = []
        futures = []
        
        logger.info(f"Starting batch scoring for {len(personas)} personas...")
        start_time = time.time()
        
        # Submit all tasks efficiently
        for p_idx, (persona, persona_emails) in enumerate(zip(personas, emails)):
            for e_idx, email in enumerate(persona_emails):
                future = self.executor.submit(self.score_email, persona, email)
                futures.append(future)
                task_mapping.append((p_idx, e_idx))
        
        logger.info(f"Submitted {len(futures)} scoring tasks")
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(futures, timeout=self.timeout):
            try:
                # Find which task this is
                future_idx = futures.index(future)
                p_idx, e_idx = task_mapping[future_idx]
                
                # Get result and store it
                result = future.result(timeout=5.0)  # Short timeout for completed futures
                results[p_idx][e_idx] = result
                
                completed_count += 1
                if completed_count % 20 == 0:  # Log progress every 20 completions
                    logger.debug(f"Completed {completed_count}/{len(futures)} scoring tasks")
                    
            except Exception as e:
                # Find which task failed
                future_idx = futures.index(future)
                p_idx, e_idx = task_mapping[future_idx]
                
                logger.error(f"Scoring task failed for persona {p_idx}, email {e_idx}: {e}")
                results[p_idx][e_idx] = {
                    'score': 0.0,
                    'reasoning': f"Timeout or error: {str(e)}",
                    'success': False,
                    'error': str(e)
                }
        
        elapsed = time.time() - start_time
        successful_scores = sum(1 for persona_results in results for result in persona_results if result.get('success', False))
        logger.info(f"Completed batch scoring: {successful_scores}/{len(futures)} successful in {elapsed:.2f}s ({len(futures)/elapsed:.1f} scores/sec)")
        
        return results
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'cache_size': len(self._score_cache),
                'cache_max_size': 1000
            }
    
    def clear_cache(self):
        """Clear the scoring cache."""
        with self._cache_lock:
            self._score_cache.clear()
            logger.info("Cleared scoring cache")
    
    def __del__(self):
        """Clean up executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

class ValidationEmailScorer:
    """
    Email scorer using the binary decision_maker module for validation.
    Handles concurrent scoring with rate limiting and caching.
    """
    
    def __init__(self, max_concurrent: int = 15, timeout: float = 30.0):  # Increased from 10 to 15
        """
        Initialize validation email scorer.
        
        Args:
            max_concurrent: Maximum concurrent scoring requests
            timeout: Timeout for individual scoring requests
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._binary_cache = {}  # Cache for binary decisions
        self._cache_lock = threading.Lock()
        
    def _create_cache_key(self, persona: str, email: str) -> str:
        """Create a cache key for persona-email pairs."""
        return f"{hash(persona)}_{hash(email)}"
    
    def score_email(self, persona: str, email: str) -> Dict[str, Any]:
        """
        Score a single email using binary decision maker with caching.
        
        Args:
            persona: Persona description
            email: Email content
            
        Returns:
            Dictionary with decision and reasoning
        """
        # Check cache first
        cache_key = self._create_cache_key(persona, email)
        with self._cache_lock:
            if cache_key in self._binary_cache:
                return self._binary_cache[cache_key].copy()
        
        try:
            result = decide_binary_response(persona, email)
            decision_result = {
                'decision': bool(result['decision']),
                'reasoning': result['reasoning'],
                'success': True,
                'error': None
            }
            
            # Cache the result
            with self._cache_lock:
                if len(self._binary_cache) < 1000:  # Limit cache size
                    self._binary_cache[cache_key] = decision_result.copy()
            
            return decision_result
            
        except Exception as e:
            logger.error(f"Binary scoring failed: {e}")
            error_result = {
                'decision': False,
                'reasoning': f"Binary scoring failed: {str(e)}",
                'success': False,
                'error': str(e)
            }
            return error_result
    
    def score_emails_batch(
        self, 
        personas: List[str], 
        emails: List[List[str]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Score multiple emails for multiple personas concurrently using binary decisions.
        
        Args:
            personas: List of persona descriptions
            emails: List of lists of emails (one list per persona)
            
        Returns:
            List of lists of binary scoring results
        """
        all_tasks = []
        persona_indices = []
        email_indices = []
        
        # Create tasks for all persona-email pairs
        for p_idx, (persona, persona_emails) in enumerate(zip(personas, emails)):
            for e_idx, email in enumerate(persona_emails):
                future = self.executor.submit(self.score_email, persona, email)
                all_tasks.append(future)
                persona_indices.append(p_idx)
                email_indices.append(e_idx)
        
        # Collect results
        results = [[] for _ in personas]
        
        logger.info(f"Binary scoring {len(all_tasks)} emails...")
        start_time = time.time()
        
        for i, future in enumerate(as_completed(all_tasks, timeout=self.timeout)):
            try:
                result = future.result()
                p_idx = persona_indices[i]
                e_idx = email_indices[i]
                
                # Ensure we have the right number of sublists
                while len(results[p_idx]) <= e_idx:
                    results[p_idx].append({})
                    
                results[p_idx][e_idx] = result
                
            except Exception as e:
                logger.error(f"Binary scoring task {i} failed: {e}")
                p_idx = persona_indices[i]
                e_idx = email_indices[i]
                
                while len(results[p_idx]) <= e_idx:
                    results[p_idx].append({})
                    
                results[p_idx][e_idx] = {
                    'decision': False,
                    'reasoning': f"Timeout or error: {str(e)}",
                    'success': False,
                    'error': str(e)
                }
        
        elapsed = time.time() - start_time
        logger.info(f"Binary scored {len(all_tasks)} emails in {elapsed:.2f}s")
        
        return results
    
    def __del__(self):
        """Clean up executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

def load_personas(file_path: str, max_personas: Optional[int] = None) -> List[str]:
    """
    Load personas from a JSON file.
    
    Args:
        file_path: Path to JSON file with personas
        max_personas: Maximum number of personas to load
        
    Returns:
        List of persona strings
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Simple array format - each item should be a string
            persona_strings = []
            for item in data:
                if isinstance(item, str):
                    persona_strings.append(item)
                elif isinstance(item, dict):
                    # Try common persona fields
                    for field in ['persona', 'description', 'text', 'content']:
                        if field in item:
                            persona_strings.append(item[field])
                            break
                    else:
                        # Convert dict to string as fallback
                        persona_strings.append(str(item))
                else:
                    # Convert any other type to string
                    persona_strings.append(str(item))
                    
        elif isinstance(data, dict):
            # Try common keys for nested structures
            personas = None
            for key in ['personas', 'data', 'items']:
                if key in data:
                    personas = data[key]
                    break
            
            if personas is None:
                # If no common key, take first list value
                for value in data.values():
                    if isinstance(value, list):
                        personas = value
                        break
                else:
                    raise ValueError("No persona list found in JSON")
            
            # Extract persona strings
            persona_strings = []
            for persona in personas:
                if isinstance(persona, str):
                    persona_strings.append(persona)
                elif isinstance(persona, dict):
                    # Try common persona fields
                    for field in ['persona', 'description', 'text', 'content']:
                        if field in persona:
                            persona_strings.append(persona[field])
                            break
                    else:
                        # Convert dict to string as fallback
                        persona_strings.append(str(persona))
                else:
                    persona_strings.append(str(persona))
        else:
            raise ValueError("JSON must contain list or dict")
        
        # Filter out empty strings
        persona_strings = [p.strip() for p in persona_strings if p and p.strip()]
        
        if max_personas:
            persona_strings = persona_strings[:max_personas]
            
        logger.info(f"Loaded {len(persona_strings)} personas from {file_path}")
        return persona_strings
        
    except Exception as e:
        logger.error(f"Failed to load personas from {file_path}: {e}")
        raise

def compute_group_advantages(
    scores: List[List[float]], 
    group_size: int,
    normalize: bool = True
) -> List[List[float]]:
    """
    Compute group-relative advantages for GRPO.
    
    Args:
        scores: List of lists of scores (one list per persona)
        group_size: Size of each group for relative ranking
        normalize: Whether to normalize advantages
        
    Returns:
        List of lists of advantages
    """
    all_advantages = []
    
    for persona_scores in scores:
        persona_advantages = []
        
        # Process in groups
        for i in range(0, len(persona_scores), group_size):
            group_scores = persona_scores[i:i + group_size]
            
            if len(group_scores) < 2:
                # Single item group - zero advantage
                persona_advantages.extend([0.0] * len(group_scores))
                continue
            
            # Compute relative advantages within group
            group_mean = np.mean(group_scores)
            group_advantages = [score - group_mean for score in group_scores]
            
            # Normalize within group if requested
            if normalize and len(group_advantages) > 1:
                group_std = np.std(group_advantages)
                if group_std > 1e-8:
                    group_advantages = [adv / group_std for adv in group_advantages]
            
            persona_advantages.extend(group_advantages)
        
        all_advantages.append(persona_advantages)
    
    return all_advantages

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    prefix: str = "checkpoint"
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        step: Current step
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
        prefix: Checkpoint filename prefix
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"{prefix}_epoch_{epoch}_step_{step}.pt"
    )
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    return checkpoint_path

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    is_lora: bool = False
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file or LoRA adapter directory
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        is_lora: Whether loading LoRA adapters
        
    Returns:
        Checkpoint metadata
    """
    if is_lora:
        # Load LoRA adapters
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model.load_adapter(checkpoint_path, "default")
            logger.info(f"Loaded LoRA adapters from {checkpoint_path}")
            
            # Look for training state file
            training_state_path = None
            if os.path.isdir(checkpoint_path):
                parent_dir = checkpoint_path
            else:
                parent_dir = os.path.dirname(checkpoint_path)
                
            for filename in os.listdir(parent_dir):
                if filename.startswith("training_state") and filename.endswith(".pt"):
                    training_state_path = os.path.join(parent_dir, filename)
                    break
            
            if training_state_path and os.path.exists(training_state_path):
                state_dict = torch.load(training_state_path, map_location='cpu')
                if optimizer and 'optimizer_state_dict' in state_dict:
                    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                
                return {
                    'epoch': state_dict.get('epoch', 0),
                    'step': state_dict.get('step', 0),
                    'loss': 0.0,
                    'best_score': state_dict.get('best_score', float('-inf'))
                }
            else:
                logger.warning("No training state file found")
                return {'epoch': 0, 'step': 0, 'loss': 0.0, 'best_score': float('-inf')}
        else:
            raise ValueError("Model must be a PeftModel to load LoRA adapters")
    else:
        # Load full model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'loss': checkpoint.get('loss', float('inf'))
        }

def setup_logging(log_level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def merge_lora_adapters(model, output_path: str):
    """
    Merge LoRA adapters into base model and save.
    
    Args:
        model: PeftModel with LoRA adapters
        output_path: Path to save merged model
    """
    from peft import PeftModel
    
    if isinstance(model, PeftModel):
        logger.info("Merging LoRA adapters into base model...")
        merged_model = model.merge_and_unload()
        
        # Save merged model
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)
        logger.info(f"Saved merged model to {output_path}")
        
        return merged_model
    else:
        logger.warning("Model is not a PeftModel, saving as-is")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        return model

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)