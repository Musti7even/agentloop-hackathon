# GRPO Email Training

Group Relative Policy Optimization (GRPO) training system for fine-tuning language models to generate high-scoring cold emails using reinforcement learning.

## Overview

This system trains a small language model (Qwen3-0.6B) to generate cold emails that score highly on your custom scoring function. It uses a GRPO-style approach where:

1. **Generation**: Model generates multiple emails for each persona
2. **Scoring**: Each email is scored using your `decision_maker_score.py` function
3. **Ranking**: Emails are ranked within persona groups
4. **Training**: Model is updated using group-relative advantages

## Features

- **QLoRA Efficiency**: Uses 4-bit quantization + LoRA for 75% memory reduction
- **Small Model**: Uses Qwen3-0.6B for fast training and inference
- **Custom Scoring**: Integrates with your existing email scoring system
- **Group Relative Learning**: GRPO algorithm for stable policy optimization
- **Concurrent Scoring**: Parallel email scoring for efficiency
- **Experiment Tracking**: Weights & Biases integration
- **Smart Checkpointing**: Automatic LoRA adapter and full model saving
- **Flexible Configuration**: JSON config files and command-line arguments

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
export ANTHROPIC_API_KEY="your_api_key_here"
```

## Quick Start

1. **Prepare your personas** in a JSON file (simple array format):
```json
[
  "Sarah Chen, Director of Operations at a fast-growing 50-person B2B marketing tech startup. Age 34, analytical and data-driven, prefers email communication with clear metrics. Values efficiency and ROI above all...",
  "Marcus Rodriguez, IT Operations Manager at a 500-person mid-sized financial services software company. Age 45, methodical and process-oriented, prefers scheduled video calls with detailed agendas..."
]
```

Or with structured objects:
```json
[
  {
    "persona": "Marketing Director at a SaaS company..."
  },
  {
    "persona": "CTO at a fintech startup..."  
  }
]
```

2. **Run training**:
```bash
# Using your B2B SaaS personas
python train_email_model.py \
    --personas_file data/personas/b2b_saas_personas_train.json \
    --num_epochs 5 \
    --output_dir ./checkpoints

# Or with custom personas file
python train_email_model.py \
    --personas_file personas.json \
    --num_epochs 5 \
    --output_dir ./checkpoints
```

3. **Monitor progress** with Weights & Biases (optional):
```bash
python train_email_model.py \
    --personas_file personas.json \
    --wandb_project my-email-training \
    --wandb_run_name experiment-1
```

## Configuration

### Command Line Arguments

- `--personas_file`: Path to JSON file with personas (required)
- `--val_personas_file`: Separate validation personas file
- `--model_name`: Hugging Face model name (default: Qwen/Qwen3-0.6B)
- `--num_epochs`: Number of training epochs (default: 5)  
- `--learning_rate`: Learning rate (default: 1e-5)
- `--batch_size`: Training batch size (default: 16)
- `--num_emails_per_persona`: Emails generated per persona (default: 8)
- `--output_dir`: Checkpoint directory (default: ./grpo_checkpoints)

### QLoRA-Specific Arguments
- `--use_qlora`: Enable QLoRA training (default: True)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA scaling parameter (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.1)

### Config File

Use a JSON config file for complex configurations:

```json
{
  "model_name": "Qwen/Qwen3-0.6B",
  "num_epochs": 10,
  "learning_rate": 1e-5,
  "batch_size": 16,
  "num_emails_per_persona": 8,
  "group_size": 8,
  "temperature": 0.8,
  "top_p": 0.9,
  "ppo_epochs": 4,
  "clip_ratio": 0.2,
  "use_qlora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "wandb_project": "grpo-email-training"
}
```

Run with config file:
```bash
python train_email_model.py --personas_file personas.json --config_file config.json
```

## Architecture

### Core Components

- **EmailGenerator**: Handles Qwen3-0.6B for email generation
- **EmailScorer**: Concurrent scoring using your decision_maker_score
- **GRPOTrainer**: Implements group relative policy optimization
- **TrainingConfig**: Configuration management

### GRPO Algorithm

1. **Group Formation**: Emails for the same persona form a group
2. **Relative Ranking**: Scores are converted to relative advantages within groups
3. **Policy Update**: PPO-style updates using group-relative advantages
4. **Reward Shaping**: Advantages are normalized for stable training

### Training Process

```
For each epoch:
  1. Generate N emails per persona
  2. Score all emails using decision_maker_score
  3. Compute group-relative advantages
  4. Run PPO epochs with weighted loss
  5. Evaluate on validation set
  6. Save checkpoints
```

## File Structure

```
grpo_training/
├── train_email_model.py    # Main training script
├── email_generator.py      # Email generation with Qwen3-0.6B
├── grpo_trainer.py         # Core GRPO training algorithm
├── config.py               # Configuration classes
├── utils.py                # Helper functions and scoring
├── requirements.txt        # Python dependencies
├── example_usage.py        # Example usage script
└── README.md              # This file
```

## Examples

### Basic Training
```bash
# Using the provided B2B SaaS personas
python train_email_model.py \
    --personas_file data/personas/b2b_saas_personas_train.json \
    --num_epochs 5 \
    --output_dir ./my_checkpoints

# Using custom personas
python train_email_model.py \
    --personas_file my_personas.json \
    --num_epochs 5 \
    --output_dir ./my_checkpoints
```

### Advanced Training with Monitoring
```bash
python train_email_model.py \
    --personas_file train_personas.json \
    --val_personas_file val_personas.json \
    --config_file advanced_config.json \
    --wandb_project email-optimization \
    --device cuda \
    --mixed_precision
```

### Resume from Checkpoint
```bash
python train_email_model.py \
    --personas_file personas.json \
    --resume_from_checkpoint ./checkpoints/grpo_checkpoint_best_epoch_3_step_1500.pt
```

## Monitoring

The system logs comprehensive metrics:

- **Training Loss**: Policy gradient loss with advantage weighting
- **Average Advantage**: Mean group-relative advantage
- **Validation Scores**: Mean/std/max/min scores on validation set
- **Learning Rate**: Current learning rate from scheduler

View metrics in:
- Console logs
- Weights & Biases dashboard (if configured)
- TensorBoard logs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - QLoRA should reduce memory by ~75% automatically
   - Reduce `batch_size` or `generation_batch_size` further
   - Use `--mixed_precision` flag
   - Try smaller model like Qwen2.5-0.5B

2. **Scoring Timeouts**:
   - Increase `scorer_timeout` in config
   - Reduce `scorer_max_concurrent`
   - Check ANTHROPIC_API_KEY and rate limits

3. **Poor Training Performance**:
   - Increase `num_emails_per_persona` for more diverse training data
   - Adjust `learning_rate` (try 5e-6 or 2e-5)
   - Tune `reward_scaling` parameter

### Performance Tips

- **QLoRA is enabled by default** for maximum efficiency
- Use GPU for faster training (QLoRA works great on single GPU)
- Increase concurrent scoring if you have API quota
- Use validation set to monitor overfitting
- LoRA adapters are automatically saved at checkpoints
- For inference, use the merged model in `final_model_merged/`

## Extending the System

### Custom Scoring Functions

Replace the scoring integration in `utils.py`:

```python
def score_email(self, persona: str, email: str) -> Dict[str, Any]:
    # Your custom scoring logic here
    score = your_scoring_function(persona, email)
    return {'score': score, 'reasoning': '', 'success': True, 'error': None}
```

### Different Models

Change the model in config (QLoRA works with most transformer models):

```python
config = TrainingConfig(
    model_name="microsoft/DialoGPT-small",  # or any causal LM
    use_qlora=True,  # Enable for any model
    lora_r=8,  # Smaller rank for smaller models
    # ... other settings
)
```

### Disable QLoRA (Full Fine-tuning)

```python
config = TrainingConfig(
    use_qlora=False,  # Disable QLoRA
    learning_rate=5e-6,  # Lower LR for full fine-tuning
    # ... other settings
)
```

### Custom Reward Shaping

Modify advantage computation in `grpo_trainer.py`:

```python
# Custom advantage weighting
advantage_weights = torch.sigmoid(advantages * self.config.reward_scaling)
weighted_loss = loss * advantage_weights.mean()
```
