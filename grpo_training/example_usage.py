#!/usr/bin/env python3
"""
Example usage script for GRPO email training.
"""

import json
import tempfile
from pathlib import Path

def create_example_personas():
    """Create example personas for testing."""
    personas = [
        {
            "persona": "Marketing Director at a mid-size SaaS company focused on customer acquisition and retention. I'm responsible for a team of 8 and have a budget of $2M annually for marketing campaigns. I'm particularly interested in marketing automation tools, customer segmentation strategies, and ROI measurement. I prefer concise, data-driven communications and value clear metrics. I'm most responsive to emails that demonstrate immediate value and include case studies from similar companies."
        },
        {
            "persona": "CTO at a fintech startup with 50 employees. I oversee the engineering team and make decisions about our tech stack. Current challenges include scaling our infrastructure, ensuring compliance with financial regulations, and maintaining security. I'm interested in cloud solutions, developer tools, and security platforms. I prefer technical details and appreciate emails that understand our specific industry constraints."
        },
        {
            "persona": "VP of Sales at a manufacturing company with traditional sales processes. I manage a team of 15 sales reps and am always looking for ways to improve their productivity and close rates. I'm interested in CRM systems, sales enablement tools, and training programs. I value ROI-focused communications and prefer to see testimonials from other manufacturing companies."
        }
    ]
    
    return personas

def create_sample_config():
    """Create sample training configuration."""
    config = {
        "model_name": "Qwen/Qwen3-0.6B",
        "num_epochs": 3,
        "learning_rate": 1e-5,
        "batch_size": 8,
        "num_emails_per_persona": 4,
        "output_dir": "./example_checkpoints",
        "wandb_project": "grpo-email-example",
        "log_level": "INFO"
    }
    
    return config

def main():
    """Run example training."""
    print("GRPO Email Training Example")
    print("=" * 40)
    
    # Create temporary files for the example
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        personas = create_example_personas()
        json.dump(personas, f, indent=2)
        personas_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = create_sample_config()
        json.dump(config, f, indent=2)
        config_file = f.name
    
    print(f"Created example personas file: {personas_file}")
    print(f"Created example config file: {config_file}")
    
    # Example command to run training
    example_command = f"""
python train_email_model.py \\
    --personas_file {personas_file} \\
    --config_file {config_file} \\
    --max_personas 3
"""
    
    print("\nExample training command:")
    print(example_command)
    
    print("\nTo run the example:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up your ANTHROPIC_API_KEY environment variable")
    print("3. Run the command above")
    
    # Clean up
    import os
    try:
        os.unlink(personas_file)
        os.unlink(config_file)
    except:
        pass

if __name__ == "__main__":
    main()