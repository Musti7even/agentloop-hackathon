import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from generation_model import generate_message

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MessageData:
    """Data structure for a generated message"""
    persona: str
    message: str
    system_prompt: str

class MessageGenerationService:
    """Service for generating messages for personas"""
    
    def __init__(self):
        """Initialize the service"""
        # Create data/messages directory if it doesn't exist
        self.messages_dir = "data/messages"
        os.makedirs(self.messages_dir, exist_ok=True)
        
        self.personas_dir = "data/personas"
        
    def load_personas_from_file(self, filename: str) -> List[str]:
        """Load personas from a JSON file in data/personas directory"""
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = os.path.join(self.personas_dir, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Personas file not found: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                personas = json.load(f)
            
            if not isinstance(personas, list):
                raise ValueError("Personas file must contain a JSON array")
            
            logger.info(f"Loaded {len(personas)} personas from {filepath}")
            return personas
            
        except Exception as e:
            logger.error(f"Error loading personas from file: {e}")
            raise
    
    def generate_messages_for_personas(
        self, 
        personas_filename: str, 
        system_prompt: str, 
        messages_per_persona: int = 3
    ) -> List[MessageData]:
        """
        Generate messages for all personas in a file
        
        Args:
            personas_filename: Name of the personas file (with or without .json)
            system_prompt: The user context/system prompt to use for generation
            messages_per_persona: Number of messages to generate per persona
            
        Returns:
            List of MessageData objects containing generated messages
        """
        try:
            logger.info(f"Starting message generation for {personas_filename}")
            logger.info(f"Messages per persona: {messages_per_persona}")
            
            # Load personas
            personas = self.load_personas_from_file(personas_filename)
            
            all_messages = []
            total_personas = len(personas)
            
            for persona_idx, persona in enumerate(personas, 1):
                logger.info(f"Generating messages for persona {persona_idx}/{total_personas}")
                
                for message_idx in range(messages_per_persona):
                    try:
                        # Generate message using the generation model
                        generated_message = generate_message(system_prompt, persona)
                        
                        # Create message data object
                        message_data = MessageData(
                            persona=persona,
                            message=generated_message,
                            system_prompt=system_prompt
                        )
                        
                        all_messages.append(message_data)
                        
                        logger.info(f"Generated message {message_idx + 1}/{messages_per_persona} for persona {persona_idx}")
                        
                    except Exception as e:
                        logger.error(f"Error generating message {message_idx + 1} for persona {persona_idx}: {e}")
                        continue
            
            logger.info(f"Successfully generated {len(all_messages)} total messages")
            
            # Save messages to file
            self.save_messages_to_file(all_messages, personas_filename)
            
            return all_messages
            
        except Exception as e:
            logger.error(f"Error generating messages: {e}")
            raise
    
    def save_messages_to_file(self, messages: List[MessageData], personas_filename: str) -> None:
        """Save generated messages to a JSON file in data/messages directory"""
        try:
            # Create output filename based on personas filename
            base_filename = personas_filename.replace('.json', '')
            output_filename = f"{base_filename}_messages.json"
            output_filepath = os.path.join(self.messages_dir, output_filename)
            
            # Delete old file if it exists
            if os.path.exists(output_filepath):
                os.remove(output_filepath)
                logger.info(f"Deleted existing file: {output_filepath}")
            
            # Convert messages to dictionary format for JSON serialization
            messages_data = []
            for msg in messages:
                messages_data.append({
                    "persona": msg.persona,
                    "message": msg.message,
                    "system_prompt": msg.system_prompt
                })
            
            # Save to file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(messages_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(messages)} messages to {output_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving messages to file: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    service = MessageGenerationService()
    
    # Example system prompt (user context)
    system_prompt = """
    Company: FlowAI
    Industry: B2B SaaS workflow automation 
    Product: AI-powered workflow automation platform for SMBs
    Value Proposition: Reduce manual work by 60% and increase team productivity
    Target Market: SMB operations managers, CTOs, and business owners with 10-200 employees
    Stage: Series A startup
    Focus: Companies struggling with repetitive manual processes and operational overhead
    """
    
    try:
        # Generate messages for training personas
        messages = service.generate_messages_for_personas(
            personas_filename="b2b_saas_personas_train.json",
            system_prompt=system_prompt,
            messages_per_persona=2
        )
        
        # Print first few generated messages
        for i, msg_data in enumerate(messages[:3], 1):
            print(f"\n--- Message {i} ---")
            print(f"Persona: {msg_data.persona[:100]}...")
            print(f"Message: {msg_data.message}")
            print("-" * 50)
        
        print(f"\nTotal messages generated: {len(messages)}")
        
    except Exception as e:
        print(f"Error: {e}") 