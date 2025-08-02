import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import anthropic
from anthropic import Anthropic
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerationService:
    """Service for generating virtual personas using Claude Sonnet"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the service with Claude API key"""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
        
        self.client = Anthropic(api_key=self.api_key)
        
        # Create data/personas directory if it doesn't exist
        self.data_dir = "data/personas"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _create_persona_generation_prompt(self, domain_context: str) -> str:
        """Create a prompt for generating personas"""
        return f"""
You are an expert data generator for ML training datasets. Generate 5 realistic and diverse personas for the following context:

We need personas who are in this DOMAIN:
{domain_context}

Generate 5 diverse personas that would be realistic targets for this business. Each persona should be detailed and authentic, representing different segments and personalities of the target market.

For each persona, create a detailed string description that includes their name, industry, occupation, preferred communication style, values, personality traits, decision-making factors, pain points, and any other relevant characteristics that would be important for outreach and messaging.

Return ONLY a valid JSON array of persona strings in this exact format:
[
  "Persona description 1 with all relevant details...",
  "Persona description 2 with all relevant details...",
  "Persona description 3 with all relevant details...",
  "Persona description 4 with all relevant details...",
  "Persona description 5 with all relevant details..."
]

Return nothing else except the JSON array of strings.
"""

    async def _generate_single_batch(self, domain_context: str) -> List[str]:
        """Generate a single batch of 5 persona strings"""
        try:
            prompt = self._create_persona_generation_prompt(domain_context)
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.8,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse the JSON response
            response_text = response.content[0].text.strip()
            personas_data = json.loads(response_text)
            
            # Ensure we have a list of strings
            if not isinstance(personas_data, list):
                raise ValueError("Response is not a list")
            
            for persona in personas_data:
                if not isinstance(persona, str):
                    raise ValueError("Persona is not a string")
            
            logger.info(f"Successfully generated batch of {len(personas_data)} persona strings")
            return personas_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            raise ValueError("Claude returned invalid JSON response")
        except Exception as e:
            logger.error(f"Error generating personas batch: {e}")
            raise

    async def generate_personas(self, domain_context: str, count: int = 50, filename: str = "personas") -> List[str]:
        """Generate virtual personas using Claude Sonnet with batch processing"""
        try:
            logger.info(f"Generating {count} personas for domain: {domain_context}")
            
            # Calculate number of batches needed (5 personas per batch)
            batch_count = count // 5
            if count % 5 != 0:
                batch_count += 1
            
            all_personas = []
            
            # Generate personas in batches
            for i in range(batch_count):
                logger.info(f"Generating batch {i+1}/{batch_count}")
                batch_personas = await self._generate_single_batch(domain_context)
                all_personas.extend(batch_personas)
                
                # Add small delay between requests to be respectful to the API
                if i < batch_count - 1:
                    await asyncio.sleep(1)
            
            # Trim to exact count if we generated more than requested
            all_personas = all_personas[:count]
            
            logger.info(f"Successfully generated {len(all_personas)} total personas")
            
            # Save to file
            self.save_personas_to_file(all_personas, filename)
            
            return all_personas
            
        except Exception as e:
            logger.error(f"Error generating personas: {e}")
            raise

    def generate_personas_sync(self, domain_context: str, count: int = 50, filename: str = "personas") -> List[str]:
        """Synchronous wrapper for generate_personas"""
        return asyncio.run(self.generate_personas(domain_context, count, filename))

    def save_personas_to_file(self, personas: List[str], filename: str) -> None:
        """Save generated personas to a JSON file in data/personas directory"""
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = os.path.join(self.data_dir, filename)
            
            # Delete old file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted existing file: {filepath}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(personas, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(personas)} personas to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving personas to file: {e}")
            raise

    def load_personas_from_file(self, filename: str) -> List[str]:
        """Load personas from a JSON file in data/personas directory"""
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                personas = json.load(f)
            
            logger.info(f"Loaded {len(personas)} personas from {filepath}")
            return personas
            
        except Exception as e:
            logger.error(f"Error loading personas from file: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    service = DataGenerationService()
    
    domain_context = "B2B SaaS cold email outreach for workflow automation solutions targeting SMBs"
    
    try:
        # Generate 50 personas (10 batches of 5 each)
        personas = service.generate_personas_sync(
            domain_context=domain_context, 
            count=10, 
            filename="b2b_saas_personas"
        )
        
        # Print first few generated personas
        for i, persona in enumerate(personas[:3], 1):
            print(f"\n--- Persona {i} ---")
            print(f"{persona[:200]}...")  # First 200 chars
        
        print(f"\nTotal personas generated: {len(personas)}")
        
    except Exception as e:
        print(f"Error: {e}")
