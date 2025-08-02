import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import anthropic
from anthropic import Anthropic
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Persona:
    """Represents a generated persona with relevant attributes"""
    name: str
    context: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'context': self.context
        }

class DataGenerationService:
    """Service for generating virtual personas using Claude Sonnet"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the service with Claude API key"""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
        
        self.client = Anthropic(api_key=self.api_key)
        
    def _create_persona_generation_prompt(self, domain_context: str, persona_count: int) -> str:
        """Create a prompt for generating personas"""
        return f"""
You are an expert data generator for ML training datasets. Generate {persona_count} realistic and diverse personas for the following context:

We need personal who are in this DOMAIN:
{domain_context}

Generate {persona_count} diverse personas that would be realistic targets for this business. Each persona should be detailed and authentic, representing different segments and personalitiies of the target market.

For each persona you generate a string that contains information like the name, industry, their prefered style, how hard their are on something their values ...

Return ONLY a valid JSON array of personas, no additional text or formatting.
"""

    async def generate_personas(self, user_context: Dict[str, Any], domain_context: str, count: int = 10) -> List[Persona]:
        """Generate virtual personas using Claude Sonnet"""
        try:
            logger.info(f"Generating {count} personas for domain: {domain_context}")
            
            prompt = self._create_persona_generation_prompt(user_context, domain_context, count)
            
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
            personas_data = json.loads(response.content[0].text)
            
            # Convert to Persona objects
            personas = []
            for persona_dict in personas_data:
                persona = Persona(**persona_dict)
                personas.append(persona)
            
            logger.info(f"Successfully generated {len(personas)} personas")
            return personas
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            raise ValueError("Claude returned invalid JSON response")
        except Exception as e:
            logger.error(f"Error generating personas: {e}")
            raise

    def generate_personas_sync(self, user_context: Dict[str, Any], domain_context: str, count: int = 10) -> List[Persona]:
        """Synchronous wrapper for generate_personas"""
        return asyncio.run(self.generate_personas(user_context, domain_context, count))

    def save_personas_to_file(self, personas: List[Persona], filename: str) -> None:
        """Save generated personas to a JSON file"""
        try:
            personas_dict = [persona.to_dict() for persona in personas]
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(personas_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(personas)} personas to {filename}")
        except Exception as e:
            logger.error(f"Error saving personas to file: {e}")
            raise

    def load_personas_from_file(self, filename: str) -> List[Persona]:
        """Load personas from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                personas_dict = json.load(f)
            
            personas = []
            for persona_dict in personas_dict:
                persona = Persona(**persona_dict)
                personas.append(persona)
            
            logger.info(f"Loaded {len(personas)} personas from {filename}")
            return personas
        except Exception as e:
            logger.error(f"Error loading personas from file: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    service = DataGenerationService()
    
    # Example user context
    user_context = {
        "company_name": "TechFlow Solutions",
        "industry": "B2B SaaS",
        "product_description": "AI-powered workflow automation platform for small to medium businesses",
        "target_market": "SMB operations managers, CTOs, and business owners",
        "value_proposition": "Reduce manual work by 60% and increase team productivity",
        "company_stage": "Series A startup",
        "additional_context": "Focusing on companies with 10-200 employees who struggle with repetitive manual processes"
    }
    
    domain_context = "B2B SaaS cold email outreach for workflow automation solutions targeting SMBs"
    
    try:
        # Generate personas
        personas = service.generate_personas_sync(user_context, domain_context, count=5)
        
        # Print generated personas
        for i, persona in enumerate(personas, 1):
            print(f"\n--- Persona {i} ---")
            print(f"Name: {persona.name}")
            print(f"Occupation: {persona.occupation}")
            print(f"Industry: {persona.industry}")
            print(f"Pain Points: {', '.join(persona.pain_points)}")
            print(f"Communication Style: {persona.communication_style}")
        
        # Save to file
        service.save_personas_to_file(personas, "generated_personas.json")
        
    except Exception as e:
        print(f"Error: {e}")
