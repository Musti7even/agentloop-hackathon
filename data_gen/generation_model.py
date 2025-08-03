import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

default_system_prompt = """You are an expert cold outreach specialist who creates highly personalized, compelling messages.

Your goal is to write a cold outreach message that:
1. Is highly personalized to the recipient
2. Clearly communicates value proposition
3. Has a strong call-to-action
4. Feels authentic and not salesy
5. Is concise and respectful of their time

Write in a professional but approachable tone. Keep it under 150 words."""

def generate_message(user_context: str, persona: str, system_prompt: str = default_system_prompt, user_prompt: str = None) -> str:
    """
    Generate optimized cold outreach message using user context and persona.
    
    Args:
        user_context (C): String containing user/startup context information
        persona (p): String description of the target persona
        
    Returns:
        str: Generated cold outreach message
        
    Raises:
        ValueError: If required inputs are missing
        Exception: If API call fails
    """
    if not user_context or not user_context.strip():
        raise ValueError("User context (C) cannot be empty")
    if not persona or not persona.strip():
        raise ValueError("Persona (p) cannot be empty")
    
    # Construct user prompt if not provided
    if user_prompt is None:
        user_prompt = f"""USER/STARTUP CONTEXT (C):
{user_context}

TARGET PERSONA (P):
{persona}

Write a personalized cold outreach message for this specific persona. Make it highly relevant to their situation, pain points, and communication style. Return only the message text."""
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    client = Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        
        generated_message = response.content[0].text.strip()
        return generated_message
        
    except Exception as e:
        raise Exception(f"Message generation failed: {e}")


class GenerationModel:
    """Simple class wrapper for the generation function"""
    
    def __init__(self):
        """Initialize the generation model"""
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    def generate(self, user_context: str, persona: str) -> str:
        """Generate message using the function"""
        return generate_message(user_context, persona)


# Example usage and testing
if __name__ == "__main__":
    # Example user context (C) - now just a string
    user_context = """
    Company: FlowAI
    Industry: B2B SaaS workflow automation 
    Product: AI-powered workflow automation platform for SMBs
    Value Proposition: Reduce manual work by 60% and increase team productivity
    Target Market: SMB operations managers, CTOs, and business owners with 10-200 employees
    Stage: Series A startup
    Focus: Companies struggling with repetitive manual processes and operational overhead
    """
    
    # Example persona (p) - from the generated data
    persona = "Michael Rodriguez, VP of Business Operations at a 2000-employee enterprise SaaS company in Austin. Age 45, strategic thinker with MBA from Wharton. Oversees global operations across 4 regions. Traditional in approach but open to innovation when well-proven. Prefers formal communication and scheduled meetings. Values vendor reliability and enterprise-grade security. Key decision maker but consults extensively with IT and Finance. Pain points: complex compliance requirements, lack of visibility across departments, integration issues with legacy systems. Budget conscious but willing to invest in long-term solutions."
    
    try:
        # Test function approach
        message = generate_message(user_context, persona)
        print("Generated Message:")
        print("-" * 50)
        print(message)
        print("-" * 50)
        
        # Test class approach
        model = GenerationModel()
        message2 = model.generate(user_context, persona)
        print("\nGenerated Message (class):")
        print("-" * 50)
        print(message2)
        
    except Exception as e:
        print(f"Error: {e}") 