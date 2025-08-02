import os
import json
import anthropic
from typing import Dict, TypedDict
from dotenv import load_dotenv

load_dotenv()

class DecisionResponse(TypedDict):
    """Structure for decision response with reasoning."""
    decision: bool
    reasoning: str


def decide_response(persona: str, message: str) -> DecisionResponse:
    """
    B2B professional cold email decision maker using Claude Sonnet 4.0 with structured output.
    
    Args:
        persona: String containing B2B professional persona information
        message: The cold email message content
        
    Returns:
        DecisionResponse: Contains decision (bool) and reasoning (str)
        
    Raises:
        ValueError: If required inputs are missing or invalid
        Exception: If API call fails
    """
    if not persona or not persona.strip():
        raise ValueError("Persona cannot be empty")
    if not message or not message.strip():
        raise ValueError("Message cannot be empty")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are an experienced B2B professional evaluating cold outreach emails.

Consider these factors:
1. Relevance to your role, company, and current challenges
2. Legitimacy and professionalism of the sender
3. Clear value proposition that addresses your pain points
4. Reasonable resource requirements and implementation effort
5. Alignment with your decision-making style and preferences
6. Timing and approach matching your communication preferences

Respond with valid JSON only, using this exact structure:
{
  "decision": true,
  "reasoning": "Brief explanation of your decision"
}

The decision field is boolean (true/false) and reasoning is a string explanation.

Please be very hard on the actualy decision and who you reply to. Your time is crucial and do not fall into every trap. It has to be a great fit."""

    user_prompt = f"""Your Persona:
{persona}

Cold Email Message:
{message}

Based on your persona, role, responsibilities, communication preferences, pain points, and decision-making style, would you respond positively to this cold email? Consider whether this message would catch your attention and prompt you to engage."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "{"}]
                }
            ]
        )
        
        # Parse structured JSON response
        response_text = "{" + response.content[0].text.strip()
        try:
            parsed_response = json.loads(response_text)
            return {
                'decision': bool(parsed_response.get('decision', False)),
                'reasoning': str(parsed_response.get('reasoning', 'No reasoning provided'))
            }
        except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
            # Fallback parsing if JSON is malformed
            text = response_text.lower()
            decision = "true" in text or "yes" in text
            return {
                'decision': decision,
                'reasoning': f"Fallback parsing used due to malformed response: {parse_error}"
            }
    except Exception as e:
        raise Exception(f"API call failed: {e}")