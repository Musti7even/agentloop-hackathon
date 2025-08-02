import os
import json
import anthropic
from typing import Dict, TypedDict
from dotenv import load_dotenv

load_dotenv()

class DecisionResponse(TypedDict):
    """Structure for decision response with score and reasoning."""
    score: float
    reasoning: str


def decide_response(persona: str, message: str) -> DecisionResponse:
    """
    B2B professional cold email decision maker using Claude Sonnet 3.5 with scored output.
    
    Args:
        persona: String containing B2B professional persona information
        message: The cold email message content
        
    Returns:
        DecisionResponse: Contains score (0.0-1.0) and reasoning (str)
        
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
  "score": 0.7,
  "reasoning": "Brief explanation of your score"
}

The score field is a float between 0.0 and 1.0:
- 0.0-0.2: Definitely won't respond (spam, irrelevant, unprofessional)
- 0.3-0.4: Unlikely to respond (poor fit, timing issues)
- 0.5-0.6: Might respond (some relevance but concerns)
- 0.7-0.8: Likely to respond (good fit, clear value)
- 0.9-1.0: Definitely will respond (perfect fit, urgent need)

Please be very hard on the actual scoring. Your time is crucial and do not fall into every trap. It has to be a great fit to score highly."""

    user_prompt = f"""Your Persona:
{persona}

Cold Email Message:
{message}

Based on your persona, role, responsibilities, communication preferences, pain points, and decision-making style, how likely are you to respond positively to this cold email? Provide a score from 0.0 to 1.0 indicating your likelihood of responding."""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.7,
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
            score = float(parsed_response.get('score', 0.0))
            # Ensure score is within valid range
            score = max(0.0, min(1.0, score))
            return {
                'score': score,
                'reasoning': str(parsed_response.get('reasoning', 'No reasoning provided'))
            }
        except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
            # Fallback parsing if JSON is malformed
            text = response_text.lower()
            # Try to extract a numeric score from the text
            import re
            score_match = re.search(r'"score":\s*([0-9]*\.?[0-9]+)', text)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    score = 0.0
            else:
                # Last resort - look for keywords
                if any(word in text for word in ["definitely", "excellent", "perfect"]):
                    score = 0.8
                elif any(word in text for word in ["likely", "good", "interested"]):
                    score = 0.6
                elif any(word in text for word in ["maybe", "possibly", "might"]):
                    score = 0.4
                else:
                    score = 0.2
                    
            return {
                'score': score,
                'reasoning': f"Fallback parsing used due to malformed response: {parse_error}"
            }
    except Exception as e:
        raise Exception(f"API call failed: {e}")