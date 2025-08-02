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


def decide_response(smb_context: Dict, message: str) -> DecisionResponse:
    """
    SMB cold email decision maker using Claude Sonnet 4.0 with structured output.
    
    Args:
        smb_context: Dictionary containing SMB information (business_type, size, goals, etc.)
        message: The cold email message content
        
    Returns:
        DecisionResponse: Contains decision (bool) and reasoning (str)
        
    Raises:
        ValueError: If required inputs are missing or invalid
        Exception: If API call fails
    """
    if not smb_context:
        raise ValueError("SMB context cannot be empty")
    if not message or not message.strip():
        raise ValueError("Message cannot be empty")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are an experienced SMB decision maker evaluating cold emails.

Consider these factors:
1. Relevance to business type and current needs
2. Legitimacy and professionalism of sender
3. Clear value proposition
4. Reasonable resource requirements
5. Alignment with business goals

Respond with valid JSON only, using this exact structure:
{
  "decision": true,
  "reasoning": "Brief explanation of your decision"
}

The decision field is boolean (true/false) and reasoning is a string explanation."""

    user_prompt = f"""SMB Context:
Business Type: {smb_context.get('business_type', 'Not specified')}
Company Size: {smb_context.get('size', 'Not specified')}
Industry: {smb_context.get('industry', 'Not specified')}
Current Goals: {smb_context.get('goals', 'Not specified')}
Budget Range: {smb_context.get('budget_range', 'Not specified')}
Key Challenges: {smb_context.get('challenges', 'Not specified')}

Cold Email Message:
{message}

Should this SMB respond to this cold email?"""

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