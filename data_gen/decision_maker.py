import os
import json
import anthropic
from typing import Dict, TypedDict, List
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DecisionResponse(TypedDict):
    """Structure for decision response with reasoning."""
    decision: bool
    reasoning: str

default_system_prompt = """You are an experienced SMB decision maker evaluating cold emails.

Consider these factors:
1. Relevance to business type and current needs
2. Legitimacy and professionalism of sender
3. Clear value proposition
4. Reasonable resource requirements
5. Alignment with business goals

Respond with valid JSON only, using this exact structure:
{{
  "decision": true/false,
  "reasoning": "Brief explanation of your decision"
}}

The decision field is boolean (true/false) and reasoning is a string explanation.

{persona}

Should this SMB respond to this cold email?"""


def decide_response(persona: str, message: str) -> DecisionResponse:
    """
    cold email decision maker using Claude Sonnet 4.0 with structured output.
    
    Args:
        persona: persona of the person that you want to reach out to and who decides if to resposne or not
        message: The cold email message content
        
    Returns:
        DecisionResponse: Contains decision (bool) and reasoning (str)
        
    Raises:
        ValueError: If required inputs are missing or invalid
        Exception: If API call fails
    """
    if not persona:
        raise ValueError("Persona cannot be empty")
    if not message or not message.strip():
        raise ValueError("Message cannot be empty")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = default_system_prompt.format(persona=persona)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": message}]
                },
            ]
        )
        
        # Parse structured JSON response
        response_text = response.content[0].text.strip()
        logger.debug(f"Raw response: {response_text[:200]}...")
        
        try:
            parsed_response = json.loads(response_text)
            return {
                'decision': bool(parsed_response.get('decision', False)),
                'reasoning': str(parsed_response.get('reasoning', 'No reasoning provided'))
            }
        except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
            logger.warning(f"JSON parsing failed for response: {response_text[:100]}")
            logger.warning(f"Parse error: {parse_error}")
            
            # Enhanced fallback parsing
            text = response_text.lower()
            if '"decision": true' in text or '"decision":true' in text:
                decision = True
            elif '"decision": false' in text or '"decision":false' in text:
                decision = False
            else:
                decision = "true" in text or "yes" in text or "respond" in text
            
            # Try to extract reasoning if available
            reasoning = "Could not parse structured response"
            if '"reasoning"' in text:
                try:
                    # Try to extract reasoning text between quotes
                    start = text.find('"reasoning"')
                    if start != -1:
                        colon_pos = text.find(':', start)
                        if colon_pos != -1:
                            quote_start = text.find('"', colon_pos)
                            if quote_start != -1:
                                quote_end = text.find('"', quote_start + 1)
                                if quote_end != -1:
                                    reasoning = response_text[quote_start + 1:quote_end]
                except:
                    pass
            
            return {
                'decision': decision,
                'reasoning': reasoning
            }
    except Exception as e:
        raise Exception(f"API call failed: {e}")
