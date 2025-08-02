import os
import anthropic
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


def decide_response(smb_context: Dict, message: str) -> bool:
    """
    SMB cold email decision maker using Claude Sonnet 4.0.
    
    Args:
        smb_context: Dictionary containing SMB information (business_type, size, goals, etc.)
        message: The cold email message content
        
    Returns:
        bool: True if SMB should respond, False otherwise
        
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
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to initialize Anthropic client: {e}")
    
    system_prompt = """You are an experienced small-to-medium business (SMB) decision maker. Your job is to evaluate cold emails and decide whether they're worth responding to.

Consider these factors when making your decision:
1. Relevance to the business type and current needs
2. Legitimacy and professionalism of the sender
3. Clear value proposition that could benefit the SMB
4. Reasonable resource requirements (time, budget, complexity)
5. Alignment with business goals and growth stage

Respond with only "YES" if the SMB should respond, or "NO" if they should not respond."""

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
            max_tokens=10,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ]
        )
        
        decision = response.content[0].text.strip().upper()
        return decision == "YES"
    except Exception as e:
        raise Exception(f"API call failed: {e}")


def decide_response_with_reasoning(smb_context: Dict, message: str) -> Dict:
    """
    SMB cold email decision maker with detailed reasoning.
    
    Args:
        smb_context: Dictionary containing SMB information
        message: The cold email message content
        
    Returns:
        Dict: Contains 'decision' (bool) and 'reasoning' (str)
        
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
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to initialize Anthropic client: {e}")
    
    system_prompt = """You are an experienced small-to-medium business (SMB) decision maker. Evaluate cold emails and provide a decision with reasoning.

Consider these factors:
1. Relevance to the business type and current needs
2. Legitimacy and professionalism of the sender
3. Clear value proposition that could benefit the SMB
4. Reasonable resource requirements (time, budget, complexity)
5. Alignment with business goals and growth stage

Format your response as:
DECISION: YES or NO
REASONING: Brief explanation of your decision"""

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
                }
            ]
        )
        
        response_text = response.content[0].text.strip()
        
        lines = response_text.split('\n')
        decision_line = next((line for line in lines if line.startswith('DECISION:')), '')
        reasoning_line = next((line for line in lines if line.startswith('REASONING:')), '')
        
        decision = 'YES' in decision_line.upper()
        reasoning = reasoning_line.replace('REASONING:', '').strip() if reasoning_line else response_text
        
        return {
            'decision': decision,
            'reasoning': reasoning
        }
    except Exception as e:
        raise Exception(f"API call failed: {e}")