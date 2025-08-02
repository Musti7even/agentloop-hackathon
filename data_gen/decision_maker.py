import os
import json
import anthropic
from typing import Dict, TypedDict, List
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
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


def decide_response_batch(filename: str) -> List[DecisionResponse]:
    """
    Batch decision maker for cold email responses.

    It updates after every run the file with the new decision for each object/row within the file.
    
    Args:
        filename: Name of the file containing the messages to run the decision maker on (contains also the persona)
        
    Returns:
        List of DecisionResponse objects containing decisions and reasoning
    """
    try:
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Load messages file
        messages_dir = "data/messages"
        filepath = os.path.join(messages_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Messages file not found: {filepath}")
        
        logger.info(f"Loading messages from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            messages_data = json.load(f)
        
        if not isinstance(messages_data, list):
            raise ValueError("Messages file must contain a JSON array")
        
        logger.info(f"Processing {len(messages_data)} messages for decision making")
        
        decisions = []
        updated_messages = []
        
        for i, message_obj in enumerate(messages_data, 1):
            try:
                # Extract required fields
                persona = message_obj.get('persona', '')
                message = message_obj.get('message', '')
                
                if not persona or not message:
                    logger.warning(f"Skipping message {i}: missing persona or message")
                    # Keep original object without decision
                    updated_messages.append(message_obj)
                    continue
                
                # Make decision
                logger.info(f"Processing message {i}/{len(messages_data)}")
                decision_response = decide_response(persona, message)
                
                # Add decision to the message object
                message_obj_with_decision = message_obj.copy()
                message_obj_with_decision['decision'] = decision_response['decision']
                message_obj_with_decision['decision_reasoning'] = decision_response['reasoning']
                
                updated_messages.append(message_obj_with_decision)
                decisions.append(decision_response)
                
                logger.info(f"Decision for message {i}: {decision_response['decision']}")
                
            except Exception as e:
                logger.error(f"Error processing message {i}: {e}")
                # Keep original object without decision
                updated_messages.append(message_obj)
                continue
        
        # Save updated messages back to file
        logger.info(f"Saving {len(updated_messages)} updated messages to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(updated_messages, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully processed {len(decisions)} decisions")
        logger.info(f"Positive decisions: {sum(1 for d in decisions if d['decision'])}")
        logger.info(f"Negative decisions: {sum(1 for d in decisions if not d['decision'])}")
        
        return decisions
        
    except Exception as e:
        logger.error(f"Error in batch decision processing: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    try:
        # Test batch decision making
        decisions = decide_response_batch("b2b_saas_personas_train_messages.json")
        
        print(f"\nProcessed {len(decisions)} decisions:")
        positive_count = sum(1 for d in decisions if d['decision'])
        negative_count = len(decisions) - positive_count
        
        print(f"Positive responses: {positive_count}")
        print(f"Negative responses: {negative_count}")
        print(f"Response rate: {positive_count/len(decisions)*100:.1f}%")
        
        # Show first few decisions
        for i, decision in enumerate(decisions[:3], 1):
            print(f"\n--- Decision {i} ---")
            print(f"Decision: {decision['decision']}")
            print(f"Reasoning: {decision['reasoning'][:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")