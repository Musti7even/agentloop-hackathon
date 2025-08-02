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

    system_prompt = """You are an experienced B2B buyer screening cold outreach in a crowded inbox. Default decision is FALSE unless criteria are met. Target acceptance ~25–30%; when uncertain, reject.

This dataset may be redacted (e.g., “[Company]”, “[Your name]”). Treat placeholders as privacy redactions, NOT errors. Ignore sender identity entirely; judge only content and fit to the persona.

Respond with valid JSON only, using this exact structure:
{
  "decision": true,
  "reasoning": "Brief explanation of your decision"
}
The decision field is boolean (true/false). The reasoning is one concise sentence. Return ONLY the JSON object.

— GATE CHECKS (apply in order) —
1) Persona Fit Gate (HARD): The email shows CLEAR specificity to the persona with at least TWO concrete ties (e.g., role/pains, tech stack/tooling, scale/constraints, regulatory context, current initiative).
   • If fewer than two ties → reject.

2) Evidence FLOOR (SOFTENED vs. strict proof): The email must include at least TWO of the following FOUR elements:
   (a) Specific metric(s) (e.g., hours saved, % reduction, time cut “2 days → 4 hours”),
   (b) Mechanism explaining *how* results are achieved for this context,
   (c) Context details about the example (industry/size/function/use case) — anonymized OK,
   (d) A linkable or offered artifact (case study/whitepaper/docs/demo link).
   • If fewer than two elements → reject.
   • Timeframe/scope boosts credibility but is not required for passing this floor.

3) Next-Step Gate (HARD): The ask is reasonable (not too many minutes or offers async materials first) and appropriate to the persona’s style.


— SCORING RUBRIC (0–10) — only if all gates pass
1) Relevance & Specificity (0–3): Concrete tie to persona’s situation (not just industry buzzwords).
2) Credibility & Proof Quality (0–3): Strength/coherence of metrics, mechanism, context, timeframe/scope, or artifact.
3) Value & Mechanism (0–2): Tangible outcomes (time/cost/error) AND a clear “how” that fits the environment.
4) Effort/Risk Fit (0–1): Low-lift path (pilot/trial/docs), mentions key integrations/compliance if relevant.
5) Tone & Ask Fit (0–1): Professional, concise, respectful ask.

— DECISION RULE —
• Approve ONLY if total score ≥ 6.5. If score = 6 (borderline), REJECT.
• Deduct points for missing details rather than inferring best case.
• Aim for ~25–30% acceptance overall.

Reasoning: In a couple of words (max 7 words), explain your decision.

"""

    user_prompt = f"""Your Persona:
{persona}

Cold Email Message:
{message}

Treat placeholders as redactions and IGNORE sender identity. Apply the GATE CHECKS in order:
1) Persona Fit Gate (need ≥2 concrete ties),
2) Evidence FLOOR (need ≥2 of metric/mechanism/context/artifact),
3) Next-Step Gate (low-lift ask).
If any gate fails, reject. If all pass, score with the rubric and approve only if total ≥ 8 (reject 7/borderline). Do not infer unstated facts; deduct for missing detail. Return ONLY the JSON object in the required schema.

"""

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