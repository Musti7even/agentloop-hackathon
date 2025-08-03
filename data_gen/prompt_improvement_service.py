#!/usr/bin/env python3
"""
Prompt Improvement Service

This service takes in a results file, analyzes all the reasoning from decision makers,
and uses an LLM to improve the system prompt to achieve better response rates.
It then runs a new evaluation cycle with the improved prompt.

Usage:
    python prompt_improvement_service.py --results-file b2b_saas_personas_train_results_20250802_144126.json
"""

import json
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from anthropic import Anthropic

# Import our existing services
from generation_model import generate_message
from decision_maker import decide_response

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptImprovementService:
    """Service for improving system prompts based on decision maker feedback"""
    
    def __init__(self):
        """Initialize the improvement service"""
        self.data_dir = Path("data")
        self.results_dir = self.data_dir / "results"
        self.personas_dir = self.data_dir / "personas"
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = Anthropic(api_key=api_key)
    
    def load_results_file(self, filename: str) -> Dict[str, Any]:
        """Load results from JSON file in data/results directory"""
        try:
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.results_dir / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"Results file not found: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.info(f"Loaded results from {filepath}")
            logger.info(f"Original success rate: {results['metadata']['successful_personas']}/{results['metadata']['total_personas']} personas")
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading results file: {e}")
            raise
    
    def extract_all_reasoning(self, results: Dict[str, Any]) -> List[str]:
        """Extract all reasoning from decision maker responses"""
        all_reasoning = []
        
        for persona_result in results["results"]:
            for message_result in persona_result["messages"]:
                reasoning = message_result["decision"]["reasoning"]
                decision = message_result["decision"]["decision"]
                
                # Include both positive and negative reasoning for learning
                status = "POSITIVE" if decision else "NEGATIVE"
                all_reasoning.append(f"[{status}] {reasoning}")
        
        logger.info(f"Extracted {len(all_reasoning)} reasoning examples")
        return all_reasoning
    
    def improve_system_prompt(self, original_prompt: str, all_reasoning: List[str]) -> str:
        """Use LLM to improve system prompt based on decision maker reasoning"""
        
        # Concatenate all reasoning
        reasoning_text = "\n\n".join(all_reasoning)
        
        improvement_prompt = f"""You are an expert at optimizing cold outreach messaging systems. Your task is to analyze feedback from B2B decision makers and improve a system prompt used for generating cold outreach messages.

CURRENT SYSTEM PROMPT:
{original_prompt}

DECISION MAKER FEEDBACK (with POSITIVE/NEGATIVE labels):
{reasoning_text}

ANALYSIS TASK:
Analyze the feedback above and identify patterns in what makes decision makers respond positively vs negatively. Then improve the system prompt to achieve higher response rates.

KEY FOCUS AREAS:
1. What specific elements in messages lead to positive responses?
2. What mistakes or approaches lead to negative responses?
3. How can the system prompt be refined to generate more compelling messages?
4. What personalization strategies work best?
5. What value propositions resonate most?

REQUIREMENTS:
- Keep the improved prompt concise but comprehensive
- Focus on actionable instructions for message generation
- Emphasize proven effective elements from positive feedback
- Address common issues identified in negative feedback
- Maintain professional tone while improving engagement

Return ONLY the improved system prompt, no additional commentary."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": improvement_prompt
                    }
                ]
            )
            
            improved_prompt = response.content[0].text.strip()
            logger.info("Successfully generated improved system prompt")
            logger.info(f"Improved prompt length: {len(improved_prompt)} characters")
            
            return improved_prompt
            
        except Exception as e:
            logger.error(f"Error improving system prompt: {e}")
            raise
    
    def load_original_personas(self, persona_filename: str) -> List[str]:
        """Load original personas from the personas directory"""
        try:
            if not persona_filename.endswith('.json'):
                persona_filename += '.json'
            
            filepath = self.personas_dir / persona_filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"Personas file not found: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                personas = json.load(f)
            
            if not isinstance(personas, list):
                raise ValueError("Personas file must contain a JSON array")
            
            logger.info(f"Loaded {len(personas)} personas from {filepath}")
            return personas
            
        except Exception as e:
            logger.error(f"Error loading personas: {e}")
            raise
    
    def _process_single_message(self, persona: str, improved_prompt: str, persona_idx: int, message_idx: int, messages_per_persona: int) -> Dict[str, Any]:
        """Process a single message for a persona"""
        try:
            logger.info(f"  Generating message {message_idx+1}/{messages_per_persona} for persona {persona_idx+1}")
            message = generate_message(improved_prompt, persona)
            
            logger.info(f"  Evaluating message {message_idx+1}/{messages_per_persona} for persona {persona_idx+1}")
            decision = decide_response(persona, message)
            
            logger.info(f"  Message {message_idx+1} for persona {persona_idx+1} complete - Decision: {decision['decision']}")
            
            return {
                "message": message,
                "decision": decision,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing message {message_idx+1} for persona {persona_idx+1}: {e}")
            return {"success": False, "error": str(e)}

    def run_evaluation_with_improved_prompt(
        self, 
        personas: List[str], 
        improved_prompt: str, 
        messages_per_persona: int = 1
    ) -> Dict[str, Any]:
        """Run evaluation using the improved system prompt with parallel processing"""
        
        logger.info(f"Starting evaluation with improved prompt")
        logger.info(f"Processing {len(personas)} personas with {messages_per_persona} messages each")
        logger.info(f"Using 6 parallel workers for faster processing")
        
        results = {
            "metadata": {
                "messages_per_persona": messages_per_persona,
                "total_personas": len(personas),
                "total_messages": 0,
                "processing_date": datetime.now().isoformat(),
                "system_prompt": improved_prompt,
                "improvement_run": True,
                "persona_file": "extracted_from_previous_results",
                "successful_personas": 0
            },
            "results": []
        }
        
        # Prepare all tasks for parallel processing
        tasks = []
        for i, persona in enumerate(personas):
            for j in range(messages_per_persona):
                tasks.append((persona, improved_prompt, i, j, messages_per_persona))
        
        # Process all tasks in parallel using ThreadPoolExecutor
        persona_results = {}  # Store results by persona index
        total_successful_messages = 0
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_message, *task): (task[2], task[3])  # persona_idx, message_idx
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                persona_idx, message_idx = future_to_task[future]
                result = future.result()
                
                if result["success"]:
                    # Initialize persona result if not exists
                    if persona_idx not in persona_results:
                        persona_results[persona_idx] = {
                            "persona": personas[persona_idx],
                            "messages": []
                        }
                    
                    # Add message result
                    persona_results[persona_idx]["messages"].append({
                        "message": result["message"],
                        "decision": result["decision"]
                    })
                    total_successful_messages += 1
        
        # Convert to list format and sort by persona index
        for persona_idx in sorted(persona_results.keys()):
            if persona_results[persona_idx]["messages"]:
                results["results"].append(persona_results[persona_idx])
        
        # Update metadata with actual results
        successful_personas = len(results["results"])
        results["metadata"]["total_messages"] = total_successful_messages
        results["metadata"]["successful_personas"] = successful_personas
        
        logger.info(f"Evaluation complete!")
        logger.info(f"Successfully processed: {successful_personas}/{len(personas)} personas")
        logger.info(f"Total messages generated: {total_successful_messages}")
        
        return results
    
    def save_improvement_results(
        self, 
        results: Dict[str, Any], 
        original_filename: str, 
        version: int = 1
    ) -> str:
        """Save improvement results with versioned filename"""
        try:
            # Create versioned filename
            base_name = original_filename.replace('.json', '')
            if base_name.endswith('_results'):
                base_name = base_name[:-8]  # Remove '_results' suffix
            
            # Remove timestamp if present
            import re
            base_name = re.sub(r'_\d{8}_\d{6}$', '', base_name)
            
            output_filename = f"{base_name}_improvement_run_v{version}.json"
            output_filepath = self.results_dir / output_filename
            
            # If file exists, increment version
            while output_filepath.exists():
                version += 1
                output_filename = f"{base_name}_improvement_run_v{version}.json"
                output_filepath = self.results_dir / output_filename
            
            # Save results
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Improvement results saved to: {output_filepath}")
            return str(output_filepath)
            
        except Exception as e:
            logger.error(f"Error saving improvement results: {e}")
            raise
    
    def calculate_improvement_metrics(
        self, 
        original_results: Dict[str, Any], 
        improved_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate improvement metrics comparing original and improved results"""
        
        # Original metrics
        orig_total = original_results["metadata"]["total_messages"]
        orig_positive = sum(
            1 for persona_result in original_results["results"]
            for message_result in persona_result["messages"]
            if message_result["decision"]["decision"]
        )
        orig_rate = (orig_positive / orig_total) * 100 if orig_total > 0 else 0
        
        # Improved metrics
        imp_total = improved_results["metadata"]["total_messages"]
        imp_positive = sum(
            1 for persona_result in improved_results["results"]
            for message_result in persona_result["messages"]
            if message_result["decision"]["decision"]
        )
        imp_rate = (imp_positive / imp_total) * 100 if imp_total > 0 else 0
        
        # Calculate improvement
        rate_improvement = imp_rate - orig_rate
        relative_improvement = ((imp_rate - orig_rate) / orig_rate) * 100 if orig_rate > 0 else 0
        
        return {
            "original": {
                "positive_responses": orig_positive,
                "total_messages": orig_total,
                "response_rate": orig_rate
            },
            "improved": {
                "positive_responses": imp_positive,
                "total_messages": imp_total,
                "response_rate": imp_rate
            },
            "improvement": {
                "absolute_improvement": rate_improvement,
                "relative_improvement": relative_improvement
            }
        }
    
    def run_improvement_cycle(
        self, 
        results_filename: str, 
        messages_per_persona: int = 1
    ) -> Dict[str, Any]:
        """Run complete improvement cycle: analyze, improve, evaluate"""
        
        logger.info("="*60)
        logger.info("STARTING PROMPT IMPROVEMENT CYCLE")
        logger.info("="*60)
        
        # Step 1: Load original results
        logger.info("\n\n🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜\n\nStep 1: Loading original results...")
        original_results = self.load_results_file(results_filename)
        original_prompt = original_results["metadata"]["system_prompt"]
        
        # Handle both original results and improvement results
        if "persona_file" in original_results["metadata"] and original_results["metadata"]["persona_file"] != "extracted_from_previous_results":
            persona_filename = original_results["metadata"]["persona_file"]
        else:
            # For improvement runs, extract personas directly from results
            logger.info("No persona_file in metadata or file was extracted from previous results - extracting personas from results...")
            personas = [result["persona"] for result in original_results["results"]]
            persona_filename = None
        
        # Step 2: Extract reasoning
        logger.info("\n\n🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜\n\nStep 2: Extracting decision maker reasoning...")
        all_reasoning = self.extract_all_reasoning(original_results)
        
        # Step 3: Improve system prompt
        logger.info("\n\n🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜\n\nStep 3: Improving system prompt...")
        improved_prompt = self.improve_system_prompt(original_prompt, all_reasoning)
        
        logger.info("PROMPT IMPROVEMENT:")
        logger.info("-" * 40)
        logger.info("ORIGINAL PROMPT:")
        logger.info(original_prompt)
        logger.info("-" * 40)
        logger.info("IMPROVED PROMPT:")
        logger.info(improved_prompt)
        logger.info("-" * 40)
        
        # Step 4: Load original personas
        logger.info("\n\n🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜\n\nStep 4: Loading original personas...")
        if persona_filename is not None:
            personas = self.load_original_personas(persona_filename)
        # personas were already extracted above if persona_filename is None
        
        # Step 5: Run evaluation with improved prompt
        logger.info("\n\n🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜\n\nStep 5: Running evaluation with improved prompt...")
        improved_results = self.run_evaluation_with_improved_prompt(
            personas, improved_prompt, messages_per_persona
        )
        
        # Step 6: Save results
        logger.info("\n\n🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜\n\nStep 6: Saving improvement results...")
        output_file = self.save_improvement_results(improved_results, results_filename)
        
        # Step 7: Calculate metrics
        logger.info("\n\n🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜\n\nStep 7: Calculating improvement metrics...")
        metrics = self.calculate_improvement_metrics(original_results, improved_results)
        
        # Print final summary
        logger.info("="*60)
        logger.info("\n\n🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜🪜\n\nIMPROVEMENT CYCLE COMPLETE")
        logger.info("="*60)
        logger.info(f"Original response rate: {metrics['original']['response_rate']:.1f}% ({metrics['original']['positive_responses']}/{metrics['original']['total_messages']})")
        logger.info(f"Improved response rate: {metrics['improved']['response_rate']:.1f}% ({metrics['improved']['positive_responses']}/{metrics['improved']['total_messages']})")
        logger.info(f"Absolute improvement: {metrics['improvement']['absolute_improvement']:+.1f} percentage points")
        logger.info(f"Relative improvement: {metrics['improvement']['relative_improvement']:+.1f}%")
        logger.info(f"Results saved to: {output_file}")
        logger.info("="*60)
        
        return {
            "improved_results": improved_results,
            "metrics": metrics,
            "output_file": output_file,
            "improved_prompt": improved_prompt
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Improve system prompts based on decision maker feedback"
    )
    
    parser.add_argument(
        "--results-file",
        required=True,
        help="Name of the results JSON file in data/results/ directory"
    )
    
    parser.add_argument(
        "--messages-per-persona",
        type=int,
        default=1,
        help="Number of messages to generate per persona in improvement run (default: 1)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize service
        service = PromptImprovementService()
        
        # Run improvement cycle
        results = service.run_improvement_cycle(
            args.results_file,
            args.messages_per_persona
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Improvement cycle interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Improvement cycle failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())