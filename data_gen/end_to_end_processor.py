#!/usr/bin/env python3
"""
End-to-end processor for B2B persona message generation and decision evaluation.

This script:
1. Loads personas from a JSON file
2. Generates N messages for each persona
3. Runs each message through the decision maker
4. Saves structured results to a JSON file

Usage:
    python end_to_end_processor.py --persona-file b2b_saas_personas_large_train.json --messages-per-persona 3
"""

import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Import our existing services
from generation_model import generate_message
from decision_maker import decide_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndToEndProcessor:
    """Processes personas through the complete pipeline"""
    
    def __init__(self):
        """Initialize the processor"""
        self.data_dir = Path("data")
        self.personas_dir = self.data_dir / "personas"
        self.results_dir = self.data_dir / "results"
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Default system prompt for message generation
        self.default_system_prompt = """
        Company: FlowAI
        Industry: B2B SaaS workflow automation 
        Product: AI-powered workflow automation platform for SMBs and enterprises
        Value Proposition: Reduce manual work by 60% and increase team productivity through intelligent automation
        Target Market: Operations managers, CTOs, and business owners with 10-2000 employees
        Stage: Series A startup with proven traction
        Focus: Companies struggling with repetitive manual processes, operational overhead, and scaling challenges
        """
    
    def load_personas(self, filename: str) -> List[str]:
        """Load personas from JSON file in data/personas directory"""
        try:
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.personas_dir / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"Personas file not found: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                personas = json.load(f)
            
            if not isinstance(personas, list):
                raise ValueError("Personas file must contain a JSON array")
            
            logger.info(f"Loaded {len(personas)} personas from {filepath}")
            return personas
            
        except Exception as e:
            logger.error(f"Error loading personas from file: {e}")
            raise
    
    def process_persona(self, persona: str, messages_per_persona: int, system_prompt: str) -> Dict[str, Any]:
        """Process a single persona through the complete pipeline"""
        try:
            persona_result = {
                "persona": persona,
                "messages": []
            }
            
            logger.info(f"Processing persona: {persona[:50]}...")
            
            for i in range(messages_per_persona):
                try:
                    # Generate message
                    logger.info(f"  Generating message {i+1}/{messages_per_persona}")
                    message = generate_message(system_prompt, persona)
                    
                    # Get decision
                    logger.info(f"  Evaluating message {i+1}/{messages_per_persona}")
                    decision = decide_response(persona, message)
                    
                    # Store result
                    message_result = {
                        "message": message,
                        "decision": decision
                    }
                    persona_result["messages"].append(message_result)
                    
                    logger.info(f"  Message {i+1} complete - Decision: {decision['decision']}")
                    
                except Exception as e:
                    logger.error(f"Error processing message {i+1} for persona: {e}")
                    # Continue with next message rather than failing the entire persona
                    continue
            
            logger.info(f"Completed persona with {len(persona_result['messages'])} messages")
            return persona_result
            
        except Exception as e:
            logger.error(f"Error processing persona: {e}")
            raise
    
    def process_all_personas(
        self, 
        personas_filename: str, 
        messages_per_persona: int = 3
    ) -> Dict[str, Any]:
        """Process all personas in the file through the complete pipeline"""
        try:
            # Use default system prompt
            system_prompt = self.default_system_prompt
            
            # Load personas
            personas = self.load_personas(personas_filename)
            
            # Initialize results structure
            results = {
                "metadata": {
                    "persona_file": personas_filename,
                    "messages_per_persona": messages_per_persona,
                    "total_personas": len(personas),
                    "total_messages": 0,
                    "processing_date": datetime.now().isoformat(),
                    "system_prompt": system_prompt.strip()
                },
                "results": []
            }
            
            logger.info(f"Starting processing of {len(personas)} personas")
            logger.info(f"Messages per persona: {messages_per_persona}")
            
            # Process each persona
            total_messages = 0
            for i, persona in enumerate(personas, 1):
                try:
                    logger.info(f"Processing persona {i}/{len(personas)}")
                    persona_result = self.process_persona(persona, messages_per_persona, system_prompt)
                    results["results"].append(persona_result)
                    total_messages += len(persona_result["messages"])
                    
                except Exception as e:
                    logger.error(f"Failed to process persona {i}: {e}")
                    # Continue with next persona rather than failing the entire batch
                    continue
            
            # Update metadata with actual totals
            results["metadata"]["total_messages"] = total_messages
            results["metadata"]["successful_personas"] = len(results["results"])
            
            logger.info(f"Processing complete!")
            logger.info(f"Successfully processed: {len(results['results'])}/{len(personas)} personas")
            logger.info(f"Total messages generated: {total_messages}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing personas: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], personas_filename: str) -> str:
        """Save results to JSON file in data/results directory"""
        try:
            # Create output filename based on input filename and timestamp
            base_name = personas_filename.replace('.json', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_results_{timestamp}.json"
            output_filepath = self.results_dir / output_filename
            
            # Save results
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_filepath}")
            return str(output_filepath)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Process B2B personas through message generation and decision evaluation pipeline"
    )
    
    parser.add_argument(
        "--persona-file",
        required=True,
        help="Name of the persona JSON file in data/personas/ directory"
    )
    
    parser.add_argument(
        "--messages-per-persona",
        type=int,
        default=3,
        help="Number of messages to generate per persona (default: 3)"
    )
    
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = EndToEndProcessor()
        
        
        # Process personas
        results = processor.process_all_personas(
            args.persona_file,
            args.messages_per_persona
        )
        
        # Save results
        output_file = processor.save_results(results, args.persona_file)
        
        # Print summary
        metadata = results["metadata"]
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Persona file: {metadata['persona_file']}")
        print(f"Messages per persona: {metadata['messages_per_persona']}")
        print(f"Total personas: {metadata['total_personas']}")
        print(f"Successfully processed: {metadata['successful_personas']}")
        print(f"Total messages generated: {metadata['total_messages']}")
        print(f"Results saved to: {output_file}")
        print("="*60)
        
        # Print decision statistics
        total_positive = sum(
            1 for persona_result in results["results"]
            for message_result in persona_result["messages"]
            if message_result["decision"]["decision"]
        )
        total_messages = metadata["total_messages"]
        if total_messages > 0:
            positive_rate = (total_positive / total_messages) * 100
            print(f"Positive responses: {total_positive}/{total_messages} ({positive_rate:.1f}%)")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())