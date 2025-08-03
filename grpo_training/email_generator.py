import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import List, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache

logger = logging.getLogger(__name__)

class EmailGenerator:
    """
    Email generator using Qwen3-0.6B for cold email generation.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_qlora: bool = True,
        lora_config: Optional[dict] = None
    ):
        """
        Initialize the email generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on (auto-detected if None)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            use_qlora: Whether to use QLoRA for efficient training
            lora_config: LoRA configuration dictionary
        """
        # Enhanced device detection for Mac M4 GPU support
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
        else:
            self.device = "cpu"
        self.max_length = max_length
        self.use_qlora = use_qlora
        
        logger.info(f"Loading model {model_name} on {self.device} with QLoRA: {use_qlora}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config for QLoRA (only on CUDA)
        quantization_config = None
        if use_qlora and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_qlora and self.device == "mps":
            logger.warning("QLoRA quantization not supported on MPS (Apple Silicon). Disabling quantization but keeping LoRA.")
            use_qlora = False  # Disable quantization but we'll keep LoRA
            
        # Load base model with proper dtype and device handling
        if self.device == "cuda":
            torch_dtype = torch.float16
            device_map = "auto"
        elif self.device == "mps":
            torch_dtype = torch.float32  # Use float32 on MPS for numerical stability
            device_map = None
        else:
            torch_dtype = torch.float32
            device_map = None
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Apply LoRA if using QLoRA
        if use_qlora:
            # Default LoRA config
            default_lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
                "target_modules": self._get_target_modules(model_name)
            }
            
            # Override with provided config
            if lora_config:
                default_lora_config.update(lora_config)
            
            # Create LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **default_lora_config
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"Applied LoRA with config: {default_lora_config}")
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        elif self.device in ["mps", "cpu"]:
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0
        )
        
        # Performance optimizations
        self._prompt_cache = {}
        self.max_workers = min(mp.cpu_count(), 8)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    @lru_cache(maxsize=1000)
    def create_email_prompt(self, persona: str) -> str:
        """
        Create a prompt for email generation.
        
        Args:
            persona: Target persona description
            context: Additional context (optional)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Name: John Smith
Company: FlowAI
Industry: B2B SaaS workflow automation 
Product: AI-powered workflow automation platform for SMBs and enterprises
Value Proposition: Reduce manual work by 60% and increase team productivity through intelligent automation
Target Market: Operations managers, CTOs, and business owners with 10-2000 employees
Stage: Series A startup with proven traction
Focus: Companies struggling with repetitive manual processes, operational overhead, and scaling challenges

You work for flow, and are reaching out to sell your product to the following persona. Write a professional cold outreach email for the following persona:

Persona: Lisa Wang - Head of Operations at FoodieConnect, a 65-person B2B restaurant technology platform. Age 35, based in New York, NY. Operations background at previous food-tech startups, self-taught in process optimization. Energetic, results-focused communication style - likes quick calls, screen shares, and immediate problem-solving. Values speed, practicality, and customer satisfaction. Personality: action-oriented, pragmatic, customer-obsessed. Decision-making factors: implementation speed, customer impact, user adoption rates. Pain points: manual customer support ticket routing causing 2-hour response delays, struggling with partner onboarding workflows, difficulty tracking feature requests across teams. Budget flexibility $25K-75K annually. Prefers rapid deployment and iterative improvements over perfect solutions.

Email: Hi Lisa,\n\nI noticed FoodieConnect's rapid growth in the restaurant tech space - congratulations! Your focus on customer experience really stands out, especially with your recent partnership expansion.\n\nHaving worked with similar food-tech platforms, I've seen how manual ticket routing and partner onboarding can create bottlenecks as companies scale. Our customers typically cut response times from 2 hours to 15 minutes using FlowAI's intelligent automation.\n\nWould you be open to a 20-minute screen share next week? I'd love to show you how we helped MealTech (similar to FoodieConnect) automate their support workflow and partner onboarding process in just 2 weeks.\n\nQuick question - would Tuesday at 11am ET work for a brief demo focused specifically on your ticket routing and onboarding challenges?\n\nBest,\n[Name]\n\nP.S. Happy to share a quick case study showing how we reduced manual ops work by 65% for another NY-based food-tech platform.

Persona: {persona}

Email: """
        
        return prompt
    
    def generate_emails(
        self,
        personas: List[str],
        num_emails_per_persona: int = 4,
        batch_size: int = 16
    ) -> List[List[str]]:
        """
        Generate multiple emails for each persona with parallel processing.
        
        Args:
            personas: List of persona descriptions
            num_emails_per_persona: Number of emails to generate per persona
            batch_size: Batch size for generation (increased default)
            
        Returns:
            List of lists, each containing emails for a persona
        """
        logger.info(f"Generating {num_emails_per_persona} emails for {len(personas)} personas (batch_size={batch_size})")
        
        # Pre-create all prompts to leverage caching
        all_prompts = []
        persona_indices = []
        
        for i, persona in enumerate(personas):
            logger.debug(f"Preparing prompts for persona {i+1}/{len(personas)}")
            # Use cached prompt creation
            base_prompt = self.create_email_prompt(persona)
            
            for _ in range(num_emails_per_persona):
                all_prompts.append(base_prompt)
                persona_indices.append(i)
        
        # Generate all emails in optimized batches
        all_generated_emails = []
        total_batches = (len(all_prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
            logger.debug(f"Processing batch {batch_idx//batch_size + 1}/{total_batches} ({len(batch_prompts)} prompts)")
            
            batch_emails = self._generate_batch(batch_prompts)
            all_generated_emails.extend(batch_emails)
        
        # Organize emails by persona
        persona_emails = [[] for _ in personas]
        for email, persona_idx in zip(all_generated_emails, persona_indices):
            persona_emails[persona_idx].append(email)
        
        # Log sample results (reduced verbosity)
        for i, (persona, emails) in enumerate(zip(personas, persona_emails)):
            logger.debug(f"Persona {i+1}: Generated {len(emails)} emails")
            if i == 0:  # Only log first persona sample
                logger.debug(f"Sample email: {emails[0][:100]}...")
        
        return persona_emails
    
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate emails for a batch of prompts with memory optimization.
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of generated emails
        """
        if not prompts:
            return []
            
        # Tokenize prompts with optimized settings
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8000,  # Increased from 256 for better context
            return_attention_mask=True
        ).to(self.device)
        
        # Generate with memory optimization
        with torch.no_grad():
            # Disable gradient checkpointing during inference to avoid caching conflicts
            original_checkpointing = False
            if hasattr(self.model, 'gradient_checkpointing'):
                original_checkpointing = self.model.gradient_checkpointing
                if original_checkpointing:
                    self.model.gradient_checkpointing_disable()
            
            try:
                # Add custom generation with numerical stability fixes
                outputs = self._generate_with_stability_fixes(
                    inputs,
                    self.generation_config
                )
            finally:
                # Restore gradient checkpointing state if it was enabled
                if hasattr(self.model, 'gradient_checkpointing_enable') and original_checkpointing:
                    self.model.gradient_checkpointing_enable()
        
        # Decode and extract emails efficiently
        emails = []
        input_lengths = inputs['attention_mask'].sum(dim=1)
        
        for output, input_length in zip(outputs, input_lengths):
            # Remove input tokens more efficiently
            generated_tokens = output[input_length:]
            
            # Decode with optimized settings
            email = self.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up email
            email = self._clean_email(email)
            emails.append(email)
            
        return emails
    
    def _generate_with_stability_fixes(self, inputs, generation_config):
        """Generate text with numerical stability fixes to prevent inf/nan/negative probabilities."""
        try:
            # First attempt: normal generation (disable use_cache to avoid conflicts)
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False,  # Disable to avoid gradient checkpointing conflicts
                do_sample=generation_config.do_sample
            )
            return outputs
            
        except RuntimeError as e:
            if "probability tensor contains either" in str(e):
                logger.warning(f"Numerical instability detected: {e}. Attempting fallback generation...")
                
                # Fallback 1: Lower temperature and use more conservative sampling
                conservative_config = GenerationConfig(
                    max_length=generation_config.max_length,
                    temperature=0.3,  # Much lower temperature
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
                
                try:
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=conservative_config,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=False,  # Disable to avoid conflicts
                        do_sample=True
                    )
                    logger.info("Fallback generation with lower temperature succeeded")
                    return outputs
                    
                except RuntimeError as e2:
                    if "probability tensor contains either" in str(e2):
                        logger.warning(f"Conservative sampling also failed: {e2}. Using greedy decoding...")
                        
                        # Fallback 2: Greedy decoding (no sampling)
                        greedy_config = GenerationConfig(
                            max_length=generation_config.max_length,
                            do_sample=False,  # Greedy decoding
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                            length_penalty=1.0
                        )
                        
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=greedy_config,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_cache=False,  # Disable to avoid conflicts
                            do_sample=False
                        )
                        logger.info("Greedy decoding fallback succeeded")
                        return outputs
                    else:
                        raise e2
            else:
                raise e
    
    def _clean_email(self, email: str) -> str:
        """
        Clean and format generated email.
        
        Args:
            email: Raw generated email
            
        Returns:
            Cleaned email
        """
        # Remove excessive whitespace
        email = "\n".join(line.strip() for line in email.split("\n"))
        
        # Remove trailing repetitions
        lines = email.split("\n")
        clean_lines = []
        for line in lines:
            if line and line not in clean_lines[-3:]:  # Avoid recent repetitions
                clean_lines.append(line)
            elif not line:
                clean_lines.append(line)
                
        return "\n".join(clean_lines).strip()
    
    def update_generation_config(self, **kwargs):
        """
        Update generation configuration.
        
        Args:
            **kwargs: Generation config parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                setattr(self.generation_config, key, value)
            else:
                logger.warning(f"Unknown generation config parameter: {key}")
    
    def _get_target_modules(self, model_name: str) -> List[str]:
        """
        Get appropriate target modules for LoRA based on model architecture.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            List of target module names
        """
        model_name_lower = model_name.lower()
        
        if "qwen" in model_name_lower:
            # Qwen models typically use these modules
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "llama" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt" in model_name_lower:
            return ["c_attn", "c_proj", "c_fc"]
        else:
            # Default fallback - common attention modules
            logger.warning(f"Unknown model architecture for {model_name}, using default target modules")
            return ["q_proj", "v_proj"]
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)