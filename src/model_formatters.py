#!/usr/bin/env python3
"""
Model Formatters for LLM Watermarking

This module provides model-specific prompt formatting functions for different
model families. Each formatter handles the specific requirements of a model family,
such as special tokens or template requirements.
"""


from typing import Optional
from transformers import PreTrainedTokenizer
from src.utils import load_hf_token

def format_prompt_for_model(
    prompt: str, 
    model_name: str, 
    tokenizer: Optional[PreTrainedTokenizer] = None,
    verbose: bool = False
) -> str:
    """
    Format a prompt according to the requirements of the specific model.
    
    Args:
        prompt: Input text prompt
        model_name: HuggingFace model identifier
        tokenizer: Tokenizer for the model (required for some formatting)
        
    Returns:
        Formatted prompt ready for the model
    """
    # Check if we can use AutoProcessor (preferred modern approach)
    if _can_import_auto_processor():
        try:
            return format_with_processor(prompt, model_name, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[WARNING] Error using AutoProcessor for {model_name}: {e}")
                print("[WARNING] Falling back to manual formatting\n")
    
    # If processor approach failed, just return the prompt unchanged
    return prompt

def _can_import_auto_processor():
    """Check if AutoProcessor can be imported."""
    try:
        import transformers.models.auto.processing_auto
        return True
    except ImportError:
        return False

def format_with_processor(prompt: str, model_name: str, verbose: bool = False) -> str:
    """
    Format a prompt using AutoProcessor's apply_chat_template method.
    
    Args:
        prompt: Input text prompt
        model_name: HuggingFace model identifier
        
    Returns:
        Formatted prompt ready for the model
    """
    from transformers import AutoProcessor
    
    if verbose:
        print(f"[DEBUG] Using AutoProcessor chat template for: {model_name}")
    
    # Convert prompt to messages format
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Get HuggingFace token
        token = load_hf_token()
        
        # Create processor and apply chat template
        processor = AutoProcessor.from_pretrained(model_name, token=token)
        formatted_prompt = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False,
            enable_thinking=False
        )
        
        if verbose:
            print(f"[DEBUG] Successfully formatted with processor")
        return formatted_prompt
        
    except Exception as e:
        if verbose:
            print(f"[DEBUG] Error using processor template: {str(e)}")
        raise e
