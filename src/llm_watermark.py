#!/usr/bin/env python3
"""
LLM Watermarking Implementation based on "A Watermark for Large Language Models (Kirchenbauer et al., 2023)
https://arxiv.org/abs/2301.10226
"""

import hashlib
import random
import torch
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Union
from abc import ABC, abstractmethod
from itertools import combinations
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
from src.model_formatters import format_prompt_for_model
from src.utils import load_hf_token
import src.paths as paths


class LLMWatermarkerBase(ABC):
    """
    Abstract base class for LLM watermarking encoder and decoder.
    Contains shared functionality for both encoding and decoding operations.
    """
    
    def __init__(
        self,
        model_name: str,
        secret_key: str,
        n: int,
        u: int,
        green_list_fraction: float,
        seed: int,
        cache_dir: str,
        device: Optional[str],
        verbose: bool
    ):
        """
        Initialize the base watermarker with shared parameters.
        
        Args:
            model_name: HuggingFace model identifier
            secret_key: Secret key for watermarking
            n: Field size parameter (GF(2^n))
            green_list_fraction: Fraction of tokens to include in green list (default: 0.5)
            seed: Random seed for reproducibility
            cache_dir: Directory to cache models
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
            verbose: Whether to show detailed output
        """
        self.model_name = model_name
        self.secret_key = secret_key
        self.n = n
        self.u = u
        self.green_list_fraction = green_list_fraction
        self.seed = seed
        self.cache_dir = cache_dir
        self.device = device
        self.verbose = verbose
        
        # Load tokenizer (shared by both encoder and decoder)
        self._load_tokenizer()
        self.vocab_size = len(self.tokenizer)
        
    def _load_tokenizer(self):
        """Load the tokenizer."""
        if self.verbose:
            print(f"Loading tokenizer for: {self.model_name}")
        
        # Get HuggingFace token if available
        token = load_hf_token()
        
        # Configure tokenizer options
        tokenizer_kwargs = {
            "cache_dir": paths.MODELS_CACHE_DIR,  # Use models subdirectory
        }
        if token:
            tokenizer_kwargs["token"] = token
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            **tokenizer_kwargs
        )
        
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _get_red_green_tokens(self, token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate red and green token lists based on the hash of a token and secret key.
        This creates a deterministic division of the vocabulary into "green" and "red" tokens.
        
        Args:
            token_id: Single token ID to hash
            secret_key: Secret key for watermarking
            
        Returns:
            Tuple of (green_tokens, red_tokens) as tensors
        """
        # Create a hash from the token ID and secret key
        hash_input = f"{token_id}-{self.secret_key}"
        hash_object = hashlib.sha256(hash_input.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert the hexadecimal hash to a 32-bit integer for use as a random seed
        # This ensures deterministic outcomes for the same input token and secret key
        hash_seed = int(hash_hex, 16) % (2**32)
        
        # Create a generator and set its seed
        rng_generator = torch.Generator(device=self.device)
        rng_generator.manual_seed(hash_seed)
        
        # Use torch.randperm for efficient permutation generation
        permutation = torch.randperm(
            self.vocab_size,
            generator=rng_generator,
            requires_grad=False,
            device=self.device
        )
        
        # Split the permuted indices into "green" and "red" lists
        split_point = int(self.vocab_size * self.green_list_fraction)

        green_tokens = permutation[:split_point]
        red_tokens = permutation[split_point:]
        
        return green_tokens, red_tokens


class LLMWatermarkEncoder(LLMWatermarkerBase):
    def __init__(
        self,
        model_name: str,
        secret_key: str,
        n: int,
        u: int,
        green_list_fraction: float = 0.5,
        bias: float = 6.0,
        seed: int = 42,
        cache_dir: str = paths.CACHE_DIR,
        device: Optional[str] = "cpu",
        context_window: int = 1500,
        temperature: float = 0.0,
        hash_window: int = 1,
        ti_correction: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the watermarker with the specified model and parameters.

        Args:
            model_name: HuggingFace model identifier
            secret_key: Secret key for watermarking
            n:
            u:
            green_list_fraction: Fraction of tokens to include in green list (default: 0.5)
            bias: Logit bias to apply to green tokens (default: 6.0)
            seed: Random seed for reproducibility
            cache_dir: Directory to cache models
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
            context_window: Maximum number of tokens to use as context for generation (default: 1024)
            temperature: Sampling temperature (default: 0.0 = greedy sampling, higher = more random)
            hash_window: Number of previous tokens to hash together (default: 1)
            ti_correction: Enable tokenization inconsistency correction (default: True)
        """
        # Initialize base class
        super().__init__(model_name, secret_key, n, u, green_list_fraction, seed, cache_dir, device, verbose)

        # Encoder-specific parameters
        self.bias = bias
        self.context_window = context_window
        self.temperature = temperature
        self.hash_window = hash_window
        self.ti_correction = ti_correction
        
        # Load model (tokenizer already loaded by base class)
        self._load_model()
        
    def _load_model(self):
        """Load the model."""
        if self.verbose:
            print(f"Loading model: {self.model_name}")
        
        # Get HuggingFace token if available
        token = load_hf_token()
        
        # Configure model loading options
        model_kwargs = {
            "cache_dir": paths.MODELS_CACHE_DIR,  # Use models subdirectory
            "use_cache": True,  # Ensure K/V caching is enabled
        }
        if token:
            model_kwargs["token"] = token
        
        # Load model with appropriate settings for the device
        if self.device.startswith("cuda"):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                device_map=self.device,  # "cuda", "cuda:0", "cuda:1", etc.
                **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.model.to(self.device)
            
    def _check_token_consistency(self, generated_ids: List[int]) -> bool:
        """Check if token list survives a decode→encode round-trip with same length."""
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        retokenized = self.tokenizer.encode(text, add_special_tokens=False)
        return len(retokenized) == len(generated_ids)

    def _sample_token(
        self,
        modified_logits: torch.Tensor,
        all_ids: List[int],
        prompt_length: int,
    ) -> Tuple[Optional[int], int]:
        """
        Sample a TI-consistent token from modified logits.

        Returns:
            (token_id, ti_retries) where token_id is None if EOS was sampled.
        """
        excluded_tokens: set = set()
        max_retries = 100
        token_id: int = 0

        for _ in range(max_retries):
            if excluded_tokens:
                sampling_logits = modified_logits.clone()
                exc = torch.tensor(list(excluded_tokens), device=sampling_logits.device)
                sampling_logits[0, 0, exc] = float('-inf')
            else:
                sampling_logits = modified_logits

            probs = torch.nn.functional.softmax(sampling_logits, dim=-1)

            if self.temperature == 0 or self.temperature < 1e-6:
                token_id = int(torch.argmax(probs, dim=-1).item())
            else:
                token_id = int(torch.multinomial(probs.squeeze(), 1).item())

            # EOS — don't add to sequence, signal early stop
            if token_id == self.tokenizer.eos_token_id:
                return None, len(excluded_tokens)

            all_ids.append(token_id)

            if not self.ti_correction or self._check_token_consistency(all_ids[prompt_length:]):
                return token_id, len(excluded_tokens)

            # Inconsistent — remove and retry
            all_ids.pop()
            excluded_tokens.add(token_id)

        # Max retries exhausted — keep last token
        all_ids.append(token_id)
        return token_id, len(excluded_tokens)

    def _trim_past_key_values(self, past_key_values, trim_amount: int):
        """
        Trim the K/V cache by removing the oldest entries.
        Supports both legacy tuple format and modern Cache objects (transformers 4.36+).

        This method is essential for K/V caching because:
        1. Memory Management: K/V caches grow linearly with sequence length, potentially causing OOM errors
        2. Context Window Limits: Models have maximum context windows (e.g., 2048, 4096 tokens)
        3. Performance: Without trimming, the cache would grow indefinitely during long text generation
        4. Correctness: Prevents exceeding model's positional encoding limits

        The method removes the oldest tokens from the beginning of the cache while preserving
        the most recent context, which is typically most relevant for generation.

        Args:
            past_key_values: Tuple of past key values OR a Cache object from the model
            trim_amount: Number of positions to remove from the beginning

        Returns:
            Trimmed past_key_values (same type as input)
        """
        # Handle modern DynamicCache objects (transformers 4.36+)
        # These have key_cache and value_cache attributes as lists of tensors
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            # Trim in place by slicing each layer's tensors
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, trim_amount:, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, trim_amount:, :]
            return past_key_values

        # Handle legacy tuple format (older transformers versions)
        # Each layer_past is a tuple of (key, value) tensors
        trimmed_past = []
        for layer_past in past_key_values:
            # Each layer_past is a tuple of (key, value) tensors
            # Shape: (batch_size, num_heads, sequence_length, head_dim)
            key, value = layer_past

            # Trim the sequence dimension (dimension 2) by removing oldest tokens
            trimmed_key = key[:, :, trim_amount:, :]
            trimmed_value = value[:, :, trim_amount:, :]

            # Append the trimmed tuple for this layer
            trimmed_past.append((trimmed_key, trimmed_value))

        return tuple(trimmed_past)
        
    def _modify_logits(self, logits: torch.Tensor, token_id: int, bias_type: str) -> torch.Tensor:
        """
        Modify logits by adding bias to either green or red tokens.
        This is the core watermarking function that biases the model's predictions
        toward selecting tokens from the specified list determined by previous token and secret key.
        
        Args:
            logits: Original logits from the model (prediction scores for each token)
            token_id: Previous token ID to use for hashing
            secret_key: Secret key for watermarking
            bias_type: Type of tokens to bias ("green" or "red")
            
        Returns:
            Modified logits tensor with bias added to the specified tokens
        """
        # Ensure logits are in the right shape (batch_size, seq_len, vocab_size)
        # The tensor might be (1, 1, vocab_size) or just (1, vocab_size)
        if len(logits.shape) == 3:
            # Get the last token's logits and flatten to 1D
            # Example: if logits is tensor([[[1.2, 0.8, -0.5, ...]]])
            # This becomes tensor([1.2, 0.8, -0.5, ...])
            logits = logits[0, -1, :]
        elif len(logits.shape) == 2:
            # Already (batch_size, vocab_size), get first batch
            # Example: if logits is tensor([[1.2, 0.8, -0.5, ...]])
            # This becomes tensor([1.2, 0.8, -0.5, ...])
            logits = logits[0, :]
        
        # Get vocabulary size from logits
        # Example: if logits tensor has 50,000 elements, vocab_size will be 50,000
        vocab_size = logits.shape[-1]
        
        # Get green and red tokens as tensors using the deterministic hash of previous token and secret key
        # Example: based on token_id 101 and secret key, this might return
        # green_tokens as tensor([42, 900, 5, ...]) and red_tokens (which we don't use here)
        green_tokens, red_tokens = self._get_red_green_tokens(token_id)
        
        # Move tensors to the same device as logits
        green_tokens = green_tokens.to(logits.device)
        red_tokens = red_tokens.to(logits.device)
        
        # Clone logits for modification to avoid affecting the original tensor
        # This creates a new tensor with the same values that we can safely modify
        modified_logits = logits.clone()
        
        if bias_type == "green":
            # Filter green tokens to ensure they're within vocabulary bounds using tensor operations
            # Example: if vocab_size is 50,000 and green_tokens contains a value 60,000,
            # we create a boolean mask tensor([True, True, True, False, ...]) for values < 50,000
            mask = green_tokens < vocab_size
            # Apply the mask to get only valid green tokens
            # Example: valid_green_tokens becomes tensor([42, 900, 5, ...]) without any out-of-bounds tokens
            valid_tokens_to_bias = green_tokens[mask]
        elif bias_type == "red":
            # Filter red tokens to ensure they're within vocabulary bounds
            mask = red_tokens < vocab_size
            valid_tokens_to_bias = red_tokens[mask]
        else:
            raise ValueError("bias_type must be 'green' or 'red'")
        
        # Vectorized bias application - apply bias to all valid tokens at once
        # This means tokens in the specified list will have higher probability of being selected
        if valid_tokens_to_bias.numel() > 0:  # Only proceed if we have valid tokens
            # Example: if self.bias is 6.0, this adds 6.0 to the logits for all specified tokens
            # For the token indices in valid_tokens_to_bias
            modified_logits[valid_tokens_to_bias] += self.bias
        
        # Reshape back to original format for compatibility with the model's expectations
        # Example: from tensor([1.2, 0.8, -0.5, ...]) back to tensor([[[1.2, 0.8, -0.5, ...]]])
        return modified_logits.unsqueeze(0).unsqueeze(0)
        
    def generate_text(
        self,
        prompt: str,
        bit_code: List[int],
    ) -> Tuple[str, str, List[int], str, Dict[str, int]]:
        """
        Generate watermarked text by encoding a single codeword (u tokens).

        Args:
            prompt: Input text prompt
            bit_code: Binary codeword to encode (length must equal u)

        Returns:
            Tuple of (full_text, generated_text, generated_ids, formatted_prompt, statistics)
        """
        # Format the prompt for the specific model
        formatted_prompt = format_prompt_for_model(prompt, self.model_name, self.tokenizer, verbose=self.verbose)

        # Tokenize the formatted prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

        # Store generated ids
        all_ids = input_ids.clone()[0].tolist()

        # Initialize K/V cache
        past_key_values = None
        cache_position = 0

        assert len(bit_code) == self.u
        prompt_length = len(input_ids[0])

        # Setup progress tracking
        progress_bar = tqdm(range(self.u), disable=not self.verbose)
        tokens_generated = 0
        green_tokens_selected = 0
        red_tokens_selected = 0
        properly_encoded_tokens = 0
        ti_correction_count = 0

        # Generate u tokens, one per bit in the codeword
        for bit_idx in range(self.u):
            # Prepare input for the model
            if past_key_values is None:
                # First generation - use the full prompt
                if len(all_ids) > self.context_window:
                    model_input_ids = torch.tensor([all_ids[-self.context_window:]], device=self.device)
                    cache_position = 0
                else:
                    model_input_ids = torch.tensor([all_ids], device=self.device)
            else:
                # Subsequent generations - only use the last generated token
                model_input_ids = torch.tensor([[all_ids[-1]]], device=self.device)

                # Check if we need to trim the cache due to context window limits
                if cache_position >= self.context_window:
                    trim_amount = cache_position - self.context_window + 1
                    past_key_values = self._trim_past_key_values(past_key_values, trim_amount)
                    cache_position = self.context_window - 1

            # Get logits from the model with K/V caching
            with torch.no_grad():
                outputs = self.model(
                    model_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits[:, -1:, :]
                past_key_values = outputs.past_key_values
                cache_position += model_input_ids.shape[1]

            # Determine bias type based on current bit
            current_bit = bit_code[bit_idx]
            bias_type = "green" if current_bit == 1 else "red"

            # For vocabulary splitting - use token_id = 0 for very first token, otherwise use previous generated token
            if tokens_generated == 0:
                current_previous_token = 0
            else:
                current_previous_token = all_ids[-1]

            # Modify logits with watermark
            modified_logits = self._modify_logits(logits, current_previous_token, bias_type)

            # Apply temperature if set
            if self.temperature > 0:
                modified_logits = modified_logits / self.temperature

            # Sample token with TI correction
            next_token_id, ti_retries = self._sample_token(modified_logits, all_ids, prompt_length)
            ti_correction_count += ti_retries

            if next_token_id is None:
                break  # EOS

            tokens_generated += 1

            # Track whether the selected token was from the green or red list
            green_tokens, red_tokens = self._get_red_green_tokens(current_previous_token)

            vocab_size = len(self.tokenizer)
            if next_token_id < vocab_size:
                next_token_tensor = torch.tensor(next_token_id, device=self.device)
                is_green = (green_tokens == next_token_tensor).any().item()
                encoded_bit = 1 if is_green else 0

                if is_green:
                    green_tokens_selected += 1
                else:
                    red_tokens_selected += 1

                if current_bit == encoded_bit:
                    properly_encoded_tokens += 1

            # Update progress bar
            progress_bar.set_description(f"Bit {bit_idx+1}/{self.u}, Green: {green_tokens_selected}, Red: {red_tokens_selected}")
            progress_bar.update(1)

        progress_bar.close()

        # Separate prompt from generated content
        if len(all_ids) > prompt_length:
            generated_ids = all_ids[prompt_length:]
            assert len(generated_ids) == tokens_generated
        else:
            generated_ids = []

        full_text = self.tokenizer.decode(all_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""

        statistics = {
            "green_tokens": green_tokens_selected,
            "red_tokens": red_tokens_selected,
            "total_tokens_generated": tokens_generated,
            "green_ratio": green_tokens_selected / (green_tokens_selected + red_tokens_selected + 1e-10),
            "properly_encoded_tokens": properly_encoded_tokens,
            "ti_correction_count": ti_correction_count
        }

        return full_text, generated_text, generated_ids, formatted_prompt, statistics


class LLMWatermarkDecoder(LLMWatermarkerBase):
    """
    Decoder for extracting watermark information from LLM-generated text.
    Reverses the encoding process to extract x-coordinates and y-bit sequences.
    """
    
    def __init__(
        self,
        model_name: str,
        secret_key: str,
        n: int,
        u: int,
        green_list_fraction: float = 0.5,
        seed: int = 42,
        cache_dir: str = paths.CACHE_DIR,
        device: Optional[str] = "cpu",
        verbose: bool = False,
    ):
        """
        Initialize the decoder with the same parameters used for encoding.
        
        Args:
            model_name: HuggingFace model identifier (same as used for encoding)
            secret_key: Secret key for watermarking (same as used for encoding)
            n: Field size parameter (GF(2^n)) (same as used for encoding)
            green_list_fraction: Fraction of tokens in green list (same as used for encoding)
            seed: Random seed for reproducibility (same as used for encoding)
            cache_dir: Directory to cache models
            device: Device to run on ('cuda', 'cpu')
            verbose: Whether to show detailed output
            error_correction_k: Number of bits to flip for error correction (0 = disabled, must be < n)
        """
        # Initialize base class (loads tokenizer)
        super().__init__(model_name, secret_key, n, u, green_list_fraction, seed, cache_dir, device, verbose)

        
    def decode(self, generated: Union[str, List[int]]) -> Tuple[List[int], int, List[int]]:
        """
        Decode watermark information from generated text or pre-tokenized IDs.

        Args:
            generated: Union[str, List[int]] - Either the generated text (prompt removed) or a list of token IDs

        Returns:
            Tuple of (bit_code, decoded_tokens_length, token_ids):
            - bit_code: Extracted bit sequence (list of 0/1), may be shorter than u if fewer tokens available
            - decoded_tokens_length: Number of tokens used for decoding
            - token_ids: The token IDs used for decoding
        """
        # Determine input type: list of ids or string text
        if isinstance(generated, list):
            token_ids = generated
            decoded_tokens_length = len(token_ids)
        elif isinstance(generated, str):
            # Tokenize the generated-only text directly without any formatting
            token_ids = self.tokenizer.encode(generated, add_special_tokens=False)
            decoded_tokens_length = len(token_ids)
        else:
            raise TypeError(f"generated must be str or List[int], got {type(generated).__name__}")

        # Extract bits by classifying each token as green (1) or red (0)
        bit_code = []
        for i, token_id in enumerate(token_ids[:self.u]):
            if i == 0:
                vocab_split_token = 0
            else:
                vocab_split_token = token_ids[i - 1]

            green_tokens, red_tokens = self._get_red_green_tokens(vocab_split_token)
            is_green = (green_tokens == token_id).any().item()
            bit_code.append(1 if is_green else 0)

        return bit_code, decoded_tokens_length, token_ids
