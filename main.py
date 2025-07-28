import argparse
import galois
import random
import numpy as np
import torch
import pandas as pd
import json
import os
import time
from datetime import datetime
import src.paths as paths
from src.utils import get_shuffled_essays
from src.llm_watermark import LLMWatermarkEncoder, LLMWatermarkDecoder, MCPSolver
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="LLM Text Watermarking based on Lagrange Interpolation")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=304, help="Maximum tokens to generate")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green/red tokens")
    parser.add_argument("--n", type=int, default=8, help="Size of the field and the code table")
    parser.add_argument("--u", type=int, default=100, help="Size of the codes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, help="Custom prompt (uses random essay from dataset if not provided)")
    parser.add_argument("--dataset", type=str, nargs=4, default=["ChristophSchuhmann/essays-with-instructions", "default", "train", "instructions"], 
                        help="Hugging Face dataset to use for prompts. Format: '<org>/<dataset name>' '<subset>' '<split>' '<column>'. Default: 'ChristophSchuhmann/essays-with-instructions' 'default' 'train' 'instructions'")
    parser.add_argument("--cache-dir", type=str, default=paths.CACHE_DIR, help="Cache directory for models and datasets")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--context-window", type=int, default=1500, help="Maximum number of tokens to use as context for generation (default: 1500)")
    parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy sampling, higher = more random)")
    parser.add_argument("--hash-window", type=int, default=1, help="Number of previous tokens to hash together (default: 1)") 
    parser.add_argument("--n-prompts", type=int, default=1, help="Number of prompts to process (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output and progress information")
    parser.add_argument("--stats", action="store_true", help="Show statistics summary in console (statistics are always saved to file)")
    parser.add_argument("--force-tokenization", action="store_true", help="Force detokenization and retokenization of text for decoding instead of using token IDs directly")

    args = parser.parse_args()

    # Set global cache
    paths.set_cache_dir(args.cache_dir)
    paths.ensure_directories()

    # Initialize random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine device
    if args.no_cuda:
        device = "cpu"
        print(f"Device: {device} (CUDA disabled by --no-cuda flag)\n")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"Device: {device} (CUDA available and enabled)\n")
        else:
            print(f"Device: {device} (CUDA not available, falling back to CPU)\n")
        
    # Create a Galois field of size 2^n
    field_size = 2 ** args.n
    gf = galois.GF(field_size)
    
    
    # Generate a deterministic 64-bit secret key based on seed
    secret_key_int = random.getrandbits(64)
    secret_key = f"{secret_key_int:016x}"  # Format as 16-character hex string
    if args.verbose:
        print(f"Generated seed based 64-bit secret key: {secret_key}\n")
    
    # Create watermarker instance
    watermarker = LLMWatermarkEncoder(
        model_name=args.model,
        green_list_fraction=args.green_fraction,
        bias=args.bias,
        seed=args.seed,
        device=device,
        context_window=args.context_window,
        temperature=args.temperature,
        hash_window=args.hash_window,
        verbose=args.verbose
    )

    # Get dataset info
    dataset_name = args.dataset[0]
    dataset_subset = args.dataset[1]
    dataset_split = args.dataset[2]
    dataset_column = args.dataset[3]

    # Get prompts
    if args.prompt:
        # Single custom prompt
        prompts = [args.prompt]
        if args.verbose:
            print(f"\n--- Using Custom Prompt ---")
            print(args.prompt[:200] + "..." if len(args.prompt) > 200 else args.prompt)
            print("---------------------------\n")

    else:
        # Get shuffled essays from dataset
        prompts = get_shuffled_essays(
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            dataset_split=dataset_split,
            dataset_column=dataset_column,
            seed=args.seed,
            n_prompts=args.n_prompts
        )
        if args.verbose:
            print(f"\n--- Prompt from {dataset_name}/{dataset_subset} ({dataset_split} split, {dataset_column} column) ---")
            print(prompts[0][:200] + "..." if len(prompts[0]) > 200 else prompts[0])
            print(f"{'='*60}\n")
    
    total_prompts = len(prompts)
    if args.verbose:
        print(f"Generating {args.max_tokens} tokens per prompt with watermarking...")
        print(f"Using dataset: {dataset_name}")
        print(f"Processing {total_prompts} prompt(s) with model: {args.model}")

    # Create dynamic output directory and file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = args.model.replace("/", "_")
    dataset_name_clean = dataset_name.replace("/", "_")
    
    output_subdir = f"{model_name_clean}_{dataset_name_clean}_n{args.n}_{timestamp}"
    output_dir = os.path.join("output", output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    stats_file = os.path.join(output_dir, "statistics.csv")
    stats_parquet_file = os.path.join(output_dir, "statistics.parquet")
    
    # Initialize DataFrame with all required columns
    columns = [
        "field_size", "prompt", "generated_text", "tokens_length", "secret_key", "properly_encoded_tokens",
        "watermark_recovered", "encoding_time", "decoding_time"
    ]
    
    # Define explicit column types for proper data handling
    column_dtypes = {
        "field_size": "int64",
        "prompt": "string",
        "generated_text": "string", 
        "tokens_length": "int64",
        "secret_key": "string",
        "properly_encoded_tokens": "int64",
        "watermark_recovered": "boolean",
        "encoding_time": "float64",
        "decoding_time": "float64"
    }
    
    stats_df = pd.DataFrame(index=range(total_prompts), columns=columns)

    for prompt_idx, prompt in enumerate(prompts, 0):
        if total_prompts > 1:
            print(f"\n{'='*60}")
            print(f"Processing Prompt {prompt_idx+1}/{total_prompts}")
            if args.verbose:
                print(f"{'='*60}")
                print(f"Prompt preview: {prompt[:100] + '...' if len(prompt) > 100 else prompt}\n")
        
        if args.verbose:
            # Encoding
            print(f"\n{'-'*60}")
            print("Encoding")
            print(f"{'-'*60}")

        # Time encoding
        encoding_start = time.time()
        full_text, generated_text, generated_ids, formatted_prompt, generation_statistics, watermark_blocks_info = watermarker.generate_text(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose
        )
        encoding_time = time.time() - encoding_start
        
        if args.verbose:
            print(f"{'-'*60}")
            print(f"\nPrompt:\n{prompt}")
            print(f"\nGenerated text:\n{generated_text}")

            print(f"\n{'-'*60}")

            print(f"\nGeneration statistics:")
            print(f"  Total tokens: {generation_statistics['total_tokens_generated']}")
            print(f"  Green tokens: {generation_statistics['green_tokens']}")
            print(f"  Red tokens: {generation_statistics['red_tokens']}")
            print(f"  Greeen ratio: {generation_statistics['green_ratio']}")
            
            # Decoding
            print(f"\n{'-'*60}")
            print("Decoding")
            print(f"{'-'*60}")
        
        # Create decoder with same parameters
        decoder = LLMWatermarkDecoder(
            model_name=args.model,
            secret_key=secret_key,
            green_list_fraction=args.green_fraction,
            seed=args.seed,
            device=device,
            verbose=args.verbose
        )
        

if __name__ == "__main__":
    main()
