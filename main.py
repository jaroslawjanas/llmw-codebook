import argparse
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
from src.llm_watermark import LLMWatermarkEncoder, LLMWatermarkDecoder
from src.codebook import CodebookManager
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="LLM Text Watermarking based on Lagrange Interpolation")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green/red tokens")
    parser.add_argument("--n", type=int, default=16, help="Size of the code table")
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
    
    # Generate a deterministic 64-bit secret key based on seed
    secret_key_int = random.getrandbits(64)
    secret_key = f"{secret_key_int:016x}"  # Format as 16-character hex string
    if args.verbose:
        print(f"Generated seed based 64-bit secret key: {secret_key}\n")

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
    
    output_subdir = f"{model_name_clean}_{dataset_name_clean}_n{args.n}_u{args.u}_{timestamp}"
    output_dir = os.path.join("output", output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    stats_file = os.path.join(output_dir, "statistics.csv")
    stats_parquet_file = os.path.join(output_dir, "statistics.parquet")
    
    # Initialize DataFrame with all required columns
    columns = [
        "n", "u", "prompt", "generated_text", "tokens_length", "secret_key", "properly_encoded_tokens",
        "watermark_recovered", "encoding_time", "decoding_time", "codebook_search_time"
    ]
    
    # Define explicit column types for proper data handling
    column_dtypes = {
        "n": "int64",
        "u": "int64",
        "prompt": "string",
        "generated_text": "string", 
        "tokens_length": "int64",
        "secret_key": "string",
        "properly_encoded_tokens": "int64",
        "watermark_recovered": "boolean",
        "encoding_time": "float64",
        "decoding_time": "float64",
        "codebook_search_time": "float64"
    }
    
    stats_df = pd.DataFrame(index=range(total_prompts), columns=columns)

    #----------------------------------------#
    #              Watermarking              #
    #----------------------------------------#

    # Create watermarker instance
    watermarker = LLMWatermarkEncoder(
        model_name=args.model,
        n=args.n,
        u=args.u,
        secret_key=secret_key,
        green_list_fraction=args.green_fraction,
        bias=args.bias,
        seed=args.seed,
        device=device,
        context_window=args.context_window,
        temperature=args.temperature,
        hash_window=args.hash_window,
        verbose=args.verbose
    )

    codebook = CodebookManager(
        n=args.n,
        u=args.u,
        secret_key=secret_key,
        seed=args.seed
    )

    successful_watermark_recoveries = 0
    top_watermark_recoveries = 0
    successful_hamming_distances = []  # For top 1 matches
    top100_hamming_distances = []     # For top 100 matches
    full_generation_prompts = 0        # Count prompts with full max-tokens generation

    for prompt_idx, prompt in enumerate(prompts, 0):

        bit_code, watermark_identity = codebook.get_random_row(args.seed + prompt_idx)

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
            bit_code=bit_code,
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
            n=args.n,
            u=args.u,
            green_list_fraction=args.green_fraction,
            seed=args.seed,
            device=device,
            verbose=args.verbose
        )
        
        decoding_start = time.time()
        code_blocks, decoded_tokens_length = decoder.decode_text(generated_ids=generated_ids)
        decoding_time = time.time() - decoding_start
        
        codebook_search_time = 0
        successful_code_matches = 0
        top_code_matches = 0

        # Skip watermark recovery if generated text is shorter than max-tokens
        if len(generated_ids) < args.max_tokens:
            if args.verbose:
                print(f"Skipping watermark recovery: generated {len(generated_ids)} tokens < {args.max_tokens} max tokens")
            continue
        
        # Count prompts with full generation
        full_generation_prompts += 1
        
        for code_block in code_blocks:
            # Get the top 100 closest matches, handling potential None return
            codebook_search_start = time.time()
            matches = codebook.find_closest_matches(code_block, top=100)
            codebook_search_time += time.time() - codebook_search_start
            
            if matches is None:
                continue

            # Check if the correct code is in top 100
            found_in_top = False
            for i, (recovered_code, recovered_identity, hamming_distance) in enumerate(matches):
                is_successful = np.array_equal(recovered_code, bit_code)
                if is_successful:
                    found_in_top = True
                    # Record hamming distance for top 100 recoveries
                    top100_hamming_distances.append(hamming_distance)
                    if args.verbose:
                        print(f"Successful recovery at position {i+1} with hamming distance: {hamming_distance}")
                    if i == 0:  # First match (top 1)
                        successful_code_matches += 1
                        # Record hamming distance for successful (top 1) recoveries
                        successful_hamming_distances.append(hamming_distance)
                    break
            
            if found_in_top:
                top_code_matches += 1

        if successful_code_matches >= 1:
            successful_watermark_recoveries += 1
            
        if top_code_matches >= 1:
            top_watermark_recoveries += 1

    # Create results file with parameters and statistics
    results_file = os.path.join(output_dir, "results.txt")
    
    with open(results_file, 'w') as f:
        # Write parameter information at the top
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT PARAMETERS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {dataset_name}/{dataset_subset} ({dataset_split} split, {dataset_column} column)\n")
        f.write(f"n (code table size): {args.n}\n")
        f.write(f"u (code size): {args.u}\n")
        f.write(f"Number of prompts: {args.n_prompts}\n")
        f.write(f"Max tokens: {args.max_tokens}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Secret key: {secret_key}\n")
        f.write(f"Green fraction: {args.green_fraction}\n")
        f.write(f"Bias: {args.bias}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\n")
        
        # Write results
        f.write("=" * 60 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total prompts processed: {total_prompts}\n")
        f.write(f"Prompts with full generation: {full_generation_prompts}\n")
        f.write(f"Successful watermark recoveries (top 1): {successful_watermark_recoveries} out of {full_generation_prompts}\n")
        f.write(f"Watermark recoveries in top 100: {top_watermark_recoveries} out of {full_generation_prompts}\n")
        
        if full_generation_prompts > 0:
            success_rate = successful_watermark_recoveries / full_generation_prompts * 100
            top100_rate = top_watermark_recoveries / full_generation_prompts * 100
            f.write(f"Success rate (top 1): {success_rate:.2f}%\n")
            f.write(f"Success rate (top 100): {top100_rate:.2f}%\n")
        
        f.write("\n")
        
        # Write hamming distance statistics
        if successful_hamming_distances:
            avg_successful_hamming = np.mean(successful_hamming_distances)
            min_successful_hamming = np.min(successful_hamming_distances)
            max_successful_hamming = np.max(successful_hamming_distances)
            f.write(f"Average hamming distance for successful recoveries (top 1): {avg_successful_hamming:.2f}\n")
            f.write(f"Min hamming distance for successful recoveries (top 1): {min_successful_hamming}\n")
            f.write(f"Max hamming distance for successful recoveries (top 1): {max_successful_hamming}\n")
        else:
            f.write("No successful recoveries (top 1) to calculate hamming distance statistics\n")
        
        f.write("\n")
        
        if top100_hamming_distances:
            avg_top100_hamming = np.mean(top100_hamming_distances)
            min_top100_hamming = np.min(top100_hamming_distances)
            max_top100_hamming = np.max(top100_hamming_distances)
            f.write(f"Average hamming distance for top 100 recoveries: {avg_top100_hamming:.2f}\n")
            f.write(f"Min hamming distance for top 100 recoveries: {min_top100_hamming}\n")
            f.write(f"Max hamming distance for top 100 recoveries: {max_top100_hamming}\n")
        else:
            f.write("No top 100 recoveries to calculate hamming distance statistics\n")

    print(f"Results saved to: {results_file}")
    print(f"Made {successful_watermark_recoveries} successful watermark recoveries out of {full_generation_prompts}")
    print(f"Made {top_watermark_recoveries} watermark recoveries in top 100 out of {full_generation_prompts}")
    
    # Calculate and display average hamming distances
    if successful_hamming_distances:
        avg_successful_hamming = np.mean(successful_hamming_distances)
        print(f"Average hamming distance for successful recoveries (top 1): {avg_successful_hamming:.2f}")
    else:
        print("No successful recoveries (top 1) to calculate average hamming distance")
    
    if top100_hamming_distances:
        avg_top100_hamming = np.mean(top100_hamming_distances)
        print(f"Average hamming distance for top 100 recoveries: {avg_top100_hamming:.2f}")
    else:
        print("No top 100 recoveries to calculate average hamming distance")

if __name__ == "__main__":
    main()
