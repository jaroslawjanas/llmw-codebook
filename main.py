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
    parser.add_argument("--secret-key", type=str, default="25111915", help="Secret key used for watermarking and codebook generation (string)")
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
    
    print(f"Using secret key: {args.secret_key}\n")

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
    generations_txt_file = os.path.join(output_dir, "generations.txt")
    try:
        with open(generations_txt_file, 'w', encoding='utf-8') as _f:
            _f.write("")
    except Exception as e:
        print(f"Warning: Failed to initialize generations.txt: {e}")
    
    # Initialize per-generation outputs DataFrame (only full-length generations will be recorded)
    generations_columns = ["bit_code", "watermark", "encoded_bit_code", "retrieved_watermark", "successful_watermark", "bit_code_hamming_distance"]
    generations_df = pd.DataFrame(columns=generations_columns)
    
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
        secret_key=args.secret_key,
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
        secret_key=args.secret_key
    )

    successful_watermark_recoveries = 0
    successful_hamming_distances = []  # For top-1 matches
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
            secret_key=args.secret_key,
            n=args.n,
            u=args.u,
            green_list_fraction=args.green_fraction,
            seed=args.seed,
            device=device,
            verbose=args.verbose
        )
        
        decoding_start = time.time()
        if args.force_tokenization:
            if args.verbose:
                print("Decoding using retokenization path (--force-tokenization).")
            code_blocks, decoded_tokens_length = decoder.decode_text(generated_text=generated_text)
        else:
            code_blocks, decoded_tokens_length = decoder.decode_text(generated_ids=generated_ids)
        decoding_time = time.time() - decoding_start
        
        codebook_search_time = 0

        # Skip watermark recovery if generated text is shorter than max-tokens
        if len(generated_ids) < args.max_tokens:
            if args.verbose:
                print(f"Skipping watermark recovery: generated {len(generated_ids)} tokens < {args.max_tokens} max tokens")
            continue
        
        # Count prompts with full generation
        full_generation_prompts += 1

        # Append to generations.txt for full-length generations
        try:
            with open(generations_txt_file, 'a', encoding='utf-8') as gf:
                gf.write("==========\n")
                gf.write(f"Prompt {prompt_idx+1}:\n")
                gf.write(f"{prompt}\n\n")
                gf.write("Generated:\n")
                gf.write(f"{generated_text}\n")
                gf.write("==========\n")
        except Exception as e:
            print(f"Warning: Failed to write generations.txt: {e}")
        
        # Track best top-1 match across all decoded blocks for this prompt
        best_block_idx = None
        best_identity = None
        best_distance = None
        
        for block_idx, code_block in enumerate(code_blocks):
            # Exact top-1 match using Faiss
            codebook_search_start = time.time()
            closest_bits, closest_identity, dist = codebook.find_closest_match(code_block)
            codebook_search_time += time.time() - codebook_search_start

            # Track best candidate across blocks for CSV
            if (best_distance is None) or (dist < best_distance):
                best_distance = int(dist)
                best_identity = int(closest_identity)
                best_block_idx = block_idx


        if best_identity is not None and best_identity == watermark_identity:
            successful_watermark_recoveries += 1
            if best_distance is not None:
                successful_hamming_distances.append(best_distance)

        # Append per-generation row to DataFrame (full-length generations only)
        bit_code_str = ''.join(map(str, bit_code))
        if best_block_idx is not None and best_block_idx < len(watermark_blocks_info):
            encoded_bits_block = watermark_blocks_info[best_block_idx].get("encoded_bits", [])
            encoded_bit_code_str = ''.join(map(str, encoded_bits_block)) if encoded_bits_block else ""
        else:
            encoded_bit_code_str = ""
        retrieved_watermark_val = int(best_identity) if best_identity is not None else pd.NA
        bit_code_hamming_distance_val = int(best_distance) if best_distance is not None else pd.NA
        successful_watermark_flag = bool(best_identity == watermark_identity) if best_identity is not None else False

        generations_df.loc[len(generations_df)] = {
            "bit_code": bit_code_str,
            "watermark": int(watermark_identity),
            "encoded_bit_code": encoded_bit_code_str,
            "retrieved_watermark": retrieved_watermark_val,
            "successful_watermark": successful_watermark_flag,
            "bit_code_hamming_distance": bit_code_hamming_distance_val,
        }

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
        f.write(f"Secret key: {args.secret_key}\n")
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
        
        if full_generation_prompts > 0:
            success_rate = successful_watermark_recoveries / full_generation_prompts * 100
            f.write(f"Success rate (top 1): {success_rate:.2f}%\n")
        
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
        

    print(f"Results saved to: {results_file}")
    print(f"Made {successful_watermark_recoveries} successful watermark recoveries out of {full_generation_prompts}")
    
    # Calculate and display average hamming distances
    if successful_hamming_distances:
        avg_successful_hamming = np.mean(successful_hamming_distances)
        print(f"Average hamming distance for successful recoveries (top 1): {avg_successful_hamming:.2f}")
    else:
        print("No successful recoveries (top 1) to calculate average hamming distance")

    # Save per-generation outputs (CSV and Parquet) alongside results.txt
    generations_csv = os.path.join(output_dir, "generations.csv")
    generations_parquet = os.path.join(output_dir, "generations.parquet")
    try:
        generations_df.to_csv(generations_csv, index=False)
    except Exception as e:
        print(f"Warning: Failed to save CSV: {e}")
    try:
        generations_df.to_parquet(generations_parquet, index=False)
    except Exception as e:
        print(f"Warning: Failed to save Parquet: {e}")

if __name__ == "__main__":
    main()
