import argparse
import random
import numpy as np
import torch
import pandas as pd
import json
import os
import time
from datetime import datetime
from threading import Thread
from queue import Queue
import src.paths as paths
from src.utils import get_shuffled_essays
from src.llm_watermark import LLMWatermarkEncoder, LLMWatermarkDecoder
from src.codebooks import CodebookBase, CodebookHamming, CodebookLevenshtein, RNGCodewordGenerator, BCHCodewordGenerator
from tqdm import tqdm


def gpu_generation_worker(watermarker, batch_items, results_queue, effective_max_tokens):
    """
    Producer: generates text for each prompt and puts results in queue.
    Runs in a background thread to allow GPU work to overlap with CPU search.
    
    Args:
        watermarker: LLMWatermarkEncoder instance
        batch_items: List of (prompt_idx, prompt, codeword, watermark_identity) tuples
        results_queue: Queue to put results into
        effective_max_tokens: Max tokens to generate per prompt
    """
    for prompt_idx, prompt, codeword, watermark_identity in batch_items:
        try:
            result = watermarker.generate_text(
                prompt=prompt,
                bit_code=codeword,
                max_new_tokens=effective_max_tokens
            )
            full_text, generated_text, generated_ids, formatted_prompt, generation_statistics, watermark_blocks_info = result
            
            results_queue.put({
                'prompt_idx': prompt_idx,
                'prompt': prompt,
                'codeword': codeword,
                'watermark_identity': watermark_identity,
                'full_text': full_text,
                'generated_text': generated_text,
                'generated_ids': generated_ids,
                'generation_statistics': generation_statistics,
                'success': True,
            })
        except Exception as e:
            results_queue.put({
                'prompt_idx': prompt_idx,
                'prompt': prompt,
                'codeword': codeword,
                'watermark_identity': watermark_identity,
                'error': str(e),
                'success': False,
            })
    
    # Signal completion
    results_queue.put(None)


def main():
    parser = argparse.ArgumentParser(description="LLM Text Watermarking based on Lagrange Interpolation")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens to generate (default: match --u)")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green/red tokens")
    parser.add_argument("--n", type=int, default=16, help="Size of the code table")
    parser.add_argument("--u", type=int, default=100, help="Size of the codes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Secret key used for watermarking and codebook generation (string)")
    parser.add_argument("--codeword-generation", type=str, choices=["rng", "bch"], default="rng", help="Codeword generation strategy for base codebook: 'rng' (random baseline) or 'bch' (BCH-based)")
    parser.add_argument("--distance", type=str, choices=["hamming", "levenshtein"], default="hamming", help="Distance metric for codebook search")
    parser.add_argument("--bch-t", type=int, default=None, help="BCH designed error-correction parameter t (if None, auto-choose the largest t with k >= n)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (n0=2^m-1). If None, minimal m with n0 >= u is chosen")
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
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for codebook generation (default: 4)")
    parser.add_argument("--chunk-rows", type=int, default=None, help="Optional override for generation chunk size in rows. If not set, auto policy is used.")

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

    # Determine effective max tokens: default to u when not provided, and ensure multiple of u
    if args.max_tokens is None:
        effective_max_tokens = args.u
    else:
        effective_max_tokens = args.max_tokens
        if effective_max_tokens % args.u != 0:
            new_max = ((effective_max_tokens + args.u - 1) // args.u) * args.u
            print(f"Note: --max-tokens ({effective_max_tokens}) is not divisible by u={args.u}. Adjusting to {new_max}.")
            effective_max_tokens = new_max

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
        print(f"Generating {effective_max_tokens} tokens per prompt with watermarking...")
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
    generations_txt_file = os.path.join(output_dir, "generations.txt")
    try:
        with open(generations_txt_file, 'w', encoding='utf-8') as _f:
            _f.write("")
    except Exception as e:
        print(f"Warning: Failed to initialize generations.txt: {e}")
    
    # Initialize per-generation outputs DataFrame (only full-length generations will be recorded)
    generations_columns = ["codeword", "watermark", "decoded_codeword", "retrieved_watermark", "successful_watermark", "distance"]
    generations_df = pd.DataFrame(columns=generations_columns)

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
        verbose=False
    )

    if args.codeword_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs = {}
    else:
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"t": args.bch_t, "m": args.bch_m}

    base_codebook = CodebookBase(
        n=args.n,
        u=args.u,
        secret_key=args.secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
        workers=args.workers,
        chunk_rows=args.chunk_rows
    )
    if args.distance == "hamming":
        codebook = CodebookHamming(base_codebook)
    else:
        codebook = CodebookLevenshtein(base_codebook)

    # Create decoder once (shared across all prompts)
    decoder = LLMWatermarkDecoder(
        model_name=args.model,
        secret_key=args.secret_key,
        n=args.n,
        u=args.u,
        green_list_fraction=args.green_fraction,
        seed=args.seed,
        device=device,
        verbose=False  # Disable verbose for decoder in consumer loop
    )

    # Statistics accumulators
    successful_watermark_recoveries = 0
    successful_search_distances     = []    # For matches (Hamming or Levenshtein)
    full_generation_prompts         = 0     # Count prompts with full max-tokens generation
    properly_encoded_tokens         = 0     # Across full-length generations only
    total_tokens_generated          = 0     # Across full-length generations only
    codebook_search_times           = []

    #----------------------------------------#
    #     Prepare Batch Items (Producer)     #
    #----------------------------------------#
    
    # Pre-collect all (prompt_idx, prompt, codeword, watermark_identity) tuples
    batch_items = []
    for prompt_idx, prompt in enumerate(prompts):
        codeword, watermark_identity = codebook.get_random_row(args.seed + prompt_idx)
        batch_items.append((prompt_idx, prompt, codeword, watermark_identity))
    
    if args.verbose:
        print(f"\nPrepared {len(batch_items)} prompts for generation queue")

    #----------------------------------------#
    #   Start GPU Worker Thread (Producer)   #
    #----------------------------------------#
    
    results_queue = Queue()
    gpu_thread = Thread(
        target=gpu_generation_worker,
        args=(watermarker, batch_items, results_queue, effective_max_tokens),
        daemon=True
    )
    gpu_thread.start()
    
    if args.verbose:
        print("GPU generation worker started")

    #----------------------------------------#
    #    Consumer Loop (Main Thread)         #
    #----------------------------------------#
    
    # Progress bar for processing (search + stats)
    processing_progress = tqdm(
        total=total_prompts, 
        desc="Processing", 
        unit="prompt", 
        disable=args.verbose
    )
    
    processed_count = 0
    
    while True:
        # Block until a result is available from the GPU worker
        item = results_queue.get()
        
        # Sentinel: None signals all generations are complete
        if item is None:
            break
        
        # Handle generation errors
        if not item['success']:
            print(f"Generation failed for prompt {item['prompt_idx'] + 1}: {item.get('error', 'Unknown error')}")
            processing_progress.update(1)
            processed_count += 1
            continue
        
        prompt_idx = item['prompt_idx']
        prompt = item['prompt']
        codeword = item['codeword']
        watermark_identity = item['watermark_identity']
        generated_ids = item['generated_ids']
        generated_text = item['generated_text']
        generation_statistics = item['generation_statistics']
        
        # Verbose output for this prompt
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Processing Prompt {prompt_idx + 1}/{total_prompts}")
            print(f"{'='*60}")
            print(f"Prompt prompt:\n{prompt[:100] + '...' if len(prompt) > 100 else prompt}\n")
            print(f"\nGenerated text:\n{generated_text[:100]+ '...' if len(generated_text) > 100 else generated_text}")
            print(f"{'_'*14}\n")


        # Skip watermark recovery if generated text is shorter than max-tokens
        if len(generated_ids) < effective_max_tokens:
            if args.verbose:
                print(f"Skipping watermark recovery for prompt {prompt_idx + 1}: generated {len(generated_ids)} tokens < {effective_max_tokens} max tokens")
            processing_progress.update(1)
            processed_count += 1
            continue
        
        # Count prompts with full generation
        full_generation_prompts += 1
        
        # Decode watermark
        if args.force_tokenization:
            code_blocks, decoded_tokens_length = decoder.decode(generated_text)
        else:
            code_blocks, decoded_tokens_length = decoder.decode(generated_ids)
        
        # Single block setup (legacy support for multi-block)
        decoded_codeword = code_blocks[0]
        
        # Accumulate properly encoded token stats for full-length generations
        properly_encoded_tokens += generation_statistics.get("properly_encoded_tokens", 0)
        total_tokens_generated += generation_statistics.get("total_tokens_generated", 0)
        
        # Append to generations.txt for full-length generations
        try:
            with open(generations_txt_file, 'a', encoding='utf-8') as gf:
                gf.write(f"-- Prompt {prompt_idx + 1}: --\n")
                gf.write(f"{prompt}\n\n")
                gf.write("-- Generated: --\n")
                gf.write(f"{generated_text}\n")
                gf.write(f"{'_'*14}\n")
        except Exception as e:
            print(f"Warning: Failed to write generations.txt: {e}")
        
        # Find closest match using codebook search (CPU-intensive)
        search_start = time.time()
        best_codeword_match, retrieved_watermark, search_distance = codebook.find_closest_match(decoded_codeword)
        search_time = time.time() - search_start
        codebook_search_times.append(search_time)
        
        if args.verbose:
            print(f"Search time: {search_time * 1000:.2f} ms")
        
        # Convert to string for easier output
        codeword_str = ''.join(map(str, codeword))
        decoded_codeword_str = ''.join(map(str, decoded_codeword))
        
        # Check if watermarking was successful
        successful_watermark_flag = bool(retrieved_watermark == watermark_identity)
        if successful_watermark_flag:
            successful_watermark_recoveries += 1
            successful_search_distances.append(search_distance)
        
        # Record in DataFrame
        generations_df.loc[len(generations_df)] = {
            "codeword": codeword_str,
            "watermark": int(watermark_identity),
            "decoded_codeword": decoded_codeword_str,
            "retrieved_watermark": retrieved_watermark,
            "successful_watermark": successful_watermark_flag,
            "distance": search_distance,
        }
        
        processing_progress.update(1)
        processed_count += 1
    
    processing_progress.close()
    
    # Wait for GPU thread to finish (should already be done after sentinel)
    gpu_thread.join()

    #----------------------------------------#
    #           Results Summary              #
    #----------------------------------------#

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
        f.write(f"Max tokens (requested): {args.max_tokens if args.max_tokens is not None else 'None (default -> u)'}\n")
        f.write(f"Max tokens (effective): {effective_max_tokens}\n")
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
        f.write(f"Successful watermark recoveries: {successful_watermark_recoveries} out of {full_generation_prompts}\n")
        
        if full_generation_prompts > 0:
            success_rate = successful_watermark_recoveries / full_generation_prompts * 100
            f.write(f"Success rate: {success_rate:.2f}%\n")
        
        # Properly encoded tokens summary across full-length generations
        if total_tokens_generated > 0:
            pct_proper = properly_encoded_tokens / total_tokens_generated * 100
        else:
            pct_proper = 0.0
        f.write(f"Properly encoded tokens: {properly_encoded_tokens} / {total_tokens_generated} ({pct_proper:.2f}%)\n")
        
        f.write("\n")
        
        # Write distance statistics
        if successful_search_distances:
            avg_successful = np.mean(successful_search_distances)
            min_successful = np.min(successful_search_distances)
            max_successful = np.max(successful_search_distances)
            f.write(f"Average {args.distance} distance for successful recoveries: {avg_successful:.2f}\n")
            f.write(f"Min {args.distance} distance for successful recoveries: {int(min_successful)}\n")
            f.write(f"Max {args.distance} distance for successful recoveries: {int(max_successful)}\n")
        else:
            f.write(f"No successful recoveries to calculate {args.distance} distance statistics\n")

        # Search time statistics
        avg_codebook_search_time = 1000 * float(np.mean(codebook_search_times)) if codebook_search_times else 0.0
        f.write(f"Avg search ({args.distance}): {avg_codebook_search_time:.2f} ms\n")

        

    print(f"\bResults saved to: {results_file}")
    print(f"Made {successful_watermark_recoveries} successful watermark recoveries out of {full_generation_prompts}")
    
    # Calculate and display average distances
    if successful_search_distances:
        avg_successful = np.mean(successful_search_distances)
        print(f"Average {args.distance} distance for successful recoveries: {avg_successful:.2f}")
    else:
        print(f"No successful recoveries to calculate average {args.distance} distance")
    
    # Display average search time
    avg_codebook_search_time = 1000 * float(np.mean(codebook_search_times)) if codebook_search_times else 0.0
    print(f"Average {args.distance} search time: {avg_codebook_search_time:.2f} ms")

    # Save per-generation outputs (CSV) alongside results.txt
    generations_csv = os.path.join(output_dir, "generations.csv")
    try:
        generations_df.to_csv(generations_csv, index=False)
    except Exception as e:
        print(f"Warning: Failed to save CSV: {e}")

if __name__ == "__main__":
    main()
