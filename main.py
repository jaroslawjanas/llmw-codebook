import argparse
import hashlib
import random
import numpy as np
import torch
import pandas as pd
import json
import os
import time
from datetime import datetime
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import src.paths as paths
from src.utils import get_shuffled_essays
from src.llm_watermark import LLMWatermarkEncoder, LLMWatermarkDecoder
from src.codebooks import CodebookBase, CodebookHamming, CodebookLevenshtein, CodebookLevenshteinTriton, RNGCodewordGenerator, BCHCodewordGenerator, LinearCodeGenerator
from tqdm import tqdm


def gpu_generation_worker(watermarker_config, batch_items, results_queue, effective_max_tokens):
    """
    Producer: generates text for each prompt and puts results in queue.
    Runs in a separate process to allow GPU work to overlap with CPU search.
    
    Args:
        watermarker_config: Dict with LLMWatermarkEncoder configuration
        batch_items: List of (prompt_idx, prompt, codeword, watermark_identity) tuples
        results_queue: multiprocessing.Queue to put results into
        effective_max_tokens: Max tokens to generate per prompt
    """
    # Create watermarker in this process (models can't be pickled across processes)
    watermarker = LLMWatermarkEncoder(**watermarker_config)
    
    for prompt_idx, prompt, codeword, watermark_identity in batch_items:
        try:
            result = watermarker.generate_text(
                prompt=prompt,
                bit_code=codeword,
                max_new_tokens=effective_max_tokens
            )
            full_text, generated_text, generated_ids, formatted_prompt, generation_statistics, watermark_blocks_info = result
            
            # Convert tensor to list for pickling across processes
            generated_ids_list = list(generated_ids) if isinstance(generated_ids, list) else generated_ids.tolist()
            
            results_queue.put({
                'prompt_idx': prompt_idx,
                'prompt': prompt,
                'codeword': codeword,
                'watermark_identity': watermark_identity,
                'full_text': full_text,
                'generated_text': generated_text,
                'generated_ids': generated_ids_list,
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
    parser.add_argument("--codeword-generation", type=str, choices=["rng", "bch", "linear"], default="rng", help="Codeword generation strategy for base codebook: 'rng' (random baseline), 'bch' (BCH-based), or 'linear' (random linear code)")
    parser.add_argument("--distance", type=str, choices=["hamming", "levenshtein", "levenshtein-triton"], default="hamming", help="Distance metric for codebook search")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (n0=2^m-1). If None, auto-select optimal")
    parser.add_argument("--prompt", type=str, help="Custom prompt (uses random essay from dataset if not provided)")
    parser.add_argument("--dataset", type=str, nargs=4, default=["ChristophSchuhmann/essays-with-instructions", "default", "train", "instructions"], 
                        help="Hugging Face dataset to use for prompts. Format: '<org>/<dataset name>' '<subset>' '<split>' '<column>'. Default: 'ChristophSchuhmann/essays-with-instructions' 'default' 'train' 'instructions'")
    parser.add_argument("--cache-dir", type=str, default=paths.CACHE_DIR, help="Cache directory for models and datasets")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--context-window", type=int, default=1500, help="Maximum number of tokens to use as context for generation (default: 1500)")
    parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy sampling, higher = more random)")
    parser.add_argument("--hash-window", type=int, default=1, help="Number of previous tokens to hash together (default: 1)") 
    parser.add_argument("--n-prompts", type=str, default="1", help="Number of prompts to process, or 'all' for entire dataset (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output and progress information")
    parser.add_argument("--stats", action="store_true", help="Show statistics summary in console (statistics are always saved to file)")
    parser.add_argument("--force-tokenization", action="store_true", help="Force detokenization and retokenization of text for decoding instead of using token IDs directly")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for codebook generation (default: 4)")
    parser.add_argument("--chunk-rows", type=int, default=None, help="Optional override for generation chunk size in rows. If not set, auto policy is used.")
    # If set, run the entire pipeline in the main process (no separate GPU worker / no IPC).
    # Useful for debugging and environments where multiprocessing is problematic or when you want deterministic,
    # step-by-step execution without producer/consumer overlap.
    parser.add_argument("--synch", action="store_true", help="Run synchronously in a single process (disable GPU worker and queue)")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs for parallel generation (0 = all available, default: 0)")
    parser.add_argument("--dedicated-search-gpu", action="store_true",
        help="Reserve last GPU for Triton search (requires --distance levenshtein-triton, >= 2 GPUs)")
    parser.add_argument("--delayed-search", action="store_true",
        help="All GPUs generate first, then all search in parallel (requires --distance levenshtein-triton)")

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

    # Determine device and number of GPUs
    if args.no_cuda:
        device = "cpu"
        num_gpus = 0
        print(f"Device: {device} (CUDA disabled by --no-cuda flag)\n")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            available_gpus = torch.cuda.device_count()
            if args.num_gpus == 0:
                num_gpus = available_gpus
            else:
                if args.num_gpus > available_gpus:
                    print(f"Warning: --num-gpus {args.num_gpus} exceeds available GPUs ({available_gpus}), using {available_gpus}")
                num_gpus = min(args.num_gpus, available_gpus)
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
            print(f"Device: {device} ({num_gpus} GPU{'s' if num_gpus > 1 else ''}: {', '.join(gpu_names)})\n")
        else:
            num_gpus = 0
            print(f"Device: {device} (CUDA not available, falling back to CPU)\n")

    # Validate --dedicated-search-gpu and --delayed-search flags
    if args.dedicated_search_gpu and args.delayed_search:
        parser.error("--dedicated-search-gpu and --delayed-search are mutually exclusive")
    if args.dedicated_search_gpu:
        if args.distance != "levenshtein-triton":
            parser.error("--dedicated-search-gpu requires --distance levenshtein-triton")
        if num_gpus < 2:
            parser.error(f"--dedicated-search-gpu requires at least 2 GPUs (found {num_gpus})")
        if args.synch:
            parser.error("--dedicated-search-gpu cannot be used with --synch")
    if args.delayed_search:
        if args.distance != "levenshtein-triton":
            parser.error("--delayed-search requires --distance levenshtein-triton")
        if args.synch:
            parser.error("--delayed-search cannot be used with --synch")

    # Compute generation GPU count and search device
    if args.dedicated_search_gpu:
        generation_gpus = num_gpus - 1
        search_device = f"cuda:{num_gpus - 1}"
        print(f"Dedicated search GPU: {search_device} (generation on {generation_gpus} GPU{'s' if generation_gpus > 1 else ''})")
    else:
        generation_gpus = num_gpus
        search_device = None

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

    # Parse --n-prompts: "all" or integer
    if args.n_prompts.lower() == "all":
        n_prompts = None
    else:
        try:
            n_prompts = int(args.n_prompts)
        except ValueError:
            parser.error(f"--n-prompts must be a positive integer or 'all', got '{args.n_prompts}'")
        if n_prompts < 1:
            parser.error(f"--n-prompts must be a positive integer or 'all', got '{args.n_prompts}'")

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
            n_prompts=n_prompts
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
    
    output_subdir = f"{model_name_clean}_{dataset_name_clean}_n{args.n}_u{args.u}_c-{args.codeword_generation}_{args.distance}_{timestamp}"
    output_dir = os.path.join("output", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment configuration as JSON for downstream tooling
    config = {
        "model": {
            "name": args.model,
            "context_window": args.context_window,
        },
        "dataset": {
            "name": dataset_name,
            "subset": dataset_subset,
            "split": dataset_split,
            "column": dataset_column,
        },
        "watermark": {
            "n": args.n,
            "u": args.u,
            "secret_key": args.secret_key,
            "green_fraction": args.green_fraction,
            "bias": args.bias,
            "hash_window": args.hash_window,
            "codeword_generation": args.codeword_generation,
            "distance": args.distance,
            "bch_m": args.bch_m,
        },
        "generation": {
            "max_tokens": args.max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "temperature": args.temperature,
            "n_prompts": total_prompts,
            "seed": args.seed,
        },
        "experiment": {
            "timestamp": timestamp,
            "device": device,
            "num_gpus": num_gpus,
            "dedicated_search_gpu": args.dedicated_search_gpu,
            "delayed_search": args.delayed_search,
        },
    }
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

    # Watermarker config (passed to GPU worker process)
    watermarker_config = {
        'model_name': args.model,
        'n': args.n,
        'u': args.u,
        'secret_key': args.secret_key,
        'green_list_fraction': args.green_fraction,
        'bias': args.bias,
        'seed': args.seed,
        'device': device,
        'context_window': args.context_window,
        'temperature': args.temperature,
        'hash_window': args.hash_window,
        'verbose': args.verbose and args.synch
    }

    if args.codeword_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs = {}
    elif args.codeword_generation == "linear":
        generator_cls = LinearCodeGenerator
        generator_kwargs = {}
    else:
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"m": args.bch_m}

    base_codebook = CodebookBase(
        n=args.n,
        u=args.u,
        secret_key=args.secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
        workers=args.workers,
        chunk_rows=args.chunk_rows
    )
    codebook_hash = hashlib.sha256(base_codebook.packed_codes.tobytes()).hexdigest()
    config["watermark"]["codebook_hash"] = codebook_hash
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if args.distance == "hamming":
        codebook = CodebookHamming(base_codebook)
    elif args.distance == "levenshtein-triton":
        if args.delayed_search:
            codebook = None  # Created after generation to save GPU memory
        elif args.dedicated_search_gpu:
            codebook = CodebookLevenshteinTriton(base_codebook, device=search_device)
        else:
            codebook = CodebookLevenshteinTriton(base_codebook)
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
        verbose=args.verbose and args.synch  # Disable verbose for decoder in consumer loop
    )

    # Statistics accumulators
    successful_watermark_recoveries = 0
    successful_search_distances     = []    # For matches (Hamming or Levenshtein)
    full_generation_prompts         = 0     # Count prompts with full max-tokens generation
    properly_encoded_tokens         = 0     # Across full-length generations only
    total_tokens_generated          = 0     # Across full-length generations only
    codebook_search_times           = []
    queue_wait_times                = []    # Time spent waiting for GPU queue

    #----------------------------------------#
    #     Prepare Batch Items (Producer)     #
    #----------------------------------------#
    
    # Pre-collect all (prompt_idx, prompt, codeword, watermark_identity) tuples
    row_source = codebook if codebook is not None else base_codebook
    batch_items = []
    for prompt_idx, prompt in enumerate(prompts):
        codeword, watermark_identity = row_source.get_random_row(args.seed + prompt_idx)
        batch_items.append((prompt_idx, prompt, codeword, watermark_identity))
    
    if args.verbose:
        print(f"\nPrepared {len(batch_items)} prompts for generation queue")

    #----------------------------------------#
    #   Start Generation Workers             #
    #----------------------------------------#

    results_queue = Queue()
    gpu_processes = []
    num_active_workers = 0

    if args.synch:
        if args.verbose:
            print("Running generation worker synchronously (--synch)")
        gpu_generation_worker(
            watermarker_config=watermarker_config,
            batch_items=batch_items,
            results_queue=results_queue,
            effective_max_tokens=effective_max_tokens
        )
        num_active_workers = 1  # synch mode sends one sentinel
    elif num_gpus <= 1:
        # Single GPU worker (original behavior)
        p = Process(
            target=gpu_generation_worker,
            args=(watermarker_config, batch_items, results_queue, effective_max_tokens),
            daemon=True,
        )
        p.start()
        gpu_processes.append(p)
        num_active_workers = 1
        if args.verbose:
            print("GPU generation worker process started")
    else:
        # Multi-GPU: round-robin distribute prompts across generation workers
        for gpu_id in range(generation_gpus):
            worker_items = [item for i, item in enumerate(batch_items) if i % generation_gpus == gpu_id]
            if not worker_items:
                continue
            worker_config = {**watermarker_config, 'device': f'cuda:{gpu_id}'}
            p = Process(
                target=gpu_generation_worker,
                args=(worker_config, worker_items, results_queue, effective_max_tokens),
                daemon=True,
            )
            p.start()
            gpu_processes.append(p)
            num_active_workers += 1
        if args.verbose:
            print(f"Started {num_active_workers} GPU generation workers across {generation_gpus} GPUs")

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
    sentinels_received = 0
    pending_searches = []  # For --delayed-search: collect decoded queries for batch search

    while True:
        wait_start = time.time()
        item = results_queue.get()
        wait_time = time.time() - wait_start
        queue_wait_times.append(wait_time)

        # Sentinel: each worker sends None when done
        if item is None:
            sentinels_received += 1
            if sentinels_received >= num_active_workers:
                break
            continue
        
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
            print(f"\tProcessing Prompt {prompt_idx + 1}/{total_prompts}")
            print(f"{'='*60}")
            print(f"Prompt:\n{prompt[:100] + '...' if len(prompt) > 100 else prompt}\n\n")
            print(f"Generated text:\n{generated_text[:100]+ '...' if len(generated_text) > 100 else generated_text}")
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
        
        if args.delayed_search:
            # Collect data for batch search after generation completes
            pending_searches.append({
                'prompt_idx': prompt_idx,
                'codeword': codeword,
                'watermark_identity': watermark_identity,
                'decoded_codeword': decoded_codeword,
            })
        else:
            # Real-time search (default or --dedicated-search-gpu)
            search_start = time.time()
            best_codeword_match, retrieved_watermark, search_distance = codebook.find_closest_match(decoded_codeword)
            search_time = time.time() - search_start
            codebook_search_times.append(search_time)

            if args.verbose:
                print(f"\tQueue wait: {wait_time * 1000:.2f} ms")
                print(f"\tSearch time: {search_time * 1000:.2f} ms\n")

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

    for p in gpu_processes:
        p.join()

    #----------------------------------------#
    #    Delayed Batch Search Phase          #
    #----------------------------------------#

    if args.delayed_search and pending_searches:
        print(f"\nGeneration complete. Starting delayed batch search across {num_gpus} GPUs...")

        # Workers are joined; GPU memory is freed. Create Triton codebook per GPU.
        search_codebooks = []
        for gpu_id in range(num_gpus):
            cb = CodebookLevenshteinTriton(base_codebook, device=f'cuda:{gpu_id}')
            search_codebooks.append(cb)

        # Distribute queries round-robin across GPUs
        per_gpu_queries = [[] for _ in range(num_gpus)]
        per_gpu_indices = [[] for _ in range(num_gpus)]
        for i, ps in enumerate(pending_searches):
            gpu_id = i % num_gpus
            per_gpu_queries[gpu_id].append(np.asarray(ps['decoded_codeword'], dtype=np.uint8))
            per_gpu_indices[gpu_id].append(i)

        def search_on_gpu(gpu_id):
            """Run batch search on a single GPU. Returns list of (orig_idx, watermark, distance)."""
            queries = per_gpu_queries[gpu_id]
            indices = per_gpu_indices[gpu_id]
            if not queries:
                return []
            cb = search_codebooks[gpu_id]
            _, watermark_ids, distances = cb.find_closest_match_batch(queries)
            return [(indices[j], int(watermark_ids[j]), int(distances[j])) for j in range(len(indices))]

        # Run search in parallel (Triton releases GIL during kernel execution)
        search_start = time.time()
        all_search_results = {}

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(search_on_gpu, gpu_id): gpu_id for gpu_id in range(num_gpus)}
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    for orig_idx, wm, dist in future.result():
                        all_search_results[orig_idx] = (wm, dist)
                except Exception as e:
                    print(f"Error: Triton search failed on cuda:{gpu_id}: {e}")
                    raise

        total_search_time = time.time() - search_start
        avg_search_per_query = total_search_time / len(pending_searches)
        print(f"Delayed batch search completed in {total_search_time:.2f}s "
              f"({len(pending_searches)} queries across {num_gpus} GPUs, "
              f"{avg_search_per_query * 1000:.1f} ms/query)")

        # Populate statistics and DataFrame from search results
        for i, ps in enumerate(pending_searches):
            if i not in all_search_results:
                continue
            retrieved_watermark, search_distance = all_search_results[i]
            codebook_search_times.append(avg_search_per_query)

            codeword_str = ''.join(map(str, ps['codeword']))
            decoded_codeword_str = ''.join(map(str, ps['decoded_codeword']))
            watermark_identity = ps['watermark_identity']

            successful_watermark_flag = bool(retrieved_watermark == watermark_identity)
            if successful_watermark_flag:
                successful_watermark_recoveries += 1
                successful_search_distances.append(search_distance)

            generations_df.loc[len(generations_df)] = {
                "codeword": codeword_str,
                "watermark": int(watermark_identity),
                "decoded_codeword": decoded_codeword_str,
                "retrieved_watermark": retrieved_watermark,
                "successful_watermark": successful_watermark_flag,
                "distance": search_distance,
            }

    #----------------------------------------#
    #           Results Summary              #
    #----------------------------------------#

    # Calculate timing averages once
    avg_codebook_search_time = 1000 * float(np.mean(codebook_search_times)) if codebook_search_times else 0.0
    avg_queue_wait_time = 1000 * float(np.mean(queue_wait_times)) if queue_wait_times else 0.0
    total_queue_wait_time_seconds = float(np.sum(queue_wait_times)) if queue_wait_times else 0.0

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
        f.write(f"Number of prompts: {total_prompts}\n")
        f.write(f"Max tokens (requested): {args.max_tokens if args.max_tokens is not None else 'None (default -> u)'}\n")
        f.write(f"Max tokens (effective): {effective_max_tokens}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Secret key: {args.secret_key}\n")
        f.write(f"Green fraction: {args.green_fraction}\n")
        f.write(f"Bias: {args.bias}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Device: {device}\n")
        if args.dedicated_search_gpu:
            f.write(f"Search mode: dedicated-search-gpu (search on {search_device})\n")
        elif args.delayed_search:
            f.write(f"Search mode: delayed-search ({num_gpus} GPUs)\n")
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

        # Timing statistics
        f.write(f"Average search time ({args.distance}): {avg_codebook_search_time:.2f} ms\n")
        f.write(f"Average queue wait time: {avg_queue_wait_time:.2f} ms\n")
        f.write(f"Total queue wait time: {total_queue_wait_time_seconds:.2f} s\n")

    print(f"\bResults saved to: {results_file}")
    print(f"Made {successful_watermark_recoveries} successful watermark recoveries out of {full_generation_prompts}")
    
    # Calculate and display average distances
    if successful_search_distances:
        avg_successful = np.mean(successful_search_distances)
        print(f"Average {args.distance} distance for successful recoveries: {avg_successful:.2f}")
    else:
        print(f"No successful recoveries to calculate average {args.distance} distance")
    
    # Display average search time and queue wait time
    print(f"Average {args.distance} search time: {avg_codebook_search_time:.2f} ms")
    print(f"Average queue wait time: {avg_queue_wait_time:.2f} ms")

    # Save per-generation outputs (CSV) alongside results.txt
    generations_csv = os.path.join(output_dir, "generations.csv")
    try:
        generations_df.to_csv(generations_csv, index=False)
    except Exception as e:
        print(f"Warning: Failed to save CSV: {e}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
