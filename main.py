"""
BCH-based LLM text watermarking.

Uses BCH error-correcting codes to embed and recover watermarks from LLM-generated text.
Unlike codebook-based approaches, BCH directly recovers the watermark via syndrome decoding.
"""

import argparse
import random
import json
import numpy as np
import torch
import pandas as pd
import os
import time
from datetime import datetime
from multiprocessing import Process, Queue
import src.paths as paths
from src.utils import get_shuffled_essays
from src.llm_watermark import LLMWatermarkEncoder, LLMWatermarkDecoder
from src.bch_codec import BCHCodec
from src.bch_params import format_bch_summary
from tqdm import tqdm


def gpu_generation_worker(watermarker_config, batch_items, results_queue, effective_max_tokens):
    """
    Producer: generates text for each prompt and puts results in queue.
    Runs in a separate process to allow GPU work to overlap with CPU decode.

    Args:
        watermarker_config: Dict with LLMWatermarkEncoder configuration
        batch_items: List of (prompt_idx, prompt, codeword, watermark_int) tuples
        results_queue: multiprocessing.Queue to put results into
        effective_max_tokens: Max tokens to generate per prompt
    """
    # Create watermarker in this process (models can't be pickled across processes)
    watermarker = LLMWatermarkEncoder(**watermarker_config)

    for prompt_idx, prompt, codeword, watermark_int in batch_items:
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
                'watermark_int': watermark_int,
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
                'watermark_int': watermark_int,
                'error': str(e),
                'success': False,
            })

    # Signal completion
    results_queue.put(None)


def main():
    parser = argparse.ArgumentParser(description="BCH-based LLM Text Watermarking")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens to generate (default: match --u)")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green/red tokens")
    parser.add_argument("--n", type=int, default=16, help="Watermark message length in bits (supports 2^n unique watermarks)")
    parser.add_argument("--u", type=int, default=100, help="Codeword length in bits (tokens per watermark)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Secret key used for watermarking (string)")
    parser.add_argument("--bch-t", type=int, default=None, help="BCH error-correction capability t (if None, auto-select maximum)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (n0=2^m-1). If None, auto-select optimal")
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
    # If set, run the entire pipeline in the main process (no separate GPU worker / no IPC).
    # Useful for debugging and environments where multiprocessing is problematic or when you want deterministic,
    # step-by-step execution without producer/consumer overlap.
    parser.add_argument("--synch", action="store_true", help="Run synchronously in a single process (disable GPU worker and queue)")

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
        print(f"Generating {effective_max_tokens} tokens per prompt with BCH watermarking...")
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

    # Initialize per-generation outputs DataFrame
    generations_columns = ["watermark", "codeword", "decoded_codeword", "recovered_watermark", "bit_errors", "decode_success"]
    generations_df = pd.DataFrame(columns=generations_columns)

    #----------------------------------------#
    #           BCH Codec Setup              #
    #----------------------------------------#

    bch_codec = BCHCodec(
        n=args.n,
        u=args.u,
        bch_m=args.bch_m,
        bch_t=args.bch_t,
        verbose=args.verbose
    )

    print(f"\n{format_bch_summary(bch_codec.params, args.n)}")
    print(f"Watermark space: 2^{args.n} = {2**args.n:,} unique watermarks\n")

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
    successful_decodes = 0              # BCH decode succeeded AND matched original
    decode_failures = 0                 # BCH decode failed (too many errors)
    bit_errors_list = []                # For successful decodes
    full_generation_prompts = 0         # Count prompts with full max-tokens generation
    properly_encoded_tokens = 0         # Across full-length generations only
    total_tokens_generated = 0          # Across full-length generations only
    bch_decode_times = []
    queue_wait_times = []               # Time spent waiting for GPU queue

    #----------------------------------------#
    #     Prepare Batch Items (Producer)     #
    #----------------------------------------#

    # Pre-collect all (prompt_idx, prompt, codeword, watermark_int) tuples
    rng = np.random.default_rng(args.seed)
    batch_items = []
    for prompt_idx, prompt in enumerate(prompts):
        # Random watermark in range [0, 2^n - 1]
        watermark_int = int(rng.integers(0, 2**args.n))
        codeword = bch_codec.encode_int(watermark_int)
        batch_items.append((prompt_idx, prompt, codeword, watermark_int))

    if args.verbose:
        print(f"\nPrepared {len(batch_items)} prompts for generation queue")

    #----------------------------------------#
    #   Start Generation Worker              #
    #----------------------------------------#

    results_queue = Queue()
    # Unified interface: both modes push results into the same Queue; the consumer loop below is identical.
    gpu_process = None
    if args.synch:
        if args.verbose:
            print("Running generation worker synchronously (--synch)")
        gpu_generation_worker(
            watermarker_config=watermarker_config,
            batch_items=batch_items,
            results_queue=results_queue,
            effective_max_tokens=effective_max_tokens
        )
    else:
        gpu_process = Process(
            target=gpu_generation_worker,
            args=(watermarker_config, batch_items, results_queue, effective_max_tokens),
            daemon=True
        )
        gpu_process.start()
        if args.verbose:
            print("GPU generation worker process started")

    #----------------------------------------#
    #    Consumer Loop (Main Thread)         #
    #----------------------------------------#

    # Progress bar for processing (decode + stats)
    processing_progress = tqdm(
        total=total_prompts,
        desc="Processing",
        unit="prompt",
        disable=args.verbose
    )

    processed_count = 0

    while True:
        # Block until a result is available.
        wait_start = time.time()
        item = results_queue.get()
        wait_time = time.time() - wait_start
        queue_wait_times.append(wait_time)

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
        watermark_int = item['watermark_int']
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

        # Decode watermark bits from generated text
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

        # BCH decode to recover watermark
        decode_start = time.time()
        recovered_watermark, bit_errors = bch_codec.decode_int(decoded_codeword)
        decode_time = time.time() - decode_start
        bch_decode_times.append(decode_time)

        # Determine decode success
        # BCH decode fails if bit_errors > t (indicated by bit_errors == n0)
        decode_failed = (bit_errors >= bch_codec.native_codeword_length)
        decode_success = not decode_failed and (recovered_watermark == watermark_int)

        if decode_failed:
            decode_failures += 1
        elif recovered_watermark == watermark_int:
            successful_decodes += 1
            bit_errors_list.append(bit_errors)

        if args.verbose:
            print(f"\tQueue wait: {wait_time * 1000:.2f} ms")
            print(f"\tBCH decode time: {decode_time * 1000:.2f} ms")
            print(f"\tOriginal watermark: {watermark_int}")
            print(f"\tRecovered watermark: {recovered_watermark}")
            print(f"\tBit errors: {bit_errors}")
            print(f"\tDecode success: {decode_success}\n")

        # Convert to string for easier output
        codeword_str = ''.join(map(str, codeword))
        decoded_codeword_str = ''.join(map(str, decoded_codeword))

        # Record in DataFrame
        generations_df.loc[len(generations_df)] = {
            "watermark": int(watermark_int),
            "codeword": codeword_str,
            "decoded_codeword": decoded_codeword_str,
            "recovered_watermark": int(recovered_watermark),
            "bit_errors": int(bit_errors),
            "decode_success": decode_success,
        }

        processing_progress.update(1)
        processed_count += 1

    processing_progress.close()

    # Wait for worker process to finish (only applicable in async mode).
    if gpu_process is not None:
        gpu_process.join()

    #----------------------------------------#
    #           Results Summary              #
    #----------------------------------------#

    # Calculate timing averages once
    avg_bch_decode_time = 1000 * float(np.mean(bch_decode_times)) if bch_decode_times else 0.0
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
        f.write(f"n (watermark bits): {args.n}\n")
        f.write(f"u (codeword length): {args.u}\n")
        f.write(f"BCH parameters: {format_bch_summary(bch_codec.params, args.n)}\n")
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
        f.write(f"Successful watermark recoveries: {successful_decodes} out of {full_generation_prompts}\n")
        f.write(f"BCH decode failures (too many errors): {decode_failures}\n")

        if full_generation_prompts > 0:
            success_rate = successful_decodes / full_generation_prompts * 100
            f.write(f"Success rate: {success_rate:.2f}%\n")

        # Properly encoded tokens summary across full-length generations
        if total_tokens_generated > 0:
            pct_proper = properly_encoded_tokens / total_tokens_generated * 100
        else:
            pct_proper = 0.0
        f.write(f"Properly encoded tokens: {properly_encoded_tokens} / {total_tokens_generated} ({pct_proper:.2f}%)\n")

        f.write("\n")

        # Write bit error statistics
        if bit_errors_list:
            avg_bit_errors = np.mean(bit_errors_list)
            min_bit_errors = np.min(bit_errors_list)
            max_bit_errors = np.max(bit_errors_list)
            f.write(f"Average bit errors for successful recoveries: {avg_bit_errors:.2f}\n")
            f.write(f"Min bit errors for successful recoveries: {int(min_bit_errors)}\n")
            f.write(f"Max bit errors for successful recoveries: {int(max_bit_errors)}\n")
        else:
            f.write(f"No successful recoveries to calculate bit error statistics\n")

        # Timing statistics
        f.write(f"Average BCH decode time: {avg_bch_decode_time:.2f} ms\n")
        f.write(f"Average queue wait time: {avg_queue_wait_time:.2f} ms\n")
        f.write(f"Total queue wait time: {total_queue_wait_time_seconds:.2f} s\n")

    print(f"\nResults saved to: {results_file}")
    print(f"Made {successful_decodes} successful watermark recoveries out of {full_generation_prompts}")
    if decode_failures > 0:
        print(f"BCH decode failures: {decode_failures}")

    # Calculate and display average bit errors
    if bit_errors_list:
        avg_bit_errors = np.mean(bit_errors_list)
        print(f"Average bit errors for successful recoveries: {avg_bit_errors:.2f}")
    else:
        print(f"No successful recoveries to calculate average bit errors")

    # Display average decode time and queue wait time
    print(f"Average BCH decode time: {avg_bch_decode_time:.2f} ms")
    print(f"Average queue wait time: {avg_queue_wait_time:.2f} ms")

    # Save per-generation outputs (CSV) alongside results.txt
    generations_csv = os.path.join(output_dir, "generations.csv")
    try:
        generations_df.to_csv(generations_csv, index=False)
    except Exception as e:
        print(f"Warning: Failed to save CSV: {e}")

    # Save config.json with experiment parameters
    config = {
        "experiment": {
            "timestamp": timestamp,
            "seed": args.seed,
        },
        "model": {
            "name": args.model,
            "device": device,
        },
        "dataset": {
            "name": dataset_name,
            "config": dataset_subset,
            "split": dataset_split,
            "column": dataset_column,
            "n_prompts": args.n_prompts,
        },
        "watermark": {
            "n": args.n,
            "u": args.u,
            "secret_key": args.secret_key,
            "green_fraction": args.green_fraction,
            "bias": args.bias,
        },
        "bch": {
            "m": bch_codec.params.m,
            "n0": bch_codec.params.n0,
            "k": bch_codec.params.k,
            "t": bch_codec.params.t,
            "padding": bch_codec.params.padding,
            "puncturing": bch_codec.params.puncturing,
        },
        "generation": {
            "max_tokens_requested": args.max_tokens,
            "max_tokens_effective": effective_max_tokens,
            "context_window": args.context_window,
            "temperature": args.temperature,
            "hash_window": args.hash_window,
        },
    }

    config_file = os.path.join(output_dir, "config.json")
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save config.json: {e}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
