#!/usr/bin/env python3
"""
Benchmark the overhead of TI (tokenization inconsistency) consistency checks.

Generates text with a given model and measures the wall-clock cost of the
decode→encode round-trip check at every token position.

Usage:
  python scripts/benchmark_tokenizer.py
  python scripts/benchmark_tokenizer.py --model meta-llama/Llama-3.2-3B --n-prompts 10 --max-tokens 200
  python scripts/benchmark_tokenizer.py --model meta-llama/Llama-2-7b-hf --n-prompts 500 --num-gpus 4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.paths as paths
from src.model_formatters import format_prompt_for_model
from src.utils import get_shuffled_essays, get_shuffled_texts, load_local_texts


def check_token_consistency(tokenizer, generated_ids: list[int]) -> bool:
    """Check if token list survives a decode→encode round-trip with same length."""
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    retokenized = tokenizer.encode(text, add_special_tokens=False)
    return len(retokenized) == len(generated_ids)


def gpu_worker(config: dict, batch_items: list[tuple[int, str]], results_queue: Queue):
    """
    GPU worker: loads model, generates text, measures TI check times.
    Each worker runs in a separate process on its assigned GPU.
    """
    device = config["device"]
    model_cache = str(Path(config["cache_dir"]) / "models")
    max_tokens = config["max_tokens"]
    model_name = config["model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=model_cache,
        torch_dtype="auto" if device != "cpu" else torch.float32,
    ).to(device)
    model.eval()

    for prompt_idx, prompt in batch_items:
        formatted = format_prompt_for_model(prompt, model_name, tokenizer)
        input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
        prompt_length = input_ids.shape[1]

        all_ids = input_ids[0].tolist()
        past_key_values = None
        gen_tok_time = 0.0
        gen_inconsistencies = 0
        check_times: list[float] = []

        gen_t0 = time.perf_counter()

        for _ in range(max_tokens):
            if past_key_values is None:
                model_input = torch.tensor([all_ids], device=device)
            else:
                model_input = torch.tensor([[all_ids[-1]]], device=device)

            with torch.no_grad():
                outputs = model(model_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

            # Greedy sampling, skip EOS
            sorted_ids = torch.argsort(logits, dim=-1, descending=True)[0]
            token_id = sorted_ids[0].item()
            if token_id == tokenizer.eos_token_id:
                token_id = sorted_ids[1].item()

            all_ids.append(token_id)
            generated_ids = all_ids[prompt_length:]

            # Time the consistency check
            t0 = time.perf_counter()
            consistent = check_token_consistency(tokenizer, generated_ids)
            elapsed = time.perf_counter() - t0

            gen_tok_time += elapsed
            check_times.append(elapsed)
            if not consistent:
                gen_inconsistencies += 1

        gen_elapsed = time.perf_counter() - gen_t0

        results_queue.put({
            "prompt_idx": prompt_idx,
            "tokenization_time_ms": gen_tok_time * 1000,
            "generation_time_ms": gen_elapsed * 1000,
            "inconsistencies": gen_inconsistencies,
            "check_times": check_times,
        })

    # Sentinel
    results_queue.put(None)


def main():
    parser = argparse.ArgumentParser(description="Benchmark TI consistency check overhead")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--dataset", type=str, nargs='+',
                        default=["ChristophSchuhmann/essays-with-instructions", "default", "train", "instructions"],
                        help="Text source: local file (.json/.txt) or HF dataset (4 args: '<name>' '<subset>' '<split>' '<column>')")
    parser.add_argument("--n-prompts", type=str, default="1", help="Number of prompts, or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-tokens", type=int, default=100, help="Tokens to generate per prompt")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (0 = all available, default: 0)")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--cache-dir", type=str, default=paths.CACHE_DIR, help="Cache directory")
    args = parser.parse_args()

    # Parse n_prompts
    if args.n_prompts.lower() == "all":
        n_prompts = None
    else:
        n_prompts = int(args.n_prompts)

    # Device / GPU count
    if args.no_cuda or not torch.cuda.is_available():
        num_gpus = 0
    else:
        available = torch.cuda.device_count()
        num_gpus = available if args.num_gpus == 0 else min(args.num_gpus, available)

    # Load prompts
    if len(args.dataset) == 1:
        path = args.dataset[0]
        ext = os.path.splitext(path)[1].lower()
        if ext in (".json", ".txt"):
            texts = load_local_texts(path)
            prompts = get_shuffled_texts(texts, args.seed, n_prompts)
        else:
            parser.error(
                f"'{path}' is not a supported file. Use a .json or .txt file, "
                f"or provide 4 arguments for a HuggingFace dataset."
            )
    elif len(args.dataset) == 4:
        prompts = get_shuffled_essays(
            dataset_name=args.dataset[0],
            dataset_subset=args.dataset[1],
            dataset_split=args.dataset[2],
            dataset_column=args.dataset[3],
            seed=args.seed,
            n_prompts=n_prompts,
        )
    else:
        parser.error(
            "--dataset must be a .json/.txt file (1 arg) or a HuggingFace dataset (4 args: name subset split column)."
        )

    device_str = f"{num_gpus} GPU(s)" if num_gpus > 0 else "CPU"
    print(f"Model: {args.model}")
    if len(args.dataset) == 1:
        print(f"Source: file: {args.dataset[0]}")
    else:
        print(f"Source: dataset: {args.dataset[0]}")
    print(f"Prompts: {len(prompts)}, Max tokens: {args.max_tokens}, Device: {device_str}")
    print()

    # Prepare batch items: (prompt_idx, prompt)
    batch_items = list(enumerate(prompts))

    worker_config = {
        "model": args.model,
        "cache_dir": args.cache_dir,
        "max_tokens": args.max_tokens,
        "device": "cpu",
    }

    results_queue: Queue = Queue()
    processes: list[Process] = []

    if num_gpus <= 1:
        # Single GPU or CPU
        worker_config["device"] = "cuda:0" if num_gpus == 1 else "cpu"
        p = Process(target=gpu_worker, args=(worker_config, batch_items, results_queue), daemon=True)
        p.start()
        processes.append(p)
    else:
        # Multi-GPU: round-robin distribution
        for gpu_id in range(num_gpus):
            worker_items = [item for i, item in enumerate(batch_items) if i % num_gpus == gpu_id]
            if not worker_items:
                continue
            config = {**worker_config, "device": f"cuda:{gpu_id}"}
            p = Process(target=gpu_worker, args=(config, worker_items, results_queue), daemon=True)
            p.start()
            processes.append(p)
        print(f"Started {len(processes)} GPU workers")

    # Collect results
    csv_rows: list[dict] = []
    all_check_times: list[float] = []
    total_inconsistencies = 0
    total_tokens = 0
    sentinels_received = 0

    progress = tqdm(total=len(prompts), desc="Generating")
    while sentinels_received < len(processes):
        item = results_queue.get()
        if item is None:
            sentinels_received += 1
            continue

        csv_rows.append({
            "prompt_idx": item["prompt_idx"],
            "tokenization_time_ms": item["tokenization_time_ms"],
            "generation_time_ms": item["generation_time_ms"],
            "inconsistencies": item["inconsistencies"],
        })
        all_check_times.extend(item["check_times"])
        total_inconsistencies += item["inconsistencies"]
        total_tokens += len(item["check_times"])
        progress.update(1)

    progress.close()
    for p in processes:
        p.join()

    # Sort by prompt_idx (results may arrive out of order with multi-GPU)
    csv_rows.sort(key=lambda r: r["prompt_idx"])

    # Report
    tok_times = [r["tokenization_time_ms"] for r in csv_rows]
    avg_per_gen = sum(tok_times) / len(tok_times)
    avg_per_token = sum(all_check_times) / len(all_check_times) * 1000
    total_time = sum(tok_times)

    lines = [
        "=" * 50,
        f"Model:               {args.model}",
        f"Source:              {'file: ' + args.dataset[0] if len(args.dataset) == 1 else 'dataset: ' + args.dataset[0]}",
        f"Generations:         {len(prompts)}",
        f"Tokens/generation:   {args.max_tokens}",
        f"Total tokens:        {total_tokens}",
        f"GPUs:                {max(num_gpus, 1)}",
        "",
        f"Avg TI check/gen:    {avg_per_gen:.3f} ms",
        f"Avg TI check/token:  {avg_per_token:.4f} ms",
        f"Total TI check time: {total_time:.1f} ms",
        f"Inconsistencies:     {total_inconsistencies} / {total_tokens} ({total_inconsistencies / total_tokens * 100:.2f}%)",
    ]

    report = "\n".join(lines)
    print(report)

    # Save to output/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = args.model.replace("/", "_")
    output_dir = os.path.join("output", f"ti_benchmark_{model_clean}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(report + "\n")

    csv_path = os.path.join(output_dir, "generations.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_idx", "tokenization_time_ms", "generation_time_ms", "inconsistencies"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
