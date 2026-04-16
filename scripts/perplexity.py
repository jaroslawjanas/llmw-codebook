"""
Perplexity Evaluation Script.

Computes token-level perplexity for up to three text conditions:
  1. Human-written text (from HuggingFace dataset or experiment directory)
  2. No-watermark generated text (from experiment directory or dataset)
  3. Watermarked generated text (from experiment directory or dataset)

Uses a sliding-window approach for texts exceeding the evaluator model's
context window, following the HuggingFace perplexity guide.

Supports multi-GPU parallelism: one model instance per GPU, texts
distributed round-robin across workers (same pattern as main.py).

Usage examples:
  python scripts/perplexity.py \\
      --human-text jaroslawjanas/open-generation-data default train "targets[0]" \\
      --no-watermark-text output/no_wm_experiment/ \\
      --watermarked-text output/wm_experiment/ \\
      --model openai/gpt-oss-20b --num-gpus 4 --verbose

  python scripts/perplexity.py \\
      --watermarked-text output/wm_experiment/ \\
      --model gpt2 --max-length 200 --min-length 100 --n-texts 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import src.paths as paths
from src.utils import load_hf_token
from scripts.utils import load_texts, parse_text_source, source_description


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation for human, no-watermark, and watermarked text",
    )

    # Text sources
    parser.add_argument(
        "--human-text", type=str, nargs="+", default=None,
        help=(
            "Human text source. Either: a local directory path (experiment output "
            "dir with generations.csv), OR 4 args: '<dataset> <subset> <split> <column>' "
            "for a HuggingFace dataset. For list-type columns, append [<index>] to the "
            "column name (e.g., 'targets[0]')."
        ),
    )
    parser.add_argument(
        "--no-watermark-text", type=str, nargs="+", default=None,
        help="No-watermark text source. Same format as --human-text.",
    )
    parser.add_argument(
        "--watermarked-text", type=str, nargs="+", default=None,
        help="Watermarked text source. Same format as --human-text.",
    )

    # Evaluator model
    parser.add_argument(
        "--model", type=str, required=True,
        help="Evaluator model (HuggingFace model ID, e.g., gpt2, gpt2-xl, openai/gpt-oss-20b)",
    )

    # Optional parameters
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Truncate texts to N tokens (evaluator tokenizer) for fair comparison",
    )
    parser.add_argument(
        "--min-length", type=int, default=None,
        help="Skip texts shorter than N tokens (evaluator tokenizer)",
    )
    parser.add_argument(
        "--n-texts", type=str, default="all",
        help="Number of texts to evaluate per condition, or 'all' (default: all)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0,
        help="Number of GPUs for parallel evaluation (0 = all available, default: 0)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=paths.CACHE_DIR,
        help="Model/dataset cache directory (default: ./cache)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Disable CUDA even if available",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--save-dir", type=str, default="output",
        help="Base directory to save outputs (default: output)",
    )

    args = parser.parse_args()

    if not any([args.human_text, args.no_watermark_text, args.watermarked_text]):
        parser.error("At least one of --human-text, --no-watermark-text, --watermarked-text is required")

    return args


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def _get_context_window(model) -> int:
    """Determine model's max context window from config."""
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        val = getattr(model.config, attr, None)
        if val is not None:
            return int(val)
    return 1024


def compute_perplexity(
    text: str,
    model,
    tokenizer,
    context_window: int,
    max_length: int | None = None,
    min_length: int | None = None,
) -> tuple[float, int, bool]:
    """Compute perplexity for a single text.

    Uses a sliding-window approach when the text exceeds the model's context
    window, with stride = context_window // 2.

    Returns:
        (perplexity, num_scored_tokens, skipped)
        If skipped=True, perplexity=nan and n_tokens=actual token count.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids  # (1, seq_len)

    seq_len = input_ids.size(1)

    # Min-length filter (before truncation)
    if min_length is not None and seq_len < min_length:
        return float("nan"), seq_len, True

    if max_length is not None and seq_len > max_length:
        input_ids = input_ids[:, :max_length]
        seq_len = max_length

    if seq_len <= 1:
        return float("inf"), 0, False

    stride = context_window // 2
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + context_window, seq_len)
        trg_len = end - prev_end

        window_ids = input_ids[:, begin:end].to(model.device)
        target_ids = window_ids.clone()

        # Mask overlap region so each token is scored exactly once
        if trg_len < target_ids.size(1):
            target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(window_ids, labels=target_ids)

        # Count tokens that actually contributed to the loss.
        # HF computes loss on shift_labels = labels[:, 1:], so count the
        # non-ignored positions in that shifted view.
        #
        # For the first (unmasked) window this equals window_size-1: the last
        # logit position predicts a token outside the window and is dropped by
        # the shift, so one token is lost.
        #
        # For subsequent windows where the overlap region is masked at the
        # START of target_ids, the shift drops that leading -100 and all
        # trg_len valid tokens remain — the correct count is trg_len, NOT
        # trg_len-1.  Counting from the shifted view handles both cases.
        num_loss_tokens = (target_ids[0, 1:] != -100).sum().item()

        if num_loss_tokens > 0:
            nll_sum += outputs.loss.item() * num_loss_tokens
            n_tokens += num_loss_tokens

        prev_end = end
        if end == seq_len:
            break

    if n_tokens == 0:
        return float("inf"), 0, False

    ppl = float(np.exp(nll_sum / n_tokens))
    return ppl, n_tokens, False


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------

def _perplexity_worker(worker_config: dict, text_items: list, results_queue: Queue):
    """GPU worker: loads evaluator model on assigned device, computes PPL for each text.

    Args:
        worker_config: dict with model_name, device, max_length, min_length,
                       models_cache_dir, token
        text_items: list of (global_idx, condition_name, local_idx, text) tuples
        results_queue: multiprocessing.Queue for sending results back
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = worker_config["device"]
    model_name = worker_config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=worker_config["models_cache_dir"],
        token=worker_config["token"],
    )

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=worker_config["models_cache_dir"],
            token=worker_config["token"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=worker_config["models_cache_dir"],
            torch_dtype="auto",
            token=worker_config["token"],
            device_map={"": device},
        )
    model.eval()

    context_window = _get_context_window(model)

    for global_idx, condition_name, local_idx, text in text_items:
        t_start = time.perf_counter()
        try:
            ppl, n_tokens, skipped = compute_perplexity(
                text, model, tokenizer, context_window,
                worker_config["max_length"],
                worker_config["min_length"],
            )
            compute_time = time.perf_counter() - t_start
            results_queue.put({
                "global_idx": global_idx,
                "condition": condition_name,
                "local_idx": local_idx,
                "perplexity": ppl,
                "n_tokens": n_tokens,
                "compute_time": compute_time,
                "skipped": skipped,
                "success": True,
            })
        except Exception as e:
            compute_time = time.perf_counter() - t_start
            results_queue.put({
                "global_idx": global_idx,
                "condition": condition_name,
                "local_idx": local_idx,
                "perplexity": float("nan"),
                "n_tokens": 0,
                "compute_time": compute_time,
                "skipped": False,
                "success": False,
                "error": str(e),
            })

    # Sentinel
    results_queue.put(None)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _output_results(
    condition_results: dict[str, list[dict]],
    source_descriptions: dict[str, str],
    args: argparse.Namespace,
) -> None:
    """Aggregate results, print summary table, save CSVs."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = args.model.replace("/", "_")
    run_dir = os.path.join(args.save_dir, f"perplexity_{model_clean}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Per-condition CSVs and stats
    summary_records = []

    for condition, records in condition_results.items():
        df = pd.DataFrame(records)

        # Save per-condition CSV
        csv_path = os.path.join(run_dir, f"{condition}.csv")
        df.to_csv(csv_path, index=False)

        # Compute stats (exclude skipped and failed)
        valid = df[
            ~df["skipped"]
            & df["perplexity"].notna()
            & np.isfinite(df["perplexity"])
        ]
        ppls = valid["perplexity"].values
        n_skipped = int(df["skipped"].sum())
        stats = {
            "condition": condition,
            "count": len(df),
            "skipped": n_skipped,
            "valid": len(valid),
            "mean": float(np.mean(ppls)) if len(ppls) > 0 else float("nan"),
            "median": float(np.median(ppls)) if len(ppls) > 0 else float("nan"),
            "std": float(np.std(ppls)) if len(ppls) > 0 else float("nan"),
            "min": float(np.min(ppls)) if len(ppls) > 0 else float("nan"),
            "max": float(np.max(ppls)) if len(ppls) > 0 else float("nan"),
        }
        summary_records.append(stats)

    # Summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # Build text report
    lines = [
        "=" * 70,
        "Perplexity Evaluation Results",
        "=" * 70,
        "",
        f"Evaluator model: {args.model}",
        f"Max length: {args.max_length or 'none (full text)'}",
        f"Min length: {args.min_length or 'none (no filter)'}",
        "",
    ]
    for cond, desc in source_descriptions.items():
        lines.append(f"  {cond}: {desc}")
    lines.append("")
    lines.append("-" * 70)
    lines.append(
        f"{'Condition':<16} {'Valid':>6} {'Skipped':>8} {'Mean PPL':>10} "
        f"{'Median PPL':>12} {'Std':>10} {'Min':>10} {'Max':>10}"
    )
    lines.append("-" * 70)

    for stats in summary_records:
        lines.append(
            f"{stats['condition']:<16} {stats['valid']:>6} {stats['skipped']:>8} "
            f"{stats['mean']:>10.2f} {stats['median']:>12.2f} {stats['std']:>10.2f} "
            f"{stats['min']:>10.2f} {stats['max']:>10.2f}"
        )

    lines.append("-" * 70)
    lines.append("")

    text_output = "\n".join(lines)
    print(text_output)

    results_path = os.path.join(run_dir, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(text_output)

    print(f"Results saved to: {run_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    paths.set_cache_dir(args.cache_dir)
    paths.ensure_directories()

    # Determine device and GPU count
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

    # Parse --n-texts
    n_texts = None
    if args.n_texts.lower() != "all":
        n_texts = int(args.n_texts)

    # --- Load texts ---
    conditions: dict[str, list[str]] = {}
    source_descriptions: dict[str, str] = {}

    for attr_name, condition_name in [
        ("human_text", "human"),
        ("no_watermark_text", "no_watermark"),
        ("watermarked_text", "watermarked"),
    ]:
        flag_value = getattr(args, attr_name)
        if flag_value is None:
            continue

        source = parse_text_source(flag_value)
        texts = load_texts(source, seed=args.seed, n_texts=n_texts)

        if not texts:
            print(f"Warning: no texts loaded for --{attr_name.replace('_', '-')}")
            continue

        conditions[condition_name] = texts
        source_descriptions[condition_name] = source_description(source)
        print(f"Loaded {len(texts)} texts for '{condition_name}' from {source_descriptions[condition_name]}")

    if not conditions:
        print("Error: no texts loaded from any source")
        sys.exit(1)

    # --- Build flat work list ---
    work_items = []  # (global_idx, condition_name, local_idx, text)
    for condition_name, texts in conditions.items():
        for idx, text in enumerate(texts):
            work_items.append((len(work_items), condition_name, idx, text))

    total_items = len(work_items)

    token = load_hf_token()
    worker_config = {
        "model_name": args.model,
        "device": device,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "models_cache_dir": paths.MODELS_CACHE_DIR,
        "token": token,
    }

    # --- Spawn workers ---
    print(f"\nLoading evaluator model: {args.model}")

    results_queue = Queue()
    gpu_processes = []
    num_active_workers = 0

    if num_gpus <= 1:
        # Single GPU or CPU: run in one worker process
        if num_gpus == 1:
            worker_config["device"] = "cuda:0"
        p = Process(
            target=_perplexity_worker,
            args=(worker_config, work_items, results_queue),
            daemon=True,
        )
        p.start()
        gpu_processes.append(p)
        num_active_workers = 1
    else:
        # Multi-GPU: round-robin distribute across workers
        for gpu_id in range(num_gpus):
            gpu_items = [item for i, item in enumerate(work_items) if i % num_gpus == gpu_id]
            if not gpu_items:
                continue
            gpu_config = {**worker_config, "device": f"cuda:{gpu_id}"}
            p = Process(
                target=_perplexity_worker,
                args=(gpu_config, gpu_items, results_queue),
                daemon=True,
            )
            p.start()
            gpu_processes.append(p)
            num_active_workers += 1
        print(f"Started {num_active_workers} GPU workers across {num_gpus} GPUs")

    # --- Consumer loop ---
    condition_results: dict[str, list[dict]] = {cond: [] for cond in conditions}
    sentinels_received = 0
    progress = tqdm(total=total_items, desc="Computing perplexity", unit="text", disable=args.verbose)

    while sentinels_received < num_active_workers:
        item = results_queue.get()

        if item is None:
            sentinels_received += 1
            continue

        condition_results[item["condition"]].append({
            "index": item["local_idx"],
            "perplexity": item["perplexity"],
            "n_tokens": item["n_tokens"],
            "compute_time": item["compute_time"],
            "skipped": item["skipped"],
        })

        if not item["success"]:
            print(f"  [{item['condition']}] idx={item['local_idx']} ERROR: {item.get('error', 'Unknown')}")
        elif item["skipped"]:
            if args.verbose:
                print(
                    f"  [{item['condition']}] idx={item['local_idx']} "
                    f"SKIPPED (tokens={item['n_tokens']})"
                )
        elif args.verbose:
            print(
                f"  [{item['condition']}] idx={item['local_idx']} "
                f"PPL={item['perplexity']:.2f} tokens={item['n_tokens']} "
                f"time={item['compute_time']:.2f}s"
            )

        progress.update(1)

    progress.close()

    for p in gpu_processes:
        p.join()

    # --- Output ---
    print()
    _output_results(condition_results, source_descriptions, args)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
