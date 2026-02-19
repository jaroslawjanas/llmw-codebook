#!/usr/bin/env python3
"""
Attack Robustness Analysis Script.

Tests how watermark detection degrades under substitution, insertion, and
deletion attacks applied to decoded codewords, reporting TPR at a fixed
FPR threshold.

Supports both Hamming and Levenshtein distance metrics.

For Hamming (fixed-length): insertions are truncated, deletions are zero-padded
to maintain length u. For Levenshtein (variable-length): attacked codewords are
passed as-is since edit distance naturally handles length changes.

Usage examples:
  python scripts/attack_robustness.py --n 20 --u 128 --z 10000 --distance hamming --attack-budget 0.1 --verbose
  python scripts/attack_robustness.py --n 20 --u 128 --z 10000 --distance levenshtein --attack-budget 1 4 9 13 --verbose
  python scripts/attack_robustness.py --n 20 --u 128 --z 10000 --distance hamming --attack-budget 1 3 5 7 10 --fpr-target 0.01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import src.paths as paths
from scripts.tpr_at_fpr import (
    find_threshold,
    load_or_generate_negatives,
    load_positive_data,
    sanitize_filename,
    shorten_model_name,
)
from scripts.utils import (
    create_triton_codebooks,
    multi_gpu_triton_batch_search,
    resolve_triton_gpu_count,
)
from src.codebooks import (
    BCHCodewordGenerator,
    CodebookBase,
    CodebookHamming,
    CodebookLevenshtein,
    LinearCodeGenerator,
    RNGCodewordGenerator,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test codebook watermark robustness against substitution, insertion, and deletion attacks"
    )
    # Required
    parser.add_argument("--n", type=int, required=True, help="Codebook size parameter (2^n entries)")
    parser.add_argument("--u", type=int, required=True, help="Code length in bits")
    parser.add_argument("--distance", type=str, required=True, choices=["hamming", "levenshtein", "levenshtein-triton"], help="Distance metric")
    parser.add_argument("--z", type=int, required=True, help="Number of random negative samples for FPR threshold")

    # Attack parameters
    parser.add_argument(
        "--attack-budget", type=float, nargs="+", required=True,
        help="Attack budgets. A single value < 1.0 is a fraction of u and generates range 0..ceil(frac*u) "
             "(e.g., 0.1 = every budget from 0 to ceil(10%%*u)). "
             "Integer values are specific bit counts (e.g., 1 4 9 13). Budget 0 is always included.",
    )

    # Codebook parameters
    parser.add_argument("--codebook-generation", type=str, default="rng", choices=["rng", "bch", "linear"], help="Codeword generation algorithm (default: rng)")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Codebook secret key (default: 25111915)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (n0=2^m-1). If None, auto-select optimal")

    # FPR parameters
    parser.add_argument("--fpr-target", type=float, default=0.01, help="Target FPR (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Data parameters
    parser.add_argument("--output-root", type=str, default="output", help="Root directory with experiment outputs (default: output)")
    parser.add_argument("--save-dir", type=str, default="output", help="Directory to save analysis outputs (default: output)")
    parser.add_argument("--include-models", type=str, nargs="+", default=None, help="Substring filter for model names")

    # GPU
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs for Triton search (0 = all available, default: 0)")

    # Display
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")

    args = parser.parse_args()

    # Resolve budgets: single fraction → range, otherwise explicit bit values
    raw = args.attack_budget
    if len(raw) == 1 and raw[0] < 1.0:
        # Percentage mode: generate range 0..ceil(frac * u)
        import math
        max_budget = math.ceil(raw[0] * args.u)
        budgets = list(range(0, max_budget + 1))
    else:
        # Explicit bit values
        budgets = sorted(set([0] + [int(b) for b in raw]))

    if any(b < 0 for b in budgets):
        parser.error("Attack budgets must be non-negative")
    args.budgets = budgets
    args.max_budget = budgets[-1]

    return args


# ---------------------------------------------------------------------------
# Attack functions (metric-agnostic)
# ---------------------------------------------------------------------------

def apply_substitution_attack(
    codeword: np.ndarray, budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Flip `budget` random bits. Returns array of length u (unchanged)."""
    if budget == 0:
        return codeword.copy()
    attacked = codeword.copy()
    indices = rng.choice(len(codeword), size=min(budget, len(codeword)), replace=False)
    attacked[indices] = 1 - attacked[indices]
    return attacked


def apply_insertion_attack(
    codeword: np.ndarray, budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Insert `budget` random bits at random positions. Returns array of length u + budget."""
    if budget == 0:
        return codeword.copy()
    result = list(codeword)
    for _ in range(budget):
        pos = rng.integers(0, len(result) + 1)
        bit = int(rng.integers(0, 2))
        result.insert(pos, bit)
    return np.array(result, dtype=np.uint8)


def apply_deletion_attack(
    codeword: np.ndarray, budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Delete `budget` random bits. Returns array of length u - budget."""
    if budget == 0:
        return codeword.copy()
    u = len(codeword)
    n_to_delete = min(budget, u - 1)  # Keep at least 1 bit
    delete_indices = set(rng.choice(u, size=n_to_delete, replace=False).tolist())
    result = [codeword[i] for i in range(u) if i not in delete_indices]
    return np.array(result, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Metric-specific adaptation
# ---------------------------------------------------------------------------

def adapt_for_hamming(attacked: np.ndarray, u: int) -> np.ndarray:
    """Truncate or pad to exactly length u for Hamming search."""
    if len(attacked) == u:
        return attacked
    elif len(attacked) > u:
        return attacked[:u]
    else:
        padded = np.zeros(u, dtype=np.uint8)
        padded[: len(attacked)] = attacked
        return padded


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _bits_to_str(arr: np.ndarray) -> str:
    """Convert a uint8 bit array to a bitstring like '01101'."""
    return "".join(str(int(b)) for b in arr)


def evaluate_attack_robustness(
    decoded_codewords: list[np.ndarray],
    original_watermarks: np.ndarray,
    original_codewords: list[str],
    total_samples: int,
    codebook,
    attack_fn: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
    budgets: list[int],
    threshold: int,
    u: int,
    seed: int,
    attack_name: str,
    codebooks: list | None = None,
) -> tuple[dict[int, float], list[dict]]:
    """
    Evaluate TPR at each specified budget level.

    Args:
        decoded_codewords: Decoded codewords from successful watermark samples only.
        original_watermarks: Original watermark identities for each decoded codeword.
        original_codewords: Bitstrings of the original codebook codewords (pre-computed).
        total_samples: Total number of samples (all rows) for TPR denominator.
        codebook: CodebookHamming or CodebookLevenshtein instance.
        attack_fn: Attack function to apply.
        budgets: List of budget values to evaluate (must include 0 for baseline).
        threshold: Distance threshold from FPR analysis.
        u: Codeword length.
        seed: Random seed.
        attack_name: Name for progress bar display.

    Returns:
        Tuple of (budget_to_tpr dict, per-sample detail rows).
    """
    is_hamming = isinstance(codebook, CodebookHamming)
    results = {}
    detail_rows: list[dict] = []

    for budget in tqdm(budgets, desc=f"  {attack_name}"):
        rng = np.random.default_rng(seed + budget)

        if is_hamming:
            # Batch: collect all attacked codewords, pack, search in one Faiss call
            attacked_list = []
            for cw in decoded_codewords:
                attacked = attack_fn(cw, budget, rng)
                adapted = adapt_for_hamming(attacked, u)
                attacked_list.append(adapted)

            if attacked_list:
                attacked_matrix = np.stack(attacked_list)
                queries = CodebookBase._pack_bits_2d(attacked_matrix, u)
                distances_arr, indices = codebook.index.search(queries, 1)
                retrieved_wm = codebook.base.watermark_identities[indices[:, 0]]
                dist_flat = distances_arr[:, 0]
                correct_id = (retrieved_wm == original_watermarks)
                tp = int((correct_id & (dist_flat <= threshold)).sum())

                for i in range(len(attacked_list)):
                    rec_cw = codebook.base.get_codeword_by_index(int(retrieved_wm[i]))
                    detail_rows.append({
                        "sample_index": i,
                        "budget": budget,
                        "original_watermark": int(original_watermarks[i]),
                        "original_codeword": original_codewords[i],
                        "decoded_codeword": _bits_to_str(decoded_codewords[i]),
                        "attacked_codeword": _bits_to_str(attacked_list[i]),
                        "recovered_watermark": int(retrieved_wm[i]),
                        "recovered_codeword": _bits_to_str(rec_cw),
                        "distance": int(dist_flat[i]),
                        "successful_watermark": bool(correct_id[i]),
                    })
            else:
                tp = 0
        else:
            # Levenshtein: batch search (handles variable-length codewords)
            attacked_list = [attack_fn(cw, budget, rng) for cw in decoded_codewords]
            if attacked_list:
                if codebooks and len(codebooks) > 1:
                    best_indices, retrieved_wm, dist_arr = multi_gpu_triton_batch_search(
                        codebooks, attacked_list,
                    )
                else:
                    best_indices, retrieved_wm, dist_arr = codebook.find_closest_match_batch(attacked_list)
                correct_id = (retrieved_wm == original_watermarks)
                tp = int((correct_id & (dist_arr <= threshold)).sum())

                for i in range(len(attacked_list)):
                    rec_cw = codebook.base.get_codeword_by_index(int(best_indices[i]))
                    detail_rows.append({
                        "sample_index": i,
                        "budget": budget,
                        "original_watermark": int(original_watermarks[i]),
                        "original_codeword": original_codewords[i],
                        "decoded_codeword": _bits_to_str(decoded_codewords[i]),
                        "attacked_codeword": _bits_to_str(attacked_list[i]),
                        "recovered_watermark": int(retrieved_wm[i]),
                        "recovered_codeword": _bits_to_str(rec_cw),
                        "distance": int(dist_arr[i]),
                        "successful_watermark": bool(correct_id[i]),
                    })
            else:
                tp = 0

        tpr = tp / total_samples if total_samples > 0 else 0.0
        results[budget] = tpr

    return results, detail_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_model_plot(
    model: str,
    attack_results: dict[str, dict[int, float]],
    actual_fpr: float,
    args: argparse.Namespace,
    save_path: str,
) -> None:
    """Create PNG with 3 subplots: substitution, insertion, deletion."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for (attack_name, results), ax in zip(attack_results.items(), axes):
        budgets = list(results.keys())
        tpr_pcts = [r * 100 for r in results.values()]
        baseline = tpr_pcts[0]

        ax.plot(budgets, tpr_pcts, "-", color="#4A90D9", linewidth=1.5)
        ax.plot(budgets[1:], tpr_pcts[1:], "o", color="#4A90D9", markersize=5)

        # Baseline
        ax.axhline(y=baseline, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(
            budgets[-1] - 0.5, baseline - 4, f"{baseline:.1f}%",
            color="green", fontsize=9, ha="right", va="top",
        )

        # Percentage labels for each point (skip first — shown by baseline)
        for x, y in zip(budgets[1:], tpr_pcts[1:]):
            y_offset = 5 if y < 90 else -12
            ax.annotate(
                f"{y:.0f}%", (x, y), textcoords="offset points",
                xytext=(0, y_offset), ha="center", fontsize=7, color="#333333",
            )

        ax.set_xlabel("Number of adversarial interventions")
        ax.set_ylabel(f"TPR @ FPR={actual_fpr * 100:.2f}%")
        ax.set_title(attack_name)
        ax.set_ylim(0, 105)
        ax.set_xlim(budgets[0], budgets[-1])
        if len(budgets) <= 20:
            ax.set_xticks(budgets)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Attack Robustness: {shorten_model_name(model)}\n"
        f"(n={args.n}, u={args.u}, dist={args.distance}, FPR={actual_fpr * 100:.2f}%)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results_txt(
    args: argparse.Namespace,
    threshold: int,
    actual_fpr: float,
    all_results: dict[str, dict[str, dict[int, float]]],
    save_path: str,
) -> None:
    """Write formatted results with one table per attack type."""
    lines = []
    lines.append("=" * 70)
    lines.append("Codebook Attack Robustness Analysis")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Parameters:")
    lines.append(f"  n (watermark bits): {args.n}")
    lines.append(f"  u (codeword length): {args.u}")
    lines.append(f"  distance metric: {args.distance}")
    lines.append(f"  codebook generation: {args.codebook_generation}")
    lines.append(f"  attack budgets: {args.budgets}")
    lines.append(f"  z (negative samples): {args.z}")
    lines.append(f"  target FPR: {args.fpr_target * 100:.2f}%")
    lines.append(f"  distance threshold: {threshold}")
    lines.append(f"  actual FPR: {actual_fpr * 100:.2f}%")
    lines.append(f"  seed: {args.seed}")
    lines.append("")

    # Collect model short names and attack names
    models = list(all_results.keys())
    short_names = [shorten_model_name(m) for m in models]
    model_col_width = max(len(s) for s in short_names) + 2
    budget_col_width = 7  # enough for "100.00"

    # Get attack names from first model
    attack_names = list(next(iter(all_results.values())).keys())
    budgets = args.budgets

    for attack_name in attack_names:
        lines.append(f"TPR (%) @ FPR={actual_fpr * 100:.2f}% — {attack_name}")
        # Header row: model column + budget columns
        header = " " * model_col_width
        header += "".join(f"{b:>{budget_col_width}}" for b in budgets)
        lines.append(header)
        lines.append("-" * len(header))

        for model, short in zip(models, short_names):
            results = all_results[model][attack_name]
            row = f"{short:<{model_col_width}}"
            row += "".join(f"{results[b] * 100:>{budget_col_width}.2f}" for b in budgets)
            lines.append(row)

        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)


def write_model_detail_csv(
    detail_rows: list[dict],
    save_path: str,
) -> None:
    """Write per-sample attack detail rows as CSV for offline re-analysis."""
    pd.DataFrame(detail_rows).to_csv(save_path, index=False)


def write_metadata_csv(
    model_metadata: list[dict],
    save_path: str,
) -> None:
    """Write per-model metadata CSV with sample counts and FPR info."""
    pd.DataFrame(model_metadata).to_csv(save_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    paths.ensure_directories()

    # Build codebook (same pattern as tpr_at_fpr.py)
    if args.codebook_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs: dict = {}
    elif args.codebook_generation == "linear":
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
    )
    codebooks = None
    if args.distance == "hamming":
        codebook = CodebookHamming(base_codebook)
    elif args.distance == "levenshtein-triton":
        num_gpus = resolve_triton_gpu_count(args.num_gpus)
        codebooks = create_triton_codebooks(base_codebook, num_gpus)
        codebook = codebooks[0]
        if args.verbose:
            print(f"Triton search across {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
    else:
        codebook = CodebookLevenshtein(base_codebook)

    # Step 1: Compute FPR threshold
    neg_distances = load_or_generate_negatives(args, codebook, codebooks=codebooks)
    threshold, actual_fpr = find_threshold(neg_distances, args.fpr_target)
    print(f"Threshold: {threshold} | Target FPR: {args.fpr_target * 100:.2f}% | Actual FPR: {actual_fpr * 100:.2f}%")

    # Step 2: Load positive data
    model_to_df = load_positive_data(args)
    if not model_to_df:
        print(f"No matching experiments found (n={args.n}, u={args.u}, distance={args.distance}, codebook={args.codebook_generation})")
        sys.exit(1)

    # Step 3: Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use raw CLI values for folder name (e.g., "budget-0.1" or "budget-1-4-9-13")
    budget_str = "-".join(
        str(int(b)) if b == int(b) else str(b) for b in args.attack_budget
    )
    run_dir_name = (
        f"attack_robustness_n{args.n}_u{args.u}"
        f"_dist-{args.distance}_{args.codebook_generation}"
        f"_budget-{budget_str}_{timestamp}"
    )
    run_save_dir = os.path.join(args.save_dir, run_dir_name)
    os.makedirs(run_save_dir, exist_ok=True)

    # Step 4: Run attacks per model
    all_results: dict[str, dict[str, dict[int, float]]] = {}
    model_metadata: list[dict] = []

    for model, df in sorted(model_to_df.items()):
        print(f"\nProcessing {shorten_model_name(model)}...")

        total_samples = len(df)
        successful = df[df["successful_watermark"] == True]

        model_metadata.append({
            "model": model,
            "total_samples": total_samples,
            "successful_samples": len(successful),
            "fpr_threshold": threshold,
            "fpr_percentage": actual_fpr * 100,
        })

        # Parse decoded codewords and original watermark identities from successful samples
        decoded_codewords = [
            np.array([int(b) for b in str(cw)], dtype=np.uint8)
            for cw in successful["decoded_codeword"].values
        ]
        original_watermarks = successful["watermark"].values.astype(np.int64)

        # Pre-compute original codebook codewords as bitstrings
        original_codewords = [
            _bits_to_str(codebook.base.get_codeword_by_index(int(wm)))
            for wm in original_watermarks
        ]

        print(f"  Loaded {len(decoded_codewords)} successful / {total_samples} total samples")

        attack_results: dict[str, dict[int, float]] = {}
        model_detail_rows: list[dict] = []
        for attack_name, attack_fn in [
            ("Substitution", apply_substitution_attack),
            ("Insertion", apply_insertion_attack),
            ("Deletion", apply_deletion_attack),
        ]:
            tpr_dict, detail_rows = evaluate_attack_robustness(
                decoded_codewords=decoded_codewords,
                original_watermarks=original_watermarks,
                original_codewords=original_codewords,
                total_samples=total_samples,
                codebook=codebook,
                attack_fn=attack_fn,
                budgets=args.budgets,
                threshold=threshold,
                u=args.u,
                seed=args.seed,
                attack_name=attack_name,
                codebooks=codebooks,
            )
            attack_results[attack_name] = tpr_dict
            for row in detail_rows:
                row["attack"] = attack_name
            model_detail_rows.extend(detail_rows)

        all_results[model] = attack_results

        # Per-model detail CSV
        safe_model = sanitize_filename(model)
        csv_path = os.path.join(run_save_dir, f"{safe_model}_details.csv")
        write_model_detail_csv(model_detail_rows, csv_path)
        print(f"  Saved detail CSV: {csv_path}")

        # Per-model plot
        plot_path = os.path.join(run_save_dir, f"{safe_model}_robustness.png")
        create_model_plot(model, attack_results, actual_fpr, args, plot_path)
        print(f"  Saved plot: {plot_path}")

    # Step 5: Write results
    results_path = os.path.join(run_save_dir, "results.txt")
    write_results_txt(args, threshold, actual_fpr, all_results, results_path)

    metadata_path = os.path.join(run_save_dir, "metadata.csv")
    write_metadata_csv(model_metadata, metadata_path)

    print(f"\nOutputs saved to: {run_save_dir}")


if __name__ == "__main__":
    main()
