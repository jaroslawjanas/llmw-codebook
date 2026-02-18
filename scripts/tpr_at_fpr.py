#!/usr/bin/env python3
"""
TPR @ Target FPR Analysis

- Generates negative-set distances from random bit sequences against a codebook to compute FPR.
- Loads positive-set results from output/ (each run directory), aggregates per canonical model.
- Finds the integer threshold closest to the target FPR.
- Prints a summary table and generates a per-model histogram plot.

Usage examples:
  python scripts/tpr_at_fpr.py --n 20 --u 128 --z 10000 --distance hamming --verbose
  python scripts/tpr_at_fpr.py --n 20 --u 128 --z 50000 --distance hamming --fpr-target 0.01
  python scripts/tpr_at_fpr.py --n 20 --u 128 --z 10000 --distance hamming --include-models Llama-3.2-3B
  python scripts/tpr_at_fpr.py --n 16 --u 128 --z 10000 --distance levenshtein --codebook-generation bch --bch-m 8 --bch-t 7
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import src.paths as paths
from scripts.utils import discover_experiments, load_and_merge
from src.codebooks import (
    BCHCodewordGenerator,
    CodebookBase,
    CodebookHamming,
    CodebookLevenshtein,
    CodebookLevenshteinTriton,
    LinearCodeGenerator,
    RNGCodewordGenerator,
)


def sanitize_filename(name: str) -> str:
    """Replace non-alphanumeric characters (except dash/underscore) with dash."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "-", name)


def shorten_model_name(name: str) -> str:
    """Extract short model name from canonical HuggingFace name (last component after /)."""
    return name.split("/")[-1] if "/" in name else name


# ---------------------------------------------------------------------------
# Negative distance generation + caching
# ---------------------------------------------------------------------------

def _build_cache_basename(args: argparse.Namespace) -> str:
    """Build a deterministic cache filename base from params (no extension)."""
    sk_clean = sanitize_filename(args.secret_key)
    distance = args.distance.replace("-triton", "")
    parts = [
        "fpr_negatives",
        args.codebook_generation,
        f"n{args.n}",
        f"u{args.u}",
        f"z{args.z}",
        f"seed{args.seed}",
        f"sk-{sk_clean}",
        distance,
    ]
    if args.codebook_generation == "bch":
        t_tag = "auto" if args.bch_t is None else str(args.bch_t)
        m_tag = "auto" if args.bch_m is None else str(args.bch_m)
        parts.append(f"bchm{m_tag}_bcht{t_tag}")
    return "_".join(parts)


def _build_cache_config(args: argparse.Namespace) -> dict:
    """Build the config dict stored alongside the cached NPZ."""
    return {
        "n": args.n,
        "u": args.u,
        "z": args.z,
        "seed": args.seed,
        "secret_key": args.secret_key,
        "codebook_generation": args.codebook_generation,
        "distance": args.distance,
        "bch_t": args.bch_t,
        "bch_m": args.bch_m,
    }


def _validate_cache_config(cached: dict, current: dict) -> bool:
    """Check that all fields in current match cached."""
    for key in current:
        if cached.get(key) != current[key]:
            return False
    return True


def load_or_generate_negatives(args: argparse.Namespace, codebook) -> np.ndarray:
    """Load cached negative distances or generate and cache them."""
    os.makedirs(paths.SCRIPTS_CACHE_DIR, exist_ok=True)

    basename = _build_cache_basename(args)
    npz_path = os.path.join(paths.SCRIPTS_CACHE_DIR, f"{basename}.npz")
    json_path = os.path.join(paths.SCRIPTS_CACHE_DIR, f"{basename}.json")
    current_config = _build_cache_config(args)

    # Try loading from cache
    if os.path.isfile(npz_path) and os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cached_config = json.load(f)
            if _validate_cache_config(cached_config, current_config):
                with np.load(npz_path) as data:
                    distances = data["distances"]
                if len(distances) == args.z:
                    print(f"Loaded negative distances from cache: {npz_path}")
                    return distances.astype(np.int32, copy=False)
        except Exception:
            pass  # Fall through to generation

    # Generate from scratch
    print(f"Generating {args.z} negative distances (distance={args.distance}, codebook={args.codebook_generation})...")
    distances = _generate_negative_distances(codebook, args.z, args.u, args.seed, args.verbose)

    # Save cache
    try:
        np.savez_compressed(npz_path, distances=distances.astype(np.int32, copy=False))
        config_to_save = {**current_config, "timestamp": datetime.now().isoformat()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, indent=2)
        print(f"Saved negative distances cache: {npz_path}")
    except Exception as e:
        print(f"Warning: failed to save cache: {e}")

    return distances


def _generate_negative_distances(codebook, z: int, u: int, seed: int, verbose: bool) -> np.ndarray:
    """Generate z random bit sequences and find closest codebook match for each.

    For Hamming (Faiss), uses batch search for speed.
    For Levenshtein (Numba), uses batch search for speed.
    """
    rng = np.random.default_rng(seed)

    if isinstance(codebook, CodebookHamming):
        # Batch Faiss search: generate all random bits, pack, and search in one call
        if verbose:
            print(f"Generating {z} random bit sequences...")
        random_bits = rng.integers(0, 2, size=(z, u), dtype=np.uint8)
        queries = CodebookBase._pack_bits_2d(random_bits, u)
        if verbose:
            print(f"Running batch Faiss search over {z} queries...")
        distances_arr, _ = codebook.index.search(queries, 1)
        return distances_arr[:, 0].astype(np.int32)
    else:
        # Levenshtein: batch Numba search
        if verbose:
            print(f"Generating {z} random bit sequences...")
        random_bits = rng.integers(0, 2, size=(z, u), dtype=np.uint8)
        if verbose:
            print(f"Running batch Levenshtein search over {z} queries...")
        return codebook.find_closest_match_batch(random_bits, distances_only=True).astype(np.int32)


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def find_threshold(neg_distances: np.ndarray, fpr_target: float) -> tuple[int, float]:
    """Find the integer threshold t that minimizes |FPR(t) - fpr_target|.

    Returns (threshold, actual_fpr).
    """
    sorted_dists = np.sort(neg_distances)
    z = len(sorted_dists)

    candidates = np.unique(sorted_dists)
    # Include one below minimum to cover FPR=0
    candidates = np.concatenate([[candidates[0] - 1], candidates])

    best_t = int(candidates[0])
    best_fpr = 0.0
    best_diff = abs(fpr_target)

    for t in candidates:
        count_leq = int(np.searchsorted(sorted_dists, t, side="right"))
        fpr_t = count_leq / z
        diff = abs(fpr_t - fpr_target)
        if diff < best_diff or (diff == best_diff and int(t) > best_t):
            best_t = int(t)
            best_fpr = fpr_t
            best_diff = diff

    return best_t, best_fpr


# ---------------------------------------------------------------------------
# Positive data loading
# ---------------------------------------------------------------------------

def load_positive_data(
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    """Discover experiments, filter by params, group by model, merge across datasets.

    Returns mapping model_name -> merged DataFrame.
    """
    exps = discover_experiments(args.output_root)
    if exps.empty:
        return {}

    # levenshtein-triton is the same metric as levenshtein (just a faster backend)
    distance_filter = args.distance.replace("-triton", "")
    mask = (
        (exps["n"] == args.n)
        & (exps["u"] == args.u)
        & (exps["distance"] == distance_filter)
        & (exps["codeword_generation"] == args.codebook_generation)
    )
    filtered = exps[mask]

    # Optional model substring filter
    if args.include_models:
        def matches_any(model: str) -> bool:
            target = model.lower()
            return any(s.lower() in target for s in args.include_models)
        filtered = filtered[filtered["model"].apply(matches_any)]

    if filtered.empty:
        return {}

    model_to_df: dict[str, pd.DataFrame] = {}
    for model, group in filtered.groupby("model"):
        try:
            df = load_and_merge(list(group["path"]))
            if not df.empty:
                model_to_df[str(model)] = df
        except Exception as e:
            print(f"Warning: failed to load data for {model}: {e}")

    return model_to_df


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_model_stats(
    model_to_df: dict[str, pd.DataFrame],
    threshold: int,
) -> list[dict]:
    """Compute per-model stats, sorted by TPR descending.

    Returns list of dicts with keys: model, total, true_positive, match_rate, tpr.
    """
    rows = []
    for model, df in model_to_df.items():
        total = len(df)
        true_positive = int(df["successful_watermark"].sum())
        match_rate = true_positive / total if total > 0 else 0.0
        successful = df[df["successful_watermark"] == True]
        detected = int((successful["distance"] <= threshold).sum())
        tpr = detected / total if total > 0 else 0.0
        rows.append({
            "model": model,
            "total": total,
            "true_positive": true_positive,
            "match_rate": match_rate,
            "tpr": tpr,
        })

    # Sort by model name
    rows.sort(key=lambda r: r["model"])
    return rows


# ---------------------------------------------------------------------------
# results.txt output
# ---------------------------------------------------------------------------

def write_results_txt(
    stats: list[dict],
    threshold: int,
    actual_fpr: float,
    neg_below: int,
    args: argparse.Namespace,
    save_path: str,
) -> None:
    """Write a formatted results.txt file."""
    fpr_pct = actual_fpr * 100
    tpr_header = f"TPR @ FPR={fpr_pct:.2f}%"

    lines = []
    lines.append("=" * 60)
    lines.append(f"TPR @ FPR = {args.fpr_target*100:.2f}% Analysis")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Parameters:")
    lines.append(f"  n (watermark bits): {args.n}")
    lines.append(f"  u (codeword length): {args.u}")
    lines.append(f"  z (negative samples): {args.z}")
    lines.append(f"  seed: {args.seed}")
    lines.append(f"  target FPR: {args.fpr_target*100:.2f}%")
    lines.append("")
    lines.append("FPR Threshold Calculation:")
    lines.append(f"  bit_error threshold: {threshold}")
    lines.append(f"  actual FPR: {fpr_pct:.2f}% ({neg_below}/{args.z} negatives below threshold)")
    lines.append("")
    lines.append("Model Results:")

    for row in stats:
        lines.append("-" * 60)
        lines.append(f"Model: {row['model']}")
        lines.append(f"  Total samples: {row['total']}")
        lines.append(f"  True Positives (TP): {row['true_positive']}")
        lines.append(f"  Match rate: {row['match_rate']*100:.2f}%")
        lines.append(f"  TPR: {row['tpr']*100:.2f}%")

    lines.append("-" * 60)
    lines.append("")
    lines.append("Summary Table:")

    # Find longest model name for alignment
    max_name = max(len(r["model"]) for r in stats)
    col_w = max(max_name, 5) + 2
    lines.append(f"  {'Model':<{col_w}} {'Match':<10} {tpr_header}")
    for row in stats:
        lines.append(
            f"  {row['model']:<{col_w}} {row['match_rate']*100:.2f}%"
            f"    {row['tpr']*100:.2f}%"
        )
    lines.append("")

    text = "\n".join(lines)

    # Print to console
    print(text)

    # Save to file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Results saved to: {save_path}")


# ---------------------------------------------------------------------------
# Table image output
# ---------------------------------------------------------------------------

def create_table_image(
    stats: list[dict],
    threshold: int,
    actual_fpr: float,
    args: argparse.Namespace,
    save_path: str,
) -> None:
    """Render a styled table as a PNG image."""
    fpr_pct = actual_fpr * 100
    n_data_rows = len(stats)

    # Build cell data
    col_labels = ["Model", "Total", "True Positives", "Match Rate", f"TPR@{fpr_pct:.2f}%"]
    cell_text = []
    for row in stats:
        cell_text.append([
            shorten_model_name(row["model"]),
            str(row["total"]),
            str(row["true_positive"]),
            f"{row['match_rate']*100:.1f}%",
            f"{row['tpr']*100:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(9, max(2, n_data_rows * 0.5 + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Explicit column widths
    col_widths = [0.32, 0.12, 0.16, 0.15, 0.18]
    for j, width in enumerate(col_widths):
        for i in range(n_data_rows + 1):
            table[(i, j)].set_width(width)

    # Style header row (blue background, white bold text)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")

    # Style data rows (alternating white / light blue)
    for i in range(1, n_data_rows + 1):
        bg = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(bg)

    plt.title(
        f"TPR @ FPR={fpr_pct:.2f}% (n={args.n}, u={args.u}, z={args.z})",
        fontsize=11, fontweight="bold", pad=20,
    )
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Table image saved to: {save_path}")


# ---------------------------------------------------------------------------
# Histogram plot
# ---------------------------------------------------------------------------

def create_histogram(
    neg_distances: np.ndarray,
    model_to_df: dict[str, pd.DataFrame],
    threshold: int,
    actual_fpr: float,
    args: argparse.Namespace,
    save_path: str,
) -> None:
    """Create a multi-subplot histogram figure (one subplot per model)."""
    models = sorted(model_to_df.keys())
    n_models = len(models)
    if n_models == 0:
        return

    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)

    # Shared bins for visual consistency
    # Collect all TP distances across models
    all_tp_dists = []
    model_tp_dists: dict[str, np.ndarray] = {}
    for model in models:
        df = model_to_df[model]
        tp_mask = df["successful_watermark"] == True
        dists = df.loc[tp_mask, "distance"].dropna().astype(int).to_numpy()
        model_tp_dists[model] = dists
        if len(dists) > 0:
            all_tp_dists.append(dists)

    all_tp_concat = np.concatenate(all_tp_dists) if all_tp_dists else np.array([], dtype=int)
    max_val = int(neg_distances.max())
    if len(all_tp_concat) > 0:
        max_val = max(max_val, int(all_tp_concat.max()))
    bins = np.arange(0, max_val + 2) - 0.5  # Center bins on integer values

    for idx, model in enumerate(models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        tp_dists = model_tp_dists[model]
        z = len(neg_distances)

        # Random (negative) histogram
        ax.hist(
            neg_distances, bins=bins, density=True,
            alpha=0.45, color="red", label=f"Random (n={z})",
        )

        # TP (successful watermark) histogram
        tp_count = len(tp_dists)
        if tp_count > 0:
            ax.hist(
                tp_dists, bins=bins, density=True,
                alpha=0.6, color="darkgreen", label=f"TP (n={tp_count})",
            )

        # Threshold line (offset +0.5 to center between bins)
        ax.axvline(x=threshold + 0.5, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={threshold}")

        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")
        ax.set_title(shorten_model_name(model))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fpr_pct = actual_fpr * 100
    plt.suptitle(
        f"Distance Distribution (n={args.n}, u={args.u}, FPR={fpr_pct:.2f}%)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to: {save_path}")

    if args.show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TPR @ Target FPR Analysis")

    # Required
    parser.add_argument("--n", type=int, required=True, help="Codebook size parameter (2^n entries)")
    parser.add_argument("--u", type=int, required=True, help="Code length in bits")
    parser.add_argument("--distance", type=str, required=True, choices=["hamming", "levenshtein", "levenshtein-triton"], help="Distance metric")

    # Optional
    parser.add_argument("--codebook-generation", type=str, default="rng", choices=["rng", "bch", "linear"], help="Codeword generation algorithm (default: rng)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative generation (default: 42)")
    parser.add_argument("--z", type=int, required=True, help="Number of random negative samples")
    parser.add_argument("--fpr-target", type=float, default=0.01, help="Target FPR (default: 0.01)")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Codebook secret key (default: 25111915)")
    parser.add_argument("--bch-t", type=int, default=None, help="BCH error-correction parameter t")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m")
    parser.add_argument("--output-root", type=str, default="output", help="Root directory with experiment outputs (default: output)")
    parser.add_argument("--save-dir", type=str, default="output", help="Directory to save analysis outputs (default: output)")
    parser.add_argument("--include-models", type=str, nargs="+", default=None, help="Substring filter for model names")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")

    args = parser.parse_args()

    # Setup paths
    paths.ensure_directories()

    # Build codebook for negative generation
    if args.codebook_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs: dict = {}
    elif args.codebook_generation == "linear":
        generator_cls = LinearCodeGenerator
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
    )
    if args.distance == "hamming":
        codebook = CodebookHamming(base_codebook)
    elif args.distance == "levenshtein-triton":
        codebook = CodebookLevenshteinTriton(base_codebook)
    else:
        codebook = CodebookLevenshtein(base_codebook)

    # Step 1: Generate or load negative distances
    neg_distances = load_or_generate_negatives(args, codebook)
    if args.verbose:
        print(f"Negative distances: {len(neg_distances)} samples, range [{neg_distances.min()}, {neg_distances.max()}]")

    # Step 2: Find threshold closest to target FPR
    threshold, actual_fpr = find_threshold(neg_distances, args.fpr_target)
    print(f"Threshold: {threshold} | Target FPR: {args.fpr_target*100:.2f}% | Actual FPR: {actual_fpr*100:.2f}%")

    # Step 3: Load positive data
    model_to_df = load_positive_data(args)
    if not model_to_df:
        print(f"No matching experiments found (n={args.n}, u={args.u}, distance={args.distance}, codebook={args.codebook_generation})")
        sys.exit(1)

    if args.verbose:
        for model, df in sorted(model_to_df.items()):
            print(f"  {model}: {len(df)} generations")

    # Step 4: Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = (
        f"tpr_at_fpr_n{args.n}_u{args.u}_z{args.z}"
        f"_dist-{args.distance}_{args.codebook_generation}"
        f"_fpr{args.fpr_target}_{timestamp}"
    )
    run_save_dir = os.path.join(args.save_dir, run_dir_name)
    os.makedirs(run_save_dir, exist_ok=True)

    # Step 5: Compute stats + write results.txt + table.png
    stats = compute_model_stats(model_to_df, threshold)
    neg_below = int((neg_distances <= threshold).sum())

    results_path = os.path.join(run_save_dir, "results.txt")
    write_results_txt(stats, threshold, actual_fpr, neg_below, args, results_path)

    table_img_path = os.path.join(run_save_dir, "table.png")
    create_table_image(stats, threshold, actual_fpr, args, table_img_path)

    # Step 6: Generate histogram
    histogram_path = os.path.join(run_save_dir, "histogram.png")
    create_histogram(neg_distances, model_to_df, threshold, actual_fpr, args, histogram_path)

    print(f"\nOutputs saved to: {run_save_dir}")


if __name__ == "__main__":
    main()
