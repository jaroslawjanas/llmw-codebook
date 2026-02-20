#!/usr/bin/env python3
"""
Codebook Spread Analysis.

Measures the internal spread of a codebook by sampling k codewords and finding
each one's nearest neighbor in the full codebook (excluding self-match).
Reports distance statistics: min, max, mean, median, std, percentiles.

Usage examples:
  python scripts/codebook_spread.py --n 16 --u 100 --distance hamming --z 1000
  python scripts/codebook_spread.py --n 24 --u 127 --distance levenshtein-triton --z 500 --num-gpus 2
  python scripts/codebook_spread.py --n 24 --u 127 --distance hamming --codebook-generation linear --z 5000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.codebooks import (
    BCHCodewordGenerator,
    CodebookBase,
    CodebookHamming,
    CodebookLevenshtein,
    LinearCodeGenerator,
    RNGCodewordGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure codebook internal spread via nearest-neighbor distances"
    )
    # Codebook parameters
    parser.add_argument("--n", type=int, required=True, help="Codebook size parameter (2^n entries)")
    parser.add_argument("--u", type=int, required=True, help="Code length in bits")
    parser.add_argument("--distance", type=str, required=True, choices=["hamming", "levenshtein", "levenshtein-triton"], help="Distance metric")
    parser.add_argument("--codebook-generation", type=str, default="rng", choices=["rng", "bch", "linear"], help="Codeword generation algorithm (default: rng)")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Codebook secret key (default: 25111915)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (auto-select if None)")

    # Spread parameters
    parser.add_argument("--z", type=int, default=1000, help="Number of codewords to sample (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--num-gpus", type=int, default=0, help="GPUs for Triton search (0=all, default: 0)")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")

    return parser.parse_args()


def spread_hamming(
    codebook: CodebookHamming,
    sample_indices: np.ndarray,
    verbose: bool,
) -> np.ndarray:
    """Find nearest-neighbor distances for sampled codewords using Hamming search.

    Uses Faiss top-2 search: rank 0 is always self (distance 0), rank 1 is nearest other.
    """
    k = len(sample_indices)

    # Extract sampled codewords as packed uint8 for Faiss
    packed = codebook.base.packed_codes[sample_indices]  # (k, bytes_per_code)

    if verbose:
        print(f"Running Faiss top-2 search for {k} queries...")

    start = time.time()
    distances, indices = codebook.index.search(packed, 2)
    elapsed = time.time() - start

    if verbose:
        print(f"Search completed in {elapsed:.2f}s ({elapsed / k * 1000:.1f} ms/query)")

    # For each query, pick the nearest that isn't self
    nn_distances = np.empty(k, dtype=np.int32)
    for i in range(k):
        if indices[i, 0] == sample_indices[i]:
            # Rank 0 is self-match, use rank 1
            nn_distances[i] = distances[i, 1]
        else:
            # Rank 0 is already a different codeword (shouldn't happen normally)
            nn_distances[i] = distances[i, 0]

    return nn_distances


def spread_levenshtein(
    codebook,
    sample_indices: np.ndarray,
    base_codebook: CodebookBase,
    verbose: bool,
    codebooks: list | None = None,
) -> np.ndarray:
    """Find nearest-neighbor distances for sampled codewords using Levenshtein search.

    Uses top_k=2 search, then picks the non-self result.
    """
    k = len(sample_indices)

    # Extract sampled codewords as unpacked uint8 arrays
    queries = []
    for idx in sample_indices:
        cw = base_codebook.get_codeword_by_index(int(idx))
        queries.append(np.asarray(cw, dtype=np.uint8))

    if verbose:
        print(f"Running Levenshtein top-2 search for {k} queries...")

    start = time.time()

    if codebooks and len(codebooks) > 1:
        from scripts.utils.search import multi_gpu_triton_batch_search
        all_indices, _, all_distances = multi_gpu_triton_batch_search(
            codebooks, queries, top_k=2,
        )
    else:
        all_indices, _, all_distances = codebook.find_closest_match_batch(
            queries, top_k=2,
        )

    elapsed = time.time() - start

    if verbose:
        print(f"Search completed in {elapsed:.2f}s ({elapsed / k * 1000:.1f} ms/query)")

    # For each query, pick the nearest that isn't self
    nn_distances = np.empty(k, dtype=np.int32)
    for i in range(k):
        if all_indices[i, 0] == sample_indices[i]:
            nn_distances[i] = all_distances[i, 1]
        else:
            nn_distances[i] = all_distances[i, 0]

    return nn_distances


def save_results(
    distances: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Save histogram plot and raw distances CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    distance_label = args.distance.replace("-triton", "")
    output_dir = os.path.join(
        "output",
        f"spread_c-{args.codebook_generation}_n{args.n}_u{args.u}_z{len(distances)}_{distance_label}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    total = 2 ** args.n
    d_min = int(np.min(distances))
    d_max = int(np.max(distances))
    d_median = float(np.median(distances))

    title = (
        f"Nearest-Neighbor Distance Distribution\n"
        f"({args.codebook_generation}, n={args.n}, u={args.u}, {distance_label}, "
        f"z={len(distances):,} / {total:,})"
    )

    # Integer bins centered on distance values
    bins = np.arange(d_min, d_max + 2) - 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    counts, _, bars = ax.hist(distances, bins=bins, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, count,
                f"{int(count)}", ha="center", va="bottom", fontsize=7,
            )
    ax.axvline(d_median, color="red", linestyle="--", linewidth=1.5, label=f"Median = {d_median:.0f}")
    ax.set_xlabel(f"{distance_label.capitalize()} Distance")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(range(d_min, d_max + 1))
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spread.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nHistogram saved to: {plot_path}")

    # Save raw distances as CSV
    csv_path = os.path.join(output_dir, "spread.csv")
    np.savetxt(csv_path, distances, fmt="%d", header="nn_distance", comments="")
    print(f"Distances saved to: {csv_path}")


def print_stats(distances: np.ndarray, args: argparse.Namespace) -> None:
    """Print nearest-neighbor distance statistics."""
    total = 2 ** args.n

    print(f"\nCodebook spread analysis ({args.codebook_generation}, n={args.n}, u={args.u}, {args.distance})")
    print(f"Sampled {len(distances)} / {total:,} codewords")
    print()
    print("Nearest-neighbor distances:")
    print(f"  Min:    {int(np.min(distances))}")
    print(f"  Max:    {int(np.max(distances))}")
    print(f"  Mean:   {np.mean(distances):.1f}")
    print(f"  Median: {int(np.median(distances))}")
    print(f"  Std:    {np.std(distances):.1f}")
    print(f"  P5:     {int(np.percentile(distances, 5))}")
    print(f"  P25:    {int(np.percentile(distances, 25))}")
    print(f"  P75:    {int(np.percentile(distances, 75))}")
    print(f"  P95:    {int(np.percentile(distances, 95))}")


def main() -> None:
    args = parse_args()

    total = 2 ** args.n
    k = min(args.z, total)

    # Build codebook
    if args.codebook_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs = {}
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

    # Sample k codeword indices
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(total, size=k, replace=False)
    sample_indices.sort()

    if args.verbose:
        print(f"Sampled {k} codeword indices from codebook with {total:,} entries")

    # Build metric-specific codebook and search
    if args.distance == "hamming":
        codebook = CodebookHamming(base_codebook)
        nn_distances = spread_hamming(codebook, sample_indices, args.verbose)

    elif args.distance == "levenshtein-triton":
        from scripts.utils.search import (
            create_triton_codebooks,
            resolve_triton_gpu_count,
        )

        num_gpus = resolve_triton_gpu_count(args.num_gpus)
        codebooks = create_triton_codebooks(base_codebook, num_gpus)
        codebook = codebooks[0]
        nn_distances = spread_levenshtein(
            codebook, sample_indices, base_codebook, args.verbose,
            codebooks=codebooks if num_gpus > 1 else None,
        )

    else:  # levenshtein (Numba)
        codebook = CodebookLevenshtein(base_codebook)
        nn_distances = spread_levenshtein(
            codebook, sample_indices, base_codebook, args.verbose,
        )

    print_stats(nn_distances, args)
    save_results(nn_distances, args)


if __name__ == "__main__":
    main()
