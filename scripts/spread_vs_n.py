#!/usr/bin/env python3
"""
Codebook Spread vs n.

Sweeps n from 4 to a given maximum, measuring nearest-neighbor distance
statistics at each step for a fixed code length u. Produces a CSV table
and a line plot showing how spread changes with codebook density.

Usage examples:
  python scripts/spread_vs_n.py --n 20 --u 127 --distance hamming --z 1000
  python scripts/spread_vs_n.py --n 16 --u 127 --distance hamming --codebook-generation rng --z 2000
  python scripts/spread_vs_n.py --n 20 --u 127 --distance levenshtein-triton --z 500 --num-gpus 2
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
from scripts.codebook_spread import spread_hamming, spread_levenshtein


MIN_N = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure codebook spread across n values (4 to n) for fixed u"
    )
    # Codebook parameters
    parser.add_argument("--n", type=int, required=True, help="Maximum n value (sweep from 4 to n)")
    parser.add_argument("--u", type=int, required=True, help="Code length in bits")
    parser.add_argument("--distance", type=str, required=True, choices=["hamming", "levenshtein", "levenshtein-triton"], help="Distance metric")
    parser.add_argument("--codebook-generation", type=str, default="linear", choices=["rng", "bch", "linear"], help="Codeword generation algorithm (default: linear)")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Codebook secret key (default: 25111915)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (auto-select if None)")

    # Sweep parameters
    parser.add_argument("--z", type=int, default=1000, help="Samples per n step, capped at 2^n (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--num-gpus", type=int, default=0, help="GPUs for Triton search (0=all, default: 0)")
    parser.add_argument("--verbose", action="store_true", help="Detailed per-step output")

    return parser.parse_args()


def build_codebook_base(
    n_val: int,
    args: argparse.Namespace,
) -> CodebookBase:
    """Build a CodebookBase for the given n value."""
    if args.codebook_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs: dict = {}
    elif args.codebook_generation == "linear":
        generator_cls = LinearCodeGenerator
        generator_kwargs = {}
    else:
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"m": args.bch_m}

    return CodebookBase(
        n=n_val,
        u=args.u,
        secret_key=args.secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
    )


def measure_spread(
    n_val: int,
    args: argparse.Namespace,
) -> dict:
    """Measure nearest-neighbor distance statistics for a single n value."""
    total = 2 ** n_val
    k = min(args.z, total)

    base_codebook = build_codebook_base(n_val, args)

    # Sample codeword indices
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(total, size=k, replace=False)
    sample_indices.sort()

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

    return {
        "n": n_val,
        "samples": k,
        "min": int(np.min(nn_distances)),
        "mean": float(np.mean(nn_distances)),
        "median": float(np.median(nn_distances)),
        "max": int(np.max(nn_distances)),
    }


def save_results(
    rows: list[dict],
    args: argparse.Namespace,
) -> None:
    """Save CSV and line plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    distance_label = args.distance.replace("-triton", "")
    output_dir = os.path.join(
        "output",
        f"spread_vs_n_c-{args.codebook_generation}_u{args.u}_z{args.z}_{distance_label}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(output_dir, "spread_vs_n.csv")
    with open(csv_path, "w") as f:
        f.write("n,samples,min,mean,median,max\n")
        for r in rows:
            f.write(f"{r['n']},{r['samples']},{r['min']},{r['mean']:.1f},{r['median']:.1f},{r['max']}\n")
    print(f"\nCSV saved to: {csv_path}")

    # Plot
    ns = [r["n"] for r in rows]
    mins = [r["min"] for r in rows]
    means = [r["mean"] for r in rows]
    medians = [r["median"] for r in rows]
    maxs = [r["max"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(ns, mins, maxs, alpha=0.15, color="C0", label="Min–Max range")
    ax.plot(ns, means, "o-", color="C0", linewidth=2, markersize=4, label="Mean")
    ax.plot(ns, medians, "s--", color="C1", linewidth=1.5, markersize=4, label="Median")
    ax.plot(ns, mins, "v:", color="C2", linewidth=1, markersize=3, label="Min")
    ax.plot(ns, maxs, "^:", color="C3", linewidth=1, markersize=3, label="Max")

    ax.set_xlabel("n (codebook has $2^n$ entries)", fontsize=12)
    ax.set_ylabel(f"Nearest-Neighbor {distance_label.capitalize()} Distance", fontsize=12)
    ax.set_title(
        f"Codebook Spread vs n\n"
        f"({args.codebook_generation}, u={args.u}, {distance_label}, z={args.z})",
        fontsize=13,
    )
    ax.set_xticks(ns)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spread_vs_n.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


def main() -> None:
    args = parse_args()

    if args.n < MIN_N:
        print(f"Error: --n must be at least {MIN_N}")
        sys.exit(1)

    print(f"Spread vs n sweep: n={MIN_N}..{args.n}, u={args.u}, "
          f"generation={args.codebook_generation}, distance={args.distance}, z={args.z}")
    print()

    # Header
    print(f"{'n':>4}  {'2^n':>12}  {'samples':>7}  {'min':>5}  {'mean':>6}  {'median':>6}  {'max':>5}  {'time':>8}")
    print("-" * 68)

    rows: list[dict] = []

    for n_val in range(MIN_N, args.n + 1):
        start = time.time()
        row = measure_spread(n_val, args)
        elapsed = time.time() - start

        rows.append(row)

        total = 2 ** n_val
        print(
            f"{n_val:>4}  {total:>12,}  {row['samples']:>7,}  "
            f"{row['min']:>5}  {row['mean']:>6.1f}  {row['median']:>6.1f}  "
            f"{row['max']:>5}  {elapsed:>7.1f}s"
        )

    save_results(rows, args)


if __name__ == "__main__":
    main()
