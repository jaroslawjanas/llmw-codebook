#!/usr/bin/env python3
"""
Levenshtein Robustness Experiment — Insertion (Suffix) Attack

Description:
- Builds/loads a Levenshtein codebook (RNG or BCH-backed base codebook).
- Runs z trials. On each trial:
  * Select a random row (watermark_identity) using a deterministic RNG seeded by --seed
  * Get its codeword (length u)
  * Generate --suffix-size random bits (deterministic via a separate RNG from the same seed)
  * Create attacked_codeword = original_codeword + suffix (length u + suffix-size)
  * Query the Levenshtein codebook for the closest match
  * Check if retrieved watermark equals the original watermark_identity
- Prints match rate and distance stats for successful and unsuccessful matches.

Determinism:
- Two independent RNGs are created from a SeedSequence(args.seed):
  row_rng selects random row indices; suffix_rng generates random bits.
- Given identical args (including --seed), results are fully reproducible.

Usage examples (PowerShell-friendly):
  python scripts/levenshtein_robustness_experiment.py --n 16 --u 100 --z 5000 --suffix-size 10 --verbose
  python scripts/levenshtein_robustness_experiment.py --n 16 --u 100 --z 1000 --suffix-size 20 --codebook bch --bch-m 8 --bch-t 7
"""

import os
import sys

# 1) Compute the project root (one level up from scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 2) Prepend it to sys.path so Python can see src/
sys.path.insert(0, PROJECT_ROOT)

import argparse
import random
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.paths as paths
from src.codebooks import (
    CodebookBase,
    CodebookLevenshtein,
    RNGCodewordGenerator,
    BCHCodewordGenerator,
)


def positive_int(value: str) -> int:
    v = int(value)
    if v < 0:
        raise argparse.ArgumentTypeError("--suffix-size must be >= 0")
    return v


def main():
    parser = argparse.ArgumentParser(description="Levenshtein insertion-attack (suffix) robustness experiment")
    parser.add_argument("--n", type=int, default=16, help="Size of the code table (2^n entries)")
    parser.add_argument("--u", type=int, default=100, help="Size of the codes (bit length)")
    parser.add_argument("--z", type=int, default=1000, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Secret key used for codebook generation (string)")
    parser.add_argument("--suffix-size", type=positive_int, required=True, help="Number of random bits to append as a suffix (insertion attack)")
    parser.add_argument("--codebook", type=str, choices=["rng", "bch"], default="rng", help="Base codeword generator: 'rng' or 'bch'")
    parser.add_argument("--bch-t", type=int, default=None, help="BCH designed error-correction parameter t (if None, auto-choose largest t with k >= n)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (n0=2^m-1). If None, minimal m with n0 >= u")
    parser.add_argument("--verbose", action="store_true", help="Show progress information")
    args = parser.parse_args()

    # Optional global seeding for any accidental global RNG use
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Ensure cache directories exist
    paths.ensure_directories()

    if args.verbose:
        print(f"Params: n={args.n}, u={args.u}, z={args.z}, seed={args.seed}, secret_key={args.secret_key}")
        print(f"Codebook: {args.codebook} | suffix-size={args.suffix_size}")

    # Build/load base codebook
    if args.codebook == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs = {}
        cb_tag = "rng"
    else:
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"t": args.bch_t, "m": args.bch_m}
        t_tag = "auto" if args.bch_t is None else str(args.bch_t)
        m_tag = "auto" if args.bch_m is None else str(args.bch_m)
        cb_tag = f"bch_m{m_tag}_t{t_tag}"

    base = CodebookBase(
        n=args.n,
        u=args.u,
        secret_key=args.secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
        cache_dir=paths.CODEBOOKS_CACHE_DIR,
    )
    levenshtein_cb = CodebookLevenshtein(base)

    # Deterministic, independent RNGs for rows and suffix bits from a single seed
    ss = np.random.SeedSequence(args.seed)
    row_rng, suffix_rng = [np.random.default_rng(s) for s in ss.spawn(2)]

    N = 2 ** args.n
    expected_len = args.u + args.suffix_size

    rows = []
    iterator = range(args.z)
    if args.verbose:
        iterator = tqdm(iterator, desc="Insertion attack (suffix)")

    for i in iterator:
        # Select random watermark identity
        watermark_identity = int(row_rng.integers(0, N))

        # Fetch original codeword (length u)
        orig_bits = base.get_codeword_by_index(watermark_identity)  # np.ndarray (u,) uint8

        # Generate suffix bits (length suffix-size)
        if args.suffix_size > 0:
            suffix_bits = suffix_rng.integers(0, 2, size=args.suffix_size, dtype=np.uint8)
            attacked_codeword = np.concatenate([orig_bits, suffix_bits]).astype(np.uint8, copy=False)
        else:
            attacked_codeword = orig_bits  # no attack

        # Sanity check expected length
        if attacked_codeword.shape[0] != expected_len:
            # Only warn once in verbose mode
            if args.verbose:
                print(f"Warning: attacked_codeword length {attacked_codeword.shape[0]} != expected {expected_len}")

        # Query nearest by Levenshtein distance
        _, retrieved_watermark, distance = levenshtein_cb.find_closest_match(attacked_codeword)

        success = bool(retrieved_watermark == watermark_identity)

        rows.append({
            "iteration": i,
            "watermark_identity": int(watermark_identity),
            "retrieved_watermark": int(retrieved_watermark),
            "success": success,
            "distance": int(distance),
            "attacked_length": int(attacked_codeword.shape[0]),
        })

    # Aggregate in a DataFrame (kept in-memory; not saved to disk)
    df = pd.DataFrame(rows)

    total = len(df)
    success_count = int(df["success"].sum())
    success_rate = (success_count / total * 100.0) if total > 0 else 0.0

    # Split by success
    succ_df = df[df["success"] == True]["distance"]
    fail_df = df[df["success"] == False]["distance"]

    def stats_str(series: pd.Series) -> str:
        if series.empty:
            return "count=0"
        return f"count={len(series)}, mean={series.mean():.3f}, min={int(series.min())}, max={int(series.max())}"

    # Console output
    print("\nLevenshtein Robustness Experiment — Insertion (Suffix) Attack")
    print("==============================================================")
    print(f"n={args.n}, u={args.u}, z={args.z}, codebook={cb_tag}, secret_key={args.secret_key}")
    print(f"seed={args.seed}, suffix-size={args.suffix_size} -> attacked length={expected_len}")
    print(f"Matches: {success_count} / {total} ({success_rate:.2f}%)")
    print(f"Successful distances:   {stats_str(succ_df)}")
    print(f"Unsuccessful distances: {stats_str(fail_df)}")


if __name__ == "__main__":
    main()
