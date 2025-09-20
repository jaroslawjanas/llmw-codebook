#!/usr/bin/env python3
"""
Intra-Codebook Spread (Nearest Neighbor Excluding Self)

Creates/loads a codebook (RNG or BCH) and samples z identities uniformly from within
the codebook. For each sampled identity, finds the nearest neighbor among all
other codewords (excluding itself) using exact Hamming distance via Faiss.

Outputs:
- CSV: nn_distances.csv with columns [watermark, neighbor_id, distance]
- JSON: parameters.json (run configuration)
- Text: summary.txt (basic statistics)

Outputs are saved under scripts/output/intra_codebook_spread_n{n}_u{u}_z{z}_{timestamp}/
Codebook caches (.npz and Faiss index) are stored in cache/codebooks via src.paths.
"""

import os
import sys

# 1) Compute the project root (one level up from scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 2) Prepend it to sys.path so Python can see src/
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.codebook import CodebookRNG, CodebookBCH
import src.paths as paths


def calculate_basic_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    arr = values.astype(np.int64, copy=False)
    stats = {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "variance": float(arr.var()),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Measure intra-codebook spread via nearest neighbor (excluding self)")
    parser.add_argument("--n", type=int, default=16, help="Size of the code table (2^n entries)")
    parser.add_argument("--u", type=int, default=100, help="Size of the codes (bit length)")
    parser.add_argument("--z", type=int, default=1000, help="Number of identities to sample from inside the codebook")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Secret key used for codebook generation (string)")
    parser.add_argument("--codebook", type=str, choices=["rng", "bch"], default="rng", help="Codebook type: 'rng' or 'bch'")
    parser.add_argument("--bch-t", type=int, default=None, help="BCH t (if None, auto-choose largest t with k >= n)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH m (n0=2^m-1). If None, minimal m with n0 >= u")
    parser.add_argument("--verbose", action="store_true", help="Show progress information")
    parser.add_argument("--output-dir", type=str, help="Output directory (auto-generated if not provided)")

    args = parser.parse_args()

    # RNG for reproducibility
    rng = np.random.default_rng(args.seed)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("scripts", "output", f"intra_codebook_spread_n{args.n}_u{args.u}_z{args.z}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    if args.verbose:
        print(f"Output directory: {output_dir}")

    # Build/load codebook (caches under cache/codebooks)
    if args.codebook == "rng":
        codebook = CodebookRNG(
            n=args.n,
            u=args.u,
            secret_key=args.secret_key,
            npz_dir=paths.CODEBOOKS_CACHE_DIR,
        )
    else:
        codebook = CodebookBCH(
            n=args.n,
            u=args.u,
            secret_key=args.secret_key,
            t=args.bch_t,
            m=args.bch_m,
            npz_dir=paths.CODEBOOKS_CACHE_DIR,
        )

    # Number of identities in codebook
    N = 2 ** args.n
    if N < 2:
        print("Codebook has fewer than 2 entries; nearest neighbor undefined.")
        return

    # Sample z identities without replacement from [0..N-1]
    z_eff = min(int(args.z), N)
    sample_ids = rng.choice(N, size=z_eff, replace=False)

    if args.verbose:
        print(f"Sampled {z_eff} identities from {N} total")

    # Query nearest neighbor excluding self:
    # Search for top-2; if the top-1 is self, take top-2
    queries = np.ascontiguousarray(codebook.packed_codes[sample_ids], dtype=np.uint8)
    D, I = codebook.index.search(queries, 2)  # (z_eff, 2)

    rows = []
    nn_distances = []

    for r in range(z_eff):
        qid = int(sample_ids[r])
        n0 = int(I[r, 0])
        d0 = int(D[r, 0])
        if n0 == qid:
            nn_id = int(I[r, 1])
            nn_d = int(D[r, 1])
        else:
            # Edge case: self wasn't ranked first (rare). Use first neighbor.
            nn_id = n0
            nn_d = d0

        rows.append({"watermark": qid, "neighbor_id": nn_id, "distance": nn_d})
        nn_distances.append(nn_d)

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "nn_distances.csv")
    df.to_csv(csv_path, index=False)

    # Save parameters
    params = {
        "n": args.n,
        "u": args.u,
        "z_requested": args.z,
        "z_effective": z_eff,
        "seed": args.seed,
        "secret_key": args.secret_key,
        "codebook": args.codebook,
        "bch_t": args.bch_t,
        "bch_m": args.bch_m,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Summary
    nn_arr = np.array(nn_distances, dtype=np.int32)
    stats = calculate_basic_stats(nn_arr)
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("INTRA-CODEBOOK NEAREST NEIGHBOR (EXCLUDING SELF)\n")
        f.write("=" * 60 + "\n")
        f.write(f"n={args.n}, u={args.u}, z_effective={z_eff}\n")
        f.write(f"codebook={args.codebook}, secret_key={args.secret_key}\n")
        f.write("\n")
        if stats:
            for k in ["count", "min", "max", "mean", "median", "std", "variance", "q25", "q75"]:
                f.write(f"{k}: {stats[k]}\n")
        else:
            f.write("No distances computed.\n")

    # Boxplot
    if nn_arr.size > 0:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.boxplot(nn_arr, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax.set_ylabel('Nearest Neighbor Hamming Distance')
            ax.set_title(f'Intra-Codebook 1-NN Distances (n={args.n}, u={args.u}, z={z_eff})')
            ax.grid(True, alpha=0.3)
            plot_path = os.path.join(output_dir, 'boxplot_nn_distances.png')
            fig.tight_layout()
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            if args.verbose:
                print(f"Boxplot saved to: {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to create boxplot: {e}")

    # Console summary
    print("\nIntra-codebook nearest neighbor analysis complete.")
    print(f"Saved results to: {output_dir}")
    print(f"Rows: {len(df)}")
    if stats:
        print(f"Distance range: {stats['min']} - {stats['max']}")
        print(f"Mean: {stats['mean']:.4f} | Median: {stats['median']:.4f} | Std: {stats['std']:.4f}")


if __name__ == "__main__":
    main()
