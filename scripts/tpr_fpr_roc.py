#!/usr/bin/env python3
"""
TPR/FPR ROC Construction and Plotting

- Generates negative-set Hamming distances from random bit sequences against a codebook to compute FPR.
- Loads positive-set results from output/ (each run directory), aggregates per canonical model, and computes TPR.
- Produces per-model ROC plots and a combined ROC plot.
- Displays TPR at a target FPR (default 1%) in the legend using linear interpolation.

Usage examples (PowerShell-friendly):
  python scripts/tpr_fpr_roc.py --n 20 --u 128 --z 10000 --fpr-target 0.01 --verbose
  python scripts/tpr_fpr_roc.py --n 20 --u 128 --include-models meta-llama/Llama-3.2-3B --no-show
  python scripts/tpr_fpr_roc.py --n 20 --u 128 --z 10000 --tmin 8 --tmax 40 --fpr-target 0.01
  python scripts/tpr_fpr_roc.py --n 20 --u 128 --tmax 64 --include-models meta-llama/Llama-3.2-3B --no-show
"""

import os
import sys
import re
import argparse
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 1) Compute the project root (one level up from scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 2) Prepend it to sys.path so Python can see src/
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import src.paths as paths
from src.codebook import CodebookManager


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def sanitize_filename(name: str) -> str:
    # Replace any character that is not alphanumeric, dash, or underscore
    return re.sub(r"[^A-Za-z0-9_\-]+", "-", name)


def parse_results_txt(run_dir: str) -> Optional[Dict[str, str]]:
    """
    Extract canonical model name and n/u from results.txt within a run directory.
    Returns dict with keys: model, n, u. Returns None if not found or parse error.
    """
    results_path = os.path.join(run_dir, "results.txt")
    if not os.path.isfile(results_path):
        return None

    model = None
    n_val = None
    u_val = None

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped.startswith("Model:"):
                    # Model: meta-llama/Llama-3.2-3B-Instruct
                    model = line_stripped.split("Model:", 1)[1].strip()
                elif line_stripped.startswith("n (code table size):"):
                    # n (code table size): 20
                    try:
                        n_val = int(line_stripped.split(":", 1)[1].strip())
                    except Exception:
                        pass
                elif line_stripped.startswith("u (code size):"):
                    # u (code size): 128
                    try:
                        u_val = int(line_stripped.split(":", 1)[1].strip())
                    except Exception:
                        pass
    except Exception:
        return None

    if model is None or n_val is None or u_val is None:
        return None

    return {"model": model, "n": n_val, "u": u_val}


def load_generations_df(run_dir: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Load generations.parquet if available else generations.csv.
    Returns None if neither exists or load fails.
    """
    parquet_path = os.path.join(run_dir, "generations.parquet")
    csv_path = os.path.join(run_dir, "generations.csv")

    # Prefer Parquet
    if os.path.isfile(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            if verbose:
                print(f"Loaded {parquet_path} ({len(df)} rows)")
            return df
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to read parquet {parquet_path}: {e}")

    # Fallback to CSV
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if verbose:
                print(f"Loaded {csv_path} ({len(df)} rows)")
            return df
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to read csv {csv_path}: {e}")

    return None


def collect_positives_by_model(
    root: str,
    n: int,
    u: int,
    include_models: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Iterate over run directories in root, filter by n/u via results.txt, and aggregate positives per canonical model.
    Returns mapping model_name -> concatenated DataFrame of that model across datasets.
    """
    if not os.path.isdir(root):
        if verbose:
            print(f"Output root does not exist: {root}")
        return {}

    model_to_frames: Dict[str, List[pd.DataFrame]] = {}

    entries = sorted(os.listdir(root))
    for entry in entries:
        run_dir = os.path.join(root, entry)
        if not os.path.isdir(run_dir):
            continue

        meta = parse_results_txt(run_dir)
        if meta is None:
            if verbose:
                print(f"Skipping {run_dir}: missing or unparsable results.txt")
            continue

        if meta["n"] != n or meta["u"] != u:
            if verbose:
                print(f"Skipping {run_dir}: n/u mismatch (found n={meta['n']}, u={meta['u']})")
            continue

        model_name = meta["model"]
        if include_models:
            # substring match, case-insensitive
            target = model_name.lower()
            if not any(substr.lower() in target for substr in include_models):
                continue

        df = load_generations_df(run_dir, verbose=verbose)
        if df is None or df.empty:
            if verbose:
                print(f"Skipping {run_dir}: no generations file")
            continue

        # Ensure required columns exist
        if "successful_watermark" not in df.columns or "bit_code_hamming_distance" not in df.columns:
            if verbose:
                print(f"Skipping {run_dir}: required columns missing")
            continue

        # Robust type conversions
        # successful_watermark may be boolean or string 'True'/'False'
        if df["successful_watermark"].dtype != "bool" and df["successful_watermark"].dtype != "boolean":
            df["successful_watermark"] = df["successful_watermark"].map(
                lambda x: True if str(x).lower() in ("true", "1", "t", "yes") else False
            )

        df["bit_code_hamming_distance"] = pd.to_numeric(df["bit_code_hamming_distance"], errors="coerce")

        if model_name not in model_to_frames:
            model_to_frames[model_name] = []
        model_to_frames[model_name].append(df[["successful_watermark", "bit_code_hamming_distance"]].copy())

    # Concatenate per model
    out: Dict[str, pd.DataFrame] = {}
    for model, frames in model_to_frames.items():
        if frames:
            out[model] = pd.concat(frames, ignore_index=True)
            if verbose:
                print(f"Aggregated model {model}: {len(out[model])} rows")
    return out


def generate_negative_distances(
    n: int,
    u: int,
    z: int,
    seed: int,
    secret_key: str,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate z random sequences of length u, find closest codebook matches, and return their Hamming distances.
    """
    # Ensure paths setup for codebook cache
    paths.ensure_directories()

    codebook = CodebookManager(
        n=n,
        u=u,
        secret_key=secret_key,
        seed=seed,
        # Let CodebookManager use default cache dir (paths.CODEBOOKS_CACHE_DIR)
    )

    distances: List[int] = []

    itr = range(z)
    if verbose:
        itr = tqdm(itr, desc="Generating negatives (random sequences)")

    for _ in itr:
        # Generate random bit sequence of length u
        seq = [random.randint(0, 1) for _ in range(u)]
        matches = codebook.find_closest_matches(seq, top=1)
        if matches:
            _, _, d = matches[0]
            distances.append(int(d))

    return np.array(distances, dtype=np.int32)


def compute_curves(
    neg_distances: np.ndarray,
    positives_df: pd.DataFrame,
    u: int,
    tmin: int = 0,
    tmax: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute thresholds, FPR, and TPR arrays for thresholds in [tmin..tmax] inclusive.
    If tmax is None, it defaults to u.

    TPR definition:
      - Let Total = total rows in positives_df (all are positives by design)
      - Unsuccessful rows (successful_watermark == False) are always FN regardless of threshold
      - TP(t) = count(successful_watermark == True AND bit_code_hamming_distance <= t)
      - TPR(t) = TP(t) / Total

    FPR definition:
      - Negatives are the generated random sequences
      - FPR(t) = mean(neg_distances <= t)
    """
    # Clamp tmin/tmax to [0, u]
    if tmax is None:
        tmax = u
    tmin = int(max(0, min(u, int(tmin))))
    tmax = int(max(0, min(u, int(tmax))))
    if tmin > tmax:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    thresholds_full = np.arange(0, u + 1, dtype=np.int32)

    # FPR (full, then slice)
    if neg_distances.size == 0:
        fpr_full = np.zeros_like(thresholds_full, dtype=np.float64)
    else:
        hist_neg = np.bincount(neg_distances, minlength=u + 1)
        cdf_neg = np.cumsum(hist_neg)
        fpr_full = cdf_neg / float(len(neg_distances))

    # TPR (full, then slice)
    total = len(positives_df)
    if total == 0:
        tpr_full = np.zeros_like(thresholds_full, dtype=np.float64)
    else:
        mask_success = positives_df["successful_watermark"] == True
        succ_distances = (
            positives_df.loc[mask_success, "bit_code_hamming_distance"]
            .dropna()
            .astype(int)
            .to_numpy()
        )

        if succ_distances.size == 0:
            tp_counts = np.zeros_like(thresholds_full, dtype=np.int64)
        else:
            hist_succ = np.bincount(succ_distances, minlength=u + 1)
            cdf_succ = np.cumsum(hist_succ)
            tp_counts = cdf_succ

        tpr_full = tp_counts / float(total)

    # Slice to requested threshold range
    thresholds = np.arange(tmin, tmax + 1, dtype=np.int32)
    fpr = fpr_full[tmin : tmax + 1]
    tpr = tpr_full[tmin : tmax + 1]
    return thresholds, fpr, tpr


def interpolate_tpr_at_fpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target: float,
) -> Tuple[float, int, int, float]:
    """
    Linearly interpolate TPR at a given target FPR.
    Returns (tpr_at_target, idx_lo, idx_hi, alpha), where:
      - idx_lo, idx_hi define the bracketing indices in fpr
      - alpha is interpolation factor between them
    Clamps to endpoints if target is out of range or if a plateau occurs.
    """
    # Ensure arrays are floats
    fpr = fpr.astype(float)
    tpr = tpr.astype(float)

    # Handle degenerate cases
    if len(fpr) == 0 or len(tpr) == 0 or len(fpr) != len(tpr):
        return 0.0, 0, 0, 0.0

    # If target is below the minimum
    if target <= fpr[0]:
        return float(tpr[0]), 0, 0, 0.0
    # If above the maximum
    if target >= fpr[-1]:
        return float(tpr[-1]), len(fpr) - 1, len(fpr) - 1, 0.0

    # Find bracketing indices
    idx_hi = int(np.searchsorted(fpr, target, side="right"))
    idx_lo = idx_hi - 1
    if idx_lo < 0:
        idx_lo = 0
        idx_hi = 0

    x0, x1 = fpr[idx_lo], fpr[idx_hi]
    y0, y1 = tpr[idx_lo], tpr[idx_hi]

    if x1 == x0:
        # Plateau: return average or y0
        return float(y0), idx_lo, idx_hi, 0.0

    alpha = (target - x0) / (x1 - x0)
    y = y0 + alpha * (y1 - y0)
    return float(y), idx_lo, idx_hi, float(alpha)


def plot_per_model(
    model: str,
    thresholds: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    tpr_at_target: float,
    fpr_target: float,
    save_dir: str,
    n: int,
    u: int,
    z: int,
) -> str:
    """
    Produce and save a per-model ROC plot.
    Returns the path to the saved PNG.
    """
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"{model} (TPR@{int(fpr_target*100)}%FPR={tpr_at_target*100:.1f}%)", linewidth=2)
    # Mark the target point
    plt.scatter([fpr_target], [tpr_at_target], color="red", s=30, zorder=5)

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC — {model}\n(n={n}, u={u}, negatives z={z})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    model_clean = sanitize_filename(model)
    out_path = os.path.join(save_dir, f"roc_{model_clean}_n{n}_u{u}_z{z}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_combined(
    model_to_curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    fpr_target: float,
    save_dir: str,
    n: int,
    u: int,
    z: int,
) -> str:
    """
    Produce and save a combined ROC plot for all models.
    model_to_curves: model -> (thresholds, fpr, tpr, tpr_at_target)
    Returns saved path.
    """
    plt.figure(figsize=(8, 7))

    # Random baseline
    xx = np.linspace(0, 1, 1001)
    plt.plot(xx, xx, linestyle="--", color="gray", linewidth=1, label="Random baseline")

    for model, (thresholds, fpr, tpr, tpr_at_target) in model_to_curves.items():
        label = f"{model} (TPR@{int(fpr_target*100)}%FPR={tpr_at_target*100:.1f}%)"
        plt.plot(fpr, tpr, linewidth=2, label=label)

        # Mark the target point on this curve (x = fpr_target, y = interpolated TPR)
        plt.scatter([fpr_target], [tpr_at_target], s=25)

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC (Combined)\n(n={n}, u={u}, negatives z={z})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"roc_combined_n{n}_u{u}_z{z}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def save_roc_csv(save_dir: str, model: str, thresholds: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, n: int, u: int, z: int) -> str:
    model_clean = sanitize_filename(model)
    out_csv = os.path.join(save_dir, f"roc_points_{model_clean}_n{n}_u{u}_z{z}.csv")
    df = pd.DataFrame({
        "threshold": thresholds.astype(int),
        "FPR": fpr.astype(float),
        "TPR": tpr.astype(float),
    })
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Construct ROC curves (TPR/FPR) for models and report TPR@1%%FPR")
    parser.add_argument("--n", type=int, default=16, help="Size of the code table (2^n entries)")
    parser.add_argument("--u", type=int, default=100, help="Size of the codes (bit length)")
    parser.add_argument("--z", type=int, default=10000, help="Number of negative random sequences to test")
    parser.add_argument("--tmin", type=int, default=None, help="Minimum Hamming threshold (inclusive). Defaults to 0.")
    parser.add_argument("--tmax", type=int, default=None, help="Maximum Hamming threshold (inclusive). Defaults to u.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fpr-target", type=float, default=0.01, help="Target FPR for reporting TPR (default=0.01)")
    parser.add_argument("--output-root", type=str, default="output", help="Root directory containing model run outputs")
    parser.add_argument("--save-dir", type=str, default=os.path.join("scripts", "output"), help="Directory to save plots/CSVs")
    parser.add_argument("--include-models", type=str, nargs="+", help="Optional substrings to filter canonical model names")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (useful in headless runs)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output and progress information")

    args = parser.parse_args()

    # Paths setup
    paths.ensure_directories()
    ensure_dirs(args.save_dir)

    # Create timestamped run subfolder under save-dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"tpr_fpr_roc_n{args.n}_u{args.u}_z{args.z}_{timestamp}"
    run_save_dir = os.path.join(args.save_dir, run_dir_name)
    ensure_dirs(run_save_dir)
    if args.verbose:
        print(f"Saving outputs to: {run_save_dir}")

    # Initialize RNG for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generate a deterministic 64-bit secret key based on seed (same approach as main.py/hamming_distance_analysis.py)
    secret_key_int = random.getrandbits(64)
    secret_key = f"{secret_key_int:016x}"

    if args.verbose:
        print(f"Secret key: {secret_key}")
        print(f"Generating negatives: z={args.z}, u={args.u}")
        print(f"Collecting positives from: {args.output_root} (n={args.n}, u={args.u})")

    # Resolve threshold range
    tmin = 0 if args.tmin is None else max(0, min(args.u, int(args.tmin)))
    tmax = args.u if args.tmax is None else max(0, min(args.u, int(args.tmax)))
    if tmin > tmax:
        print(f"Invalid threshold range: tmin ({tmin}) > tmax ({tmax}).")
        sys.exit(1)
    if args.verbose:
        print(f"Threshold range: [{tmin}..{tmax}] (u={args.u})")

    # 1) FPR: negative set distances
    neg_distances = generate_negative_distances(
        n=args.n, u=args.u, z=args.z, seed=args.seed, secret_key=secret_key, verbose=args.verbose
    )
    if args.verbose:
        print(f"Collected {len(neg_distances)} negative distances")

    # 2) TPR: positives aggregated per canonical model
    model_to_df = collect_positives_by_model(
        root=args.output_root,
        n=args.n,
        u=args.u,
        include_models=args.include_models,
        verbose=args.verbose,
    )
    if not model_to_df:
        print("No matching model runs found for the specified n/u. Nothing to plot.")
        return

    # 3) Compute curves and 4) Interpolate TPR @ FPR target
    model_to_curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    for model, df in model_to_df.items():
        thresholds, fpr, tpr = compute_curves(neg_distances, df, args.u, tmin=tmin, tmax=tmax)
        tpr_at_target, _, _, _ = interpolate_tpr_at_fpr(fpr, tpr, args.fpr_target)
        model_to_curves[model] = (thresholds, fpr, tpr, tpr_at_target)

        # Save per-model ROC CSV and plot
        csv_path = save_roc_csv(run_save_dir, model, thresholds, fpr, tpr, args.n, args.u, args.z)
        png_path = plot_per_model(model, thresholds, fpr, tpr, tpr_at_target, args.fpr_target, run_save_dir, args.n, args.u, args.z)

        print(f"[{model}] TPR@{int(args.fpr_target*100)}%FPR = {tpr_at_target*100:.2f}% | Saved: {os.path.basename(png_path)}, {os.path.basename(csv_path)}")

    # 5) Combined plot
    combined_path = plot_combined(model_to_curves, args.fpr_target, run_save_dir, args.n, args.u, args.z)
    print(f"Combined ROC saved to: {combined_path}")

    # Optional: show plots interactively (if not in headless environment)
    if not args.no_show:
        # No live figures to show because we saved and closed them; this is intentional to avoid GUI dependence.
        pass


if __name__ == "__main__":
    main()
