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
import zlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypedDict, cast

# 1) Compute the project root (one level up from scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 2) Prepend it to sys.path so Python can see src/
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from tqdm import tqdm
import src.paths as paths
from src.codebooks import CodebookBase, CodebookHamming, CodebookLevenshtein, RNGCodewordGenerator, BCHCodewordGenerator


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def sanitize_filename(name: str) -> str:
    # Replace any character that is not alphanumeric, dash, or underscore
    return re.sub(r"[^A-Za-z0-9_\-]+", "-", name)


class RunMeta(TypedDict):
    model: str
    n: int
    u: int


def parse_results_txt(run_dir: str) -> Optional[RunMeta]:
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

    return {"model": str(model), "n": int(n_val), "u": int(u_val)}


def load_generations_df(run_dir: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Load generations.csv.
    Returns None if not found or load fails.
    """
    csv_path = os.path.join(run_dir, "generations.csv")

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
        if "successful_watermark" not in df.columns or "distance" not in df.columns:
            if verbose:
                print(f"Skipping {run_dir}: required columns missing")
            continue

        # Robust type conversions
        # successful_watermark may be boolean or string 'True'/'False'
        if df["successful_watermark"].dtype != "bool" and df["successful_watermark"].dtype != "boolean":
            df["successful_watermark"] = df["successful_watermark"].map(
                lambda x: True if str(x).lower() in ("true", "1", "t", "yes") else False
            )

        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

        if model_name not in model_to_frames:
            model_to_frames[model_name] = []
        sub_df = cast(pd.DataFrame, df[["successful_watermark", "distance"]].copy())
        model_to_frames[model_name].append(sub_df)

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
    codebook: str,
    distance: str,
    bch_t: Optional[int],
    bch_m: Optional[int],
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate or load cached negative distances (Hamming/Levenshtein distances of random sequences to the closest code).
    Caching strategy:
      - Exact hit: load cache file for (n,u,z,seed,secret_key,codebook,distance).
      - Downsample: if a larger cache exists with same (n,u,seed,secret_key,codebook,distance) and z_big >= z,
        deterministically sample z entries without replacement, save the exact cache, and return.
      - Miss: generate with a local seeded RNG, save, and return.
    The cache key includes the codebook family, BCH params, and the distance metric to avoid mixing results across algorithms.
    """
    # Ensure paths setup for caches
    paths.ensure_directories()
    os.makedirs(paths.SCRIPTS_CACHE_DIR, exist_ok=True)

    sk_clean = sanitize_filename(secret_key)
    if codebook == "rng":
        base_tag = "rng"
    else:
        # Note: we include provided parameters; if auto, include 'auto' in tag to reflect resolution
        t_tag = "auto" if bch_t is None else str(bch_t)
        m_tag = "auto" if bch_m is None else str(bch_m)
        base_tag = f"bch_m{m_tag}_t{t_tag}"
    # Include distance metric to avoid cache/file collisions across metrics
    cb_tag = f"{base_tag}-{distance}"

    def build_cache_path(z_val: int) -> str:
        fname = f"negative_distances_z{z_val}_s{seed}_n{n}_u{u}_cb{cb_tag}_sk{sk_clean}.npz"
        return os.path.join(paths.SCRIPTS_CACHE_DIR, fname)

    exact_path = build_cache_path(z)

    # 1) Try exact cache
    if os.path.isfile(exact_path):
        try:
            with np.load(exact_path) as npz:
                distances = npz["distances"].astype(np.int32, copy=False)
            if distances.shape[0] == z:
                print(f"Loaded negative distances from cache: {exact_path} (z={z}, seed={seed}, n={n}, u={u}, cb={cb_tag}, sk={secret_key})")
                return distances
        except Exception:
            # Corrupt cache, fall through to search/generate
            pass

    # 2) Try larger cache for downsampling (same codebook tag)
    pattern = re.compile(rf"^negative_distances_z(\d+)_s{seed}_n{n}_u{u}_cb{re.escape(cb_tag)}_sk{re.escape(sk_clean)}\.npz$")
    best_path = None
    best_z = -1
    try:
        for name in os.listdir(paths.SCRIPTS_CACHE_DIR):
            m = pattern.match(name)
            if not m:
                continue
            z_found = int(m.group(1))
            if z_found > best_z:
                best_z = z_found
                best_path = os.path.join(paths.SCRIPTS_CACHE_DIR, name)
    except Exception:
        best_path = None
        best_z = -1

    if best_path is not None and best_z >= z and os.path.abspath(best_path) != os.path.abspath(exact_path):
        try:
            with np.load(best_path) as npz:
                big = npz["distances"].astype(np.int32, copy=False)
            if big.shape[0] >= z:
                # Deterministic subset using a derived seed (stable across runs)
                sk_crc = zlib.crc32(secret_key.encode("utf-8")) & 0xFFFFFFFF
                ss = np.random.SeedSequence([seed, z, n, u, sk_crc])
                rng = np.random.default_rng(ss)
                idx = rng.choice(big.shape[0], size=z, replace=False)
                distances = big[idx]
                # Save the downsampled exact cache
                try:
                    np.savez_compressed(exact_path, distances=distances.astype(np.int32, copy=False))
                except Exception:
                    pass
                print(f"Loaded larger cache and downsampled {best_z}→{z} from {best_path}; wrote: {exact_path}")
                return distances
        except Exception:
            # Fall back to generation
            pass

    # 3) Generate from scratch with a local seeded RNG (independent of global state)
    if codebook == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs = {}
    else:
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"t": bch_t, "m": bch_m}

    base = CodebookBase(
        n=n,
        u=u,
        secret_key=secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
    )
    if distance == "hamming":
        cb = CodebookHamming(base)
    else:
        cb = CodebookLevenshtein(base)

    distances_list: List[int] = []
    itr = range(z)
    if verbose:
        itr = tqdm(itr, desc="Generating negatives (random sequences)")

    rng = np.random.default_rng(seed)
    for _ in itr:
        # Generate random bit sequence of length u using local RNG
        seq = rng.integers(0, 2, size=u, dtype=np.uint8).tolist()
        # Exact top-1 match using Faiss
        _, _, d = cb.find_closest_match(seq)
        distances_list.append(int(d))

    distances = np.array(distances_list, dtype=np.int32)

    # Save cache
    try:
        np.savez_compressed(exact_path, distances=distances)
        print(f"Generated negative distances and saved cache to: {exact_path} (z={len(distances)})")
    except Exception:
        print(f"Generated negative distances (cache save failed): {exact_path}")

    return distances


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
      - TP(t) = count(successful_watermark == True AND distance <= t)
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
            positives_df.loc[mask_success, "distance"]
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
    cb_tag: str,
    scale: str = "linear",
    x_eps: Optional[float] = None,
    fpr_min: Optional[float] = None,
    fpr_max: Optional[float] = None,
) -> str:
    """
    Produce and save a per-model ROC plot.
    Returns the path to the saved PNG.
    Supports axis scaling modes via `scale`: "linear", "logx", "symlogx", "loglog".
    - logx: logarithmic x-axis (FPR), replaces FPR=0 with a small epsilon.
    - symlogx: symmetric log x-axis with small linear region around 0 (keeps FPR=0 visible).
    - loglog: logarithmic x and y axes (TPR zeros replaced with epsilon for plotting).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    s = (scale or "linear").lower()
    fpr_plot = fpr.copy()
    tpr_plot = tpr.copy()

    # Determine epsilons and x range
    min_positive_fpr = float(np.min(fpr[fpr > 0])) if np.any(fpr > 0) else None
    auto_eps = max((min_positive_fpr / 10.0 if min_positive_fpr is not None else 1.0 / (10.0 * max(z, 1))), 1e-12)
    eps = x_eps if x_eps is not None else auto_eps

    if s == "linear":
        ax.set_xlim(0.0, 1.0)
    elif s in ("logx", "loglog"):
        fpr_plot = np.where(fpr > 0, fpr, eps)
        ax.set_xscale("log")
        x_min = fpr_min if fpr_min is not None else max(min_positive_fpr if min_positive_fpr is not None else eps, eps)
        x_max = fpr_max if fpr_max is not None else 1.0
        ax.set_xlim(x_min, x_max)
        if s == "loglog":
            min_positive_tpr = float(np.min(tpr[tpr > 0])) if np.any(tpr > 0) else None
            y_eps = max((min_positive_tpr / 10.0 if min_positive_tpr is not None else 1e-4), 1e-12)
            tpr_plot = np.where(tpr > 0, tpr, y_eps)
            ax.set_yscale("log")
    elif s == "symlogx":
        # Keep zeros; show a small linear neighborhood near 0
        linthresh = eps
        ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
        x_min = fpr_min if fpr_min is not None else 0.0
        x_max = fpr_max if fpr_max is not None else 1.0
        ax.set_xlim(x_min, x_max)

    ax.plot(fpr_plot, tpr_plot, label=f"{model} (TPR@{int(fpr_target*100)}%FPR={tpr_at_target*100:.1f}%, cb={cb_tag})", linewidth=2)
    # Mark the target point
    ax.scatter([fpr_target if s == "linear" else max(fpr_target, eps)], [tpr_at_target], color="red", s=30, zorder=5)

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC — {model}\n(n={n}, u={u}, negatives z={z}, cb={cb_tag})")
    if s != "loglog":
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    model_clean = sanitize_filename(model)
    scale_suffix = f"_{s}" if s != "linear" else ""
    out_path = os.path.join(save_dir, f"roc_{model_clean}_n{n}_u{u}_z{z}_cb-{cb_tag}{scale_suffix}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_combined(
    model_to_curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    fpr_target: float,
    save_dir: str,
    n: int,
    u: int,
    z: int,
    cb_tag: str,
    scale: str = "linear",
    x_eps: Optional[float] = None,
    fpr_min: Optional[float] = None,
    fpr_max: Optional[float] = None,
) -> str:
    """
    Produce and save a combined ROC plot for all models.
    model_to_curves: model -> (thresholds, fpr, tpr, tpr_at_target)
    Returns saved path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    s = (scale or "linear").lower()

    # Determine plotting parameters across all curves for good defaults
    all_fpr = np.concatenate([v[1] for v in model_to_curves.values()]) if model_to_curves else np.array([])
    min_positive_fpr = float(np.min(all_fpr[all_fpr > 0])) if np.any(all_fpr > 0) else None
    auto_eps = max((min_positive_fpr / 10.0 if min_positive_fpr is not None else 1.0 / (10.0 * max(z, 1))), 1e-12)
    eps = x_eps if x_eps is not None else auto_eps

    # Set axis scales and baseline domain
    if s == "linear":
        ax.set_xlim(0.0, 1.0)
        baseline_x = np.linspace(0.0, 1.0, 1001)
    elif s in ("logx", "loglog"):
        ax.set_xscale("log")
        x_min = fpr_min if fpr_min is not None else max(min_positive_fpr if min_positive_fpr is not None else eps, eps)
        x_max = fpr_max if fpr_max is not None else 1.0
        ax.set_xlim(x_min, x_max)
        baseline_x = np.geomspace(x_min, 1.0, 1001)
        if s == "loglog":
            ax.set_yscale("log")
    elif s == "symlogx":
        linthresh = eps
        ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
        x_min = fpr_min if fpr_min is not None else 0.0
        x_max = fpr_max if fpr_max is not None else 1.0
        ax.set_xlim(x_min, x_max)
        baseline_x = np.geomspace(max(linthresh, 1e-12), 1.0, 1001)
    else:
        baseline_x = np.linspace(0.0, 1.0, 1001)

    # Random baseline
    ax.plot(baseline_x, baseline_x, linestyle="--", color="gray", linewidth=1, label="Random baseline")

    for model, (thresholds, fpr, tpr, tpr_at_target) in model_to_curves.items():
        if s == "linear":
            fpr_plot, tpr_plot = fpr, tpr
            target_x = fpr_target
            target_y = tpr_at_target
        elif s in ("logx", "loglog"):
            fpr_plot = np.where(fpr > 0, fpr, eps)
            if s == "loglog":
                min_positive_tpr = float(np.min(tpr[tpr > 0])) if np.any(tpr > 0) else None
                y_eps = max((min_positive_tpr / 10.0 if min_positive_tpr is not None else 1e-4), 1e-12)
                tpr_plot = np.where(tpr > 0, tpr, y_eps)
                target_y = max(tpr_at_target, y_eps)
            else:
                tpr_plot = tpr
                target_y = tpr_at_target
            target_x = max(fpr_target, eps)
        elif s == "symlogx":
            fpr_plot = fpr
            tpr_plot = tpr
            target_x = fpr_target
            target_y = tpr_at_target
        else:
            fpr_plot, tpr_plot = fpr, tpr
            target_x, target_y = fpr_target, tpr_at_target

        label = f"{model} (TPR@{int(fpr_target*100)}%FPR={tpr_at_target*100:.1f}%, cb={cb_tag})"
        ax.plot(fpr_plot, tpr_plot, linewidth=2, label=label)
        ax.scatter([target_x], [target_y], s=25)

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC (Combined)\n(n={n}, u={u}, negatives z={z}, cb={cb_tag})")
    if s != "loglog":
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    scale_suffix = f"_{s}" if s != "linear" else ""
    out_path = os.path.join(save_dir, f"roc_combined_n{n}_u{u}_z{z}_cb-{cb_tag}{scale_suffix}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_roc_csv(save_dir: str, model: str, thresholds: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, n: int, u: int, z: int, cb_tag: str) -> str:
    model_clean = sanitize_filename(model)
    out_csv = os.path.join(save_dir, f"roc_points_{model_clean}_n{n}_u{u}_z{z}_cb-{cb_tag}.csv")
    df = pd.DataFrame({
        "threshold": thresholds.astype(int),
        "FPR": fpr.astype(float),
        "TPR": tpr.astype(float),
    })
    df.to_csv(out_csv, index=False)
    return out_csv


def generate_power_of_ten_ticks(fpr_min: float, fpr_max: float):
    """
    Generate power-of-ten ticks 10^p between [fpr_min, fpr_max] inclusive, where possible.
    Returns (ticks, labels, exponents).
    """
    fpr_min = float(fpr_min)
    fpr_max = float(fpr_max)
    if fpr_min <= 0 or fpr_max <= 0:
        raise ValueError("fpr_min and fpr_max must be > 0 for power-of-ten ticks")
    if fpr_min > fpr_max:
        fpr_min, fpr_max = fpr_max, fpr_min

    log_min = np.log10(fpr_min)
    log_max = np.log10(fpr_max)
    # Use tiny epsilon to handle exact powers cleanly
    p_start = int(np.ceil(log_min - 1e-12))
    p_end = int(np.floor(log_max + 1e-12))
    if p_start > p_end:
        # Degenerate window: include the nearest power to fpr_min
        p_start = int(np.round(log_min))
        p_end = p_start
    exponents = list(range(p_start, p_end + 1))
    ticks = [10.0 ** p for p in exponents]
    # Format labels using Unicode superscripts, e.g., 10⁻⁴ instead of 10^-4
    sup = {"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}
    def to_superscript(num: int) -> str:
        s = str(int(num))
        return "".join(sup.get(ch, ch) for ch in s)
    labels = [f"10{to_superscript(p)}" for p in exponents]
    return ticks, labels, exponents


def thresholds_for_fpr_ticks(fpr_array: np.ndarray, thresholds_array: np.ndarray, ticks: List[float]) -> List[int]:
    """
    Map each target FPR tick to the smallest threshold such that FPR(threshold) >= tick.
    Clamps to endpoints if tick is out of range.
    """
    fpr_arr = fpr_array.astype(float)
    idxs = np.searchsorted(fpr_arr, np.array(ticks, dtype=float), side="left")
    idxs = np.clip(idxs, 0, len(thresholds_array) - 1)
    return thresholds_array[idxs].astype(int).tolist()


def save_table_image(save_dir: str, n: int, u: int, z: int,
                     model_names: List[str], tick_labels: List[str],
                     hd_values: List[int], tpr_rows: List[List[float]], cb_tag: str) -> str:
    """
    Render a table as a PNG:
      - Header row: Model + FPR ticks (10^p labels)
      - Second row: HD + thresholds for each tick
      - One row per model with TPR at each tick as percentages
    """
    # Prepare cell text
    header = ["FPR"] + tick_labels
    hd_row = ["HD Threshold"] + [str(int(v)) for v in hd_values]

    # Format TPR rows as percentages with one decimal
    tpr_str_rows = [[f"{(val * 100.0):.1f}%" if np.isfinite(val) else "NA" for val in row] for row in tpr_rows]

    cell_text = [header, hd_row]
    for model, vals in zip(model_names, tpr_str_rows):
        cell_text.append([model] + vals)

    num_cols = len(header)
    num_rows = len(cell_text)

    # Figure sizing heuristics
    fig_w = max(6.0, 1.1 * num_cols + 1.0)  # widen with more columns
    fig_h = max(3.0, 0.55 * num_rows + 0.5)  # height scales with rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Column widths: wider first column
    if num_cols > 1:
        other_w = 0.65 / (num_cols - 1)
        col_widths = [0.35] + [other_w] * (num_cols - 1)
    else:
        col_widths = [1.0]

    table = ax.table(cellText=cell_text, cellLoc="center", colWidths=col_widths, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    # Style header (no shading)
    for j in range(num_cols):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold")
    # Style HD label
    table[(1, 0)].set_text_props(weight="bold")

    ax.set_title(f"TPR at power-of-ten FPR ticks with HD thresholds\n(n={n}, u={u}, negatives z={z}, cb={cb_tag})", fontsize=10, pad=10)
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"roc_table_n{n}_u{u}_z{z}_cb-{cb_tag}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Construct ROC curves (TPR/FPR) for models and report TPR@1%%FPR")
    parser.add_argument("--n", type=int, default=16, help="Size of the code table (2^n entries)")
    parser.add_argument("--u", type=int, default=100, help="Size of the codes (bit length)")
    parser.add_argument("--z", type=int, default=10000, help="Number of negative random sequences to test")
    parser.add_argument("--tmin", type=int, default=None, help="Minimum Hamming threshold (inclusive). Defaults to 0.")
    parser.add_argument("--tmax", type=int, default=None, help="Maximum Hamming threshold (inclusive). Defaults to u.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Secret key used for codebook generation (string)")
    parser.add_argument("--codebook", type=str, choices=["rng", "bch"], default="rng", help="Codebook type for negatives: 'rng' or 'bch'")
    parser.add_argument("--distance", type=str, choices=["hamming", "levenshtein"], default="hamming", help="Distance metric for negatives search: 'hamming' or 'levenshtein'")
    parser.add_argument("--bch-t", type=int, default=None, help="BCH designed error-correction parameter t (if None, auto-choose the largest t with k >= n)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (n0=2^m-1). If None, minimal m with n0 >= u is chosen")
    parser.add_argument("--fpr-target", type=float, default=0.01, help="Target FPR for reporting TPR (default=0.01)")
    parser.add_argument("--output-root", type=str, default="output", help="Root directory containing model run outputs")
    parser.add_argument("--save-dir", type=str, default=os.path.join("scripts", "output"), help="Directory to save plots/CSVs")
    parser.add_argument("--include-models", type=str, nargs="+", help="Optional substrings to filter canonical model names")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (useful in headless runs)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output and progress information")
    # Plot scaling and axis controls
    parser.add_argument("--scale", type=str, choices=["linear", "logx", "symlogx", "loglog"], default="linear",
                        help="Axis scaling for plots: linear (default), logx (log FPR), symlogx (log FPR with linear region near 0), loglog (log FPR and TPR)")
    parser.add_argument("--x-eps", dest="x_eps", type=float, default=None,
                        help="Small epsilon used when plotting FPR=0 on log axes. Default auto-computed.")
    parser.add_argument("--fpr-min", dest="fpr_min", type=float, default=1e-4,
                        help="Minimum FPR for x-axis limits (default 1e-4; useful with --scale logx/symlogx).")
    parser.add_argument("--fpr-max", dest="fpr_max", type=float, default=1.0,
                        help="Maximum FPR for x-axis limits (default 1.0).")

    args = parser.parse_args()

    # Paths setup
    paths.ensure_directories()
    ensure_dirs(args.save_dir)

    # Create timestamped run subfolder under save-dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Build codebook tag for filenames/plots (codeword generator + distance metric)
    if args.codebook == "rng":
        _base_tag = "rng"
    else:
        _t_tag = "auto" if args.bch_t is None else str(args.bch_t)
        _m_tag = "auto" if args.bch_m is None else str(args.bch_m)
        _base_tag = f"bch_m{_m_tag}_t{_t_tag}"
    cb_tag = f"{_base_tag}-{args.distance}"

    run_dir_name = f"tpr_fpr_roc_n{args.n}_u{args.u}_z{args.z}_cb-{cb_tag}_{timestamp}"
    run_save_dir = os.path.join(args.save_dir, run_dir_name)
    ensure_dirs(run_save_dir)
    if args.verbose:
        print(f"Saving outputs to: {run_save_dir}")

    # Initialize RNG for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.verbose:
        print(f"Secret key: {args.secret_key}")
        print(f"Generating negatives: z={args.z}, u={args.u}, codebook={args.codebook}, distance={args.distance}")
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
        n=args.n, u=args.u, z=args.z, seed=args.seed, secret_key=args.secret_key,
        codebook=args.codebook, distance=args.distance, bch_t=args.bch_t, bch_m=args.bch_m, verbose=args.verbose
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
        csv_path = save_roc_csv(run_save_dir, model, thresholds, fpr, tpr, args.n, args.u, args.z, cb_tag)
        png_path = plot_per_model(
            model, thresholds, fpr, tpr, tpr_at_target, args.fpr_target,
            run_save_dir, args.n, args.u, args.z, cb_tag,
            scale=args.scale, x_eps=args.x_eps, fpr_min=args.fpr_min, fpr_max=args.fpr_max
        )

        print(f"[{model}] TPR@{int(args.fpr_target*100)}%FPR = {tpr_at_target*100:.2f}% | Saved: {os.path.basename(png_path)}, {os.path.basename(csv_path)}")

    # 5) Combined plot
    combined_path = plot_combined(
        model_to_curves, args.fpr_target, run_save_dir, args.n, args.u, args.z, cb_tag,
        scale=args.scale, x_eps=args.x_eps, fpr_min=args.fpr_min, fpr_max=args.fpr_max
    )
    print(f"Combined ROC saved to: {combined_path}")

    # Table image at power-of-ten FPR ticks with HD row
    try:
        ticks, tick_labels, _ = generate_power_of_ten_ticks(args.fpr_min, args.fpr_max)
        # Reference arrays from any model (FPR is based on negatives, same across models)
        ref_model = next(iter(model_to_curves))
        ref_thresholds, ref_fpr, _, _ = model_to_curves[ref_model]
        hd_values = thresholds_for_fpr_ticks(ref_fpr, ref_thresholds, ticks)

        # Per-model TPR at ticks
        model_names = sorted(model_to_curves.keys())
        tpr_rows = []
        for m in model_names:
            _, fpr_arr, tpr_arr, _ = model_to_curves[m]
            vals = [interpolate_tpr_at_fpr(fpr_arr, tpr_arr, t)[0] for t in ticks]
            tpr_rows.append(vals)

        table_path = save_table_image(run_save_dir, args.n, args.u, args.z, model_names, tick_labels, hd_values, tpr_rows, cb_tag)
        print(f"ROC table saved to: {table_path}")
    except Exception as e:
        print(f"Warning: failed to generate table image: {e}")

    # Optional: show plots interactively (if not in headless environment)
    if not args.no_show:
        # No live figures to show because we saved and closed them; this is intentional to avoid GUI dependence.
        pass


if __name__ == "__main__":
    main()
