"""
AUROC Analysis for Codebook-Based Watermarks (master branch).

Computes the full ROC curve and AUROC score for a single experiment
directory by comparing successful watermark distances against a large pool
of random negative samples.

Algorithm:
1. Load experiment config and generations.csv from --input-dir
2. Extract n, u, secret_key, codeword_generation, distance, bch_m from config
3. Build the appropriate codebook and generate z random negative samples
4. Compute ROC curve and AUROC
5. Output raw curve data as CSV plus summary text and plot
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.negative_generation import load_or_generate_negatives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate AUROC for codebook-based watermarks from an experiment directory"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Path to experiment directory containing config.json and generations.csv"
    )
    parser.add_argument(
        "--z", type=int, default=10000,
        help="Number of random negative samples for FPR estimation (default: 10000, min: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for negative generation (default: 42)"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0,
        help="Number of GPUs for Triton search (0 = all available, default: 0)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: auto-generated under output/)"
    )
    parser.add_argument(
        "--distance", type=str, default=None,
        choices=["hamming", "levenshtein", "levenshtein-triton"],
        help="Override distance metric from config (default: use config value)"
    )
    args = parser.parse_args()

    if args.z < 1000:
        parser.error(f"--z must be at least 1000, got {args.z}")

    if not args.input_dir.exists():
        parser.error(f"--input-dir does not exist: {args.input_dir}")

    return args


def load_experiment(input_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Load config.json and generations.csv from experiment directory."""
    config_path = input_dir / "config.json"
    generations_path = input_dir / "generations.csv"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {input_dir}")
    if not generations_path.exists():
        raise FileNotFoundError(f"generations.csv not found in {input_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    df = pd.read_csv(generations_path)
    return config, df


def extract_positive_distances(df: pd.DataFrame) -> np.ndarray:
    """
    Extract distances for successfully decoded watermarks.

    Positives = rows where successful_watermark == True.
    """
    successful = df[df["successful_watermark"]]
    return successful["distance"].values.astype(np.int64)


def compute_roc_curve(
    positive_distances: np.ndarray,
    negative_distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve from positive and negative distances.

    Thresholds are chosen from unique negative distances. For each threshold T:
    - FPR = count(negatives <= T) / total_negatives
    - TPR = count(positives <= T) / total_positives

    Args:
        positive_distances: distances for successful watermarks
        negative_distances: distances for random negative samples

    Returns:
        Tuple of (thresholds, fpr_array, tpr_array)
    """
    if len(positive_distances) == 0:
        raise ValueError("No successful watermarks found — cannot compute ROC curve")

    pos_sorted = np.sort(positive_distances)
    neg_sorted = np.sort(negative_distances)

    unique_thresholds, counts = np.unique(neg_sorted, return_counts=True)
    cumulative_neg_counts = np.cumsum(counts)

    n_pos = len(pos_sorted)
    n_neg = len(neg_sorted)

    pos_counts = np.searchsorted(pos_sorted, unique_thresholds, side="right")

    fpr = cumulative_neg_counts / n_neg
    tpr = pos_counts / n_pos

    # Prepend (0, 0) to start the curve from origin
    thresholds = np.concatenate([[unique_thresholds[0] - 1], unique_thresholds])
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])

    # Append (1, 1) if not already there
    if fpr[-1] < 1.0 or tpr[-1] < 1.0:
        max_threshold = max(unique_thresholds[-1], pos_sorted[-1]) + 1
        thresholds = np.append(thresholds, max_threshold)
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)

    return thresholds, fpr, tpr


def compute_auroc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute AUROC using the trapezoid rule."""
    return float(np.trapezoid(tpr, fpr))


def find_fpr_threshold(
    negative_distances: np.ndarray,
    fpr_target: float
) -> tuple[int, float]:
    """Find distance threshold that achieves target FPR."""
    sorted_dists = np.sort(negative_distances)
    z = len(sorted_dists)

    fpr_index = int(fpr_target * z) - 1
    fpr_index = max(0, fpr_index)

    threshold = int(sorted_dists[fpr_index])
    actual_fpr = np.mean(negative_distances <= threshold)

    return threshold, actual_fpr


def create_results_txt(
    args: argparse.Namespace,
    config: dict,
    input_dir: Path,
    n_positives: int,
    auroc: float,
    threshold: int,
    fpr_at_threshold: float,
    tpr_at_threshold: float,
    output_path: Path,
) -> None:
    """Write detailed results to text file."""
    model = config["model"]["name"]
    n = config["watermark"]["n"]
    u = config["watermark"]["u"]
    codeword_generation = config["watermark"]["codeword_generation"]
    distance = args.distance if args.distance else config["watermark"]["distance"]

    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AUROC Analysis for Codebook Watermarks\n")
        f.write("=" * 60 + "\n\n")

        f.write("Experiment:\n")
        f.write(f"  Input directory: {input_dir}\n")
        f.write(f"  Model: {model}\n")
        f.write(f"  Timestamp: {config['experiment']['timestamp']}\n\n")

        f.write("Parameters:\n")
        f.write(f"  n (watermark bits): {n}\n")
        f.write(f"  u (codeword length): {u}\n")
        f.write(f"  z (negative samples): {args.z}\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  codeword generation: {codeword_generation}\n")
        f.write(f"  distance metric: {distance}\n")
        if args.distance:
            f.write(f"  (Overridden from config value: {config['watermark']['distance']})\n")
        f.write("\n")

        f.write("Results:\n")
        f.write(f"  Total positive samples (successful watermarks): {n_positives}\n")
        f.write(f"  AUROC: {auroc:.6f}\n\n")

        f.write("Operating Point at 1% FPR (for reference):\n")
        f.write(f"  Threshold: distance <= {threshold}\n")
        f.write(f"  FPR: {fpr_at_threshold * 100:.4f}%\n")
        f.write(f"  TPR: {tpr_at_threshold * 100:.4f}%\n\n")

        f.write("Note: See roc_curve.csv for full (threshold, fpr, tpr) data.\n")


def create_roc_csv(
    thresholds: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    output_path: Path,
) -> None:
    """Write raw ROC curve data to CSV."""
    df = pd.DataFrame({
        "threshold": thresholds,
        "fpr": fpr,
        "tpr": tpr,
    })
    df.to_csv(output_path, index=False)


def create_roc_png(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    model_name: str,
    output_path: Path,
    fpr_1pct: float | None = None,
    tpr_1pct: float | None = None,
) -> None:
    """Create ROC curve plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping roc_curve.png")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        color="darkblue",
        linewidth=2,
        label=f"ROC curve (AUROC = {auroc:.4f})"
    )

    # Mark 1% FPR operating point
    if fpr_1pct is not None and tpr_1pct is not None:
        ax.plot(
            fpr_1pct, tpr_1pct,
            marker="o",
            markersize=8,
            color="darkblue",
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
            label=f"1% FPR (TPR = {tpr_1pct * 100:.1f}%)",
            linestyle="None"
        )

    # Plot diagonal reference (random classifier)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random (AUROC = 0.5)")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Zoom y-axis to where the curve actually varies
    meaningful_tpr = tpr[fpr > 0]
    if len(meaningful_tpr) > 0:
        min_tpr = meaningful_tpr.min()
        # Headroom above 1.0 = 5% of the visible y-range, so the curve
        # never touches the border but wasted space scales with the zoom.
        y_range = 1.0 - min_tpr
        ax.set_ylim(min_tpr - y_range * 0.2, 1.0 + y_range * 0.05)
        ticks = ax.get_yticks()
        ticks = ticks[ticks <= 1.0]
        if 1.0 not in ticks:
            ticks = np.append(ticks, 1.0)
        ax.set_yticks(np.sort(ticks))
    else:
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    # Load experiment
    print(f"Loading experiment from: {args.input_dir}")
    config, df = load_experiment(args.input_dir)

    n = config["watermark"]["n"]
    u = config["watermark"]["u"]
    secret_key = config["watermark"]["secret_key"]
    codeword_generation = config["watermark"]["codeword_generation"]
    distance = args.distance if args.distance else config["watermark"]["distance"]
    bch_m = config["watermark"].get("bch_m")
    model = config["model"]["name"]

    print(f"Model: {model}")
    print(f"n={n}, u={u}, generation={codeword_generation}, distance={distance}")
    if args.distance:
        print(f"(Overridden from config value: {config['watermark']['distance']})")

    # Extract positives
    positive_distances = extract_positive_distances(df)
    n_positives = len(positive_distances)
    print(f"Positive samples (successful watermarks): {n_positives}")

    if n_positives == 0:
        print("Error: No successful watermarks found in experiment")
        sys.exit(1)

    # Generate or load cached negatives
    print(f"\nGenerating {args.z} random negative samples...")
    negative_distances = load_or_generate_negatives(
        prefix="negative_distances",
        n=n,
        u=u,
        secret_key=secret_key,
        codeword_generation=codeword_generation,
        distance=distance,
        bch_m=bch_m,
        z=args.z,
        seed=args.seed,
        num_gpus=args.num_gpus,
    )

    # Compute ROC curve
    print("\nComputing ROC curve...")
    thresholds, fpr, tpr = compute_roc_curve(positive_distances, negative_distances)
    auroc = compute_auroc(fpr, tpr)

    print(f"AUROC: {auroc:.6f}")
    print(f"ROC points: {len(thresholds)}")

    # Reference operating point at 1% FPR
    ref_threshold, ref_fpr = find_fpr_threshold(negative_distances, 0.01)
    ref_tpr = float(np.mean(positive_distances <= ref_threshold))
    print(f"At 1% FPR threshold (distance <= {ref_threshold}): TPR = {ref_tpr * 100:.2f}%")

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "output" / (
            f"auroc_{model.replace('/', '_')}_n{n}_u{u}_{codeword_generation}_{distance}_{timestamp}"
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    create_roc_csv(thresholds, fpr, tpr, output_path=output_dir / "roc_curve.csv")
    create_results_txt(
        args=args,
        config=config,
        input_dir=args.input_dir,
        n_positives=n_positives,
        auroc=auroc,
        threshold=ref_threshold,
        fpr_at_threshold=ref_fpr,
        tpr_at_threshold=ref_tpr,
        output_path=output_dir / "results.txt"
    )
    create_roc_png(
        fpr=fpr,
        tpr=tpr,
        auroc=auroc,
        model_name=model,
        output_path=output_dir / "roc_curve.png",
        fpr_1pct=ref_fpr,
        tpr_1pct=ref_tpr,
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"  - roc_curve.csv  (raw curve data)")
    print(f"  - results.txt    (summary)")
    print(f"  - roc_curve.png  (plot)")


if __name__ == "__main__":
    main()
