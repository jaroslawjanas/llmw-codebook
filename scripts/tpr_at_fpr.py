"""
TPR @ FPR Analysis Script.

Calculates True Positive Rate at a fixed False Positive Rate (default 1%)
for all models with matching (n, u) parameters.

Algorithm:
1. Generate z random binary sequences of length u
2. BCH decode each to get bit_errors (distance to nearest valid codeword)
3. Find threshold where approximately fpr_target% of random sequences are "false positives"
4. For each model, calculate TPR = (samples with bit_errors <= threshold) / total
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils import discover_experiments, load_by_model
from src.bch_codec import BCHCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate TPR at a fixed FPR threshold"
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="Watermark bits (must match experiment n)"
    )
    parser.add_argument(
        "--u", type=int, required=True,
        help="Codeword length (must match experiment u)"
    )
    parser.add_argument(
        "--z", type=int, default=10000,
        help="Number of random samples for FPR threshold estimation (default: 10000, min: 1000)"
    )
    parser.add_argument(
        "--fpr-target", type=float, default=0.01,
        help="Target FPR (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    if args.z < 1000:
        parser.error(f"--z must be at least 1000, got {args.z}")

    if not (0 < args.fpr_target < 1):
        parser.error(f"--fpr-target must be between 0 and 1 (exclusive), got {args.fpr_target}")

    return args


def shorten_model_name(name: str) -> str:
    """Shorten model name by removing org prefix."""
    if "/" in name:
        return name.split("/")[-1]
    return name


def generate_negative_bit_errors(
    bch_codec: BCHCodec,
    z: int,
    seed: int,
) -> np.ndarray:
    """
    Generate bit_errors for z random binary sequences.

    Each random sequence is BCH decoded to find its distance to the
    nearest valid codeword.

    Args:
        bch_codec: BCH codec instance
        z: Number of random samples
        seed: Random seed

    Returns:
        Array of bit_errors values for each random sequence
    """
    rng = np.random.default_rng(seed)
    bit_errors_list = []

    for _ in tqdm(range(z), desc="Generating negatives"):
        random_codeword = rng.integers(0, 2, size=bch_codec.u, dtype=np.uint8)
        _, bit_errors = bch_codec.decode(random_codeword)
        bit_errors_list.append(bit_errors)

    return np.array(bit_errors_list)


def find_fpr_threshold(
    negative_bit_errors: np.ndarray,
    fpr_target: float
) -> tuple[int, float]:
    """
    Find bit_error threshold that achieves target FPR.

    FPR = proportion of negative samples with bit_errors <= threshold.

    Args:
        negative_bit_errors: Array of bit_errors for negative samples
        fpr_target: Target FPR (e.g., 0.01 for 1%)

    Returns:
        Tuple of (threshold, actual_fpr):
        - threshold: bit_error value at which FPR is achieved
        - actual_fpr: actual FPR at this threshold
    """
    sorted_errors = np.sort(negative_bit_errors)
    z = len(sorted_errors)

    # Find index corresponding to target FPR percentile
    # FPR = count(bit_errors <= threshold) / z
    # We want threshold such that fpr_target * z samples are below it
    fpr_index = int(fpr_target * z) - 1
    fpr_index = max(0, fpr_index)

    threshold = int(sorted_errors[fpr_index])

    # Calculate actual FPR at this threshold
    actual_fpr = np.mean(negative_bit_errors <= threshold)

    return threshold, actual_fpr


def calculate_tpr_for_model(
    model: str,
    threshold: int,
) -> tuple[dict, np.ndarray]:
    """
    Calculate TPR for a specific model at the given threshold.

    Args:
        model: Model name
        threshold: bit_error threshold (samples <= threshold are positives)

    Returns:
        Tuple of (stats_dict, bit_errors_array):
        - stats_dict: model, total, tp, tpr, match_rate
        - bit_errors_array: bit_errors for successful watermarks (for histogram)
    """
    df = load_by_model(model)

    total_samples = len(df)

    # Apply threshold only to successfully decoded watermark rows
    successful = df[df["successful_watermark"]]
    tp = int((successful["bit_errors"] <= threshold).sum())
    tpr = tp / total_samples if total_samples > 0 else 0.0

    # Match rate: successful decodes / total samples
    n_successful = int(df["successful_watermark"].sum())
    match_rate = n_successful / total_samples if total_samples > 0 else 0.0

    stats = {
        "model": model,
        "total": total_samples,
        "tp": tp,
        "tpr": tpr,
        "match_rate": match_rate,
    }

    bit_errors = successful["bit_errors"].values

    return stats, bit_errors


def create_results_txt(
    args: argparse.Namespace,
    threshold: int,
    actual_fpr: float,
    negative_bit_errors: np.ndarray,
    results: list[dict],
    output_path: Path
) -> None:
    """Write detailed results to text file."""
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"TPR @ FPR = {args.fpr_target * 100:.2f}% Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("Parameters:\n")
        f.write(f"  n (watermark bits): {args.n}\n")
        f.write(f"  u (codeword length): {args.u}\n")
        f.write(f"  z (negative samples): {args.z}\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  target FPR: {args.fpr_target * 100:.2f}%\n\n")

        f.write("FPR Threshold Calculation:\n")
        f.write(f"  bit_error threshold: {threshold}\n")
        negatives_below = int(np.sum(negative_bit_errors <= threshold))
        f.write(f"  actual FPR: {actual_fpr * 100:.2f}% ")
        f.write(f"({negatives_below}/{args.z} negatives below threshold)\n\n")

        f.write("Model Results:\n")
        f.write("-" * 60 + "\n")

        for r in results:
            f.write(f"Model: {r['model']}\n")
            f.write(f"  Total samples: {r['total']}\n")
            f.write(f"  True Positives (TP): {r['tp']}\n")
            f.write(f"  Match rate: {r['match_rate'] * 100:.2f}%\n")
            f.write(f"  TPR: {r['tpr'] * 100:.2f}%\n")
            f.write("-" * 60 + "\n")

        f.write("\nSummary Table:\n")
        max_model_len = max(len(r["model"]) for r in results)
        header = f"  {'Model':<{max_model_len}}  Match     TPR @ FPR={actual_fpr * 100:.2f}%"
        f.write(header + "\n")
        for r in results:
            f.write(f"  {r['model']:<{max_model_len}}  {r['match_rate'] * 100:5.2f}%    {r['tpr'] * 100:.2f}%\n")


def create_table_png(
    results: list[dict],
    actual_fpr: float,
    args: argparse.Namespace,
    output_path: Path
) -> None:
    """Create summary table as PNG image."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping table.png")
        return

    cell_data = []
    for r in results:
        cell_data.append([
            shorten_model_name(r["model"]),
            f"{r['total']}",
            f"{r['tp']}",
            f"{r['match_rate'] * 100:.1f}%",
            f"{r['tpr'] * 100:.1f}%"
        ])

    col_labels = [
        "Model",
        "Total",
        "True Positives",
        "Match Rate",
        f"TPR@{actual_fpr * 100:.2f}%",
    ]

    fig, ax = plt.subplots(figsize=(9, max(2, len(results) * 0.5 + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Set column widths: wider for model name, narrower for numbers
    col_widths = [0.32, 0.12, 0.16, 0.15, 0.18]
    for i, width in enumerate(col_widths):
        for row in range(len(cell_data) + 1):
            table[(row, i)].set_width(width)

    # Style header row
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", weight="bold")

    # Alternate row colors
    for i in range(1, len(cell_data) + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    plt.title(
        f"TPR @ FPR={actual_fpr * 100:.2f}% (n={args.n}, u={args.u}, z={args.z})",
        fontsize=11,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_histogram_png(
    negative_bit_errors: np.ndarray,
    model_bit_errors: dict[str, np.ndarray],
    threshold: int,
    actual_fpr: float,
    args: argparse.Namespace,
    output_path: Path
) -> None:
    """Create per-model histogram subplots comparing each model's bit errors to negatives."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping histogram.png")
        return

    n_models = len(model_bit_errors)
    if n_models == 0:
        return

    # Determine grid layout (2 columns, variable rows)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 4 * n_rows),
        squeeze=False
    )

    # Determine shared bin edges across all data
    all_positives = np.concatenate(list(model_bit_errors.values()))
    max_errors = max(negative_bit_errors.max(), all_positives.max())
    bins = np.arange(0, max_errors + 2) - 0.5

    # Plot each model in its subplot
    for idx, (model, pos_errors) in enumerate(model_bit_errors.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Plot negative distribution
        ax.hist(
            negative_bit_errors,
            bins=bins,
            alpha=0.45,
            label=f"Random (n={args.z})",
            color="red",
            density=True
        )

        # Plot successfully recovered watermarks (true positives)
        n_tp = len(pos_errors)

        if n_tp > 0:
            ax.hist(
                pos_errors,
                bins=bins,
                alpha=0.6,
                label=f"TP (n={n_tp})",
                color="darkgreen",
                density=True
            )

        # Draw threshold line
        ax.axvline(
            threshold + 0.5,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold={threshold}"
        )

        ax.set_xlabel("Bit Errors")
        ax.set_ylabel("Density")
        ax.set_title(shorten_model_name(model))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.suptitle(
        f"Bit Error Distributions (n={args.n}, u={args.u}, FPR={actual_fpr * 100:.2f}%)",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    # Discover experiments with matching n and u
    experiments = discover_experiments()

    if experiments.empty:
        print("Error: No experiments found in output directory")
        sys.exit(1)

    matching = experiments[(experiments["n"] == args.n) & (experiments["u"] == args.u)]

    if matching.empty:
        available = experiments[["n", "u"]].drop_duplicates().values.tolist()
        print(f"Error: No experiments found with n={args.n}, u={args.u}")
        print(f"Available (n, u) combinations: {available}")
        sys.exit(1)

    models = sorted(matching["model"].unique())

    print(f"Found {len(models)} models with n={args.n}, u={args.u}:")
    for m in models:
        print(f"  - {m}")
    print()

    # Create BCH codec
    bch_codec = BCHCodec(n=args.n, u=args.u, verbose=True)

    # Phase 1: Generate negative bit errors
    print(f"Generating {args.z} random samples for FPR threshold...")

    negative_bit_errors = generate_negative_bit_errors(
        bch_codec=bch_codec,
        z=args.z,
        seed=args.seed,
    )

    # Phase 2: Find FPR threshold
    threshold, actual_fpr = find_fpr_threshold(negative_bit_errors, args.fpr_target)

    print(f"\nFPR threshold: bit_errors <= {threshold}")
    print(f"Actual FPR: {actual_fpr * 100:.2f}%\n")

    # Phase 3: Calculate TPR per model
    results = []
    model_bit_errors = {}  # Per-model bit errors for histogram

    for model in tqdm(models, desc="Calculating TPR"):
        stats, bit_errors = calculate_tpr_for_model(model, threshold)
        results.append(stats)
        model_bit_errors[model] = bit_errors

    # Sort results by TPR descending
    results.sort(key=lambda x: x["tpr"], reverse=True)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "output" / (
        f"tpr_at_fpr_n{args.n}_u{args.u}_z{args.z}_fpr{args.fpr_target}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 4: Generate outputs
    create_results_txt(
        args=args,
        threshold=threshold,
        actual_fpr=actual_fpr,
        negative_bit_errors=negative_bit_errors,
        results=results,
        output_path=output_dir / "results.txt"
    )

    create_table_png(
        results=results,
        actual_fpr=actual_fpr,
        args=args,
        output_path=output_dir / "table.png"
    )

    create_histogram_png(
        negative_bit_errors=negative_bit_errors,
        model_bit_errors=model_bit_errors,
        threshold=threshold,
        actual_fpr=actual_fpr,
        args=args,
        output_path=output_dir / "histogram.png"
    )

    # Print summary
    print(f"\nResults saved to: {output_dir}")
    print(f"\nThreshold: bit_errors <= {threshold} (FPR = {actual_fpr * 100:.2f}%)")
    print("\nSummary:")
    print("-" * 50)
    for r in results:
        print(f"  {shorten_model_name(r['model']):<35} TPR = {r['tpr'] * 100:.2f}%")


if __name__ == "__main__":
    main()
