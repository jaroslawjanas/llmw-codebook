"""
Combine multiple AUROC experiment results into a single comparison plot.

Reads roc_curve.csv from each --input-dir and overlays all ROC curves
on one axes with adaptive zoom.

Example:
  python scripts/combine_auroc.py \\
    --input-dir output/auroc_bch_* --label BCH \\
    --input-dir output/auroc_*_hamming_* --label Hamming \\
    --input-dir output/auroc_*_levenshtein* --label Levenshtein \\
    --output combined_roc.png
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required")
    sys.exit(1)


def discover_dirs(pattern: str) -> list[Path]:
    """Expand a glob pattern to sorted directory paths."""
    matches = sorted(glob.glob(pattern))
    dirs = [Path(m) for m in matches if Path(m).is_dir()]
    if not dirs:
        print(f"Warning: no directories matched pattern: {pattern}")
    return dirs


def load_roc_csv(input_dir: Path) -> pd.DataFrame | None:
    """Load roc_curve.csv from an AUROC output directory."""
    csv_path = input_dir / "roc_curve.csv"
    if not csv_path.exists():
        print(f"Warning: roc_curve.csv not found in {input_dir}")
        return None
    return pd.read_csv(csv_path)


def extract_auroc_from_results(input_dir: Path) -> float | None:
    """Parse AUROC value from results.txt if present."""
    txt_path = input_dir / "results.txt"
    if not txt_path.exists():
        return None
    try:
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("AUROC:"):
                return float(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def compute_auroc_from_csv(df: pd.DataFrame) -> float:
    """Compute AUROC from (fpr, tpr) pairs using trapezoid rule."""
    fpr = df["fpr"].to_numpy()
    tpr = df["tpr"].to_numpy()
    return float(np.trapezoid(tpr, fpr))


def find_1pct_point(df: pd.DataFrame) -> tuple[float, float] | None:
    """Find (fpr, tpr) closest to 1% FPR."""
    fpr = df["fpr"].to_numpy()
    tpr = df["tpr"].to_numpy()
    idx = np.argmin(np.abs(fpr - 0.01))
    return float(fpr[idx]), float(tpr[idx])


def shorten_label(name: str) -> str:
    """Extract short model name from HuggingFace path."""
    return name.split("/")[-1] if "/" in name else name


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine AUROC curves into one plot")
    parser.add_argument(
        "--input-dir", type=str, action="append", required=True,
        help="Glob pattern matching AUROC output directories (can repeat)"
    )
    parser.add_argument(
        "--label", type=str, action="append", required=True,
        help="Label for each --input-dir group (must match count)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/combined_auroc.png"),
        help="Output image path (default: output/combined_auroc.png)"
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Plot title (default: auto-generated)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plot interactively"
    )
    args = parser.parse_args()

    if len(args.input_dir) != len(args.label):
        parser.error("Number of --label must match number of --input-dir")

    # Resolve directories
    all_dirs: list[list[Path]] = []
    for pattern in args.input_dir:
        dirs = discover_dirs(pattern)
        all_dirs.append(dirs)

    # Color cycle
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(13, 12))

    global_min_tpr = 1.0
    plotted_any = False

    for group_idx, (dirs, label) in enumerate(zip(all_dirs, args.label)):
        color = colors[group_idx % len(colors)]
        for d in dirs:
            df = load_roc_csv(d)
            if df is None or df.empty:
                continue

            auroc = extract_auroc_from_results(d)
            if auroc is None:
                auroc = compute_auroc_from_csv(df)

            fpr = df["fpr"].to_numpy()
            tpr = df["tpr"].to_numpy()

            # Update global min for zoom
            meaningful = tpr[fpr > 0]
            if len(meaningful) > 0:
                global_min_tpr = min(global_min_tpr, meaningful.min())

            # Strip any (u=...) suffix from the group label for clean legend
            clean_label = label.split(" (")[0]
            curve_label = f"{clean_label} (AUROC = {auroc:.4f})"
            ax.plot(
                fpr, tpr,
                color=color,
                linewidth=3,
                label=curve_label,
            )

            # 1% FPR marker
            pt = find_1pct_point(df)
            if pt:
                fpr_1, tpr_1 = pt
                ax.plot(
                    fpr_1, tpr_1,
                    marker="o",
                    markersize=8,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    zorder=5,
                    linestyle="None",
                )

            plotted_any = True

    if not plotted_any:
        print("Error: no valid ROC curves found")
        sys.exit(1)

    # No diagonal reference — all curves are near-perfect, random baseline is off-scale

    # Fixed y-axis zoom from 0.9992 to just above 1.0
    y_min = 0.9992
    y_top = 1.00002
    ax.set_ylim(y_min, y_top)

    # Explicit ticks every 0.0002 from 0.9992 to 1.0
    step = 0.0002
    tick_vals = np.arange(y_min, 1.0 + step, step)
    tick_vals = tick_vals[tick_vals <= 1.0]
    if 1.0 not in tick_vals:
        tick_vals = np.append(tick_vals, 1.0)
    ax.set_yticks(np.sort(tick_vals))

    ax.set_xlim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=22)
    ax.set_ylabel("True Positive Rate", fontsize=22)

    # No title — clean plot
    if args.title:
        ax.set_title(args.title, fontsize=13, fontweight="bold")

    # Reorder legend: Levenshtein, Hamming, BCH
    handles, labels = ax.get_legend_handles_labels()
    desired_order = ["Levenshtein", "Hamming", "BCH"]
    ordered = []
    for name in desired_order:
        for h, lab in zip(handles, labels):
            if lab.startswith(name) and (h, lab) not in ordered:
                ordered.append((h, lab))
                break
    # Append any unmatched entries (fallback)
    for h, lab in zip(handles, labels):
        if (h, lab) not in ordered:
            ordered.append((h, lab))
    if ordered:
        ax.legend([h for h, _ in ordered], [lab for _, lab in ordered], loc="lower right", fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved combined plot to: {args.output}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
