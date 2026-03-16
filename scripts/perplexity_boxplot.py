"""
Perplexity Box Plot.

Draws vertical box plots comparing perplexity distributions across
human, no-watermark, and watermarked text conditions.

Reads CSV files produced by scripts/perplexity.py.

Usage:
  python scripts/perplexity_boxplot.py \\
      --input output/perplexity_openai_gpt-oss-20b_20260316_182208/

  python scripts/perplexity_boxplot.py \\
      --input output/perplexity_openai_gpt-oss-20b_20260316_182208/ \\
      --max-ppl 200 --no-outliers
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = ["human", "no_watermark", "watermarked"]
CONDITION_LABELS = {
    "human": "Human",
    "no_watermark": "No Watermark",
    "watermarked": "Watermarked",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Perplexity box plot")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to perplexity output directory (containing human.csv, etc.)",
    )
    parser.add_argument(
        "--max-ppl", type=float, default=None,
        help="Cap y-axis at this value (default: auto)",
    )
    parser.add_argument(
        "--show-outliers", action="store_true",
        help="Show outlier points (hidden by default)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save figure to this path (default: show interactively)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)

    # Load available conditions
    data = {}
    for condition in CONDITION_ORDER:
        csv_path = input_dir / f"{condition}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Filter out skipped texts
            df = df[df["skipped"] == False]
            data[condition] = df["perplexity"].values

    if not data:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    # Prepare data in order
    labels = []
    values = []
    for condition in CONDITION_ORDER:
        if condition in data:
            labels.append(CONDITION_LABELS[condition])
            values.append(data[condition])

    # Plot
    fig, ax = plt.subplots(figsize=(6, 7))

    bp = ax.boxplot(
        values,
        tick_labels=labels,
        showfliers=args.show_outliers,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
    )

    # Colors
    colors = ["#4DAF4A", "#377EB8", "#E41A1C"]  # green, blue, red
    for patch, color in zip(bp["boxes"], colors[:len(values)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add median annotations
    for i, v in enumerate(values):
        median = np.median(v)
        ax.annotate(
            f"{median:.1f}",
            xy=(i + 1, median),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
        )

    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Perplexity Distribution by Text Condition", fontsize=13)

    if args.max_ppl is not None:
        ax.set_ylim(top=args.max_ppl)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = args.output if args.output else str(input_dir / "boxplot.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
