#!/usr/bin/env python3
"""
Compare watermark robustness against Qu et al., Yoo et al., and Cohen et al.

Generates a grouped bar chart (similar to Figure 6 in Qu et al.) comparing
match rates across attack types.

Usage:
  python scripts/compare_robustness.py --input-dir output/word_attack_...
  python scripts/compare_robustness.py --input-dir output/word_attack_... --show
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------
# Baseline values estimated from Figure 6 in Qu et al. (b=20, T=200, 10%)
# -----------------------------------------------------------------------

ATTACKS = ["No Attack", "Word Delete", "Word Add", "Synonym Replace", "Paraphrase"]

# Map from our attack names in results.txt to display names
ATTACK_NAME_MAP = {
    "No Attack": "No Attack",
    "deletion": "Word Delete",
    "insertion": "Word Add",
    "synonym": "Synonym Replace",
    "paraphrase": "Paraphrase",
}

BASELINES = {
    "Qu et al.": {
        "No Attack": 98,
        "Word Delete": 88,
        "Word Add": 92,
        "Synonym Replace": 88,
        "Paraphrase": 80,
    },
    "Yoo et al.": {
        "No Attack": 50,
        "Word Delete": 28,
        "Word Add": 35,
        "Synonym Replace": 30,
        "Paraphrase": 15,
    },
    "Cohen et al.": {
        "No Attack": 94,
        "Word Delete": 67,
        "Word Add": 78,
        "Synonym Replace": 67,
        "Paraphrase": 25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare watermark robustness with published baselines"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to word attack output directory (must contain results.txt)",
    )
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path (default: comparison.png in input-dir)",
    )
    return parser.parse_args()


def parse_results_txt(results_path: str) -> dict[str, float]:
    """Parse match rates from results.txt."""
    rates: dict[str, float] = {}

    with open(results_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Match "No Attack   7109   0   99.8%   17.6"
        m = re.match(r"^(No Attack|deletion|insertion|synonym|paraphrase)\s+\d+\s+\d+\s+([\d.]+)%", line)
        if m:
            attack_name = m.group(1)
            match_rate = float(m.group(2))
            display_name = ATTACK_NAME_MAP.get(attack_name, attack_name)
            rates[display_name] = match_rate

    return rates


def create_comparison_chart(
    ours: dict[str, float],
    output_path: str,
    show: bool = False,
) -> None:
    """Create grouped bar chart comparing match rates."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch

    # Only include attacks that we have data for
    attacks = [a for a in ATTACKS if a in ours]

    n_attacks = len(attacks)
    n_methods = 1 + len(BASELINES)  # Ours + baselines
    method_names = ["Ours (Levenshtein)"] + list(BASELINES.keys())

    # Colors: modern flat palette matching the reference style
    colors = ["#4CAF50", "#2196F3", "#BDBDBD", "#9E9E9E"]

    x = np.arange(n_attacks)
    width = 0.18
    gap = 0.02

    fig, ax = plt.subplots(figsize=(14, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Collect all values per attack to find the best
    all_values: list[list[float]] = []
    for i, method in enumerate(method_names):
        if method not in BASELINES:
            values = [ours.get(a, 0) for a in attacks]
        else:
            values = [BASELINES[method].get(a, 0) for a in attacks]
        all_values.append(values)

    best_per_attack = [max(all_values[m][a] for m in range(n_methods)) for a in range(n_attacks)]

    for i, method in enumerate(method_names):
        values = all_values[i]

        offset = (i - n_methods / 2 + 0.5) * (width + gap)
        bars = ax.bar(
            x + offset, values, width,
            color=colors[i],
            edgecolor="none",
            zorder=3,
        )

        # Value labels on top of bars (bold if best)
        for j, (bar, val) in enumerate(zip(bars, values)):
            label = f"{val:.1f}" if isinstance(val, float) and val != int(val) else f"{val:.0f}"
            is_best = val == best_per_attack[j]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                label,
                ha="center", va="bottom",
                fontsize=8,
                fontweight="bold" if is_best else "normal",
                color="#333333",
            )

    # Only keep left and bottom spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")

    # Y-axis scale
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor="#555555")
    ax.tick_params(axis="x", length=0)

    # X-axis labels as bold black text
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=10, color="black")

    # Legend at top
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[i], edgecolor="none")
        for i in range(n_methods)
    ]
    legend = ax.legend(
        legend_handles, method_names,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=n_methods,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=2.0,
    )

    plt.subplots_adjust(bottom=0.10, top=0.88)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Plot saved: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    results_path = os.path.join(args.input_dir, "results.txt")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        sys.exit(1)

    ours = parse_results_txt(results_path)
    if not ours:
        print("Error: Could not parse any match rates from results.txt")
        sys.exit(1)

    print(f"Parsed match rates from: {args.input_dir}")
    for attack, rate in ours.items():
        print(f"  {attack}: {rate:.1f}%")

    output_path = args.output or os.path.join(args.input_dir, "comparison.png")
    create_comparison_chart(ours, output_path, show=args.show)


if __name__ == "__main__":
    main()
