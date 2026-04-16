"""
Diversity Evaluation Script.

Computes word-level lexical diversity metrics for up to three text conditions:
  1. Human-written text (from HuggingFace dataset or experiment directory)
  2. No-watermark generated text (from experiment directory or dataset)
  3. Watermarked generated text (from experiment directory or dataset)

Metrics computed per text:
  distinct_1             — unique unigrams / total unigrams
  distinct_2             — unique bigrams  / total bigrams  (nan if < 2 words)
  repeated_trigram_ratio — 1 − (unique trigrams / total trigrams)  (nan if < 3 words)
  n_words                — word count after re.findall(r'\\b\\w+\\b', text.lower())

No model or GPU required; pure CPU computation.

Usage examples:
  python scripts/diversity.py \\
      --human-text jaroslawjanas/open-generation-data default train "targets[0]" \\
      --no-watermark-text output/no_wm_experiment/ \\
      --watermarked-text output/wm_experiment/ \\
      --verbose

  python scripts/diversity.py \\
      --watermarked-text output/wm_experiment/ \\
      --min-words 50 --max-words 500 --n-texts 200
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.utils import load_texts, parse_text_source, source_description


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lexical diversity evaluation for human, no-watermark, and watermarked text",
    )

    parser.add_argument(
        "--human-text", type=str, nargs="+", default=None,
        help=(
            "Human text source. Either: a local directory path (experiment output "
            "dir with generations.csv), OR 4 args: '<dataset> <subset> <split> <column>' "
            "for a HuggingFace dataset. For list-type columns, append [<index>] to the "
            "column name (e.g., 'targets[0]')."
        ),
    )
    parser.add_argument(
        "--no-watermark-text", type=str, nargs="+", default=None,
        help="No-watermark text source. Same format as --human-text.",
    )
    parser.add_argument(
        "--watermarked-text", type=str, nargs="+", default=None,
        help="Watermarked text source. Same format as --human-text.",
    )

    parser.add_argument(
        "--max-words", type=int, default=None,
        help="Truncate texts to N words (word-level, after tokenization) for fair comparison",
    )
    parser.add_argument(
        "--min-words", type=int, default=None,
        help="Skip texts with fewer than N words (word-level, before truncation)",
    )
    parser.add_argument(
        "--n-texts", type=str, default="all",
        help="Number of texts to evaluate per condition, or 'all' (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-text output")
    parser.add_argument(
        "--save-dir", type=str, default="output",
        help="Base directory to save outputs (default: output)",
    )

    args = parser.parse_args()

    if not any([args.human_text, args.no_watermark_text, args.watermarked_text]):
        parser.error("At least one of --human-text, --no-watermark-text, --watermarked-text is required")

    return args


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_diversity_metrics(
    text: str,
    max_words: int | None = None,
    min_words: int | None = None,
) -> dict:
    """Compute word-level diversity metrics for a single text.

    Returns dict with keys: n_words, distinct_1, distinct_2,
    repeated_trigram_ratio, skipped.
    """
    nan = float("nan")
    words = re.findall(r"\b\w+\b", text.lower())
    n = len(words)

    if n == 0 or (min_words is not None and n < min_words):
        return {
            "n_words": n,
            "distinct_1": nan,
            "distinct_2": nan,
            "repeated_trigram_ratio": nan,
            "skipped": True,
        }

    if max_words is not None and n > max_words:
        words = words[:max_words]
        n = max_words

    distinct_1 = len(set(words)) / n

    if n >= 2:
        bigrams = [(words[i], words[i + 1]) for i in range(n - 1)]
        distinct_2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct_2 = nan

    if n >= 3:
        trigrams = [(words[i], words[i + 1], words[i + 2]) for i in range(n - 2)]
        repeated_trigram_ratio = 1.0 - len(set(trigrams)) / len(trigrams)
    else:
        repeated_trigram_ratio = nan

    return {
        "n_words": n,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repeated_trigram_ratio": repeated_trigram_ratio,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    fv = float(v)
    return f"{fv:>10.3f}" if not math.isnan(fv) else f"{'nan':>10}"


_CONDITION_ORDER = ["human", "no_watermark", "watermarked"]
_CONDITION_LABELS = {
    "human": "Human",
    "no_watermark": "No Watermark",
    "watermarked": "Watermarked",
}
_COLORS = ["#4DAF4A", "#377EB8", "#E41A1C"]


def _plot_results(condition_results: dict[str, list[dict]], run_dir: str) -> str | None:
    """Generate a 3-panel boxplot figure and save to run_dir/diversity.png."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return None

    metric_defs = [
        ("distinct_1", "Distinct-1\n(↑ more diverse)"),
        ("distinct_2", "Distinct-2\n(↑ more diverse)"),
        ("repeated_trigram_ratio", "Repeated Trigram Ratio\n(↓ less repetitive)"),
    ]

    present = [c for c in _CONDITION_ORDER if c in condition_results]
    labels = [_CONDITION_LABELS[c] for c in present]
    colors = [_COLORS[_CONDITION_ORDER.index(c)] for c in present]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (metric_key, metric_title) in zip(axes, metric_defs):
        values = []
        for condition in present:
            df = pd.DataFrame(condition_results[condition])
            valid_mask = ~df["skipped"] & df[metric_key].notna()
            vals = df.loc[valid_mask, metric_key].values.astype(float)
            values.append(vals[~np.isnan(vals)])

        bp = ax.boxplot(
            values,
            tick_labels=labels,
            showfliers=False,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, vals in enumerate(values):
            if len(vals) > 0:
                ax.annotate(
                    f"{np.median(vals):.3f}",
                    xy=(i + 1.3, float(np.median(vals))),
                    fontsize=8,
                    ha="left",
                    va="center",
                )

        ax.set_title(metric_title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Diversity Evaluation", fontsize=13)
    fig.tight_layout()

    output_path = os.path.join(run_dir, "diversity.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _output_results(
    condition_results: dict[str, list[dict]],
    source_descriptions: dict[str, str],
    args: argparse.Namespace,
) -> None:
    """Aggregate results, print summary tables, save CSVs."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, f"diversity_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    metrics = [
        ("distinct_1", "Distinct-1 (unique unigrams / total unigrams; higher = more diverse)"),
        ("distinct_2", "Distinct-2 (unique bigrams / total bigrams; higher = more diverse)"),
        ("repeated_trigram_ratio", "Repeated Trigram Ratio (1 − unique/total trigrams; lower = less repetitive)"),
    ]

    summary_records = []

    for condition, records in condition_results.items():
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(run_dir, f"{condition}.csv"), index=False)

        n_skipped = int(df["skipped"].sum())
        for metric_key, _ in metrics:
            valid_mask = ~df["skipped"] & df[metric_key].notna()
            vals = df.loc[valid_mask, metric_key].values.astype(float)
            summary_records.append({
                "condition": condition,
                "metric": metric_key,
                "count": len(df),
                "valid": int(valid_mask.sum()),
                "skipped": n_skipped,
                "mean": float(np.mean(vals)) if len(vals) > 0 else float("nan"),
                "median": float(np.median(vals)) if len(vals) > 0 else float("nan"),
                "std": float(np.std(vals)) if len(vals) > 0 else float("nan"),
                "min": float(np.min(vals)) if len(vals) > 0 else float("nan"),
                "max": float(np.max(vals)) if len(vals) > 0 else float("nan"),
            })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # Build text report
    lines = [
        "=" * 70,
        "Diversity Evaluation Results",
        "=" * 70,
        "",
        f"Max words: {args.max_words or 'none (full text)'}",
        f"Min words: {args.min_words or 'none (no filter)'}",
        "",
    ]
    for cond, desc in source_descriptions.items():
        lines.append(f"  {cond}: {desc}")
    lines.append("")

    for metric_key, metric_label in metrics:
        metric_rows = summary_df[summary_df["metric"] == metric_key]
        lines.append(f"--- {metric_label} ---")
        lines.append("")
        lines.append(
            f"{'Condition':<16} {'Valid':>6} {'Skipped':>8} {'Mean':>10} "
            f"{'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
        )
        lines.append("-" * 70)
        for _, row in metric_rows.iterrows():
            lines.append(
                f"{row['condition']:<16} {int(row['valid']):>6} {int(row['skipped']):>8} "
                f"{_fmt(row['mean'])} {_fmt(row['median'])} {_fmt(row['std'])} "
                f"{_fmt(row['min'])} {_fmt(row['max'])}"
            )
        lines.append("")

    lines.append("=" * 70)
    text_output = "\n".join(lines)
    print(text_output)

    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(text_output)

    plot_path = _plot_results(condition_results, run_dir)
    if plot_path:
        print(f"Plot saved to:    {plot_path}")

    print(f"Results saved to: {run_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    n_texts = None
    if args.n_texts.lower() != "all":
        n_texts = int(args.n_texts)

    # --- Load texts ---
    conditions: dict[str, list[str]] = {}
    source_descs: dict[str, str] = {}

    for attr_name, condition_name in [
        ("human_text", "human"),
        ("no_watermark_text", "no_watermark"),
        ("watermarked_text", "watermarked"),
    ]:
        flag_value = getattr(args, attr_name)
        if flag_value is None:
            continue

        source = parse_text_source(flag_value)
        texts = load_texts(source, seed=args.seed, n_texts=n_texts)

        if not texts:
            print(f"Warning: no texts loaded for --{attr_name.replace('_', '-')}")
            continue

        conditions[condition_name] = texts
        source_descs[condition_name] = source_description(source)
        print(f"Loaded {len(texts)} texts for '{condition_name}' from {source_descs[condition_name]}")

    if not conditions:
        print("Error: no texts loaded from any source")
        sys.exit(1)

    # --- Compute metrics ---
    condition_results: dict[str, list[dict]] = {cond: [] for cond in conditions}
    total_texts = sum(len(t) for t in conditions.values())
    progress = tqdm(
        total=total_texts, desc="Computing diversity metrics", unit="text", disable=args.verbose,
    )

    for condition_name, texts in conditions.items():
        for idx, text in enumerate(texts):
            result = compute_diversity_metrics(text, args.max_words, args.min_words)
            condition_results[condition_name].append({"index": idx, **result})

            if args.verbose:
                if result["skipped"]:
                    print(f"  [{condition_name}] idx={idx} SKIPPED (words={result['n_words']})")
                else:
                    d2 = f"{result['distinct_2']:.3f}" if not math.isnan(result["distinct_2"]) else "nan"
                    rtr = f"{result['repeated_trigram_ratio']:.3f}" if not math.isnan(result["repeated_trigram_ratio"]) else "nan"
                    print(
                        f"  [{condition_name}] idx={idx} "
                        f"d1={result['distinct_1']:.3f} d2={d2} rtr={rtr} "
                        f"words={result['n_words']}"
                    )

            progress.update(1)

    progress.close()

    print()
    _output_results(condition_results, source_descs, args)


if __name__ == "__main__":
    main()
