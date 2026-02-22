"""
BCH Attack Robustness Analysis Script.

Tests how watermark detection degrades under substitution, insertion, and
deletion attacks applied to decoded codewords.

For each attack type, iterates from budget=0 to max_budget and measures
TPR at a fixed FPR threshold (default 1%).
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.tpr_at_fpr import find_fpr_threshold
from scripts.utils import discover_experiments, load_and_merge
from scripts.utils.parallel import (
    create_bch_pool,
    evaluate_attack_robustness_parallel,
    generate_negative_bit_errors_parallel,
    resolve_workers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test BCH watermark robustness against substitution, insertion, and deletion attacks"
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
        "--attack-budget", type=float, default=0.1,
        help="Attack budget: if < 1.0, percentage of u; if >= 1, absolute number of bits (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--z", type=int, default=50000,
        help="Number of random samples for FPR threshold estimation (default: 50000, min: 1000)"
    )
    parser.add_argument(
        "--fpr-target", type=float, default=0.01,
        help="Target FPR (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers (default: 0 = auto = cpu_count)"
    )

    args = parser.parse_args()

    if args.z < 1000:
        parser.error(f"--z must be at least 1000, got {args.z}")

    if not (0 < args.fpr_target < 1):
        parser.error(f"--fpr-target must be between 0 and 1 (exclusive), got {args.fpr_target}")

    # Convert budget to integer
    if args.attack_budget < 1.0:
        args.max_budget = math.ceil(args.attack_budget * args.u)
    else:
        args.max_budget = int(args.attack_budget)

    if args.max_budget <= 0:
        parser.error(f"Attack budget must result in at least 1 bit, got {args.max_budget}")

    return args


def shorten_model_name(name: str) -> str:
    """Shorten model name by removing org prefix."""
    if "/" in name:
        return name.split("/")[-1]
    return name


def apply_substitution_attack(
    codeword: np.ndarray,
    budget: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Flip `budget` random bits in the codeword.

    Args:
        codeword: Binary array of length u
        budget: Number of bits to flip
        rng: Random number generator

    Returns:
        Attacked codeword (copy with bits flipped)
    """
    if budget == 0:
        return codeword.copy()

    attacked = codeword.copy()
    indices = rng.choice(len(codeword), size=budget, replace=False)
    attacked[indices] = 1 - attacked[indices]  # Flip bits
    return attacked


def apply_insertion_attack(
    codeword: np.ndarray,
    budget: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Insert `budget` random bits at random positions, then truncate to original length.

    This simulates an attacker inserting extra tokens/bits, which shifts the alignment.
    After insertion, we truncate from the end to maintain length u.

    Args:
        codeword: Binary array of length u
        budget: Number of bits to insert
        rng: Random number generator

    Returns:
        Attacked codeword (with insertions and truncated to length u)
    """
    if budget == 0:
        return codeword.copy()

    u = len(codeword)
    result = list(codeword)

    # Generate random positions and bits to insert
    # Insert one at a time to avoid index shifting complexity
    for _ in range(budget):
        pos = rng.integers(0, len(result) + 1)
        bit = rng.integers(0, 2)
        result.insert(pos, bit)

    # Truncate to original length u
    return np.array(result[:u], dtype=np.uint8)


def apply_deletion_attack(
    codeword: np.ndarray,
    budget: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Delete `budget` random bits, then pad with 0s to original length.

    This simulates an attacker deleting tokens/bits from the watermarked text.
    After deletion, we pad with zeros at the end to maintain length u.

    Args:
        codeword: Binary array of length u
        budget: Number of bits to delete
        rng: Random number generator

    Returns:
        Attacked codeword (with deletions and padded to length u)
    """
    if budget == 0:
        return codeword.copy()

    u = len(codeword)

    # Select indices to delete (ensure we don't delete more than available)
    n_to_delete = min(budget, u)
    delete_indices = set(rng.choice(u, size=n_to_delete, replace=False))

    # Keep non-deleted bits
    result = [codeword[i] for i in range(u) if i not in delete_indices]

    # Pad with zeros to reach original length
    while len(result) < u:
        result.append(0)

    return np.array(result, dtype=np.uint8)


def evaluate_attack_robustness(
    df,
    bch_codec: BCHCodec,
    attack_fn: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
    max_budget: int,
    threshold: int,
    seed: int,
    attack_name: str,
) -> dict[int, float]:
    """
    Evaluate TPR at each attack budget level.

    For each budget from 0 to max_budget:
    1. Apply attack to each row's decoded_codeword
    2. BCH decode the attacked codeword
    3. Check if bit_errors <= threshold (true positive)
    4. Calculate TPR = true_positives / total_samples

    Args:
        df: DataFrame with 'decoded_codeword' column
        bch_codec: BCH codec instance
        attack_fn: Attack function (substitution, insertion, or deletion)
        max_budget: Maximum attack budget
        threshold: bit_error threshold from FPR analysis
        seed: Random seed
        attack_name: Name of attack for progress bar

    Returns:
        Dict mapping budget -> tpr
    """
    results = {}

    # Pre-convert data for efficiency (only successful watermarks, but total includes all)
    total = len(df)
    successful = df[df["successful_watermark"]]
    decoded_codewords = [
        np.array([int(b) for b in str(cw)], dtype=np.uint8)
        for cw in successful["decoded_codeword"].values
    ]
    original_watermarks = successful["watermark"].values.astype(np.int64)

    for budget in tqdm(range(max_budget + 1), desc=f"  {attack_name}"):
        # Use seed + budget for reproducibility with different random selections per budget
        rng = np.random.default_rng(seed + budget)

        tp = 0

        for i, decoded_codeword in enumerate(decoded_codewords):
            # Apply attack
            attacked_codeword = attack_fn(decoded_codeword, budget, rng)

            # BCH decode
            recovered_watermark, bit_errors = bch_codec.decode_int(attacked_codeword)

            # Check if correct identity AND within threshold
            if recovered_watermark == original_watermarks[i] and bit_errors <= threshold:
                tp += 1

        tpr = tp / total if total > 0 else 0.0
        results[budget] = tpr

    return results


def create_model_plot(
    model: str,
    substitution_results: dict[int, float],
    insertion_results: dict[int, float],
    deletion_results: dict[int, float],
    actual_fpr: float,
    args: argparse.Namespace,
    output_path: Path
) -> None:
    """
    Create a PNG with 3 subplots (substitution, insertion, deletion).

    X-axis: Number of adversarial interventions
    Y-axis: TPR at target FPR (%)
    Green dashed baseline showing budget=0 TPR.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot generation")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    attack_data = [
        ("Substitution", substitution_results, axes[0]),
        ("Insertion", insertion_results, axes[1]),
        ("Deletion", deletion_results, axes[2]),
    ]

    for attack_name, results, ax in attack_data:
        budgets = sorted(results.keys())
        match_rates = [results[b] * 100 for b in budgets]
        baseline = match_rates[0] if budgets[0] == 0 else 100.0

        # Plot line (all points) but markers only for points after the first
        ax.plot(budgets, match_rates, '-', color='#4A90D9', linewidth=1.5)
        ax.plot(budgets[1:], match_rates[1:], 'o', color='#4A90D9', markersize=5)

        # Green dashed baseline
        ax.axhline(y=baseline, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.text(budgets[-1] - 0.5, baseline - 4, f'{baseline:.1f}%',
                color='green', fontsize=9, ha='right', va='top')

        # Add percentage labels for each point (skip first point - baseline shows it)
        for i, (x, y) in enumerate(zip(budgets[1:], match_rates[1:]), start=1):
            y_offset = 5 if y < 90 else -12
            ax.annotate(f'{y:.0f}%', (x, y), textcoords="offset points",
                        xytext=(0, y_offset), ha='center', fontsize=7, color='#333333')

        ax.set_xlabel('Number of adversarial interventions')
        ax.set_ylabel(f'TPR @ FPR={actual_fpr * 100:.2f}%')
        ax.set_title(attack_name)
        ax.set_ylim(0, 100)
        ax.set_xlim(budgets[0], budgets[-1])
        # Space out x-axis ticks to avoid overlap
        max_ticks = 20
        if len(budgets) > max_ticks:
            step = len(budgets) // max_ticks + 1
            ax.set_xticks(budgets[::step])
        else:
            ax.set_xticks(budgets)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Attack Robustness: {shorten_model_name(model)} (n={args.n}, u={args.u}, FPR={actual_fpr * 100:.2f}%)',
        fontsize=12,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_txt(
    args: argparse.Namespace,
    threshold: int,
    actual_fpr: float,
    all_results: dict[str, dict[str, dict[int, float]]],
    output_path: Path
) -> None:
    """Write formatted results with one table per attack type."""
    lines = []
    lines.append("=" * 70)
    lines.append("BCH Attack Robustness Analysis")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Parameters:")
    lines.append(f"  n (watermark bits): {args.n}")
    lines.append(f"  u (codeword length): {args.u}")
    lines.append(f"  max attack budget: {args.max_budget} bits")
    lines.append(f"  z (negative samples): {args.z}")
    lines.append(f"  target FPR: {args.fpr_target * 100:.2f}%")
    lines.append(f"  bit_error threshold: {threshold}")
    lines.append(f"  actual FPR: {actual_fpr * 100:.2f}%")
    lines.append(f"  seed: {args.seed}")
    lines.append("")

    # Collect model short names and attack names
    models = list(all_results.keys())
    short_names = [shorten_model_name(m) for m in models]
    model_col_width = max(len(s) for s in short_names) + 2
    budget_col_width = 7

    attack_names = list(next(iter(all_results.values())).keys())
    budgets = sorted(next(iter(next(iter(all_results.values())).values())).keys())

    for attack_name in attack_names:
        lines.append(f"TPR (%) @ FPR={actual_fpr * 100:.2f}% — {attack_name}")
        header = " " * model_col_width
        header += "".join(f"{b:>{budget_col_width}}" for b in budgets)
        lines.append(header)
        lines.append("-" * len(header))

        for model, short in zip(models, short_names):
            results = all_results[model][attack_name]
            row = f"{short:<{model_col_width}}"
            row += "".join(f"{results[b] * 100:>{budget_col_width}.2f}" for b in budgets)
            lines.append(row)

        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    args = parse_args()

    # Discover experiments
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

    # Create worker pool (handles BCHCodec construction per worker)
    workers = resolve_workers(args.workers)
    print(f"Using {workers} parallel workers\n")

    with create_bch_pool(n=args.n, u=args.u, workers=workers) as pool:
        # Compute FPR threshold once
        print(f"Generating {args.z} random samples for FPR threshold...")
        negative_bit_errors = generate_negative_bit_errors_parallel(
            pool=pool,
            u=args.u,
            z=args.z,
            seed=args.seed,
        )
        threshold, actual_fpr = find_fpr_threshold(negative_bit_errors, args.fpr_target)
        print(f"FPR threshold: bit_errors <= {threshold}")
        print(f"Actual FPR: {actual_fpr * 100:.2f}%\n")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "output" / (
            f"attack_robustness_n{args.n}_u{args.u}_budget{args.max_budget}_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each model
        all_results = {}

        for model in models:
            print(f"\nProcessing {shorten_model_name(model)}...")
            model_paths = matching[matching["model"] == model]["path"].tolist()
            df = load_and_merge(model_paths)
            print(f"  Loaded {len(df)} samples")

            attack_results = evaluate_attack_robustness_parallel(
                pool=pool,
                df=df,
                attack_names=["Substitution", "Insertion", "Deletion"],
                max_budget=args.max_budget,
                threshold=threshold,
                seed=args.seed,
            )

            all_results[model] = attack_results

            # Create per-model plot
            short_model = model.replace("/", "_")
            create_model_plot(
                model=model,
                substitution_results=attack_results["Substitution"],
                insertion_results=attack_results["Insertion"],
                deletion_results=attack_results["Deletion"],
                actual_fpr=actual_fpr,
                args=args,
                output_path=output_dir / f"{short_model}_robustness.png"
            )
            print(f"  Saved plot: {short_model}_robustness.png")

    # Create summary results file
    create_results_txt(args, threshold, actual_fpr, all_results, output_dir / "results.txt")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}\n")

    print("Summary:")
    print("-" * 60)
    for model, attack_results in all_results.items():
        print(f"\n{shorten_model_name(model)}:")
        for attack_type, results in attack_results.items():
            print(f"  {attack_type:<15} baseline={results[0]:.2%}  final={results[args.max_budget]:.2%}")


if __name__ == "__main__":
    main()
