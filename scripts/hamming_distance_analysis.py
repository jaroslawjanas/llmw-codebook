#!/usr/bin/env python3
"""
Hamming Distance Analysis for Codebook Randomness Testing

This script analyzes the distribution of Hamming distances when searching for 
random bit sequences in a generated codebook. It helps evaluate the randomness 
and collision properties of the codebook generation algorithm.
"""

import os
import sys

# 1) Compute the project root (one level up from scripts/)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)

# 2) Prepend it to sys.path so Python can see src/
sys.path.insert(0, PROJECT_ROOT)

import argparse
import random
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from src.codebook import CodebookRNG, CodebookBCH
import src.paths as paths
from tqdm import tqdm


def gaussian(x, amplitude, mean, stddev):
    """Gaussian function for curve fitting."""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def moving_average(data, window_size):
    """Calculate moving average of data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def theoretical_binomial_pmf(u, max_distance=None):
    """Calculate theoretical binomial PMF for random bit comparison."""
    if max_distance is None:
        max_distance = u
    k_values = np.arange(0, min(max_distance + 1, u + 1))
    # Binomial PMF: C(u,k) * (0.5)^u
    pmf_values = [stats.binom.pmf(k, u, 0.5) for k in k_values]
    return k_values, pmf_values


def fit_gaussian_to_histogram(hist_values, bin_centers):
    """Fit Gaussian distribution to histogram data."""
    try:
        # Initial parameter estimates
        mean_guess = np.average(bin_centers, weights=hist_values)
        stddev_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist_values))
        amplitude_guess = np.max(hist_values)
        
        # Fit the curve
        popt, _ = curve_fit(gaussian, bin_centers, hist_values, 
                           p0=[amplitude_guess, mean_guess, stddev_guess])
        return popt
    except:
        return None


def create_analysis_plots(distances, args, output_dir):
    """Create comprehensive analysis plots."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Hamming Distance Analysis (n={args.n}, u={args.u}, z={args.z}, cb={args.codebook})',
                 fontsize=16, fontweight='bold')
    
    # 1. Histogram with multiple fits
    ax1 = axes[0, 0]
    
    # Create histogram
    n_bins = min(50, max(20, int(np.sqrt(len(distances)))))
    counts, bin_edges, _ = ax1.hist(distances, bins=n_bins, alpha=0.7, color='skyblue', 
                                   edgecolor='black', linewidth=0.5, density=True, label=f'Histogram (cb={args.codebook})')
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit Gaussian
    gaussian_params = fit_gaussian_to_histogram(counts, bin_centers)
    if gaussian_params is not None:
        gaussian_curve = gaussian(bin_centers, *gaussian_params)
        ax1.plot(bin_centers, gaussian_curve, 'r-', linewidth=2, 
                label=f'Gaussian Fit (μ={gaussian_params[1]:.1f}, σ={gaussian_params[2]:.1f})')
    
    # Theoretical binomial
    binom_k, binom_pmf = theoretical_binomial_pmf(args.u, max(distances))
    if len(binom_k) > 0:
        ax1.plot(binom_k, binom_pmf, 'g--', linewidth=2, 
                label=f'Theoretical Binomial (p=0.5)')
    
    ax1.set_xlabel('Hamming Distance')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Distribution of Hamming Distances')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Moving average trend
    ax2 = axes[0, 1]
    sorted_distances = np.sort(distances)
    cumulative_probs = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    
    # Calculate moving average on CDF
    window_size = max(10, len(sorted_distances) // 50)
    if len(sorted_distances) > window_size:
        ma_distances = moving_average(sorted_distances, window_size)
        ma_probs = np.linspace(0, 1, len(ma_distances))
        ax2.plot(ma_probs, ma_distances, 'b-', linewidth=2, label=f'Moving Average (window={window_size})')
        ax2.scatter(cumulative_probs[::max(1, len(cumulative_probs)//100)], 
                    sorted_distances[::max(1, len(sorted_distances)//100)], 
                    alpha=0.3, s=1, color='gray', label='Data points')
    else:
        ax2.plot(cumulative_probs, sorted_distances, 'b-', linewidth=2)
    
    ax2.set_xlabel('Cumulative Probability')
    ax2.set_ylabel('Hamming Distance')
    ax2.set_title('Cumulative Distribution with Trend')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = axes[1, 0]
    ax3.boxplot(distances, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax3.set_ylabel('Hamming Distance')
    ax3.set_title('Box Plot of Hamming Distances')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q plot against normal distribution
    ax4 = axes[1, 1]
    stats.probplot(distances, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot vs Normal Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    cb_tag = f"cb-{args.codebook}"
    plot_path = os.path.join(output_dir, f'hamming_distribution_analysis_{cb_tag}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def calculate_statistics(distances):
    """Calculate comprehensive statistics."""
    if len(distances) == 0:
        return {}
    
    distances_array = np.array(distances)
    
    stats_dict = {
        'count': len(distances),
        'min': int(np.min(distances_array)),
        'max': int(np.max(distances_array)),
        'mean': float(np.mean(distances_array)),
        'median': float(np.median(distances_array)),
        'std': float(np.std(distances_array)),
        'variance': float(np.var(distances_array)),
        'q25': float(np.percentile(distances_array, 25)),
        'q75': float(np.percentile(distances_array, 75)),
        'mode': int(stats.mode(distances_array, keepdims=True).mode[0])
    }
    
    return stats_dict


def main():
    parser = argparse.ArgumentParser(description="Hamming Distance Analysis for Codebook Randomness Testing")
    parser.add_argument("--n", type=int, default=16, help="Size of the code table (2^n entries)")
    parser.add_argument("--u", type=int, default=100, help="Size of the codes (bit length)")
    parser.add_argument("--z", type=int, default=1000, help="Number of random sequences to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Secret key used for codebook generation (string)")
    parser.add_argument("--codebook", type=str, choices=["rng", "bch"], default="rng", help="Codebook type for analysis negatives: 'rng' or 'bch'")
    parser.add_argument("--bch-t", type=int, default=None, help="BCH t (if None, auto-choose largest t with k >= n)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH m (n0=2^m-1). If None, minimal m with n0 >= u")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output and progress information")
    parser.add_argument("--output-dir", type=str, help="Output directory (auto-generated if not provided)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    # Initialize random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    
    if args.verbose:
        print(f"Using secret key: {args.secret_key}")
        print(f"Creating codebook with n={args.n}, u={args.u}")

    # Create dynamic output directory if not provided
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cb_tag = f"cb-{args.codebook}"
        output_dir = os.path.join("scripts", "output", f"hamming_analysis_n{args.n}_u{args.u}_z{args.z}_{cb_tag}_{timestamp}")
        
    os.makedirs(output_dir, exist_ok=True)
    if args.verbose:
        print(f"Output directory: {output_dir}")

    # Create codebook
    if args.codebook == "rng":
        codebook = CodebookRNG(
            n=args.n,
            u=args.u,
            secret_key=args.secret_key,
            npz_dir=paths.CODEBOOKS_CACHE_DIR
        )
    else:
        codebook = CodebookBCH(
            n=args.n,
            u=args.u,
            secret_key=args.secret_key,
            t=args.bch_t,
            m=args.bch_m,
            npz_dir=paths.CODEBOOKS_CACHE_DIR
        )

    # Generate random sequences and find closest matches
    hamming_distances = []
    sequence_data = []
    
    if args.verbose:
        print(f"Generating {args.z} random sequences of length {args.u}...")
    
    for i in tqdm(range(args.z), desc="Testing random sequences", disable=not args.verbose):
        # Generate random bit sequence of length u
        random_sequence = [random.randint(0, 1) for _ in range(args.u)]
        
        # Find closest match (exact top-1 via Faiss)
        try:
            _, _, hamming_distance = codebook.find_closest_match(random_sequence)
            hamming_distances.append(hamming_distance)
            sequence_data.append({
                'sequence_id': i,
                'hamming_distance': hamming_distance
            })
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to process sequence {i}: {e}")
            continue

    if len(hamming_distances) == 0:
        print("Error: No valid hamming distances collected!")
        return

    # Calculate statistics
    stats_dict = calculate_statistics(hamming_distances)
    
    # Theoretical expectations for comparison
    theoretical_mean = args.u / 2
    theoretical_std = np.sqrt(args.u / 4)
    
    # Create results summary
    results_file = os.path.join(output_dir, 'hamming_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("HAMMING DISTANCE ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment Parameters:\n")
        f.write(f"  n (code table size): {args.n}\n")
        f.write(f"  u (code length): {args.u}\n")
        f.write(f"  z (test sequences): {args.z}\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  secret_key: {args.secret_key}\n")
        f.write(f"  timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        f.write("Statistical Results:\n")
        f.write(f"  Valid sequences processed: {stats_dict['count']}\n")
        f.write(f"  Minimum hamming distance: {stats_dict['min']}\n")
        f.write(f"  Maximum hamming distance: {stats_dict['max']}\n")
        f.write(f"  Mean hamming distance: {stats_dict['mean']:.4f}\n")
        f.write(f"  Median hamming distance: {stats_dict['median']:.4f}\n")
        f.write(f"  Standard deviation: {stats_dict['std']:.4f}\n")
        f.write(f"  Variance: {stats_dict['variance']:.4f}\n")
        f.write(f"  Mode: {stats_dict['mode']}\n")
        f.write("\n")
        
        # Test for normality
        if len(hamming_distances) > 8:
            shapiro_stat, shapiro_p = stats.shapiro(hamming_distances[:5000])  # Limit for performance
            f.write(f"\nNormality Test (Shapiro-Wilk):\n")
            f.write(f"  Statistic: {shapiro_stat:.6f}\n")
            f.write(f"  p-value: {shapiro_p:.6f}\n")
            f.write(f"  Normal distribution likely: {'Yes' if shapiro_p > 0.05 else 'No'} (p > 0.05)\n")

    # Save raw data
    df = pd.DataFrame(sequence_data)
    csv_file = os.path.join(output_dir, 'hamming_distances.csv')
    df.to_csv(csv_file, index=False)
    
    # Save parameters
    params_file = os.path.join(output_dir, 'parameters.json')
    params = {
        'n': args.n,
        'u': args.u,
        'z': args.z,
        'seed': args.seed,
        'secret_key': args.secret_key,
        'codebook': args.codebook,
        'bch_t': args.bch_t,
        'bch_m': args.bch_m,
        'timestamp': datetime.now().isoformat(),
        'statistics': stats_dict,
        'theoretical_mean': theoretical_mean,
        'theoretical_std': theoretical_std
    }
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    # Create plots
    if not args.no_plots and len(hamming_distances) > 0:
        try:
            plot_path = create_analysis_plots(hamming_distances, args, output_dir)
            print(f"Analysis plots saved to: {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to create plots: {e}")

    # Print summary
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Valid sequences processed: {stats_dict['count']}")
    print(f"Hamming distance range: {stats_dict['min']} - {stats_dict['max']}")
    print(f"Mean: {stats_dict['mean']:.4f}")
    print(f"Std: {stats_dict['std']:.4f}")
    
    if len(hamming_distances) > 8:
        shapiro_stat, shapiro_p = stats.shapiro(hamming_distances[:5000])
        print(f"Normal distribution likely: {'Yes' if shapiro_p > 0.05 else 'No'}")


if __name__ == "__main__":
    main()
