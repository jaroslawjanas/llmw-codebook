#!/usr/bin/env python3
"""
Re-run codebook search on existing experiment results with a different distance metric.

Avoids re-generating text (expensive) when you only want to compare search results
across metrics. Uses the codebook SHA-256 hash in config.json to verify the same
codebook is used.

Usage:
  python scripts/re_search.py output/<experiment_folder> --distance levenshtein --verbose
  python scripts/re_search.py output/<experiment_folder> --distance levenshtein-triton
  python scripts/re_search.py output/<experiment_folder> --distance hamming
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import src.paths as paths
from scripts.utils import (
    create_triton_codebooks,
    load_experiment,
    multi_gpu_triton_batch_search,
    resolve_triton_gpu_count,
)
from src.codebooks import (
    BCHCodewordGenerator,
    CodebookBase,
    CodebookHamming,
    CodebookLevenshtein,
    LinearCodeGenerator,
    RNGCodewordGenerator,
)


def main():
    parser = argparse.ArgumentParser(
        description="Re-run codebook search on existing experiment results with a different distance metric"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to existing experiment output directory",
    )
    parser.add_argument(
        "--distance",
        type=str,
        required=True,
        choices=["hamming", "levenshtein", "levenshtein-triton"],
        help="Target distance metric",
    )
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs for Triton search (0 = all available, default: 0)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")

    args = parser.parse_args()

    paths.ensure_directories()

    # ------------------------------------------------------------------
    # 1. Load source experiment
    # ------------------------------------------------------------------
    experiment = load_experiment(args.experiment_dir)
    config = experiment["config"]
    generations = experiment["generations"]
    source_dir = experiment["path"]

    wm = config["watermark"]
    n = wm["n"]
    u = wm["u"]
    secret_key = wm["secret_key"]
    codeword_generation = wm.get("codeword_generation", "rng")
    bch_m = wm.get("bch_m", None)
    source_distance = wm.get("distance", "hamming")

    # Check for codebook hash
    codebook_hash = wm.get("codebook_hash")
    if codebook_hash is None:
        print(
            "Error: Source experiment has no codebook_hash in config.json.\n"
            "Re-run with the latest main.py to generate a config with codebook_hash,\n"
            "or add the hash manually."
        )
        sys.exit(1)

    # Warn if same distance metric
    source_normalized = source_distance.replace("-triton", "")
    target_normalized = args.distance.replace("-triton", "")
    if source_normalized == target_normalized:
        print(
            f"Warning: Target distance '{args.distance}' normalizes to same metric "
            f"as source '{source_distance}'. Proceeding anyway (useful for verification)."
        )

    if args.verbose:
        print(f"Source experiment: {source_dir}")
        print(f"  n={n}, u={u}, secret_key={secret_key}")
        print(f"  codeword_generation={codeword_generation}, bch_m={bch_m}")
        print(f"  source distance: {source_distance}")
        print(f"  target distance: {args.distance}")
        print(f"  rows in CSV: {len(generations)}")

    # ------------------------------------------------------------------
    # 2. Build codebook + verify hash
    # ------------------------------------------------------------------
    if codeword_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs: dict = {}
    elif codeword_generation == "linear":
        generator_cls = LinearCodeGenerator
        generator_kwargs = {}
    else:
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"m": bch_m}

    base_codebook = CodebookBase(
        n=n,
        u=u,
        secret_key=secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
    )

    computed_hash = hashlib.sha256(base_codebook.packed_codes.tobytes()).hexdigest()
    if computed_hash != codebook_hash:
        print(
            f"Error: Codebook hash mismatch.\n"
            f"  Expected (from config): {codebook_hash}\n"
            f"  Computed:               {computed_hash}\n"
            f"The codebook parameters may have changed or the cache is stale."
        )
        sys.exit(1)

    if args.verbose:
        print(f"Codebook hash verified: {computed_hash[:16]}...")

    # Wrap with target metric
    codebooks = None
    if args.distance == "hamming":
        codebook = CodebookHamming(base_codebook)
    elif args.distance == "levenshtein-triton":
        num_gpus = resolve_triton_gpu_count(args.num_gpus)
        codebooks = create_triton_codebooks(base_codebook, num_gpus)
        codebook = codebooks[0]
        if args.verbose:
            print(f"Triton search across {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
    else:
        codebook = CodebookLevenshtein(base_codebook)

    # ------------------------------------------------------------------
    # 3. Re-search
    # ------------------------------------------------------------------
    if generations.empty:
        print("Warning: Source generations.csv is empty. Producing empty output.")
        retrieved_watermarks = np.array([], dtype=np.int64)
        distances = np.array([], dtype=np.int32)
    else:
        # Parse decoded_codeword strings → uint8 arrays
        queries = []
        for cw_str in generations["decoded_codeword"]:
            queries.append(np.array([int(c) for c in str(cw_str)], dtype=np.uint8))

        if args.verbose:
            print(f"Running {args.distance} search over {len(queries)} queries...")

        search_start = time.time()

        if isinstance(codebook, CodebookHamming):
            # Batch Faiss search
            query_matrix = np.stack(queries)
            packed = CodebookBase._pack_bits_2d(query_matrix, u)
            dist_arr, idx_arr = codebook.index.search(packed, 1)
            distances = dist_arr[:, 0].astype(np.int32)
            retrieved_watermarks = base_codebook.watermark_identities[idx_arr[:, 0]]
        elif codebooks and len(codebooks) > 1:
            # Multi-GPU Triton search
            _, retrieved_watermarks, distances = multi_gpu_triton_batch_search(codebooks, queries)
        else:
            # Single-GPU Levenshtein / Triton batch search
            _, retrieved_watermarks, distances = codebook.find_closest_match_batch(queries)

        search_time = time.time() - search_start

        if args.verbose:
            print(f"Search completed in {search_time:.2f}s ({search_time / len(queries) * 1000:.1f} ms/query)")

    # Build result columns
    watermark_col = generations["watermark"].values
    successful = np.array(
        [int(r) == int(w) for r, w in zip(retrieved_watermarks, watermark_col)],
        dtype=bool,
    )

    # ------------------------------------------------------------------
    # 4. Write new output folder
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = config["model"]["name"].replace("/", "_")
    dataset_name_clean = config["dataset"]["name"].replace("/", "_")
    output_subdir = f"{model_name_clean}_{dataset_name_clean}_n{n}_u{u}_{args.distance}_{timestamp}"
    output_dir = os.path.join("output", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # config.json — deep copy, update distance and timestamp
    new_config = copy.deepcopy(config)
    new_config["watermark"]["distance"] = args.distance
    new_config["experiment"]["timestamp"] = timestamp
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)

    # generations.csv — keep source columns, replace search results
    new_gen = pd.DataFrame({
        "codeword": generations["codeword"].values,
        "watermark": watermark_col,
        "decoded_codeword": generations["decoded_codeword"].values,
        "retrieved_watermark": retrieved_watermarks.astype(int) if len(retrieved_watermarks) > 0 else [],
        "successful_watermark": successful if len(successful) > 0 else [],
        "distance": distances.astype(int) if len(distances) > 0 else [],
    })
    new_gen.to_csv(os.path.join(output_dir, "generations.csv"), index=False)

    # generations.txt — copy from source
    source_txt = source_dir / "generations.txt"
    dest_txt = os.path.join(output_dir, "generations.txt")
    if source_txt.exists():
        shutil.copy2(str(source_txt), dest_txt)

    # results.txt
    total_rows = len(new_gen)
    n_successful = int(successful.sum()) if len(successful) > 0 else 0
    successful_distances = distances[successful] if len(successful) > 0 else np.array([])

    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT PARAMETERS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {config['model']['name']}\n")
        ds = config["dataset"]
        f.write(f"Dataset: {ds['name']}/{ds.get('subset', '')} ({ds.get('split', '')} split, {ds.get('column', '')} column)\n")
        f.write(f"n (code table size): {n}\n")
        f.write(f"u (code size): {u}\n")
        f.write(f"Distance metric: {args.distance}\n")
        f.write(f"Codeword generation: {codeword_generation}\n")
        f.write(f"Secret key: {secret_key}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Successful watermark recoveries: {n_successful} out of {total_rows}\n")
        if total_rows > 0:
            f.write(f"Success rate: {n_successful / total_rows * 100:.2f}%\n")
        f.write("\n")

        if len(successful_distances) > 0:
            f.write(f"Average {args.distance} distance for successful recoveries: {np.mean(successful_distances):.2f}\n")
            f.write(f"Min {args.distance} distance for successful recoveries: {int(np.min(successful_distances))}\n")
            f.write(f"Max {args.distance} distance for successful recoveries: {int(np.max(successful_distances))}\n")
        else:
            f.write(f"No successful recoveries to calculate {args.distance} distance statistics\n")

        if not generations.empty:
            f.write(f"Search time: {search_time:.2f}s ({search_time / len(queries) * 1000:.1f} ms/query)\n")

    print(f"Output saved to: {output_dir}")
    print(f"Successful watermark recoveries: {n_successful} out of {total_rows}")
    if len(successful_distances) > 0:
        print(f"Average {args.distance} distance for successful recoveries: {np.mean(successful_distances):.2f}")


if __name__ == "__main__":
    main()
