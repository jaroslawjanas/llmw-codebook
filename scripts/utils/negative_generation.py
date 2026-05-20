"""Shared negative-distance generation and caching utilities.

Used by auroc.py, matchrate_at_fpr.py, and any future script that needs
negative-sample distances against a codebook.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Callable

import numpy as np
from tqdm import tqdm

import src.paths as paths
from scripts.utils.search import (
    create_triton_codebooks,
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


def _build_cache_basename(
    prefix: str,
    n: int,
    u: int,
    z: int,
    seed: int,
    secret_key: str,
    codeword_generation: str,
    distance: str,
    bch_m: int | None,
) -> str:
    """Build a deterministic cache filename base from params (no extension)."""
    distance_norm = distance.replace("-triton", "")
    parts = [
        prefix,
        f"c-{codeword_generation}",
        f"n{n}",
        f"u{u}",
        f"z{z}",
        f"seed{seed}",
        f"sk-{secret_key}",
        distance_norm,
    ]
    if codeword_generation == "bch":
        m_tag = "auto" if bch_m is None else str(bch_m)
        parts.append(f"bchm{m_tag}")
    return "_".join(parts)


def _build_cache_config(
    n: int,
    u: int,
    z: int,
    seed: int,
    secret_key: str,
    codeword_generation: str,
    distance: str,
    bch_m: int | None,
) -> dict:
    """Build the config dict stored alongside the cached NPZ."""
    return {
        "n": n,
        "u": u,
        "z": z,
        "seed": seed,
        "secret_key": secret_key,
        "codeword_generation": codeword_generation,
        "distance": distance,
        "bch_m": bch_m,
    }


def _validate_cache_config(cached: dict, current: dict) -> bool:
    """Check that all fields in current match cached."""
    for key in current:
        if cached.get(key) != current[key]:
            return False
    return True


def generate_negative_distances(
    n: int,
    u: int,
    secret_key: str,
    codeword_generation: str,
    distance: str,
    bch_m: int | None,
    z: int,
    seed: int,
    num_gpus: int,
) -> np.ndarray:
    """Generate distances for z random binary sequences against the codebook.

    Args:
        n, u, secret_key, codeword_generation, distance, bch_m: Codebook params.
        z: Number of random negative samples.
        seed: Random seed.
        num_gpus: GPUs for Triton search (0 = all).

    Returns:
        np.ndarray of shape (z,) dtype int32 — distances for each sequence.
    """
    # Build generator
    if codeword_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs: dict = {}
    elif codeword_generation == "linear":
        generator_cls = LinearCodeGenerator
        generator_kwargs = {}
    elif codeword_generation == "bch":
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"m": bch_m}
    else:
        raise ValueError(f"Unknown codeword_generation: {codeword_generation}")

    base_codebook = CodebookBase(
        n=n,
        u=u,
        secret_key=secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
    )

    # Normalize distance name for experiment matching
    distance_norm = distance.replace("-triton", "")

    # Generate random bits
    print(f"Generating {z} random bit sequences...")
    rng = np.random.default_rng(seed)
    random_bits = rng.integers(0, 2, size=(z, u), dtype=np.uint8)

    # Search based on distance metric
    if distance_norm == "hamming":
        codebook = CodebookHamming(base_codebook)
        queries = CodebookBase._pack_bits_2d(random_bits, u)
        # Chunked Faiss search with progress bar for large z
        chunk_size = min(5000, max(1000, z // 100))
        n_chunks = (z + chunk_size - 1) // chunk_size
        distances_list = []
        for i in tqdm(range(0, z, chunk_size), total=n_chunks, desc="Faiss search", unit="chunk"):
            chunk = queries[i:i + chunk_size]
            d, _ = codebook.index.search(chunk, 1)
            distances_list.append(d[:, 0])
        return np.concatenate(distances_list).astype(np.int32)

    elif distance == "levenshtein-triton":
        num_gpus = resolve_triton_gpu_count(num_gpus)
        codebooks = create_triton_codebooks(base_codebook, num_gpus)
        print(f"Running Triton search over {z} queries across {num_gpus} GPU{'s' if num_gpus > 1 else ''}...")
        return multi_gpu_triton_batch_search(
            codebooks, random_bits, distances_only=True,
        ).astype(np.int32)

    else:  # levenshtein
        codebook = CodebookLevenshtein(base_codebook)
        print(f"Running batch Levenshtein search over {z} queries...")
        return codebook.find_closest_match_batch(
            random_bits, distances_only=True,
        ).astype(np.int32)


def load_or_generate_negatives(
    prefix: str,
    n: int,
    u: int,
    secret_key: str,
    codeword_generation: str,
    distance: str,
    bch_m: int | None,
    z: int,
    seed: int,
    num_gpus: int,
    generate_fn: Callable[[], np.ndarray] | None = None,
) -> np.ndarray:
    """Load cached negative distances or generate and cache them.

    Args:
        prefix: Cache key prefix (e.g. "auroc_negatives" or "fpr_negatives").
        generate_fn: Optional callable overriding the default generation.
                     If None, uses ``generate_negative_distances`` with the
                     same parameters.

    Returns:
        np.ndarray of shape (z,) dtype int32.
    """
    paths.ensure_directories()
    basename = _build_cache_basename(
        prefix, n, u, z, seed, secret_key, codeword_generation, distance, bch_m,
    )
    npz_path = os.path.join(paths.SCRIPTS_CACHE_DIR, f"{basename}.npz")
    json_path = os.path.join(paths.SCRIPTS_CACHE_DIR, f"{basename}.json")
    current_config = _build_cache_config(
        n, u, z, seed, secret_key, codeword_generation, distance, bch_m,
    )

    # Try loading from cache
    if os.path.isfile(npz_path) and os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cached_config = json.load(f)
            if _validate_cache_config(cached_config, current_config):
                with np.load(npz_path) as data:
                    distances = data["distances"]
                if len(distances) == z:
                    print(f"Loaded negative distances from cache: {npz_path}")
                    return distances.astype(np.int32, copy=False)
        except Exception:
            pass  # Fall through to generation

    # Generate from scratch
    if generate_fn is None:
        distances = generate_negative_distances(
            n=n, u=u, secret_key=secret_key,
            codeword_generation=codeword_generation,
            distance=distance, bch_m=bch_m,
            z=z, seed=seed, num_gpus=num_gpus,
        )
    else:
        distances = generate_fn()

    # Save cache
    try:
        np.savez_compressed(npz_path, distances=distances.astype(np.int32, copy=False))
        config_to_save = {**current_config, "timestamp": datetime.now().isoformat()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, indent=2)
        print(f"Saved negative distances cache: {npz_path}")
    except Exception as e:
        print(f"Warning: failed to save cache: {e}")

    return distances
