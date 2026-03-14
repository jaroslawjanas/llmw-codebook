"""
Parallel BCH decode utilities using multiprocessing Pool.

Uses the Pool initializer pattern to create one BCHCodec per worker process,
avoiding the ~10s galois.BCH() constructor cost per task and sidestepping
pickling issues with galois objects.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Optional

import numpy as np
from tqdm import tqdm


# ---- Module-level worker state (set by _init_worker) ----

_worker_codec = None  # BCHCodec instance, one per worker process


def _init_worker(n: int, u: int, bch_m: Optional[int]) -> None:
    """Pool initializer: create BCHCodec in each worker process."""
    global _worker_codec

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.bch_codec import BCHCodec
    _worker_codec = BCHCodec(n=n, u=u, bch_m=bch_m)


# ---- Worker functions (module-level for pickling) ----

def _negative_chunk_worker(args: tuple) -> list[int]:
    """
    Process a chunk of random codeword decodes for FPR threshold estimation.

    Args:
        args: (chunk_size, seed_entropy) where seed_entropy comes from SeedSequence

    Returns:
        List of bit_errors for this chunk
    """
    chunk_size, seed_entropy = args
    rng = np.random.default_rng(seed_entropy)

    results = []
    for _ in range(chunk_size):
        random_codeword = rng.integers(0, 2, size=_worker_codec.u, dtype=np.uint8)
        _, bit_errors = _worker_codec.decode(random_codeword)
        results.append(bit_errors)
    return results


def _attack_budget_worker(args: tuple) -> tuple[str, int, int]:
    """
    Evaluate one (attack_type, budget) pair across all samples.

    Args:
        args: (attack_type, budget, decoded_codewords, original_watermarks,
               threshold, seed)

    Returns:
        (attack_type, budget, tp_count)
    """
    attack_type, budget, decoded_codewords, original_watermarks, threshold, seed = args

    from scripts.attack_robustness import (
        apply_substitution_attack,
        apply_insertion_attack,
        apply_deletion_attack,
    )
    attack_fn_map = {
        "substitution": apply_substitution_attack,
        "insertion": apply_insertion_attack,
        "deletion": apply_deletion_attack,
    }
    attack_fn = attack_fn_map[attack_type]

    rng = np.random.default_rng(seed + budget)
    tp = 0

    for i, decoded_codeword in enumerate(decoded_codewords):
        attacked_codeword = attack_fn(decoded_codeword, budget, rng)
        recovered_watermark, bit_errors = _worker_codec.decode_int(attacked_codeword)
        if recovered_watermark == original_watermarks[i] and bit_errors <= threshold:
            tp += 1

    return attack_type, budget, tp


# ---- Public API ----

def create_bch_pool(
    n: int,
    u: int,
    workers: int,
    bch_m: Optional[int] = None,
) -> mp.Pool:
    """
    Create a multiprocessing Pool with BCHCodec-initialized workers.

    Each worker process creates its own BCHCodec instance at init time,
    paying the ~10s galois.BCH() constructor cost once per worker.

    Args:
        n: Watermark message length in bits
        u: Codeword length in bits
        workers: Number of worker processes
        bch_m: Optional BCH field degree (auto-select if None)

    Returns:
        Initialized Pool (use as context manager)
    """
    ctx = mp.get_context('spawn')
    return ctx.Pool(workers, initializer=_init_worker, initargs=(n, u, bch_m))


def generate_negative_bit_errors_parallel(
    pool: mp.Pool,
    u: int,
    z: int,
    seed: int,
) -> np.ndarray:
    """
    Parallel version of generate_negative_bit_errors.

    Splits z samples into chunks distributed across pool workers.
    Uses np.random.SeedSequence for proper deterministic seed derivation.

    Note: Results differ from sequential version due to different RNG stream
    partitioning, but are deterministic for a given (seed, workers) combination.

    Args:
        pool: Initialized Pool with BCHCodec workers
        u: Codeword length (for progress description)
        z: Number of random samples
        seed: Base random seed

    Returns:
        Array of bit_errors values
    """
    num_workers = pool._processes
    num_chunks = min(z, num_workers * 8)  # Fine granularity for smooth progress
    chunk_size = z // num_chunks
    remainder = z % num_chunks

    # Derive deterministic child seeds
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(num_chunks)

    chunks = []
    for i in range(num_chunks):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append((size, child_seeds[i].entropy))

    all_errors = []
    for chunk_result in tqdm(
        pool.imap(_negative_chunk_worker, chunks),
        total=num_chunks,
        desc="Generating negatives",
    ):
        all_errors.extend(chunk_result)

    return np.array(all_errors)


def evaluate_attack_robustness_parallel(
    pool: mp.Pool,
    df,
    attack_names: list[str],
    max_budget: int,
    threshold: int,
    seed: int,
) -> dict[str, dict[int, float]]:
    """
    Parallel attack robustness evaluation for all attack types.

    Flattens all (attack_type, budget) pairs into independent tasks and
    distributes across the pool. Each task processes all samples for one
    (attack_type, budget) combination.

    Args:
        pool: Initialized Pool with BCHCodec workers
        df: DataFrame with 'decoded_codeword', 'watermark', 'successful_watermark'
        attack_names: List of attack names (e.g., ["Substitution", "Insertion", "Deletion"])
        max_budget: Maximum attack budget
        threshold: bit_error threshold from FPR analysis
        seed: Random seed

    Returns:
        Dict mapping attack_name -> dict[budget -> match_rate_at_fpr]
    """
    total = len(df)
    successful = df[df["successful_watermark"]]

    # Pre-convert data
    decoded_codewords = [
        np.array([int(b) for b in str(cw)], dtype=np.uint8)
        for cw in successful["decoded_codeword"].values
    ]
    original_watermarks = successful["watermark"].values.astype(np.int64)

    # Build task list
    tasks = []
    for attack_name in attack_names:
        for budget in range(max_budget + 1):
            tasks.append((
                attack_name.lower(),
                budget,
                decoded_codewords,
                original_watermarks,
                threshold,
                seed,
            ))

    # Process in parallel
    output = {name: {} for name in attack_names}
    for attack_type, budget, tp in tqdm(
        pool.imap_unordered(_attack_budget_worker, tasks),
        total=len(tasks),
        desc="Attack evaluation",
    ):
        # Map back to original case
        for name in attack_names:
            if name.lower() == attack_type:
                output[name][budget] = tp / total if total > 0 else 0.0
                break

    return output


def resolve_workers(workers: int) -> int:
    """Resolve worker count: 0 = auto (cpu_count), otherwise use as-is."""
    if workers == 0:
        return os.cpu_count() or 1
    return workers
