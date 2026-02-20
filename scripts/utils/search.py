"""Multi-GPU Triton search utilities for scripts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def resolve_triton_gpu_count(requested: int) -> int:
    """Resolve requested GPU count to actual available GPUs.

    Args:
        requested: Number of GPUs requested (0 = all available).

    Returns:
        Actual number of GPUs to use (>= 1).
    """
    import torch

    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("No CUDA GPUs available for Triton search")
    if requested == 0:
        return available
    return min(requested, available)


def create_triton_codebooks(base_codebook, num_gpus: int) -> list:
    """Create CodebookLevenshteinTriton instances across GPUs.

    Args:
        base_codebook: CodebookBase instance.
        num_gpus: Number of GPUs to use.

    Returns:
        List of CodebookLevenshteinTriton instances, one per GPU.
    """
    from src.codebooks import CodebookLevenshteinTriton

    return [
        CodebookLevenshteinTriton(base_codebook, device=f"cuda:{i}")
        for i in range(num_gpus)
    ]


def multi_gpu_triton_batch_search(
    codebooks: list,
    queries: list[np.ndarray] | np.ndarray,
    *,
    distances_only: bool = False,
    top_k: int = 1,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run batch Levenshtein search across multiple Triton codebook instances.

    Distributes queries round-robin across GPUs and merges results.
    For a single codebook, delegates directly without threading overhead.

    Args:
        codebooks: List of CodebookLevenshteinTriton instances (one per GPU).
        queries: List of 1D uint8 arrays or 2D (B, length) uint8 array.
        distances_only: If True, return only distances.
        top_k: Number of closest entries per query.

    Returns:
        top_k == 1: same as find_closest_match_batch with (B,) shaped arrays.
        top_k > 1: same but with (B, top_k) shaped arrays.
    """
    num_gpus = len(codebooks)
    n_queries = len(queries)

    # Single GPU: no threading overhead
    if num_gpus <= 1:
        return codebooks[0].find_closest_match_batch(
            queries, distances_only=distances_only, top_k=top_k,
        )

    # Distribute queries round-robin across GPUs
    per_gpu_queries: list[list] = [[] for _ in range(num_gpus)]
    per_gpu_indices: list[list[int]] = [[] for _ in range(num_gpus)]
    for i in range(n_queries):
        gpu_id = i % num_gpus
        per_gpu_queries[gpu_id].append(queries[i])
        per_gpu_indices[gpu_id].append(i)

    def search_on_gpu(gpu_id: int):
        q = per_gpu_queries[gpu_id]
        if not q:
            return gpu_id, None
        # Stack into 2D array if all same length (common case)
        if isinstance(queries, np.ndarray) and queries.ndim == 2:
            q = np.stack(q)
        return gpu_id, codebooks[gpu_id].find_closest_match_batch(
            q, distances_only=distances_only, top_k=top_k,
        )

    # Run in parallel — Triton releases GIL during kernel execution
    gpu_results: dict = {}
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(search_on_gpu, gid): gid for gid in range(num_gpus)}
        for future in as_completed(futures):
            gid, result = future.result()
            if result is not None:
                gpu_results[gid] = result

    # Merge results back in original order
    out_shape = (n_queries,) if top_k <= 1 else (n_queries, top_k)

    if distances_only:
        out_distances = np.empty(out_shape, dtype=np.int32)
        for gid, dists in gpu_results.items():
            idx = np.array(per_gpu_indices[gid])
            out_distances[idx] = dists
        return out_distances
    else:
        out_indices = np.empty(out_shape, dtype=np.int64)
        out_watermarks = np.empty(out_shape, dtype=np.int64)
        out_distances = np.empty(out_shape, dtype=np.int32)
        for gid, (indices, watermarks, dists) in gpu_results.items():
            idx = np.array(per_gpu_indices[gid])
            out_indices[idx] = indices
            out_watermarks[idx] = watermarks
            out_distances[idx] = dists
        return out_indices, out_watermarks, out_distances
