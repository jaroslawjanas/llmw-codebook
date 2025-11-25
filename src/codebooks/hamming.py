"""
Hamming-distance specialized codebook that consumes a base NPZ and builds a Faiss index.

Responsibilities:
- Consume CodebookBase (canonical codewords stored as packed bits in NPZ)
- Build and cache a Faiss IndexBinaryFlat for exact Hamming search
- Provide nearest-neighbor search by Hamming distance
- Delegate non-search helpers (like get_random_row) to the base

File naming:
- Base:   codebook_{algo}_n{n}_u{u}_sk-<secret-key>.npz
- Hamming index: codebook_{algo}_n{n}_u{u}_sk-<secret-key>_metric-hamming.faiss
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import resolution depends on environment
    import faiss  # type: ignore
except ModuleNotFoundError:
    faiss = None
    try:
        import faiss_cpu as faiss  # type: ignore
    except ModuleNotFoundError:
        try:
            import faiss_gpu as faiss  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "Faiss is required for Hamming codebooks. Install `faiss-cpu` (CPU only) or `faiss-gpu` "
                "matching your platform, e.g. `pip install faiss-cpu`."
            ) from exc

from .base import CodebookBase


__all__ = ["CodebookHamming"]


class CodebookHamming:
    """
    Hamming-distance codebook built on top of a CodebookBase.

    Construction:
        base = CodebookBase(...)
        hamming = CodebookHamming(base)

    Usage:
        # Select a deterministic random watermark row (no FAISS needed)
        bits, watermark = hamming.get_random_row(seed)

        # Find closest match by Hamming distance (uses FAISS)
        bits, watermark, distance = hamming.find_closest_match(target_bits)
    """

    def __init__(self, base: CodebookBase, *, cache_dir: Optional[str] = None) -> None:
        if base.packed_codes is None:
            raise RuntimeError("Base codebook not loaded")

        self.base = base
        self.cache_dir = cache_dir or base.cache_dir

        self.filename_base = f"{self.base.filename_base}_metric-hamming"
        self.index_path = os.path.join(self.cache_dir, f"{self.filename_base}.faiss")

        self.index: Optional[Any] = None
        self.bytes_per_code: int = int(self.base.bytes_per_code)
        self.dim_bits: int = int(self.base.dim_bits)

        self._load_or_build_index()

    # ------------------------------------------------------------------
    # Delegated helpers
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def u(self) -> int:
        return self.base.u

    def get_random_row(self, seed: int) -> tuple[list[int], int]:
        return self.base.get_random_row(seed)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_closest_match(self, target_codeword: Sequence[int] | np.ndarray) -> tuple[list[int], int, int]:
        """
        Locate the closest codeword by exact Hamming distance using Faiss.
        Returns (closest_codeword_bits, watermark_identity, distance_value).
        """
        target_array = np.asarray(target_codeword, dtype=np.uint8)
        if target_array.ndim != 1 or target_array.shape[0] != self.u:
            raise ValueError(f"Target codeword must be a 1D sequence of length u={self.u}")

        if self.index is None or self.base.watermark_identities is None or self.base.packed_codes is None:
            raise RuntimeError("Codebook index not initialized")

        query = CodebookBase._pack_bits_1d(target_array, self.u)
        distances, indices = self.index.search(query, 1)  # type: ignore
        idx = int(indices[0, 0])
        dist = int(distances[0, 0])
        watermark = int(self.base.watermark_identities[idx])
        bits = self.base.get_codeword_by_index(idx)
        return bits.tolist(), watermark, dist

    # ------------------------------------------------------------------
    # Index build/load helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        if self.base.packed_codes is None:
            raise RuntimeError("Packed codes unavailable for indexing")

        codes = np.ascontiguousarray(self.base.packed_codes, dtype=np.uint8)
        self.bytes_per_code = int(codes.shape[1])
        self.dim_bits = int(self.bytes_per_code * 8)

        index: Any = faiss.IndexBinaryFlat(self.dim_bits)  # type: ignore[attr-defined]
        index.add(codes)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        writer = getattr(faiss, "write_index_binary", None)
        if callable(writer):
            writer(index, self.index_path)
        else:  # fallback older faiss versions
            faiss.write_index(index, self.index_path)  # type: ignore
        self.index = index

    def _load_or_build_index(self) -> None:
        try:
            reader = getattr(faiss, "read_index_binary", None)
            if callable(reader) and os.path.exists(self.index_path):
                index = reader(self.index_path)
            elif os.path.exists(self.index_path):
                index = faiss.read_index(self.index_path)  # type: ignore
            else:
                raise FileNotFoundError("Faiss index file not found")

            # Validate cached index
            if int(getattr(index, "d")) != int(self.base.dim_bits):  # type: ignore[attr-defined]
                raise ValueError("Cached Faiss index dimension mismatch")
            if int(getattr(index, "ntotal")) != int(self.base.packed_codes.shape[0]):  # type: ignore[attr-defined]
                raise ValueError("Cached Faiss index size mismatch")

            self.index = index
            self.dim_bits = int(getattr(index, "d"))  # type: ignore[attr-defined]
            self.bytes_per_code = int(self.dim_bits // 8)
        except Exception:
            # On any failure (missing/invalid), rebuild index
            self._build_index()
