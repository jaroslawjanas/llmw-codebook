"""
Hamming-distance codebook implementations backed by Faiss IndexBinaryFlat.

Builds upon the generic CodebookBase combined with specific codeword generators.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Type

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
from .generation import BCHCodewordGenerator, CodewordGenerator, RNGCodewordGenerator

__all__ = [
    "HammingCodebookBase",
    "HammingCodebookRNG",
    "HammingCodebookBCH",
    "CodebookRNG",
    "CodebookBCH",
]


class HammingCodebookBase(CodebookBase):
    """
    Concrete CodebookBase implementation that stores packed codewords and
    indexes them with Faiss IndexBinaryFlat for exact Hamming search.
    """

    def __init__(
        self,
        *,
        n: int,
        u: int,
        secret_key: str,
        generator_cls: Type[CodewordGenerator],
        generator_kwargs: Optional[dict] = None,
        cache_dir: Optional[str] = None,
        workers: int = 4,
        chunk_rows: Optional[int] = None,
    ) -> None:
        self.npz_path: Optional[str] = None
        self.index_path: Optional[str] = None
        self.packed_codes: Optional[np.ndarray] = None
        self.watermark_identities: Optional[np.ndarray] = None
        self.bytes_per_code: int = 0
        self.dim_bits: int = 0
        self.index: Optional[Any] = None

        super().__init__(
            n=n,
            u=u,
            secret_key=secret_key,
            generator_cls=generator_cls,
            generator_kwargs=generator_kwargs,
            cache_dir=cache_dir,
            workers=workers,
            chunk_rows=chunk_rows,
        )

    # ------------------------------------------------------------------
    # Abstract hook implementations
    # ------------------------------------------------------------------

    def _configure_paths(self) -> None:
        assert self.cache_dir is not None
        filename = f"{self.filename_base}_metric-hamming"
        self.npz_path = os.path.join(self.cache_dir, f"{filename}.npz")
        self.index_path = os.path.join(self.cache_dir, f"{filename}.faiss")

    def _cache_exists(self) -> bool:
        return bool(self.npz_path and os.path.exists(self.npz_path)) and bool(
            self.index_path and os.path.exists(self.index_path)
        )

    def _store_generated_codewords(self, codewords: np.ndarray) -> None:
        if self.npz_path is None:
            raise RuntimeError("Cache path not configured")

        packed_codes = self._pack_bits_2d(codewords, self.u)
        watermark_identities = np.arange(codewords.shape[0], dtype=np.int64)

        os.makedirs(os.path.dirname(self.npz_path), exist_ok=True)
        np.savez_compressed(
            self.npz_path,
            packed_codes=packed_codes,
            watermark_identities=watermark_identities,
            u=self.u,
        )

        # Build index from freshly generated codes
        self.packed_codes = packed_codes
        self.watermark_identities = watermark_identities
        self.bytes_per_code = int(packed_codes.shape[1])
        self.dim_bits = int(self.bytes_per_code * 8)
        self._build_index()

    def _load_cache(self) -> None:
        if self.npz_path is None:
            raise RuntimeError("Cache path not configured")

        data = np.load(self.npz_path)
        if "packed_codes" not in data or "watermark_identities" not in data or "u" not in data:
            raise ValueError(f"Invalid codebook cache format: {self.npz_path}")

        self.packed_codes = np.ascontiguousarray(data["packed_codes"], dtype=np.uint8)
        self.watermark_identities = np.asarray(data["watermark_identities"], dtype=np.int64)
        cached_u = int(data["u"])
        if cached_u != self.u:
            raise ValueError(f"Cached codebook u={cached_u} does not match requested u={self.u}")

        if self.packed_codes.shape[0] != self.watermark_identities.shape[0]:
            raise ValueError("Cached codebook metadata mismatch (rows)")

        self.bytes_per_code = int(self.packed_codes.shape[1])
        self.dim_bits = int(self.bytes_per_code * 8)
        self._load_or_build_index_from_cache()

    def _get_codeword_bits_by_index(self, index: int) -> np.ndarray:
        if self.packed_codes is None:
            raise RuntimeError("Codebook not loaded")
        packed_row = self.packed_codes[index]
        return self._unpack_bits_1d(packed_row, self.u)

    def find_closest_match(self, target_codeword: Sequence[int] | np.ndarray) -> tuple[list[int], int, int]:
        target_array = np.asarray(target_codeword, dtype=np.uint8)
        if target_array.ndim != 1 or target_array.shape[0] != self.u:
            raise ValueError(f"Target codeword must be a 1D sequence of length u={self.u}")

        if (
            self.index is None
            or self.watermark_identities is None
            or self.packed_codes is None
        ):
            raise RuntimeError("Codebook index not initialized")

        query = self._pack_bits_1d(target_array, self.u)
        distances, indices = self.index.search(query, 1)  # type: ignore
        idx = int(indices[0, 0])
        dist = int(distances[0, 0])
        watermark = int(self.watermark_identities[idx])
        bits = self._unpack_bits_1d(self.packed_codes[idx], self.u)
        return bits.tolist(), watermark, dist

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        if self.packed_codes is None or self.bytes_per_code == 0:
            raise RuntimeError("Packed codes unavailable for indexing")

        dim_bits = int(self.bytes_per_code * 8)
        self.dim_bits = dim_bits
        index = faiss.IndexBinaryFlat(dim_bits)  # type: ignore
        index.add(self.packed_codes)
        if self.index_path is not None:
            writer = getattr(faiss, "write_index_binary", None)
            if callable(writer):
                writer(index, self.index_path)
            else:  # fallback older faiss versions
                faiss.write_index(index, self.index_path)  # type: ignore
        self.index = index

    def _load_or_build_index_from_cache(self) -> None:
        assert self.index_path is not None
        try:
            reader = getattr(faiss, "read_index_binary", None)
            if callable(reader):
                index = reader(self.index_path)
            else:
                index = faiss.read_index(self.index_path)  # type: ignore
            if index.d != self.dim_bits:  # type: ignore[attr-defined]
                raise ValueError("Cached Faiss index dimension mismatch")
            if index.ntotal != self.packed_codes.shape[0]:  # type: ignore[attr-defined]
                raise ValueError("Cached Faiss index size mismatch")
            self.index = index
        except Exception:
            # rebuild
            self._build_index()

    # ------------------------------------------------------------------
    # Static packing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_bits_2d(bits_01: np.ndarray, u: int) -> np.ndarray:
        if bits_01.ndim != 2 or bits_01.shape[1] != u:
            raise ValueError("Expected (N, u) bit matrix")
        remainder = u % 8
        if remainder != 0:
            pad = 8 - remainder
            bits_01 = np.pad(bits_01, pad_width=((0, 0), (0, pad)), mode="constant", constant_values=0)
        return np.packbits(bits_01, axis=1, bitorder="big")

    @staticmethod
    def _pack_bits_1d(bits_01: np.ndarray, u: int) -> np.ndarray:
        arr = np.asarray(bits_01, dtype=np.uint8)
        if arr.ndim != 1 or arr.shape[0] != u:
            raise ValueError(f"Expected 1D array of length u={u}")
        remainder = u % 8
        if remainder != 0:
            pad = 8 - remainder
            arr = np.pad(arr, pad_width=(0, pad), mode="constant", constant_values=0)
        packed = np.packbits(arr, bitorder="big")
        return packed.reshape(1, -1)

    @staticmethod
    def _unpack_bits_1d(packed_row: np.ndarray, u: int) -> np.ndarray:
        bits = np.unpackbits(packed_row, bitorder="big")
        return bits[:u].astype(np.uint8)


# ---------------------------------------------------------------------------
# Concrete classes (retain previous naming)
# ---------------------------------------------------------------------------


class HammingCodebookRNG(HammingCodebookBase):
    def __init__(
        self,
        *,
        n: int,
        u: int,
        secret_key: str,
        cache_dir: Optional[str] = None,
        workers: int = 4,
        chunk_rows: Optional[int] = None,
    ) -> None:
        super().__init__(
            n=n,
            u=u,
            secret_key=secret_key,
            generator_cls=RNGCodewordGenerator,
            cache_dir=cache_dir,
            workers=workers,
            chunk_rows=chunk_rows,
        )


class HammingCodebookBCH(HammingCodebookBase):
    def __init__(
        self,
        *,
        n: int,
        u: int,
        secret_key: str,
        t: Optional[int] = None,
        m: Optional[int] = None,
        permute: bool = True,
        xor_mask: bool = True,
        cache_dir: Optional[str] = None,
        workers: int = 4,
        chunk_rows: Optional[int] = None,
    ) -> None:
        generator_kwargs = dict(t=t, m=m, permute=permute, xor_mask=xor_mask)
        super().__init__(
            n=n,
            u=u,
            secret_key=secret_key,
            generator_cls=BCHCodewordGenerator,
            generator_kwargs=generator_kwargs,
            cache_dir=cache_dir,
            workers=workers,
            chunk_rows=chunk_rows,
        )


# Backwards-compatible aliases
CodebookRNG = HammingCodebookRNG
CodebookBCH = HammingCodebookBCH
