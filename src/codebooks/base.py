"""
Concrete base codebook: generates and caches the fundamental codebook (NPZ).

Responsibilities:
- Deterministically generate (2^n, u) bit-code table using a pluggable generator (RNG/BCH)
- Store packed bits to minimize disk and RAM usage
- Provide direct lookup by index and a seeded random-row selection
- Handle cache detection/loading and on-demand generation

This base NPZ is intentionally decoupled from any search/index structure (e.g., Faiss).
Specialized codebooks (like Hamming) will consume this base artifact for performance indexing.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Type

import numpy as np

import src.paths as paths
from .generation import CodewordGenerator


class CodebookBase:
    """
    Fundamental codebook that persists the canonical bit table as a packed NPZ.

    - File name format:
        codebook_{algorithm_tag}_n{n}_u{u}_sk-{secret_key}.npz
      Example:
        codebook_rng_n24_u128_sk-25111915.npz

    - NPZ contents:
        - packed_codes: uint8 array of shape (2^n, ceil(u/8))
        - watermark_identities: int64 array [0 .. 2^n-1]
        - n: int
        - u: int
        - generator_tag: str
        - secret_key: str
    """

    def __init__(
        self,
        *,
        n: int,
        u: int,
        secret_key: str,
        generator_cls: Type[CodewordGenerator],
        generator_kwargs: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        workers: int = 4,
        chunk_rows: Optional[int] = None,
    ) -> None:
        self.n = int(n)
        self.u = int(u)
        self.secret_key = str(secret_key)
        self.workers = max(1, int(workers))
        self.chunk_rows = None if chunk_rows is None else max(1, int(chunk_rows))

        generator_kwargs = generator_kwargs or {}
        self.generator: CodewordGenerator = generator_cls(
            n=self.n,
            u=self.u,
            secret_key=self.secret_key,
            **generator_kwargs,
        )

        cache_root = cache_dir or paths.CODEBOOKS_CACHE_DIR
        os.makedirs(cache_root, exist_ok=True)
        self.cache_dir = cache_root

        # Paths and in-memory artifacts
        self.filename_base = self._build_filename_base()
        self.npz_path = os.path.join(self.cache_dir, f"{self.filename_base}.npz")

        self.packed_codes: Optional[np.ndarray] = None
        self.watermark_identities: Optional[np.ndarray] = None
        self.bytes_per_code: int = 0
        self.dim_bits: int = 0  # bytes_per_code * 8

        self._load_or_build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def table_size(self) -> int:
        return 2 ** self.n

    def get_random_row(self, seed: int) -> tuple[list[int], int]:
        """
        Select a deterministic random row (watermark identity) using the provided seed,
        and return (bit_vector_as_list, watermark_identity).
        """
        if self.packed_codes is None:
            raise RuntimeError("Codebook not loaded")
        rng = np.random.default_rng(seed)
        watermark = int(rng.integers(0, self.table_size))
        bits = self.get_codeword_by_index(watermark)
        return bits.tolist(), watermark

    def get_codeword_by_index(self, index: int) -> np.ndarray:
        """
        Direct lookup by watermark identity (row index). Returns a 1D array of length u with 0/1 values.
        """
        if self.packed_codes is None:
            raise RuntimeError("Codebook not loaded")
        if index < 0 or index >= self.packed_codes.shape[0]:
            raise IndexError(f"Index {index} out of range for table size {self.packed_codes.shape[0]}")
        packed_row = self.packed_codes[index]
        return self._unpack_bits_1d(packed_row, self.u)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filename_base(self) -> str:
        # Use hyphen for the secret-key portion as requested
        return f"codebook_{self.generator.algorithm_tag}_n{self.n}_u{self.u}_sk-{self.secret_key}"

    def _cache_exists(self) -> bool:
        return os.path.exists(self.npz_path)

    def _load_or_build(self) -> None:
        if self._cache_exists():
            self._load_cache()
            return

        # Generate raw 0/1 bits (N, u)
        raw_codewords = self.generator.generate(
            workers=self.workers,
            chunk_rows=self.chunk_rows,
        )
        # Persist packed representation
        self._store_generated_codewords(raw_codewords)
        # Load into memory for use
        self._load_cache()

    def _store_generated_codewords(self, codewords_bits_01: np.ndarray) -> None:
        if codewords_bits_01.ndim != 2 or codewords_bits_01.shape[1] != self.u:
            raise ValueError(f"Expected raw bit matrix of shape (2^n, u), got {codewords_bits_01.shape}")

        packed_codes = self._pack_bits_2d(codewords_bits_01, self.u)
        watermark_identities = np.arange(codewords_bits_01.shape[0], dtype=np.int64)

        os.makedirs(os.path.dirname(self.npz_path), exist_ok=True)
        np.savez_compressed(
            self.npz_path,
            packed_codes=packed_codes,
            watermark_identities=watermark_identities,
            n=self.n,
            u=self.u,
            generator_tag=self.generator.algorithm_tag,
            secret_key=np.array(self.secret_key),
        )

    def _load_cache(self) -> None:
        data = np.load(self.npz_path, allow_pickle=True)
        required = {"packed_codes", "watermark_identities", "n", "u", "generator_tag", "secret_key"}
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Invalid codebook cache: missing keys {missing} in {self.npz_path}")

        cached_n = int(data["n"])
        cached_u = int(data["u"])
        cached_tag = str(data["generator_tag"])
        cached_secret = str(data["secret_key"])

        if cached_n != self.n or cached_u != self.u:
            raise ValueError(f"Cached n/u mismatch: cached (n={cached_n}, u={cached_u}) != requested (n={self.n}, u={self.u})")
        if cached_tag != self.generator.algorithm_tag:
            raise ValueError(f"Generator tag mismatch: cached '{cached_tag}' != requested '{self.generator.algorithm_tag}'")
        if cached_secret != self.secret_key:
            raise ValueError("Secret key mismatch for cached codebook")

        self.packed_codes = np.ascontiguousarray(data["packed_codes"], dtype=np.uint8)
        self.watermark_identities = np.asarray(data["watermark_identities"], dtype=np.int64)

        if self.packed_codes.shape[0] != self.watermark_identities.shape[0]:
            raise ValueError("Cached codebook metadata mismatch (row count)")

        self.bytes_per_code = int(self.packed_codes.shape[1])
        self.dim_bits = int(self.bytes_per_code * 8)

    # ------------------------------------------------------------------
    # Static packing helpers (shared with specialized codebooks)
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
