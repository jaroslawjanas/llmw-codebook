"""
Deterministic codeword generation strategies.

Provides pluggable generator classes that produce packed bit arrays (shape: N, ceil(u/8))
for watermark identities, where N = 2 ** n.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Optional, Tuple

import numpy as np
from tqdm import tqdm

def _pack_bits_2d(bits_01: np.ndarray, u: int) -> np.ndarray:
    if bits_01.ndim != 2 or bits_01.shape[1] != u:
        raise ValueError("Expected (N, u) bit matrix")
    remainder = u % 8
    if remainder != 0:
        pad = 8 - remainder
        bits_01 = np.pad(bits_01, pad_width=((0, 0), (0, pad)), mode="constant", constant_values=0)
    return np.packbits(bits_01, axis=1, bitorder="big")

__all__ = [
    "CodewordGenerator",
    "RNGCodewordGenerator",
    "BCHCodewordGenerator",
    "LinearCodeGenerator",
]


def _derive_seed(secret_key: str, tag: str) -> int:
    digest = hashlib.sha256(f"{secret_key}|{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _int_to_bits(value: int, length: int) -> np.ndarray:
    bits = np.zeros(length, dtype=np.uint8)
    for idx in range(length):
        bits[idx] = (value >> (length - 1 - idx)) & 1
    return bits


def _build_message_matrix(start: int, end: int, n: int) -> np.ndarray:
    """Build (end-start, n) matrix where row i is the n-bit binary representation of (start+i)."""
    identities = np.arange(start, end, dtype=np.uint32)
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint32)
    return ((identities[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(np.uint8)


@dataclass
class CodewordGenerator(ABC):
    n: int
    u: int
    secret_key: str

    @property
    @abstractmethod
    def algorithm_tag(self) -> str:
        ...

    @abstractmethod
    def generate(self, *, workers: int, chunk_rows: Optional[int]) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# RNG baseline
# ---------------------------------------------------------------------------


def _rng_chunk_worker(args: Tuple[int, int, int, str]) -> Tuple[int, np.ndarray]:
    start, end, u, secret_key = args
    rows = end - start
    bits = np.empty((rows, u), dtype=np.uint8)
    for offset, i in enumerate(range(start, end)):
        digest = hashlib.sha256(f"{secret_key}:{i}".encode("utf-8")).digest()
        identity_seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(identity_seed)
        bits[offset] = rng.integers(0, 2, size=u, dtype=np.uint8)
    return start, bits


@dataclass
class RNGCodewordGenerator(CodewordGenerator):
    @property
    def algorithm_tag(self) -> str:
        return "rng"

    def generate(self, *, workers: int, chunk_rows: Optional[int]) -> np.ndarray:
        table_size = 2 ** self.n
        bytes_per_code = (self.u + 7) // 8
        chunk_rows = chunk_rows or self._default_chunk_rows()
        chunks = [(start, min(table_size, start + chunk_rows)) for start in range(0, table_size, chunk_rows)]

        # Pre-allocate PACKED array (N, ceil(u/8))
        packed_bits = np.empty((table_size, bytes_per_code), dtype=np.uint8)

        args_iter = ((start, end, self.u, self.secret_key) for start, end in chunks)
        with Pool(processes=workers) as pool:
            for start, chunk_bits in tqdm(
                pool.imap_unordered(_rng_chunk_worker, args_iter, chunksize=1),
                total=len(chunks),
                desc="Generating RNG codewords",
                unit="chunk",
            ):
                end = start + chunk_bits.shape[0]
                # Pack each chunk immediately to minimize memory
                packed_bits[start:end] = _pack_bits_2d(chunk_bits, self.u)

        return packed_bits

    def _default_chunk_rows(self) -> int:
        target_bytes = 4 * 1024 * 1024
        rows = max(4096, min(262144, target_bytes // max(1, self.u)))
        return int(rows)


# ---------------------------------------------------------------------------
# BCH generator (optional dependency on galois)
# ---------------------------------------------------------------------------

# Valid k values for BCH codes, keyed by m where n0 = 2^m - 1.
# Each list is sorted by k descending (largest message capacity first).
# The actual t for each k is determined by galois.BCH(n0, k=k) at runtime.
_BCH_VALID_K: dict[int, list[int]] = {
    3: [4, 1],
    4: [11, 7, 5, 1],
    5: [26, 21, 16, 11, 6, 1],
    6: [57, 51, 45, 39, 36, 30, 24, 18, 16, 10, 7, 1],
    7: [120, 113, 106, 99, 92, 85, 78, 71, 64, 57, 50, 43, 36, 29, 22, 15, 8, 1],
    8: [247, 239, 231, 223, 215, 207, 199, 191, 187, 179, 171, 163, 155, 147, 139, 131, 123, 115, 107, 99, 91, 87, 79, 71, 63, 55, 47, 45, 37, 29, 21, 13, 9, 1],
    9: [502, 493, 484, 475, 466, 457, 448, 439, 430, 421, 412, 403, 394, 385, 376, 367, 358, 349, 340, 331, 322, 313, 304, 295, 286, 277, 268, 259, 250, 241, 238, 229, 220, 211, 202, 193, 184, 175, 166, 157, 148, 139, 130, 121, 112, 103, 94, 85, 76, 67, 58, 49, 40, 31, 28, 19, 10, 1],
}


def _find_smallest_valid_k(n0: int, min_k: int) -> Optional[int]:
    """Find the smallest valid BCH k >= min_k for a given n0."""
    m = (n0 + 1).bit_length() - 1
    if m not in _BCH_VALID_K:
        return None
    result = None
    for k in _BCH_VALID_K[m]:  # sorted descending
        if k >= min_k:
            result = k
    return result


_BCH: Any = None
_BCH_k: Optional[int] = None
_BCH_watermark_bits: Optional[int] = None
_BCH_u: Optional[int] = None
_BCH_padding: int = 0
_BCH_secret_key: str = ""


def _bch_pool_init(n: int, u: int, m: int, k: int, padding: int, secret_key: str) -> None:
    global _BCH, _BCH_k, _BCH_watermark_bits, _BCH_u, _BCH_padding, _BCH_secret_key

    import galois  # type: ignore

    _BCH_watermark_bits = int(n)
    _BCH_u = int(u)
    _BCH_padding = int(padding)
    _BCH_secret_key = secret_key

    n0 = (2**m) - 1
    _BCH = galois.BCH(n0, k=k)
    _BCH_k = int(k)


def _bch_chunk_worker(args: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    global _BCH, _BCH_k, _BCH_watermark_bits, _BCH_u, _BCH_padding, _BCH_secret_key
    start, end = args
    if (
        _BCH is None
        or _BCH_k is None
        or _BCH_watermark_bits is None
        or _BCH_u is None
    ):
        raise RuntimeError("BCH worker not initialized")

    rows = end - start
    bits = np.empty((rows, _BCH_u), dtype=np.uint8)

    for offset, i in enumerate(range(start, end)):
        msg = np.zeros(_BCH_k, dtype=np.uint8)
        msg[:_BCH_watermark_bits] = _int_to_bits(i, _BCH_watermark_bits)
        if _BCH_k > _BCH_watermark_bits:
            pad_seed = _derive_seed(_BCH_secret_key, f"bch-pad-{i}")
            pad_rng = np.random.default_rng(pad_seed)
            msg[_BCH_watermark_bits:] = pad_rng.integers(0, 2, size=_BCH_k - _BCH_watermark_bits, dtype=np.uint8)

        cw_full = np.array(_BCH.encode(msg), dtype=np.uint8)
        if _BCH_padding > 0:
            cw_u = np.zeros(_BCH_u, dtype=np.uint8)
            cw_u[:len(cw_full)] = cw_full
        else:
            cw_u = cw_full[:_BCH_u]

        bits[offset] = cw_u

    return start, bits


@dataclass
class BCHCodewordGenerator(CodewordGenerator):
    m: Optional[int] = None

    @property
    def algorithm_tag(self) -> str:
        m, k, _, _ = self._resolve_bch_parameters()
        return f"bch_m{m}_k{k}"

    def generate(self, *, workers: int, chunk_rows: Optional[int]) -> np.ndarray:
        resolved_m, resolved_k, padding, _ = self._resolve_bch_parameters()
        table_size = 2 ** self.n
        bytes_per_code = (self.u + 7) // 8
        chunk_rows = chunk_rows or self._default_chunk_rows()
        chunks = [(start, min(table_size, start + chunk_rows)) for start in range(0, table_size, chunk_rows)]

        # Pre-allocate PACKED array (N, ceil(u/8))
        packed_bits = np.empty((table_size, bytes_per_code), dtype=np.uint8)

        with Pool(
            processes=workers,
            initializer=_bch_pool_init,
            initargs=(self.n, self.u, resolved_m, resolved_k, padding, self.secret_key),
        ) as pool:
            for start, chunk_bits in tqdm(
                pool.imap_unordered(_bch_chunk_worker, chunks, chunksize=1),
                total=len(chunks),
                desc="Generating BCH codewords",
                unit="chunk",
            ):
                end = start + chunk_bits.shape[0]
                # Pack each chunk immediately to minimize memory
                packed_bits[start:end] = _pack_bits_2d(chunk_bits, self.u)

        return packed_bits

    def _default_chunk_rows(self) -> int:
        target_bytes = 4 * 1024 * 1024
        rows = max(1024, min(65536, target_bytes // max(1, self.u)))
        return int(rows)

    def _resolve_bch_parameters(self) -> Tuple[int, int, int, int]:
        """Resolve BCH parameters using lookup table. Returns (m, k, padding, puncturing)."""
        if self.m is not None:
            m = int(self.m)
            n0 = (2**m) - 1
            k = _find_smallest_valid_k(n0, min_k=self.n)
            if k is None:
                raise ValueError(
                    f"No valid BCH code for m={m} (n0={n0}) with k >= {self.n}. "
                    f"Try a larger --bch-m."
                )
            padding = max(0, self.u - n0)
            puncturing = max(0, n0 - self.u)
            return m, k, padding, puncturing

        # Auto-select: find all viable (m, n0, k)
        viable = []
        for m_candidate in sorted(_BCH_VALID_K.keys()):
            n0 = (2**m_candidate) - 1
            k = _find_smallest_valid_k(n0, min_k=self.n)
            if k is not None:
                viable.append((m_candidate, n0, k))

        if not viable:
            raise ValueError(
                f"No valid BCH code found for n={self.n}. "
                f"Specify --bch-m manually for m > 9."
            )

        # Prefer largest n0 <= u (minimize padding), else smallest n0 > u (minimize puncturing)
        padding_opts = [(m, n0, k) for m, n0, k in viable if n0 <= self.u]
        puncturing_opts = [(m, n0, k) for m, n0, k in viable if n0 > self.u]

        if padding_opts:
            m, n0, k = max(padding_opts, key=lambda x: x[1])
        else:
            m, n0, k = min(puncturing_opts, key=lambda x: x[1])

        padding = max(0, self.u - n0)
        puncturing = max(0, n0 - self.u)
        return m, k, padding, puncturing


# ---------------------------------------------------------------------------
# Random linear code over GF(2)
# ---------------------------------------------------------------------------


@dataclass
class LinearCodeGenerator(CodewordGenerator):
    """
    Random linear code over GF(2) with affine offset.

    For each identity i in [0, 2^n):
        codeword(i) = (bits_of(i) @ G) mod 2  XOR  r

    where G is a random (n, u) binary generator matrix and r is a random u-bit
    affine offset vector, both derived deterministically from secret_key.
    """

    @property
    def algorithm_tag(self) -> str:
        return "linear"

    def generate(self, *, workers: int, chunk_rows: Optional[int] = None) -> np.ndarray:
        table_size = 2 ** self.n
        bytes_per_code = (self.u + 7) // 8
        chunk_rows = chunk_rows or self._default_chunk_rows()

        # Derive G and r deterministically from secret_key
        seed_G = _derive_seed(self.secret_key, "linear-G")
        seed_r = _derive_seed(self.secret_key, "linear-r")

        rng_G = np.random.default_rng(seed_G)
        G = rng_G.integers(0, 2, size=(self.n, self.u), dtype=np.uint8)

        rng_r = np.random.default_rng(seed_r)
        r = rng_r.integers(0, 2, size=self.u, dtype=np.uint8)

        # Pre-allocate packed output
        packed_bits = np.empty((table_size, bytes_per_code), dtype=np.uint8)

        chunks = [(start, min(table_size, start + chunk_rows))
                  for start in range(0, table_size, chunk_rows)]

        for start, end in tqdm(chunks, desc="Generating linear codewords", unit="chunk"):
            M = _build_message_matrix(start, end, self.n)
            codewords = np.asarray((M @ G) % 2, dtype=np.uint8)
            codewords ^= r
            packed_bits[start:end] = _pack_bits_2d(codewords, self.u)

        return packed_bits

    def _default_chunk_rows(self) -> int:
        target_bytes = 32 * 1024 * 1024
        rows = max(4096, min(1048576, target_bytes // max(1, self.u)))
        return int(rows)
