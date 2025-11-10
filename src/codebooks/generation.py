"""
Deterministic codeword generation strategies.

Provides pluggable generator classes that produce raw bit arrays (shape: N, u)
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

__all__ = [
    "CodewordGenerator",
    "RNGCodewordGenerator",
    "BCHCodewordGenerator",
]


def _derive_seed(secret_key: str, tag: str) -> int:
    digest = hashlib.sha256(f"{secret_key}|{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _int_to_bits(value: int, length: int) -> np.ndarray:
    bits = np.zeros(length, dtype=np.uint8)
    for idx in range(length):
        bits[idx] = (value >> (length - 1 - idx)) & 1
    return bits


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
        chunk_rows = chunk_rows or self._default_chunk_rows()
        chunks = [(start, min(table_size, start + chunk_rows)) for start in range(0, table_size, chunk_rows)]

        bits = np.empty((table_size, self.u), dtype=np.uint8)

        args_iter = ((start, end, self.u, self.secret_key) for start, end in chunks)
        with Pool(processes=workers) as pool:
            for start, chunk_bits in tqdm(
                pool.imap_unordered(_rng_chunk_worker, args_iter, chunksize=1),
                total=len(chunks),
                desc="Generating RNG codewords",
                unit="chunk",
            ):
                end = start + chunk_bits.shape[0]
                bits[start:end] = chunk_bits

        return bits

    def _default_chunk_rows(self) -> int:
        target_bytes = 4 * 1024 * 1024
        rows = max(4096, min(262144, target_bytes // max(1, self.u)))
        return int(rows)


# ---------------------------------------------------------------------------
# BCH generator (optional dependency on galois)
# ---------------------------------------------------------------------------

_BCH: Any = None
_BCH_perm: Optional[np.ndarray] = None
_BCH_mask: Optional[np.ndarray] = None
_BCH_k: Optional[int] = None
_BCH_watermark_bits: Optional[int] = None
_BCH_u: Optional[int] = None


def _bch_pool_init(n: int, u: int, secret_key: str, m: int, t: int, permute: bool, xor_mask: bool) -> None:
    global _BCH, _BCH_perm, _BCH_mask, _BCH_k, _BCH_watermark_bits, _BCH_u

    import galois  # type: ignore

    _BCH_watermark_bits = int(n)
    _BCH_u = int(u)

    n0 = (2**m) - 1
    d = 2 * int(t) + 1

    _BCH = galois.BCH(n0, d=d)
    _BCH_k = int(_BCH.k)

    if permute:
        perm_rng = np.random.default_rng(_derive_seed(secret_key, "bch-permutation"))
        permutation = np.arange(n0, dtype=np.int64)
        perm_rng.shuffle(permutation)
        _BCH_perm = permutation
    else:
        _BCH_perm = np.arange(n0, dtype=np.int64)

    if xor_mask:
        mask_rng = np.random.default_rng(_derive_seed(secret_key, "bch-mask"))
        _BCH_mask = mask_rng.integers(0, 2, size=_BCH_u, dtype=np.uint8)
    else:
        _BCH_mask = None


def _bch_chunk_worker(args: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    global _BCH, _BCH_perm, _BCH_mask, _BCH_k, _BCH_watermark_bits, _BCH_u
    start, end = args
    if (
        _BCH is None
        or _BCH_perm is None
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

        cw_full = np.array(_BCH.encode(msg), dtype=np.uint8)
        cw_perm = cw_full[_BCH_perm]
        cw_u = cw_perm[:_BCH_u]

        if _BCH_mask is not None:
            cw_u = np.bitwise_xor(cw_u, _BCH_mask)

        bits[offset] = cw_u

    return start, bits


@dataclass
class BCHCodewordGenerator(CodewordGenerator):
    t: Optional[int] = None
    m: Optional[int] = None
    permute: bool = True
    xor_mask: bool = True

    @property
    def algorithm_tag(self) -> str:
        t_tag = str(self.t) if self.t is not None else "auto"
        m_tag = str(self.m) if self.m is not None else "auto"
        return f"bch_m{m_tag}_t{t_tag}"

    def generate(self, *, workers: int, chunk_rows: Optional[int]) -> np.ndarray:
        resolved_m, resolved_t, _ = self._resolve_bch_parameters()
        table_size = 2 ** self.n
        chunk_rows = chunk_rows or self._default_chunk_rows()
        chunks = [(start, min(table_size, start + chunk_rows)) for start in range(0, table_size, chunk_rows)]

        bits = np.empty((table_size, self.u), dtype=np.uint8)

        with Pool(
            processes=workers,
            initializer=_bch_pool_init,
            initargs=(self.n, self.u, self.secret_key, resolved_m, resolved_t, self.permute, self.xor_mask),
        ) as pool:
            for start, chunk_bits in tqdm(
                pool.imap_unordered(_bch_chunk_worker, chunks, chunksize=1),
                total=len(chunks),
                desc="Generating BCH codewords",
                unit="chunk",
            ):
                end = start + chunk_bits.shape[0]
                bits[start:end] = chunk_bits

        return bits

    def _default_chunk_rows(self) -> int:
        target_bytes = 4 * 1024 * 1024
        rows = max(1024, min(65536, target_bytes // max(1, self.u)))
        return int(rows)

    def _resolve_bch_parameters(self) -> Tuple[int, int, int]:
        try:
            import galois  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("BCH codebook generation requires the 'galois' package") from exc

        if self.m is None:
            m = 2
            while (2**m - 1) < self.u:
                m += 1
        else:
            m = int(self.m)
            if (2**m - 1) < self.u:
                raise ValueError(f"Provided m={m} yields n0={2**m - 1} < u={self.u}")

        n0 = (2**m) - 1

        if self.t is not None:
            d = 2 * int(self.t) + 1
            bch = galois.BCH(n0, d=d)
            if bch.k < self.n:
                raise ValueError(f"Provided t={self.t} yields k={bch.k} < n={self.n}")
            resolved_t = int(self.t)
            resolved_k = int(bch.k)
        else:
            resolved_t = None
            resolved_k = None
            for t_candidate in range(min(n0 // 2, 64), 0, -1):
                try:
                    d = 2 * t_candidate + 1
                    bch_candidate = galois.BCH(n0, d=d)
                    if bch_candidate.k >= self.n:
                        resolved_t = t_candidate
                        resolved_k = int(bch_candidate.k)
                        break
                except Exception:
                    continue
            if resolved_t is None or resolved_k is None:
                raise RuntimeError(
                    f"Unable to resolve BCH parameters for n0={n0}, n={self.n}. "
                    "Increase m or decrease n."
                )

        return m, resolved_t, resolved_k
