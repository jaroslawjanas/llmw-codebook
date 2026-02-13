"""
Levenshtein-distance specialized codebook that consumes a base NPZ and builds
Numba-JIT-compiled Myers' bit-parallel search over packed uint64 arrays.

Responsibilities:
- Consume CodebookBase (canonical codewords stored as packed bits in NPZ)
- Build and cache uint64-packed codebook for Numba-based Levenshtein search
- Provide nearest-neighbor search by edit distance (insertions, deletions, substitutions)
- Support both single-query and batch-query search
- Delegate non-search helpers (like get_random_row) to the base

File naming:
- Base:        codebook_{algo}_n{n}_u{u}_sk-<secret-key>.npz
- Levenshtein: codebook_{algo}_n{n}_u{u}_sk-<secret-key>_metric-levenshtein.npz
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import numba
    from numba import njit, prange
except ModuleNotFoundError as exc:
    raise ImportError(
        "Numba is required for Levenshtein codebooks. Install with `pip install numba`."
    ) from exc

from .base import CodebookBase

__all__ = ["CodebookLevenshtein"]


# ---------------------------------------------------------------------------
# Numba-JIT kernels (module-level, compiled once on first import)
# ---------------------------------------------------------------------------
# Bit packing convention: **little-endian within each uint64 word**.
#   bit[i] is stored at position (i % 64) of word (i // 64).
#   i.e. LSB of word 0 = bit 0, MSB of word 0 = bit 63, LSB of word 1 = bit 64, ...
# This aligns with Myers' original formulation where left-shift moves to higher
# pattern positions, and the boundary condition sets bit 0 (LSB).
# ---------------------------------------------------------------------------


@njit(cache=True)
def _pack_bits_to_uint64(bits: np.ndarray, length: int) -> np.ndarray:
    """Pack a 1D uint8 {0,1} array into LE uint64 words.

    Args:
        bits: 1D array of 0/1 values, length >= ``length``.
        length: number of valid bits to pack.

    Returns:
        1D uint64 array of shape (ceil(length / 64),).
    """
    num_words = (length + 63) // 64
    packed = np.zeros(num_words, dtype=np.uint64)
    for i in range(length):
        if bits[i]:
            packed[i // 64] |= np.uint64(1) << np.uint64(i % 64)
    return packed


@njit(cache=True)
def _myers_levenshtein(
    query_packed: np.ndarray,
    query_len: int,
    text_packed: np.ndarray,
    text_len: int,
) -> int:
    """Compute Levenshtein edit distance between two binary sequences.

    Uses Myers' bit-parallel algorithm (1999) adapted for binary alphabet.
    Multi-word support for sequences longer than 64 bits.

    The *pattern* is the query (determines Peq and block count).
    The *text* is the codebook entry (scanned character by character).

    For binary alphabet {0, 1}:
        Peq[1][b] = query_packed[b]   (bit j set where query bit b*64+j == 1)
        Peq[0][b] = ~query_packed[b]  (complement)

    Args:
        query_packed: LE-packed query, shape (num_query_words,) uint64.
        query_len: actual query length in bits.
        text_packed: LE-packed codebook entry, shape (num_text_words,) uint64.
        text_len: actual codebook entry length in bits.

    Returns:
        Levenshtein edit distance (int).
    """
    m = query_len  # pattern length
    n = text_len   # text length

    if m == 0:
        return n
    if n == 0:
        return m

    num_blocks = (m + 63) // 64

    # Mask for the last block (only the lowest (m % 64) bits are valid).
    last_bits = m % 64
    if last_bits == 0:
        last_mask = ~np.uint64(0)  # all bits valid
    else:
        last_mask = (np.uint64(1) << np.uint64(last_bits)) - np.uint64(1)

    # Initialize: VP = all-ones (within valid bits), VN = all-zeros, score = m
    Pv = np.empty(num_blocks, dtype=np.uint64)
    Mv = np.zeros(num_blocks, dtype=np.uint64)
    for b in range(num_blocks):
        Pv[b] = ~np.uint64(0)
    # Mask last block of Pv
    Pv[num_blocks - 1] &= last_mask

    score = m

    # High-bit mask for score update: the bit at position (last_bits - 1) in last block,
    # or bit 63 if last_bits == 0.
    if last_bits == 0:
        high_bit = np.uint64(1) << np.uint64(63)
    else:
        high_bit = np.uint64(1) << np.uint64(last_bits - 1)

    # Process each character in the text
    for i in range(n):
        # Extract text[i] from packed representation
        word_idx = i // 64
        bit_pos = i % 64
        text_bit = (text_packed[word_idx] >> np.uint64(bit_pos)) & np.uint64(1)

        # Build Eq for this text character.
        # Eq[b] = Peq[text_bit][b]
        # For binary alphabet: Peq[1] = query_packed, Peq[0] = ~query_packed
        # Carry propagation across blocks for the addition (Eq & Pv) + Pv
        carry = np.uint64(0)
        prev_ph_msb = np.uint64(0)
        prev_mh_msb = np.uint64(0)
        for b in range(num_blocks):
            if text_bit:
                Eq = query_packed[b]
            else:
                Eq = ~query_packed[b]

            # Mask last block
            if b == num_blocks - 1:
                Eq &= last_mask

            Xv = Eq | Mv[b]

            # Xh = (((Eq & Pv[b]) + Pv[b] + carry) ^ Pv[b]) | Eq
            Eq_and_Pv = Eq & Pv[b]
            sum_val = Eq_and_Pv + Pv[b] + carry
            # Detect carry out: carry occurs if sum wrapped around
            if carry:
                new_carry = np.uint64(1) if sum_val <= Eq_and_Pv else np.uint64(0)
            else:
                new_carry = np.uint64(1) if sum_val < Eq_and_Pv else np.uint64(0)
            carry = new_carry

            Xh = (sum_val ^ Pv[b]) | Eq

            Ph = Mv[b] | ~(Xh | Pv[b])
            Mh = Pv[b] & Xh

            # Mask last block for Ph and Mh
            if b == num_blocks - 1:
                Ph &= last_mask
                Mh &= last_mask

            # Score update from the high bit of the last block
            if b == num_blocks - 1:
                if Ph & high_bit:
                    score += 1
                if Mh & high_bit:
                    score -= 1

            # Shift Ph and Mh left by 1 (with carry between blocks)
            # Save the MSB before shifting (it carries into the next block)
            ph_msb = (Ph >> np.uint64(63)) & np.uint64(1)
            mh_msb = (Mh >> np.uint64(63)) & np.uint64(1)

            Ph_shifted = Ph << np.uint64(1)
            Mh_shifted = Mh << np.uint64(1)

            if b == 0:
                # Boundary condition: set bit 0 of Ph (LSB of first block)
                Ph_shifted |= np.uint64(1)
            else:
                # Carry in from previous block
                Ph_shifted |= prev_ph_msb
                Mh_shifted |= prev_mh_msb

            prev_ph_msb = ph_msb
            prev_mh_msb = mh_msb

            # Update Pv and Mv for next iteration
            Pv[b] = Mh_shifted | ~(Xv | Ph_shifted)
            Mv[b] = Ph_shifted & Xv

            # Mask last block
            if b == num_blocks - 1:
                Pv[b] &= last_mask
                Mv[b] &= last_mask

    return score


@njit(cache=True, parallel=True)
def _search_codebook(
    query_packed: np.ndarray,
    query_len: int,
    codebook_packed: np.ndarray,
    code_len: int,
) -> Tuple[int, int]:
    """Find the closest codebook entry to a single query.

    Parallelized across codebook chunks using prange for multi-core utilization.

    Returns:
        (best_index, best_distance)
    """
    N = codebook_packed.shape[0]
    CHUNK = 65536
    num_chunks = (N + CHUNK - 1) // CHUNK

    chunk_best_idx = np.empty(num_chunks, dtype=np.int64)
    chunk_best_dist = np.empty(num_chunks, dtype=np.int32)

    for c in prange(num_chunks):
        start = c * CHUNK
        end = min(start + CHUNK, N)
        local_best_idx = start
        local_best_dist = np.int32(query_len + code_len + 1)

        for i in range(start, end):
            d = _myers_levenshtein(query_packed, query_len, codebook_packed[i], code_len)
            if d < local_best_dist:
                local_best_dist = np.int32(d)
                local_best_idx = i

        chunk_best_idx[c] = local_best_idx
        chunk_best_dist[c] = local_best_dist

    # Reduce across chunks (sequential, num_chunks is small)
    best_idx = np.int64(chunk_best_idx[0])
    best_dist = np.int32(chunk_best_dist[0])
    for c in range(1, num_chunks):
        if chunk_best_dist[c] < best_dist:
            best_dist = chunk_best_dist[c]
            best_idx = chunk_best_idx[c]

    return int(best_idx), int(best_dist)


@njit(cache=True, parallel=True)
def _search_codebook_batch(
    queries_packed: np.ndarray,
    query_lens: np.ndarray,
    codebook_packed: np.ndarray,
    code_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find closest codebook entry for each query in a batch.

    Parallelized across queries using prange. Each query scans the entire
    codebook sequentially (but different queries run on different cores).

    Args:
        queries_packed: (B, max_words) uint64, padded queries.
        query_lens: (B,) int32, actual length of each query.
        codebook_packed: (N, num_words) uint64, codebook entries.
        code_len: codeword length in bits (u).

    Returns:
        (best_indices, best_distances) — each shape (B,).
    """
    B = queries_packed.shape[0]
    N = codebook_packed.shape[0]

    best_indices = np.empty(B, dtype=np.int64)
    best_distances = np.empty(B, dtype=np.int32)

    for q in prange(B):
        q_packed = queries_packed[q]
        q_len = int(query_lens[q])
        # Trim query packed to actual number of words needed
        q_num_words = (q_len + 63) // 64
        q_trimmed = q_packed[:q_num_words]

        local_best_idx = np.int64(0)
        local_best_dist = np.int32(q_len + code_len + 1)

        for i in range(N):
            d = _myers_levenshtein(q_trimmed, q_len, codebook_packed[i], code_len)
            if d < local_best_dist:
                local_best_dist = np.int32(d)
                local_best_idx = np.int64(i)

        best_indices[q] = local_best_idx
        best_distances[q] = local_best_dist

    return best_indices, best_distances


@njit(cache=True, parallel=True)
def _batch_convert_uint8_to_uint64(
    packed_uint8: np.ndarray, u: int, num_words: int
) -> np.ndarray:
    """Convert codebook from uint8 big-endian packed to uint64 LE-packed.

    Reads bit i from uint8 array (byte i//8, bit 7 - i%8) and sets
    bit i%64 of uint64 word i//64.

    Args:
        packed_uint8: (N, ceil(u/8)) uint8 array from CodebookBase.
        u: codeword length in bits.
        num_words: ceil(u/64), number of uint64 words per row.

    Returns:
        (N, num_words) uint64 array.
    """
    N = packed_uint8.shape[0]
    result = np.zeros((N, num_words), dtype=np.uint64)

    for row in prange(N):
        for i in range(u):
            byte_idx = i // 8
            bit_in_byte = 7 - (i % 8)  # big-endian within byte
            bit_val = (packed_uint8[row, byte_idx] >> bit_in_byte) & 1
            if bit_val:
                word_idx = i // 64
                bit_in_word = i % 64  # LE within uint64 word
                result[row, word_idx] |= np.uint64(1) << np.uint64(bit_in_word)

    return result


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------


class CodebookLevenshtein:
    """
    Levenshtein-distance codebook built on top of a CodebookBase.

    Uses Numba-JIT-compiled Myers' bit-parallel algorithm optimized for
    binary alphabet, operating on packed uint64 arrays.

    Construction:
        base = CodebookBase(...)
        levenshtein = CodebookLevenshtein(base)

    Usage:
        # Select a deterministic random watermark row
        bits, watermark = levenshtein.get_random_row(seed)

        # Find closest match by Levenshtein edit distance (single query)
        bits, watermark, distance = levenshtein.find_closest_match(target_bits)

        # Find closest matches for multiple queries (batch)
        indices, watermarks, distances = levenshtein.find_closest_match_batch(queries)
    """

    def __init__(self, base: CodebookBase, *, cache_dir: Optional[str] = None) -> None:
        if base.packed_codes is None:
            raise RuntimeError("Base codebook not loaded")

        self.base = base
        self.cache_dir = cache_dir or base.cache_dir

        self.filename_base = f"{self.base.filename_base}_metric-levenshtein"
        self.index_path = os.path.join(self.cache_dir, f"{self.filename_base}.npz")

        self.codebook_packed: Optional[np.ndarray] = None
        self.num_words: int = (base.u + 63) // 64
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

    def get_random_row(self, seed: int) -> Tuple[List[int], int]:
        return self.base.get_random_row(seed)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_closest_match(
        self, target_codeword: Sequence[int] | np.ndarray
    ) -> Tuple[List[int], int, int]:
        """
        Locate the closest codeword by Levenshtein edit distance.
        Accepts target codeword as a 1D sequence of 0/1 of ANY length (variable length allowed).
        Returns (closest_codeword_bits, watermark_identity, distance_value).

        - closest_codeword_bits: the matched codebook codeword (length = self.u)
        - watermark_identity: watermark id (row index in base)
        - distance_value: raw edit distance (insertions + deletions + substitutions)
        """
        if self.codebook_packed is None or self.base.watermark_identities is None:
            raise RuntimeError("Levenshtein codebook not initialized")

        target_array = np.asarray(target_codeword, dtype=np.uint8).reshape(-1)
        # Be robust to accidental non-binary values by mapping to {0,1}
        if target_array.size == 0:
            # Empty query: distance = u for every codebook entry; return first.
            idx = 0
            watermark = int(self.base.watermark_identities[0])
            bits = self.base.get_codeword_by_index(0)
            return bits.tolist(), watermark, self.u
        else:
            target_array = (target_array != 0).astype(np.uint8)

        query_len = int(target_array.shape[0])
        query_packed = _pack_bits_to_uint64(target_array, query_len)

        idx, dist = _search_codebook(query_packed, query_len, self.codebook_packed, self.u)

        idx = int(idx)
        watermark = int(self.base.watermark_identities[idx])
        bits = self.base.get_codeword_by_index(idx)
        return bits.tolist(), watermark, int(dist)

    def find_closest_match_batch(
        self,
        queries: list[np.ndarray] | np.ndarray,
        *,
        distances_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """
        Batch search: find closest codebook entry for each query.

        Args:
            queries: List of 1D uint8 arrays (variable length allowed), or
                     a 2D uint8 array of shape (B, length) for uniform-length queries.
            distances_only: If True, return only distances array (shape (B,), int32).

        Returns:
            If distances_only=False: Tuple of (best_indices, watermark_ids, distances)
                - best_indices: (B,) int64
                - watermark_ids: (B,) int64
                - distances: (B,) int32
            If distances_only=True: distances array only, shape (B,) int32.
        """
        if self.codebook_packed is None or self.base.watermark_identities is None:
            raise RuntimeError("Levenshtein codebook not initialized")

        # Normalize input to list of 1D arrays
        if isinstance(queries, np.ndarray) and queries.ndim == 2:
            query_list = [queries[i] for i in range(queries.shape[0])]
        else:
            query_list = list(queries)

        B = len(query_list)
        if B == 0:
            empty_i = np.empty(0, dtype=np.int64)
            empty_d = np.empty(0, dtype=np.int32)
            if distances_only:
                return empty_d
            return empty_i, empty_i.copy(), empty_d

        # Normalize each query to {0,1} uint8 and determine lengths
        normalized: list[np.ndarray] = []
        query_lens = np.empty(B, dtype=np.int32)
        for i, q in enumerate(query_list):
            q_arr = np.asarray(q, dtype=np.uint8).reshape(-1)
            if q_arr.size > 0:
                q_arr = (q_arr != 0).astype(np.uint8)
            normalized.append(q_arr)
            query_lens[i] = q_arr.shape[0]

        max_len = int(query_lens.max())
        if max_len == 0:
            # All queries are empty; distance = u for all
            dists = np.full(B, self.u, dtype=np.int32)
            if distances_only:
                return dists
            indices = np.zeros(B, dtype=np.int64)
            wm_ids = self.base.watermark_identities[indices]
            return indices, wm_ids, dists

        max_words = (max_len + 63) // 64

        # Pack all queries into (B, max_words) uint64 array
        queries_packed = np.zeros((B, max_words), dtype=np.uint64)
        for i, q_arr in enumerate(normalized):
            q_len = int(query_lens[i])
            if q_len > 0:
                packed = _pack_bits_to_uint64(q_arr, q_len)
                queries_packed[i, : packed.shape[0]] = packed

        # Batch search
        best_indices, best_distances = _search_codebook_batch(
            queries_packed, query_lens, self.codebook_packed, self.u
        )

        if distances_only:
            return best_distances

        watermark_ids = self.base.watermark_identities[best_indices]
        return best_indices, watermark_ids, best_distances

    # ------------------------------------------------------------------
    # Build/load cache helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Convert base uint8 packed codes to uint64 LE packed and cache as .npz."""
        if self.base.packed_codes is None:
            raise RuntimeError("Packed codes unavailable for index build")

        print(f"Building Levenshtein uint64 index (n={self.base.n}, u={self.base.u})...")
        codebook_packed = _batch_convert_uint8_to_uint64(
            self.base.packed_codes, self.base.u, self.num_words
        )

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        np.savez_compressed(
            self.index_path,
            codebook_packed=codebook_packed,
            n=np.int64(self.base.n),
            u=np.int64(self.base.u),
            generator_tag=np.array(self.base.generator.algorithm_tag),
            secret_key=np.array(self.base.secret_key),
            count=np.int64(codebook_packed.shape[0]),
        )

        self.codebook_packed = codebook_packed

    def _load_or_build_index(self) -> None:
        try:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError("Levenshtein NPZ not found")

            data = np.load(self.index_path, allow_pickle=True)

            # Validate metadata matches the base
            if int(data["n"]) != self.base.n or int(data["u"]) != self.base.u:
                raise ValueError("Cached Levenshtein codebook n/u mismatch")
            if str(data["generator_tag"]) != self.base.generator.algorithm_tag:
                raise ValueError("Cached Levenshtein generator tag mismatch")
            if str(data["secret_key"]) != self.base.secret_key:
                raise ValueError("Cached Levenshtein secret key mismatch")
            if self.base.watermark_identities is None or int(data["count"]) != int(
                self.base.watermark_identities.shape[0]
            ):
                raise ValueError("Cached Levenshtein row count mismatch")

            self.codebook_packed = np.ascontiguousarray(
                data["codebook_packed"], dtype=np.uint64
            )

        except Exception:
            # On any failure (missing/invalid), rebuild index
            self._build_index()
