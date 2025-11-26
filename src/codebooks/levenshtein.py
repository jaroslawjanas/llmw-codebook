"""
Levenshtein-distance specialized codebook that consumes a base NPZ and builds a RapidFuzz-ready
string list for fast edit-distance search.

Responsibilities:
- Consume CodebookBase (canonical codewords stored as packed bits in NPZ)
- Build and cache a list of bitstrings for RapidFuzz Levenshtein search
- Provide nearest-neighbor search by edit distance (insertions, deletions, substitutions)
- Delegate non-search helpers (like get_random_row) to the base

File naming:
- Base:        codebook_{algo}_n{n}_u{u}_sk-&lt;secret-key&gt;.npz
- Levenshtein: codebook_{algo}_n{n}_u{u}_sk-&lt;secret-key&gt;_metric-levenshtein.pkl
"""

from __future__ import annotations

import os
import pickle
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .base import CodebookBase

__all__ = ["CodebookLevenshtein"]


class CodebookLevenshtein:
    """
    Levenshtein-distance codebook built on top of a CodebookBase.

    Construction:
        base = CodebookBase(...)
        levenshtein = CodebookLevenshtein(base)

    Usage:
        # Select a deterministic random watermark row (no RF data needed)
        bits, watermark = levenshtein.get_random_row(seed)

        # Find closest match by Levenshtein edit distance (uses RapidFuzz)
        bits, watermark, distance = levenshtein.find_closest_match(target_bits)
    """

    def __init__(self, base: CodebookBase, *, cache_dir: Optional[str] = None) -> None:
        if base.packed_codes is None:
            raise RuntimeError("Base codebook not loaded")

        self.base = base
        self.cache_dir = cache_dir or base.cache_dir

        self.filename_base = f"{self.base.filename_base}_metric-levenshtein"
        self.index_path = os.path.join(self.cache_dir, f"{self.filename_base}.pkl")

        self.codewords_str: Optional[List[str]] = None
        self.bytes_per_code: int = int(self.base.bytes_per_code)
        self.dim_bits: int = int(self.base.dim_bits)

        self._load_or_build_strings()

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

    def find_closest_match(self, target_codeword: Sequence[int] | np.ndarray) -> Tuple[List[int], int, int]:
        """
        Locate the closest codeword by Levenshtein edit distance using RapidFuzz.
        Accepts target codeword as a 1D sequence of 0/1 of ANY length (variable length allowed).
        Returns (closest_codeword_bits, watermark_identity, distance_value).

        - closest_codeword_bits: the matched codebook codeword (length = self.u)
        - watermark_identity: watermark id (row index in base)
        - distance_value: raw edit distance (insertions + deletions + substitutions)
        """
        if self.codewords_str is None or self.base.watermark_identities is None:
            raise RuntimeError("Levenshtein codebook not initialized")

        target_array = np.asarray(target_codeword, dtype=np.uint8).reshape(-1)
        # Be robust to accidental non-binary values by mapping to {0,1}
        if target_array.size == 0:
            # Allow empty input; Levenshtein distance will equal len(candidate)
            query_str = ""
        else:
            target_array = (target_array != 0).astype(np.uint8)
            # Convert to a contiguous string "0101..."
            query_str = "".join("1" if b else "0" for b in target_array.tolist())

        # RapidFuzz extractOne returns (best_match_str, score_or_distance, index)
        # With scorer=Levenshtein.distance, the second element is the raw distance (lower is better).
        best = process.extractOne(
            query_str,
            self.codewords_str,
            scorer=Levenshtein.distance,
            processor=None,  # our choices are already strings, skip internal pre-processing
        )
        if best is None:
            # Should not happen as choices are non-empty; guard for completeness
            raise RuntimeError("RapidFuzz returned no result")

        _, distance, idx = best
        idx = int(idx)
        watermark = int(self.base.watermark_identities[idx])
        bits = self.base.get_codeword_by_index(idx)
        return bits.tolist(), watermark, int(distance)

    # ------------------------------------------------------------------
    # Build/load cache helpers
    # ------------------------------------------------------------------

    def _build_strings(self) -> None:
        """
        Build the bitstring list from base.packed_codes and store to pickle cache.
        Uses a streaming row-by-row conversion to avoid large temporary allocations.
        """
        if self.base.packed_codes is None:
            raise RuntimeError("Packed codes unavailable for string build")

        N = int(self.base.packed_codes.shape[0])
        codewords_str: List[str] = []
        codewords_str_append = codewords_str.append

        # Convert row by row to keep peak memory low at large N
        for i in range(N):
            packed_row = self.base.packed_codes[i]
            bits = CodebookBase._unpack_bits_1d(packed_row, self.base.u)  # (u,) uint8
            s = "".join("1" if b else "0" for b in bits.tolist())
            codewords_str_append(s)

        metadata = {
            "n": self.base.n,
            "u": self.base.u,
            "generator_tag": self.base.generator.algorithm_tag,
            "secret_key": self.base.secret_key,
            "count": N,
        }

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"codewords_str": codewords_str, "metadata": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.codewords_str = codewords_str

    def _load_or_build_strings(self) -> None:
        try:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError("Levenshtein pickle not found")

            with open(self.index_path, "rb") as f:
                payload: Any = pickle.load(f)

            if not isinstance(payload, dict) or "codewords_str" not in payload or "metadata" not in payload:
                raise ValueError("Invalid Levenshtein pickle structure")

            codewords_str = payload["codewords_str"]
            metadata = payload["metadata"]

            if not isinstance(codewords_str, list) or any(not isinstance(s, str) for s in codewords_str):
                raise ValueError("Invalid codewords_str in Levenshtein pickle")

            # Validate metadata matches the base
            if int(metadata.get("n")) != self.base.n or int(metadata.get("u")) != self.base.u:
                raise ValueError("Cached Levenshtein codebook n/u mismatch")
            if str(metadata.get("generator_tag")) != self.base.generator.algorithm_tag:
                raise ValueError("Cached Levenshtein generator tag mismatch")
            if str(metadata.get("secret_key")) != self.base.secret_key:
                raise ValueError("Cached Levenshtein secret key mismatch")
            # Validate row count against loaded watermark identities length
            if self.base.watermark_identities is None or int(metadata.get("count")) != int(self.base.watermark_identities.shape[0]):
                raise ValueError("Cached Levenshtein row count mismatch")

            self.codewords_str = codewords_str
        except Exception:
            # On any failure (missing/invalid), rebuild strings
            self._build_strings()
