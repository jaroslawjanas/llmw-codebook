#!/usr/bin/env python3
"""
Codebook generation and exact Hamming top-1 search with Faiss (Binary Index)

Overview
- We provide a small hierarchy that separates "how codewords are generated" from the
  common mechanics (caching, byte-packing, Faiss exact Hamming search, API).
- Two implementations:
  1) CodebookRNG: reproduces the existing deterministic RNG-based codeword generation
     per watermark identity, driven by the secret_key (baseline).
  2) CodebookBCH: generates codewords from a binary BCH code to increase separation
     (Hamming distance) between different watermarks.

Terminology
- watermark: an integer in [0, 2^n), representing the identity.
- codeword: a u-bit vector assigned to a given watermark.
- Hamming distance: number of positions where two equal-length bit vectors differ.

BCH parameters (used only in CodebookBCH; explained here for clarity)
- In a binary BCH code, you pick a field size parameter m (an integer >= 2).
- The canonical (primitive) BCH block length is n0 = 2^m - 1 bits.
- The code has a dimension k (number of information bits), so there are 2^k codewords.
- The designed error-correcting capability is t; the code guarantees minimum distance
  d_min >= 2*t + 1. Increasing t generally decreases k (fewer information bits).
- Constraint for our use: we need to uniquely map 2^n watermarks to codewords; hence
  we must have k >= n (at least n information bits).
- We produce a length-u codeword by:
  1) Encoding a k-bit message whose first n bits are the watermark in binary and the
     remaining (k - n) bits are zeros (injective mapping -> unique codewords).
  2) Applying a deterministic, secret-key-derived permutation Π over the n0 positions,
     then puncturing (keeping) the first u positions after permutation. This is a simple
     way to reach arbitrary u. Permutation preserves Hamming distances; puncturing
     may reduce the formal minimum distance but empirically maintains strong separation
     when u is not too close to n0.
  3) Optionally applying a deterministic XOR mask of length u derived from the secret key.
     XOR does not change Hamming distances; it ties the codebook to your secret.

Notes on Faiss:
- Faiss binary indexes operate on bit vectors with dimension divisible by 8. We zero-pad
  codewords to the next multiple of 8 bits when packing; zero-padding both database and
  query preserves the Hamming distances between their original u-bit portions.

Public API (common to all codebooks)
- get_random_row(seed) -> (codeword_bits: List[int], watermark: int)
- find_closest_match(target_codeword_bits) -> (closest_codeword_bits: List[int], watermark: int, hamming_distance: int)

We keep the API focused on exact top-1 search via Faiss IndexBinaryFlat.
"""

import os
import hashlib
from typing import List, Tuple, Optional

import numpy as np
import faiss  # Faiss (CPU) is required and used unconditionally

import src.paths as paths


class CodebookBase:
    """
    Base class that implements:
    - cache filenames, NPZ save/load (packed codewords),
    - bit packing helpers,
    - building/loading a Faiss IndexBinaryFlat for exact Hamming top-1 search,
    - public API methods: get_random_row() and find_closest_match().

    Subclasses implement:
    - _algorithm_tag(): str  -> short string for filenames (e.g., 'rng', 'bch_m8_t7')
    - generate_codewords(): np.ndarray (2**n, u) of dtype uint8 with values {0,1}
    """

    def __init__(self, n: int, u: int, secret_key: str, npz_dir: Optional[str] = ""):
        """
        Args:
            n: number of bits in watermark identities; table size is 2**n (watermarks 0..2**n-1)
            u: length (in bits) of each codeword
            secret_key: secret string controlling deterministic codebook generation
            npz_dir: optional custom directory for storing the codebook .npz (defaults to paths.CODEBOOKS_CACHE_DIR)
        """
        self.n = int(n)
        self.u = int(u)
        self.secret_key = str(secret_key)

        # Resolve cache paths with algorithm tag to avoid collisions
        self.filename_base = self._get_filename_base()
        if npz_dir:
            os.makedirs(npz_dir, exist_ok=True)
            self.npz_path = os.path.join(npz_dir, f"{self.filename_base}.npz")
            self.index_path = os.path.join(npz_dir, f"{self.filename_base}__binary_flat.faiss")
        else:
            os.makedirs(paths.CODEBOOKS_CACHE_DIR, exist_ok=True)
            self.npz_path = os.path.join(paths.CODEBOOKS_CACHE_DIR, f"{self.filename_base}.npz")
            self.index_path = os.path.join(paths.CODEBOOKS_CACHE_DIR, f"{self.filename_base}__binary_flat.faiss")

        # Load or build the codebook (packed storage) and watermarks
        self.packed_codes, self.watermark_identities = self._load_or_generate_packed()

        # Dimension in bits for Faiss must be a multiple of 8
        self.bytes_per_code = int(self.packed_codes.shape[1])
        self.dim_bits = self.bytes_per_code * 8  # padded bit length used by Faiss

        # Load or build the exact Faiss binary index
        self.index = self._load_or_build_index()

    # -------------------------------
    # To be provided/overridden by subclasses
    # -------------------------------

    def _algorithm_tag(self) -> str:
        """
        Short, filesystem-friendly identifier that encodes the algorithm and important params.
        Example: 'rng' or 'bch_m8_t7'
        """
        raise NotImplementedError

    def generate_codewords(self) -> np.ndarray:
        """
        Generate codewords of shape (2**n, u) as 0/1 uint8 array.
        Each row corresponds to the codeword for a watermark in [0, 2**n).
        """
        raise NotImplementedError

    # -------------------------------
    # Common logic: caching & packing
    # -------------------------------

    def _get_filename_base(self) -> str:
        """
        Build a deterministic filename base for cache artifacts. Include the algorithm tag so RNG and BCH caches don't collide.
        """
        algo = self._algorithm_tag()
        return f"codebook_{algo}_n{self.n}_u{self.u}_sk_{self.secret_key}"

    def _load_or_generate_packed(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load existing packed codebook arrays (.npz) or generate new ones.

        Returns:
            packed_codes: ndarray (N, B) uint8, where N = 2**n and B = ceil(u/8)
            watermark_identities: ndarray (N,) int64
        """
        if os.path.exists(self.npz_path):
            print(f"Loading existing codebook (packed) from: {self.npz_path}")
            data = np.load(self.npz_path)
            if "packed_codes" in data and "watermark_identities" in data and "u" in data:
                packed_codes = data["packed_codes"].astype(np.uint8, copy=False)
                watermark_identities = data["watermark_identities"].astype(np.int64, copy=False)
                if watermark_identities.shape[0] != packed_codes.shape[0]:
                    raise ValueError("Corrupt codebook cache: identities and codes row count mismatch.")
                if int(data["u"].item()) != self.u:
                    raise ValueError("Loaded codebook 'u' does not match requested 'u'.")
                print(f"Loaded codebook with {len(packed_codes)} entries, {self.u} bits each (packed into {packed_codes.shape[1]} bytes)")
                return packed_codes, watermark_identities
            raise ValueError("Unrecognized codebook cache format.")

        # Generate new codebook
        print(f"Generating new codebook: {self.filename_base}")
        packed_codes, watermark_identities = self._generate_packed_arrays()

        # Save packed arrays
        np.savez_compressed(self.npz_path, packed_codes=packed_codes, watermark_identities=watermark_identities, u=self.u)
        print(f"Saved codebook to: {self.npz_path}")

        return packed_codes, watermark_identities

    def _generate_packed_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate watermark identities and their codewords bits, then pack into bytes.

        Returns:
            packed_codes: ndarray (N, B) uint8
            watermark_identities: ndarray (N,) int64
        """
        table_size = 2 ** self.n
        print(f"Generating codebook with {table_size} entries...")

        watermark_identities = np.arange(table_size, dtype=np.int64)

        # Delegate codeword generation to subclass
        codeword_bits = self.generate_codewords()  # shape (N, u), dtype uint8 {0,1}

        if not (isinstance(codeword_bits, np.ndarray) and codeword_bits.shape == (table_size, self.u) and codeword_bits.dtype == np.uint8):
            raise ValueError("generate_codewords() must return np.ndarray of shape (2**n, u) and dtype uint8 with values {0,1}")

        # Pack 0/1 bits into bytes; if u not divisible by 8, right-pad with zeros
        packed_codes = self._pack_bits_2d(codeword_bits, self.u)

        # Free large matrix ASAP (let GC handle if references remain)
        del codeword_bits

        return packed_codes, watermark_identities

    @staticmethod
    def _pack_bits_2d(bits_01: np.ndarray, u: int) -> np.ndarray:
        """
        Pack a 2D array of 0/1 bits (N, u) into bytes (N, B), padding to next multiple of 8 if necessary.
        Uses big-endian bit order for consistency with Faiss typical conventions.

        Args:
            bits_01: ndarray of shape (N, u) with dtype uint8 and values {0,1}
            u: original bit length

        Returns:
            packed: ndarray of shape (N, B) with dtype uint8
        """
        if bits_01.ndim != 2 or bits_01.shape[1] != u:
            raise ValueError("Input bits_01 must have shape (N, u).")
        remainder = u % 8
        if remainder != 0:
            pad = 8 - remainder
            bits_01 = np.pad(bits_01, pad_width=((0, 0), (0, pad)), mode="constant", constant_values=0)
        packed = np.packbits(bits_01, axis=1, bitorder="big")
        return packed

    @staticmethod
    def _pack_bits_1d(bits_01: np.ndarray, u: int) -> np.ndarray:
        """
        Pack a 1D array/list of 0/1 bits (length u) into a single row of bytes (1, B).
        """
        arr = np.asarray(bits_01, dtype=np.uint8)
        if arr.ndim != 1 or arr.shape[0] != u:
            raise ValueError(f"Expected 1D array of length u={u}.")
        remainder = u % 8
        if remainder != 0:
            pad = 8 - remainder
            arr = np.pad(arr, pad_width=(0, pad), mode="constant", constant_values=0)
        packed = np.packbits(arr, axis=0, bitorder="big").astype(np.uint8, copy=False)
        return packed.reshape(1, -1)  # shape (1, B)

    @staticmethod
    def _unpack_bits_1d(packed_row: np.ndarray, u: int) -> List[int]:
        """
        Unpack a single packed row (B,) into a list of u bits (truncate any padding).
        """
        arr = np.asarray(packed_row, dtype=np.uint8).reshape(-1)
        bits = np.unpackbits(arr, bitorder="big")
        return bits[:u].astype(np.uint8).tolist()

    # -------------------------------
    # Faiss exact binary index
    # -------------------------------

    def _load_or_build_index(self):
        """
        Load an existing Faiss IndexBinaryFlat from disk, or build a new one and save it.
        """
        # Try load
        if os.path.exists(self.index_path):
            try:
                reader = getattr(faiss, "read_index_binary", None)
                index = reader(self.index_path) if callable(reader) else faiss.read_index(self.index_path)
                # Validate it's a binary index with matching dimension
                if not hasattr(index, "d") or index.d != self.dim_bits:
                    raise ValueError("Index dimension mismatch or non-binary index file.")
                if index.ntotal != self.packed_codes.shape[0]:
                    raise ValueError("Index ntotal mismatch; rebuilding index.")
                print(f"Loaded Faiss binary index from: {self.index_path} (ntotal={index.ntotal}, dim_bits={index.d})")
                return index
            except Exception as e:
                print(f"Warning: Failed to load existing index ({e}); rebuilding...")

        # Build new exact IndexBinaryFlat
        print(f"Building Faiss IndexBinaryFlat (dim_bits={self.dim_bits})...")
        index = faiss.IndexBinaryFlat(self.dim_bits)
        codes = np.ascontiguousarray(self.packed_codes, dtype=np.uint8)
        index.add(codes)
        writer_bin = getattr(faiss, "write_index_binary", None)
        if callable(writer_bin):
            writer_bin(index, self.index_path)
        else:
            faiss.write_index(index, self.index_path)
        print(f"Saved Faiss index to: {self.index_path} (ntotal={index.ntotal})")
        return index

    # -------------------------------
    # Public API
    # -------------------------------

    def get_random_row(self, seed: int) -> Tuple[List[int], int]:
        """
        Get a random watermark and its codeword using a seeded RNG.
        Returns:
            (codeword_bits_as_list, watermark)
        """
        rng = np.random.default_rng(seed)
        watermark = int(rng.integers(0, 2 ** self.n))

        packed_row = self.packed_codes[watermark]
        codeword_bits = self._unpack_bits_1d(packed_row, self.u)
        return codeword_bits, watermark

    def find_closest_match(self, target_codeword: List[int]) -> Tuple[List[int], int, int]:
        """
        Find the single closest codeword by exact Hamming distance using Faiss binary flat index.

        Args:
            target_codeword: list/array of 0/1 bits of length u

        Returns:
            (closest_codeword_bits, watermark, hamming_distance)
        """
        # Basic validation
        if len(target_codeword) != self.u:
            raise ValueError(f"Target codeword must have length u={self.u}")
        if not all(bit in (0, 1) for bit in target_codeword):
            raise ValueError("Target codeword must contain only 0s and 1s")

        # Pack query to bytes (shape (1,B)); padded zeros match database padding
        query_packed = self._pack_bits_1d(np.asarray(target_codeword, dtype=np.uint8), self.u)

        # Perform exact 1-NN search
        distances, indices = self.index.search(query_packed, 1)  # exact Hamming, top-1
        idx = int(indices[0, 0])
        dist = int(distances[0, 0])

        # Map idx to watermark and retrieve original bits
        watermark = int(self.watermark_identities[idx])
        closest_bits = self._unpack_bits_1d(self.packed_codes[idx], self.u)
        return closest_bits, watermark, dist


class CodebookRNG(CodebookBase):
    """
    Deterministic RNG-based codewords (reproduces the existing behavior).
    For each watermark i, seed a local RNG with SHA256(secret_key : i) and draw u bits.
    """

    def _algorithm_tag(self) -> str:
        return "rng"

    def generate_codewords(self) -> np.ndarray:
        table_size = 2 ** self.n
        bits = np.zeros((table_size, self.u), dtype=np.uint8)
        for i in range(table_size):
            seed_material = f"{self.secret_key}:{i}".encode("utf-8")
            digest = hashlib.sha256(seed_material).digest()
            identity_seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(identity_seed)
            bits[i] = rng.integers(0, 2, size=self.u, dtype=np.uint8)
        return bits


class CodebookBCH(CodebookBase):
    """
    BCH-based codewords for larger inter-watermark separation.

    Construction:
    - Choose m (either provided or minimal such that n0 = 2^m - 1 >= u).
    - Choose t (either provided or the largest t s.t. k >= n); d_min >= 2t+1.
    - For each watermark (integer in [0, 2^n)):
        * Build a k-bit message: first n bits are the watermark in binary, remaining (k-n) are zeros.
        * Encode with BCH -> codeword length n0.
        * Apply secret-key permutation Π over n0 positions, then puncture (keep) the first u positions.
        * Optionally XOR a secret-key-derived mask of length u.
    - Permutation and XOR preserve Hamming distances. Puncturing may reduce the formal minimum distance,
      but empirically maintains good separation if u is not too close to n0.
    """

    def __init__(
        self,
        n: int,
        u: int,
        secret_key: str,
        t: Optional[int] = None,
        m: Optional[int] = None,
        permute: bool = True,
        xor_mask: bool = True,
        npz_dir: Optional[str] = "",
    ):
        self.t = t
        self.m = m
        self.permute = bool(permute)
        self.xor_mask = bool(xor_mask)
        super().__init__(n=n, u=u, secret_key=secret_key, npz_dir=npz_dir)

    def _algorithm_tag(self) -> str:
        # Use provided params in tag; actual resolved values are printed at build time
        t_tag = str(self.t) if self.t is not None else "auto"
        m_tag = str(self.m) if self.m is not None else "auto"
        return f"bch_m{m_tag}_t{t_tag}"

    @staticmethod
    def _int_to_bits(value: int, length: int) -> np.ndarray:
        """
        Convert integer 'value' to a big-endian bit array of given 'length' (uint8 {0,1}).
        """
        bits = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            # Fill from MSB to LSB (big-endian): position i corresponds to bit (length-1-i)
            bits[i] = (value >> (length - 1 - i)) & 1
        return bits

    def _derive_seed(self, tag: str) -> int:
        """
        Derive a 64-bit seed from (secret_key, tag).
        """
        h = hashlib.sha256(f"{self.secret_key}|{tag}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big", signed=False)

    def _derive_mask_bits(self, length: int) -> np.ndarray:
        """
        Derive a deterministic 0/1 mask of given length from the secret key.
        """
        rng = np.random.default_rng(self._derive_seed("bch-mask"))
        return rng.integers(0, 2, size=length, dtype=np.uint8)

    def generate_codewords(self) -> np.ndarray:
        try:
            import galois  # local import to avoid hard dependency when using RNG
        except Exception as e:
            raise ImportError("CodebookBCH requires the 'galois' package. Ensure it's installed.") from e

        # Resolve m (n0) so that n0 >= u
        if self.m is None:
            m = 2
            while (2 ** m - 1) < self.u:
                m += 1
        else:
            m = int(self.m)
            if (2 ** m - 1) < self.u:
                raise ValueError(f"Provided m={m} gives n0={2**m-1} < u={self.u}. Increase m or decrease u.")
        n0 = 2 ** m - 1

        # Choose t, maximizing separation subject to k >= n
        resolved_t = None
        resolved_k = None
        if self.t is not None:
            d = 2*int(self.t) + 1
            bch = galois.BCH(n0, d=d)
            if bch.k < self.n:
                raise ValueError(f"Provided t={self.t} (d={d}) yields k={bch.k} < n={self.n}. Decrease t or increase m.")
            resolved_t = int(self.t)
            resolved_k = int(bch.k)
        else:
            # Try descending t to find the largest that still gives k >= n
            # Set an upper bound; t cannot exceed n0//2. A practical small cap keeps search quick.
            t_max = min(n0 // 2, 64)
            bch = None
            for t_candidate in range(t_max, 0, -1):
                try:
                    d = 2*t_candidate + 1
                    bch_candidate = galois.BCH(n0, d=d)
                    if bch_candidate.k >= self.n:
                        bch = bch_candidate
                        resolved_t = t_candidate
                        resolved_k = int(bch_candidate.k)
                        break
                except Exception:
                    # Some d may be invalid for a given n0; skip
                    continue
            if bch is None:
                raise RuntimeError(
                    f"Failed to find BCH parameters for n0={n0} (m={m}) with k >= n={self.n}. "
                    f"Try increasing m or decreasing n."
                )

        print(f"BCH parameters resolved: m={m} (n0={n0}), t={resolved_t}, k={resolved_k}, d_min >= {2*resolved_t+1}")

        # Prepare permutation Π over n0 positions (derived from secret key), if enabled
        if self.permute:
            perm_rng = np.random.default_rng(self._derive_seed("bch-permutation"))
            permutation = np.arange(n0, dtype=np.int64)
            perm_rng.shuffle(permutation)
        else:
            permutation = np.arange(n0, dtype=np.int64)

        # Prepare XOR mask of length u (derived from secret key), if enabled
        mask = self._derive_mask_bits(self.u) if self.xor_mask else None

        # Generate codewords
        table_size = 2 ** self.n
        codewords = np.zeros((table_size, self.u), dtype=np.uint8)

        # Build and encode for each watermark
        for i in range(table_size):
            # k-bit message: first n bits are watermark (big-endian), remainder zeros
            msg = np.zeros(resolved_k, dtype=np.uint8)
            msg[: self.n] = self._int_to_bits(i, self.n)

            # BCH encode -> GF2 array length n0
            cw_full = bch.encode(msg)  # type: ignore[attr-defined]
            cw_full = np.array(cw_full, dtype=np.uint8)  # convert to 0/1 uint8

            # Apply permutation, then puncture to u bits
            cw_perm = cw_full[permutation]
            cw_u = cw_perm[: self.u]

            # Optional secret-key XOR mask (does not change Hamming distances)
            if mask is not None:
                cw_u = np.bitwise_xor(cw_u, mask)

            codewords[i] = cw_u

        return codewords


# Backward-incompatible change: we no longer export CodebookManager.
# Update callers to import and instantiate either CodebookRNG or CodebookBCH as needed.
# Example:
#   from src.codebook import CodebookRNG, CodebookBCH
#   codebook = CodebookRNG(n=n, u=u, secret_key=secret_key)
#   # or
#   codebook = CodebookBCH(n=n, u=u, secret_key=secret_key, t=7)  # for example
