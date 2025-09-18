#!/usr/bin/env python3
"""
Codebook generation and exact Hamming top-1 search with Faiss (Binary Index)

Parallel generation
- Work is split into fixed-size portions (chunks of rows).
- A process pool (default 4 workers) generates packed byte chunks concurrently.
- Parent process drains results and writes them into the final packed matrix.
- Progress is shown with a tqdm progress bar.

Overview
- We provide a small hierarchy that separates "how codewords are generated" from the
  common mechanics (caching, byte-packing, Faiss exact Hamming search, API).
- Two implementations:
  1) CodebookRNG: deterministic RNG-based codeword generation per watermark identity,
     driven by the secret_key (baseline).
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
import math
import hashlib
from typing import List, Tuple, Optional

import numpy as np
import faiss  # Faiss (CPU) is required and used
from tqdm import tqdm
from multiprocessing import Pool

import src.paths as paths


# -------------------------------
# Helpers shared by workers
# -------------------------------

def _derive_seed(secret_key: str, tag: str) -> int:
    """
    Derive a 64-bit seed from (secret_key, tag).
    """
    h = hashlib.sha256(f"{secret_key}|{tag}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _int_to_bits(value: int, length: int) -> np.ndarray:
    """
    Convert integer 'value' to a big-endian bit array of given 'length' (uint8 {0,1}).
    """
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        bits[i] = (value >> (length - 1 - i)) & 1
    return bits


def _pack_bits_2d_local(bits_01: np.ndarray, u: int) -> np.ndarray:
    """
    Local packbits for worker results: pack (R, u) into (R, B) with big-endian, zero-pad to a multiple of 8 bits.
    """
    remainder = u % 8
    if remainder != 0:
        pad = 8 - remainder
        bits_01 = np.pad(bits_01, pad_width=((0, 0), (0, pad)), mode="constant", constant_values=0)
    return np.packbits(bits_01, axis=1, bitorder="big").astype(np.uint8, copy=False)


# -------------------------------
# RNG worker (top-level for Windows pickling)
# -------------------------------

def _rng_chunk_worker(args: Tuple[int, int, int, str]) -> Tuple[int, np.ndarray]:
    """
    Generate RNG-based codewords for a chunk [start, end), pack into bytes, and return (start, packed_chunk).
    args: (start, end, u, secret_key)
    """
    start, end, u, secret_key = args
    rows = end - start
    bits = np.empty((rows, u), dtype=np.uint8)
    for offset, i in enumerate(range(start, end)):
        seed_material = f"{secret_key}:{i}".encode("utf-8")
        digest = hashlib.sha256(seed_material).digest()
        identity_seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(identity_seed)
        bits[offset] = rng.integers(0, 2, size=u, dtype=np.uint8)
    packed = _pack_bits_2d_local(bits, u)
    return start, packed


# -------------------------------
# BCH workers (initializer + worker) - top-level for Windows pickling
# -------------------------------

_BCH = None
_BCH_k = None
_BCH_perm = None
_BCH_mask = None
_BCH_n = None
_BCH_u = None


def _bch_pool_init(n: int, u: int, secret_key: str, m: int, t: int, permute: bool, xor_mask: bool) -> None:
    """
    Per-process initializer: build BCH, permutation and mask once in each worker process.
    """
    global _BCH, _BCH_k, _BCH_perm, _BCH_mask, _BCH_n, _BCH_u
    _BCH_n = int(n)
    _BCH_u = int(u)
    n0 = (2 ** m) - 1
    d = 2 * int(t) + 1

    # Import galois inside worker to avoid requiring it when not using BCH codebook
    import galois  # type: ignore

    _BCH = galois.BCH(n0, d=d)
    _BCH_k = int(_BCH.k)

    # Permutation over n0 positions
    if permute:
        perm_rng = np.random.default_rng(_derive_seed(secret_key, "bch-permutation"))
        permutation = np.arange(n0, dtype=np.int64)
        perm_rng.shuffle(permutation)
        _BCH_perm = permutation
    else:
        _BCH_perm = np.arange(n0, dtype=np.int64)

    # XOR mask of length u
    if xor_mask:
        mask_rng = np.random.default_rng(_derive_seed(secret_key, "bch-mask"))
        _BCH_mask = mask_rng.integers(0, 2, size=_BCH_u, dtype=np.uint8)
    else:
        _BCH_mask = None


def _bch_chunk_worker(args: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    """
    Generate BCH-based codewords for a chunk [start, end), pack into bytes, and return (start, packed_chunk).
    Uses globals initialized by _bch_pool_init.
    """
    global _BCH, _BCH_k, _BCH_perm, _BCH_mask, _BCH_n, _BCH_u
    start, end = args
    rows = end - start
    bits = np.empty((rows, _BCH_u), dtype=np.uint8)

    for offset, i in enumerate(range(start, end)):
        # k-bit message: first n bits are watermark (big-endian), remainder zeros
        msg = np.zeros(_BCH_k, dtype=np.uint8)
        msg[:_BCH_n] = _int_to_bits(i, _BCH_n)

        # Encode -> length n0
        cw_full = _BCH.encode(msg)  # type: ignore[attr-defined]
        cw_full = np.array(cw_full, dtype=np.uint8)

        # Permute then puncture to u
        cw_perm = cw_full[_BCH_perm]
        cw_u = cw_perm[:_BCH_u]

        # Optional XOR mask
        if _BCH_mask is not None:
            cw_u = np.bitwise_xor(cw_u, _BCH_mask)

        bits[offset] = cw_u

    packed = _pack_bits_2d_local(bits, _BCH_u)
    return start, packed


class CodebookBase:
    """
    Base class that implements:
    - cache filenames, NPZ save/load (packed codewords),
    - bit packing helpers,
    - building/loading a Faiss IndexBinaryFlat for exact Hamming top-1 search,
    - public API methods: get_random_row() and find_closest_match().

    Subclasses implicitly define:
    - Algorithm tag via _algorithm_tag()
    - Generation mode (RNG or BCH) used to select worker functions
    """

    def __init__(
        self,
        n: int,
        u: int,
        secret_key: str,
        npz_dir: Optional[str] = "",
        workers: int = 4,
        chunk_rows: Optional[int] = None,
    ):
        """
        Args:
            n: number of bits in watermark identities; table size is 2**n (watermarks 0..2**n-1)
            u: length (in bits) of each codeword
            secret_key: secret string controlling deterministic codebook generation
            npz_dir: optional custom directory for storing the codebook .npz (defaults to paths.CODEBOOKS_CACHE_DIR)
            workers: number of parallel worker processes to use for generation (default: 4)
            chunk_rows: optional override for the chunk size in rows (default: auto policy)
        """
        self.n = int(n)
        self.u = int(u)
        self.secret_key = str(secret_key)
        self.workers = max(1, int(workers))
        self.chunk_rows = None if chunk_rows is None else max(1, int(chunk_rows))

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
        packed_codes, watermark_identities = self._generate_packed_arrays_parallel()

        # Save packed arrays
        np.savez_compressed(self.npz_path, packed_codes=packed_codes, watermark_identities=watermark_identities, u=self.u)
        print(f"Saved codebook to: {self.npz_path}")

        return packed_codes, watermark_identities

    def _default_chunk_rows(self) -> int:
        """
        Default chunk size policy: target ~4 MiB of raw 0/1 bit storage per chunk.
        rows ~= 4*MiB / u. Clamp to [4096, 262144] to keep throughput/RAM balanced.
        """
        target_bytes = 4 * 1024 * 1024
        rows = max(4096, min(262144, target_bytes // max(1, self.u)))
        return int(rows)

    def _generate_packed_arrays_parallel(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parallel generation of packed codewords:
        - Split rows [0..N) into chunks of chunk_rows
        - Use a process pool to generate each chunk as packed bytes
        - Assemble into a preallocated (N, B) uint8 array

        Returns:
            packed_codes: ndarray (N, B) uint8
            watermark_identities: ndarray (N,) int64
        """
        N = 2 ** self.n
        B = (self.u + 7) // 8

        # Note: For future optimization, we can consider a memory-mapped array (numpy.memmap)
        # to stream chunks directly to disk (to further lower peak RAM). Not necessary now.
        packed_codes = np.empty((N, B), dtype=np.uint8)

        watermark_identities = np.arange(N, dtype=np.int64)
        chunk_rows = self.chunk_rows or self._default_chunk_rows()
        if chunk_rows <= 0:
            chunk_rows = self._default_chunk_rows()

        # Prepare chunk ranges
        chunks: List[Tuple[int, int]] = []
        for start in range(0, N, chunk_rows):
            end = min(N, start + chunk_rows)
            chunks.append((start, end))

        # Select worker strategy based on subclass
        is_rng = isinstance(self, CodebookRNG)
        is_bch = isinstance(self, CodebookBCH)

        # Progress bar over total rows
        progress = tqdm(total=N, desc=f"Generating {self._algorithm_tag()} codewords (workers={self.workers})", unit="cw")

        if is_rng:
            # RNG workers don't need an initializer
            args_iter = ((start, end, self.u, self.secret_key) for (start, end) in chunks)
            with Pool(processes=self.workers) as pool:
                for start, packed_chunk in pool.imap_unordered(_rng_chunk_worker, args_iter, chunksize=1):
                    end = start + packed_chunk.shape[0]
                    packed_codes[start:end] = packed_chunk
                    progress.update(end - start)

        elif is_bch:
            # Resolve BCH parameters in parent, then pass to initializer
            # (We resolve m,t,k exactly as in the original logic)
            resolved_m, resolved_t, resolved_k = self._resolve_bch_parameters()
            initargs = (self.n, self.u, self.secret_key, resolved_m, resolved_t, self.permute, self.xor_mask)
            with Pool(processes=self.workers, initializer=_bch_pool_init, initargs=initargs) as pool:
                for start, packed_chunk in pool.imap_unordered(_bch_chunk_worker, chunks, chunksize=1):
                    end = start + packed_chunk.shape[0]
                    packed_codes[start:end] = packed_chunk
                    progress.update(end - start)
        else:
            progress.close()
            raise RuntimeError("Unknown codebook subclass: expected CodebookRNG or CodebookBCH.")

        progress.close()
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

    Note: Generation is now parallelized in the base class via chunked workers.
    """

    def _algorithm_tag(self) -> str:
        return "rng"

    # Keeping this method for reference/backwards discussion; not used by the new parallel path.
    def generate_codewords(self) -> np.ndarray:
        table_size = 2 ** self.n
        bits = np.zeros((table_size, self.u), dtype=np.uint8)
        for i in tqdm(range(table_size), desc="Generating RNG codewords", unit="cw"):
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
        workers: int = 4,
        chunk_rows: Optional[int] = None,
    ):
        self.t = t
        self.m = m
        self.permute = bool(permute)
        self.xor_mask = bool(xor_mask)
        super().__init__(n=n, u=u, secret_key=secret_key, npz_dir=npz_dir, workers=workers, chunk_rows=chunk_rows)

    def _algorithm_tag(self) -> str:
        # Use provided params in tag; actual resolved values are printed at build time
        t_tag = str(self.t) if self.t is not None else "auto"
        m_tag = str(self.m) if self.m is not None else "auto"
        return f"bch_m{m_tag}_t{t_tag}"

    def _resolve_bch_parameters(self) -> Tuple[int, int, int]:
        """
        Resolve BCH parameters (m, t, k) with the same semantics as before:
        - If m not provided, choose minimal m s.t. n0 >= u
        - If t not provided, choose largest t s.t. k >= n
        Returns (resolved_m, resolved_t, resolved_k)
        """
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
            d = 2 * int(self.t) + 1
            bch = galois.BCH(n0, d=d)
            if bch.k < self.n:
                raise ValueError(f"Provided t={self.t} (d={d}) yields k={bch.k} < n={self.n}. Decrease t or increase m.")
            resolved_t = int(self.t)
            resolved_k = int(bch.k)
        else:
            # Try descending t to find the largest that still gives k >= n
            t_max = min(n0 // 2, 64)
            bch = None
            for t_candidate in range(t_max, 0, -1):
                try:
                    d = 2 * t_candidate + 1
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

        print(f"BCH parameters resolved: m={m} (n0={n0}), t={resolved_t}, k={resolved_k}, d_min >= {2 * resolved_t + 1}")
        return m, resolved_t, resolved_k


# Backward-incompatible note: the generation is now chunked and parallelized.
# Callers should import and instantiate CodebookRNG or CodebookBCH as needed.
# Example:
#   from src.codebook import CodebookRNG, CodebookBCH
#   codebook = CodebookRNG(n=n, u=u, secret_key=secret_key, workers=4)
#   # or
#   codebook = CodebookBCH(n=n, u=u, secret_key=secret_key, t=7, workers=4)  # for example
