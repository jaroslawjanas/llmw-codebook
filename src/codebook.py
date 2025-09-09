#!/usr/bin/env python3
"""
Codebook Manager with Faiss Binary Index (exact, top-1)

This implementation:
- Generates deterministic bit codes per watermark identity (same logic as before)
- Packs bit codes into bytes (8x memory reduction) to use Faiss binary indexes
- Builds/loads a Faiss IndexBinaryFlat (exact Hamming search) and always returns the single closest match
- Removes NumPy-based distance search (Faiss-only as requested)
- Provides a new API: find_closest_match(...) → (bit_code_bits, watermark_identity, hamming_distance)
- Keeps a temporary compatibility wrapper find_closest_matches(..., top=1) that returns a single-element list
  (this wrapper will be removed after callers are migrated)

Notes on packing:
- Faiss binary indexes operate on bit vectors with bit dimension divisible by 8.
- If u is not a multiple of 8, we pad each code and query with zeros up to the next multiple of 8 bits before packing.
  Padding with zeros on both sides preserves exact Hamming distances between the original u-bit vectors because
  the appended bits are identical (0) and do not contribute to the distance.
"""

import os
import hashlib
from typing import List, Tuple, Optional

import numpy as np
import faiss  # Faiss (CPU) is required and used unconditionally

import src.paths as paths


class CodebookManager:
    """
    Generates and manages codebook tables for watermarking, and performs exact nearest neighbor
    search (top-1) in Hamming space using a Faiss IndexBinaryFlat.

    Public API:
      - get_random_row(seed) -> (bit_code_bits: List[int], watermark_identity: int)
      - find_closest_match(target_bit_code_bits) -> (closest_bit_code_bits, watermark_identity, hamming_distance)

    Temporary compatibility (to be removed after callers migrate):
      - find_closest_matches(target_bit_code_bits, top=1) -> [(closest_bit_code_bits, watermark_identity, hamming_distance)]
        Only top=1 is supported; any other value raises.
    """

    def __init__(self, n: int, u: int, secret_key: str, npz_dir: Optional[str] = ""):
        """
        Initialize the codebook manager, load or generate the packed codebook arrays,
        and build/load the Faiss exact binary index.

        Args:
            n: number of address bits, table size will be 2**n (watermark identities 0..2**n-1)
            u: bit length of each code (will be padded to next multiple of 8 for storage/indexing)
            secret_key: secret controlling deterministic per-identity code generation
            npz_dir: optional custom directory for storing the codebook .npz (defaults to paths.CODEBOOKS_CACHE_DIR)
        """
        self.n = int(n)
        self.u = int(u)
        self.secret_key = str(secret_key)

        # Resolve cache paths
        self.filename_base = self._get_filename_base()
        if npz_dir:
            os.makedirs(npz_dir, exist_ok=True)
            self.npz_path = os.path.join(npz_dir, f"{self.filename_base}.npz")
            self.index_path = os.path.join(npz_dir, f"{self.filename_base}__binary_flat.faiss")
        else:
            os.makedirs(paths.CODEBOOKS_CACHE_DIR, exist_ok=True)
            self.npz_path = os.path.join(paths.CODEBOOKS_CACHE_DIR, f"{self.filename_base}.npz")
            self.index_path = os.path.join(paths.CODEBOOKS_CACHE_DIR, f"{self.filename_base}__binary_flat.faiss")

        # Load or build the codebook (packed storage) and identities
        self.packed_codes, self.watermark_identities = self._load_or_generate_packed()

        # Bit dimension for Faiss must be a multiple of 8
        self.bytes_per_code = int(self.packed_codes.shape[1])
        self.dim_bits = self.bytes_per_code * 8  # padded bit length used by Faiss

        # Load or build the exact Faiss binary index
        self.index = self._load_or_build_index()

    # -------------------------------
    # Codebook generation and packing
    # -------------------------------

    def _get_filename_base(self) -> str:
        """
        Build a deterministic filename base for cache artifacts.
        We keep the original pattern for clarity and reproducibility.
        """
        return f"codebook_n{self.n}_u{self.u}_sk_{self.secret_key}"

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
            # Preferred new format
            if "packed_codes" in data and "watermark_identities" in data and "u" in data:
                packed_codes = data["packed_codes"].astype(np.uint8, copy=False)
                watermark_identities = data["watermark_identities"].astype(np.int64, copy=False)
                # Validate dimensions
                if watermark_identities.shape[0] != packed_codes.shape[0]:
                    raise ValueError("Corrupt codebook cache: identities and codes row count mismatch.")
                if int(data["u"].item()) != self.u:
                    raise ValueError("Loaded codebook 'u' does not match requested 'u'.")
                print(f"Loaded codebook with {len(packed_codes)} entries, {self.u} bits each (packed into {packed_codes.shape[1]} bytes)")
                return packed_codes, watermark_identities

            # Backward conversion: existing file may contain uncompressed bit_codes (0/1 uint8)
            if "bit_codes" in data and "watermark_identities" in data:
                print("Found legacy codebook format; converting to packed bytes and updating cache...")
                bit_codes = data["bit_codes"].astype(np.uint8, copy=False)
                watermark_identities = data["watermark_identities"].astype(np.int64, copy=False)
                packed_codes = self._pack_bits_2d(bit_codes, self.u)
                # Overwrite NPZ with the new compact format
                np.savez_compressed(self.npz_path, packed_codes=packed_codes, watermark_identities=watermark_identities, u=self.u)
                print(f"Converted and saved packed codebook to: {self.npz_path}")
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
        Generate watermark identities and their bit codes, then pack codes into bytes.

        Returns:
            packed_codes: ndarray (N, B) uint8
            watermark_identities: ndarray (N,) int64
        """
        table_size = 2 ** self.n
        print(f"Generating codebook with {table_size} entries...")

        # Identities 0..2**n-1
        watermark_identities = np.arange(table_size, dtype=np.int64)

        # Generate u-bit codes as 0/1 uint8, per identity (deterministic RNG per identity)
        # Note: This is the same generation logic as before, kept for reproducibility.
        bit_codes = np.zeros((table_size, self.u), dtype=np.uint8)
        for i, identity in enumerate(watermark_identities):
            seed_material = f"{self.secret_key}:{int(identity)}".encode("utf-8")
            digest = hashlib.sha256(seed_material).digest()
            identity_seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(identity_seed)
            bit_codes[i] = rng.integers(0, 2, size=self.u, dtype=np.uint8)

        # Pack 0/1 bits into bytes; if u not divisible by 8, right-pad with zeros (see _pack_bits_2d)
        packed_codes = self._pack_bits_2d(bit_codes, self.u)
        # Free the large 0/1 matrix ASAP
        del bit_codes

        return packed_codes, watermark_identities

    @staticmethod
    def _pack_bits_2d(bits_01: np.ndarray, u: int) -> np.ndarray:
        """
        Pack a 2D array of 0/1 bits (N, u) into bytes (N, B), padding to next multiple of 8 if necessary.
        Uses big-endian bit order for consistency.

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
                index = faiss.read_index(self.index_path)
                # Validate it's a binary index with matching dimension
                if not hasattr(index, "d") or index.d != self.dim_bits:
                    raise ValueError("Index dimension mismatch or non-binary index file.")
                if index.ntotal != self.packed_codes.shape[0]:
                    # If counts differ (e.g., cache mismatch), rebuild
                    raise ValueError("Index ntotal mismatch; rebuilding index.")
                print(f"Loaded Faiss binary index from: {self.index_path} (ntotal={index.ntotal}, dim_bits={index.d})")
                return index
            except Exception as e:
                print(f"Warning: Failed to load existing index ({e}); rebuilding...")

        # Build new exact IndexBinaryFlat
        print(f"Building Faiss IndexBinaryFlat (dim_bits={self.dim_bits})...")
        index = faiss.IndexBinaryFlat(self.dim_bits)
        # Faiss expects contiguous uint8 with shape (N, dim_bits/8)
        codes = np.ascontiguousarray(self.packed_codes, dtype=np.uint8)
        index.add(codes)
        faiss.write_index(index, self.index_path)
        print(f"Saved Faiss index to: {self.index_path} (ntotal={index.ntotal})")
        return index

    # -------------------------------
    # Public API
    # -------------------------------

    def get_random_row(self, seed: int) -> Tuple[List[int], int]:
        """
        Get a random watermark identity and its corresponding bit code using a seeded RNG.

        Returns:
            (bit_code_bits_as_list, watermark_identity)
        """
        rng = np.random.default_rng(seed)
        watermark_identity = int(rng.integers(0, 2 ** self.n))

        # Retrieve packed row and unpack to bits for the caller
        packed_row = self.packed_codes[watermark_identity]
        bit_code_bits = self._unpack_bits_1d(packed_row, self.u)
        return bit_code_bits, watermark_identity

    def find_closest_match(self, target_bit_code: List[int]) -> Tuple[List[int], int, int]:
        """
        Find the single closest code by exact Hamming distance using Faiss binary flat index.

        Args:
            target_bit_code: list/array of 0/1 bits of length u

        Returns:
            (closest_bit_code_bits, watermark_identity, hamming_distance)
        """
        # Basic validation
        if len(target_bit_code) != self.u:
            raise ValueError(f"Target bit code must have length u={self.u}")
        if not all(bit in (0, 1) for bit in target_bit_code):
            raise ValueError("Target bit code must contain only 0s and 1s")

        # Pack query to bytes (shape (1,B)); padded zeros match database padding
        query_packed = self._pack_bits_1d(np.asarray(target_bit_code, dtype=np.uint8), self.u)

        # Perform exact 1-NN search
        distances, indices = self.index.search(query_packed, 1)  # exact Hamming, top-1
        idx = int(indices[0, 0])
        dist = int(distances[0, 0])

        # Map idx to identity and retrieve original bits for convenience
        identity = int(self.watermark_identities[idx])
        closest_bits = self._unpack_bits_1d(self.packed_codes[idx], self.u)
        return closest_bits, identity, dist

    # -------------------------------
    # Temporary compatibility wrapper
    # -------------------------------

    def find_closest_matches(self, target_bit_code: List[int], top: int = 1) -> List[Tuple[np.ndarray, int, int]]:
        """
        Deprecated compatibility wrapper for the old API. Only supports top=1.
        Returns a single-element list with the closest match tuple.

        To migrate, switch to:
            find_closest_match(target_bit_code) -> (bits, identity, distance)
        """
        if top != 1:
            raise ValueError("This codebook only supports exact top-1 search. Call find_closest_match(...).")
        bits, identity, dist = self.find_closest_match(target_bit_code)
        # Preserve old return structure: list of a single tuple
        return [(np.asarray(bits, dtype=np.uint8), identity, dist)]
