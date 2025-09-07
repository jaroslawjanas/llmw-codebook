#!/usr/bin/env python3
"""
Codebook Generator for LLM Watermarking
Generates lookup tables mapping watermark identities to bit codes using pseudo-random functions.
"""

import os
import hashlib
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import src.paths as paths


class CodebookManager:
    """
    Generates and manages codebook tables for watermarking.
    Each table maps watermark identities to u-bit codes using a pseudo-random function.
    """
    
    def __init__(self, n: int, u: int, secret_key: str, npz_dir: Optional[str] = ""):
        """Initialize the codebook manager and load/generate the codebook arrays."""
        self.n = n
        self.u = u
        self.secret_key = secret_key

        self.filename_base = self._get_filename_base()

        if npz_dir:
            self.npz_path = os.path.join(npz_dir, f"{self.filename_base}.npz")
        else:
            self.npz_path = os.path.join(paths.CODEBOOKS_CACHE_DIR, f"{self.filename_base}.npz")
        
        # Load or generate the codebook arrays directly
        self.bit_codes, self.watermark_identities = self._load_or_generate_arrays()
    
    def _load_or_generate_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load existing codebook arrays or generate new ones.
        
        Returns:
            Tuple of (bit_codes_2d_array, watermark_identities_array)
        """
        # Check if cached NPZ file exists
        if os.path.exists(self.npz_path):
            print(f"Loading existing codebook from: {self.npz_path}")
            data = np.load(self.npz_path)
            bit_codes = data['bit_codes']
            watermark_identities = data['watermark_identities']
            print(f"Loaded codebook with {len(bit_codes)} entries, {bit_codes.shape[1]} bits each")
            return bit_codes, watermark_identities
        
        # Generate new arrays
        print(f"Generating new codebook: {self.filename_base}")
        bit_codes, watermark_identities = self._generate_arrays()
        
        # Save to NPZ cache with compression
        np.savez_compressed(
            self.npz_path, 
            bit_codes=bit_codes, 
            watermark_identities=watermark_identities
        )
        print(f"Saved codebook to: {self.npz_path}")
        
        return bit_codes, watermark_identities
    
    def _generate_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the codebook arrays directly without DataFrame overhead.
        
        Returns:
            Tuple of (bit_codes_2d_array, watermark_identities_array)
        """
        # Calculate table size
        table_size = 2 ** self.n
        print(f"Generating codebook with {table_size} entries...")
        
        # Generate watermark identities (0 to 2^n - 1) with int64 dtype
        watermark_identities = np.arange(table_size, dtype=np.int64)
        
        # Generate bit codes using vectorized operations
        bit_codes = self._generate_bit_codes_vectorized(watermark_identities)
        
        return bit_codes, watermark_identities
    
    def _get_filename_base(self) -> str:
        """
        Generate standardized filename base based on parameters.
        
        Returns:
            Filename base string
        """
        return f"codebook_n{self.n}_u{self.u}_sk_{self.secret_key}"
    
    def _generate_bit_codes_vectorized(self, identities: np.ndarray) -> np.ndarray:
        """
        Generate bit codes for all identities using vectorized operations.
        
        Args:
            identities: Array of watermark identities
            
        Returns:
            2D numpy array of bit codes (n_identities, u_bits)
        """
        # Create bit codes array
        bit_codes = np.zeros((len(identities), self.u), dtype=np.uint8)
        
        for i, identity in enumerate(identities):
            # Derive a deterministic per-identity seed from secret_key and identity only
            seed_material = f"{self.secret_key}:{int(identity)}".encode("utf-8")
            digest = hashlib.sha256(seed_material).digest()
            identity_seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(identity_seed)
            
            # Generate u random bits directly
            bit_codes[i] = rng.integers(0, 2, size=self.u, dtype=np.uint8)
        
        return bit_codes
    
    def get_random_row(self, seed: int) -> Tuple[List[int], int]:
        """
        Get a random watermark identity and its corresponding bit code using a seeded RNG.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (bit_code, watermark_identity)
        """
        # Create a seeded random number generator
        rng = np.random.default_rng(seed)
        
        # Generate a random watermark identity within the valid range [0, 2^n - 1]
        watermark_identity = rng.integers(0, 2 ** self.n)
        
        # Get the corresponding bit code directly from the array and convert to list
        bit_code = self.bit_codes[watermark_identity].tolist()
        
        return bit_code, watermark_identity
    
    def _calculate_hamming_distance_vectorized(self, target_code: np.ndarray, all_codes: np.ndarray) -> np.ndarray:
        """
        Calculate Hamming distances between target code and all codes using vectorization.
        
        Args:
            target_code: Target bit code as numpy array
            all_codes: All bit codes as 2D numpy array (n_codes, u_bits)
            
        Returns:
            Array of Hamming distances
        """
        # Vectorized comparison and sum - count differing bits
        return np.sum(target_code != all_codes, axis=1)
    
    def find_closest_matches(self, target_bit_code: List[int], top: int = 1) -> List[Tuple[np.ndarray, int, int]]:
        """
        Find the closest matching bit codes in the codebook based on Hamming distance.
        
        Args:
            target_bit_code: The bit code to search for
            top: Number of closest matches to return (default: 1)
            
        Returns:
            List of tuples (bit_code, watermark_identity, hamming_distance) sorted by distance
            
        Raises:
            ValueError: If target_bit_code has wrong length, contains invalid values, or top is invalid
        """
        # Input validation
        if len(target_bit_code) != self.u:
            return None
        
        if not all(bit in [0, 1] for bit in target_bit_code):
            raise ValueError("Target bit code must contain only 0s and 1s")
        
        if top <= 0:
            raise ValueError("Parameter 'top' must be a positive integer")
        
        if len(self.bit_codes) == 0:
            raise ValueError("Codebook is empty")
        
        # Convert target to numpy array for vectorized operations
        target_array = np.array(target_bit_code, dtype=np.uint8)
        
        # Calculate all distances at once using vectorization
        distances = self._calculate_hamming_distance_vectorized(target_array, self.bit_codes)
        
        # Use different strategies based on top value
        if top == 1:
            # Early termination for single result - find minimum directly
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            return [(self.bit_codes[min_idx], self.watermark_identities[min_idx], min_distance)]
        
        else:
            # Use partial sorting for efficiency when top << total_entries
            # argpartition: O(n) - partitions array so first 'top' elements are smallest (unsorted)
            top_indices = np.argpartition(distances, top)[:top]
            # argsort: O(k log k) - sort only the top k elements by their distance values
            top_indices = top_indices[np.argsort(distances[top_indices])]
            
            return [(self.bit_codes[idx], self.watermark_identities[idx], distances[idx]) 
                   for idx in top_indices]
