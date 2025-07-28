#!/usr/bin/env python3
"""
Codebook Generator for LLM Watermarking
Generates lookup tables mapping watermark identities to bit codes using pseudo-random functions.
"""

import os
import hashlib
import pandas as pd
import numpy as np
from typing import List
import src.paths as paths


class CodebookGenerator:
    """
    Generates and manages codebook tables for watermarking.
    Each table maps watermark identities to u-bit codes using a pseudo-random function.
    """
    
    def __init__(self, n: int, u: int, secret_key: str, seed: int = 42):
        """Initialize the codebook generator"""
        self.n = n
        self.u = u
        self.secret_key = secret_key
        self.seed = seed

        self.filename_base = self._get_filename_base(n, u, secret_key, seed)
        self.parquet_path = os.path.join(paths.CODEBOOKS_CACHE_DIR, f"{self.filename_base}.parquet")
        self.csv_path = os.path.join(paths.CODEBOOKS_CACHE_DIR, f"{self.filename_base}.csv")
    
    def get_codebook(self) -> pd.DataFrame:
        """
        Generate a codebook table or load existing one from cache.
            
        Returns:
            pandas DataFrame with columns ['watermark_identity', 'bit_code']
        """
        # Check if cached parquet version exists (fast loading)
        if os.path.exists(self.parquet_path):
            print(f"Loading existing codebook from: {self.parquet_path}")
            df = pd.read_parquet(self.parquet_path)
            # Convert bit_code arrays back to lists
            df['bit_code'] = df['bit_code'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else list(x))
            return df
        
        # Generate new table
        print(f"Generating new codebook: {self.filename_base}")
        table = self._generate_table(self.n, self.u, self.secret_key, self.seed)
        
        # Save to cache as parquet (fast loading)
        df_to_save = table.copy()
        # Convert bit lists to numpy arrays for efficient storage
        df_to_save['bit_code'] = df_to_save['bit_code'].apply(np.array)
        df_to_save.to_parquet(self.parquet_path, index=False)
        
        # Also save as CSV for human readability
        df_csv = table.copy()
        df_csv['bit_code'] = df_csv['bit_code'].apply(self._bit_list_to_string)
        df_csv.to_csv(self.csv_path, index=False)
        
        print(f"Saved codebook to: {self.parquet_path} (fast loading)")
        print(f"Saved human-readable copy to: {self.csv_path}")
        
        return table
    
    def _get_filename_base(self, n: int, u: int, secret_key: str, seed: int) -> str:
        """
        Generate standardized filename base based on parameters.
        
        Args:
            n: Field size parameter
            u: Bit code length
            secret_key: Secret key
            seed: Random seed
            
        Returns:
            Filename base string
        """
        # Create a short hash of the secret key for filename
        secret_hash = hashlib.md5(secret_key.encode()).hexdigest()[:8]
        return f"codebook_n{n}_u{u}_sk_{secret_hash}_s_{seed}"
    
    def _generate_table(self, n: int, u: int, secret_key: str, seed: int) -> pd.DataFrame:
        """
        Generate the codebook table with 2^n rows.
        
        Args:
            n: Field size parameter - table will have 2^n rows
            u: Bit code length - each code will be u bits long
            secret_key: Secret key for pseudo-random function
            seed: Random seed for reproducibility
            
        Returns:
            pandas DataFrame with columns ['watermark_identity', 'bit_code']
        """
        # Create a random generator with the specified seed
        # Might prove useful if this were to be changed to a different approach
        # rng = np.random.default_rng(seed)
        
        # Calculate table size
        table_size = 2 ** n
        print(f"Generating codebook with {table_size} entries...")
        
        # Generate watermark identities (0 to 2^n - 1)
        watermark_identities = list(range(table_size))
        
        # Generate bit codes using pseudo-random function
        bit_codes = []
        for identity in watermark_identities:
            bit_code = self._generate_bit_code(identity)
            bit_codes.append(bit_code)
        
        # Create DataFrame
        df = pd.DataFrame({
            'watermark_identity': watermark_identities,
            'bit_code': bit_codes
        })
        
        return df
    
    def _generate_bit_code(self, identity: int) -> List[int]:
        """
        Generate u-bit code for given identity using pseudo-random function.
        
        Args:
            identity: Watermark identity (0 to 2^n - 1)
            
        Returns:
            List of u bits (0s and 1s)
        """
        # Create hash input combining secret key and identity
        hash_input = f"{self.secret_key}-{identity}"
        hash_object = hashlib.sha256(hash_input.encode())
        hash_bytes = hash_object.digest()
        
        # Convert hash bytes to bits
        bit_code = []
        for i in range(self.u):
            # Use different bytes for different bits to maximize randomness
            byte_index = i // 8
            bit_index = i % 8
            
            # If we need more bytes than available, cycle through the hash
            if byte_index >= len(hash_bytes):
                byte_index = byte_index % len(hash_bytes)
            
            # Extract bit from byte
            byte_val = hash_bytes[byte_index]
            bit_val = (byte_val >> bit_index) & 1
            bit_code.append(bit_val)
        
        return bit_code
    
    def _bit_list_to_string(self, bit_list: List[int]) -> str:
        """
        Convert bit list to string representation for CSV storage.
        
        Args:
            bit_list: List of bits
            
        Returns:
            String representation
        """
        return ''.join(map(str, bit_list))


# Example usage function
def generate_standard_codebook():
    """
    Generate a standard codebook for common parameters.
    """
    # Example: Generate codebook with n=4 (16 entries) and u=8-bit codes
    n = 16
    u = 100
    secret_key = "42"
    seed = 42
    
    generator = CodebookGenerator(n, u, secret_key, seed)
    table = generator.get_codebook()
    
    print(f"Generated codebook with {len(table)} entries")
    print(f"First 5 rows:")
    print(table.head())
    print(f"Sample bit code: {table.iloc[0]['bit_code']}")
    
    return table


if __name__ == "__main__":
    # Test the codebook generation
    generate_standard_codebook()
