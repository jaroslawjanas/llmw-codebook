"""
Codebook package initialization.

Provides access to codeword generators and Hamming-based codebook implementations.
"""

from .generation import CodewordGenerator, RNGCodewordGenerator, BCHCodewordGenerator
from .hamming import (
    HammingCodebookBase,
    HammingCodebookRNG,
    HammingCodebookBCH,
    CodebookRNG,
    CodebookBCH,
)

__all__ = [
    "CodewordGenerator",
    "RNGCodewordGenerator",
    "BCHCodewordGenerator",
    "HammingCodebookBase",
    "HammingCodebookRNG",
    "HammingCodebookBCH",
    "CodebookRNG",
    "CodebookBCH",
]
