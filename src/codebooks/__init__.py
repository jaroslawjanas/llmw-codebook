"""
Codebook package initialization.

Exports:
- CodebookBase: canonical base codebook (NPZ) generator/loader
- CodebookHamming: FAISS-backed Hamming distance search consuming the base codebook
- Generators: RNGCodewordGenerator, BCHCodewordGenerator (pluggable strategies)
"""

from .base import CodebookBase
from .hamming import CodebookHamming
from .levenshtein import CodebookLevenshtein
from .levenshtein_triton import CodebookLevenshteinTriton
from .generation import CodewordGenerator, RNGCodewordGenerator, BCHCodewordGenerator, LinearCodeGenerator

__all__ = [
    "CodebookBase",
    "CodebookHamming",
    "CodebookLevenshtein",
    "CodebookLevenshteinTriton",
    "CodewordGenerator",
    "RNGCodewordGenerator",
    "BCHCodewordGenerator",
    "LinearCodeGenerator",
]
