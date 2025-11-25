"""
Codebook package initialization.

Exports:
- CodebookBase: canonical base codebook (NPZ) generator/loader
- CodebookHamming: FAISS-backed Hamming distance search consuming the base codebook
- Generators: RNGCodewordGenerator, BCHCodewordGenerator (pluggable strategies)
"""

from .base import CodebookBase
from .hamming import CodebookHamming
from .generation import CodewordGenerator, RNGCodewordGenerator, BCHCodewordGenerator

__all__ = [
    "CodebookBase",
    "CodebookHamming",
    "CodewordGenerator",
    "RNGCodewordGenerator",
    "BCHCodewordGenerator",
]
