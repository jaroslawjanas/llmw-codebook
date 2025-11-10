"""
Legacy import shim for codebook implementations.

The codebase has been refactored into the modular `src.codebooks` package.
This module re-exports the Hamming-based codebook classes to maintain
backwards compatibility with existing imports elsewhere in the project.
"""

from __future__ import annotations

from src.codebooks.hamming import (
    CodebookBCH,
    CodebookRNG,
    HammingCodebookBCH,
    HammingCodebookBase,
    HammingCodebookRNG,
)

__all__ = [
    "HammingCodebookBase",
    "HammingCodebookRNG",
    "HammingCodebookBCH",
    "CodebookRNG",
    "CodebookBCH",
]
