"""
DEPRECATED MODULE

This module has been removed in favor of the modular package `src.codebooks`.

New API:
- from src.codebooks import CodebookBase, CodebookHamming
- from src.codebooks import RNGCodewordGenerator, BCHCodewordGenerator

Example:
    from src.codebooks import CodebookBase, CodebookHamming, RNGCodewordGenerator

    base = CodebookBase(
        n=24, u=128, secret_key="25111915",
        generator_cls=RNGCodewordGenerator,
    )
    codebook = CodebookHamming(base)
    bits, wm = codebook.get_random_row(seed=42)
    closest_bits, wm_id, dist = codebook.find_closest_match(bits)
"""

raise ImportError(
    "src.codebook is deprecated. Use src.codebooks "
    "(CodebookBase, CodebookHamming, RNGCodewordGenerator, BCHCodewordGenerator)."
)
