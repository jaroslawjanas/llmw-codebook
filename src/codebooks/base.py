"""
Abstract base classes for codebook implementations.

Provides a shared scaffold for deterministic codebook generation, cache management,
and public API surface (get_random_row / find_closest_match) while delegating storage
and indexing details to concrete subclasses.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np

import src.paths as paths
from .generation import CodewordGenerator


class CodebookBase(ABC):
    """
    Base class for codebook implementations backed by deterministic generators.

    Subclasses must implement storage/index specifics via the abstract hooks.
    """

    def __init__(
        self,
        *,
        n: int,
        u: int,
        secret_key: str,
        generator_cls: Type[CodewordGenerator],
        generator_kwargs: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        workers: int = 4,
        chunk_rows: Optional[int] = None,
    ) -> None:
        self.n = int(n)
        self.u = int(u)
        self.secret_key = str(secret_key)
        self.workers = max(1, int(workers))
        self.chunk_rows = None if chunk_rows is None else max(1, int(chunk_rows))

        generator_kwargs = generator_kwargs or {}
        self.generator: CodewordGenerator = generator_cls(
            n=self.n,
            u=self.u,
            secret_key=self.secret_key,
            **generator_kwargs,
        )

        cache_root = cache_dir or paths.CODEBOOKS_CACHE_DIR
        os.makedirs(cache_root, exist_ok=True)
        self.cache_dir = cache_root

        self.filename_base = self._build_filename_base()
        self._configure_paths()
        self._load_or_build()

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _configure_paths(self) -> None:
        """
        Initialize filesystem-related attributes (e.g., cache/index paths).
        Called once during __init__.
        """

    @abstractmethod
    def _cache_exists(self) -> bool:
        """
        Whether all artifacts required by the subclass are present on disk.
        """

    @abstractmethod
    def _store_generated_codewords(self, codewords: np.ndarray) -> None:
        """
        Persist freshly generated codewords & build any indices required.
        """

    @abstractmethod
    def _load_cache(self) -> None:
        """
        Load cached artifacts into memory (codeword storage, index handles, etc.).
        """

    @abstractmethod
    def _get_codeword_bits_by_index(self, index: int) -> np.ndarray:
        """
        Return the bit vector for a given watermark index.
        """

    @abstractmethod
    def find_closest_match(self, target_codeword: np.ndarray) -> tuple[list[int], int, int]:
        """
        Locate the closest codeword according to the subclass distance metric.
        Must return (closest_codeword_bits, watermark_identity, distance_value).
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_random_row(self, seed: int) -> tuple[list[int], int]:
        rng = np.random.default_rng(seed)
        watermark = int(rng.integers(0, 2 ** self.n))
        bits = self._get_codeword_bits_by_index(watermark)
        return bits.tolist(), watermark

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filename_base(self) -> str:
        return f"codebook_{self.generator.algorithm_tag}_n{self.n}_u{self.u}_sk_{self.secret_key}"

    def _load_or_build(self) -> None:
        if self._cache_exists():
            self._load_cache()
            return

        raw_codewords = self.generator.generate(
            workers=self.workers,
            chunk_rows=self.chunk_rows,
        )
        self._store_generated_codewords(raw_codewords)
        self._load_cache()
