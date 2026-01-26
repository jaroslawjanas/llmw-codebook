"""
BCH parameter selection and validation.

Provides utilities to find optimal BCH code parameters for given message length (n)
and desired codeword length (u), maximizing error correction capability (t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    import galois
except ImportError as e:
    raise ImportError(
        "The 'galois' package is required for BCH codes. "
        "Install it with: pip install galois"
    ) from e


@dataclass
class BCHParams:
    """
    BCH code parameters for a configured codec.

    Attributes:
        m: Field degree. The native BCH codeword length is n0 = 2^m - 1.
           Valid range: 3 to 16 (n0 from 7 to 65535).
        n0: Native BCH codeword length (n0 = 2^m - 1). This is the length
            of the BCH codeword BEFORE padding/puncturing.
        k: Message (information) length. The number of data bits the BCH code
            can protect. Must be >= the watermark size n.
        t: Error correction capability. The BCH code can correct up to t bit
            errors in the received codeword.
        padding: Number of zero bits appended when u > n0.
                 These padding bits are unprotected by BCH error correction.
        puncturing: Number of bits removed when u < n0.
                    Puncturing reduces error correction capability since these
                    positions become erasures at the decoder.
    """
    m: int
    n0: int
    k: int
    t: int
    padding: int
    puncturing: int

    @property
    def is_perfect_fit(self) -> bool:
        """True if no padding or puncturing needed."""
        return self.padding == 0 and self.puncturing == 0


def get_valid_bch_codes(n0: int) -> List[Tuple[int, int]]:
    """
    Get all valid (k, t) pairs for a given native BCH codeword length n0.

    Not all combinations of (n0, k, t) are valid BCH codes. This function
    enumerates all valid codes by trying different design distances d = 2t+1
    and recording which k values are achievable.

    Returns:
        List of (k, t) tuples sorted by k descending. The smallest k for
        a given n0 corresponds to the largest t (most error correction).
    """
    valid = []
    # Try different values of t to find valid codes
    for t_candidate in range(1, n0 // 2 + 1):
        try:
            d = 2 * t_candidate + 1
            bch = galois.BCH(n0, d=d)
            k = int(bch.k)
            valid.append((k, t_candidate))
        except (ValueError, Exception):
            # Invalid combination, skip
            continue

    # Remove duplicates and sort by k descending
    valid = list(set(valid))
    valid.sort(key=lambda x: -x[0])
    return valid


def find_best_k_t_for_n0(n0: int, min_k: int) -> Optional[Tuple[int, int]]:
    """
    Find the (k, t) pair for a given n0 that maximizes error correction.

    For a fixed n0, smaller k means larger t (more parity bits).
    We want the smallest k that is still >= min_k (to fit our message).

    Returns:
        (k, t) tuple with maximum t such that k >= min_k.
        Returns None if no valid code exists for this n0.
    """
    valid_codes = get_valid_bch_codes(n0)

    # Find smallest k >= min_k (which gives maximum t)
    best = None
    for k, t in valid_codes:
        if k >= min_k:
            if best is None or t > best[1]:
                best = (k, t)

    return best


def select_bch_parameters(
    n: int,
    u: int,
    m: Optional[int] = None,
    t: Optional[int] = None,
    allow_padding: bool = True,
) -> BCHParams:
    """
    Select optimal BCH parameters for given message and codeword lengths.

    Args:
        n: Message length in bits (watermark size)
        u: Desired output codeword length in bits
        m: Optional manual field degree (n0 = 2^m - 1)
        t: Optional manual error correction capability
        allow_padding: If True, allows n0 < u (padding with zeros). If False,
                       requires n0 >= u (for codebook generation which truncates).

    Returns:
        BCHParams with optimal configuration

    Raises:
        ValueError: If no valid BCH code exists for the given constraints
    """
    if n <= 0:
        raise ValueError(f"Message length n must be positive, got {n}")
    if u <= 0:
        raise ValueError(f"Codeword length u must be positive, got {u}")
    if n >= u:
        raise ValueError(f"Message length n={n} must be less than codeword length u={u}")

    if m is not None and t is not None:
        # Fully manual: validate and use
        return _select_manual_m_t(n, u, m, t, allow_padding)
    elif m is not None:
        # Manual m, auto t
        return _select_manual_m(n, u, m, allow_padding)
    elif t is not None:
        # Manual t, auto m
        return _select_manual_t(n, u, t, allow_padding)
    else:
        # Full auto
        return _select_auto(n, u, allow_padding)


def _select_manual_m_t(n: int, u: int, m: int, t: int, allow_padding: bool) -> BCHParams:
    """Select BCH with manually specified m and t."""
    n0 = 2**m - 1

    try:
        d = 2 * t + 1
        bch = galois.BCH(n0, d=d)
        k = int(bch.k)
    except Exception as e:
        raise ValueError(
            f"BCH(n0={n0}, t={t}) is not a valid code. "
            f"Underlying error: {e}"
        ) from e

    if k < n:
        raise ValueError(
            f"BCH(n0={n0}, t={t}) has k={k} < n={n}. "
            f"Reduce --bch-t or increase --bch-m."
        )

    if not allow_padding and n0 < u:
        raise ValueError(
            f"BCH(n0={n0}) < u={u} requires padding, but padding is not allowed. "
            f"Increase --bch-m to get n0 >= u."
        )

    padding = max(0, u - n0)
    puncturing = max(0, n0 - u)

    return BCHParams(m=m, n0=n0, k=k, t=t, padding=padding, puncturing=puncturing)


def _select_manual_m(n: int, u: int, m: int, allow_padding: bool) -> BCHParams:
    """Select BCH with manually specified m, auto-select t."""
    n0 = 2**m - 1

    if not allow_padding and n0 < u:
        raise ValueError(
            f"BCH(n0={n0}) < u={u} requires padding, but padding is not allowed. "
            f"Increase --bch-m to get n0 >= u."
        )

    result = find_best_k_t_for_n0(n0, min_k=n)
    if result is None:
        raise ValueError(
            f"No valid BCH code for m={m} (n0={n0}) with k >= {n}. "
            f"Increase --bch-m."
        )

    k, t = result
    padding = max(0, u - n0)
    puncturing = max(0, n0 - u)

    return BCHParams(m=m, n0=n0, k=k, t=t, padding=padding, puncturing=puncturing)


def _select_manual_t(n: int, u: int, t: int, allow_padding: bool) -> BCHParams:
    """Select BCH with manually specified t, auto-select m."""
    # Find smallest m that supports this t with k >= n
    # If padding not allowed, also require n0 >= u
    for m_candidate in range(3, 17):
        n0 = 2**m_candidate - 1
        if not allow_padding and n0 < u:
            continue  # Skip, would need padding
        try:
            d = 2 * t + 1
            bch = galois.BCH(n0, d=d)
            k = int(bch.k)
            if k >= n:
                padding = max(0, u - n0)
                puncturing = max(0, n0 - u)
                return BCHParams(m=m_candidate, n0=n0, k=k, t=t, padding=padding, puncturing=puncturing)
        except Exception:
            continue

    raise ValueError(
        f"No valid BCH code with t={t} and k >= {n} found. "
        f"Try a smaller --bch-t value."
    )


def _select_auto(n: int, u: int, allow_padding: bool) -> BCHParams:
    """
    Fully automatic BCH parameter selection.

    Strategy when allow_padding=True (BCH mode):
    1. Find all valid (m, n0, k, t) where k >= n
    2. Prefer n0 <= u (padding case) to minimize unprotected bits
    3. Among padding options, pick largest n0 (minimize padding)
    4. If no n0 <= u, pick smallest n0 > u (minimize puncturing)

    Strategy when allow_padding=False (codebook generation):
    1. Find all valid (m, n0, k, t) where k >= n AND n0 >= u
    2. Pick smallest n0 >= u (minimize puncturing/truncation)
    """
    viable: List[Tuple[int, int, int, int]] = []  # (m, n0, k, t)

    for m_candidate in range(3, 17):
        n0 = 2**m_candidate - 1
        result = find_best_k_t_for_n0(n0, min_k=n)
        if result is not None:
            k, t = result
            viable.append((m_candidate, n0, k, t))

    if not viable:
        raise ValueError(
            f"No valid BCH code found for message length n={n}. "
            f"This is unexpected for reasonable n values."
        )

    # Separate into padding (n0 <= u) and puncturing/truncation (n0 > u) options
    padding_options = [(m, n0, k, t) for m, n0, k, t in viable if n0 <= u]
    puncturing_options = [(m, n0, k, t) for m, n0, k, t in viable if n0 > u]

    if allow_padding and padding_options:
        # Pick largest n0 <= u (minimize zero-padding)
        best = max(padding_options, key=lambda x: x[1])
    elif puncturing_options:
        # Pick smallest n0 > u (minimize puncturing/truncation)
        best = min(puncturing_options, key=lambda x: x[1])
    elif padding_options:
        # Only padding options available but padding not allowed
        raise ValueError(
            f"No BCH code with n0 >= u={u} found for n={n}. "
            f"All viable codes have n0 < u, which requires padding."
        )
    else:
        raise ValueError(
            f"No viable BCH configuration for n={n}, u={u}. "
            f"This should not happen."
        )

    m, n0, k, t = best
    padding = max(0, u - n0)
    puncturing = max(0, n0 - u)

    return BCHParams(m=m, n0=n0, k=k, t=t, padding=padding, puncturing=puncturing)


def print_bch_warnings(
    n: int,
    u: int,
    params: BCHParams,
    manual_m: bool = False,
    manual_t: bool = False,
) -> None:
    """
    Print warnings when BCH configuration is not a perfect fit.

    Args:
        n: Requested message length
        u: Requested codeword length
        params: Selected BCH parameters
        manual_m: Whether m was manually specified
        manual_t: Whether t was manually specified
    """
    warnings = []

    if params.padding > 0:
        warnings.append(
            f"Padding {params.padding} zero bits (unprotected by BCH). "
            f"Consider --u {params.n0} for exact BCH match."
        )

    if params.puncturing > 0:
        warnings.append(
            f"Puncturing {params.puncturing} bits (reduced error correction). "
            f"Consider --u {params.n0} for exact BCH match."
        )

    if params.k > n:
        warnings.append(
            f"Message uses {n}/{params.k} available bits. "
            f"Consider --n {params.k} for full message capacity."
        )

    # If manual params, check if auto would give better t
    if (manual_m or manual_t) and not (manual_m and manual_t):
        try:
            auto_params = _select_auto(n, u, allow_padding=True)
            if auto_params.t > params.t:
                warnings.append(
                    f"Auto-selection would give t={auto_params.t} "
                    f"(vs current t={params.t})."
                )
        except ValueError:
            pass

    for w in warnings:
        print(f"Warning: {w}")


def format_bch_summary(params: BCHParams, n: int) -> str:
    """Format a summary string for the BCH configuration."""
    action = ""
    if params.padding > 0:
        action = f", padding {params.padding} zeros"
    elif params.puncturing > 0:
        action = f", puncturing {params.puncturing} bits"

    return (
        f"BCH(n0={params.n0}, k={params.k}, t={params.t}) "
        f"[m={params.m}, message={n}/{params.k} bits{action}]"
    )
