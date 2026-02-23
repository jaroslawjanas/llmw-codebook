"""
BCH parameter selection and validation.

Provides utilities to find optimal BCH code parameters for given message length (n)
and desired codeword length (u). The error correction capability (t) is determined
dynamically by galois when the BCH code is created.
"""

from __future__ import annotations

from dataclasses import dataclass


# Valid k values for BCH codes, keyed by m where n0 = 2^m - 1
# Each list is sorted by k descending (largest message capacity first)
# The actual t for each k is determined by galois.BCH(n0, k=k) at runtime
# Supports m=3..9 (n0 up to 511, covering u up to ~500)
BCH_VALID_K: dict[int, list[int]] = {
    3: [4, 1],
    4: [11, 7, 5, 1],
    5: [26, 21, 16, 11, 6, 1],
    6: [57, 51, 45, 39, 36, 30, 24, 18, 16, 10, 7, 1],
    7: [120, 113, 106, 99, 92, 85, 78, 71, 64, 57, 50, 43, 36, 29, 22, 15, 8, 1],
    8: [247, 239, 231, 223, 215, 207, 199, 191, 187, 179, 171, 163, 155, 147, 139, 131, 123, 115, 107, 99, 91, 87, 79, 71, 63, 55, 47, 45, 37, 29, 21, 13, 9, 1],
    9: [502, 493, 484, 475, 466, 457, 448, 439, 430, 421, 412, 403, 394, 385, 376, 367, 358, 349, 340, 331, 322, 313, 304, 295, 286, 277, 268, 259, 250, 241, 238, 229, 220, 211, 202, 193, 184, 175, 166, 157, 148, 139, 130, 121, 112, 103, 94, 85, 76, 67, 58, 49, 40, 31, 28, 19, 10, 1],
}


@dataclass
class BCHParams:
    """
    BCH code parameters for a configured codec.

    Attributes:
        m: Field degree. The native BCH codeword length is n0 = 2^m - 1.
           Valid range: 3 to 9 (n0 from 7 to 511).
        n0: Native BCH codeword length (n0 = 2^m - 1). This is the length
            of the BCH codeword BEFORE padding/puncturing.
        k: Message (information) length. The number of data bits the BCH code
            can protect. Must be >= the watermark size n.
        t: Error correction capability. The BCH code can correct up to t bit
            errors in the received codeword. This is set by BCHCodec after
            creating the galois.BCH instance (not from the lookup table).
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


def get_valid_k_values(n0: int) -> list[int]:
    """
    Get all valid k values for a given native BCH codeword length n0.

    Uses pre-computed lookup table for instant results (m=3..9, n0 up to 511).

    Returns:
        List of valid k values sorted by k descending.

    Raises:
        ValueError: If n0 is not a valid BCH length or not in lookup table.
    """
    # Calculate m from n0: n0 = 2^m - 1
    m = (n0 + 1).bit_length() - 1

    # Validate n0 is a valid BCH length
    if 2**m - 1 != n0:
        raise ValueError(f"n0={n0} is not a valid BCH codeword length (must be 2^m - 1)")

    if m not in BCH_VALID_K:
        raise ValueError(
            f"m={m} (n0={n0}) not in lookup table. "
            f"Supported: m=3..9 (n0=7..511, u up to ~500)"
        )

    return BCH_VALID_K[m]


def find_smallest_k(n0: int, min_k: int) -> int | None:
    """
    Find the smallest valid k >= min_k for a given n0.

    Smaller k means more parity bits, which gives higher error correction (t).
    We want the smallest k that still fits our message.

    Returns:
        Smallest k >= min_k, or None if no valid k exists.
    """
    valid_k = get_valid_k_values(n0)

    # valid_k is sorted descending, iterate to find smallest >= min_k
    result = None
    for k in valid_k:
        if k >= min_k:
            result = k  # Keep updating (last valid one is smallest)

    return result


def select_bch_parameters(
    n: int,
    u: int,
    m: int | None = None,
    allow_padding: bool = True,
) -> BCHParams:
    """
    Select optimal BCH parameters for given message and codeword lengths.

    Args:
        n: Message length in bits (watermark size)
        u: Desired output codeword length in bits
        m: Optional manual field degree (n0 = 2^m - 1)
        allow_padding: If True, allows n0 < u (padding with zeros). If False,
                       requires n0 >= u (for codebook generation which truncates).

    Returns:
        BCHParams with optimal configuration. Note: t is set to 0 as a placeholder;
        the actual t is determined by BCHCodec when creating the galois.BCH instance.

    Raises:
        ValueError: If no valid BCH code exists for the given constraints
    """
    if n <= 0:
        raise ValueError(f"Message length n must be positive, got {n}")
    if u <= 0:
        raise ValueError(f"Codeword length u must be positive, got {u}")
    if n >= u:
        raise ValueError(f"Message length n={n} must be less than codeword length u={u}")

    if m is not None:
        return _select_manual_m(n, u, m, allow_padding)
    else:
        return _select_auto(n, u, allow_padding)


def _select_manual_m(n: int, u: int, m: int, allow_padding: bool) -> BCHParams:
    """Select BCH with manually specified m, auto-select k."""
    n0 = 2**m - 1

    if not allow_padding and n0 < u:
        raise ValueError(
            f"BCH(n0={n0}) < u={u} requires padding, but padding is not allowed. "
            f"Increase --bch-m to get n0 >= u."
        )

    k = find_smallest_k(n0, min_k=n)
    if k is None:
        raise ValueError(
            f"No valid BCH code for m={m} (n0={n0}) with k >= {n}. "
            f"Increase --bch-m."
        )

    padding = max(0, u - n0)
    puncturing = max(0, n0 - u)

    # t=0 is a placeholder; actual t is set by BCHCodec
    return BCHParams(m=m, n0=n0, k=k, t=0, padding=padding, puncturing=puncturing)


def _select_auto(n: int, u: int, allow_padding: bool) -> BCHParams:
    """
    Fully automatic BCH parameter selection.

    Strategy when allow_padding=True (BCH mode):
    1. Find all valid (m, n0, k) where k >= n
    2. Prefer exact match (n0 == u), then puncturing (smallest n0 > u)
    3. Fallback to n0 < u (padding) only when no exact or puncturing option exists

    Strategy when allow_padding=False (codebook generation):
    1. Find all valid (m, n0, k) where k >= n AND n0 >= u
    2. Prefer exact match (n0 == u), then smallest n0 > u (minimize puncturing)
    """
    # Validate within lookup table range (m=3..9 covers n0 up to 511)
    if n > 500 or u > 500:
        raise ValueError(
            f"Auto-selection only supports n <= 500 and u <= 500. "
            f"Got n={n}, u={u}. Please specify --bch-m manually."
        )

    viable: list[tuple[int, int, int]] = []  # (m, n0, k)

    for m_candidate in sorted(BCH_VALID_K.keys()):
        n0 = 2**m_candidate - 1
        k = find_smallest_k(n0, min_k=n)
        if k is not None:
            viable.append((m_candidate, n0, k))

    if not viable:
        raise ValueError(
            f"No valid BCH code found for message length n={n}. "
            f"This is unexpected for reasonable n values."
        )

    # Prefer: exact match > puncturing (all bits BCH-protected) > padding fallback
    exact_options = [(m, n0, k) for m, n0, k in viable if n0 == u]
    puncturing_options = [(m, n0, k) for m, n0, k in viable if n0 > u]
    padding_options = [(m, n0, k) for m, n0, k in viable if n0 < u]

    if exact_options:
        best = exact_options[0]
    elif puncturing_options:
        # Pick smallest n0 > u (minimize puncturing, all bits BCH-protected)
        best = min(puncturing_options, key=lambda x: x[1])
    elif allow_padding and padding_options:
        # Fallback: largest n0 < u (minimize unprotected zero-padding)
        best = max(padding_options, key=lambda x: x[1])
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

    m, n0, k = best
    padding = max(0, u - n0)
    puncturing = max(0, n0 - u)

    # t=0 is a placeholder; actual t is set by BCHCodec
    return BCHParams(m=m, n0=n0, k=k, t=0, padding=padding, puncturing=puncturing)


def print_bch_warnings(n: int, u: int, params: BCHParams) -> None:
    """
    Print warnings when BCH configuration is not a perfect fit.

    Args:
        n: Requested message length
        u: Requested codeword length
        params: Selected BCH parameters
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
