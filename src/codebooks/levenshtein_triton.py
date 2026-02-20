"""
Triton GPU-accelerated Levenshtein-distance codebook search using Myers'
bit-parallel algorithm, with one GPU thread per codebook entry.

Key difference from the failed PyTorch approach: the entire Myers' sequential
loop (over text positions) runs INSIDE a single Triton kernel per codebook
entry, keeping all state in GPU registers. This avoids the per-iteration
kernel-launch overhead that made the PyTorch version 10x slower than Numba.

Responsibilities:
- Consume CodebookBase (canonical codewords stored as packed bits in NPZ)
- Build and cache uint64-packed codebook (same NPZ format as levenshtein.py)
- Provide nearest-neighbor search by edit distance on GPU
- Support both single-query and batch-query search
- Delegate non-search helpers (like get_random_row) to the base

File naming (shared with CPU Levenshtein):
- Base:        codebook_{algo}_n{n}_u{u}_sk-<secret-key>.npz
- Levenshtein: codebook_{algo}_n{n}_u{u}_sk-<secret-key>_metric-levenshtein.npz
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from .base import CodebookBase

__all__ = ["CodebookLevenshteinTriton"]


# ---------------------------------------------------------------------------
# Pure-numpy helpers (no Numba dependency)
# ---------------------------------------------------------------------------
# Bit packing convention: **little-endian within each uint64 word**.
#   bit[i] is stored at position (i % 64) of word (i // 64).
#   i.e. LSB of word 0 = bit 0, MSB of word 0 = bit 63, LSB of word 1 = bit 64, ...
# This aligns with Myers' original formulation where left-shift moves to higher
# pattern positions, and the boundary condition sets bit 0 (LSB).
# ---------------------------------------------------------------------------


def _convert_uint8_be_to_int64_le(packed_uint8: np.ndarray, u: int) -> np.ndarray:
    """Convert codebook from uint8 big-endian packed to int64 LE-packed.

    Processes in row-chunks to cap working memory at ~64 MB regardless of
    codebook size (the previous all-at-once approach created a temporary
    (N, num_words, 64) uint64 array — 32+ GB for n=24 u=256).

    Args:
        packed_uint8: (N, ceil(u/8)) uint8 array from CodebookBase.
        u: codeword length in bits.

    Returns:
        (N, num_words) array, dtype int64 (same bits as uint64, reinterpreted).
    """
    N = packed_uint8.shape[0]
    num_words = (u + 63) // 64
    result = np.empty((N, num_words), dtype=np.int64)

    # Cap intermediate memory at ~64 MB: each row expands to (num_words, 64) uint64
    row_bytes = num_words * 64 * 8  # bytes per row in the uint64 intermediate
    chunk_rows = max(1, (64 * 1024 * 1024) // row_bytes)

    # LE bit weights — computed once, reused across chunks
    powers = np.uint64(1) << np.arange(64, dtype=np.uint64)  # (64,)

    for start in range(0, N, chunk_rows):
        end = min(start + chunk_rows, N)
        chunk = packed_uint8[start:end]

        all_bits = np.unpackbits(chunk, axis=1, bitorder="big")[:, :u]

        chunk_n = end - start
        padded = np.zeros((chunk_n, num_words * 64), dtype=np.uint8)
        padded[:, :u] = all_bits
        padded = padded.reshape(chunk_n, num_words, 64)

        result[start:end] = (padded.astype(np.uint64) * powers).sum(axis=2).view(np.int64)

    return result


def _pack_query_to_int64(bits: np.ndarray, length: int) -> np.ndarray:
    """Pack 1D uint8 {0,1} array to int64 LE words.

    Args:
        bits: 1D uint8 array of {0,1} values.
        length: number of valid bits.

    Returns:
        (num_words,) int64 array.
    """
    num_words = (length + 63) // 64
    padded = np.zeros(num_words * 64, dtype=np.uint8)
    padded[:length] = bits[:length]
    padded = padded.reshape(num_words, 64)
    powers = np.uint64(1) << np.arange(64, dtype=np.uint64)
    packed = (padded.astype(np.uint64) * powers).sum(axis=1)
    return packed.view(np.int64)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
# We generate specialized kernels for different numbers of 64-bit blocks
# (NUM_BLOCKS = ceil(query_len / 64)).  For typical use cases:
#   u <= 64:   NUM_BLOCKS=1 (single-word)
#   u <= 128:  NUM_BLOCKS=2 (most common: u=100, u=127, u=128)
#   u <= 192:  NUM_BLOCKS=3
#
# Each program instance processes one codebook entry. The entire Myers'
# sequential loop over text positions runs in-kernel with state in registers.
# ---------------------------------------------------------------------------

@triton.jit
def _myers_lev_kernel_1block(
    # Pointers
    codebook_ptr,       # (N, num_cb_words) int64 — codebook packed
    query_w0: tl.int64, # query word 0
    result_ptr,         # (N,) int32 — output distances
    # Dimensions
    N: tl.int32,        # number of codebook entries
    text_len: tl.int32, # code length (u)
    num_cb_words: tl.int32,  # stride: words per codebook row
    last_mask: tl.int64,
    high_bit: tl.int64,
    query_len: tl.int32,
    # Grid
    BLOCK_SIZE: tl.constexpr,  # entries per program
):
    """Myers' kernel for NUM_BLOCKS=1 (query_len <= 64)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Initialize Pv, Mv, score
    Pv0 = last_mask  # scalar broadcast
    Mv0: tl.int64 = 0
    score = query_len  # broadcast to vector implicitly via ops with offs

    # We need score as a vector — initialize from scalar
    score_vec = tl.full([BLOCK_SIZE], query_len, dtype=tl.int32)

    q0 = query_w0
    q0_masked = q0 & last_mask
    nq0_masked = (~q0) & last_mask

    # Vectorize Pv, Mv across entries
    Pv0_vec = tl.full([BLOCK_SIZE], Pv0, dtype=tl.int64)
    Mv0_vec = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)

    # Process each text position
    for i in range(text_len):
        word_idx = i // 64
        bit_pos = i % 64

        # Load text_packed[entry, word_idx] for each entry
        cb_offs = offs * num_cb_words + word_idx
        text_word = tl.load(codebook_ptr + cb_offs, mask=mask, other=0).to(tl.int64)
        text_bit = (text_word >> bit_pos) & 1  # (BLOCK_SIZE,) 0 or 1

        # Eq = query_word if text_bit==1, else ~query_word (masked)
        Eq = tl.where(text_bit == 1, q0_masked, nq0_masked)

        # Xv = Eq | Mv
        Xv = Eq | Mv0_vec

        # sum = (Eq & Pv) + Pv  (with overflow handled by unsigned semantics)
        a = Eq & Pv0_vec
        sum_val = a + Pv0_vec

        # Xh = (sum ^ Pv) | Eq
        Xh = (sum_val ^ Pv0_vec) | Eq

        # Ph = Mv | ~(Xh | Pv)
        Ph = Mv0_vec | (~(Xh | Pv0_vec))
        # Mh = Pv & Xh
        Mh = Pv0_vec & Xh

        # Mask
        Ph = Ph & last_mask
        Mh = Mh & last_mask

        # Score update from high bit
        ph_high = (Ph & high_bit) != 0
        mh_high = (Mh & high_bit) != 0
        score_vec = score_vec + ph_high.to(tl.int32) - mh_high.to(tl.int32)

        # Shift Ph, Mh left by 1; set bit 0 of Ph (boundary)
        Ph_shifted = (Ph << 1) | 1
        Mh_shifted = Mh << 1

        # Update Pv, Mv
        Pv0_vec = (Mh_shifted | (~(Xv | Ph_shifted))) & last_mask
        Mv0_vec = (Ph_shifted & Xv) & last_mask

    # Store results
    tl.store(result_ptr + offs, score_vec, mask=mask)


@triton.jit
def _myers_lev_kernel_2block(
    # Pointers
    codebook_ptr,
    query_w0: tl.int64,
    query_w1: tl.int64,
    result_ptr,
    # Dimensions
    N: tl.int32,
    text_len: tl.int32,
    num_cb_words: tl.int32,
    last_mask: tl.int64,
    high_bit: tl.int64,
    query_len: tl.int32,
    # Grid
    BLOCK_SIZE: tl.constexpr,
):
    """Myers' kernel for NUM_BLOCKS=2 (64 < query_len <= 128)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    ALL_ONES: tl.constexpr = -1  # 0xFFFFFFFFFFFFFFFF as int64

    # Block 0: full 64 bits valid (mask = all-ones)
    Pv0 = tl.full([BLOCK_SIZE], ALL_ONES, dtype=tl.int64)
    Mv0 = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
    # Block 1: last_mask bits valid
    Pv1 = tl.full([BLOCK_SIZE], last_mask, dtype=tl.int64)
    Mv1 = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)

    score_vec = tl.full([BLOCK_SIZE], query_len, dtype=tl.int32)

    q0 = query_w0
    nq0 = ~query_w0
    q1_masked = query_w1 & last_mask
    nq1_masked = (~query_w1) & last_mask

    for i in range(text_len):
        word_idx = i // 64
        bit_pos = i % 64

        cb_offs = offs * num_cb_words + word_idx
        text_word = tl.load(codebook_ptr + cb_offs, mask=mask, other=0).to(tl.int64)
        text_bit = (text_word >> bit_pos) & 1

        # ---- Block 0 (full 64-bit word, no masking needed) ----
        Eq0 = tl.where(text_bit == 1, q0, nq0)
        Xv0 = Eq0 | Mv0

        a0 = Eq0 & Pv0
        sum0 = a0 + Pv0
        # Carry detection for unsigned addition (& 1 needed: Triton >> is arithmetic)
        carry0 = (((a0 & Pv0) | ((a0 | Pv0) & (~sum0))) >> 63) & 1

        Xh0 = (sum0 ^ Pv0) | Eq0
        Ph0 = Mv0 | (~(Xh0 | Pv0))
        Mh0 = Pv0 & Xh0

        # Shift Ph0, Mh0 left by 1; carry MSB to block 1
        ph0_msb = (Ph0 >> 63) & 1
        mh0_msb = (Mh0 >> 63) & 1
        Ph0_shifted = (Ph0 << 1) | 1  # boundary condition: set bit 0
        Mh0_shifted = Mh0 << 1

        # Update Pv0, Mv0
        Pv0 = Mh0_shifted | (~(Xv0 | Ph0_shifted))
        Mv0 = Ph0_shifted & Xv0

        # ---- Block 1 (last block, masked) ----
        Eq1 = tl.where(text_bit == 1, q1_masked, nq1_masked)
        Xv1 = Eq1 | Mv1

        a1 = Eq1 & Pv1
        sum1_no_carry = a1 + Pv1
        sum1 = sum1_no_carry + carry0

        Xh1 = (sum1 ^ Pv1) | Eq1
        Ph1 = Mv1 | (~(Xh1 | Pv1))
        Mh1 = Pv1 & Xh1

        Ph1 = Ph1 & last_mask
        Mh1 = Mh1 & last_mask

        # Score update from high bit of last block
        ph1_high = (Ph1 & high_bit) != 0
        mh1_high = (Mh1 & high_bit) != 0
        score_vec = score_vec + ph1_high.to(tl.int32) - mh1_high.to(tl.int32)

        # Shift Ph1, Mh1 left by 1, carry in from block 0
        Ph1_shifted = (Ph1 << 1) | ph0_msb
        Mh1_shifted = (Mh1 << 1) | mh0_msb

        # Update Pv1, Mv1
        Pv1 = (Mh1_shifted | (~(Xv1 | Ph1_shifted))) & last_mask
        Mv1 = (Ph1_shifted & Xv1) & last_mask

    tl.store(result_ptr + offs, score_vec, mask=mask)


@triton.jit
def _myers_lev_kernel_3block(
    codebook_ptr,
    query_w0: tl.int64,
    query_w1: tl.int64,
    query_w2: tl.int64,
    result_ptr,
    N: tl.int32,
    text_len: tl.int32,
    num_cb_words: tl.int32,
    last_mask: tl.int64,
    high_bit: tl.int64,
    query_len: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Myers' kernel for NUM_BLOCKS=3 (128 < query_len <= 192)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    ALL_ONES: tl.constexpr = -1

    Pv0 = tl.full([BLOCK_SIZE], ALL_ONES, dtype=tl.int64)
    Mv0 = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
    Pv1 = tl.full([BLOCK_SIZE], ALL_ONES, dtype=tl.int64)
    Mv1 = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
    Pv2 = tl.full([BLOCK_SIZE], last_mask, dtype=tl.int64)
    Mv2 = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)

    score_vec = tl.full([BLOCK_SIZE], query_len, dtype=tl.int32)

    q0 = query_w0
    nq0 = ~query_w0
    q1 = query_w1
    nq1 = ~query_w1
    q2_masked = query_w2 & last_mask
    nq2_masked = (~query_w2) & last_mask

    for i in range(text_len):
        word_idx = i // 64
        bit_pos = i % 64

        cb_offs = offs * num_cb_words + word_idx
        text_word = tl.load(codebook_ptr + cb_offs, mask=mask, other=0).to(tl.int64)
        text_bit = (text_word >> bit_pos) & 1

        # ---- Block 0 ----
        Eq0 = tl.where(text_bit == 1, q0, nq0)
        Xv0 = Eq0 | Mv0
        a0 = Eq0 & Pv0
        sum0 = a0 + Pv0
        carry0 = (((a0 & Pv0) | ((a0 | Pv0) & (~sum0))) >> 63) & 1
        Xh0 = (sum0 ^ Pv0) | Eq0
        Ph0 = Mv0 | (~(Xh0 | Pv0))
        Mh0 = Pv0 & Xh0
        ph0_msb = (Ph0 >> 63) & 1
        mh0_msb = (Mh0 >> 63) & 1
        Ph0_shifted = (Ph0 << 1) | 1
        Mh0_shifted = Mh0 << 1
        Pv0 = Mh0_shifted | (~(Xv0 | Ph0_shifted))
        Mv0 = Ph0_shifted & Xv0

        # ---- Block 1 ----
        Eq1 = tl.where(text_bit == 1, q1, nq1)
        Xv1 = Eq1 | Mv1
        a1 = Eq1 & Pv1
        sum1_nc = a1 + Pv1
        carry1_add = (((a1 & Pv1) | ((a1 | Pv1) & (~sum1_nc))) >> 63) & 1
        sum1 = sum1_nc + carry0
        carry1_ovf = carry0 & (sum1_nc == ALL_ONES).to(tl.int64)
        carry1 = carry1_add | carry1_ovf
        Xh1 = (sum1 ^ Pv1) | Eq1
        Ph1 = Mv1 | (~(Xh1 | Pv1))
        Mh1 = Pv1 & Xh1
        ph1_msb = (Ph1 >> 63) & 1
        mh1_msb = (Mh1 >> 63) & 1
        Ph1_shifted = (Ph1 << 1) | ph0_msb
        Mh1_shifted = (Mh1 << 1) | mh0_msb
        Pv1 = Mh1_shifted | (~(Xv1 | Ph1_shifted))
        Mv1 = Ph1_shifted & Xv1

        # ---- Block 2 (last, masked) ----
        Eq2 = tl.where(text_bit == 1, q2_masked, nq2_masked)
        Xv2 = Eq2 | Mv2
        a2 = Eq2 & Pv2
        sum2_nc = a2 + Pv2
        sum2 = sum2_nc + carry1
        Xh2 = (sum2 ^ Pv2) | Eq2
        Ph2 = Mv2 | (~(Xh2 | Pv2))
        Mh2 = Pv2 & Xh2
        Ph2 = Ph2 & last_mask
        Mh2 = Mh2 & last_mask

        ph2_high = (Ph2 & high_bit) != 0
        mh2_high = (Mh2 & high_bit) != 0
        score_vec = score_vec + ph2_high.to(tl.int32) - mh2_high.to(tl.int32)

        Ph2_shifted = (Ph2 << 1) | ph1_msb
        Mh2_shifted = (Mh2 << 1) | mh1_msb
        Pv2 = (Mh2_shifted | (~(Xv2 | Ph2_shifted))) & last_mask
        Mv2 = (Ph2_shifted & Xv2) & last_mask

    tl.store(result_ptr + offs, score_vec, mask=mask)


# ---------------------------------------------------------------------------
# Generalized kernel (NUM_BLOCKS >= 4, buffer-based state)
# ---------------------------------------------------------------------------
# For u > 192, the number of 64-bit blocks exceeds the hand-unrolled kernels
# above. This kernel handles any NUM_BLOCKS by storing Pv/Mv state in
# pre-allocated GPU memory buffers instead of register variables.
#
# The inner block loop uses tl.static_range(NUM_BLOCKS) so the compiler
# unrolls it at compile time — all `b == 0` / `b == NUM_BLOCKS - 1` checks
# become compile-time constants with no runtime branching.
# ---------------------------------------------------------------------------

@triton.jit
def _myers_lev_kernel_general(
    # Pointers
    codebook_ptr,       # (chunk_N, num_cb_words) int64 — codebook packed
    query_ptr,          # (NUM_BLOCKS,) int64 — query words on device
    pv_ptr,             # (NUM_BLOCKS * chunk_N,) int64 — Pv state buffer, row-major
    mv_ptr,             # (NUM_BLOCKS * chunk_N,) int64 — Mv state buffer, row-major
    result_ptr,         # (chunk_N,) int32 — output distances
    # Dimensions
    N: tl.int32,        # chunk_N (number of codebook entries in this chunk)
    text_len: tl.int32, # code length (u)
    num_cb_words: tl.int32,  # stride: words per codebook row
    last_mask: tl.int64,
    high_bit: tl.int64,
    query_len: tl.int32,
    # Grid
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """Myers' kernel for arbitrary NUM_BLOCKS (query_len up to NUM_BLOCKS*64).

    Uses global memory buffers for Pv/Mv state with coalesced access pattern:
    buffer layout is (NUM_BLOCKS, N) row-major, so block b's state for entry
    offset `offs` is at ptr + b * N + offs.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    ALL_ONES: tl.constexpr = -1  # 0xFFFFFFFFFFFFFFFF as int64

    score_vec = tl.full([BLOCK_SIZE], query_len, dtype=tl.int32)

    # Process each text position
    for i in range(text_len):
        word_idx = i // 64
        bit_pos = i % 64

        # Load text bit for each entry
        cb_offs = offs * num_cb_words + word_idx
        text_word = tl.load(codebook_ptr + cb_offs, mask=mask, other=0).to(tl.int64)
        text_bit = (text_word >> bit_pos) & 1

        # Carry and MSB propagation across blocks
        carry: tl.int64 = 0
        prev_ph_msb: tl.int64 = 0
        prev_mh_msb: tl.int64 = 0

        for b in tl.static_range(NUM_BLOCKS):
            # Load query word for this block
            q_word = tl.load(query_ptr + b).to(tl.int64)

            # Build Eq: mask last block to valid bits only
            if b == NUM_BLOCKS - 1:
                q_masked = q_word & last_mask
                nq_masked = (~q_word) & last_mask
                Eq = tl.where(text_bit == 1, q_masked, nq_masked)
            else:
                Eq = tl.where(text_bit == 1, q_word, ~q_word)

            # Load Pv, Mv for this block (coalesced: adjacent offs are adjacent)
            state_offs = b * N + offs
            Pv = tl.load(pv_ptr + state_offs, mask=mask, other=0).to(tl.int64)
            Mv = tl.load(mv_ptr + state_offs, mask=mask, other=0).to(tl.int64)

            Xv = Eq | Mv

            # Addition with carry propagation
            a = Eq & Pv
            if b == 0:
                # First block: no carry-in
                sum_val = a + Pv
                new_carry = (((a & Pv) | ((a | Pv) & (~sum_val))) >> 63) & 1
            else:
                # Subsequent blocks: two-step carry (overflow check)
                sum_nc = a + Pv
                carry_add = (((a & Pv) | ((a | Pv) & (~sum_nc))) >> 63) & 1
                sum_val = sum_nc + carry
                carry_ovf = carry & (sum_nc == ALL_ONES).to(tl.int64)
                new_carry = carry_add | carry_ovf

            carry = new_carry

            Xh = (sum_val ^ Pv) | Eq
            Ph = Mv | (~(Xh | Pv))
            Mh = Pv & Xh

            # Last block: mask and update score
            if b == NUM_BLOCKS - 1:
                Ph = Ph & last_mask
                Mh = Mh & last_mask
                ph_high = (Ph & high_bit) != 0
                mh_high = (Mh & high_bit) != 0
                score_vec = score_vec + ph_high.to(tl.int32) - mh_high.to(tl.int32)

            # Extract MSBs before shifting
            ph_msb = (Ph >> 63) & 1
            mh_msb = (Mh >> 63) & 1

            # Shift Ph, Mh left by 1
            if b == 0:
                Ph_shifted = (Ph << 1) | 1  # boundary condition: set bit 0
                Mh_shifted = Mh << 1
            else:
                Ph_shifted = (Ph << 1) | prev_ph_msb
                Mh_shifted = (Mh << 1) | prev_mh_msb

            prev_ph_msb = ph_msb
            prev_mh_msb = mh_msb

            # Update Pv, Mv
            new_Pv = Mh_shifted | (~(Xv | Ph_shifted))
            new_Mv = Ph_shifted & Xv

            if b == NUM_BLOCKS - 1:
                new_Pv = new_Pv & last_mask
                new_Mv = new_Mv & last_mask

            # Store updated state back to buffer
            tl.store(pv_ptr + state_offs, new_Pv, mask=mask)
            tl.store(mv_ptr + state_offs, new_Mv, mask=mask)

    # Store results
    tl.store(result_ptr + offs, score_vec, mask=mask)


# ---------------------------------------------------------------------------
# Host-side search function
# ---------------------------------------------------------------------------

def _myers_search_triton(
    query_packed: torch.Tensor,
    query_len: int,
    codebook_packed: torch.Tensor,
    code_len: int,
    chunk_size: int = 1 << 20,
    top_k: int = 1,
) -> Tuple[int, int] | Tuple[np.ndarray, np.ndarray]:
    """Find closest codebook entry(ies) to a single query using Triton kernel.

    Args:
        query_packed: (num_query_words,) int64 tensor on device.
        query_len: actual query length in bits.
        codebook_packed: (N, num_text_words) int64 tensor on device.
        code_len: codeword length in bits (u).
        chunk_size: max entries per GPU chunk to cap memory.
        top_k: number of closest entries to return.

    Returns:
        top_k == 1: (best_index, best_distance) as Python ints.
        top_k > 1: (indices, distances) as numpy arrays of shape (top_k,).
    """
    device = codebook_packed.device
    torch.cuda.set_device(device)  # Ensure Triton launches kernel on correct GPU
    N = codebook_packed.shape[0]
    num_cb_words = codebook_packed.shape[1]

    m = query_len
    n = code_len
    num_blocks = (m + 63) // 64

    # Last-block mask — must be Python int (not np.int64) for Triton kernel args.
    # Triton treats int64 params as signed, so we convert to signed representation.
    last_bits = m % 64
    if last_bits == 0:
        last_mask = -1  # all ones (0xFFFFFFFFFFFFFFFF as signed int64)
        high_bit = -9223372036854775808  # 1 << 63 as signed int64
    else:
        last_mask = (1 << last_bits) - 1
        high_bit = 1 << (last_bits - 1)

    # Extract query words as Python ints for kernel params
    q_words = [int(query_packed[b].item()) for b in range(num_blocks)]

    BLOCK_SIZE = 256  # entries per Triton program instance

    # For the general kernel (num_blocks >= 4), cap Pv/Mv buffer memory at ~128 MB
    if num_blocks > 3:
        max_buffer_bytes = 128 * 1024 * 1024
        adaptive_chunk = max_buffer_bytes // (2 * num_blocks * 8)
        effective_chunk = min(chunk_size, max(65536, adaptive_chunk))
    else:
        effective_chunk = chunk_size

    # Top-k tracking
    max_dist = m + n + 1
    if top_k <= 1:
        best_idx = 0
        best_dist = max_dist
    else:
        topk_dists = np.full(top_k, max_dist, dtype=np.int32)
        topk_idxs = np.zeros(top_k, dtype=np.int64)

    for chunk_start in range(0, N, effective_chunk):
        chunk_end = min(chunk_start + effective_chunk, N)
        chunk_N = chunk_end - chunk_start

        # Result buffer for this chunk
        result = torch.empty(chunk_N, dtype=torch.int32, device=device)

        # Select kernel based on number of blocks
        grid = ((chunk_N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        cb_chunk_ptr = codebook_packed[chunk_start:chunk_end]

        if num_blocks == 1:
            _myers_lev_kernel_1block[grid](
                cb_chunk_ptr,
                q_words[0],
                result,
                chunk_N,
                n,
                num_cb_words,
                last_mask,
                high_bit,
                m,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif num_blocks == 2:
            _myers_lev_kernel_2block[grid](
                cb_chunk_ptr,
                q_words[0],
                q_words[1],
                result,
                chunk_N,
                n,
                num_cb_words,
                last_mask,
                high_bit,
                m,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif num_blocks == 3:
            _myers_lev_kernel_3block[grid](
                cb_chunk_ptr,
                q_words[0],
                q_words[1],
                q_words[2],
                result,
                chunk_N,
                n,
                num_cb_words,
                last_mask,
                high_bit,
                m,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # General kernel: allocate Pv/Mv state buffers on GPU
            # Layout: (NUM_BLOCKS, chunk_N) row-major for coalesced access
            Pv_buf = torch.full(
                (num_blocks, chunk_N), -1, dtype=torch.int64, device=device
            )
            Pv_buf[-1, :] = last_mask  # last block: only valid bits
            Mv_buf = torch.zeros(
                (num_blocks, chunk_N), dtype=torch.int64, device=device
            )

            _myers_lev_kernel_general[grid](
                cb_chunk_ptr,
                query_packed,       # (num_blocks,) int64 on device
                Pv_buf,
                Mv_buf,
                result,
                chunk_N,
                n,
                num_cb_words,
                last_mask,
                high_bit,
                m,
                BLOCK_SIZE=BLOCK_SIZE,
                NUM_BLOCKS=num_blocks,
            )

            del Pv_buf, Mv_buf

        # Reduce this chunk
        if top_k <= 1:
            chunk_min_val, chunk_min_local_idx = result.min(dim=0)
            chunk_dist = chunk_min_val.item()
            if chunk_dist < best_dist:
                best_dist = chunk_dist
                best_idx = chunk_start + chunk_min_local_idx.item()
        else:
            k_chunk = min(top_k, chunk_N)
            chunk_vals, chunk_local = torch.topk(
                result[:chunk_N], k=k_chunk, largest=False
            )
            chunk_dists = chunk_vals.cpu().numpy().astype(np.int32)
            chunk_idxs = (chunk_start + chunk_local.cpu().numpy()).astype(np.int64)
            # Merge with running top-k
            merged_dists = np.concatenate([topk_dists, chunk_dists])
            merged_idxs = np.concatenate([topk_idxs, chunk_idxs])
            order = np.argsort(merged_dists)[:top_k]
            topk_dists = merged_dists[order]
            topk_idxs = merged_idxs[order]

    if top_k <= 1:
        return best_idx, best_dist
    return topk_idxs, topk_dists


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------


class CodebookLevenshteinTriton:
    """
    Triton GPU-accelerated Levenshtein-distance codebook search.

    Uses custom Triton kernels implementing Myers' bit-parallel algorithm
    with one GPU thread per codebook entry. The entire sequential loop
    over text positions runs inside the kernel, avoiding kernel-launch
    overhead that plagues naive PyTorch vectorizations.

    Construction:
        base = CodebookBase(...)
        lev_triton = CodebookLevenshteinTriton(base)

    Usage:
        bits, watermark = lev_triton.get_random_row(seed)
        bits, watermark, distance = lev_triton.find_closest_match(target_bits)
        indices, watermarks, distances = lev_triton.find_closest_match_batch(queries)
    """

    def __init__(
        self,
        base: CodebookBase,
        *,
        cache_dir: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        chunk_size: int = 1 << 20,
    ) -> None:
        if base.packed_codes is None:
            raise RuntimeError("Base codebook not loaded")

        self.base = base
        self.cache_dir = cache_dir or base.cache_dir
        self.chunk_size = chunk_size

        self.filename_base = f"{self.base.filename_base}_metric-levenshtein"
        self.index_path = os.path.join(self.cache_dir, f"{self.filename_base}.npz")

        self.num_words: int = (base.u + 63) // 64
        self.bytes_per_code: int = int(self.base.bytes_per_code)
        self.dim_bits: int = int(self.base.dim_bits)

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError(
                    "CUDA not available. CodebookLevenshteinTriton requires a GPU. "
                    "Use CodebookLevenshtein (Numba) for CPU-only environments."
                )
        else:
            self.device = torch.device(device)

        # Load or build the packed codebook, then move to GPU
        self._codebook_np: Optional[np.ndarray] = None
        self._codebook_gpu: Optional[torch.Tensor] = None
        self._load_or_build_index()

    # ------------------------------------------------------------------
    # Delegated helpers
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def u(self) -> int:
        return self.base.u

    def get_random_row(self, seed: int) -> Tuple[List[int], int]:
        return self.base.get_random_row(seed)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_closest_match(
        self, target_codeword: Sequence[int] | np.ndarray
    ) -> Tuple[List[int], int, int]:
        """
        Locate the closest codeword by Levenshtein edit distance on GPU.
        Accepts target codeword as a 1D sequence of 0/1 of ANY length.
        Returns (closest_codeword_bits, watermark_identity, distance_value).
        """
        if self._codebook_gpu is None or self.base.watermark_identities is None:
            raise RuntimeError("Levenshtein Triton codebook not initialized")

        target_array = np.asarray(target_codeword, dtype=np.uint8).reshape(-1)

        if target_array.size == 0:
            idx = 0
            watermark = int(self.base.watermark_identities[0])
            bits = self.base.get_codeword_by_index(0)
            return bits.tolist(), watermark, self.u

        target_array = (target_array != 0).astype(np.uint8)
        query_len = int(target_array.shape[0])
        query_packed_np = _pack_query_to_int64(target_array, query_len)
        query_packed = torch.from_numpy(query_packed_np).to(self.device)

        idx, dist = _myers_search_triton(
            query_packed, query_len, self._codebook_gpu, self.u, self.chunk_size
        )

        idx = int(idx)
        watermark = int(self.base.watermark_identities[idx])
        bits = self.base.get_codeword_by_index(idx)
        return bits.tolist(), watermark, int(dist)

    def find_closest_match_batch(
        self,
        queries: list[np.ndarray] | np.ndarray,
        *,
        distances_only: bool = False,
        top_k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """
        Batch search: find closest codebook entry(ies) for each query on GPU.

        Args:
            queries: List of 1D uint8 arrays (variable length allowed), or
                     a 2D uint8 array of shape (B, length) for uniform-length queries.
            distances_only: If True, return only distances array.
            top_k: Number of closest entries per query.

        Returns:
            top_k == 1:
                distances_only=False: (best_indices(B,), watermark_ids(B,), distances(B,))
                distances_only=True:  distances(B,) int32
            top_k > 1:
                distances_only=False: (indices(B,k), watermark_ids(B,k), distances(B,k))
                distances_only=True:  distances(B,k) int32
        """
        if self._codebook_gpu is None or self.base.watermark_identities is None:
            raise RuntimeError("Levenshtein Triton codebook not initialized")

        if isinstance(queries, np.ndarray) and queries.ndim == 2:
            query_list = [queries[i] for i in range(queries.shape[0])]
        else:
            query_list = list(queries)

        B = len(query_list)
        if B == 0:
            empty_i = np.empty(0, dtype=np.int64)
            empty_d = np.empty(0, dtype=np.int32)
            if distances_only:
                return empty_d
            return empty_i, empty_i.copy(), empty_d

        if top_k <= 1:
            out_shape = (B,)
        else:
            out_shape = (B, top_k)
        best_indices = np.empty(out_shape, dtype=np.int64)
        best_distances = np.empty(out_shape, dtype=np.int32)

        for qi in range(B):
            q_arr = np.asarray(query_list[qi], dtype=np.uint8).reshape(-1)

            if q_arr.size == 0:
                best_indices[qi] = 0
                best_distances[qi] = self.u
                continue

            q_arr = (q_arr != 0).astype(np.uint8)
            q_len = int(q_arr.shape[0])
            q_packed_np = _pack_query_to_int64(q_arr, q_len)
            q_packed = torch.from_numpy(q_packed_np).to(self.device)

            result = _myers_search_triton(
                q_packed, q_len, self._codebook_gpu, self.u, self.chunk_size,
                top_k=top_k,
            )
            if top_k <= 1:
                best_indices[qi] = result[0]
                best_distances[qi] = result[1]
            else:
                best_indices[qi] = result[0]
                best_distances[qi] = result[1]

        if distances_only:
            return best_distances

        watermark_ids = self.base.watermark_identities[best_indices]
        return best_indices, watermark_ids, best_distances

    # ------------------------------------------------------------------
    # Build/load cache helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Convert base uint8 packed codes to int64 LE packed and cache as .npz."""
        if self.base.packed_codes is None:
            raise RuntimeError("Packed codes unavailable for index build")

        print(f"Building Levenshtein uint64 index [numpy] (n={self.base.n}, u={self.base.u})...")
        codebook_packed = _convert_uint8_be_to_int64_le(
            self.base.packed_codes, self.base.u
        )

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        np.savez_compressed(
            self.index_path,
            codebook_packed=codebook_packed.view(np.uint64),  # store as uint64 for compat
            n=np.int64(self.base.n),
            u=np.int64(self.base.u),
            generator_tag=np.array(self.base.generator.algorithm_tag),
            secret_key=np.array(self.base.secret_key),
            count=np.int64(codebook_packed.shape[0]),
        )

        self._codebook_np = codebook_packed

    def _load_or_build_index(self) -> None:
        try:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError("Levenshtein NPZ not found")

            data = np.load(self.index_path, allow_pickle=True)

            if int(data["n"]) != self.base.n or int(data["u"]) != self.base.u:
                raise ValueError("Cached Levenshtein codebook n/u mismatch")
            if str(data["generator_tag"]) != self.base.generator.algorithm_tag:
                raise ValueError("Cached Levenshtein generator tag mismatch")
            if str(data["secret_key"]) != self.base.secret_key:
                raise ValueError("Cached Levenshtein secret key mismatch")
            if self.base.watermark_identities is None or int(data["count"]) != int(
                self.base.watermark_identities.shape[0]
            ):
                raise ValueError("Cached Levenshtein row count mismatch")

            self._codebook_np = np.ascontiguousarray(
                data["codebook_packed"], dtype=np.uint64
            ).view(np.int64)

        except Exception:
            self._build_index()

        # Move to GPU as int64
        self._codebook_gpu = torch.from_numpy(
            np.ascontiguousarray(self._codebook_np)
        ).to(self.device)
        # Free numpy copy
        self._codebook_np = None
