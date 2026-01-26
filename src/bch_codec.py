"""
BCH encoder/decoder with support for padding and puncturing.

Provides a codec that can encode/decode watermarks of arbitrary length to/from
codewords of arbitrary length, using BCH error correction with automatic
padding (for u > n0) or puncturing (for u < n0).
"""

from __future__ import annotations

from typing import Tuple, Union, Optional
import numpy as np

try:
    import galois
except ImportError as e:
    raise ImportError(
        "The 'galois' package is required for BCH codes. "
        "Install it with: pip install galois"
    ) from e

from src.bch_params import BCHParams, select_bch_parameters, print_bch_warnings, format_bch_summary


ArrayLike = Union[list, np.ndarray]


class BCHCodec:
    """
    BCH encoder/decoder for watermarking with arbitrary (n, u) lengths.

    This codec handles the mismatch between desired codeword length (u) and
    the native BCH codeword length (n0 = 2^m - 1) via:

    1. PADDING (u > n0): Append zero bits to reach length u.
       - Padding bits are not protected by BCH error correction.
       - Decoder strips padding before BCH decoding.

    2. PUNCTURING (u < n0): Remove bits from the end to reach length u.
       - Reduces effective error correction capability.
       - Decoder inserts zeros at punctured positions before BCH decoding.

    Attributes:
        n: Watermark/message length in bits (supports 2^n unique identities)
        u: Desired codeword length in bits (embedded in generated text)
        params: BCHParams with selected (m, n0, k, t) configuration

    Example:
        codec = BCHCodec(n=36, u=127)  # 36-bit watermark, 127-bit codeword
        codeword = codec.encode_int(watermark_value)
        recovered, bit_errors = codec.decode_int(received_bits)
    """

    def __init__(
        self,
        n: int,
        u: int,
        bch_m: Optional[int] = None,
        bch_t: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.n = n  # Message length (watermark bits)
        self.u = u  # Desired codeword length

        # Select BCH parameters
        self.params = select_bch_parameters(n, u, m=bch_m, t=bch_t)

        # Create the BCH code
        d = 2 * self.params.t + 1
        self._bch = galois.BCH(self.params.n0, d=d)
        self._GF2 = galois.GF(2)

        # Store manual flags for warnings
        self._manual_m = bch_m is not None
        self._manual_t = bch_t is not None

        if verbose:
            print(format_bch_summary(self.params, n))
            print_bch_warnings(n, u, self.params, self._manual_m, self._manual_t)

    @property
    def message_length(self) -> int:
        """Watermark message length in bits."""
        return self.n

    @property
    def codeword_length(self) -> int:
        """Output codeword length in bits (after padding/puncturing)."""
        return self.u

    @property
    def error_correction_capability(self) -> int:
        """Number of bit errors BCH can correct (before puncturing penalty)."""
        return self.params.t

    @property
    def native_codeword_length(self) -> int:
        """Native BCH codeword length n0 = 2^m - 1."""
        return self.params.n0

    def encode(self, message: ArrayLike) -> np.ndarray:
        """
        Encode a message into a BCH codeword of exactly u bits.

        Encoding process:
        1. Pad message from n bits to k bits (BCH message length)
        2. Apply BCH encoding to get n0 bits
        3. Apply padding (u > n0) or puncturing (u < n0) to get u bits

        Args:
            message: Binary array of length n (watermark bits)

        Returns:
            Binary array of length u (codeword with padding/puncturing applied)
        """
        msg_arr = self._validate_binary_input(message, self.n, "message")

        # Pad message to k bits (BCH message length)
        if self.params.k > self.n:
            padded_msg = np.zeros(self.params.k, dtype=np.uint8)
            padded_msg[:self.n] = msg_arr
        else:
            padded_msg = msg_arr

        # BCH encode
        gf_msg = self._GF2(padded_msg)
        gf_codeword = self._bch.encode(gf_msg)
        codeword = np.array(gf_codeword, dtype=np.uint8)

        # Apply padding or puncturing to get exactly u bits
        if self.params.padding > 0:
            # Append zeros
            output = np.zeros(self.u, dtype=np.uint8)
            output[:self.params.n0] = codeword
        elif self.params.puncturing > 0:
            # Remove last bits
            output = codeword[:self.u]
        else:
            # Perfect fit
            output = codeword

        return output

    def decode(self, codeword: ArrayLike) -> Tuple[np.ndarray, int]:
        """
        Decode a BCH codeword back to the original message.

        Decoding process:
        1. Undo padding (strip zeros) or puncturing (insert erasure markers)
           to recover the n0-bit native BCH codeword
        2. Apply BCH syndrome decoding to correct errors
        3. Extract the first n bits as the recovered message

        Args:
            codeword: Binary array of length u (received from decoder)

        Returns:
            Tuple of (message, bit_errors):
            - message: Binary array of length n (recovered watermark)
            - bit_errors: Number of errors corrected. If exceeds t, decoding
                          failed and the result is unreliable.
        """
        cw_arr = self._validate_binary_input(codeword, self.u, "codeword")

        # Undo padding or puncturing to get n0-bit codeword
        if self.params.padding > 0:
            # Strip padding zeros
            native_cw = cw_arr[:self.params.n0]
        elif self.params.puncturing > 0:
            # Insert erasures (zeros) at punctured positions
            # BCH will treat these as potential error locations
            native_cw = np.zeros(self.params.n0, dtype=np.uint8)
            native_cw[:self.u] = cw_arr
        else:
            native_cw = cw_arr

        # BCH decode
        gf_cw = self._GF2(native_cw)

        try:
            decoded = self._bch.decode(gf_cw)
            if isinstance(decoded, tuple):
                decoded_msg = decoded[0]
            else:
                decoded_msg = decoded

            msg = np.array(decoded_msg, dtype=np.uint8)

            # Calculate bit errors by re-encoding and comparing
            corrected_cw = np.array(self._bch.encode(self._GF2(msg)), dtype=np.uint8)

            # Compare only the non-punctured portion
            if self.params.puncturing > 0:
                bit_errors = int(np.sum(native_cw[:self.u] != corrected_cw[:self.u]))
            else:
                bit_errors = int(np.sum(native_cw != corrected_cw))

        except Exception:
            # Decoding failed (too many errors)
            # Return zeros and maximum possible bit errors
            msg = np.zeros(self.params.k, dtype=np.uint8)
            bit_errors = self.params.n0  # Indicate decoding failure

        # Extract original message (first n bits)
        return msg[:self.n], bit_errors

    def encode_int(self, watermark: int) -> np.ndarray:
        """
        Convenience method: encode an integer watermark to a codeword.

        Converts the integer to binary (MSB first) before encoding.
        Useful for directly encoding watermark IDs/indices.

        Args:
            watermark: Integer watermark value (0 to 2^n - 1)

        Returns:
            Binary array of length u
        """
        if watermark < 0 or watermark >= 2**self.n:
            raise ValueError(
                f"Watermark {watermark} out of range [0, {2**self.n - 1}]"
            )

        # Convert int to binary array (MSB first)
        message = np.array(
            [(watermark >> (self.n - 1 - i)) & 1 for i in range(self.n)],
            dtype=np.uint8
        )
        return self.encode(message)

    def decode_int(self, codeword: ArrayLike) -> Tuple[int, int]:
        """
        Convenience method: decode a codeword to an integer watermark.

        Converts the recovered binary message back to an integer.

        Args:
            codeword: Binary array of length u (received from decoder)

        Returns:
            Tuple of (watermark_int, bit_errors):
            - watermark_int: The recovered watermark as an integer
            - bit_errors: Number of bit errors detected/corrected
        """
        message, bit_errors = self.decode(codeword)

        # Convert binary array to int
        watermark = 0
        for bit in message:
            watermark = (watermark << 1) | int(bit)

        return watermark, bit_errors

    @staticmethod
    def _validate_binary_input(
        arr: ArrayLike,
        expected_length: int,
        name: str
    ) -> np.ndarray:
        """Validate and convert input to binary numpy array."""
        arr = np.asarray(arr, dtype=np.uint8)

        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {arr.shape}")

        if len(arr) != expected_length:
            raise ValueError(
                f"{name} must have length {expected_length}, got {len(arr)}"
            )

        if arr.min() < 0 or arr.max() > 1:
            raise ValueError(f"{name} must contain only binary values (0 or 1)")

        return arr


def create_bch_codec(
    n: int,
    u: int,
    bch_m: Optional[int] = None,
    bch_t: Optional[int] = None,
    verbose: bool = False,
) -> BCHCodec:
    """
    Factory function to create a BCH codec.

    Args:
        n: Message length in bits (watermark size)
        u: Desired output codeword length in bits
        bch_m: Optional manual field degree (n0 = 2^m - 1)
        bch_t: Optional manual error correction capability
        verbose: Print BCH configuration and warnings

    Returns:
        Configured BCHCodec instance
    """
    return BCHCodec(n, u, bch_m, bch_t, verbose)
