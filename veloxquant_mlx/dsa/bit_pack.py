from __future__ import annotations

import math

import numpy as np


class BitPackBuffer:
    """Compact buffer that packs b-bit unsigned integers into uint8 bytes.

    Supports b ∈ {1, 2, 3, 4}. All bit manipulation is implemented manually.

    For non-power-of-2 bit widths (b=3), 8 values fit exactly in 3 bytes:
    each group of 8 values is packed into 3 bytes (24 bits / 3 bits = 8 values).

    Args:
        b: Bits per element. Must be one of {1, 2, 3, 4}.

    Raises:
        ValueError: If b is not in {1, 2, 3, 4}.
    """

    _SUPPORTED_BITS = {1, 2, 3, 4}

    def __init__(self, b: int) -> None:
        if b not in self._SUPPORTED_BITS:
            raise ValueError(f"BitPackBuffer: b must be in {self._SUPPORTED_BITS}, got {b}")
        self.b = b
        self._max_val = (1 << b) - 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pack(self, indices: np.ndarray) -> np.ndarray:
        """Pack an array of b-bit unsigned integers into bytes.

        Args:
            indices: Array of uint8 values in range [0, 2^b). Shape (n,).

        Returns:
            Packed uint8 array. Length is ceil(n * b / 8).

        Raises:
            ValueError: If any value exceeds 2^b - 1.
        """
        indices = np.asarray(indices, dtype=np.uint8)
        if indices.ndim != 1:
            raise ValueError("pack expects a 1-D array")
        if np.any(indices > self._max_val):
            raise ValueError(f"BitPackBuffer(b={self.b}): values must be < {self._max_val + 1}")

        if self.b == 1:
            return self._pack_1bit(indices)
        elif self.b == 2:
            return self._pack_2bit(indices)
        elif self.b == 3:
            return self._pack_3bit(indices)
        else:  # b == 4
            return self._pack_4bit(indices)

    def unpack(self, packed: np.ndarray, n: int) -> np.ndarray:
        """Unpack a byte array back into n b-bit unsigned integers.

        Args:
            packed: Packed uint8 array produced by pack().
            n: Original number of values.

        Returns:
            Uint8 array of length n with values in [0, 2^b).
        """
        packed = np.asarray(packed, dtype=np.uint8)
        if self.b == 1:
            return self._unpack_1bit(packed, n)
        elif self.b == 2:
            return self._unpack_2bit(packed, n)
        elif self.b == 3:
            return self._unpack_3bit(packed, n)
        else:  # b == 4
            return self._unpack_4bit(packed, n)

    # ------------------------------------------------------------------
    # 1-bit: 8 values per byte
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_1bit(idx: np.ndarray) -> np.ndarray:
        n = len(idx)
        n_bytes = (n + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for bit in range(8):
            positions = np.arange(bit, n, 8)
            packed[: len(positions)] |= (idx[positions] & 1) << bit
        return packed

    @staticmethod
    def _unpack_1bit(packed: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.uint8)
        for bit in range(8):
            positions = np.arange(bit, n, 8)
            byte_idx = np.arange(len(positions))
            out[positions] = (packed[byte_idx] >> bit) & 1
        return out

    # ------------------------------------------------------------------
    # 2-bit: 4 values per byte
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_2bit(idx: np.ndarray) -> np.ndarray:
        n = len(idx)
        n_bytes = (n + 3) // 4
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for slot in range(4):
            positions = np.arange(slot, n, 4)
            byte_idx = np.arange(len(positions))
            packed[byte_idx] |= (idx[positions] & 0x3) << (2 * slot)
        return packed

    @staticmethod
    def _unpack_2bit(packed: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.uint8)
        for slot in range(4):
            positions = np.arange(slot, n, 4)
            byte_idx = np.arange(len(positions))
            out[positions] = (packed[byte_idx] >> (2 * slot)) & 0x3
        return out

    # ------------------------------------------------------------------
    # 3-bit: 8 values per 3 bytes (exactly)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_3bit(idx: np.ndarray) -> np.ndarray:
        """Pack 8 3-bit values into 3 bytes (24 bits).

        Group layout (bits 23..0):
            byte0 bits [7:0]  = idx[0][2:0]  idx[1][2:0]  (partial)
            byte1 bits [7:0]  = ...
            byte2 bits [7:0]  = ...

        Exact bit assignments for a group of 8 values v0..v7:
            byte0 = v0[2:0] | v1[2:0]<<3 | v2[1:0]<<6
            byte1 = v2[2:2] | v3[2:0]<<1 | v4[2:0]<<4 | v5[0:0]<<7
            byte2 = v5[2:1] | v6[2:0]<<2 | v7[2:0]<<5
        """
        n = len(idx)
        # Pad to multiple of 8
        pad = (8 - n % 8) % 8
        if pad:
            idx = np.concatenate([idx, np.zeros(pad, dtype=np.uint8)])
        n_groups = len(idx) // 8
        out = np.zeros(n_groups * 3, dtype=np.uint8)
        g = idx.reshape(n_groups, 8)
        out[0::3] = (g[:, 0] & 0x7) | ((g[:, 1] & 0x7) << 3) | ((g[:, 2] & 0x3) << 6)
        out[1::3] = (
            ((g[:, 2] >> 2) & 0x1)
            | ((g[:, 3] & 0x7) << 1)
            | ((g[:, 4] & 0x7) << 4)
            | ((g[:, 5] & 0x1) << 7)
        )
        out[2::3] = ((g[:, 5] >> 1) & 0x3) | ((g[:, 6] & 0x7) << 2) | ((g[:, 7] & 0x7) << 5)
        return out

    @staticmethod
    def _unpack_3bit(packed: np.ndarray, n: int) -> np.ndarray:
        n_groups = (n + 7) // 8
        b0 = packed[0::3][:n_groups]
        b1 = packed[1::3][:n_groups]
        b2 = packed[2::3][:n_groups]
        g = np.zeros((n_groups, 8), dtype=np.uint8)
        g[:, 0] = b0 & 0x7
        g[:, 1] = (b0 >> 3) & 0x7
        g[:, 2] = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
        g[:, 3] = (b1 >> 1) & 0x7
        g[:, 4] = (b1 >> 4) & 0x7
        g[:, 5] = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
        g[:, 6] = (b2 >> 2) & 0x7
        g[:, 7] = (b2 >> 5) & 0x7
        return g.reshape(-1)[:n]

    # ------------------------------------------------------------------
    # 4-bit: 2 values per byte
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_4bit(idx: np.ndarray) -> np.ndarray:
        n = len(idx)
        n_bytes = (n + 1) // 2
        packed = np.zeros(n_bytes, dtype=np.uint8)
        even = idx[0::2]
        byte_idx = np.arange(len(even))
        packed[byte_idx] = even & 0xF
        odd_positions = np.arange(1, n, 2)
        odd_byte_idx = np.arange(len(odd_positions))
        packed[odd_byte_idx] |= (idx[odd_positions] & 0xF) << 4
        return packed

    @staticmethod
    def _unpack_4bit(packed: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.uint8)
        even_positions = np.arange(0, n, 2)
        byte_idx = np.arange(len(even_positions))
        out[even_positions] = packed[byte_idx] & 0xF
        odd_positions = np.arange(1, n, 2)
        odd_byte_idx = np.arange(len(odd_positions))
        out[odd_positions] = (packed[odd_byte_idx] >> 4) & 0xF
        return out

    def __repr__(self) -> str:
        return f"BitPackBuffer(b={self.b})"
