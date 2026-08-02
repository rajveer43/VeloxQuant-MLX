"""Bit-packing Metal kernels for TurboQuant index storage.

Packs b-bit quantization indices (b ∈ {1, 2, 4}) into tightly bit-packed
uint8 buffers, and unpacks them back.  This is the primary memory saving for
all TurboQuant cache variants:

  b=4 → 2 indices / byte  (2× compression over uint8 storage)
  b=2 → 4 indices / byte  (4× compression)
  b=1 → 8 indices / byte  (8× compression)

Public API:
  - :func:`turboquant_bit_pack`
  - :func:`turboquant_bit_unpack`
"""

from __future__ import annotations

import mlx.core as mx

_cache: dict = {}


# ===========================================================================
# Metal source — pack
# ===========================================================================
# Grid: (N_bytes, 1, 1) — one thread per output byte.
# Each thread reads ELEMS_PER_BYTE = 8/B_BITS consecutive indices, masks each
# to B_BITS, shifts into position, and writes one packed uint8.
# ELEMS_PER_BYTE is a compile-time constant (template), so the loop unrolls.

_PACK_SRC = r"""
    constexpr int  ELEMS_PER_BYTE = 8 / B_BITS;
    constexpr uint MASK           = (1u << B_BITS) - 1u;

    uint byte_idx = thread_position_in_grid.x;
    uint base     = byte_idx * ELEMS_PER_BYTE;

    uint packed_byte = 0u;
    for (int i = 0; i < ELEMS_PER_BYTE; ++i) {
        uint val = uint(indices[base + i]) & MASK;
        packed_byte |= (val << (i * B_BITS));
    }
    packed[byte_idx] = uint8_t(packed_byte);
"""


# ===========================================================================
# Metal source — unpack
# ===========================================================================
# Grid: (N_elements, 1, 1) — one thread per output index.
# Each thread computes its source byte and bit offset, extracts B_BITS, writes.

_UNPACK_SRC = r"""
    constexpr int  ELEMS_PER_BYTE = 8 / B_BITS;
    constexpr uint MASK           = (1u << B_BITS) - 1u;

    uint elem_idx = thread_position_in_grid.x;
    uint byte_idx = elem_idx / ELEMS_PER_BYTE;
    uint bit_off  = (elem_idx % ELEMS_PER_BYTE) * B_BITS;

    indices[elem_idx] = uint8_t((uint(packed[byte_idx]) >> bit_off) & MASK);
"""


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def _pack_kernel(b: int):
    key = ("bit_pack", b)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"turboquant_bit_pack_b{b}",
            input_names=["indices"],
            output_names=["packed"],
            source=_PACK_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _unpack_kernel(b: int):
    key = ("bit_unpack", b)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"turboquant_bit_unpack_b{b}",
            input_names=["packed"],
            output_names=["indices"],
            source=_UNPACK_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def turboquant_bit_pack(indices: mx.array, b: int) -> mx.array:
    """Pack uint8 indices into tightly bit-packed uint8 storage.

    Args:
        indices: ``[N]`` uint8 with values in ``[0, 2^b)``.
                 N must be divisible by ``8 // b``.
        b: Bits per index. Must be 1, 2, or 4.

    Returns:
        ``[N * b // 8]`` uint8 packed buffer.
    """
    if b not in (1, 2, 4):
        raise ValueError(f"turboquant_bit_pack: b must be 1, 2, or 4, got {b}")
    N = indices.size
    elems_per_byte = 8 // b
    if N % elems_per_byte != 0:
        raise ValueError(f"turboquant_bit_pack: N={N} not divisible by {elems_per_byte} (= 8/b)")
    n_bytes = N * b // 8
    flat = indices.reshape(-1).astype(mx.uint8)
    outputs = _pack_kernel(b)(
        inputs=[flat],
        template=[("B_BITS", b)],
        grid=(n_bytes, 1, 1),
        threadgroup=(min(256, n_bytes), 1, 1),
        output_shapes=[(n_bytes,)],
        output_dtypes=[mx.uint8],
    )
    return outputs[0]


def turboquant_bit_unpack(packed: mx.array, N: int, b: int) -> mx.array:
    """Unpack bit-packed uint8 storage back into uint8 indices.

    Args:
        packed: ``[N * b // 8]`` uint8 packed buffer.
        N:      Number of original indices to recover.
        b:      Bits per index. Must be 1, 2, or 4.

    Returns:
        ``[N]`` uint8 indices.
    """
    if b not in (1, 2, 4):
        raise ValueError(f"turboquant_bit_unpack: b must be 1, 2, or 4, got {b}")
    flat = packed.reshape(-1).astype(mx.uint8)
    outputs = _unpack_kernel(b)(
        inputs=[flat],
        template=[("B_BITS", b)],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.uint8],
    )
    return outputs[0]


__all__ = [
    "turboquant_bit_pack",
    "turboquant_bit_unpack",
]
