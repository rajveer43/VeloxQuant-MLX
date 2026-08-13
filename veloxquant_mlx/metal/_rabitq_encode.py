"""Fused RaBitQ encode Metal kernel — rotate + binarize + pack + magnitude.

One dispatch turns raw fp16 key vectors into the centroid-free RaBitQ
cache representation consumed by :func:`rabitq_fused_attend`:

    y      = WHT(diag * k) / sqrt(D)          (randomized Hadamard rotation)
    k_bits = packbits(y >= 0)                 (little-endian, >= 0 -> 1)
    k_mag  = L1(y) / D                        (per-vector magnitude)

The rotation matches the RaBitQ quantizer's `_rotate_batch_*` path
(diagonal +-1 sign flip, then Sylvester-order Walsh-Hadamard scaled by
1/sqrt(D)), and the bit conventions match `_pack_signs`.

Sign packing uses ``simd_ballot``: each SIMD-group's 32 sign predicates
land in one 32-bit vote mask in a single instruction, which is exactly
4 bytes of little-endian packed output — no per-bit shifting loop.

Public API:
  - :func:`rabitq_encode`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — fused rotate + sign-pack + L1 magnitude
# ===========================================================================
# Grid:        (N * D, 1, 1) — MLX grid = total threads.
# Threadgroup: (D, 1, 1)     — one threadgroup per vector (D <= 1024, pow 2).
#
# Pipeline:
#   1. Load keys[n, lane], apply diagonal +-1 sign; stage in threadgroup buf.
#   2. In-place WHT: the same range-based parallel butterfly as
#      _scalar_quant.py's fused Hadamard kernel (log2(D) barrier passes).
#   3. Scale by 1/sqrt(D). Scaling by a positive constant cannot change
#      a sign, so bits depend only on the pre-scale butterfly output.
#   4. simd_ballot(y >= 0) -> 32-bit mask per SIMD-group. Metal assigns
#      SIMD lanes by linear thread index, so mask bit t of SIMD-group g
#      is dim 32g + t — exactly bytes [4g, 4g+4) of np.packbits
#      (bitorder='little'). Lane 0 of each SIMD-group stores its 4 bytes.
#   5. L1 via simd_sum, then a cross-SIMD-group reduction in threadgroup
#      memory (canonical two-stage pattern, MSL spec Sec. 6.9.2.1).

_RABITQ_ENCODE_SRC = _read_kernel_source("rabitq_encode.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _encode_kernel(d: int):
    key = ("rabitq_encode", d)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"rabitq_encode_d{d}",
            input_names=["keys", "diag"],
            output_names=["k_bits", "k_mag"],
            source=_RABITQ_ENCODE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rabitq_encode(
    keys: mx.array,  # [N, D] fp16/fp32 — raw (pre-rotation) key vectors
    diag: mx.array,  # [D] fp32 — +-1 Hadamard diagonal
) -> tuple[mx.array, mx.array]:
    """Encode keys into the centroid-free RaBitQ cache representation.

    Fuses the randomized Hadamard rotation, sign binarization, bit
    packing, and L1-magnitude computation into a single dispatch. The
    outputs plug directly into :func:`rabitq_fused_attend` as ``k_bits``
    and ``k_mag`` (with ``k_const = 0``).

    Args:
        keys: ``[N, D]`` fp16/fp32 raw key vectors. D must be a power of
              two (Walsh-Hadamard), divisible by 8, and <= 1024.
        diag: ``[D]`` fp32 +-1 diagonal, e.g. from
              ``make_hadamard_diagonal`` — must be the same diagonal used
              to rotate queries at attend time.

    Returns:
        Tuple of:
          - ``k_bits`` ``[N, D//8]`` uint8 packed sign bits of the
            rotated vectors (little-endian bit order, ``>= 0`` -> 1).
          - ``k_mag`` ``[N]`` fp32 per-vector ``L1(rotated)/D``.
    """
    if keys.ndim != 2:
        raise ValueError(f"rabitq_encode: keys must be 2D [N, D], got {keys.shape}")
    N, D = keys.shape
    if D % 8 != 0:
        raise ValueError(f"rabitq_encode: D={D} must be divisible by 8")
    if D & (D - 1) != 0:
        raise ValueError(f"rabitq_encode: D={D} must be a power of two")
    if D > 1024:
        raise ValueError(f"rabitq_encode: D={D} exceeds the 1024 limit")
    if diag.shape != (D,):
        raise ValueError(f"rabitq_encode: diag must be [{D}], got {diag.shape}")

    outputs = _encode_kernel(D)(
        inputs=[keys.astype(mx.float16), diag.astype(mx.float32)],
        template=[("MAX_D", D), ("N_BYTES", D // 8)],
        # MLX grid = total threads; D threads per threadgroup, one per vector
        grid=(N * D, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(N, D // 8), (N,)],
        output_dtypes=[mx.uint8, mx.float32],
    )
    return outputs[0], outputs[1]


__all__ = ["rabitq_encode"]
