"""RaBitQ Metal kernel — packed Hamming distance scoring.

Computes the approximate RaBitQ distance for N candidate vectors against
a single query, all in 1-bit packed uint8 representation:

    score[i] = popcount(XOR(qbits, bits[i])) * scale + Cx[i]

where scale = ||qhat - c||_1 / D (precomputed once per query-cluster pair).

Public API:
  - :func:`rabitq_hamming_score`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}

# ---------------------------------------------------------------------------
# Metal source
# ---------------------------------------------------------------------------
# Grid: (N, 1, 1) — one thread per candidate key.
# Each thread XORs D/8 bytes of packed bits, sums popcounts, scales, adds Cx.
# N_BYTES = D / 8 is a compile-time template constant.
# Metal 2.0+ provides popcount() builtin for uint.

_HAMMING_SCORE_SRC = _read_kernel_source("rabitq_hamming_score.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _hamming_kernel(n_bytes: int):
    key = ("rabitq_hamming", n_bytes)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"rabitq_hamming_score_nb{n_bytes}",
            input_names=["qbits", "bits", "Cx", "scale"],
            output_names=["scores"],
            source=_HAMMING_SCORE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rabitq_hamming_score(
    qbits: mx.array,  # [D//8] uint8 — packed query bits
    bits: mx.array,  # [N, D//8] uint8 — packed candidate bits
    Cx: mx.array,  # [N] float32 — per-candidate constant
    scale: mx.array,  # [1] float32 — ||qhat-c||_1 / D
) -> mx.array:
    """Compute RaBitQ approximate distances for N candidates.

    Args:
        qbits:  Packed query sign bits, shape [D//8], uint8.
        bits:   Packed candidate sign bits, shape [N, D//8], uint8.
        Cx:     Per-candidate precomputed constant, shape [N], float32.
        scale:  Scalar scale factor ||qhat-c||_1/D, shape [1], float32.

    Returns:
        scores: shape [N], float32. Lower = closer.
    """
    if qbits.ndim != 1:
        raise ValueError(f"rabitq_hamming_score: qbits must be 1D, got {qbits.shape}")
    if bits.ndim != 2:
        raise ValueError(f"rabitq_hamming_score: bits must be 2D [N, D//8], got {bits.shape}")
    N, n_bytes = bits.shape
    if qbits.shape[0] != n_bytes:
        raise ValueError(
            f"rabitq_hamming_score: qbits length {qbits.shape[0]} != bits cols {n_bytes}"
        )

    qbits_ = qbits.astype(mx.uint8)
    bits_ = bits.reshape(-1).astype(mx.uint8)
    Cx_ = Cx.astype(mx.float32)
    scale_ = scale.reshape(1).astype(mx.float32)

    outputs = _hamming_kernel(n_bytes)(
        inputs=[qbits_, bits_, Cx_, scale_],
        template=[("N_BYTES", n_bytes), ("N", N)],
        grid=(N, 1, 1),
        threadgroup=(min(N, 256), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


__all__ = ["rabitq_hamming_score"]
