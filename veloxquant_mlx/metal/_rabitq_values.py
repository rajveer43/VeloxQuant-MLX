"""Nibble packing Metal kernel for 4-bit value-codebook indices.

Packs two 4-bit indices per byte so the value side of the asymmetric
RaBitQ cache stores ``[.., D//2]`` uint8 instead of ``[.., D]`` uint8 —
half the memory and half the bandwidth at attend time.
:func:`rabitq_fused_attend` reads this format directly when the v_idx
shape says so.

Layout: byte ``j`` holds elements ``2j`` (low nibble) and ``2j + 1``
(high nibble).

Public API:
  - :func:`rabitq_pack_values`
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
# Grid: (N_out, 1, 1) — one thread per output byte. Indices are masked
# to 4 bits so out-of-range inputs can never corrupt a neighbour nibble.

_PACK_VALUES_SRC = _read_kernel_source("rabitq_pack_values.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _pack_values_kernel():
    key = ("rabitq_pack_values",)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name="rabitq_pack_values",
            input_names=["v_idx"],
            output_names=["packed"],
            source=_PACK_VALUES_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rabitq_pack_values(v_idx: mx.array) -> mx.array:
    """Pack 4-bit value indices two-per-byte along the last axis.

    Args:
        v_idx: uint8 indices in ``[0, 16)``, any shape with an even last
               dimension (e.g. ``[B, H, S_kv, D]``). Values are masked
               to 4 bits.

    Returns:
        uint8 array with the last dimension halved; byte ``j`` holds
        element ``2j`` in the low nibble and ``2j + 1`` in the high
        nibble. Feed directly to :func:`rabitq_fused_attend`.
    """
    if v_idx.ndim < 1:
        raise ValueError("rabitq_pack_values: v_idx must have at least 1 dim")
    last = v_idx.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"rabitq_pack_values: last dimension must be even, got {last}")

    out_shape = (*v_idx.shape[:-1], last // 2)
    n_out = 1
    for s in out_shape:
        n_out *= s

    outputs = _pack_values_kernel()(
        inputs=[v_idx.astype(mx.uint8)],
        grid=(n_out, 1, 1),
        threadgroup=(min(n_out, 256), 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[mx.uint8],
    )
    return outputs[0]


__all__ = ["rabitq_pack_values"]
