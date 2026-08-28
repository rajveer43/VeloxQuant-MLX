"""Fused TurboQuantRVQ quantize+pack Metal kernel (#251).

``TurboQuantRVQKVCache.update_and_fetch`` currently runs the two-stage RVQ
codebook quantize (stage-1 nearest-centroid, dequantize, residual, stage-2
nearest-centroid) as MLX broadcast-compare ops, then bit-packs each stage's
uint8 index array into uint32 words via a *separate* MLX dispatch
(``_pack_indices`` in ``veloxquant_mlx/cache/turboquant_rvq_cache.py``):

    rotated y -> quantize1 -> dequantize1 -> residual -> quantize2
              -> [idx1 uint8 buffer] -> pack -> [packed1 uint32]
              -> [idx2 uint8 buffer] -> pack -> [packed2 uint32]

Every arrow after the rotation is arithmetically a single streaming pass per
coordinate, but the MLX path materializes two full-size ``(N, D)`` uint8
index intermediates before packing them. This module fuses stage-1 quantize,
stage-2 (residual) quantize, and both bit-packs into one Metal dispatch that
writes directly to the two packed uint32 streams — no intermediate index
buffers, no separate pack dispatch.

Bit-exactness with the MLX path (``ScalarCodebook.quantize`` +
``_pack_indices``):

* ``ScalarCodebook.quantize`` computes ``idx = sum(y > boundary_k)`` over
  sorted Voronoi boundaries (a "boundary-count" quantizer, not a naive
  argmin) — this kernel reduces the identical ``>`` comparisons in the same
  order, so ties resolve identically.
* ``_pack_indices`` packs ``ELEMS_PER_WORD = 32 // bits`` consecutive
  coordinates into one ``uint32``, LSB-first, zero-padding any partial
  trailing word — this kernel's packing loop matches that exactly.

Public API:
  - :func:`rvq_quant_pack`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

_WORD_BITS = 32


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}

# ===========================================================================
# Metal source — fused stage-1 + stage-2 RVQ quantize, packed to uint32
# ===========================================================================
# Grid:        (N * D, 1, 1) — MLX grid = total threads.
# Threadgroup: (D, 1, 1)     — one threadgroup per rotated vector, D <= 1024.
#
# BITS and MAX_D are compile-time #defines/templates: BITS controls the
# packing width (matching TurboQuantRVQKVCache's single bit_width_inlier used
# for both RVQ stages), MAX_D sizes the threadgroup index-staging buffers.

_RVQ_QUANT_PACK_SRC = _read_kernel_source("rvq_quant_pack.metal")


def _quant_pack_kernel(d: int, bits: int):
    key = ("rvq_quant_pack", d, bits)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"rvq_quant_pack_d{d}_b{bits}",
            input_names=["rotated", "centroids1", "boundaries1", "boundaries2"],
            output_names=["packed1", "packed2"],
            header=f"#define MAX_D {d}\n#define BITS {bits}u\n",
            source=_RVQ_QUANT_PACK_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def rvq_quant_pack(
    rotated: mx.array,
    centroids1: mx.array,
    boundaries1: mx.array,
    boundaries2: mx.array,
    bits: int,
) -> tuple[mx.array, mx.array]:
    """Fused two-stage RVQ quantize, packed directly into uint32 words.

    Args:
        rotated:     ``[N, D]`` fp16/fp32 rotated key vectors (post-Hadamard
                     or post-QR rotation — the ``y`` that would otherwise go
                     into ``ScalarCodebook.quantize``). D must be a power of
                     two, <= 1024 (Metal threadgroup-size limit).
        centroids1:  ``[2**bits]`` fp32/fp16 stage-1 sorted centroids.
        boundaries1: ``[2**bits - 1]`` fp32/fp16 stage-1 sorted Voronoi
                     boundaries (midpoints between consecutive centroids).
        boundaries2: ``[2**bits - 1]`` fp32/fp16 stage-2 (Laplacian
                     residual) sorted Voronoi boundaries.
        bits:        Bits per stage (1-4). Total packed storage is
                     ``2 * bits`` bits/coordinate, split across two streams.

    Returns:
        Tuple ``(packed1, packed2)``, each ``[N, ceil(D / (32 // bits))]``
        uint32 — bit-identical to
        ``_pack_indices(codebook1.quantize(rotated), bits)`` and
        ``_pack_indices(codebook2.quantize(rotated - y_hat1), bits)``
        respectively.
    """
    if rotated.ndim != 2:
        raise ValueError(f"rvq_quant_pack: rotated must be 2D [N, D], got {rotated.shape}")
    N, D = rotated.shape
    if D & (D - 1) != 0:
        raise ValueError(f"rvq_quant_pack: D={D} must be a power of two")
    if D > 1024:
        raise ValueError(f"rvq_quant_pack: D={D} exceeds the 1024 threadgroup-size limit")
    if not (1 <= bits <= 4):
        raise ValueError(f"rvq_quant_pack: bits must be 1-4, got {bits}")
    n_levels = 1 << bits
    if centroids1.size != n_levels:
        raise ValueError(
            f"rvq_quant_pack: expected {n_levels} stage-1 centroids for bits={bits}, "
            f"got {centroids1.size}"
        )
    if boundaries1.size != n_levels - 1 or boundaries2.size != n_levels - 1:
        raise ValueError(
            f"rvq_quant_pack: expected {n_levels - 1} boundaries per stage for bits={bits}, "
            f"got {boundaries1.size} (stage 1), {boundaries2.size} (stage 2)"
        )

    el_per_word = _WORD_BITS // bits
    n_words = -(-D // el_per_word)  # ceil div

    kernel = _quant_pack_kernel(D, bits)
    outputs = kernel(
        inputs=[
            rotated.astype(mx.float16),
            centroids1.astype(mx.float32),
            boundaries1.astype(mx.float32),
            boundaries2.astype(mx.float32),
        ],
        # MLX grid = total threads; D threads per threadgroup, one per vector.
        grid=(N * D, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(N, n_words), (N, n_words)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return outputs[0], outputs[1]


__all__ = ["rvq_quant_pack"]
