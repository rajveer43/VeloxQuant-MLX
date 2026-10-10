"""Fused TurboQuantRVQ packed-key decode Metal kernel.

``TurboQuantRVQKVCache`` stores keys as two bit-packed uint32 index streams
plus a bf16 norm per vector and re-decodes the *whole* cached history on every
``update_and_fetch`` (``_dequantize_range``). The MLX path is a chain of
separate dispatches, each writing a full-size intermediate to DRAM:

    packed -> shift/mask expand -> reshape/slice/astype(uint8)   (x2 streams)
           -> codebook gather (x2) -> add -> hadamard_transform * diag
           -> fp32 norm multiply + saturate

This module fuses all of it into one dispatch. Algorithm, per cached vector
(one threadgroup of ``D`` threads, thread ``i`` owns coordinate ``i``):

  1. Unpack ``idx1``/``idx2`` for coordinate ``i`` straight from the uint32
     words (LSB-first, ``32 // bits`` codes per word, same layout as
     ``_pack_indices`` / ``mx.quantize``).
  2. ``y = half(c1[idx1] + c2[idx2])`` -- the reference sums two fp16 values.
  3. In-place Walsh-Hadamard butterfly in threadgroup memory, scaled by
     ``1/sqrt(D)`` (``mx.hadamard_transform`` is normalized), then ``* diag``
     and a round to fp16, exactly as ``HadamardPreconditioner.apply_inverse``.
  4. ``half(clamp(x_unit * norm, +-65504))`` -- ``_rescale_fp16``.

Exactness: steps 1, 2 and the final rounding mirror the reference, so the
only difference from the MLX path is the fp32 summation order inside the
butterfly (it can flip an fp16 output by 1 ulp on rare elements). The result
is NOT bit-identical to the MLX path on the rotated output; the unpack +
gather + sum stage is (see ``tests/metal/test_rvq_unpack_decode.py``).

Scope: Hadamard rotation only (``TurboQuantRVQ(use_hadamard=True)``), fp16
output, power-of-two ``D <= 1024``. bf16/fp32 keys and the dense-QR rotation
keep the MLX path in the cache.

Public API:
  - :func:`rvq_unpack_decode`
"""

from __future__ import annotations

import mlx.core as mx

from veloxquant_mlx.metal._kernel_utils import KernelCache, read_kernel_source

_WORD_BITS = 32


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return read_kernel_source(__file__, filename)


_cache = KernelCache()

# ===========================================================================
# Metal source — fused unpack + codebook gather + inverse Hadamard + rescale
# ===========================================================================
# Grid:        (N * D, 1, 1) — MLX grid = total threads.
# Threadgroup: (D, 1, 1)     — one threadgroup per cached key vector, D <= 1024.
#
# BITS and MAX_D are compile-time #defines, so the kernel is keyed on (D, bits).

_RVQ_UNPACK_DECODE_SRC = _read_kernel_source("rvq_unpack_decode.metal")


def _unpack_decode_kernel(d: int, bits: int):
    key = ("rvq_unpack_decode", d, bits)
    return _cache.get_or_create(
        key,
        lambda: mx.fast.metal_kernel(
            name=f"rvq_unpack_decode_d{d}_b{bits}",
            input_names=["packed1", "packed2", "norms", "centroids1", "centroids2", "diag", "dims"],
            output_names=["out"],
            header=f"#define MAX_D {d}\n#define BITS {bits}u\n",
            source=_RVQ_UNPACK_DECODE_SRC,
            ensure_row_contiguous=True,
        ),
    )


def rvq_unpack_decode(
    packed1: mx.array,
    packed2: mx.array,
    norms: mx.array,
    centroids1: mx.array,
    centroids2: mx.array,
    diag: mx.array,
    bits: int,
    d: int,
    seq_len: int | None = None,
) -> mx.array:
    """Decode packed RVQ key vectors to fp16 in a single dispatch.

    Args:
        packed1:    ``[N, n_words]`` uint32 stage-1 index stream, or the whole
                    cache buffer ``[BH, cap, n_words]`` together with ``seq_len``.
        packed2:    Same shape, stage-2 (residual) index stream.
        norms:      ``[N]`` / ``[N, 1]`` (or ``[BH, cap, 1]``) per-vector norm
                    (bf16/fp16/fp32), multiplied in fp32 and saturated to
                    fp16's range.
        centroids1: ``[2**bits]`` stage-1 centroids (fp16 values; any float dtype).
        centroids2: ``[2**bits]`` stage-2 centroids.
        diag:       ``[d]`` randomized-Hadamard +-1 diagonal.
        bits:       Bits per stage (1-4); ``n_words = ceil(d / (32 // bits))``.
        d:          Head dimension; a power of two, <= 1024.
        seq_len:    Live tokens per ``BH`` row (``<= cap``). The kernel reads
                    slots ``[0, seq_len)`` of each row in place, so a cache can
                    pass its full over-allocated buffer without a strided-slice
                    copy. Output rows are ordered ``(bh, token)``.

    Returns:
        ``[N, d]`` (``N = BH * seq_len``) fp16 reconstructed keys, i.e. the same values as
        ``rescale(apply_inverse(c1[idx1] + c2[idx2]), norms)`` on the MLX path.
    """
    if d <= 0 or d & (d - 1) != 0:
        raise ValueError(f"rvq_unpack_decode: d={d} must be a power of two")
    if d > 1024:
        raise ValueError(f"rvq_unpack_decode: d={d} exceeds the 1024 threadgroup-size limit")
    if not (1 <= bits <= 4):
        raise ValueError(f"rvq_unpack_decode: bits must be 1-4, got {bits}")
    el_per_word = _WORD_BITS // bits
    n_words = -(-d // el_per_word)  # ceil div
    n_levels = 1 << bits
    if packed1.ndim == 2 and packed2.ndim == 2:
        # Flat [N, n_words] rows: one "head" whose capacity is its length.
        if seq_len is not None:
            raise ValueError("rvq_unpack_decode: seq_len requires 3D [BH, cap, n_words] streams")
        bh, cap = 1, packed1.shape[0]
        seq = cap
    elif packed1.ndim == 3 and packed2.ndim == 3:
        bh, cap = packed1.shape[0], packed1.shape[1]
        seq = cap if seq_len is None else seq_len
        if not (0 <= seq <= cap):
            raise ValueError(f"rvq_unpack_decode: seq_len={seq} must be in [0, {cap}]")
    else:
        raise ValueError(
            f"rvq_unpack_decode: packed streams must be 2D [N, n_words] or 3D "
            f"[BH, cap, n_words], got {packed1.shape} and {packed2.shape}"
        )
    if packed1.shape[-1] != n_words:
        raise ValueError(
            f"rvq_unpack_decode: expected {n_words} words per row for d={d}, bits={bits}, "
            f"got packed shape {packed1.shape}"
        )
    if packed2.shape != packed1.shape:
        raise ValueError(
            f"rvq_unpack_decode: packed shape mismatch {packed1.shape} vs {packed2.shape}"
        )
    if packed1.dtype != mx.uint32 or packed2.dtype != mx.uint32:
        raise ValueError("rvq_unpack_decode: packed streams must be uint32")
    if norms.size != bh * cap:
        raise ValueError(f"rvq_unpack_decode: expected {bh * cap} norms, got {norms.size}")
    if centroids1.size != n_levels or centroids2.size != n_levels:
        raise ValueError(
            f"rvq_unpack_decode: expected {n_levels} centroids per stage for bits={bits}, "
            f"got {centroids1.size} (stage 1), {centroids2.size} (stage 2)"
        )
    if diag.size != d:
        raise ValueError(f"rvq_unpack_decode: expected diag of size {d}, got {diag.size}")
    N = bh * seq
    if N == 0:
        return mx.zeros((0, d), dtype=mx.float16)

    kernel = _unpack_decode_kernel(d, bits)
    (out,) = kernel(
        inputs=[
            packed1,
            packed2,
            norms.reshape(-1),
            centroids1.astype(mx.float32),
            centroids2.astype(mx.float32),
            diag.astype(mx.float32),
            mx.array([seq, cap], dtype=mx.uint32),
        ],
        # MLX grid = total threads; d threads per threadgroup, one per vector.
        grid=(N * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(N, d)],
        output_dtypes=[mx.float16],
    )
    return out


__all__ = ["rvq_unpack_decode"]
