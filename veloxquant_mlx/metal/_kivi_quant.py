"""KIVI asymmetric group quantization Metal kernel.

Fuses the whole ``KIVIKVCache._quant_dequant_along`` round-trip — ``moveaxis``
-> pad -> reshape -> ``min``/``max`` -> ``round``/``clip`` -> reconstruct ->
``moveaxis`` back — into a single Metal dispatch (issue #164).

The MLX path materializes several full-size intermediates (the moved-axis
copy, the padded copy, the reshaped group view, the codes, the reconstruction)
for what is arithmetically a single streaming pass. This kernel keeps the group
in registers and writes one output.

**Axis-agnostic by construction.** KIVI quantizes keys per channel (group runs
along the *token* axis, ``axis=-2``) and values per token (group runs along the
*channel* axis, ``axis=-1``). Rather than branch inside the shader, the wrapper
moves the quantization axis last and flattens to ``[R, L]``; the kernel always
groups along ``L``. The ``moveaxis`` is a cheap strided copy that MLX fuses,
and it keeps one shader serving both KIVI schemes.

**Bit-exactness with the MLX path** requires three details that are easy to get
wrong, all of which the parity tests pin:

1. *Padding replicates the last live element.* When ``L`` is not a multiple of
   ``group_size``, the MLX path pads with copies of ``x[..., -1:]``. Padding
   with zeros instead would change the final group's ``min``/``max`` and
   silently shift every reconstruction in that group.
2. *Rounding is half-to-even.* ``mx.round`` is half-to-even; Metal's ``rint()``
   is too under the default rounding mode, but ``metal::round()`` is
   half-away-from-zero and would disagree on exact ``.5`` codes — which are
   common here, since ``(v - gmin) / scale`` lands on exact halves whenever a
   group's range divides evenly.
3. *``eps`` floors the scale.* Degenerate groups (``min == max``, e.g. a
   single-element group) must divide by ``eps``, not zero.

Public API:
  - :func:`kivi_group_quant_dequant`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

# Threads per threadgroup. One threadgroup owns one group, so this is the
# reduction width, not the group size — the kernel grid-strides when
# group_size > _THREADS. Power of two: the tree reduction halves it.
_THREADS = 32

_MLX_TO_METAL_DTYPE = {
    mx.float16: "half",
    mx.float32: "float",
    mx.bfloat16: "bfloat16_t",
}


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — fused group quantize + dequantize
# ===========================================================================
# Grid:        (n_row_groups * THREADS, 1, 1) — one threadgroup per (row, group).
# Threadgroup: (THREADS, 1, 1).
#
# GROUP_SIZE, LEVELS, EPS, THREADS and the element type T are injected as
# compile-time #defines so the tree reduction unrolls. The array shape (R, L)
# is a runtime buffer, not a #define — see _group_quant_kernel.

_KIVI_GROUP_QUANT_SRC = _read_kernel_source("kivi_group_quant.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _group_quant_kernel(group_size: int, levels: int, eps: float, dtype_name: str):
    """Compile (once) the shader variant for this quantization configuration.

    Deliberately **not** keyed on the array shape: ``R``/``L`` are passed as a
    runtime ``shape`` buffer instead of ``#define``\\ s. Specializing on shape
    would compile a new shader for every distinct sequence length — one per
    decode step — so the cache would grow without bound and every step would
    pay a shader compile. The key here spans a handful of values that are
    fixed by the cache's config, so the cache stays tiny.
    """
    key = ("kivi_group_quant", group_size, levels, eps, dtype_name)
    if key not in _cache:
        header = (
            f"#define GROUP_SIZE {group_size}u\n"
            f"#define LEVELS {levels}\n"
            f"#define EPS {eps!r}f\n"
            f"#define THREADS {_THREADS}u\n"
            f"#define T {dtype_name}\n"
        )
        _cache[key] = mx.fast.metal_kernel(
            name=f"kivi_group_quant_g{group_size}_l{levels}",
            input_names=["x", "shape"],
            output_names=["out"],
            header=header,
            source=_KIVI_GROUP_QUANT_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def kivi_group_quant_dequant(
    x: mx.array,
    axis: int,
    group_size: int,
    levels: int,
    eps: float = 1e-8,
) -> mx.array:
    """Round-trip ``x`` through KIVI asymmetric group quantization on the GPU.

    Drop-in replacement for ``KIVIKVCache._quant_dequant_along``, producing
    bit-identical output on every shape the cache uses.

    Args:
        x:          ``[..., S, D]`` float array (fp16/fp32/bf16).
        axis:       Quantization axis within the last two dims — ``-2`` for
                    per-channel (KIVI keys, group along tokens), ``-1`` for
                    per-token (KIVI values, group along channels).
        group_size: Elements per min/max group along ``axis``.
        levels:     ``2**b - 1`` — the top quantization level.
        eps:        Floor on the group scale, for degenerate ``min == max``
                    groups.

    Returns:
        Array of ``x``'s shape and dtype, quantized and reconstructed.
    """
    if axis not in (-1, -2):
        raise ValueError(f"kivi_group_quant_dequant: axis must be -1 or -2, got {axis}")
    if x.ndim < 2:
        raise ValueError(f"kivi_group_quant_dequant: x must be at least 2D, got {x.shape}")
    if group_size < 1:
        raise ValueError(f"kivi_group_quant_dequant: group_size must be >= 1, got {group_size}")
    if levels < 1:
        raise ValueError(f"kivi_group_quant_dequant: levels must be >= 1, got {levels}")
    if x.dtype not in _MLX_TO_METAL_DTYPE:
        raise ValueError(
            f"kivi_group_quant_dequant: unsupported dtype {x.dtype}; "
            f"expected one of {sorted(str(d) for d in _MLX_TO_METAL_DTYPE)}"
        )
    if x.size == 0:
        return x

    orig_shape = x.shape
    # Move the quant axis last so the kernel always groups along the fastest-
    # varying dimension, then flatten everything else into rows.
    xm = mx.moveaxis(x, axis, -1) if axis != -1 else x
    moved_shape = xm.shape
    L = moved_shape[-1]
    R = xm.size // L
    flat = mx.contiguous(xm.reshape(R, L))

    n_groups = (L + group_size - 1) // group_size
    n_tg = R * n_groups

    (out,) = _group_quant_kernel(group_size, levels, eps, _MLX_TO_METAL_DTYPE[x.dtype])(
        inputs=[flat, mx.array([R, L], dtype=mx.uint32)],
        grid=(n_tg * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(R, L)],
        output_dtypes=[x.dtype],
    )

    out = out.reshape(moved_shape)
    if axis != -1:
        out = mx.moveaxis(out, -1, axis)
    return out.reshape(orig_shape)


__all__ = ["kivi_group_quant_dequant"]
