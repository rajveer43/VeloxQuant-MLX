"""KIVI asymmetric group quantization Metal kernels.

Fuses the whole ``KIVIKVCache._quant_dequant_along`` round-trip — ``moveaxis``
-> pad -> reshape -> ``min``/``max`` -> ``round``/``clip`` -> reconstruct ->
``moveaxis`` back — into a single Metal dispatch (issue #164).

The MLX path materializes several full-size intermediates (the moved-axis copy,
the padded copy, the reshaped group view, the codes, the reconstruction) for
what is arithmetically a single streaming pass.

**Two kernels, because KIVI's two schemes have opposite memory layouts.** The
cache stores ``[..., S, D]`` row-contiguous, so:

* *Per-token* values (``axis=-1``) group along ``D``, the **contiguous** axis.
  One SIMD group per quantization group: lanes split the group, loads coalesce,
  and a ``simd_shuffle_xor`` butterfly reduces without barriers.
* *Per-channel* keys (``axis=-2``) group along ``S``, the **strided** axis. Here
  one thread owns a whole group and consecutive threads take consecutive
  channels — loads still coalesce, and owning the group outright removes the
  cross-thread reduction entirely.

The obvious alternative for keys — transpose so the token axis becomes
contiguous, then reuse the value kernel — is what this replaces. That transpose
is a full-size materializing copy, and it costs more memory traffic than the
arithmetic being fused; it made the fused kernel a net loss on the key path.

**Bit-exactness with the MLX path** requires three details that are easy to get
wrong, all of which the parity tests pin:

1. *Padding replicates the last live element.* When the quantized axis is not a
   multiple of ``group_size``, the MLX path pads with copies of ``x[..., -1:]``.
   Padding with zeros instead would change the final group's ``min``/``max`` and
   silently shift every reconstruction in that group.
2. *Rounding is half-to-even.* ``mx.round`` is half-to-even; Metal's ``rint()``
   is too, but ``metal::round()`` is half-away-from-zero and would disagree on
   exact ``.5`` codes — which are common here, since ``(v - gmin) / scale``
   lands on exact halves whenever a group's range divides evenly.
3. *``eps`` floors the scale.* Degenerate groups (``min == max``, e.g. a
   single-element group) must divide by ``eps``, not zero.

Public API:
  - :func:`kivi_group_quant_dequant`
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

# Threads per threadgroup for the per-token kernel: one SIMD group, so the
# butterfly reduction stays within a warp and needs no barriers.
_SIMD_WIDTH = 32

# Occupancy knob for the per-channel kernel. Threads there never cooperate, so
# this only affects scheduling granularity.
_CHANNEL_TG = 256

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
# Metal source — per-channel (KIVI keys) and per-token (KIVI values)
# ===========================================================================
# GROUP_SIZE, LEVELS, EPS, DHEAD and the element type T are compile-time
# #defines. DHEAD is the model's head_dim — fixed per cache, and making it
# constant turns the index math into shifts/masks.
#
# The sequence length is deliberately NOT specialized: it grows every decode
# step, so baking it in would compile a fresh shader per token. It arrives via
# MLX's automatic ``x_shape`` buffer instead.

_KIVI_QUANT_CHANNEL_SRC = _read_kernel_source("kivi_group_quant_channel.metal")
_KIVI_QUANT_TOKEN_SRC = _read_kernel_source("kivi_group_quant_token.metal")


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def _header(group_size: int, levels: int, eps: float, head_dim: int, dtype_name: str) -> str:
    return (
        f"#define GROUP_SIZE {group_size}u\n"
        f"#define LEVELS {levels}\n"
        f"#define EPS {eps!r}f\n"
        f"#define DHEAD {head_dim}u\n"
        f"#define T {dtype_name}\n"
    )


def _quant_kernel(
    axis: int, group_size: int, levels: int, eps: float, head_dim: int, dtype_name: str
):
    """Compile (once) the shader variant for this quantization configuration.

    Keyed on values that are fixed by the cache's config — never on the
    sequence length. Specializing on shape would compile a new shader for every
    distinct sequence length (one per decode step), so the cache would grow
    without bound and every step would pay a shader compile.
    """
    key = ("kivi_group_quant", axis, group_size, levels, eps, head_dim, dtype_name)
    if key not in _cache:
        per_channel = axis == -2
        _cache[key] = mx.fast.metal_kernel(
            name=("kivi_group_quant_channel" if per_channel else "kivi_group_quant_token")
            + f"_g{group_size}_l{levels}_d{head_dim}",
            input_names=["x"],
            output_names=["out"],
            header=_header(group_size, levels, eps, head_dim, dtype_name),
            source=_KIVI_QUANT_CHANNEL_SRC if per_channel else _KIVI_QUANT_TOKEN_SRC,
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
    S, D = orig_shape[-2], orig_shape[-1]
    BH = x.size // (S * D)
    # Free for a contiguous array: this only flattens leading dims, it does not
    # move the quantization axis. Nothing here materializes a transpose.
    flat = mx.contiguous(x.reshape(BH, S, D))

    kernel = _quant_kernel(axis, group_size, levels, eps, D, _MLX_TO_METAL_DTYPE[x.dtype])

    if axis == -2:
        # One thread per (bh, token-group, channel).
        n_threads = BH * math.ceil(S / group_size) * D
        tg = min(_CHANNEL_TG, n_threads)
        grid = (math.ceil(n_threads / tg) * tg, 1, 1)
        threadgroup = (tg, 1, 1)
    else:
        # One SIMD group per (bh, token, channel-group).
        n_groups = BH * S * math.ceil(D / group_size)
        grid = (n_groups * _SIMD_WIDTH, 1, 1)
        threadgroup = (_SIMD_WIDTH, 1, 1)

    (out,) = kernel(
        inputs=[flat],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(BH, S, D)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(orig_shape)


__all__ = ["kivi_group_quant_dequant"]
