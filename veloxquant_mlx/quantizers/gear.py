"""GEAR quantizer — error-feedback: residual low-rank + sparse outlier correction.

Inspired by "GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless
Generative Inference of LLM" (Kang et al., arXiv:2403.05527). Documented as
"GEAR-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

What GEAR adds that the repo did not have: **error feedback**. Every other
method in the suite picks a bit-width (or a cache layout) and lives with the
quantization error. GEAR makes *any* ultra-low-bit base quantizer near-lossless
by reconstructing what it threw away, via the three-part decomposition

    X  ~=  Quant_b(X)  +  L . R  +  S

  1. ``Quant_b(X)`` — base: most entries at ultra-low precision. The paper's
     backbone is **KCVT**: keys quantized **per-channel** (§2, "Group-wise
     Quantization"), values quantized **per-token**, both without KIVI's
     fine-grained residual buffer. We reuse the repo's asymmetric min/max group
     quant (``cachegen.quantize_to_codes``) but apply it along the axis KCVT
     specifies — ``base_axis="channel"`` for keys, ``"token"`` for values — so
     GEAR composes over an existing, already-tested quantizer with the paper's
     actual backbone rather than a generic per-token quantizer for both.
  2. ``L . R``      — a low-rank approximation of the quantization residual
     ``E = X - dequant(Quant_b(X))``. The residual of a coherent KV matrix is
     itself low-rank, so a small rank recovers most of the lost signal cheaply.
  3. ``S``          — a sparse matrix correcting the top-``rho`` outlier entries
     by magnitude in the post-low-rank residual, the few the low-rank term could
     not absorb.

Adaptation:
  * The residual SVD is computed on the prefill batch (the cache wrapper owns the
    prefill-vs-decode orchestration), reusing the SVDq/PALU prefill-SVD pattern.
    The paper additionally uses a smaller rank (r=2) for streaming decode-only
    updates via a token buffer (§3, "Streaming Buffer") purely as a speed
    optimization — the reconstruction target is unaffected, so we compute the
    same low-rank recovery on whatever batch the cache holds. Not ported: no
    perf claim depends on it.
  * We do **not** ship GEAR's fused streaming-dequant CUDA kernel. The wrapper
    reconstructs fp16 then calls MLX SDPA; stored size shrinks, attend-time peak
    memory does not. Documented as a known simplification.
  * Reconstruction is a real lossy reconstruction (unlike CacheGen's lossless
    byte model): the reported ``compressed_*_bytes`` AND the values the model
    sees both reflect the error-feedback layers.

This module holds the pure numerics: base quant (borrowed, KCVT-axis-aware),
residual, residual-low-rank, sparse-outlier extraction, full
compress/reconstruct, and an honest byte estimator. The cache wrapper owns the
per-layer prefill/decode state.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import mlx.core as mx

from veloxquant_mlx.quantizers._quant_utils import _truncated_svd
from veloxquant_mlx.quantizers.cachegen import CodeStream, dequant_codes, quantize_to_codes


class GEARState(NamedTuple):
    """A GEAR-compressed tensor: base codes + low-rank residual + sparse outliers.

    Attributes:
        codes:  [n_groups, group_size, D] fp32 integer base codes.
        scale:  [n_groups, 1, D] fp32 per-group step.
        zero:   [n_groups, 1, D] fp32 per-group min.
        L:      [N, r] fp32 low-rank residual left factor (or None if rank==0).
        R:      [r, D] fp32 low-rank residual right factor (or None if rank==0).
        sp_idx: [nnz] int32 flattened indices into the [N, D] residual (or None).
        sp_val: [nnz] fp16 outlier residual values (or None).
        n_rows: int original (pre-pad) token count (the ``X.shape[0]`` GEAR was
            given — NOT necessarily the row count of ``codes``, see ``axis``).
        bits:   int base bit-width.
        rank:   int residual low-rank (0 = no low-rank term).
        axis:   str base quant axis — "channel" (KCVT key scheme) or "token"
            (KCVT value scheme). ``codes``/``scale``/``zero`` are channel-major
            (transposed, row count == ``d_cols``) when ``axis == "channel"``.
        d_cols: int original channel count ``D`` — the row count of ``codes``
            when ``axis == "channel"`` (unused, equals ``codes.shape[-1]``,
            when ``axis == "token"``).
    """

    codes: mx.array
    scale: mx.array
    zero: mx.array
    L: Optional[mx.array]
    R: Optional[mx.array]
    sp_idx: Optional[mx.array]
    sp_val: Optional[mx.array]
    n_rows: int
    bits: int
    rank: int
    axis: str = "token"
    d_cols: int = 0


def quantize_base(x: mx.array, bits: int, group_size: int = 32, axis: str = "token"):
    """Base ultra-low-bit group quant. Returns ``(CodeStream, base_recon[N, D])``.

    Borrows the repo's shared asymmetric min/max group quant (the same math
    KIVI/CacheGen use) so GEAR's base layer is not new quant logic — only the
    *axis* it groups along is chosen here, per the paper's KCVT backbone.

    Args:
        x: ``[N, D]`` one head's keys or values.
        bits: base bit-width.
        group_size: group size along the quantization axis.
        axis: ``"channel"`` (paper's Key scheme — group along the token axis,
            one ``(scale, zero)`` per channel) or ``"token"`` (paper's Value
            scheme — group along the channel axis, one ``(scale, zero)`` per
            token; also the default, matching this module's pre-KCVT behavior
            for callers that do not pass ``axis``).
    """
    if axis not in ("channel", "token"):
        raise ValueError(f"quantize_base: axis={axis!r} must be 'channel' or 'token'.")
    if axis == "token":
        stream = quantize_to_codes(x, bits, group_size)
        return stream, dequant_codes(stream)
    # channel: transpose so channels become rows, group along them, transpose back.
    stream_t = quantize_to_codes(x.T, bits, group_size)
    recon = dequant_codes(stream_t).T
    # stream_t is already channel-major: codes [d_groups, group_size, n],
    # stream_t.n_rows == d (the channel count). Pass it through as-is — the
    # caller (gear_compress) stores GEARState.d_cols separately so
    # gear_reconstruct can rebuild this exact channel-major view.
    return stream_t, recon


def _dequant_axis(stream: CodeStream, axis: str) -> mx.array:
    """Dequantize a ``CodeStream`` produced by :func:`quantize_base`, axis-aware.

    ``stream.n_rows`` must already be set to the row count *of that stream*
    (tokens for ``"token"``, channels for ``"channel"``) — callers rebuild the
    view with the right count rather than reusing ``GEARState.n_rows`` (which
    is always the original token count, ambiguous for the channel axis).
    """
    if axis == "token":
        return dequant_codes(stream)
    # channel: stream is channel-major; dequant then transpose back to [N, D].
    return dequant_codes(stream).T


def residual(x: mx.array, base_recon: mx.array) -> mx.array:
    """Quantization residual ``E = x - base_recon`` as fp32 ``[N, D]``."""
    return x.astype(mx.float32) - base_recon.astype(mx.float32)


def lowrank_error(
    E: mx.array,
    rank: Optional[int],
    energy_threshold: float = 0.90,
) -> tuple[Optional[mx.array], Optional[mx.array]]:
    """Low-rank factors ``(L, R)`` of the residual via truncated SVD.

    ``E ~= L @ R`` with ``L = U_r * s_r`` (``[N, r]``) and ``R = Vt_r`` (``[r, D]``).
    Returns ``(None, None)`` when ``rank == 0`` (pure-sparse / base-only mode).

    Args:
        E: Residual ``[N, D]`` fp32.
        rank: Explicit low-rank ``r``; 0 disables the low-rank term; None →
            chosen by ``energy_threshold``.
        energy_threshold: Singular-value energy to retain when ``rank`` is None.

    Returns:
        ``(L, R)`` fp32, or ``(None, None)`` if no low-rank term.
    """
    if rank is not None and int(rank) == 0:
        return None, None
    U_r, s_r, Vt_r = _truncated_svd(E, rank=rank, energy_threshold=energy_threshold)
    L = U_r * s_r[None, :]  # fold singular values into the left factor
    R = Vt_r  # [r, D]
    return L, R


def sparse_outliers(
    resid: mx.array,
    frac: float,
) -> tuple[Optional[mx.array], Optional[mx.array]]:
    """Top-``frac`` entries of ``resid`` by magnitude → ``(flat_idx, values)``.

    The outlier correction GEAR applies to the residual the low-rank term could
    not absorb. Selection is by absolute value over the flattened ``[N, D]``.

    Args:
        resid: Post-low-rank residual ``[N, D]`` fp32.
        frac: Fraction of entries to keep exact (0 → no sparse term).

    Returns:
        ``(flat_idx[int32], values[fp16])`` or ``(None, None)`` if no sparse term.
    """
    if frac <= 0.0:
        return None, None
    n, d = resid.shape
    total = n * d
    nnz = int(total * frac)
    if nnz <= 0:
        return None, None
    flat = resid.reshape(-1)
    mag = mx.abs(flat)
    order = mx.argsort(mag)  # ascending
    top = order[total - nnz :]  # largest-magnitude indices
    top = top.astype(mx.int32)
    vals = flat[top].astype(mx.float16)
    return top, vals


def gear_compress(
    x: mx.array,
    bits: int = 2,
    rank: Optional[int] = None,
    sparse_frac: float = 0.01,
    group_size: int = 32,
    energy_threshold: float = 0.90,
    base_axis: str = "token",
) -> GEARState:
    """Full GEAR compression of one head's K/V matrix ``[N, D]`` → ``GEARState``.

    Pipeline: base group quant (KCVT-axis-aware) → residual → low-rank residual
    → sparse outliers of the *post-low-rank* residual.

    Args:
        base_axis: ``"channel"`` for keys, ``"token"`` for values, per the
            paper's KCVT backbone. See :func:`quantize_base`.
    """
    x32 = x.astype(mx.float32)
    n, d = int(x32.shape[0]), int(x32.shape[1])
    stream, base_recon = quantize_base(x32, bits, group_size, axis=base_axis)
    E = residual(x32, base_recon)  # [N, D]

    L, R = lowrank_error(E, rank, energy_threshold)
    if L is not None:
        E_after = E - (L @ R)
        r = int(L.shape[1])
    else:
        E_after = E
        r = 0

    sp_idx, sp_val = sparse_outliers(E_after, sparse_frac)

    return GEARState(
        codes=stream.codes,
        scale=stream.scale,
        zero=stream.zero,
        L=L,
        R=R,
        sp_idx=sp_idx,
        sp_val=sp_val,
        n_rows=n,
        bits=bits,
        rank=r,
        axis=base_axis,
        d_cols=d,
    )


def gear_reconstruct(state: GEARState) -> mx.array:
    """Reconstruct fp16 ``[n_rows, D]`` from a GEARState (base + low-rank + sparse)."""
    stream_rows = state.d_cols if state.axis == "channel" else state.n_rows
    stream = _CodeStreamView(state.codes, state.scale, state.zero, stream_rows, state.bits)
    base = _dequant_axis(stream, state.axis).astype(mx.float32)
    out = base
    if state.L is not None and state.R is not None:
        out = out + (state.L @ state.R)
    if state.sp_idx is not None and state.sp_val is not None:
        n, d = out.shape
        flat = out.reshape(-1)
        flat = flat.at[state.sp_idx].add(state.sp_val.astype(mx.float32))
        out = flat.reshape(n, d)
    return out.astype(mx.float16)


class _CodeStreamView(NamedTuple):
    """Minimal CodeStream-shaped view so ``dequant_codes`` can rebuild the base."""

    codes: mx.array
    scale: mx.array
    zero: mx.array
    n_rows: int
    bits: int


def _base_code_and_param_bytes(state: GEARState) -> tuple[int, int]:
    """Shared base-layer byte math, correct for either KCVT axis.

    ``state.codes`` is ``[n_groups, group_size, W]`` where ``W`` is the
    *stream's own* column count — the original token count ``n_rows`` when
    ``axis == "token"``, or the original channel count ``d_cols`` when
    ``axis == "channel"``. Either way ``total elements == n_rows * d_cols``,
    so code size is axis-independent; only ``param_bytes`` (one scale/zero per
    group per stream-column) needs the stream's own width ``W``.
    """
    n_groups, _gs, w = state.codes.shape
    total_elems = state.n_rows * state.d_cols
    code_bytes = math.ceil(total_elems * state.bits / 8)
    param_bytes = n_groups * w * 2 * 2  # scale + zero, fp16
    return code_bytes, param_bytes


def gear_bytes(state: GEARState) -> int:
    """Honest stored size (bytes) of a GEARState.

    base codes (fixed-width packed) + fp16 group params + fp16 ``L,R`` factors +
    sparse triples (int32 index + fp16 value per nnz).
    """
    code_bytes, param_bytes = _base_code_and_param_bytes(state)
    lr_bytes = 0
    if state.L is not None and state.R is not None:
        lr_bytes = (state.L.shape[0] * state.L.shape[1] + state.R.shape[0] * state.R.shape[1]) * 2
    sp_bytes = 0
    if state.sp_idx is not None:
        nnz = int(state.sp_idx.shape[0])
        sp_bytes = nnz * (4 + 2)  # int32 index + fp16 value
    return int(code_bytes + param_bytes + lr_bytes + sp_bytes)


def base_only_bytes(state: GEARState) -> int:
    """Stored size of the base codes alone (no error-feedback) — the baseline."""
    code_bytes, param_bytes = _base_code_and_param_bytes(state)
    return int(code_bytes + param_bytes)


def gear_quant_dequant(
    x: mx.array,
    bits: int = 2,
    rank: Optional[int] = None,
    sparse_frac: float = 0.01,
    group_size: int = 32,
    energy_threshold: float = 0.90,
    base_axis: str = "token",
) -> mx.array:
    """Drop-in quant→dequant: full GEAR reconstruction of ``[N, D]`` → fp16."""
    return gear_reconstruct(
        gear_compress(x, bits, rank, sparse_frac, group_size, energy_threshold, base_axis)
    )


__all__ = [
    "GEARState",
    "quantize_base",
    "residual",
    "lowrank_error",
    "sparse_outliers",
    "gear_compress",
    "gear_reconstruct",
    "gear_bytes",
    "base_only_bytes",
    "gear_quant_dequant",
]
