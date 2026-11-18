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
from fractions import Fraction
from functools import lru_cache
from numbers import Integral, Real
from typing import NamedTuple

import mlx.core as mx

from veloxquant_mlx.quantizers._quant_utils import _truncated_svd
from veloxquant_mlx.quantizers.cachegen import CodeStream, dequant_codes, quantize_to_codes

# ---------------------------------------------------------------------------
# Argument validation. Bad values used to be accepted silently (sparse_frac>1,
# rank<0) or fail deep inside the SVD with an unrelated message. ``name`` lets
# ``GEARKVCache`` report the config field the caller actually set.
# ---------------------------------------------------------------------------
_MAX_BITS = 8


def _as_int(value: object) -> int | None:
    """``value`` as an int if it is integral (``8`` or ``8.0``), else ``None``.

    Configs routinely carry whole numbers as floats (JSON/YAML, ``8.0``), and
    the old ``int(...)`` coercion accepted them, so they must keep working.
    A fractional value (``8.5``) is rejected rather than truncated, and bools,
    NaN and Inf are not integers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        f = float(value)
        if math.isfinite(f) and f.is_integer():
            return int(f)
    return None


def _validate_bits(bits: object, name: str = "bits") -> int:
    n = _as_int(bits)
    if n is None or not 1 <= n <= _MAX_BITS:
        raise ValueError(f"{name}={bits!r} must be an integer in [1, {_MAX_BITS}].")
    return n


def _validate_group_size(group_size: object, name: str = "group_size") -> int:
    n = _as_int(group_size)
    if n is None or n < 1:
        raise ValueError(f"{name}={group_size!r} must be an integer >= 1.")
    return n


def _validate_rank(rank: object, name: str = "rank") -> int | None:
    """``None`` (energy-chosen) or an integer ``>= 0`` (0 = no low-rank term)."""
    if rank is None:
        return None
    n = _as_int(rank)
    if n is None or n < 0:
        raise ValueError(f"{name}={rank!r} must be None or an integer >= 0.")
    return n


def _validate_sparse_frac(frac: object, name: str = "sparse_frac") -> float:
    """A finite fraction in ``[0, 1]`` (0 = no sparse term)."""
    try:
        f = float(frac)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name}={frac!r} must be a finite fraction in [0, 1].") from None
    if not math.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ValueError(f"{name}={frac!r} must be a finite fraction in [0, 1].")
    return f


def _validate_energy_threshold(t: object, name: str = "energy_threshold") -> float:
    """A finite fraction in ``(0, 1]``; only used when the rank is chosen by energy."""
    try:
        f = float(t)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name}={t!r} must be a finite number in (0, 1].") from None
    if not math.isfinite(f) or not 0.0 < f <= 1.0:
        raise ValueError(f"{name}={t!r} must be a finite number in (0, 1].")
    return f


def _validate_matrix(x: mx.array, name: str = "x", *, check_finite: bool = True) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2-D [N, D] matrix, got shape {tuple(x.shape)}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {tuple(x.shape)}.")
    if check_finite and not bool(mx.all(mx.isfinite(x)).item()):
        raise ValueError(f"{name} must be finite (found NaN or Inf).")


def validate_gear_config(config: object) -> tuple[int, int | None, float, float, int]:
    """Validate the ``gear_*`` fields of a ``KVCacheConfig`` (or look-alike).

    Returns ``(bits, rank, energy_threshold, sparse_fraction, group_size)``,
    using the same defaults as :class:`GEARKVCache`. Shared by the cache
    constructor and ``KVCacheConfig.__post_init__`` so both reject the same
    values with the same messages.
    """
    return (
        _validate_bits(getattr(config, "gear_bits", 2), "gear_bits"),
        _validate_rank(getattr(config, "gear_rank", None), "gear_rank"),
        _validate_energy_threshold(
            getattr(config, "gear_energy_threshold", 0.90), "gear_energy_threshold"
        ),
        _validate_sparse_frac(
            getattr(config, "gear_sparse_fraction", 0.01), "gear_sparse_fraction"
        ),
        _validate_group_size(getattr(config, "gear_group_size", 32), "gear_group_size"),
    )


def _validate_state(state: GEARState) -> None:
    if state.d_cols <= 0:
        raise ValueError(
            f"GEARState.d_cols={state.d_cols} must be the original channel count D (> 0); "
            "it is required to rebuild the token-axis view and to bill bytes."
        )


@lru_cache(maxsize=1024)
def _nnz_for(total: int, frac: float) -> int:
    """``floor(total * frac)`` in the decimal the caller wrote.

    ``int(total * frac)`` truncates binary-float error: ``int(100 * 0.29)`` is
    28 although 0.29 of 100 is 29. ``repr`` gives the shortest decimal that
    round-trips, so the product is exact.
    """
    return int(Fraction(repr(float(frac))) * total)


class GEARState(NamedTuple):
    """A GEAR-compressed tensor: base codes + low-rank residual + sparse outliers.

    Attributes:
        codes:  [n_groups, group_size, W] fp32 integer base codes.
        scale:  [n_groups, 1, W] fp32 per-group step.
        zero:   [n_groups, 1, W] fp32 per-group min.
            ``W`` is the stream's own column count: ``D`` with ``n_groups``
            token-groups for ``axis == "channel"``, but the token count ``N``
            with ``n_groups`` channel-groups for ``axis == "token"``.
        L:      [N, r] fp32 low-rank residual left factor (or None if rank==0).
        R:      [r, D] fp32 low-rank residual right factor (or None if rank==0).
        sp_idx: [nnz] int32 flattened indices into the [N, D] residual (or None).
        sp_val: [nnz] fp16 outlier residual values (or None).
        n_rows: int original (pre-pad) token count (the ``X.shape[0]`` GEAR was
            given — NOT necessarily the row count of ``codes``, see ``axis``).
        bits:   int base bit-width.
        rank:   int residual low-rank (0 = no low-rank term).
        axis:   str base quant axis — "channel" (KCVT key scheme) or "token"
            (KCVT value scheme). ``codes``/``scale``/``zero`` are transposed
            (row count == ``d_cols``) when ``axis == "token"``.
        d_cols: int original channel count ``D`` — the row count of ``codes``
            when ``axis == "token"`` (unused, equals ``codes.shape[-1]``,
            when ``axis == "channel"``).
    """

    codes: mx.array
    scale: mx.array
    zero: mx.array
    L: mx.array | None
    R: mx.array | None
    sp_idx: mx.array | None
    sp_val: mx.array | None
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
    bits = _validate_bits(bits)
    group_size = _validate_group_size(group_size)
    if axis == "channel":
        # quantize_to_codes groups along axis 0 (rows) of whatever it is
        # given. For untransposed x [N, D], groups are runs of tokens, and
        # scale/zero come out [n_groups, 1, D] — one (scale, zero) per
        # channel, pooled over each group of tokens. That already IS the
        # paper's per-channel KCVT scheme (:issue:`615`; previously this
        # branch transposed first, which groups along channels instead and
        # produces one (scale, zero) per *token* — the "token" scheme).
        stream = quantize_to_codes(x, bits, group_size)
        return stream, dequant_codes(stream)
    # token: transpose so tokens become rows, group along them (i.e. along
    # channels per token), transpose back. One (scale, zero) per token.
    stream_t = quantize_to_codes(x.T, bits, group_size)
    recon = dequant_codes(stream_t).T
    # stream_t is already token-major-of-the-transpose: codes
    # [d_groups, group_size, n], stream_t.n_rows == d (the channel count).
    # Pass it through as-is — the caller (gear_compress) stores
    # GEARState.d_cols separately so gear_reconstruct can rebuild this exact
    # transposed-major view.
    return stream_t, recon


def _dequant_axis(stream: CodeStream, axis: str) -> mx.array:
    """Dequantize a ``CodeStream`` produced by :func:`quantize_base`, axis-aware.

    ``stream.n_rows`` must already be set to the row count *of that stream*
    (tokens for ``"token"``, channels for ``"channel"``) — callers rebuild the
    view with the right count rather than reusing ``GEARState.n_rows`` (which
    is always the original token count, ambiguous for the channel axis).
    """
    if axis == "channel":
        return dequant_codes(stream)
    # token: stream is transposed-major; dequant then transpose back to [N, D].
    return dequant_codes(stream).T


def residual(x: mx.array, base_recon: mx.array) -> mx.array:
    """Quantization residual ``E = x - base_recon`` as fp32 ``[N, D]``."""
    return x.astype(mx.float32) - base_recon.astype(mx.float32)


def lowrank_error(
    E: mx.array,
    rank: int | None,
    energy_threshold: float = 0.90,
) -> tuple[mx.array | None, mx.array | None]:
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
    rank = _validate_rank(rank)
    energy_threshold = _validate_energy_threshold(energy_threshold)
    _validate_matrix(E, "E", check_finite=False)
    if rank == 0:
        return None, None
    U_r, s_r, Vt_r = _truncated_svd(E, rank=rank, energy_threshold=energy_threshold)
    L = U_r * s_r[None, :]  # fold singular values into the left factor
    R = Vt_r  # [r, D]
    return L, R


def sparse_outliers(
    resid: mx.array,
    frac: float,
) -> tuple[mx.array | None, mx.array | None]:
    """Top-``frac`` entries of ``resid`` by magnitude → ``(flat_idx, values)``.

    The outlier correction GEAR applies to the residual the low-rank term could
    not absorb. Selection is by absolute value over the flattened ``[N, D]``.

    Args:
        resid: Post-low-rank residual ``[N, D]`` fp32.
        frac: Fraction of entries to keep exact (0 → no sparse term).

    Returns:
        ``(flat_idx[int32], values[fp16])`` or ``(None, None)`` if no sparse term.
    """
    frac = _validate_sparse_frac(frac)
    if frac <= 0.0:
        return None, None
    n, d = resid.shape
    total = n * d
    nnz = _nnz_for(total, frac)
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
    rank: int | None = None,
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
    # Finiteness is not checked up front: that costs a host sync on every call.
    # The residual SVD raises on NaN/Inf, and only then do we confirm and re-raise
    # a clear error, so valid input pays nothing. (rank=0 runs no SVD and so, as
    # before, does not look at the values.)
    _validate_matrix(x, "x", check_finite=False)
    bits = _validate_bits(bits)
    group_size = _validate_group_size(group_size)
    rank = _validate_rank(rank)
    energy_threshold = _validate_energy_threshold(energy_threshold)
    sparse_frac = _validate_sparse_frac(sparse_frac)
    x32 = x.astype(mx.float32)
    n, d = int(x32.shape[0]), int(x32.shape[1])
    stream, base_recon = quantize_base(x32, bits, group_size, axis=base_axis)
    E = residual(x32, base_recon)  # [N, D]

    try:
        L, R = lowrank_error(E, rank, energy_threshold)
    except RuntimeError:
        if not bool(mx.all(mx.isfinite(x32)).item()):
            raise ValueError("x must be finite (found NaN or Inf).") from None
        raise
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


def sparse_outliers_batched(
    resid: mx.array,
    frac: float,
) -> tuple[mx.array | None, mx.array | None]:
    """Batched-leading-axis equivalent of :func:`sparse_outliers` (#570).

    Every row of a ``_compress_and_account`` batch shares the same ``[N, D]``
    shape (same ``S``/``D`` for the whole call), so ``nnz`` — unlike SVD rank
    — is already uniform across the batch; no ragged-width padding/truncation
    is needed here (contrast :func:`_truncated_svd_batched`, whose per-row
    rank genuinely varies and must be handled that way).

    Args:
        resid: Post-low-rank residual ``[BH, N, D]`` fp32.
        frac: Fraction of entries to keep exact (0 → no sparse term).

    Returns:
        ``(flat_idx[int32], values[fp16])`` each ``[BH, nnz]``, or
        ``(None, None)`` if no sparse term — matching :func:`sparse_outliers`'
        per-row contract, just with a leading ``BH`` axis.
    """
    frac = _validate_sparse_frac(frac)
    if frac <= 0.0:
        return None, None
    bh, n, d = resid.shape
    total = n * d
    nnz = _nnz_for(total, frac)
    if nnz <= 0:
        return None, None
    flat = resid.reshape(bh, total)
    mag = mx.abs(flat)
    order = mx.argsort(mag, axis=-1)  # ascending, per row
    top = order[:, total - nnz :]  # [BH, nnz] largest-magnitude indices per row
    top = top.astype(mx.int32)
    vals = mx.take_along_axis(flat, top, axis=-1).astype(mx.float16)
    return top, vals


def _scatter_add_rows(out: mx.array, sp_idx: mx.array, sp_val: mx.array) -> mx.array:
    """Add each head's sparse outliers into ``out`` ``[BH, N, D]`` without flattening heads.

    The previous ``out.reshape(-1)`` + ``flat.at[row_base + idx]`` form needs
    one flat array of ``BH*N*D`` elements. MLX array dimensions are int32, so
    once ``BH*N*D > 2**31 - 1`` that reshape raises ``OverflowError`` and the
    reconstruction fails outright. Scattering with ``(row, col)`` index pairs
    keeps every dimension (``BH`` and ``N*D``) within int32 and no cross-head
    offset is ever formed, so it stays correct for any batch whose per-head
    ``N*D`` fits (which ``sp_idx`` being int32 already requires).
    """
    bh, n, d = out.shape
    rows = mx.broadcast_to(mx.arange(bh, dtype=mx.int32)[:, None], sp_idx.shape)
    flat = out.reshape(bh, n * d).at[rows, sp_idx].add(sp_val.astype(out.dtype))
    return flat.reshape(bh, n, d)


def gear_reconstruct_batched(
    base: mx.array,
    L: mx.array | None,
    R: mx.array | None,
    sp_idx: mx.array | None,
    sp_val: mx.array | None,
    out_dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Batched-leading-axis equivalent of :func:`gear_reconstruct` (#570).

    Replaces the ``for idx in range(B * H): gear_reconstruct(...)`` loop —
    the largest remaining per-head cost in ``GEARKVCache._compress_and_account``
    after #504 batched Passes 1-2 (~50% of wall time in isolated profiling,
    per that class's module docstring). The low-rank add and sparse scatter
    for head ``(b, h)`` only ever touch that head's own factors/mask — no
    cross-head coupling — so both are pure vectorizations over the batch axis.

    ``L``/``R`` here are the **padded, batch-max-rank** factors straight out
    of ``_truncated_svd_batched`` (already zero-masked beyond each row's own
    rank), NOT per-row-truncated to ``ranks[idx]`` — unlike storage/byte
    accounting, the padded columns are numerically inert for this batched
    matmul (zero singular values contribute exactly 0 to ``L @ R``), so
    skipping the per-row slice here is correctness-preserving and avoids
    rebuilding a ragged Python list just to re-stack it.

    Args:
        base: Batched base-layer dequant, fp32 ``[BH, N, D]`` (Pass 1's
            output — already computed once, batched, upstream).
        L: ``[BH, N, max_rank]`` fp32 or None (no low-rank term for the batch).
        R: ``[BH, max_rank, D]`` fp32 or None.
        sp_idx: ``[BH, nnz]`` int32 flattened indices into each row's
            ``[N, D]``, or None (no sparse term).
        sp_val: ``[BH, nnz]`` fp16 outlier values, or None.

    Returns:
        fp16 ``[BH, N, D]`` reconstruction, matching
        ``mx.stack([gear_reconstruct(...) for idx in range(BH)], axis=0)``.
    """
    out = base.astype(mx.float32)
    if L is not None and R is not None:
        out = out + (L @ R)
    if sp_idx is not None and sp_val is not None:
        out = _scatter_add_rows(out, sp_idx, sp_val)
    return out.astype(out_dtype)


def gear_reconstruct(state: GEARState, base: mx.array | None = None) -> mx.array:
    """Reconstruct fp16 ``[n_rows, D]`` from a GEARState (base + low-rank + sparse).

    Args:
        state: The compressed tensor.
        base: Optional precomputed base-layer dequant (fp32, ``[n_rows, D]``),
            i.e. ``dequant(Quant_b(x))`` for this exact ``state``. Callers that
            already computed this (e.g. ``GEARKVCache._compress_and_account``,
            whose Pass 1 batches the base dequant across all ``B*H`` heads
            before Pass 3 calls this function once per head) should pass it
            through here instead of leaving it to be recomputed — this
            function otherwise redoes the identical per-head
            ``_dequant_axis`` math a second time for no reason. Recomputed
            from ``state`` when omitted, preserving the original behavior for
            standalone callers (e.g. :func:`gear_quant_dequant`).
    """
    _validate_state(state)
    if base is None:
        stream_rows = state.d_cols if state.axis == "token" else state.n_rows
        stream = _CodeStreamView(state.codes, state.scale, state.zero, stream_rows, state.bits)
        base = _dequant_axis(stream, state.axis).astype(mx.float32)
    else:
        base = base.astype(mx.float32)
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
    _validate_state(state)
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
    rank: int | None = None,
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
    "sparse_outliers_batched",
    "gear_compress",
    "gear_reconstruct",
    "gear_reconstruct_batched",
    "gear_bytes",
    "base_only_bytes",
    "gear_quant_dequant",
    "validate_gear_config",
]
