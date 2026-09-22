"""Shared low-level quantization helpers used by multiple quantizers."""

from __future__ import annotations

import mlx.core as mx


def _group_quant_codes(
    x: mx.array, b: int, group_size: int = 32
) -> tuple[mx.array, mx.array, mx.array]:
    """Asymmetric min/max group quantization along axis 0. Returns raw codes.

    Groups are formed along axis 0 (token/sequence axis). Each group is
    independently scaled and zero-pointed using the group's min/max values.
    This is the split-output core shared by :func:`_group_quant_dequant`
    (round-tripped fp16) and any caller that needs the codes/scale/zero
    triple directly (e.g. to store codes at a reduced bit-width).

    Args:
        x: Input array [N, D] fp16 or fp32.
        b: Bit width (1–8).
        group_size: Number of rows per quantization group.

    Returns:
        ``(codes, scale, zero)`` where ``codes`` is ``[n_groups, group_size, D]``
        fp32 (unpacked, pre-rounding-dtype-cast), and ``scale``/``zero`` are
        ``[n_groups, 1, D]`` fp32. Note ``codes`` retains the padded row count
        (``n_groups * group_size``, not the original ``N``) — callers reshape
        and truncate as needed (see :func:`_group_dequant_codes`).
    """
    n, d = x.shape
    gs = group_size
    n_groups = (n + gs - 1) // gs
    pad = n_groups * gs - n
    x32 = x.astype(mx.float32)
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[-1:], (pad, d))], axis=0)
    xg = x32.reshape(n_groups, gs, d)
    gmin = mx.min(xg, axis=1, keepdims=True)
    gmax = mx.max(xg, axis=1, keepdims=True)
    levels = (1 << b) - 1
    eps = 1e-8
    scale = mx.maximum((gmax - gmin) / levels, eps)
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
    return codes, scale, gmin


def _group_dequant_codes(
    codes: mx.array, scale: mx.array, zero: mx.array, n: int, group_size: int = 32
) -> mx.array:
    """Reconstruct ``[n, D]`` fp32 from :func:`_group_quant_codes` output.

    Args:
        codes: ``[n_groups, group_size, D]`` (as returned by
            :func:`_group_quant_codes`) or ``[n_groups * group_size, D]``.
        scale: ``[n_groups, 1, D]`` or ``[n_groups, D]`` fp32.
        zero: ``[n_groups, 1, D]`` or ``[n_groups, D]`` fp32, same shape as
            ``scale``.
        n: Original (unpadded) row count to truncate back to.
        group_size: Number of rows per quantization group (must match the
            value used to produce ``codes``/``scale``/``zero``).

    Returns:
        Reconstructed array ``[n, D]`` fp32.
    """
    d = codes.shape[-1]
    n_groups = scale.shape[0]
    gs = group_size
    codes = codes.reshape(n_groups, gs, d)
    scale = scale.reshape(n_groups, 1, d)
    zero = zero.reshape(n_groups, 1, d)
    recon = codes * scale + zero
    return recon.reshape(n_groups * gs, d)[:n]


def _group_quant_dequant(x: mx.array, b: int, group_size: int = 32) -> mx.array:
    """Asymmetric min/max group quantization along axis 0. Returns fp16.

    Groups are formed along axis 0 (token/sequence axis). Each group is
    independently scaled and zero-pointed using the group's min/max values.

    Args:
        x: Input array [N, D] fp16 or fp32.
        b: Bit width (1–8).
        group_size: Number of rows per quantization group.

    Returns:
        Quantized-then-dequantized array [N, D] fp16.
    """
    n = x.shape[0]
    codes, scale, gmin = _group_quant_codes(x, b, group_size)
    recon = _group_dequant_codes(codes, scale, gmin, n, group_size)
    return recon.astype(mx.float16)


def _group_quant_codes_batched(
    x: mx.array, b: int, group_size: int = 32
) -> tuple[mx.array, mx.array, mx.array]:
    """Batched-leading-axis equivalent of :func:`_group_quant_codes`.

    Groups are formed along axis 1 (the per-row token/sequence axis); axis 0
    is an independent leading batch axis (e.g. flattened ``B*H`` rows).
    Replaces a Python loop calling :func:`_group_quant_codes` (or the
    equivalent ``cachegen.quantize_to_codes``, which shares this exact
    math) once per row — see GEARKVCache's ``_compress_and_account`` for a
    real caller (VeloxQuant-MLX#504: after batching this class's SVD, this
    per-head base-quantize step became the new dominant cost — 62.6% of
    wall time — confirming the same unbatched-loop pattern recurs in a
    caller's base layer even after its own heavier op is fixed).

    Args:
        x: ``[N, S, D]`` fp16 or fp32 — ``N`` independent rows, each with
            its own ``S``-length token axis to group along.
        b: Bit width (1-8), shared by every row in this call.
        group_size: Number of rows (tokens) per quantization group.

    Returns:
        ``(codes, scale, zero)`` where ``codes`` is
        ``[N, n_groups, group_size, D]`` fp32 (unpacked, padded row count),
        and ``scale``/``zero`` are ``[N, n_groups, 1, D]`` fp32. Verified
        bit-for-bit equivalent to looping :func:`_group_quant_codes` over
        axis 0.
    """
    n, s, d = x.shape
    gs = group_size
    n_groups = (s + gs - 1) // gs
    pad = n_groups * gs - s
    x32 = x.astype(mx.float32)
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[:, -1:], (n, pad, d))], axis=1)
    xg = x32.reshape(n, n_groups, gs, d)
    gmin = mx.min(xg, axis=2, keepdims=True)
    gmax = mx.max(xg, axis=2, keepdims=True)
    levels = (1 << b) - 1
    eps = 1e-8
    scale = mx.maximum((gmax - gmin) / levels, eps)
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
    return codes, scale, gmin


def _group_dequant_codes_batched(
    codes: mx.array, scale: mx.array, zero: mx.array, s: int, group_size: int = 32
) -> mx.array:
    """Batched-leading-axis equivalent of :func:`_group_dequant_codes`.

    Args:
        codes: ``[N, n_groups, group_size, D]`` (as returned by
            :func:`_group_quant_codes_batched`).
        scale: ``[N, n_groups, 1, D]`` fp32.
        zero: ``[N, n_groups, 1, D]`` fp32.
        s: Original (unpadded) per-row token count to truncate back to.
        group_size: Must match the value used to produce
            ``codes``/``scale``/``zero``.

    Returns:
        Reconstructed array ``[N, s, D]`` fp32.
    """
    n = codes.shape[0]
    d = codes.shape[-1]
    n_groups = scale.shape[1]
    gs = group_size
    codes = codes.reshape(n, n_groups, gs, d)
    scale = scale.reshape(n, n_groups, 1, d)
    zero = zero.reshape(n, n_groups, 1, d)
    recon = codes * scale + zero
    return recon.reshape(n, n_groups * gs, d)[:, :s]


def _group_quant_dequant_batched(x: mx.array, b: int, group_size: int = 32) -> mx.array:
    """Batched-leading-axis equivalent of :func:`_group_quant_dequant`.

    Groups are formed along axis 1 (the per-row token/sequence axis); axis 0
    is an independent leading batch axis (e.g. flattened ``B*H`` rows) that
    is never mixed across when forming groups — each ``[S, D]`` slice is
    quantized exactly as :func:`_group_quant_dequant` would quantize it on
    its own, just computed as one vectorized call instead of ``N`` separate
    Python-level calls into it. Verified bit-for-bit equivalent to looping
    :func:`_group_quant_dequant` over axis 0 (fp32-rounding-only difference).

    Args:
        x: ``[N, S, D]`` fp16 or fp32 — ``N`` independent rows, each with its
            own ``S``-length token axis to group along.
        b: Bit width (1-8), shared by every row in this call (callers with
            per-row bit-widths call this once per distinct bit-width and
            gather the results — see e.g. ``AdaKVCache._quantize_per_head``).
        group_size: Number of rows (tokens) per quantization group.

    Returns:
        Quantized-then-dequantized array ``[N, S, D]`` fp16.
    """
    n, s, d = x.shape
    gs = group_size
    n_groups = (s + gs - 1) // gs
    pad = n_groups * gs - s
    x32 = x.astype(mx.float32)
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[:, -1:], (n, pad, d))], axis=1)
    xg = x32.reshape(n, n_groups, gs, d)
    gmin = mx.min(xg, axis=2, keepdims=True)
    gmax = mx.max(xg, axis=2, keepdims=True)
    levels = (1 << b) - 1
    eps = 1e-8
    scale = mx.maximum((gmax - gmin) / levels, eps)
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
    recon = codes * scale + gmin
    return recon.reshape(n, n_groups * gs, d)[:, :s].astype(mx.float16)


def _truncated_svd(
    x: mx.array,
    rank: int | None = None,
    energy_threshold: float = 0.90,
) -> tuple[mx.array, mx.array, mx.array]:
    """Truncated SVD of a centered-or-raw matrix ``[N, D]`` → ``(U_r, s_r, Vt_r)``.

    Computes the economy SVD on the CPU stream (MLX ``linalg.svd`` is CPU-only)
    and truncates to ``rank`` singular components. When ``rank`` is None the rank
    is chosen as the smallest prefix whose cumulative singular-value energy
    reaches ``energy_threshold``.

    This is the shared truncation logic used by SVDq (signal SVD), PALU
    (group-head signal SVD), and GEAR (quantization-residual SVD). The caller is
    responsible for any mean-centering it needs before calling.

    Args:
        x: Input matrix ``[N, D]`` fp16 or fp32.
        rank: Explicit rank ``r``. If None, chosen by ``energy_threshold``.
        energy_threshold: Fraction of singular-value energy to retain when
            ``rank`` is None.

    Returns:
        ``(U_r, s_r, Vt_r)`` where ``U_r`` is ``[N, r]``, ``s_r`` is ``[r]``
        (descending), and ``Vt_r`` is ``[r, D]`` — all fp32.
    """
    x32 = x.astype(mx.float32)
    n, d = x32.shape
    U, s_vals, Vt = mx.linalg.svd(x32, stream=mx.cpu)
    mx.eval(U, s_vals, Vt)

    if rank is None:
        total = float(mx.sum(s_vals).item())
        if total < 1e-12:
            rank = 1
        else:
            cumsum = 0.0
            rank = int(s_vals.shape[0])
            for i, sv in enumerate(s_vals.tolist()):
                cumsum += sv
                if cumsum / total >= energy_threshold:
                    rank = i + 1
                    break
    rank = max(1, min(int(rank), int(s_vals.shape[0]), d))

    return U[:, :rank], s_vals[:rank], Vt[:rank, :]


def _truncated_svd_batched(
    x: mx.array,
    rank: int | None = None,
    energy_threshold: float = 0.90,
) -> tuple[mx.array, mx.array, list[int]]:
    """Batched-leading-axis equivalent of :func:`_truncated_svd`.

    Replaces a Python loop calling :func:`_truncated_svd` once per ``(b, h)``
    pair — the dominant cost in GEAR's real-model decode slowdown
    (VeloxQuant-MLX#504: real ``mlx_lm.generate()`` measured 72 -> 5.8 tok/s
    on this M4, ~38% of wall time inside this SVD alone). MLX's
    ``mx.linalg.svd`` natively supports a batched leading axis (verified
    directly: singular values and vectors for each row of a batched call are
    bit-identical to calling it once per row), so the SVD itself is one
    dispatch instead of ``BH`` separate CPU-stream calls.

    When ``rank is None`` (the default — energy-threshold selection), each
    row of the batch may legitimately want a *different* rank (a
    low-effective-rank residual needs few components; a high-effective-rank
    one needs many). Since a single returned array can't hold a
    per-row-variable width, this function returns factors padded to the
    **maximum** rank across the batch, with every row's per-row rank listed
    separately in the third return value — the caller MUST truncate each
    row back to its own rank (``L[i, :, :ranks[i]]``, ``R[i, :ranks[i], :]``)
    before using it for storage/byte-accounting; the padded (untruncated)
    width is only a computational convenience, and using it directly would
    overstate storage cost for any row whose true rank is smaller than the
    batch max (each padded row's singular values beyond its own rank are
    exactly 0.0 — see ``rank_mask`` below — so the extra columns are
    numerically inert for *reconstruction*, but nonzero-shaped, so a caller
    that skips the per-row truncation would silently pay to store zeros).

    Args:
        x: ``[BH, N, D]`` fp32 — ``BH`` independent residual matrices.
        rank: Explicit rank ``r`` shared by every row. If None, chosen
            per-row by ``energy_threshold`` (vectorized — no ``.tolist()``
            or per-row Python loop; verified bit-identical to looping the
            scalar cumulative-energy selection in :func:`_truncated_svd`).
        energy_threshold: Fraction of singular-value energy to retain per
            row when ``rank`` is None.

    Returns:
        ``(L, R, ranks)``: ``L`` is ``[BH, N, max_rank]`` fp32 (already
        folded with singular values, i.e. ``U_r * s_r`` — matching
        :func:`lowrank_error`'s convention, NOT :func:`_truncated_svd`'s raw
        ``U_r``); ``R`` is ``[BH, max_rank, D]`` fp32 (``Vt_r``); ``ranks``
        is a plain Python ``list[int]`` of each row's own rank (``<=
        max_rank``), for the caller's per-row truncation.
    """
    x32 = x.astype(mx.float32)
    bh, n, d = x32.shape
    U, s_vals, Vt = mx.linalg.svd(x32, stream=mx.cpu)  # [BH,N,N], [BH,min(N,D)], [BH,min(N,D),D]
    mx.eval(U, s_vals, Vt)

    r_max_possible = s_vals.shape[-1]
    if rank is not None:
        r = max(1, min(int(rank), int(r_max_possible), d))
        ranks = [r] * bh
        max_rank = r
    else:
        total = mx.sum(s_vals, axis=-1, keepdims=True)  # [BH, 1]
        is_near_zero = (total < 1e-12).reshape(bh)  # [BH] — matches the
        # scalar version's `if total < 1e-12: rank = 1` special case exactly
        # (a near-zero residual, e.g. an exactly-zero matrix): that branch
        # short-circuits BEFORE the cumulative-energy loop in the scalar
        # version, so it must be handled as its own case here too, not left
        # to fall through the (division-safe, but semantically different)
        # `frac = cumsum / maximum(total, eps)` path below, which would
        # otherwise select the FULL rank for a zero matrix instead of the
        # scalar version's rank=1 (verified this divergence directly before
        # adding this branch — an earlier draft without it silently gave
        # rank=r_max_possible for an all-zero residual instead of rank=1).
        cumsum = mx.cumsum(s_vals, axis=-1)  # [BH, R]
        frac = cumsum / mx.maximum(total, 1e-12)
        meets = frac >= energy_threshold  # [BH, R] bool
        # argmax of a 0/1 array returns the FIRST True index — the smallest
        # rank whose cumulative energy clears the threshold, matching the
        # scalar version's `break` on the first qualifying prefix exactly.
        first_true = mx.argmax(meets.astype(mx.int32), axis=-1)  # [BH]
        any_true = mx.any(meets, axis=-1)
        raw_rank = mx.where(
            any_true, first_true + 1, mx.full((bh,), r_max_possible, dtype=mx.int32)
        )
        raw_rank = mx.where(is_near_zero, mx.ones_like(raw_rank), raw_rank)
        mx.eval(raw_rank)
        raw_rank_list: list[int] = raw_rank.tolist()  # type: ignore[assignment]
        ranks = [max(1, min(r, int(r_max_possible), d)) for r in raw_rank_list]
        max_rank = max(ranks)

    U_trunc = U[:, :, :max_rank]  # [BH, N, max_rank]
    s_trunc = s_vals[:, :max_rank]  # [BH, max_rank]
    Vt_trunc = Vt[:, :max_rank, :]  # [BH, max_rank, D]

    # Zero out each row's singular values beyond its own rank so the padded
    # columns are numerically inert for reconstruction (L @ R contributes 0
    # from those columns) — callers must still slice to `ranks[i]` before
    # storage/byte-accounting (see docstring); this masking only protects
    # correctness of any direct use of the padded-width L/R.
    ranks_arr = mx.array(ranks, dtype=mx.int32)
    col_idx = mx.arange(max_rank)[None, :]  # [1, max_rank]
    rank_mask = col_idx < ranks_arr[:, None]  # [BH, max_rank]
    s_masked = mx.where(rank_mask, s_trunc, mx.zeros_like(s_trunc))

    L = U_trunc * s_masked[:, None, :]  # [BH, N, max_rank], singular values folded in
    R = Vt_trunc  # [BH, max_rank, D]
    return L, R, ranks


__all__ = [
    "_group_quant_codes",
    "_group_quant_codes_batched",
    "_group_dequant_codes",
    "_group_dequant_codes_batched",
    "_group_quant_dequant",
    "_group_quant_dequant_batched",
    "_truncated_svd",
    "_truncated_svd_batched",
]
