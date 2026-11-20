"""SVDq quantizer — sub-2-bit key compression via offline SVD + mixed-precision latent coding.

Inspired by "SVDq: Singular Value Decomposition-based KV Cache Quantization"
(arXiv:2502.15304, Feb 2025, unreviewed preprint).

Algorithm (paper Section 4.2 / Algorithm 1):
  Prefill phase (triggered once when a batch of keys arrives):
    1. Compute truncated SVD of the key matrix K ∈ R^{S×D}: K ≈ U·Σ_r·V^H + K̄
       where K̄ = mean(K, axis=0) is subtracted before decomposition.
    2. Store V (right singular vectors, shape [D, r]) and K̄ (mean key, [D])
       as layer attributes.
    3. Project keys into latent space: L = (K - K̄) @ V  →  shape [S, r]
    4. Split the r latent channels into 8 equal-size groups (ordered by
       descending singular value, i.e. group 0 = largest-λ channels) and
       apply a fixed per-group bit-width schedule `b = (b1, ..., b8)`. A
       group can be assigned 0 bits, which truncates it entirely (paper
       Eq. 6 and the worked example: `(8,4,2,1,1,0,0,0)` → b̄ = 2).

  Decode phase (per new token):
    1. Project new key: l = (k - K̄) @ V  →  shape [1, r]
    2. Quantize l with the same 8-group bit schedule.
    3. On fetch, reconstruct: K_hat = L_dequant @ V^H + K̄

This module implements the per-token latent quantizer.  The cache wrapper
(SVDqKVCache) owns the SVD state and orchestrates prefill vs decode.

Adaptation notes (what is still NOT faithful — read before trusting this as
a reproduction of the paper's numbers):
  - Rank defaults to energy_threshold=0.95 (retain ≥95% singular value energy)
    rather than a fixed d//4 — this is more robust across models, but is not
    a rule the paper specifies at all (invented here).
  - SVD is fit from the live prefill batch of a single sequence, not from an
    offline calibration set as the paper does (Section 4.2, "Load K cache
    matrix for l-th layer" — the paper's V is computed once per model from
    calibration data, not per-sequence).
  - The 8-group split now matches the paper's *shape* (Eq. 6), but the
    default schedule below is still a hand-chosen heuristic, not one fit to
    any calibration or benchmark data the way the paper's schedules were
    (Table 2/3's schedules were chosen to hit specific target bit-widths
    after empirical tuning on RULER/LongBench).
  - Values are left at fp16 (the paper notes values have weak low-rank
    structure, and explicitly does not investigate V-cache compression).
  - Documented as "SVDq-adapted" — not a faithful port; no model-level
    accuracy numbers have been reproduced (synthetic-data tests only).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import mlx.core as mx
import numpy as np

from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers._quant_utils import (
    _group_quant_dequant,
    _group_quant_dequant_batched,
)
from veloxquant_mlx.quantizers._validation import (
    check_finite_gram,
    validate_bit_schedule,
    validate_energy_threshold,
    validate_int,
    validate_rank,
)

#: Paper's worked example schedule (Section 4.2): 8 equal-size groups over the
#: latent channels, ordered from largest to smallest singular value, with the
#: last three groups truncated to 0 bits. Mean bit-width b̄ = 2.
DEFAULT_BIT_SCHEDULE: tuple[int, ...] = (8, 4, 2, 1, 1, 0, 0, 0)

_WHO = "svdq"


def validate_svdq_config(config: Any) -> dict[str, Any]:
    """Validate and coerce the ``svdq_*`` fields of a :class:`KVCacheConfig`.

    Returns the coerced values keyed by field suffix (``rank``, ``energy``,
    ``bit_schedule``, ``group_size``) using the same defaults as
    :class:`~veloxquant_mlx.cache.svdq_cache.SVDqKVCache`. Whole-number floats
    (``8.0``) are accepted and coerced to int; anything else invalid raises
    :class:`~veloxquant_mlx.core.exceptions.QuantizerConfigError` (a ``ValueError``).
    A bit schedule that truncates every group is rejected here, since the cache would
    then store nothing and return only the mean key.
    """

    def get(name: str, default: Any) -> Any:
        return getattr(config, f"svdq_{name}", default)

    return {
        "rank": validate_rank(get("rank", None), "svdq_rank", who=_WHO),
        "energy": validate_energy_threshold(
            get("energy_threshold", 0.95), "svdq_energy_threshold", who=_WHO
        ),
        "bit_schedule": validate_bit_schedule(
            get("bit_schedule", DEFAULT_BIT_SCHEDULE),
            "svdq_bit_schedule",
            who=_WHO,
            allow_all_zero=False,
        ),
        "group_size": validate_int(get("group_size", 32), "svdq_group_size", who=_WHO, lo=1),
    }


def _schedule_uncached(schedule: object) -> tuple[int, ...]:
    return validate_bit_schedule(schedule, "bit_schedule", who=_WHO)


@lru_cache(maxsize=256)
def _schedule_cached(schedule: tuple[int, ...], types: tuple[type, ...]) -> tuple[int, ...]:
    # ``types`` is part of the key because ``(True, 4)`` hashes and compares equal to
    # ``(1, 4)``: without it a cached valid schedule would also admit the bool one.
    return _schedule_uncached(schedule)


def _checked_schedule(schedule: object) -> tuple[int, ...]:
    """:func:`validate_bit_schedule` with a per-distinct-schedule cache (decode calls repeat it)."""
    try:
        sched: tuple[Any, ...] = tuple(schedule)  # type: ignore[arg-type]
        return _schedule_cached(sched, tuple(map(type, sched)))
    except TypeError:  # not iterable, or an unhashable entry: the validator rejects it
        return _schedule_uncached(schedule)


def _validate_keys(keys: mx.array, ndim: int, name: str = "keys") -> None:
    if keys.ndim != ndim:
        shape = "[S, D]" if ndim == 2 else "[H, S, D]"
        raise QuantizerConfigError(
            f"svdq: {name} must be a {ndim}-D {shape} array, got shape {tuple(keys.shape)}."
        )
    if 0 in keys.shape:
        raise QuantizerConfigError(
            f"svdq: {name} must be non-empty, got shape {tuple(keys.shape)}."
        )


def svd_compress_keys(
    keys: mx.array,
    rank: int | None = None,
    energy_threshold: float = 0.95,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute truncated SVD of a key matrix and return latents + projection.

    Args:
        keys: Shape [S, D], fp16 or fp32.
        rank: Explicit rank r. If None, determined by energy_threshold.
        energy_threshold: Fraction of total singular value energy to retain.

    Returns:
        (L, V, K_mean, singular_values) where:
          L          — latent codes [S, r] fp32
          V          — right singular vectors [D, r] fp32
          K_mean     — mean key [D] fp32
          singular_values — [r] fp32, descending
    """
    _validate_keys(keys, 2)
    rank = validate_rank(rank, "rank", who=_WHO)
    energy_threshold = validate_energy_threshold(energy_threshold, "energy_threshold", who=_WHO)
    x = keys.astype(mx.float32)
    # The SVD is a one-off, far costlier than this pass, and on NaN it only fails with
    # "SVD failed with error code -4".
    if not bool(mx.all(mx.isfinite(x)).item()):
        raise QuantizerConfigError("svdq: keys must be finite (found NaN or Inf).")
    K_mean = mx.mean(x, axis=0)  # [D]
    x_centered = x - K_mean[None, :]  # [S, D]

    # MLX svd returns (U, S, Vt) with Vt shape [D, D] (economy=False)
    # or [min(S,D), D] — use economy form via mx.linalg.svd
    U, S_vals, Vt = mx.linalg.svd(x_centered, stream=mx.cpu)
    # Vt: [min(S,D), D] → V = Vt.T: [D, min(S,D)]
    mx.eval(U, S_vals, Vt)

    if rank is None:
        total_energy = float(mx.sum(S_vals).item())
        if total_energy < 1e-12:
            rank = 1
        else:
            cumsum = 0.0
            rank = len(S_vals)
            for i, sv in enumerate(S_vals.tolist()):
                cumsum += sv
                if cumsum / total_energy >= energy_threshold:
                    rank = i + 1
                    break
    rank = min(rank, int(S_vals.shape[0]), keys.shape[-1])

    V = Vt[:rank, :].T  # [D, r]
    s_r = S_vals[:rank]  # [r]
    L = x_centered @ V  # [S, r]
    return L, V, K_mean, s_r


def _select_ranks(
    S_vals: mx.array,  # [H, R] descending singular values
    rank: int | None,
    energy_threshold: float,
    r_max_possible: int,
    D: int,
) -> tuple[list[int], int]:
    """Per-head rank: the explicit ``rank`` or the smallest rank reaching ``energy_threshold``."""
    H = S_vals.shape[0]
    if rank is not None:
        r = min(rank, r_max_possible, D)
        ranks = [r] * H
        max_rank = r
    else:
        total = mx.sum(S_vals, axis=-1, keepdims=True)  # [H, 1]
        is_near_zero = (total < 1e-12).reshape(H)  # matches the scalar
        # version's `if total_energy < 1e-12: rank = 1` short-circuit,
        # evaluated BEFORE the cumulative-energy loop there — same special
        # case as GEAR's _truncated_svd_batched (see its docstring for why
        # this must be its own branch rather than falling through the
        # division-safe cumsum/max(total,eps) path below).
        cumsum = mx.cumsum(S_vals, axis=-1)  # [H, R]
        frac = cumsum / mx.maximum(total, 1e-12)
        meets = frac >= energy_threshold  # [H, R] bool
        first_true = mx.argmax(meets.astype(mx.int32), axis=-1)  # [H]
        any_true = mx.any(meets, axis=-1)
        raw_rank = mx.where(any_true, first_true + 1, mx.full((H,), r_max_possible, dtype=mx.int32))
        raw_rank = mx.where(is_near_zero, mx.ones_like(raw_rank), raw_rank)
        mx.eval(raw_rank)
        raw_rank_list: list[int] = raw_rank.tolist()  # type: ignore[assignment]
        ranks = [min(r, int(r_max_possible), D) for r in raw_rank_list]
        max_rank = max(ranks) if ranks else 0
    return ranks, max_rank


def basis_from_gram_batched(
    gram: np.ndarray,  # [H, D, D] float64, sum over rows of x x^T (uncentered)
    row_sum: np.ndarray,  # [H, D] float64, sum over rows of x
    n_rows: int,
    rank: int | None = None,
    energy_threshold: float = 0.95,
) -> tuple[list[mx.array], list[mx.array], list[mx.array]]:
    """The basis :func:`svd_compress_keys_batched` would fit on ``n_rows`` rows,
    from their accumulated uncentered Gram matrix and row sum.

    Lets a cache refit on every prefill chunk without keeping the raw keys
    (#890). Returns ``(V_list, K_mean_list, s_list)`` with the same per-head
    rank truncation as :func:`svd_compress_keys_batched`.
    """
    rank = validate_rank(rank, "rank", who=_WHO)
    energy_threshold = validate_energy_threshold(energy_threshold, "energy_threshold", who=_WHO)
    n_rows = validate_int(n_rows, "n_rows", who=_WHO, lo=1)
    check_finite_gram(gram, "the accumulated key Gram matrix", who=_WHO)
    H, D, _ = gram.shape
    mean = row_sum / n_rows  # [H, D]
    centered = gram - n_rows * mean[:, :, None] * mean[:, None, :]
    evals, evecs = np.linalg.eigh(0.5 * (centered + np.swapaxes(centered, 1, 2)))  # ascending
    r_max_possible = min(n_rows, D)
    S_vals = mx.array(
        np.sqrt(np.clip(evals[:, ::-1][:, :r_max_possible], 0.0, None)).astype(np.float32)
    )
    v_desc = evecs[:, :, ::-1]
    ranks, max_rank = _select_ranks(S_vals, rank, energy_threshold, r_max_possible, D)
    V_padded = mx.array(v_desc[:, :, :max_rank].astype(np.float32))
    s_padded = S_vals[:, :max_rank]
    K_mean = mx.array(mean.astype(np.float32))
    return (
        [V_padded[h, :, : ranks[h]] for h in range(H)],
        [K_mean[h] for h in range(H)],
        [s_padded[h, : ranks[h]] for h in range(H)],
    )


def svd_compress_keys_batched(
    keys: mx.array,  # [H, S, D] fp16/fp32
    rank: int | None = None,
    energy_threshold: float = 0.95,
) -> tuple[list[mx.array], list[mx.array], list[mx.array], list[mx.array]]:
    """Batched-over-heads equivalent of calling :func:`svd_compress_keys` once
    per head, replacing ``SVDqKVCache._run_prefill_svd``'s ``for h in
    range(H):`` loop (VeloxQuant-MLX#569).

    ``mx.linalg.svd`` natively supports a batched leading axis (verified
    directly: singular values/vectors for each row of a batched ``[H,S,D]``
    call are bit-identical to calling it once per row), so the SVD itself is
    one dispatch instead of ``H`` separate CPU-stream calls — the same
    technique GEAR's ``_truncated_svd_batched`` (#504) already uses.

    Each head's SVD can legitimately land at a different rank under
    energy-threshold auto-rank (different heads' key distributions have
    different effective dimensionality) — since a single array can't hold a
    per-row-variable width, the SVD factors are computed at the batch's max
    rank and each head's per-head outputs are truncated back to its own
    rank before being returned, so the return contract exactly matches
    calling :func:`svd_compress_keys` per head: no padding leaks out of
    this function.

    Args:
        keys: ``[H, S, D]`` fp16 or fp32 — one head axis.
        rank: Explicit rank shared by every head. If None, chosen per-head
            by ``energy_threshold`` (vectorized cumulative-energy selection,
            not a per-row Python loop; verified bit-identical to looping
            :func:`svd_compress_keys`'s scalar rank selection).
        energy_threshold: Fraction of singular-value energy to retain per
            head when ``rank`` is None.

    Returns:
        Four length-``H`` lists ``(L_list, V_list, K_mean_list, s_list)``,
        matching :func:`svd_compress_keys`'s per-head return values exactly
        (each already truncated to that head's own rank — no shared padding
        in the returned lists).
    """
    _validate_keys(keys, 3)
    rank = validate_rank(rank, "rank", who=_WHO)
    energy_threshold = validate_energy_threshold(energy_threshold, "energy_threshold", who=_WHO)
    x = keys.astype(mx.float32)
    H, S, D = x.shape
    K_mean = mx.mean(x, axis=1)  # [H, D]
    x_centered = x - K_mean[:, None, :]  # [H, S, D]

    # Only V and the singular values are used, so eigendecompose the batched
    # D x D Gram matrix instead of SVD-ing the tall block, which also builds
    # an [S, S] U and made prefill superlinear in prompt length (#875). Formed
    # and diagonalised in float64: squaring the conditioning is harmless
    # there, whereas a float32 Gram blurs the weaker retained directions.
    xc_np = np.asarray(x_centered, dtype=np.float64)
    gram = np.swapaxes(xc_np, 1, 2) @ xc_np
    check_finite_gram(gram, "keys", who=_WHO)
    evals, evecs = np.linalg.eigh(gram)  # ascending
    r_max_possible = min(S, D)  # an SVD of [S, D] has min(S, D) singular values
    S_vals = mx.array(
        np.sqrt(np.clip(evals[:, ::-1][:, :r_max_possible], 0.0, None)).astype(np.float32)
    )  # [H, min(S,D)] descending
    v_desc = evecs[:, :, ::-1]  # [H, D, D], columns by descending singular value

    ranks, max_rank = _select_ranks(S_vals, rank, energy_threshold, r_max_possible, D)

    V_padded = mx.array(v_desc[:, :, :max_rank].astype(np.float32))  # [H, D, max_rank]
    s_padded = S_vals[:, :max_rank]  # [H, max_rank]
    L_padded = x_centered @ V_padded  # [H, S, max_rank]

    L_list = [L_padded[h, :, : ranks[h]] for h in range(H)]
    V_list = [V_padded[h, :, : ranks[h]] for h in range(H)]
    K_mean_list = [K_mean[h] for h in range(H)]
    s_list = [s_padded[h, : ranks[h]] for h in range(H)]
    return L_list, V_list, K_mean_list, s_list


def _compute_slices(r: int, n_groups: int) -> tuple[tuple[int, int], ...]:
    n_groups = validate_int(n_groups, "n_groups", who=_WHO, lo=1)
    if r <= 0:
        return ()
    n_groups = min(n_groups, r)
    base = r // n_groups
    remainder = r % n_groups
    slices = []
    start = 0
    for i in range(n_groups):
        size = base + (1 if i < remainder else 0)
        slices.append((start, start + size))
        start += size
    return tuple(slices)


# The cache asks for the same (rank, n_groups) once per head per layer on every decode
# step, so compute (and validate) each distinct pair once. typed=True: otherwise a
# cached valid ``1`` would also answer for ``True``.
_slices_cached = lru_cache(maxsize=1024, typed=True)(_compute_slices)


def latent_group_slices(r: int, n_groups: int = 8) -> list[tuple[int, int]]:
    """Split ``r`` latent channels (descending singular-value order) into
    ``n_groups`` contiguous, near-equal-size slices, matching the paper's
    Section 4.2 grouping ("we divide the d latent channels of P_V(K) into 8
    equal-sized groups"). The last group absorbs any remainder so every
    channel is covered even when ``r`` doesn't divide evenly by ``n_groups``.

    Args:
        r: Number of latent channels (already rank-truncated).
        n_groups: Number of equal-size groups (paper uses 8).

    Returns:
        List of ``(start, end)`` half-open index ranges, length
        ``min(n_groups, r)`` — fewer groups than requested when ``r`` is
        smaller than ``n_groups`` (each channel gets its own group).
    """
    try:
        return list(_slices_cached(r, n_groups))
    except TypeError:  # unhashable argument: the uncached path rejects or reports it
        return list(_compute_slices(r, n_groups))


#: Minimum channels a group must have before the schedule's 0-bit groups are
#: considered safe. Below this, a 0-bit group truncates 1-2 individual
#: channels wholesale rather than a tail of genuinely negligible energy —
#: see test_small_rank_near_group_count_can_underperform_naive, which shows
#: SVDq losing to naive 2-bit quantization once groups shrink to size 1.
MIN_SAFE_CHANNELS_PER_GROUP = 4


def min_safe_rank(n_groups: int = len(DEFAULT_BIT_SCHEDULE)) -> int:
    """Smallest rank at which every one of ``n_groups`` groups has at least
    :data:`MIN_SAFE_CHANNELS_PER_GROUP` channels.

    Below this rank, ``latent_group_slices`` still produces ``n_groups``
    groups (or fewer), but individual groups can shrink to 1-2 channels —
    at which point a 0-bit group in the schedule truncates specific
    channels outright rather than a genuinely negligible energy tail, and
    SVDq's accuracy advantage over naive quantization is not guaranteed
    (see the paper's own decay-rate assumption in Section 4.3, which this
    only holds when there are enough channels per group for the exponential
    decay model to apply).
    """
    return n_groups * MIN_SAFE_CHANNELS_PER_GROUP


def quantize_latents_mixed(
    L: mx.array,
    singular_values: mx.array,
    bit_schedule: Sequence[int] = DEFAULT_BIT_SCHEDULE,
    group_size: int = 32,
) -> mx.array:
    """Mixed-precision quantization of latent codes via an 8-group bit schedule.

    Matches the paper's Algorithm 1 / Eq. 6: the ``r`` latent channels
    (already ordered by descending singular value) are split into
    ``len(bit_schedule)`` equal-size contiguous groups, and each group is
    quantized at its own fixed bit width. A group with 0 bits is truncated
    entirely (its reconstructed value is exactly 0, matching the paper's
    example schedule ``(8,4,2,1,1,0,0,0)``).

    Args:
        L: Latent codes [S, r] fp32, columns already in descending
           singular-value order.
        singular_values: [r] fp32, descending. Unused for channel selection
            now that grouping is positional (paper Eq. 6), kept for API
            stability and to allow callers to assert ordering.
        bit_schedule: Per-group bit widths, most-significant group first.
            Defaults to the paper's worked example ``(8,4,2,1,1,0,0,0)``.
        group_size: Group size for quantization along the token axis
            (independent of the *latent-channel* grouping above).

    Returns:
        Reconstructed latents [S, r] fp16.
    """
    del singular_values  # grouping is positional; L's columns are pre-sorted
    bit_schedule = _checked_schedule(bit_schedule)
    group_size = validate_int(group_size, "group_size", who=_WHO, lo=1)
    S, r = L.shape
    slices = latent_group_slices(r, n_groups=len(bit_schedule))

    parts: list[mx.array] = []
    for group_idx, (start, end) in enumerate(slices):
        bits = bit_schedule[group_idx]
        L_group = L[:, start:end]
        if bits <= 0:
            parts.append(mx.zeros_like(L_group).astype(mx.float16))
        else:
            parts.append(_group_quant_dequant(L_group, bits, group_size))
    if not parts:
        return mx.zeros((S, 0), dtype=mx.float16)
    return mx.concatenate(parts, axis=1)


def project_quantize_reconstruct_batched(
    keys: mx.array,  # [H, S, D]
    V_list: list[mx.array],  # each [D, r_h] fp32
    K_mean_list: list[mx.array],  # each [D] fp32
    schedules: list[Sequence[int]],  # per-head effective bit schedule
    group_size: int,
    *,
    validate: bool = True,
) -> mx.array:
    """Batched-over-heads equivalent of looping project -> quantize -> reconstruct
    once per head, each through its own already-fitted SVD basis.

    Replaces ``SVDqKVCache._project_quantize_reconstruct``'s ``for h in
    range(H):`` loop — the decode-hot-path cost this function exists to
    remove (see VeloxQuant-MLX#562; distinct from the SVD-*fit* loop, tracked
    separately in #569). Each head's basis (``V_list[h]``, ``K_mean_list[h]``)
    was already fit (by :func:`svd_compress_keys`, still called once per head
    at prefill — out of scope here) and generally has its own rank ``r_h``
    and, after the small-rank safe-schedule guard, possibly its own bit
    schedule; this function only batches the per-token *application* of
    those already-fitted, ragged-rank bases.

    Heads are grouped by their exact ``(r_h, schedule_h)`` pair — in
    practice very few distinct groups (typically 1, when every head shares
    one explicit ``svdq_rank``/schedule; a handful under energy-threshold
    auto-rank) — and each group's projection, quantization, and
    reconstruction run as one batched call instead of ``H`` separate ones.
    Within a group, the projection itself is further batched across ALL
    heads at once (not just the group) by zero-padding every head's ``V``/
    latent to the batch's max rank — :func:`quantize_latents_mixed`'s
    channel-group boundaries depend on the exact rank, though, so
    quantization is still done per ``(r_h, schedule_h)`` group. Numerically
    identical to the per-head loop: a padded head's extra latent columns
    are exact zeros (unused rows of a zero-padded ``V``project to 0,
    independent of any other head's data), so they never influence another
    head's projection, quantization, or reconstruction.

    Args:
        keys: ``[H, S, D]`` fp16 or fp32 — this call's keys, one head axis.
        V_list: Per-head projection basis, length ``H``, each ``[D, r_h]``.
        K_mean_list: Per-head mean key, length ``H``, each ``[D]``.
        schedules: Per-head effective bit schedule, length ``H``.
        group_size: Token-axis group size for the latent quantizer.
        validate: Check ``group_size``, the schedules and the per-head list lengths.
            :class:`~veloxquant_mlx.cache.svdq_cache.SVDqKVCache` passes ``False`` on its
            decode path: it validated them when it was built, and re-checking on every
            token of every layer is pure overhead.

    Returns:
        ``[H, S, D]`` fp16 reconstructed keys.
    """
    H, S, D = keys.shape
    if validate:
        if not len(V_list) == len(K_mean_list) == len(schedules) == H:
            raise QuantizerConfigError(
                f"svdq: V_list, K_mean_list and schedules must each have one entry per head "
                f"({H}), got {len(V_list)}, {len(K_mean_list)} and {len(schedules)}."
            )
        group_size = validate_int(group_size, "group_size", who=_WHO, lo=1)
        schedules = [_checked_schedule(s) for s in schedules]
    ranks = [int(V.shape[1]) for V in V_list]
    r_max = max(ranks) if ranks else 0

    keys_f = keys.astype(mx.float32)

    if r_max == 0:
        return mx.zeros((H, S, D), dtype=mx.float16)

    # Zero-pad every head's V/K_mean to [H, D, r_max] / [H, D] so the
    # projection is one batched matmul across all H heads regardless of
    # each head's own (possibly smaller) rank — padded V columns are exact
    # zeros, so they contribute nothing to the padded latent columns.
    V_padded = mx.zeros((H, D, r_max), dtype=mx.float32)
    K_mean_stacked = mx.stack(K_mean_list, axis=0)  # [H, D]
    for h in range(H):
        r_h = ranks[h]
        if r_h > 0:
            V_padded[h, :, :r_h] = V_list[h]

    k_centered = keys_f - K_mean_stacked[:, None, :]  # [H, S, D]
    L = k_centered @ V_padded  # [H, S, r_max]

    # Quantize per distinct (rank, schedule) group — channel-group
    # boundaries (latent_group_slices) depend on the exact rank, so a
    # padded-to-r_max quantize call would put the wrong channels in each
    # bit-width tier for any head whose true rank differs from r_max.
    groups: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for h in range(H):
        key = (ranks[h], tuple(schedules[h]))
        groups.setdefault(key, []).append(h)

    L_q = mx.zeros((H, S, r_max), dtype=mx.float16)
    for (r_h, schedule_h), head_idxs in groups.items():
        if r_h == 0:
            continue
        idx = mx.array(head_idxs)
        L_group = mx.take(L, idx, axis=0)[:, :, :r_h]  # [n, S, r_h]
        slices = latent_group_slices(r_h, n_groups=len(schedule_h))
        parts: list[mx.array] = []
        for group_idx, (start, end) in enumerate(slices):
            bits = schedule_h[group_idx]
            chunk = L_group[:, :, start:end]
            if bits <= 0:
                parts.append(mx.zeros_like(chunk).astype(mx.float16))
            else:
                parts.append(_group_quant_dequant_batched(chunk, bits, group_size))
        recon = (
            mx.concatenate(parts, axis=2)
            if parts
            else mx.zeros((len(head_idxs), S, 0), dtype=mx.float16)
        )
        if r_h < r_max:
            pad = mx.zeros((len(head_idxs), S, r_max - r_h), dtype=mx.float16)
            recon = mx.concatenate([recon, pad], axis=2)
        L_q[idx] = recon

    K_hat = L_q.astype(mx.float32) @ mx.swapaxes(V_padded, -1, -2) + K_mean_stacked[:, None, :]
    return K_hat.astype(mx.float16)


def reconstruct_keys(
    L_q: mx.array,
    V: mx.array,
    K_mean: mx.array,
) -> mx.array:
    """Reconstruct full key matrix from quantized latents.

    Args:
        L_q: Quantized latents [S, r] fp16.
        V: Right singular vectors [D, r] fp32.
        K_mean: Mean key [D] fp32.

    Returns:
        Reconstructed keys [S, D] fp16.
    """
    K_hat = L_q.astype(mx.float32) @ V.T + K_mean[None, :]
    return K_hat.astype(mx.float16)


def equivalent_bit_width(r: int, bit_schedule: Sequence[int] = DEFAULT_BIT_SCHEDULE) -> float:
    """Mean bit-width b̄ of an 8-group schedule over ``r`` latent channels (paper Eq. 6).

    ``b̄ = (1/n_groups) * sum(bi)`` — the paper's own definition, independent
    of ``r`` since every group is weighted equally regardless of size. This
    only approximates the *per-original-key-element* cost; the cache wrapper
    additionally scales by ``r/D`` since latent dim is smaller than ``D``.
    """
    bit_schedule = _checked_schedule(bit_schedule)
    n_groups = min(len(bit_schedule), max(r, 1))
    return sum(bit_schedule[:n_groups]) / n_groups


__all__ = [
    "svd_compress_keys",
    "svd_compress_keys_batched",
    "basis_from_gram_batched",
    "quantize_latents_mixed",
    "reconstruct_keys",
    "project_quantize_reconstruct_batched",
    "latent_group_slices",
    "equivalent_bit_width",
    "validate_svdq_config",
    "min_safe_rank",
    "MIN_SAFE_CHANNELS_PER_GROUP",
    "DEFAULT_BIT_SCHEDULE",
    "_group_quant_dequant",
]
