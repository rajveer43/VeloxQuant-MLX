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

import mlx.core as mx

from veloxquant_mlx.quantizers._quant_utils import (
    _group_quant_dequant,
    _group_quant_dequant_batched,
)

#: Paper's worked example schedule (Section 4.2): 8 equal-size groups over the
#: latent channels, ordered from largest to smallest singular value, with the
#: last three groups truncated to 0 bits. Mean bit-width b̄ = 2.
DEFAULT_BIT_SCHEDULE: tuple[int, ...] = (8, 4, 2, 1, 1, 0, 0, 0)


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
    x = keys.astype(mx.float32)
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
    if r <= 0:
        return []
    n_groups = min(n_groups, r)
    base = r // n_groups
    remainder = r % n_groups
    slices = []
    start = 0
    for i in range(n_groups):
        size = base + (1 if i < remainder else 0)
        slices.append((start, start + size))
        start += size
    return slices


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

    Returns:
        ``[H, S, D]`` fp16 reconstructed keys.
    """
    H, S, D = keys.shape
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
    n_groups = min(len(bit_schedule), max(r, 1))
    return sum(bit_schedule[:n_groups]) / n_groups


__all__ = [
    "svd_compress_keys",
    "quantize_latents_mixed",
    "reconstruct_keys",
    "project_quantize_reconstruct_batched",
    "latent_group_slices",
    "equivalent_bit_width",
    "min_safe_rank",
    "MIN_SAFE_CHANNELS_PER_GROUP",
    "DEFAULT_BIT_SCHEDULE",
    "_group_quant_dequant",
]
