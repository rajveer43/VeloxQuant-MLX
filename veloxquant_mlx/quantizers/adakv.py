"""AdaKV-proxy quantizer — per-head adaptive bit allocation over KIVI.

Inspired by "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget
Allocation for Efficient LLM Inference" (arXiv:2407.11550, 2024). Documented
as "AdaKV-proxy (VeloxQuant-MLX implementation)" — a *proxy* adaptation, not a
faithful port: true Ada-KV adapts the per-head *eviction* budget using softmax
attention weights, which live outside the ``update_and_fetch`` contract. We
instead adapt the per-head *bit* budget using an attention-free proxy for head
importance.

Algorithm:
    For a key tensor K ∈ R^{B×H×S×D}:
      1. Per head h, compute a head-importance score (see "Importance signal"
         below for the two available signals and what each one measures).
      2. Rank-normalise the importances to a zero-mean spread, place each head
         at ``target_avg_bits + spread * z[h]``, then snap to the nearest value
         in the allowed set (e.g. {2, 3, 4}).
      3. A greedy round-trip correction nudges heads up/down — picking the head
         closest to a rounding boundary each time, with permutation-invariant
         tie-breaking — until the total bit budget matches
         ``n_heads × target_avg_bits`` as closely as the allowed set permits.
      4. Each head's keys are quantized at its assigned bit-width with KIVI-style
         asymmetric min/max group quantization.

Importance signal (two available, and what each actually measures):
    ``compute_head_norm_variance`` — inter-token key-norm variance,
    ``Var_t(‖k_t‖₂)``. This measures **quantization sensitivity**, not attention
    dispersion: a head whose key magnitudes vary widely has a wider dynamic
    range per min/max group, so a fixed bit-width incurs more reconstruction
    error, and extra bits buy more there. This is a sound criterion for
    allocating *bits*, but it is a **different** criterion from the paper's —
    and in fact anti-correlated with it. High norm-variance is the signature of
    a few outlier-norm tokens dominating the logits, i.e. an attention-*sparse*
    head, which the paper would *defund*. Stated plainly rather than papered
    over; see :func:`compute_head_norm_variance`.

    ``compute_head_attention_entropy`` — per-head attention entropy estimated
    from an observation window, reusing the keys-as-proxy-queries trick from
    :mod:`veloxquant_mlx.quantizers.snapkv`. This carries the **paper's sign**:
    dispersed (high-entropy) heads score higher and receive more budget, which
    is Ada-KV §3.3 / Fig. 1b. Costs one ``[w, S]`` attention matrix per head.

Budget-allocation caveat:
    Adaptation is only possible on the *open* interval
    ``(lo_bit, hi_bit)``. If ``target_avg_bits`` equals either endpoint, the
    per-head budget ``H × target`` is satisfiable only by giving every head that
    same value — raising one head must be paid for by lowering another past the
    bound. :func:`allocate_head_bits` warns once in that case rather than
    silently returning a uniform (KIVI-equivalent) allocation.

Effective bit-width:
    assigned_avg_bits = Σ_h head_bits[h] / H,  with  head_bits[h] ∈ allowed_bits
    and  Σ_h head_bits[h] ≈ H × target_avg_bits.

Adaptation notes:
    - True Ada-KV (head-adaptive eviction budget) needs softmax attention
      weights over the *real* query states — outside the update_and_fetch
      contract. Documented as the theoretical basis only.
    - Cross-layer budget sharing (a low-importance layer donating budget to a
      high-importance layer) is out of scope — it would break the single-wrapper
      contract.
    - Values are left at fp16 — AdaKV-proxy is a key-only method.
"""
from __future__ import annotations

import math
import warnings
from typing import Sequence

import mlx.core as mx

from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant


def compute_head_norm_variance(keys: mx.array) -> mx.array:
    """Inter-token key-norm variance per head — a *quantization-sensitivity* signal.

    For each head h, computes the variance over the sequence (token) axis of
    the per-token key L2 norms ‖k_t‖₂, then averages that variance over the
    batch dimension.

    .. warning::
       This is **not** a proxy for attention entropy, and it is **not** the
       criterion Ada-KV (arXiv:2407.11550) allocates on. The paper shifts budget
       toward attention-*dispersed* heads (§3.3, Fig. 1b). High key-norm
       variance indicates a few outlier-magnitude tokens dominating the
       ``q·k`` logits — the signature of an attention-*sparse* head, which the
       paper would give *less* budget. Empirically the two signals are
       anti-correlated.

       What this signal does measure, soundly, is **quantization
       sensitivity**: wider spread in ‖k_t‖₂ means a wider dynamic range inside
       each min/max quantization group, so a fixed bit-width incurs more
       reconstruction error and extra bits buy more improvement. That is a
       legitimate basis for allocating *bits* — it is simply a different
       objective from the paper's, not an approximation of it.

       For a signal carrying the paper's sign, use
       :func:`compute_head_attention_entropy`.

    Args:
        keys: Shape [B, H, S, D] fp16 or fp32.

    Returns:
        [H] fp32 array of per-head norm variances. Returns zeros if S < 2.
    """
    B, H, S, D = keys.shape
    k32 = keys.astype(mx.float32)
    norms = mx.sqrt(mx.sum(k32 * k32, axis=-1))          # [B, H, S]
    if S < 2:
        return mx.zeros((H,), dtype=mx.float32)
    var = mx.var(norms, axis=-1)                          # [B, H]
    return mx.mean(var, axis=0)                           # [H]


def compute_head_attention_entropy(
    keys: mx.array,
    obs_window: int = 32,
) -> mx.array:
    """Per-head attention entropy — the signal carrying Ada-KV's own sign.

    Estimates how *dispersed* each head's attention is, which is the criterion
    Ada-KV allocates on (§3.3, Fig. 1a-1b): sparse heads concentrate their
    weight on a few tokens and need little budget; dispersed heads spread it
    broadly and need more.

    Real query states are not visible at ``update_and_fetch`` time, so this
    reuses the keys-as-proxy-queries substitution already established by
    :func:`veloxquant_mlx.quantizers.snapkv.obs_window_attention_scores`: the
    last ``obs_window`` key rows stand in for queries. Key and query spaces are
    correlated (both projected from the same residual stream), making this a
    stronger proxy than key-norm statistics — but it remains an approximation,
    stated plainly.

    Entropy is normalised by ``ln(S)`` so scores lie in ``[0, 1]`` and are
    comparable across sequence lengths; higher means more dispersed, hence more
    budget under :func:`allocate_head_bits`.

    Args:
        keys: Shape [B, H, S, D] fp16 or fp32.
        obs_window: Trailing key rows used as proxy queries. Clamped to ``S``.

    Returns:
        [H] fp32 normalised entropies in ``[0, 1]``. Zeros if ``S < 2``.
    """
    B, H, S, D = keys.shape
    if S < 2:
        return mx.zeros((H,), dtype=mx.float32)

    w = min(max(int(obs_window), 1), S)
    k32 = keys.astype(mx.float32)
    q_proxy = k32[:, :, -w:, :]                              # [B, H, w, D]
    logits = (q_proxy @ mx.swapaxes(k32, -1, -2)) / math.sqrt(D)   # [B, H, w, S]
    attn = mx.softmax(logits, axis=-1)                       # rows sum to 1
    avg = mx.mean(attn, axis=2)                              # [B, H, S]
    ent = -mx.sum(avg * mx.log(avg + 1e-12), axis=-1)        # [B, H]
    ent = ent / float(math.log(S))                           # normalise to [0, 1]
    return mx.mean(ent, axis=0).astype(mx.float32)           # [H]


def _rank_normalize(values: Sequence[float]) -> list[float]:
    """Map values to evenly-spaced ranks in ``[-0.5, 0.5]``, ties averaged.

    Order-preserving and bounded, so downstream scaling cannot be saturated by
    a single extreme importance value (the clamp-before-normalize failure this
    replaces). Ties receive the same rank, which is what makes the resulting
    allocation invariant to head permutation.
    """
    n = len(values)
    if n == 1:
        return [0.0]

    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0          # average rank across the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    # Evenly space onto a unit interval, then centre on the *mean* rank rather
    # than the midpoint. Centring on the midpoint would push a tie group off
    # `target` whenever the ranks are asymmetric (e.g. one outlier against
    # several tied heads), spending budget the tied heads did not earn.
    scaled = [r / (n - 1) for r in ranks]
    mean = sum(scaled) / n
    return [s - mean for s in scaled]


def allocate_head_bits(
    head_importance: Sequence[float] | mx.array,
    target_avg_bits: float,
    allowed_bits: Sequence[int],
    n_heads: int,
    warn_degenerate: bool = True,
) -> list[int]:
    """Allocate per-head bit-widths under a global average-bits budget.

    Steps:
        1. Rank-normalise importances to a bounded zero-mean spread
           ``z[h] ∈ [-0.5, 0.5]`` (ties averaged).
        2. Place each head at ``target + spread * z[h]``, where ``spread``
           is the widest range that keeps every head inside
           ``[lo, hi]`` — so the mean lands on ``target`` by construction and
           the bounds are respected without a saturating clamp.
        3. Snap each head to the nearest value in ``allowed_bits``.
        4. Greedy correction: while the integer total over/undershoots
           ``n_heads × target``, bump the head whose real budget is closest to
           the boundary being crossed, breaking ties on importance, until the
           total can no longer move closer to target.

    Permutation behaviour: for heads with *distinct* importances the result is
    invariant to head ordering. When several heads are exactly tied **and** the
    budget is not divisible among them (e.g. 4 tied heads sharing a 10-bit
    budget over ``{2,3,4}``), some tied head must receive fewer bits than its
    equals — an integer-allocation fact, not a policy choice. In that case the
    lowest-indexed tied head gives way. The outcome is deterministic but does
    depend on head ordering; it cannot be otherwise without leaving the budget
    unmet.

    Why rank-normalisation rather than scaling raw importance: raw key-norm
    variances span orders of magnitude, so proportional scaling followed by a
    clamp pins nearly every head to ``lo`` or ``hi`` and discards the interior
    ordering the snap and greedy passes exist to act on. Ranks are bounded and
    order-preserving, so a single extreme head cannot flatten the allocation.

    Args:
        head_importance: [H] importance scores. Only the *ordering* matters;
            magnitudes are discarded by rank-normalisation. All-equal (incl.
            all-zero) degrades to a uniform target allocation.
        target_avg_bits: Global average bits/element target. Adaptation is only
            possible when this lies strictly inside ``(lo, hi)``.
        allowed_bits: Sorted-or-unsorted set of permissible bit-widths.
        n_heads: Number of heads H (must match len(head_importance)).
        warn_degenerate: Emit a ``UserWarning`` when ``target_avg_bits`` sits at
            or outside an endpoint of ``allowed_bits``, which forces a uniform
            (KIVI-equivalent) allocation. Set ``False`` to silence when the
            uniform result is intentional.

    Returns:
        [H] list of assigned integer bit-widths, each in ``allowed_bits``.
    """
    allowed = sorted(set(int(b) for b in allowed_bits))
    lo, hi = allowed[0], allowed[-1]

    imp = [float(x) for x in (head_importance.tolist()
                              if isinstance(head_importance, mx.array)
                              else head_importance)]
    if len(imp) != n_heads:
        raise ValueError(
            f"allocate_head_bits: head_importance has {len(imp)} entries but "
            f"n_heads={n_heads}."
        )

    # Single head: no allocation freedom — snap target to the allowed set.
    if n_heads == 1:
        return [min(allowed, key=lambda b: abs(b - target_avg_bits))]

    target = float(target_avg_bits)

    # Adaptation needs headroom on both sides: raising one head above `target`
    # must be payable by lowering another, without either leaving [lo, hi].
    # At an endpoint no such pair exists and the allocation is forced uniform.
    if warn_degenerate and not (lo < target < hi):
        warnings.warn(
            f"allocate_head_bits: target_avg_bits={target} is at or outside the "
            f"bounds of allowed_bits={allowed} (lo={lo}, hi={hi}), so every head "
            f"is forced to the same bit-width and the allocation is uniform — "
            f"equivalent to plain KIVI at that bit-width, with no per-head "
            f"adaptation regardless of head_importance. Choose a target strictly "
            f"inside ({lo}, {hi}) to enable adaptation.",
            UserWarning,
            stacklevel=2,
        )

    budget_total = n_heads * target

    # Rank-normalise to a bounded zero-mean spread. Only ordering survives,
    # which is what prevents one extreme head from saturating the allocation.
    z = _rank_normalize(imp)

    # Widest spread keeping every head inside [lo, hi]. `z` is mean-centred but
    # not symmetric, so scale by its actual extent on each side independently
    # and take the binding constraint.
    z_max = max(z)
    z_min = min(z)
    headroom_up = (hi - target) / z_max if z_max > 1e-12 else float("inf")
    headroom_dn = (target - lo) / (-z_min) if z_min < -1e-12 else float("inf")
    spread = min(headroom_up, headroom_dn)
    if spread == float("inf"):          # all heads tied -> no spread to apply
        spread = 0.0

    real = [min(max(target + spread * zh, float(lo)), float(hi)) for zh in z]

    # Snap each head to the nearest allowed bit value.
    bits = [min(allowed, key=lambda b: abs(b - r)) for r in real]

    # --- Greedy round-trip correction toward the integer budget -------------
    target_total = budget_total

    def _neighbor(b: int, direction: int) -> int | None:
        """Next allowed bit above (direction=+1) or below (direction=-1) b."""
        i = allowed.index(b)
        j = i + direction
        if 0 <= j < len(allowed):
            return allowed[j]
        return None

    # Move heads one allowed-step at a time, each time choosing the head whose
    # real budget is closest to the boundary we are crossing (smallest added
    # snap error). Stop when no single step reduces |total - target|.
    max_iters = n_heads * len(allowed) * 4
    for _ in range(max_iters):
        cur_total = sum(bits)
        diff = target_total - cur_total
        if abs(diff) < 1e-9:
            break
        direction = 1 if diff > 0 else -1

        best_h = -1
        best_key = None
        for h in range(n_heads):
            nb = _neighbor(bits[h], direction)
            if nb is None:
                continue
            # Cost = how far this head's real budget already is from the new
            # snap; prefer bumping heads whose real value most supports it.
            # Ties are broken on the head's own importance rather than its
            # index, so permuting the head axis cannot change which head is
            # starved: when stepping up, the more important head wins; when
            # stepping down, the less important one gives way first.
            cost = abs(real[h] - nb)
            key = (cost, -direction * imp[h])
            if best_key is None or key < best_key:
                best_key = key
                best_h = h

        if best_h < 0:
            break  # nothing left to move in this direction

        new_bits = list(bits)
        new_bits[best_h] = _neighbor(bits[best_h], direction)  # type: ignore[arg-type]
        # Only accept the step if it gets the total strictly closer to target.
        if abs(target_total - sum(new_bits)) < abs(diff):
            bits = new_bits
        else:
            break

    return bits


def quantize_head(keys_h: mx.array, b: int, group_size: int = 32) -> mx.array:
    """Quantize one head's keys at ``b`` bits with KIVI-style group quant.

    Args:
        keys_h: [S, D] fp16 or fp32 — a single head's key tensor.
        b: Bit-width for this head.
        group_size: Group size along the token axis.

    Returns:
        Reconstructed keys [S, D] fp16.
    """
    return _group_quant_dequant(keys_h, b, group_size).astype(mx.float16)


__all__ = [
    "compute_head_norm_variance",
    "compute_head_attention_entropy",
    "allocate_head_bits",
    "quantize_head",
]
