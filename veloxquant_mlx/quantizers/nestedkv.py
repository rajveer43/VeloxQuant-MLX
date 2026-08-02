"""NestedKV-adapted quantizer primitives — multi-scale ensembled prefill eviction.

Inspired by "NestedKV: Nested Memory Routing for Long-Context KV Cache
Compression" (Chen, Liu, Gao, Fan, Wang, Chu, Lin, Hu; arXiv:2605.26678).
Documented as "NestedKV-adapted (VeloxQuant-MLX implementation)" — not a
faithful port. **No verified peer-reviewed venue as of 2026-07-14** — this is
a one-time, user-directed exception to this repo's standing rule that every
shipped method trace to a verified peer-reviewed venue. See
``paper/research/surveys/NEW_METHOD_SURVEY_V21.md`` for the full rationale.
Every other method in this repo (38 of 39) has a verified venue; this is the
only exception, and the next method survey reverts to requiring one.

What makes this a new axis: every eviction method already in the repo scores
a token from ONE importance signal — cumulative attention (H2O), prefill
observation-window attention (SnapKV), layer-adaptive budget (PyramidKV),
key-norm proxy (Keyformer/MorphKV), reconstruction-reliance (KVzip), or joint
key-value leverage score (CurDKV). NestedKV keeps THREE parallel key-only
statistics at three time scales (stable/global, episodic/block-local,
current/recent-window), scores a token's anomaly against each independently,
and combines the three rankings via a training-free two-axis outer rule:
a per-head blend (which scale is most discriminative on this head) and a
per-token surprise gate (route to the single strongest scale when the three
disagree, rather than average them).

Adaptation limitations (stated plainly):
  - **Unpublished preprint, no verified venue** (see module header above) —
    the headline exception for this method.
  - **One-shot prefill compression, unbounded during decode.** The paper's
    own design (Appendix A) computes scores, blend weights, and surprise
    gates ONCE at the end of prefill; decoded tokens are appended normally,
    never rescored or evicted. This is a faithful port of the paper's actual
    design, not a shortcut — but it means NestedKV's cache, unlike every
    other eviction method here (H2O, CurDKV, StreamingLLM), does not stay
    bounded through a very long decode run. Mirrors SnapKV-adapted's
    prefill-once / decode-append structure, not H2O's/CurDKV's per-step loop.
  - Episodic block means are computed over fixed prefill-time token
    positions — a faithful, non-approximated port of the paper's formula
    (no eviction-collapses-indices problem exists here, since eviction only
    happens once, after all three memory scales are already computed).
  - Gate/blend constants — beta=3.0, tau=0.60, kappa=10.0, log-prior
    (0.4, 0.4, 0.2), safeguard_alpha=0.20 — are all taken directly from the
    paper's Appendix A ("Hyperparameters"), not guessed.
  - Key-only: no query/attention-score access at all, not even a proxy
    (stronger than H2O/SnapKV/CurDKV's key-as-query proxy — NestedKV never
    approximates attention in the first place).
  - No RoPE position-ID remapping after eviction; kept tokens preserve their
    relative order.
  - No PyTorch/CUDA reference kept; pure MLX from the start.

Public API
----------
NestedKVState              — per-head prefill/decode state dataclass
init_nestedkv_state         — construct empty state
nestedkv_score              — one-shot per-head anomaly scoring over full prefill
nestedkv_allocate_head_budgets — cross-head budget competition (paper's component 5)
nestedkv_compress_prefill   — one-shot per-head scoring + eviction (prefill only)
nestedkv_append_decode      — plain unscored append (decode only)
nestedkv_get_kv
nestedkv_fp16_bytes
full_nestedkv_fp16_bytes
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class NestedKVState:
    """Per-head NestedKV state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before prefill.
        values: [n_kept, D] fp16 stored value rows, or None before prefill.
        n_sink: Number of leading sink positions — never evicted.
        compressed: Whether the one-shot prefill compression has already run.
    """

    keys: mx.array | None
    values: mx.array | None
    n_sink: int
    compressed: bool = False


def init_nestedkv_state(n_sink: int) -> NestedKVState:
    """Create an empty NestedKVState before prefill."""
    return NestedKVState(keys=None, values=None, n_sink=n_sink, compressed=False)


# ---------------------------------------------------------------------------
# Per-scale continuum memory statistics (paper Section 2.2)
# ---------------------------------------------------------------------------


def _normalize_keys(keys: mx.array) -> mx.array:
    """L2-normalize each row of ``[n, D]`` keys. Zero rows stay zero (no div-by-0)."""
    norm = mx.sqrt(mx.sum(keys * keys, axis=-1, keepdims=True))
    safe_norm = mx.maximum(norm, 1e-12)
    return keys / safe_norm


def _stable_memory(k_hat: mx.array) -> mx.array:
    """mu_s = mean over ALL normalized keys. Returns [D]."""
    return mx.mean(k_hat, axis=0)


def _episodic_memory(k_hat: mx.array, block_size: int) -> mx.array:
    """mu_e(i) = mean over the contiguous block containing token i. Returns [n, D].

    Blocks are contiguous, non-overlapping, size ``block_size`` (last block may
    be shorter). Every token in a block shares that block's mean.
    """
    n = k_hat.shape[0]
    b = max(1, min(int(block_size), n))
    means = []
    for start in range(0, n, b):
        end = min(start + b, n)
        block_mean = mx.mean(k_hat[start:end], axis=0)
        means.append(mx.broadcast_to(block_mean[None, :], (end - start, k_hat.shape[1])))
    return mx.concatenate(means, axis=0)


def _current_memory(k_hat: mx.array, window: int) -> mx.array:
    """mu_c(i) = mean over the trailing causal window ending at i. Returns [n, D].

    For token i (0-indexed), the window is [max(0, i-W+1), i] inclusive.
    """
    n, d = k_hat.shape
    w = max(1, int(window))
    # Cumulative sum trick: cumsum[i] = sum(k_hat[0..i]); window sum via cumsum diff.
    cumsum = mx.cumsum(k_hat, axis=0)  # [n, D]
    idx = mx.arange(n)
    lo = mx.maximum(idx - w + 1, 0)  # [n]
    # window_sum[i] = cumsum[i] - cumsum[lo[i]-1]  (or cumsum[i] if lo[i] == 0)
    lo_list = lo.tolist()
    window_sums = []
    for i in range(n):
        lo_i = int(lo_list[i])
        if lo_i == 0:
            window_sums.append(cumsum[i])
        else:
            window_sums.append(cumsum[i] - cumsum[lo_i - 1])
    window_sum = mx.stack(window_sums, axis=0)  # [n, D]
    counts = (idx - lo + 1).astype(mx.float32)[:, None]  # [n, 1]
    return window_sum / counts


def block_size_for(n: int) -> int:
    """b = clip(floor(n / 32), 128, 256), per the paper's schedule (Section 2.2)."""
    return int(min(256, max(128, n // 32)))


# ---------------------------------------------------------------------------
# Per-scale anomaly scores + normalization (paper Section 2.3)
# ---------------------------------------------------------------------------


def _cosine_anomaly_global(k_hat: mx.array, mu: mx.array) -> mx.array:
    """-cos(k_hat_i, mu) for every row i, against a single [D] anchor. Returns [n]."""
    sims = k_hat @ mu
    return -sims


def _cosine_anomaly_per_token(k_hat: mx.array, mu_per_token: mx.array) -> mx.array:
    """-cos(k_hat_i, mu_per_token_i) for every row i. Returns [n]."""
    sims = mx.sum(k_hat * mu_per_token, axis=-1)
    return -sims


def _min_max_normalize(x: mx.array) -> mx.array:
    """Normalize [n] to [0, 1]. Constant input maps to all-zeros (no NaN)."""
    lo = mx.min(x)
    hi = mx.max(x)
    span = hi - lo
    if float(span.item()) <= 1e-12:
        return mx.zeros_like(x)
    return (x - lo) / span


def per_scale_anomaly_scores(
    k_hat: mx.array,
    block_size: int,
    window: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute (a_s_hat, a_e_hat, a_c_hat) — min-max-normalized per-scale anomaly.

    Args:
        k_hat: [n, D] L2-normalized keys.
        block_size: episodic block size b.
        window: current-memory trailing window W.

    Returns:
        Three [n] float32 arrays, each on [0, 1].
    """
    mu_s = _stable_memory(k_hat)
    mu_e = _episodic_memory(k_hat, block_size)
    mu_c = _current_memory(k_hat, window)

    a_s = _cosine_anomaly_global(k_hat, mu_s)
    a_e = _cosine_anomaly_per_token(k_hat, mu_e)
    a_c = _cosine_anomaly_per_token(k_hat, mu_c)

    return (
        _min_max_normalize(a_s),
        _min_max_normalize(a_e),
        _min_max_normalize(a_c),
    )


# ---------------------------------------------------------------------------
# Head-adaptive blend (paper Section 2.4, first axis)
# ---------------------------------------------------------------------------


def _top_bottom_gap(x: mx.array, p: float = 0.10) -> float:
    """Delta_k = mean(top-p(x)) - mean(bottom-p(x)). Small-n floor: at least 1 element."""
    n = x.shape[0]
    k = max(1, int(round(n * p)))
    sorted_x = mx.sort(x)
    bottom = sorted_x[:k]
    top = sorted_x[n - k :]
    return float((mx.mean(top) - mx.mean(bottom)).item())


def head_adaptive_blend(
    a_s_hat: mx.array,
    a_e_hat: mx.array,
    a_c_hat: mx.array,
    beta: float = 3.0,
    prior: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> mx.array:
    """Blend the three per-scale anomaly scores with a head-adaptive softmax weight.

    Delta_k measures how discriminative scale k is on this head (gap between
    its top-10% and bottom-10% scores). Larger gap -> more weight, anchored by
    a fixed log-prior so a tied/uninformative head falls back to (0.4, 0.4, 0.2).

    Returns:
        [n] float32 blended score a_blend.
    """
    n = a_s_hat.shape[0]
    if n <= 1:
        # No spread possible; fall back to the fixed prior exactly.
        w_s, w_e, w_c = prior
        return w_s * a_s_hat + w_e * a_e_hat + w_c * a_c_hat

    deltas = [
        _top_bottom_gap(a_s_hat),
        _top_bottom_gap(a_e_hat),
        _top_bottom_gap(a_c_hat),
    ]
    log_prior = [float(mx.log(mx.array(p)).item()) for p in prior]
    logits = mx.array([log_prior[i] + beta * deltas[i] for i in range(3)])
    weights = mx.softmax(logits, axis=-1)
    w_s, w_e, w_c = float(weights[0].item()), float(weights[1].item()), float(weights[2].item())
    return w_s * a_s_hat + w_e * a_e_hat + w_c * a_c_hat


# ---------------------------------------------------------------------------
# Surprise-gated routing (paper Section 2.4, second axis)
# ---------------------------------------------------------------------------


def surprise_gated_score(
    a_s_hat: mx.array,
    a_e_hat: mx.array,
    a_c_hat: mx.array,
    a_blend: mx.array,
    tau: float = 0.60,
    kappa: float = 10.0,
) -> mx.array:
    """Combine a_blend with the strongest individual scale via a surprise gate.

    s(i) = std(a_s_hat[i], a_e_hat[i], a_c_hat[i]), then min-max normalized
    within the head and mean-centered (per the paper's Appendix A: "surprise
    scores are min-max normalized within each head and mean-centered before
    applying the gate"). a_win(i) = max of the three. alpha(i) = sigmoid(kappa
    * (s(i) - tau)). a_star(i) = (1 - alpha(i)) * a_blend(i) + alpha(i) * a_win(i).

    Returns:
        [n] float32 final NestedKV score a_star.
    """
    stacked = mx.stack([a_s_hat, a_e_hat, a_c_hat], axis=0)  # [3, n]
    surprise = mx.std(stacked, axis=0)  # [n]

    surprise_norm = _min_max_normalize(surprise)
    surprise_centered = surprise_norm - mx.mean(surprise_norm)

    a_win = mx.max(stacked, axis=0)  # [n]
    alpha = mx.sigmoid(kappa * (surprise_centered - tau))
    return (1.0 - alpha) * a_blend + alpha * a_win


def nestedkv_score(
    keys: mx.array,
    window: int = 64,
    beta: float = 3.0,
    tau: float = 0.60,
    kappa: float = 10.0,
    prior: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> mx.array:
    """One-shot per-head NestedKV score over all N prefill keys.

    Args:
        keys: [n, D] fp16/fp32 key matrix for one head (full prefill).
        window: current-memory trailing window W.
        beta, tau, kappa, prior: gate/blend constants (paper Appendix A defaults).

    Returns:
        [n] float32 final score a_star (higher = more anomalous = keep-worthy).
    """
    n = keys.shape[0]
    block_size = block_size_for(n)
    k_hat = _normalize_keys(keys.astype(mx.float32))
    a_s_hat, a_e_hat, a_c_hat = per_scale_anomaly_scores(k_hat, block_size, window)
    a_blend = head_adaptive_blend(a_s_hat, a_e_hat, a_c_hat, beta=beta, prior=prior)
    return surprise_gated_score(a_s_hat, a_e_hat, a_c_hat, a_blend, tau=tau, kappa=kappa)


# ---------------------------------------------------------------------------
# Head-wise memory competition (paper Section 2.6, component 5)
# ---------------------------------------------------------------------------


def nestedkv_allocate_head_budgets(
    head_scores: list[mx.array],
    total_budget: int,
    safeguard_alpha: float = 0.20,
) -> list[int]:
    """Allocate a total layer budget across H heads by global top-B score competition.

    Two-step rule (paper Appendix A, "head-wise safeguard"):
      1. Each head h keeps a guaranteed floor of floor(safeguard_alpha * (1-r) * n_h)
         of its own highest-scoring tokens, where r is the implied eviction ratio
         (derived from total_budget vs sum of n_h, not passed separately).
      2. The remaining budget is filled by pooling all NOT-yet-guaranteed
         (head, position, score) triples across heads and taking the global
         top-K by score until the remaining budget is exhausted.

    Args:
        head_scores: length-H list of [n_h] float32 score arrays (one per head).
        total_budget: total tokens to keep across all H heads this layer.
        safeguard_alpha: per-head guaranteed-floor fraction.

    Returns:
        length-H list of ints B_h, sum(B_h) == min(total_budget, sum(n_h)).
    """
    H = len(head_scores)
    n_h = [int(s.shape[0]) for s in head_scores]
    total_n = sum(n_h)
    total_budget = max(0, min(int(total_budget), total_n))

    if total_n == 0:
        return [0] * H

    r = 1.0 - (total_budget / total_n)
    r = max(0.0, min(1.0, r))

    floors = [max(0, min(n_h[h], int((safeguard_alpha * (1.0 - r) * n_h[h])))) for h in range(H)]
    # Floors cannot collectively exceed total_budget; scale down proportionally if so.
    floor_sum = sum(floors)
    if floor_sum > total_budget:
        scale = total_budget / floor_sum if floor_sum > 0 else 0.0
        floors = [int(f * scale) for f in floors]
        floor_sum = sum(floors)

    remaining_budget = total_budget - floor_sum

    # Build the remaining pool: for each head, all tokens NOT in its own top-floor[h].
    pool: list[tuple[int, int, float]] = []  # (head, local_idx, score)
    guaranteed_counts = list(floors)
    for h in range(H):
        scores_list = head_scores[h].tolist()
        order = sorted(range(n_h[h]), key=lambda i: scores_list[i], reverse=True)
        guaranteed_idx = set(order[: floors[h]])
        for i in range(n_h[h]):
            if i not in guaranteed_idx:
                pool.append((h, i, scores_list[i]))

    pool.sort(key=lambda t: t[2], reverse=True)
    extra_counts = [0] * H
    for h, _i, _score in pool[:remaining_budget]:
        extra_counts[h] += 1

    return [guaranteed_counts[h] + extra_counts[h] for h in range(H)]


# ---------------------------------------------------------------------------
# One-shot prefill compression / decode append
# ---------------------------------------------------------------------------


def nestedkv_compress_prefill(
    state: NestedKVState,
    keys: mx.array,  # [S, D] fp16/fp32, S > 1
    values: mx.array,  # [S, D]
    budget: int,
    window: int = 64,
    beta: float = 3.0,
    tau: float = 0.60,
    kappa: float = 10.0,
) -> NestedKVState:
    """Run the one-shot NestedKV prefill compression for one head.

    Computes a_star over all S prefill tokens, pins the first n_sink positions,
    keeps the top-``budget`` scoring tokens (sinks always included), in
    ascending original-position order. Runs exactly once per sequence.
    """
    S = keys.shape[0]
    n_sink_eff = min(state.n_sink, S)
    budget_eff = max(n_sink_eff, min(budget, S))

    scores = nestedkv_score(keys.astype(mx.float32), window=window, beta=beta, tau=tau, kappa=kappa)

    if n_sink_eff > 0:
        inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
        protected = mx.concatenate([inf_block, scores[n_sink_eff:]], axis=0)
    else:
        protected = scores

    score_list = protected.tolist()
    ranked = sorted(range(S), key=lambda i: score_list[i], reverse=True)
    kept = sorted(ranked[:budget_eff])

    kept_keys = mx.stack([keys[i] for i in kept], axis=0).astype(mx.float16)
    kept_values = mx.stack([values[i] for i in kept], axis=0).astype(mx.float16)

    return NestedKVState(
        keys=kept_keys,
        values=kept_values,
        n_sink=state.n_sink,
        compressed=True,
    )


def nestedkv_append_decode(
    state: NestedKVState,
    keys: mx.array,  # [S, D], S == 1 typically
    values: mx.array,
) -> NestedKVState:
    """Plain unscored append for decode-phase tokens. Never evicts.

    If called before prefill compression has run (state.keys is None), this
    also serves as the bootstrap path.
    """
    k16 = keys.astype(mx.float16)
    v16 = values.astype(mx.float16)
    if state.keys is None:
        keys_cat, values_cat = k16, v16
    else:
        keys_cat = mx.concatenate([state.keys, k16], axis=0)
        values_cat = mx.concatenate([state.values, v16], axis=0)
    return NestedKVState(
        keys=keys_cat,
        values=values_cat,
        n_sink=state.n_sink,
        compressed=state.compressed,
    )


def nestedkv_get_kv(state: NestedKVState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def nestedkv_fp16_bytes(state: NestedKVState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_nestedkv_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "NestedKVState",
    "init_nestedkv_state",
    "block_size_for",
    "per_scale_anomaly_scores",
    "head_adaptive_blend",
    "surprise_gated_score",
    "nestedkv_score",
    "nestedkv_allocate_head_budgets",
    "nestedkv_compress_prefill",
    "nestedkv_append_decode",
    "nestedkv_get_kv",
    "nestedkv_fp16_bytes",
    "full_nestedkv_fp16_bytes",
]
