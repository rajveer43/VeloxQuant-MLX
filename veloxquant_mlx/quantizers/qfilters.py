"""Q-Filters KV eviction primitives — query-agnostic projection scoring.

Inspired by "Q-Filters: Leveraging QK Geometry for Efficient KV Cache
Compression" (arXiv:2503.02812, **preprint**). Documented as
"Q-Filters-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

The paper's finding: for a trained attention head the (Query, Key) joint
distribution is anisotropic, so there exists a single per-head direction
(the *Q-Filter*) onto which a key's projection predicts the average
attention that key will receive. Ranking cached keys by that projection
approximates attention-based importance **without computing attention and
without a query at eviction time** — a fourth eviction scorer class the repo
otherwise lacks (not attention/proxy like SnapKV/H2O/TOVA/PyramidKV/
SqueezeAttention/ChunkKV/CaM, not structural like StreamingLLM/sink, not
intrinsic-norm like L2Norm).

TWO FILTER SOURCES (which one you get changes the guarantees)
--------------------------------------------------------------
1. **Calibrated — paper-faithful (preferred).** Pass a ``filter_dir`` computed
   offline by :mod:`veloxquant_mlx.quantizers.qfilters_calibration` from the
   **SVD of query activations** (paper §3.2, Eq. 1). This is the paper's
   actual mechanism. Because the filter is known before the first token, the
   sign is correct by construction (Theorem 3.3's ``kappa^h > 0``), eviction
   is active immediately, and the kept set is **path-independent**.

2. **Key-SVD fallback (no calibration available).** With no ``filter_dir``,
   the filter is estimated from the top right-singular vector of the first
   ``calib_tokens`` observed **keys** and frozen. A cache never sees queries,
   so this substitutes a different estimator of the same head-geometry axis.
   It recovers the axis but **not which end of it is important** — the sign is
   exactly what a query disambiguates. Under this path ``sign`` is a genuine
   ablation knob, not a cosmetic one. Nothing on this path is claimed
   equivalent to the paper's filter.

Path-dependence applies to the FALLBACK ONLY
--------------------------------------------
On the key-SVD path the filter is estimated from whichever chunk first
crosses ``calib_tokens``, so prefill-in-one-block and token-by-token decode
can freeze *different* directions and diverge; we do not claim bit-for-bit
prefill/decode equivalence there, only that scoring/eviction are
order-invariant *given the same frozen filter*. With a calibrated filter the
direction never depends on traffic, so that divergence cannot arise.

Limitations (stated plainly):
  - No RoPE position-ID *renumbering* after eviction — surviving tokens keep
    their original absolute positions (so the cache reports the true token
    position as its offset and needs no re-rotation; see the cache module and
    :issue:`171`). Positions therefore become non-contiguous with gaps where
    tokens were dropped, which the paper also does not address.
  - Uniform budget / n_sink across all heads.
  - ``recent`` (trailing protected window) is an extension, off by default.
  - Calibrated filters are model-specific; reusing another model's artifact
    is silently wrong, so the loader checks the recorded model id.

Public API (mirrors quantizers/knorm.py)
-----------------------------------------
QFiltersState           — immutable per-head state dataclass
init_qfilters_state     — construct empty state (validates budget/sign guards)
estimate_filter_dir     — top right-singular vector of observed keys (frozen)
qfilters_update         — absorb a whole [S, D] block, evict if over budget
qfilters_update_batched — same selection for all [BH, n_total, D] groups at once
qfilters_get_kv         — extract current (keys, values) arrays
qfilters_fp16_bytes     — bytes stored in current state (incl. filter_dir)
full_qfilters_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class QFiltersState:
    """Per-head Q-Filters eviction state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        scores: [n_kept] float32 projection ``sign · (key · filter_dir)`` for
                each kept row — computed once when the filter freezes / at
                insertion thereafter, never updated. None until the filter is
                frozen.
        filter_dir: [D] float32 frozen query-agnostic direction (unit norm),
                or None until ``calib_tokens`` keys have been observed.
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens kept at any time (including sinks).
        recent: Trailing protected window (0 = off, paper-faithful).
        calib_tokens: Tokens observed before the filter is estimated & frozen.
        sign:   +1 = keep highest projection (the paper's direction);
                -1 = inverted selection (the benchmark's ablation arm).
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    filter_dir: mx.array | None
    n_sink: int
    budget: int
    recent: int
    calib_tokens: int
    sign: int


def init_qfilters_state(
    n_sink: int,
    budget: int,
    head_dim: int,  # noqa: ARG001 — accepted for API symmetry with init_knorm_state
    recent: int = 0,
    calib_tokens: int = 128,
    sign: int = 1,
    filter_dir: mx.array | None = None,
) -> QFiltersState:
    """Create an empty QFiltersState before any tokens arrive.

    Args:
        filter_dir: Optional ``[D]`` pre-calibrated Q-Filter from
            :mod:`veloxquant_mlx.quantizers.qfilters_calibration` (the paper's
            query-SVD direction). When supplied the filter is already frozen,
            so ``calib_tokens`` is irrelevant and eviction is active from the
            very first token — there is no warm-up window during which the
            cache grows unbounded, and no path dependence. When omitted the
            cache falls back to the key-SVD estimator (see module docstring).

    Raises:
        ValueError: if ``sign`` is not ±1, the protected positions
            (sinks + recent) leave no evictable room within the budget, or
            ``filter_dir`` is not a 1-D vector.
    """
    if sign not in (1, -1):
        raise ValueError(f"qfilters: sign must be +1 or -1, got {sign!r}")
    if n_sink + recent >= budget:
        raise ValueError(
            f"qfilters: n_sink ({n_sink}) + recent ({recent}) must be < "
            f"budget ({budget}) — no evictable positions remain"
        )
    if filter_dir is not None:
        if filter_dir.ndim != 1:
            raise ValueError(
                f"qfilters: filter_dir must be 1-D [D], got shape {tuple(filter_dir.shape)}"
            )
        filter_dir = filter_dir.astype(mx.float32)
    return QFiltersState(
        keys=None,
        values=None,
        scores=None,
        filter_dir=filter_dir,
        n_sink=n_sink,
        budget=budget,
        recent=recent,
        calib_tokens=calib_tokens,
        sign=int(sign),
    )


def estimate_filter_dir(keys: mx.array) -> mx.array:
    """Estimate the query-agnostic filter as the top singular direction of keys.

    Computes the top right-singular vector of the mean-centered key block
    ``keys`` (``[N, D]``) — equivalently the leading eigenvector of the
    ``[D, D]`` key covariance. Sign-normalized deterministically (largest-
    magnitude component forced positive) so the result is reproducible.

    NOTE (the honesty crux): the paper estimates this direction from
    query-distribution SVD offline; we estimate it from observed KEYS. A
    documented deviation — a different estimator of the same head-geometry
    direction, not claimed equivalent.

    Returns:
        [D] float32 unit vector.
    """
    x = keys.astype(mx.float32)
    x = x - mx.mean(x, axis=0, keepdims=True)  # center
    # Leading eigenvector of the covariance via eigh (symmetric [D, D]).
    cov = (x.T @ x) / max(int(x.shape[0]) - 1, 1)
    _, vecs = mx.linalg.eigh(cov, stream=mx.cpu)  # ascending eigenvalues
    direction = vecs[:, -1]  # top eigenvector
    # Deterministic sign: force the largest-magnitude component positive.
    pivot = mx.argmax(mx.abs(direction))
    direction = direction * mx.sign(direction[pivot])
    norm = mx.sqrt(mx.sum(direction**2))
    return direction / mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))


def _project(keys: mx.array, filter_dir: mx.array, sign: int) -> mx.array:
    """Projection scores ``sign · (key · filter_dir)`` for each key row (float32)."""
    return sign * (keys.astype(mx.float32) @ filter_dir)


def qfilters_update(
    state: QFiltersState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> QFiltersState:
    """Absorb a whole block of S tokens, then evict down to budget in one shot.

    Vectorized — no per-token loop. Before ``calib_tokens`` keys have been
    observed the filter is None and everything passes through (no eviction).
    Once enough keys are seen the filter is estimated & frozen, all stored
    tokens are scored, and the over-budget case is a single protected top-k.
    Kept tokens are returned in original temporal order.

    One-shot eviction == incremental eviction (calibrated path)
    -----------------------------------------------------------
    A token is scored once, against a filter frozen before token 0, and its
    score never changes while it is cached. Selecting the top ``budget`` rows
    under a fixed key is therefore order-invariant: absorbing S tokens and
    evicting once retains exactly the rows that evicting after every chunk
    would. Feeding the same prompt as one block, in 256-token chunks, or one
    token at a time yields bit-identical K/V and the same ``budget`` bound
    (:issue:`172`; ``test_calibrated_kept_set_is_invariant_to_prefill_chunk_size``).
    Incremental eviction would buy nothing here — it is the same function.

    The trade-off this DOES expose: with ``recent=0`` the trailing tokens of a
    long prefill are scored like any other, so most of the prompt's tail —
    typically where the question lives — is evicted before the first token is
    generated, which degrades generation badly at small budgets. That is a
    property of pure projection ranking, not of chunking. Set ``recent`` to
    pin a trailing window when generation quality on long prompts matters.

    On the key-SVD FALLBACK path the frozen direction still depends on which
    chunk first crosses ``calib_tokens``, so kept sets legitimately differ
    across chunkings (see the module docstring). Only the calibrated path is
    path-independent.
    """
    new_keys = new_keys.astype(mx.float16)
    new_values = new_values.astype(mx.float16)

    if state.keys is None:
        keys_cat = new_keys
        values_cat = new_values
    else:
        keys_cat = mx.concatenate([state.keys, new_keys], axis=0)
        values_cat = mx.concatenate([state.values, new_values], axis=0)

    n_total = int(keys_cat.shape[0])

    filter_dir = state.filter_dir
    # Freeze the filter as soon as we have enough observed tokens.
    if filter_dir is None and n_total >= state.calib_tokens:
        filter_dir = estimate_filter_dir(keys_cat)

    # Still pre-calibration: keep everything, no scores yet.
    if filter_dir is None:
        return QFiltersState(
            keys=keys_cat,
            values=values_cat,
            scores=None,
            filter_dir=None,
            n_sink=state.n_sink,
            budget=state.budget,
            recent=state.recent,
            calib_tokens=state.calib_tokens,
            sign=state.sign,
        )

    # Score every stored token against the frozen filter (scores never change
    # for already-kept tokens: recomputing them yields the same values).
    scores_cat = _project(keys_cat, filter_dir, state.sign)  # [n_total] float32

    if n_total > state.budget:
        n_sink_eff = min(state.n_sink, n_total)
        # We keep the HIGHEST scores. Force protected positions to +inf so
        # they always survive the descending selection.
        protect = mx.zeros((n_total,), dtype=mx.float32)
        if n_sink_eff > 0:
            protect[:n_sink_eff] = float("inf")
        if state.recent > 0:
            r_eff = min(state.recent, n_total - n_sink_eff)
            if r_eff > 0:
                protect[n_total - r_eff :] = float("inf")
        sel = scores_cat + protect

        order = mx.argsort(sel)  # ascending
        keep_idx = mx.sort(order[n_total - state.budget :])  # top-budget, temporal order
        keys_cat = keys_cat[keep_idx]
        values_cat = values_cat[keep_idx]
        scores_cat = scores_cat[keep_idx]

    return QFiltersState(
        keys=keys_cat,
        values=values_cat,
        scores=scores_cat,
        filter_dir=filter_dir,
        n_sink=state.n_sink,
        budget=state.budget,
        recent=state.recent,
        calib_tokens=state.calib_tokens,
        sign=state.sign,
    )


def qfilters_update_batched(
    keys_cat: mx.array,  # [BH, n_total, D] fp16
    values_cat: mx.array,  # [BH, n_total, D] fp16
    filter_dir: mx.array,  # [BH, D] float32, frozen
    budget: int,
    n_sink: int,
    recent: int,
    sign: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Score and evict every ``(batch, head)`` group at once — no Python loop.

    The batched twin of :func:`qfilters_update`'s post-calibration branch, for
    the case the cache actually runs in: one frozen filter per group, a budget
    / ``n_sink`` / ``recent`` / ``sign`` shared by every head, and the same
    ``n_total`` rows in every group. Those uniformities are what make the whole
    ``[BH, n_total, D]`` block a single vectorized selection (:issue:`173`).

    Selection is identical to the per-head path: protected rows are lifted to
    ``+inf``, an ascending ``argsort`` takes the top ``budget``, and survivors
    are returned in temporal order. Ties break toward the same rows because the
    sort key and ordering convention are the same.

    Args:
        keys_cat:   ``[BH, n_total, D]`` fp16 stored + newly appended keys.
        values_cat: ``[BH, n_total, D]`` fp16, same layout.
        filter_dir: ``[BH, D]`` float32 frozen per-group unit-norm Q-Filter.
        budget:     Rows retained per group, including protected rows.
        n_sink:     Leading rows never evicted.
        recent:     Trailing rows never evicted.
        sign:       ``+1`` = paper direction; ``-1`` = inverted ablation.

    Returns:
        ``(keys, values, scores)`` — keys/values ``[BH, n_kept, D]`` fp16 and
        scores ``[BH, n_kept]`` float32, temporal order. When the block is
        under budget nothing is evicted and ``n_kept == n_total``.
    """
    bh, n_total, _ = (int(x) for x in keys_cat.shape)

    # sign * <k_i, f_g> per group — one batched matvec for all BH groups.
    scores_cat = sign * mx.einsum("gnd,gd->gn", keys_cat.astype(mx.float32), filter_dir)

    if n_total <= budget:
        return keys_cat, values_cat, scores_cat

    n_sink_eff = min(n_sink, n_total)
    protect = mx.zeros((n_total,), dtype=mx.float32)
    if n_sink_eff > 0:
        protect[:n_sink_eff] = float("inf")
    if recent > 0:
        r_eff = min(recent, n_total - n_sink_eff)
        if r_eff > 0:
            protect[n_total - r_eff :] = float("inf")
    sel = scores_cat + protect  # broadcasts [n_total] across groups

    order = mx.argsort(sel, axis=-1)  # ascending, per group
    keep_idx = mx.sort(order[:, n_total - budget :], axis=-1)  # [BH, budget]

    # take_along_axis gathers a *different* row set per group in one op — the
    # part a per-head loop was doing serially.
    gather = mx.expand_dims(keep_idx, axis=-1)
    keys_out = mx.take_along_axis(keys_cat, gather, axis=1)
    values_out = mx.take_along_axis(values_cat, gather, axis=1)
    scores_out = mx.take_along_axis(scores_cat, keep_idx, axis=1)
    return keys_out, values_out, scores_out


def qfilters_get_kv(state: QFiltersState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first
    update (same contract as ``knorm_get_kv``).
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def qfilters_fp16_bytes(state: QFiltersState) -> int:
    """Bytes currently stored for K + V in fp16, plus the frozen filter_dir.

    The filter direction is counted in full (``D * 4`` bytes, float32) once it
    exists — same byte-accounting discipline as SKVQ's permutation tables.
    """
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    total = n * D * 2 * 2  # K + V, 2 bytes each
    if state.filter_dir is not None:
        total += int(state.filter_dir.shape[0]) * 4  # float32 direction
    return total


def full_qfilters_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2


__all__ = [
    "QFiltersState",
    "init_qfilters_state",
    "estimate_filter_dir",
    "qfilters_update",
    "qfilters_update_batched",
    "qfilters_get_kv",
    "qfilters_fp16_bytes",
    "full_qfilters_fp16_bytes",
]
