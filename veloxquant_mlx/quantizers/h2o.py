"""H2O (Heavy Hitter Oracle) KV eviction primitives.

Inspired by "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models" (Zhang et al., ICLR 2024, arXiv:2306.14048).
Documented as "H2O-adapted (VeloxQuant-MLX implementation)" — not a faithful
port. See adaptation limitations below and in cache/h2o_cache.py.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: the paper accumulates attention weights from the actual
    query vectors at each decode step. At cache level the query is not visible,
    so we use the incoming key vector as a proxy query to approximate the
    attention distribution. This is the same approximation used by SnapKV-adapted.
  - Uniform budget across all heads.
  - Scores are accumulated additively (sum of softmax weights) rather than
    the paper's exact formulation, which may differ in low-budget regimes.

KNOWN, CONFIRMED-ON-REAL-MODELS PROBLEM (not fixed here, see
cache/h2o_cache.py and docs-site/docs/algorithms/h2o.md for the full
writeup): every newly-arrived token starts at score 0.0, and eviction removes
the global-minimum-score token — so on essentially any real score
distribution, the brand-new token is its own eviction target almost every
time the cache is full. In practice this freezes the kept set on whichever
tokens filled the budget first (typically the prompt) and never admits
anything generated afterward. Confirmed via mlx_lm.generate on Llama-3.2-1B
and Mistral-7B: after dozens of true decode steps, the kept set was still
exactly the original prompt positions, and generation degenerated into
repetition loops from having no visibility into its own recent output. This
is a consequence of implementing the paper's own formula correctly, not a
deviation from it — fixing it needs a different eviction/scoring policy
(e.g. a grace period before eviction-eligibility), tracked as follow-up work.

RoPE position remapping (a separate, narrower, and now-fixed correctness
issue — real, but NOT the cause of the freeze above):
  Incoming keys arrive already RoPE-rotated for their true absolute position
  (rotation happens in the model's attention module, upstream of this cache).
  Evicting a row from the middle of the sequence leaves the survivors'
  *storage index* out of sync with the absolute position baked into their
  rotation — the model's attention math assumes contiguous positions, so this
  desync would corrupt every subsequent query's relative-position math *if*
  interior eviction happens (rare given the freeze above, but not impossible,
  e.g. with ``n_sink > 0`` protecting only the front). Every kept row's true
  original position is now tracked in ``H2OState.positions``, and
  ``h2o_update`` de-rotates + re-rotates keys to a gap-free contiguous layout
  via :func:`veloxquant_mlx.quantizers.a2ats_rope.rope_remap_positions`
  whenever an eviction actually changes the kept set's positions. Verified
  via a synthetic fingerprinted-token test forcing an interior eviction and
  confirming every surviving key de-rotates back to its exact original value.

Scalability fix (a separate, now-fixed bug found while testing long
context — independent of both issues above): the original ``h2o_update``
processed every incoming token with a Python ``for`` loop, issuing 4 separate
``mx.concatenate`` calls (keys, values, scores, positions) per token. On a
genuinely long prefill (~3200 tokens tested) with no eviction yet triggered,
this built an unfused lazy-eval graph large enough to exceed MLX's Metal
resource/command-buffer tracking limit — ``RuntimeError: [metal::malloc]
Resource limit (499000) exceeded`` — crashing *before generation could even
start*, regardless of budget or eviction settings. Fixed by
:func:`_batch_absorb_no_eviction`: whichever leading portion of an incoming
batch is guaranteed not to trigger eviction (because the cache has not yet
reached ``budget``) is absorbed via one batched masked-attention matmul
instead of a token-by-token loop. Verified numerically equivalent to the
sequential loop it replaces (score/position/key parity within fp16 rounding,
including across the batch-path/loop-path boundary within a single call).
Only the genuinely sequential eviction logic (each eviction depends on the
previous one) still loops per token, and only once the cache is actually at
or over budget.

Fused Metal kernel for the over-budget branch (optional, further speedup on
top of the vectorization above): the per-token eviction step itself — append,
sink-protected argmin, evict, RoPE-remap — is still a Python loop, but each
iteration's ~15 small MLX ops (concat x4, argmin, boolean-index x4, remap)
can optionally be replaced with two fused Metal dispatches via
:func:`veloxquant_mlx.metal.h2o_fused_evict` (design:
``paper/research/H2O_METAL_KERNEL_TECH_SPEC.md``). Verified bit-for-bit
equivalent to the MLX eviction branch it replaces (18 parity tests in
``veloxquant_mlx/tests/metal/test_h2o_evict.py``, including sink protection,
interior eviction, tie-break-matches-``mx.argmin``, and untouched-rows-are-
exact-copies). Measured ~3.4x faster than the pure-MLX per-token branch at
``n_kept=512, D=128``. Used automatically when
``veloxquant_mlx.metal.metal_available()`` is true; falls back to the
pure-MLX loop otherwise — this library works without Metal, per the
existing policy for every other kernel here.

Public API
----------
H2OState          — immutable per-head state dataclass
init_h2o_state    — construct empty state
h2o_update        — absorb S new tokens, evict if over budget, remap RoPE
h2o_get_kv        — extract current (keys, values) arrays
h2o_fp16_bytes    — bytes stored in current state
full_h2o_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from veloxquant_mlx.quantizers.a2ats_rope import rope_remap_positions


@dataclass
class H2OState:
    """Per-head sliding H2O state.

    Attributes:
        keys:      [n_kept, D] fp16 stored key rows, RoPE'd at ``positions``
                   (contiguous 0..n_kept-1 after any eviction), or None
                   before first update.
        values:    [n_kept, D] fp16 stored value rows, or None before first
                   update.
        scores:    [n_kept] cumulative softmax attention mass (float32), or
                   None.
        positions: [n_kept] int32 absolute positions each kept key is
                   currently rotated at. Reassigned to a contiguous range
                   whenever eviction changes which rows survive, and each
                   key is re-rotated (via ``rope_remap_positions``) to match.
        n_sink:    Number of leading sink positions — never evicted.
        budget:    Maximum tokens to keep at any time (including sinks).
        rope_base: RoPE frequency base used to de-rotate/re-rotate kept keys.
        next_pos:  Absolute position the next incoming token will occupy —
                   tracks true sequence position across calls, independent
                   of how many rows have been evicted so far.
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    positions: mx.array | None
    n_sink: int
    budget: int
    rope_base: float
    next_pos: int


def init_h2o_state(n_sink: int, budget: int, head_dim: int, rope_base: float = 10000.0) -> H2OState:  # noqa: ARG001
    """Create an empty H2OState before any tokens arrive.

    Args:
        n_sink:    Number of initial sink positions to protect from eviction.
        budget:    Maximum total tokens kept (sinks + non-sinks).
        head_dim:  Head dimension D (unused here; accepted for API symmetry
                   with StreamingLLM's init_streaming_window).
        rope_base: RoPE frequency base — must match the base the model's own
                   attention module uses, or position remapping after
                   eviction will not cancel out the original rotation
                   correctly.

    Raises:
        ValueError: if there are sink positions to protect but they leave no
            evictable room within ``budget`` (``n_sink=0, budget=0`` remains
            a valid "disabled cache" configuration).
    """
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"h2o: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable positions remain, so sinks would be evicted once "
            "the cache fills"
        )
    return H2OState(
        keys=None,
        values=None,
        scores=None,
        positions=None,
        n_sink=n_sink,
        budget=budget,
        rope_base=rope_base,
        next_pos=0,
    )


def _attention_scores(query_proxy: mx.array, keys: mx.array) -> mx.array:
    """Softmax attention weights of query_proxy against each key row.

    Args:
        query_proxy: [D] — used as a stand-in for the true query.
        keys:        [n, D] — existing key rows.

    Returns:
        [n] softmax weights summing to ~1.
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy) * scale  # [n]
    return mx.softmax(logits, axis=-1)


def _batch_absorb_no_eviction(
    prior_keys: mx.array | None,
    prior_scores: mx.array | None,
    new_keys: mx.array,
) -> mx.array:
    """Vectorized score accumulation for a batch of incoming tokens, valid
    ONLY while every one of them is guaranteed to survive (no eviction can
    occur for any of them). Equivalent to calling the per-token score-update
    step of :func:`h2o_update` once per row of ``new_keys`` in sequence, but
    computed as a single masked batched attention matmul instead of a Python
    loop with S separate ``mx.concatenate`` calls — this is what let a long
    prefill (thousands of tokens) exceed MLX's Metal resource limit before
    this fix, independent of eviction/budget settings.

    Args:
        prior_keys:   ``[P, D]`` fp32 keys already in the cache before this
                      batch (``None`` if the cache was empty).
        prior_scores: ``[P]`` fp32 cumulative scores already in the cache
                      (``None`` if the cache was empty).
        new_keys:     ``[S, D]`` fp32 incoming keys, all of which the caller
                      has verified will fit within budget with no eviction.

    Returns:
        ``[P + S]`` fp32 scores: ``prior_scores`` updated with every new
        token's contribution, concatenated with each new token's own
        (partial) accumulated score — the newest token always ends at
        exactly ``0.0``, matching the per-token loop's semantics.
    """
    S = new_keys.shape[0]
    scale = 1.0 / math.sqrt(float(new_keys.shape[-1]))

    if prior_keys is None:
        # First S tokens ever: token 0 bootstraps to 1.0 (matches
        # h2o_update's bootstrap branch); every later token j attends
        # causally over tokens 0..j-1 only.
        logits = (new_keys @ new_keys.T) * scale  # [S, S]
        causal = mx.arange(S)[:, None] > mx.arange(S)[None, :]
        masked = mx.where(causal, logits, -mx.inf)
        attn = mx.softmax(masked, axis=-1)
        attn = mx.where(mx.isnan(attn), 0.0, attn)  # row 0: all -inf -> nan -> 0
        col_sums = mx.sum(attn, axis=0)  # [S], token j's mass from tokens after it
        bootstrap = mx.concatenate([mx.array([1.0], dtype=mx.float32), mx.zeros((S - 1,))])
        return col_sums + bootstrap

    # Prior tokens already in the cache: each new token j attends over ALL
    # prior tokens (full mass) plus new tokens 0..j-1 (causal within batch).
    P = prior_keys.shape[0]
    prior_logits = (new_keys @ prior_keys.T) * scale  # [S, P]
    new_logits = (new_keys @ new_keys.T) * scale  # [S, S]
    causal = mx.arange(S)[:, None] > mx.arange(S)[None, :]
    new_logits = mx.where(causal, new_logits, -mx.inf)
    full_logits = mx.concatenate([prior_logits, new_logits], axis=-1)  # [S, P+S]
    attn = mx.softmax(full_logits, axis=-1)  # every row has >=1 real prior token -> no nan
    prior_contrib = mx.sum(attn[:, :P], axis=0)  # [P] mass each prior token gains
    new_contrib = mx.sum(attn[:, P:], axis=0)  # [S] mass each new token gains from later new tokens
    updated_prior_scores = prior_scores + prior_contrib
    return mx.concatenate([updated_prior_scores, new_contrib], axis=0)


def _batch_absorb_prefix(
    state: H2OState,
    new_keys: mx.array,
    new_values: mx.array,
) -> H2OState:
    """Absorb a batch of ``B`` incoming tokens known in advance to fit within
    budget with zero eviction (``B <= state.budget - n_current``), as a
    single vectorized update instead of ``B`` sequential per-token steps.

    Args:
        state:      Current H2OState (may be empty, i.e. ``state.keys is None``).
        new_keys:   ``[B, D]`` fp16 rows, all guaranteed to survive.
        new_values: ``[B, D]`` fp16 rows.

    Returns:
        Updated H2OState with ``n_current + B`` rows, no eviction applied.
    """
    B = new_keys.shape[0]
    new_positions = mx.arange(state.next_pos, state.next_pos + B, dtype=mx.int32)

    prior_keys32 = None if state.keys is None else state.keys.astype(mx.float32)
    scores = _batch_absorb_no_eviction(prior_keys32, state.scores, new_keys.astype(mx.float32))

    if state.keys is None:
        keys_cat = new_keys.astype(mx.float16)
        values_cat = new_values.astype(mx.float16)
        positions_cat = new_positions
    else:
        keys_cat = mx.concatenate([state.keys, new_keys.astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, new_values.astype(mx.float16)], axis=0)
        positions_cat = mx.concatenate([state.positions, new_positions], axis=0)

    return H2OState(
        keys=keys_cat,
        values=values_cat,
        scores=scores,
        positions=positions_cat,
        n_sink=state.n_sink,
        budget=state.budget,
        rope_base=state.rope_base,
        next_pos=state.next_pos + B,
    )


# How often (in loop iterations) h2o_update forces graph materialization
# during the over-budget eviction loop. Without this, a long prefill whose
# budget is exceeded almost immediately queues one eviction's worth of
# unevaluated graph nodes per token, exhausting MLX's Metal resource/
# command-buffer tracking limit before generation can even finish — see the
# flush call site below for the full explanation and how this was found.
# 32 is conservative: confirmed empirically to keep a ~3200-token, heavy-
# eviction prefill well under the ~499000-resource ceiling on the fused
# Metal path (the tighter of the two paths), with negligible eval overhead
# (mx.eval on 4 small 1-D/2-D arrays is cheap relative to the eviction work
# it flushes).
_EVAL_FLUSH_INTERVAL = 32


def _metal_evict_available() -> bool:
    """True iff the fused Metal eviction kernel can be used on this build.

    Lazily imported (not at module top-level) so this module has no hard
    dependency on ``mx.fast.metal_kernel`` being present — matches the
    pattern in :mod:`veloxquant_mlx.metal`'s own ``__getattr__``.
    """
    try:
        from veloxquant_mlx.metal import metal_available

        return metal_available()
    except Exception:
        return False


def _evict_via_mlx(
    keys_cat: mx.array,
    values_cat: mx.array,
    scores_cat: mx.array,
    positions_cat: mx.array,
    n_sink: int,
    rope_base: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Pure-MLX eviction: sink-protected argmin, drop the loser, re-rotate
    the shifted survivors. Reference implementation — see module docstring
    for why this exists (the fused Metal path in :func:`_evict_via_metal`
    must match this bit-for-bit)."""
    n_total = keys_cat.shape[0]
    n_sink_eff = min(n_sink, n_total)
    if n_sink_eff > 0:
        inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
        protected = mx.concatenate([inf_block, scores_cat[n_sink_eff:]], axis=0)
    else:
        protected = scores_cat

    evict_idx = int(mx.argmin(protected).item())
    evicted_pos = int(positions_cat[evict_idx].item())
    keep_indices = [j for j in range(n_total) if j != evict_idx]
    keys_kept = keys_cat[keep_indices]
    values_kept = values_cat[keep_indices]
    scores_kept = scores_cat[keep_indices]
    old_positions_kept = positions_cat[keep_indices]

    # Evicting row `evict_idx` leaves a size-1 gap at `evicted_pos`. Rows
    # that sat *before* the gap (sinks included) keep their exact original
    # position — nothing about their true distance from any future query
    # changes. Rows *after* the gap shift down by exactly one position each,
    # closing the gap so the model's position bookkeeping (which assumes a
    # contiguous cache) stays in sync with what is actually stored. Minimal
    # disturbance: only rows after the gap are re-rotated.
    shift = mx.where(old_positions_kept > evicted_pos, -1, 0)
    new_positions = old_positions_kept + shift
    keys_kept = rope_remap_positions(keys_kept, old_positions_kept, new_positions, base=rope_base)
    return keys_kept, values_kept, scores_kept, new_positions


def _evict_via_metal(
    keys_cat: mx.array,
    values_cat: mx.array,
    scores_cat: mx.array,
    positions_cat: mx.array,
    n_sink: int,
    rope_base: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused-Metal-kernel eviction — see
    :func:`veloxquant_mlx.metal.h2o_fused_evict` and
    ``paper/research/H2O_METAL_KERNEL_TECH_SPEC.md``. Bit-for-bit equivalent
    to :func:`_evict_via_mlx` (verified in
    ``veloxquant_mlx/tests/metal/test_h2o_evict.py``); single-(batch*head)
    call here (``BH=1``), since ``h2o_update`` operates on one head's state
    at a time — batching across heads is a natural follow-up (see spec
    section 6) not implemented yet.
    """
    from veloxquant_mlx.metal import h2o_fused_evict

    keys_out, values_out, scores_out, positions_out = h2o_fused_evict(
        keys_cat[None],
        values_cat[None],
        scores_cat[None],
        positions_cat[None],
        n_sink=n_sink,
        rope_base=rope_base,
    )
    return keys_out[0], values_out[0], scores_out[0], positions_out[0]


def h2o_update(
    state: H2OState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> H2OState:
    """Absorb S new tokens into state, evicting the lowest-score non-sink token if over budget.

    For each of the S incoming tokens:
      1. Compute approximate attention weights of the new key (as proxy query)
         over all currently stored keys.
      2. Accumulate those weights into existing per-token scores.
      3. Append the new token with score 0 (it starts accumulating next step)
         at its true absolute position (``state.next_pos``).
      4. If total tokens > budget: permanently evict the non-sink token with the
         lowest cumulative score. Survivors positioned *before* the evicted
         token's position keep their exact original rotation untouched;
         survivors *after* it shift down by exactly one position each (closing
         the gap) and are re-rotated accordingly, so the cache's storage index
         always matches a contiguous position range (see module docstring).

    Performance note: whichever *leading* portion of ``new_keys`` is
    guaranteed not to trigger eviction (because the cache has not yet
    reached ``budget``) is absorbed in one vectorized batched-attention call
    (:func:`_batch_absorb_no_eviction`) instead of a Python loop with 4
    separate ``mx.concatenate`` calls per token — this is what let a long
    prefill (thousands of tokens, no eviction yet triggered) exceed MLX's
    Metal resource limit before this fix. Only once the cache is actually at
    or over budget does the remaining per-token eviction loop run, since each
    eviction's outcome depends on the previous one and cannot be batched
    without changing which token gets evicted.

    Args:
        state:      Current H2OState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated H2OState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]
    n_current = 0 if state.keys is None else state.keys.shape[0]
    n_batchable = max(0, min(S, state.budget - n_current))

    if n_batchable > 0:
        state = _batch_absorb_prefix(state, new_keys[:n_batchable], new_values[:n_batchable])
        new_keys = new_keys[n_batchable:]
        new_values = new_values[n_batchable:]
        S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]
        cur_pos = state.next_pos

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = H2OState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                scores=mx.ones((1,), dtype=mx.float32),
                positions=mx.array([cur_pos], dtype=mx.int32),
                n_sink=state.n_sink,
                budget=state.budget,
                rope_base=state.rope_base,
                next_pos=cur_pos + 1,
            )
            continue

        # --- score update --------------------------------------------------
        attn = _attention_scores(k_i.astype(mx.float32), state.keys.astype(mx.float32))
        updated_scores = state.scores + attn  # [n_kept]

        # --- append new token (score = 0; begins accumulating next step) ---
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        scores_cat = mx.concatenate([updated_scores, mx.zeros((1,), dtype=mx.float32)], axis=0)
        positions_cat = mx.concatenate(
            [state.positions, mx.array([cur_pos], dtype=mx.int32)], axis=0
        )

        n_total = keys_cat.shape[0]

        if n_total > state.budget:
            if _metal_evict_available():
                keys_cat, values_cat, scores_cat, positions_cat = _evict_via_metal(
                    keys_cat, values_cat, scores_cat, positions_cat, state.n_sink, state.rope_base
                )
            else:
                keys_cat, values_cat, scores_cat, positions_cat = _evict_via_mlx(
                    keys_cat, values_cat, scores_cat, positions_cat, state.n_sink, state.rope_base
                )

        state = H2OState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            positions=positions_cat,
            n_sink=state.n_sink,
            budget=state.budget,
            rope_base=state.rope_base,
            next_pos=cur_pos + 1,
        )

        # Force materialization periodically. Without this, a long prefill
        # whose budget is exceeded almost immediately (heavy eviction from
        # the first over-budget token onward) queues one eviction's worth of
        # graph nodes per loop iteration — concatenations and boolean-index
        # ops on the pure-MLX path, or two kernel dispatches on the fused
        # Metal path — without ever forcing evaluation, across potentially
        # thousands of iterations. Confirmed on real models: this exhausts
        # MLX's Metal resource/command-buffer tracking limit exactly like
        # the pre-vectorization prefill crash this module's docstring
        # describes, except triggered by the eviction loop instead of the
        # since-fixed below-budget batch path — and the fused Metal path
        # hits the ceiling with FEWER iterations than the pure-MLX path
        # (each kernel dispatch appears to register more Metal-side
        # resources per call than an equivalent handful of array ops), so
        # this flush is required for both paths, not just Metal.
        if (i + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(state.keys, state.values, state.scores, state.positions)

    return state


def h2o_get_kv(state: H2OState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def h2o_fp16_bytes(state: H2OState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_h2o_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "H2OState",
    "init_h2o_state",
    "h2o_update",
    "h2o_get_kv",
    "h2o_fp16_bytes",
    "full_h2o_fp16_bytes",
]
