"""Keyformer KV eviction primitives — Gumbel-regularized accumulating scorer.

Inspired by "Keyformer: KV Cache Reduction through Key Tokens Selection for
Efficient Generative Inference" (Adnan et al., MLSys 2024, arXiv:2403.09054).
Documented as "Keyformer-adapted (VeloxQuant-MLX implementation)" — not a
faithful port. See adaptation limitations below and in cache/keyformer_cache.py.

The paper's finding: naively evicting by an accumulated attention score is
unstable — a token that scores low early (before the tokens that will attend
to it have arrived) is evicted and can never recover, even if it would have
become a heavy hitter. Keyformer regularizes the eviction decision with
**Gumbel noise** on the score logits: a temperature-controlled perturbation
that keeps the retained set stochastically "soft" enough that borderline
tokens are not deterministically pruned on the first low reading. As decoding
proceeds the temperature is annealed toward 0, so late decisions are sharp.

WHERE THIS SITS IN THE REPO
---------------------------
This is the repo's proxy-attention scorer family (SnapKV / H2O / TOVA /
PyramidKV / SqueezeAttention / ChunkKV / CaM). Structurally it is the
**H2O pair** (``quantizers/h2o.py``): an additive accumulation of proxy
attention mass with a protected-sink top-budget eviction. The *only* new
mechanism is the Gumbel-noise regularizer added to the accumulated logits
before the keep/evict selection. Setting ``tau_init = tau_end = 0`` removes
the noise and recovers H2O-adapted's deterministic behavior exactly — that is
the honest ablation, exercised by the benchmark and a dedicated test.

FIXED, CONFIRMED-BY-PAPER-COMPARISON GAP #1 — no temperature annealing.
Section 3.3.1 / Equation 10 of the paper is not a footnote: the annealed
temperature schedule ``tau = tau_init + t * delta_tau`` (``delta_tau =
(tau_end - tau_init) / T``) is *the* mechanism that makes Keyformer's
score function behave like a near-standard softmax during the prompt phase
(``tau approx 1``, no tokens discarded yet, nothing to regularize against)
and grows deliberately more randomized as more tokens are discarded during
decoding (``tau -> tau_end``, paper default ``tau_end = 2``) — see
Section 3.3.1 and Appendix A.8. A single frozen ``tau`` for the entire run
(the previous state of this module) cannot reproduce that: it is either
always too sharp (no regularization when it is needed most, late in
generation) or always too soft (unnecessary randomization during the prompt,
when every token is still present and nothing has been discarded yet). Fixed
by tracking ``tau_init``/``tau_end``/``anneal_steps`` and computing the
current step's temperature as ``tau_init + min(pos, anneal_steps) *
delta_tau`` — see :class:`KeyformerState`. ``tau_init == tau_end`` (the
default) reproduces a constant temperature exactly, so this is backward
compatible with every existing constant-``tau`` caller and test.

FIXED, CONFIRMED-BY-PAPER-COMPARISON GAP #2 — no RoPE position tracking.
This module never tracked which absolute position each kept key was rotated
at, so an interior eviction (anything evicted that is not the newest token —
now routine once sink/recent protection is combined with annealed noise
picking a different eviction target than plain recency) silently desynced
the survivors' storage index from the absolute position baked into their
rotation, corrupting every subsequent proxy-attention computation and the
model's own downstream attention math. This is the exact bug class H2O-adapted
had and fixed (see ``quantizers/h2o.py``'s module docstring, "RoPE position
remapping"). Fixed the same way: every kept row's true original position is
now tracked in ``KeyformerState.positions``, and ``keyformer_update``
re-rotates the shifted survivors via
:func:`veloxquant_mlx.quantizers.a2ats_rope.rope_remap_positions` whenever an
eviction changes the kept set's positions.

FUSED METAL KERNEL for the over-budget branch (optional, mirrors H2O-adapted's
kernel): the per-token eviction step — append, Gumbel-regularized argmin,
evict, RoPE-remap — can optionally be replaced with two fused Metal
dispatches via :func:`veloxquant_mlx.metal.keyformer_fused_evict`, structurally
identical to :func:`veloxquant_mlx.metal.h2o_fused_evict` with one addition: a
per-row frozen Gumbel value is threaded through both dispatches and folded
into the reduction as ``score + tau * gumbel``. Equivalent to the MLX eviction branch it replaces (keys within 1 fp16 ULP;
the kernel rotates in fp32) (see
``veloxquant_mlx/tests/metal/test_keyformer_evict.py``). Used automatically
when ``veloxquant_mlx.metal.metal_available()`` is true; falls back to the
pure-MLX loop otherwise.

THE HONESTY CRUX (read before trusting any number)
--------------------------------------------------
1. **Proxy query.** Like H2O/SnapKV-adapted, a cache never sees the true query
   vector, so the incoming KEY is used as a proxy query to estimate the
   attention each stored key receives. The paper accumulates the model's real
   attention logits. This is a documented substitution, not the paper's math.
2. **Frozen per-position noise, not redrawn sampling.** The paper redraws
   Gumbel noise fresh each decoding step. A cache processes blocks with no
   global step counter it can trust to redraw against, so we draw ONE
   deterministic Gumbel value per token position (seeded from a fixed base
   seed + a per-head running position) and freeze it; the (now-annealed, see
   gap #1 above) temperature scales that frozen noise. This preserves the
   mechanism's intent — borderline tokens are perturbed so a single low
   reading does not doom them — while staying reproducible and
   order-diagnosable. It is NOT the paper's redraw-every-step sampling, and
   we do not claim it is.
3. Nothing here is validated on a trained model. The regularizer's benefit is
   measured only under constructed "late-riser" geometry in the benchmark,
   with a control where it shows no advantage.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (crux 1).
  - Frozen deterministic per-position Gumbel, not redrawn each step (crux 2).
  - Uniform budget / n_sink / tau schedule across all heads.
  - ``recent`` (trailing protected window) is an extension, off by default.

Public API (mirrors quantizers/h2o.py)
---------------------------------------
KeyformerState        — immutable per-head state dataclass
init_keyformer_state  — construct empty state (validates guards)
keyformer_update      — absorb S new tokens, evict if over budget, remap RoPE
keyformer_get_kv      — extract current (keys, values) arrays
keyformer_fp16_bytes  — bytes stored in current state
full_keyformer_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from veloxquant_mlx.quantizers._eviction_common import (
    attention_scores,
    fp16_kv_bytes,
    full_fp16_kv_bytes,
    get_kv,
)
from veloxquant_mlx.quantizers.a2ats_rope import rope_remap_positions
from veloxquant_mlx.quantizers.h2o import _rope_remap_positions_batched


@dataclass
class KeyformerState:
    """Per-head Keyformer eviction state.

    Attributes:
        keys:      [n_kept, D] fp16 stored key rows, RoPE'd at ``positions``
                   (contiguous 0..n_kept-1 after any eviction), or None before
                   first update.
        values:    [n_kept, D] fp16 stored value rows, or None before first
                   update.
        scores:    [n_kept] cumulative proxy-attention mass (float32), or None.
        gumbel:    [n_kept] frozen per-position Gumbel noise (float32), or
                   None. Drawn once when a token is inserted; never redrawn.
                   Added (scaled by the current annealed temperature) to
                   ``scores`` only for the keep/evict decision — the stored
                   cumulative mass itself stays clean.
        positions: [n_kept] int32 absolute positions each kept key is
                   currently rotated at. Reassigned to a contiguous range
                   whenever eviction changes which rows survive, and each key
                   is re-rotated (via ``rope_remap_positions``) to match. Same
                   fix as H2O-adapted's ``H2OState.positions`` — see module
                   docstring gap #2.
        pos:       Running count of token positions this head has ever
                   inserted; seeds the deterministic Gumbel draw AND drives
                   the temperature-annealing schedule, so both are
                   reproducible and independent of block boundaries.
        n_sink:    Number of leading sink positions — never evicted.
        budget:    Maximum tokens kept at any time (including sinks).
        recent:    Trailing protected window (0 = off, paper-faithful).
        tau_init:  Gumbel-noise temperature (>= 0) at ``pos == 0``. The
                   paper's default is 1 (Section 3.3.1): during the prompt
                   phase, when no tokens have been discarded yet, this keeps
                   the score function close to a plain softmax.
        tau_end:   Gumbel-noise temperature (>= tau_init) once ``pos >=
                   anneal_steps``. The paper's default is 2: the more tokens
                   discarded, the more randomized the score function becomes,
                   per Equation 10. ``tau_end == tau_init`` reproduces a
                   constant temperature (backward compatible with every
                   existing constant-``tau`` caller).
        anneal_steps: Number of update steps (``T`` in the paper) over which
                   ``tau`` ramps linearly from ``tau_init`` to ``tau_end``.
                   Steps beyond this hold at ``tau_end``. Ignored (no
                   annealing) when ``tau_init == tau_end``.
        rope_base: RoPE frequency base used to de-rotate/re-rotate kept keys.
                   Must match the model's own attention module base.
        next_pos:  Absolute position the next incoming token will occupy —
                   tracks true sequence position across calls, independent of
                   how many rows have been evicted so far.
        seed:      Base seed for the deterministic per-position Gumbel draw.
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    gumbel: mx.array | None
    positions: mx.array | None
    pos: int
    n_sink: int
    budget: int
    recent: int
    tau_init: float
    tau_end: float
    anneal_steps: int
    rope_base: float
    next_pos: int
    seed: int

    @property
    def tau(self) -> float:
        """Backward-compatible alias for ``tau_init`` (pre-annealing field name)."""
        return self.tau_init


def _tau_at(state: KeyformerState) -> float:
    """Current annealed temperature at ``state.pos``, per Equation 10.

    ``tau = tau_init + min(pos, anneal_steps) * delta_tau``, where
    ``delta_tau = (tau_end - tau_init) / anneal_steps``. Holds at ``tau_end``
    once ``pos >= anneal_steps``. Degenerates to the constant ``tau_init``
    when ``tau_init == tau_end`` or ``anneal_steps <= 0``.
    """
    if state.tau_end == state.tau_init or state.anneal_steps <= 0:
        return state.tau_init
    t = min(state.pos, state.anneal_steps)
    delta_tau = (state.tau_end - state.tau_init) / state.anneal_steps
    return state.tau_init + t * delta_tau


def init_keyformer_state(
    n_sink: int,
    budget: int,
    head_dim: int,  # noqa: ARG001 — accepted for API symmetry with init_h2o_state
    recent: int = 0,
    tau: float | None = None,
    tau_init: float = 1.0,
    tau_end: float = 1.0,
    anneal_steps: int = 0,
    rope_base: float = 10000.0,
    seed: int = 0,
) -> KeyformerState:
    """Create an empty KeyformerState before any tokens arrive.

    Args:
        n_sink:       Number of leading sink positions to protect from eviction.
        budget:       Maximum total tokens kept (sinks + non-sinks + recent).
        head_dim:     Head dimension D (unused; API symmetry with init_h2o_state).
        recent:       Trailing protected window (extension, off by default).
        tau:          Deprecated alias — if given, sets both ``tau_init`` and
                      ``tau_end`` to this constant value (no annealing),
                      overriding ``tau_init``/``tau_end``. Kept so existing
                      constant-temperature callers need no changes.
        tau_init:     Gumbel-noise temperature at ``pos == 0`` (paper default 1).
        tau_end:      Gumbel-noise temperature once annealing completes (paper
                      default 2). Equal to ``tau_init`` means no annealing.
        anneal_steps: Update steps (``T``) over which ``tau`` ramps from
                      ``tau_init`` to ``tau_end``. ``0`` disables annealing
                      (temperature is constant at ``tau_init``) regardless of
                      ``tau_end``.
        rope_base:    RoPE frequency base — must match the model's own
                      attention module, or position remapping after eviction
                      will not cancel out the original rotation correctly.
        seed:         Base seed for the deterministic per-position Gumbel draw.

    Raises:
        ValueError: if ``tau_init``/``tau_end`` is negative, or the protected
            positions (sinks + recent) leave no evictable room within budget.
    """
    if tau is not None:
        tau_init = tau
        tau_end = tau
    if tau_init < 0:
        raise ValueError(f"keyformer: tau must be >= 0, got {tau_init!r}")
    if tau_end < 0:
        raise ValueError(f"keyformer: tau_end must be >= 0, got {tau_end!r}")
    if n_sink + recent >= budget:
        raise ValueError(
            f"keyformer: n_sink ({n_sink}) + recent ({recent}) must be < "
            f"budget ({budget}) — no evictable positions remain"
        )
    return KeyformerState(
        keys=None,
        values=None,
        scores=None,
        gumbel=None,
        positions=None,
        pos=0,
        n_sink=n_sink,
        budget=budget,
        recent=recent,
        tau_init=float(tau_init),
        tau_end=float(tau_end),
        anneal_steps=int(anneal_steps),
        rope_base=float(rope_base),
        next_pos=0,
        seed=int(seed),
    )


def _gumbel_at(seed: int, pos: int) -> mx.array:
    """One deterministic Gumbel(0,1) sample keyed by (seed, pos).

    Reproducible: the same (seed, pos) always yields the same value, so a given
    token position carries the same frozen noise regardless of how blocks are
    chunked. Gumbel via inverse-CDF: -log(-log(U)), U ~ Uniform(0,1).
    """
    key = mx.random.key(seed * 1_000_003 + pos)
    u = mx.random.uniform(low=1e-9, high=1.0, key=key)  # avoid log(0)
    return -mx.log(-mx.log(u))


def _draw_uniform_from_key(key: mx.array) -> mx.array:
    return mx.random.uniform(low=1e-9, high=1.0, key=key)


def _gumbel_at_batched(seeds: list[int], pos: int) -> mx.array:
    """Batched-``[BH]`` equivalent of calling :func:`_gumbel_at` once per row.

    ``mx.random.key`` only accepts a Python ``int`` (no batched-key overload
    in this MLX build — confirmed: neither ``mx.random.uniform(key=...)``
    nor ``mx.random.gumbel`` accept a leading-axis batch of keys), so the
    ``BH`` per-head keys are built with one Python-level ``mx.random.key``
    call each — cheap (pure int arithmetic + key construction, no
    data-dependent Metal work, ~34us for BH=64 measured) and only *once per
    token step*, not once per ``(head, step)`` pair like the loop this
    replaces. ``mx.vmap`` then does the actual random draw for all ``BH``
    rows in one batched call. Verified bit-for-bit equal to stacking
    :func:`_gumbel_at` calls (same ``seed * 1_000_003 + pos`` key derivation,
    same inverse-CDF formula).
    """
    keys = mx.stack([mx.random.key(s * 1_000_003 + pos) for s in seeds])
    u = mx.vmap(_draw_uniform_from_key)(keys)
    return -mx.log(-mx.log(u))


# How often (in loop iterations) keyformer_update forces graph materialization.
# Without this, a long prefill (thousands of prompt tokens processed token-by-
# token by this loop — Keyformer has no vectorized below-budget batch path
# like H2O's _batch_absorb_no_eviction) queues one eviction's worth of
# unevaluated graph nodes per token — concatenations, boolean-index ops, or
# two Metal kernel dispatches — without ever forcing evaluation, exhausting
# MLX's Metal resource/command-buffer tracking limit before generation can
# even finish (RuntimeError: [metal::malloc] Resource limit (499000)
# exceeded). Confirmed via real-model benchmarking
# (benchmark_scripts/benchmark_keyformer_real_model.py) — this crash was NOT
# present before the Metal-kernel/RoPE-remap changes; the extra per-eviction
# graph nodes those changes introduced (positions_cat concat, gumbel_cat
# concat/kernel dispatch) tipped a long, heavily-evicting prefill over the
# limit that the shorter unfused H2O-style loop did not previously hit. Same
# fix and same interval as h2o.py's identically-named constant.
_EVAL_FLUSH_INTERVAL = 32


def _metal_evict_available() -> bool:
    """True iff the fused Metal eviction kernel can be used on this build.

    Lazily imported (not at module top-level) so this module has no hard
    dependency on ``mx.fast.metal_kernel`` being present — matches the
    pattern in :mod:`veloxquant_mlx.metal`'s own ``__getattr__`` and
    ``quantizers/h2o.py``'s identically-named helper.
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
    gumbel_cat: mx.array,
    positions_cat: mx.array,
    n_sink: int,
    recent: int,
    tau: float,
    rope_base: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Pure-MLX eviction: sink/recent-protected Gumbel-regularized argmin,
    drop the loser, re-rotate the shifted survivors. Reference implementation
    — see module docstring for why this exists (the fused Metal path in
    :func:`_evict_via_metal` must match this; keys to within 1 fp16 ULP).
    """
    n_total = keys_cat.shape[0]
    sel = scores_cat + tau * gumbel_cat

    n_sink_eff = min(n_sink, n_total)
    protect = mx.zeros((n_total,), dtype=mx.float32)
    if n_sink_eff > 0:
        protect[:n_sink_eff] = float("inf")
    if recent > 0:
        r_eff = min(recent, n_total - n_sink_eff)
        if r_eff > 0:
            protect[n_total - r_eff :] = float("inf")
    sel = sel + protect

    evict_idx = int(mx.argmin(sel).item())
    evicted_pos = int(positions_cat[evict_idx].item())
    keep = [j for j in range(n_total) if j != evict_idx]
    keys_kept = keys_cat[keep]
    values_kept = values_cat[keep]
    scores_kept = scores_cat[keep]
    gumbel_kept = gumbel_cat[keep]
    old_positions_kept = positions_cat[keep]

    # Same minimal-disturbance remap as H2O-adapted: rows before the evicted
    # gap keep their exact rotation; rows after shift down by one position
    # and are re-rotated to match.
    shift = mx.where(old_positions_kept > evicted_pos, -1, 0)
    new_positions = old_positions_kept + shift
    keys_kept = rope_remap_positions(keys_kept, old_positions_kept, new_positions, base=rope_base)
    return keys_kept, values_kept, scores_kept, gumbel_kept, new_positions


def _evict_via_metal(
    keys_cat: mx.array,
    values_cat: mx.array,
    scores_cat: mx.array,
    gumbel_cat: mx.array,
    positions_cat: mx.array,
    n_sink: int,
    recent: int,
    tau: float,
    rope_base: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Fused-Metal-kernel eviction — see
    :func:`veloxquant_mlx.metal.keyformer_fused_evict`. Equivalent (keys to
    within 1 fp16 ULP) to :func:`_evict_via_mlx` (verified in
    ``veloxquant_mlx/tests/metal/test_keyformer_evict.py``); single-
    (batch*head) call here (``BH=1``), since ``keyformer_update`` operates on
    one head's state at a time, same as H2O-adapted's ``_evict_via_metal``.
    """
    from veloxquant_mlx.metal import keyformer_fused_evict

    keys_out, values_out, scores_out, gumbel_out, positions_out = keyformer_fused_evict(
        keys_cat[None],
        values_cat[None],
        scores_cat[None],
        gumbel_cat[None],
        positions_cat[None],
        n_sink=n_sink,
        rope_base=rope_base,
        tau=tau,
        recent=recent,
    )
    return keys_out[0], values_out[0], scores_out[0], gumbel_out[0], positions_out[0]


def keyformer_update(
    state: KeyformerState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> KeyformerState:
    """Absorb S new tokens, evicting the lowest Gumbel-regularized token if over budget.

    For each of the S incoming tokens:
      1. Accumulate proxy-attention mass of the new key over all stored keys
         (H2O-adapted's additive rule).
      2. Append the new token with cumulative score 0, its true absolute
         position (``state.next_pos``), and a frozen per-position Gumbel draw
         (seeded by the head's running position).
      3. If over budget: evict the non-protected token with the lowest
         ``score + tau * gumbel``, where ``tau`` is the current annealed
         temperature (Equation 10) — the Gumbel term is the Keyformer
         mechanism; at ``tau == 0`` this is exactly H2O-adapted's argmin on
         the raw score. Survivors positioned after the evicted token's
         position shift down by one and are re-rotated (RoPE remap), same as
         H2O-adapted.

    Uses the fused Metal kernel (:func:`_evict_via_metal`) when available,
    falling back to the pure-MLX reference (:func:`_evict_via_mlx`)
    otherwise — same policy as ``h2o_update``.

    Kept tokens are returned in original temporal order.
    """
    S = int(new_keys.shape[0])
    use_metal = _metal_evict_available()

    for i in range(S):
        k_i = new_keys[i].astype(mx.float16)  # [D]
        v_i = new_values[i].astype(mx.float16)  # [D]
        g_i = _gumbel_at(state.seed, state.pos)  # frozen noise for this position
        cur_pos = state.next_pos

        if state.keys is None:
            state = KeyformerState(
                keys=k_i[None],
                values=v_i[None],
                scores=mx.ones((1,), dtype=mx.float32),
                gumbel=g_i[None],
                positions=mx.array([cur_pos], dtype=mx.int32),
                pos=state.pos + 1,
                n_sink=state.n_sink,
                budget=state.budget,
                recent=state.recent,
                tau_init=state.tau_init,
                tau_end=state.tau_end,
                anneal_steps=state.anneal_steps,
                rope_base=state.rope_base,
                next_pos=cur_pos + 1,
                seed=state.seed,
            )
            continue

        # --- accumulate proxy attention over stored keys -------------------
        attn = attention_scores(k_i.astype(mx.float32), state.keys.astype(mx.float32))
        updated_scores = state.scores + attn  # [n_kept]

        # --- append new token (score 0; begins accumulating next step) -----
        keys_cat = mx.concatenate([state.keys, k_i[None]], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None]], axis=0)
        scores_cat = mx.concatenate([updated_scores, mx.zeros((1,), dtype=mx.float32)], axis=0)
        gumbel_cat = mx.concatenate([state.gumbel, g_i[None]], axis=0)
        positions_cat = mx.concatenate(
            [state.positions, mx.array([cur_pos], dtype=mx.int32)], axis=0
        )

        n_total = int(keys_cat.shape[0])

        if n_total > state.budget:
            tau = _tau_at(state)
            evict_fn = _evict_via_metal if use_metal else _evict_via_mlx
            keys_cat, values_cat, scores_cat, gumbel_cat, positions_cat = evict_fn(
                keys_cat,
                values_cat,
                scores_cat,
                gumbel_cat,
                positions_cat,
                state.n_sink,
                state.recent,
                tau,
                state.rope_base,
            )

        state = KeyformerState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            gumbel=gumbel_cat,
            positions=positions_cat,
            pos=state.pos + 1,
            n_sink=state.n_sink,
            budget=state.budget,
            recent=state.recent,
            tau_init=state.tau_init,
            tau_end=state.tau_end,
            anneal_steps=state.anneal_steps,
            rope_base=state.rope_base,
            next_pos=cur_pos + 1,
            seed=state.seed,
        )

        # Force materialization periodically — see _EVAL_FLUSH_INTERVAL.
        if (i + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(state.keys, state.values, state.scores, state.gumbel, state.positions)

    return state


def _attention_scores_batched(query_proxy: mx.array, keys: mx.array) -> mx.array:
    """Softmax attention weights, batched over a leading ``[BH]`` axis.

    Identical formula to :func:`h2o._attention_scores_batched` (not
    imported directly to avoid a private cross-module dependency); see
    that function's docstring.

    Args:
        query_proxy: ``[BH, D]``.
        keys:        ``[BH, n, D]``.

    Returns:
        ``[BH, n]`` softmax weights, each row summing to ~1.
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy[..., None])[..., 0] * scale  # [BH, n]
    return mx.softmax(logits, axis=-1)


def _evict_via_mlx_batched(
    keys_cat: mx.array,
    values_cat: mx.array,
    scores_cat: mx.array,
    gumbel_cat: mx.array,
    positions_cat: mx.array,
    n_sink: int,
    recent: int,
    tau: float,
    rope_base: float,
    return_indices: bool = False,
) -> (
    tuple[mx.array, mx.array, mx.array, mx.array, mx.array]
    | tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]
):
    """Batched-``[BH,·]`` equivalent of :func:`_evict_via_mlx`.

    Each ``bh`` row's Gumbel-regularized argmin/evict/re-rotate is
    independent of every other row — a like-for-like vectorization of
    :func:`_evict_via_mlx`'s per-head-loop math over a leading ``BH`` axis,
    not a new eviction policy. Same protected-region construction as H2O's
    :func:`h2o._evict_via_mlx_batched`, extended with the ``recent`` window
    (an extension H2O's batched helper doesn't have) and the additive
    ``tau * gumbel`` regularizer term.
    """
    bh, n_total = scores_cat.shape
    sel = scores_cat + tau * gumbel_cat

    n_sink_eff = min(n_sink, n_total)
    protect = mx.zeros((bh, n_total), dtype=mx.float32)
    if n_sink_eff > 0:
        sink_inf = mx.full((bh, n_sink_eff), float("inf"), dtype=mx.float32)
        protect = mx.concatenate([sink_inf, protect[:, n_sink_eff:]], axis=1)
    if recent > 0:
        r_eff = min(recent, n_total - n_sink_eff)
        if r_eff > 0:
            recent_inf = mx.full((bh, r_eff), float("inf"), dtype=mx.float32)
            protect = mx.concatenate([protect[:, : n_total - r_eff], recent_inf], axis=1)
    sel = sel + protect

    evict_idx = mx.argmin(sel, axis=-1, keepdims=True)  # [BH, 1]
    rows = mx.arange(n_total - 1)[None]  # [1, n_total-1]
    source = rows + (rows >= evict_idx)  # [BH, n_total-1]

    keys_kept = mx.take_along_axis(keys_cat, source[..., None], axis=1)
    values_kept = mx.take_along_axis(values_cat, source[..., None], axis=1)
    scores_kept = mx.take_along_axis(scores_cat, source, axis=1)
    gumbel_kept = mx.take_along_axis(gumbel_cat, source, axis=1)
    old_positions_kept = mx.take_along_axis(positions_cat, source, axis=1)
    evicted_pos = mx.take_along_axis(positions_cat, evict_idx, axis=1)  # [BH, 1]

    shift = mx.where(old_positions_kept > evicted_pos, -1, 0)
    new_positions = old_positions_kept + shift
    keys_kept = _rope_remap_positions_batched(
        keys_kept, old_positions_kept, new_positions, base=rope_base
    )
    if return_indices:
        return keys_kept, values_kept, scores_kept, gumbel_kept, new_positions, source
    return keys_kept, values_kept, scores_kept, gumbel_kept, new_positions


def _evict_via_metal_batched(
    keys_cat: mx.array,
    values_cat: mx.array,
    scores_cat: mx.array,
    gumbel_cat: mx.array,
    positions_cat: mx.array,
    n_sink: int,
    recent: int,
    tau: float,
    rope_base: float,
    return_indices: bool = False,
) -> (
    tuple[mx.array, mx.array, mx.array, mx.array, mx.array]
    | tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]
):
    """Batched-``[BH,·]`` fused Metal eviction — a direct pass-through to
    :func:`veloxquant_mlx.metal.keyformer_fused_evict`, which already accepts
    a ``BH`` leading dimension (see ``metal/_keyformer_evict.py``); no new
    kernel is needed here, only batching the caller (same relationship as
    H2O's ``_metal_evict_batched`` to ``h2o_fused_evict``).
    """
    from veloxquant_mlx.metal import keyformer_fused_evict

    return keyformer_fused_evict(
        keys_cat,
        values_cat,
        scores_cat,
        gumbel_cat,
        positions_cat,
        n_sink=n_sink,
        rope_base=rope_base,
        tau=tau,
        recent=recent,
        return_indices=return_indices,
    )


def keyformer_update_batched(
    keys: mx.array | None,  # [BH, n, D] fp16 or None
    values: mx.array | None,  # [BH, n, D] fp16 or None
    scores: mx.array | None,  # [BH, n] fp32 or None
    gumbel: mx.array | None,  # [BH, n] fp32 or None
    positions: mx.array | None,  # [BH, n] int32 or None
    new_keys: mx.array,  # [BH, S, D]
    new_values: mx.array,  # [BH, S, D]
    n_sink: int,
    budget: int,
    recent: int,
    tau_init: float,
    tau_end: float,
    anneal_steps: int,
    rope_base: float,
    next_pos: int,
    pos: int,
    seeds: list[int],
    return_indices: bool = False,
) -> (
    tuple[mx.array, mx.array, mx.array, mx.array, mx.array, int, int]
    | tuple[mx.array, mx.array, mx.array, mx.array, mx.array, int, int, mx.array]
):
    """Vectorized-over-``BH`` equivalent of calling :func:`keyformer_update`

    With ``return_indices=True``, append chronological indices into the
    supplied prior + new rows (or the supplied candidate rows for eviction).
    once per ``(batch, head)`` pair with identical per-row config.

    All ``BH`` rows share ``n_sink``/``budget``/``recent``/the tau schedule/
    ``rope_base``/``next_pos``/``pos`` (true for every real caller:
    :class:`KeyformerKVCache` applies one uniform config to every head; only
    each head's Gumbel noise stream differs, via a distinct ``seeds[hh]`` —
    see :func:`_gumbel_at_batched`), so the per-token score/append/evict/
    RoPE-remap math — otherwise identical for every row — can run as one
    batched MLX call per step instead of ``BH`` separate Python-level calls
    into :func:`keyformer_update`. Same fix, same template, as
    :func:`veloxquant_mlx.quantizers.h2o.h2o_update_batched` (Keyformer is
    documented as the "Metal-fused sibling" of H2O-adapted — see this
    module's docstring).

    Numerically identical to the per-head loop it replaces: every op below
    is the same formula as :func:`keyformer_update`'s bootstrap/score-update/
    evict branches, applied over a leading ``BH`` axis instead of a Python
    loop — verified bit-for-bit (fp32-rounding-only) equivalent.

    Returns:
        ``(keys, values, scores, gumbel, positions, next_pos, pos)`` — the
        first five ``[BH, n_kept, D]``/``[BH, n_kept]``, ``next_pos`` the
        updated absolute step count and ``pos`` the updated per-head running
        position count (both shared across all rows, as in the loop this
        replaces).
    """
    bh, s, d = new_keys.shape
    n_prior = 0 if keys is None else keys.shape[1]
    indices = mx.broadcast_to(mx.arange(n_prior)[None], (bh, n_prior))
    if s == 0:
        result = (keys, values, scores, gumbel, positions, next_pos, pos)
        return (*result, indices) if return_indices else result
    if n_sink + recent >= budget:
        raise ValueError(
            f"keyformer: n_sink ({n_sink}) + recent ({recent}) must be < "
            f"budget ({budget}) — no evictable positions remain"
        )

    use_metal = _metal_evict_available()
    no_anneal = tau_end == tau_init or anneal_steps <= 0
    k_dtype = new_keys.dtype
    v_dtype = new_values.dtype

    for i in range(s):
        if return_indices:
            indices = mx.concatenate([indices, mx.full((bh, 1), n_prior + i)], axis=1)
        k_i = new_keys[:, i].astype(k_dtype)  # [BH, D]
        v_i = new_values[:, i].astype(v_dtype)  # [BH, D]
        g_i = _gumbel_at_batched(seeds, pos)  # [BH]
        cur_pos = next_pos

        if keys is None:
            keys = new_keys[:, i : i + 1].astype(k_dtype)  # [BH, 1, D]
            values = v_i[:, None, :]
            scores = mx.ones((bh, 1), dtype=mx.float32)
            gumbel = g_i[:, None]
            positions = mx.full((bh, 1), cur_pos, dtype=mx.int32)
            pos += 1
            next_pos = cur_pos + 1
            continue

        attn = _attention_scores_batched(k_i.astype(mx.float32), keys.astype(mx.float32))
        updated_scores = scores + attn  # [BH, n]

        keys_cat = mx.concatenate([keys, k_i[:, None, :]], axis=1)
        values_cat = mx.concatenate([values, v_i[:, None, :]], axis=1)
        scores_cat = mx.concatenate([updated_scores, mx.zeros((bh, 1), dtype=mx.float32)], axis=1)
        gumbel_cat = mx.concatenate([gumbel, g_i[:, None]], axis=1)
        positions_cat = mx.concatenate(
            [positions, mx.full((bh, 1), cur_pos, dtype=mx.int32)], axis=1
        )

        n_total = keys_cat.shape[1]
        if n_total > budget:
            tau = (
                tau_init
                if no_anneal
                else tau_init + min(pos, anneal_steps) * ((tau_end - tau_init) / anneal_steps)
            )
            evict_fn = _evict_via_metal_batched if use_metal else _evict_via_mlx_batched
            evicted = evict_fn(
                keys_cat,
                values_cat,
                scores_cat,
                gumbel_cat,
                positions_cat,
                n_sink,
                recent,
                tau,
                rope_base,
                return_indices=return_indices,
            )
            keys_cat, values_cat, scores_cat, gumbel_cat, positions_cat = evicted[:5]
            if return_indices:
                indices = mx.take_along_axis(indices, evicted[5], axis=1)

        keys, values, scores, gumbel, positions = (
            keys_cat,
            values_cat,
            scores_cat,
            gumbel_cat,
            positions_cat,
        )
        pos += 1
        next_pos = cur_pos + 1

        if (i + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(keys, values, scores, gumbel, positions, indices)

    result = (keys, values, scores, gumbel, positions, next_pos, pos)
    return (*result, indices) if return_indices else result


def keyformer_get_kv(state: KeyformerState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update
    (same contract as ``h2o_get_kv``).
    """
    return get_kv(state.keys, state.values)


def keyformer_fp16_bytes(state: KeyformerState) -> int:
    """Bytes currently stored for K + V in fp16.

    Scores/gumbel are transient bookkeeping (float32, ``n`` each) — negligible
    beside K+V and, like H2O's scores, not counted as cache payload.
    """
    return fp16_kv_bytes(state.keys)


def full_keyformer_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return full_fp16_kv_bytes(tokens_seen, head_dim)


__all__ = [
    "KeyformerState",
    "init_keyformer_state",
    "keyformer_update",
    "keyformer_update_batched",
    "keyformer_get_kv",
    "keyformer_fp16_bytes",
    "full_keyformer_fp16_bytes",
]
