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
before the keep/evict selection. Setting ``tau = 0`` removes the noise and
recovers H2O-adapted's deterministic behavior exactly — that is the honest
ablation, exercised by the benchmark and a dedicated test.

THE HONESTY CRUX (read before trusting any number)
--------------------------------------------------
1. **Proxy query.** Like H2O/SnapKV-adapted, a cache never sees the true query
   vector, so the incoming KEY is used as a proxy query to estimate the
   attention each stored key receives. The paper accumulates the model's real
   attention logits. This is a documented substitution, not the paper's math.
2. **Frozen per-position noise, not annealed sampling.** The paper redraws
   Gumbel noise and anneals a temperature across the full generation. A cache
   processes blocks with no global step counter it can trust, so we draw ONE
   deterministic Gumbel value per token position (seeded from a fixed base
   seed + a per-head running position) and freeze it. ``tau`` scales that
   frozen noise. This preserves the mechanism's intent — borderline tokens are
   perturbed so a single low reading does not doom them — while staying
   reproducible and order-diagnosable. It is NOT the paper's annealing
   schedule, and we do not claim it is.
3. Nothing here is validated on a trained model. The regularizer's benefit is
   measured only under constructed "late-riser" geometry in the benchmark,
   with a control where it shows no advantage.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (crux 1).
  - Frozen deterministic per-position Gumbel, no annealing schedule (crux 2).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget / n_sink / tau across all heads.
  - ``recent`` (trailing protected window) is an extension, off by default.

Public API (mirrors quantizers/h2o.py)
---------------------------------------
KeyformerState        — immutable per-head state dataclass
init_keyformer_state  — construct empty state (validates guards)
keyformer_update      — absorb S new tokens, evict if over budget
keyformer_get_kv      — extract current (keys, values) arrays
keyformer_fp16_bytes  — bytes stored in current state
full_keyformer_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class KeyformerState:
    """Per-head Keyformer eviction state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        scores: [n_kept] cumulative proxy-attention mass (float32), or None.
        gumbel: [n_kept] frozen per-position Gumbel noise (float32), or None.
                Drawn once when a token is inserted; never redrawn. Added
                (scaled by ``tau``) to ``scores`` only for the keep/evict
                decision — the stored cumulative mass itself stays clean.
        pos:    Running count of token positions this head has ever inserted;
                seeds the deterministic Gumbel draw so it is reproducible and
                independent of block boundaries.
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens kept at any time (including sinks).
        recent: Trailing protected window (0 = off, paper-faithful).
        tau:    Gumbel-noise temperature (>= 0). 0 = no noise = H2O-adapted.
        seed:   Base seed for the deterministic per-position Gumbel draw.
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    gumbel: mx.array | None
    pos: int
    n_sink: int
    budget: int
    recent: int
    tau: float
    seed: int


def init_keyformer_state(
    n_sink: int,
    budget: int,
    head_dim: int,  # noqa: ARG001 — accepted for API symmetry with init_h2o_state
    recent: int = 0,
    tau: float = 1.0,
    seed: int = 0,
) -> KeyformerState:
    """Create an empty KeyformerState before any tokens arrive.

    Raises:
        ValueError: if ``tau`` is negative, or the protected positions
            (sinks + recent) leave no evictable room within the budget.
    """
    if tau < 0:
        raise ValueError(f"keyformer: tau must be >= 0, got {tau!r}")
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
        pos=0,
        n_sink=n_sink,
        budget=budget,
        recent=recent,
        tau=float(tau),
        seed=int(seed),
    )


def _attention_scores(query_proxy: mx.array, keys: mx.array) -> mx.array:
    """Softmax proxy-attention weights of ``query_proxy`` against each key row.

    Args:
        query_proxy: [D] incoming key used as a stand-in for the true query.
        keys:        [n, D] existing key rows.

    Returns:
        [n] softmax weights summing to ~1.
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy) * scale  # [n]
    return mx.softmax(logits, axis=-1)


def _gumbel_at(seed: int, pos: int) -> mx.array:
    """One deterministic Gumbel(0,1) sample keyed by (seed, pos).

    Reproducible: the same (seed, pos) always yields the same value, so a given
    token position carries the same frozen noise regardless of how blocks are
    chunked. Gumbel via inverse-CDF: -log(-log(U)), U ~ Uniform(0,1).
    """
    key = mx.random.key(seed * 1_000_003 + pos)
    u = mx.random.uniform(low=1e-9, high=1.0, key=key)  # avoid log(0)
    return -mx.log(-mx.log(u))


def keyformer_update(
    state: KeyformerState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> KeyformerState:
    """Absorb S new tokens, evicting the lowest Gumbel-regularized token if over budget.

    For each of the S incoming tokens:
      1. Accumulate proxy-attention mass of the new key over all stored keys
         (H2O-adapted's additive rule).
      2. Append the new token with cumulative score 0 and a frozen per-position
         Gumbel draw (seeded by the head's running position).
      3. If over budget: evict the non-protected token with the lowest
         ``score + tau * gumbel`` — the Gumbel term is the Keyformer mechanism;
         at ``tau == 0`` this is exactly H2O-adapted's argmin on the raw score.

    Kept tokens are returned in original temporal order.
    """
    S = int(new_keys.shape[0])

    for i in range(S):
        k_i = new_keys[i].astype(mx.float16)  # [D]
        v_i = new_values[i].astype(mx.float16)  # [D]
        g_i = _gumbel_at(state.seed, state.pos)  # frozen noise for this position

        if state.keys is None:
            state = KeyformerState(
                keys=k_i[None],
                values=v_i[None],
                scores=mx.ones((1,), dtype=mx.float32),
                gumbel=g_i[None],
                pos=state.pos + 1,
                n_sink=state.n_sink,
                budget=state.budget,
                recent=state.recent,
                tau=state.tau,
                seed=state.seed,
            )
            continue

        # --- accumulate proxy attention over stored keys -------------------
        attn = _attention_scores(k_i.astype(mx.float32), state.keys.astype(mx.float32))
        updated_scores = state.scores + attn  # [n_kept]

        # --- append new token (score 0; begins accumulating next step) -----
        keys_cat = mx.concatenate([state.keys, k_i[None]], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None]], axis=0)
        scores_cat = mx.concatenate([updated_scores, mx.zeros((1,), dtype=mx.float32)], axis=0)
        gumbel_cat = mx.concatenate([state.gumbel, g_i[None]], axis=0)

        n_total = int(keys_cat.shape[0])

        if n_total > state.budget:
            # Gumbel-regularized selection view: score + tau * frozen noise.
            sel = scores_cat + state.tau * gumbel_cat

            # Protect sinks (leading) and recent (trailing) with +inf.
            n_sink_eff = min(state.n_sink, n_total)
            protect = mx.zeros((n_total,), dtype=mx.float32)
            if n_sink_eff > 0:
                protect[:n_sink_eff] = float("inf")
            if state.recent > 0:
                r_eff = min(state.recent, n_total - n_sink_eff)
                if r_eff > 0:
                    protect[n_total - r_eff :] = float("inf")
            sel = sel + protect

            evict_idx = int(mx.argmin(sel).item())
            keep = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep]
            values_cat = values_cat[keep]
            scores_cat = scores_cat[keep]
            gumbel_cat = gumbel_cat[keep]

        state = KeyformerState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            gumbel=gumbel_cat,
            pos=state.pos + 1,
            n_sink=state.n_sink,
            budget=state.budget,
            recent=state.recent,
            tau=state.tau,
            seed=state.seed,
        )

    return state


def keyformer_get_kv(state: KeyformerState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update
    (same contract as ``h2o_get_kv``).
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def keyformer_fp16_bytes(state: KeyformerState) -> int:
    """Bytes currently stored for K + V in fp16.

    Scores/gumbel are transient bookkeeping (float32, ``n`` each) — negligible
    beside K+V and, like H2O's scores, not counted as cache payload.
    """
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_keyformer_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2


__all__ = [
    "KeyformerState",
    "init_keyformer_state",
    "keyformer_update",
    "keyformer_get_kv",
    "keyformer_fp16_bytes",
    "full_keyformer_fp16_bytes",
]
