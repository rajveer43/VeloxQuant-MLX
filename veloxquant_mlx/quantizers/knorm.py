"""L2-norm KV eviction primitives — intrinsic key-norm token importance.

Inspired by "A Simple and Effective L2 Norm-Based Strategy for KV Cache
Compression" (Devoto, Zhao, Scardapane, Minervini — EMNLP 2024;
arXiv:2406.11430, code at https://github.com/alessiodevoto/l2compress).
Documented as "L2Norm-adapted (VeloxQuant-MLX implementation)" — not a
faithful port.

The paper's finding (in trained decoder LMs): **a low L2 norm of a key
embedding usually leads to a high attention score during decoding** — a KV
pair's influence is largely determined by the key itself, before it is ever
queried. Eviction follows directly: rank cached tokens by key norm, keep the
lowest-norm ones. This is the repo's first **intrinsic** eviction scorer:
every other eviction method scores with attention / a key-as-query proxy
(SnapKV, H2O, TOVA, PyramidKV, SqueezeAttention, ChunkKV, CaM) or pure
structure (StreamingLLM, sink, sliding-window). Note the sign inversion vs
ChunkKV's ``key_norm`` option and ZipCache's saliency proxy, which treat
*high* norm as important — the inversion is the paper's empirical content.

Path-independence invariant (``recent=0``): the norm is computed once at
insertion and never updated, so evicting the worst-scoring non-sink token
whenever over budget is the classic "keep k best with a heap" algorithm —
the final kept set equals the global budget-best over all tokens seen,
regardless of how arrivals were grouped. Prefill-in-one-block and
token-by-token decode produce bit-for-bit identical kept sets. No
accumulating-score method (H2O, TOVA) has this property. (``recent > 0``
breaks it: the protected window moves with time.)

Adaptation limitations (stated plainly):
  - The low-norm ⇒ high-attention correlation is the paper's empirical claim
    about trained models; nothing here validates it on synthetic data (see
    the benchmark's isotropic control, where the method shows no advantage).
  - No RoPE position-ID *renumbering* after eviction — surviving tokens keep
    their original absolute positions (``knorm_update`` restores temporal
    order after top-k selection), so the cache layer reports the true token
    position for RoPE rather than the retained row count (see
    ``cache/knorm_cache.py`` and :issue:`171`, :issue:`174`).
  - Uniform budget and n_sink across all heads.
  - ``recent`` (trailing protected window) is an extension, off by default.

Public API (mirrors quantizers/h2o.py)
--------------------------------------
KnormState           — immutable per-head state dataclass
init_knorm_state     — construct empty state (validates budget vs guards)
knorm_update         — absorb a whole [S, D] block, evict if over budget
knorm_get_kv         — extract current (keys, values) arrays
knorm_fp16_bytes     — bytes stored in current state
full_knorm_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class KnormState:
    """Per-head L2-norm eviction state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        norms:  [n_kept] float32 L2 norm of each kept key row — computed once
                at insertion, never updated (the intrinsic-score property).
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens kept at any time (including sinks).
        recent: Trailing protected window (0 = off, paper-faithful).
        keep:   "low" = keep lowest-norm tokens (the paper's finding);
                "high" = inverted selection (the benchmark's ablation arm).
    """

    keys: mx.array | None
    values: mx.array | None
    norms: mx.array | None
    n_sink: int
    budget: int
    recent: int
    keep: str


def init_knorm_state(
    n_sink: int,
    budget: int,
    head_dim: int,  # noqa: ARG001 — accepted for API symmetry with init_h2o_state
    recent: int = 0,
    keep: str = "low",
) -> KnormState:
    """Create an empty KnormState before any tokens arrive.

    Raises:
        ValueError: if ``keep`` is invalid or the protected positions
            (sinks + recent) leave no evictable room within the budget.
    """
    if keep not in ("low", "high"):
        raise ValueError(f"knorm: keep must be 'low' or 'high', got {keep!r}")
    if n_sink + recent >= budget:
        raise ValueError(
            f"knorm: n_sink ({n_sink}) + recent ({recent}) must be < "
            f"budget ({budget}) — no evictable positions remain"
        )
    return KnormState(
        keys=None,
        values=None,
        norms=None,
        n_sink=n_sink,
        budget=budget,
        recent=recent,
        keep=keep,
    )


def knorm_update(
    state: KnormState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> KnormState:
    """Absorb a whole block of S tokens, then evict down to budget in one shot.

    Vectorized — no per-token loop: norms are intrinsic (never updated), so
    the over-budget case is a single protected top-k selection. Kept tokens
    are returned in original temporal order.
    """
    new_norms = mx.sqrt(mx.sum(new_keys.astype(mx.float32) ** 2, axis=-1))  # [S]

    if state.keys is None:
        keys_cat = new_keys.astype(mx.float16)
        values_cat = new_values.astype(mx.float16)
        norms_cat = new_norms
    else:
        keys_cat = mx.concatenate([state.keys, new_keys.astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, new_values.astype(mx.float16)], axis=0)
        norms_cat = mx.concatenate([state.norms, new_norms], axis=0)

    n_total = int(keys_cat.shape[0])
    if n_total > state.budget:
        # Unify both modes to "keep the lowest scores": negate norms for
        # keep="high", then force protected positions to -inf so they always
        # survive the ascending top-k.
        scores = norms_cat
        if state.keep == "high":
            scores = -scores

        n_sink_eff = min(state.n_sink, n_total)
        protect = mx.zeros((n_total,), dtype=mx.float32)
        if n_sink_eff > 0:
            protect[:n_sink_eff] = float("-inf")
        if state.recent > 0:
            r_eff = min(state.recent, n_total - n_sink_eff)
            if r_eff > 0:
                protect[n_total - r_eff :] = float("-inf")
        scores = scores + protect

        order = mx.argsort(scores)  # ascending; protected first
        keep_idx = mx.sort(order[: state.budget])  # restore temporal order
        keys_cat = keys_cat[keep_idx]
        values_cat = values_cat[keep_idx]
        norms_cat = norms_cat[keep_idx]

    return KnormState(
        keys=keys_cat,
        values=values_cat,
        norms=norms_cat,
        n_sink=state.n_sink,
        budget=state.budget,
        recent=state.recent,
        keep=state.keep,
    )


def knorm_get_kv(state: KnormState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first
    update (same contract as ``h2o_get_kv``).
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def knorm_fp16_bytes(state: KnormState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_knorm_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2


__all__ = [
    "KnormState",
    "init_knorm_state",
    "knorm_update",
    "knorm_get_kv",
    "knorm_fp16_bytes",
    "full_knorm_fp16_bytes",
]
