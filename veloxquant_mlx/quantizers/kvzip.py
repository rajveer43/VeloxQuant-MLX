"""KVzip KV eviction primitives — context-reconstruction reliance retention.

Inspired by "KVzip: Query-Agnostic KV Cache Compression with Context
Reconstruction" (Kim, Kim, Kwon, Lee, Yun, Song, NeurIPS 2025 (Oral),
arXiv:2505.23416, official code github.com/snu-mllab/KVzip). Documented as
"KVzip-adapted (VeloxQuant-MLX implementation)" — not a faithful port. See
adaptation limitations below and in cache/kvzip_cache.py.

The paper's finding: instead of ranking a cached KV pair by the attention it
receives from queries, rank it by how much the model *relies on it to reconstruct
its own context*. The importance profile is **query-agnostic** — computed once
against a reconstruction objective, then reused across all future queries — so one
compressed cache serves diverse downstream queries. Low-reconstruction-reliance
pairs are evicted.

WHERE THIS SITS IN THE REPO
---------------------------
This is the repo's proxy-attention scorer family (SnapKV / H2O / TOVA /
PyramidKV / SqueezeAttention / ChunkKV / CaM / Keyformer / MorphKV). The
distinguishing axis: **every** existing scorer ranks a stored token by the
attention it *receives from queries* — cumulative (H2O), latest-query (TOVA /
SnapKV), or recent-window (MorphKV). KVzip-adapted is the first to rank by
**reconstruction reliance** — the attention a stored key receives from a fixed
*reconstruction probe*, not from a live query. The honest reference behavior:

  - ``probe = "latest"`` collapses onto a **latest-token (TOVA-adapted-style)**
    eviction: the reconstruction probe is the single most-recent key, so the
    importance is just that key's attention over the keep set. This is the clean,
    assertable reduction (exercised by a dedicated test), the analogue of
    MorphKV's ``window = 1`` == TOVA and Keyformer's ``tau = 0`` == H2O collapses.
  - ``probe = "context"`` (default) uses the full kept set as the reconstruction
    probe: importance is the MAX proxy-attention each stored key receives across
    all reconstruction-probe rows. This never becomes H2O's *cumulative-forever*
    rule (KVzip recomputes from the live keep set, it does not accumulate), so we
    do NOT claim an H2O collapse — only the ``probe = "latest"`` reduction is
    pinned exactly.

THE HONESTY CRUX (read before trusting any number)
--------------------------------------------------
1. **Proxy reconstruction.** A cache never runs the real model to reconstruct
   text. Like H2O / TOVA / MorphKV-adapted, the stored/incoming KEYS are used as
   proxy reconstruction queries to estimate the attention each stored key
   receives. The paper uses the model's real reconstruction forward passes.
   Documented substitution, not the paper's math.
2. **Query-agnostic, recomputed — not accumulated.** We keep no cumulative score
   array. Each step, reconstruction importance is recomputed from the current
   keep set against the probe. Query-agnostic in the paper's sense (the probe is
   not a downstream query); constant, not a growing accumulator.
3. Nothing here is validated on a trained model. The paper's headline numbers
   (3–4× reduction, ~2× decode latency, negligible loss up to 170K tokens on
   LLaMA3.1 / Qwen2.5 / Gemma3) are the paper's, on trained models — NEVER quoted
   as ours. The mechanism's benefit is measured only under a constructed
   "reconstruction-shift" geometry in the benchmark, with a null control where it
   shows no advantage.

Adaptation limitations (stated plainly):
  - Key-as-reconstruction-probe proxy (crux 1).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget / n_sink / probe across all heads.
  - Leading ``n_sink`` tokens are protected as sinks. The reconstruction probe
    (``probe="context"``: the whole keep set; ``probe="latest"``: the last row)
    drives the ranking; no trailing window is force-protected (unlike MorphKV) —
    a token survives only if the reconstruction probe relies on it.

Public API (mirrors quantizers/morphkv.py)
------------------------------------------
KVzipState        — immutable per-head state dataclass
init_kvzip_state  — construct empty state (validates guards)
kvzip_update      — absorb S new tokens, evict least reconstruction-critical if over budget
kvzip_get_kv      — extract current (keys, values) arrays
kvzip_fp16_bytes  — bytes stored in current state
full_kvzip_fp16_bytes — hypothetical cost without eviction
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

_VALID_PROBES = ("context", "latest")


@dataclass
class KVzipState:
    """Per-head KVzip eviction state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        pos:    Running count of token positions this head has ever inserted
                (diagnostic; reconstruction importance is recomputed from
                ``keys`` and the probe each step).
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens kept at any time (including sinks).
        probe:  Reconstruction probe: ``"context"`` (full kept set drives the
                reconstruction ranking) or ``"latest"`` (the single most-recent
                key — collapses to a latest-token / TOVA-adapted ranking).
        head_dim: Head dimension D (for byte accounting before first insert).
    """

    keys: mx.array | None
    values: mx.array | None
    pos: int
    n_sink: int
    budget: int
    probe: str
    head_dim: int


def init_kvzip_state(
    n_sink: int,
    budget: int,
    head_dim: int,
    probe: str = "context",
) -> KVzipState:
    """Create an empty KVzipState before any tokens arrive.

    Raises:
        ValueError: if ``budget < 1``, ``n_sink >= budget`` (no evictable room),
            or ``probe`` is not one of ``"context"`` / ``"latest"``.
    """
    if budget < 1:
        raise ValueError(f"kvzip: budget must be >= 1, got {budget!r}")
    if n_sink >= budget:
        raise ValueError(
            f"kvzip: n_sink ({n_sink}) must be < budget ({budget}) — no evictable positions remain"
        )
    if probe not in _VALID_PROBES:
        raise ValueError(f"kvzip: probe must be one of {_VALID_PROBES}, got {probe!r}")
    return KVzipState(
        keys=None,
        values=None,
        pos=0,
        n_sink=n_sink,
        budget=budget,
        probe=str(probe),
        head_dim=int(head_dim),
    )


def _reconstruction_importance(keys: mx.array, probe: str) -> mx.array:
    """Reconstruction reliance each stored key receives from the probe.

    This is the KVzip signal: "how much does reconstructing the context rely on
    this stored token." Following the paper, importance is the **maximum**
    proxy-attention a stored key receives across the reconstruction-probe rows
    (max-over-probe, so a key that is critical to *any* probe position survives).

    Probe rows:
      - ``"context"``: every stored key row is a reconstruction-probe query
        (reconstruct the whole context from itself). Importance is the max over
        the full keep set.
      - ``"latest"``: only the single most-recent key row is the probe. Importance
        is then exactly that one key's attention over the keep set — the
        latest-token (TOVA-adapted) ranking. This is the pinned reduction.

    Args:
        keys:  [n, D] stored key rows (the keep-set candidates).
        probe: ``"context"`` | ``"latest"``.

    Returns:
        [n] reconstruction-reliance score per stored key.
    """
    keys_f = keys.astype(mx.float32)
    n = int(keys_f.shape[0])

    if probe == "latest":
        # Single most-recent key as the reconstruction probe → exactly the
        # TOVA-adapted latest-token attention over the keep set.
        return attention_scores(keys_f[n - 1], keys_f)

    # probe == "context": max over all probe rows of the attention placed on
    # each stored key. Build the [n_probe, n] matrix and reduce with max(axis=0).
    scale = 1.0 / math.sqrt(float(keys_f.shape[-1]))
    logits = (keys_f @ keys_f.T) * scale  # [n_probe, n]
    attn = mx.softmax(logits, axis=-1)  # each probe row sums to ~1
    return mx.max(attn, axis=0)  # [n] max reliance per key


def kvzip_update(
    state: KVzipState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> KVzipState:
    """Absorb S new tokens, evicting the least reconstruction-critical if over budget.

    For each of the S incoming tokens:
      1. Append the new token to the cache.
      2. If over budget: rank the current keep set by reconstruction reliance
         (:func:`_reconstruction_importance` under the configured probe), force
         the leading ``n_sink`` sinks to survive (+inf), and evict the
         non-protected token with the LOWEST reconstruction reliance.
         Constant-size: ``n_kept <= budget`` after every token.

    No cumulative score is carried across steps — the ranking is recomputed fresh
    each step from the live keep set against the probe (that is the mechanism:
    query-agnostic reconstruction reliance, not accumulation). Kept tokens are
    returned in original temporal order.
    """
    S = int(new_keys.shape[0])

    for i in range(S):
        k_i = new_keys[i].astype(mx.float16)  # [D]
        v_i = new_values[i].astype(mx.float16)  # [D]

        if state.keys is None:
            state = KVzipState(
                keys=k_i[None],
                values=v_i[None],
                pos=state.pos + 1,
                n_sink=state.n_sink,
                budget=state.budget,
                probe=state.probe,
                head_dim=state.head_dim,
            )
            continue

        # --- append new token ---------------------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None]], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None]], axis=0)

        n_total = int(keys_cat.shape[0])

        if n_total > state.budget:
            importance = _reconstruction_importance(keys_cat, state.probe)  # [n_total]

            # Protect the leading sinks with +inf so they are never chosen.
            n_sink_eff = min(state.n_sink, n_total)
            protect = mx.zeros((n_total,), dtype=mx.float32)
            if n_sink_eff > 0:
                protect[:n_sink_eff] = float("inf")
            sel = importance + protect

            evict_idx = int(mx.argmin(sel).item())
            keep = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep]
            values_cat = values_cat[keep]

        state = KVzipState(
            keys=keys_cat,
            values=values_cat,
            pos=state.pos + 1,
            n_sink=state.n_sink,
            budget=state.budget,
            probe=state.probe,
            head_dim=state.head_dim,
        )

    return state


_EVAL_FLUSH_INTERVAL = 32


def _reconstruction_importance_batched(keys: mx.array, probe: str) -> mx.array:
    """Batched-``[BH,n,D]`` equivalent of :func:`_reconstruction_importance`.

    Args:
        keys:  ``[BH, n, D]`` stored key rows (the keep-set candidates).
        probe: ``"context"`` | ``"latest"``.

    Returns:
        ``[BH, n]`` reconstruction-reliance score per stored key.
    """
    keys_f = keys.astype(mx.float32)
    n = int(keys_f.shape[1])

    if probe == "latest":
        # Single most-recent key (per row) as the reconstruction probe ->
        # exactly the TOVA-adapted latest-token attention over the keep set.
        scale = 1.0 / math.sqrt(float(keys_f.shape[-1]))
        latest = keys_f[:, n - 1]  # [BH, D]
        logits = (keys_f @ latest[..., None])[..., 0] * scale  # [BH, n]
        return mx.softmax(logits, axis=-1)

    # probe == "context": max over all probe rows of the attention placed on
    # each stored key. Build the [BH, n_probe, n] matrix and reduce with
    # max(axis=1) (the probe axis).
    scale = 1.0 / math.sqrt(float(keys_f.shape[-1]))
    logits = (keys_f @ mx.swapaxes(keys_f, -1, -2)) * scale  # [BH, n_probe, n]
    attn = mx.softmax(logits, axis=-1)  # each probe row sums to ~1
    return mx.max(attn, axis=1)  # [BH, n] max reliance per key


def kvzip_update_batched(
    keys: mx.array | None,  # [BH, n, D] fp16 or None
    values: mx.array | None,  # [BH, n, D] fp16 or None
    new_keys: mx.array,  # [BH, S, D]
    new_values: mx.array,  # [BH, S, D]
    n_sink: int,
    budget: int,
    probe: str,
    *,
    return_indices: bool = False,
) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array]:
    """Vectorized-over-``BH`` equivalent of calling :func:`kvzip_update` once
    per ``(batch, head)`` pair with identical per-row ``n_sink``/``budget``/
    ``probe`` (true for every real caller: :class:`KVzipKVCache` applies one
    uniform config to every head), so the per-token append/rank/evict math —
    otherwise identical for every row — can run as one batched MLX call per
    step instead of ``BH`` separate Python-level calls into
    :func:`kvzip_update`. Same fix, same template, as
    :func:`veloxquant_mlx.quantizers.h2o.h2o_update_batched`.

    Numerically identical to the per-head loop it replaces: every op below
    is the same formula as :func:`kvzip_update`'s bootstrap/append/evict
    branches, applied over a leading ``BH`` axis instead of a Python loop.

    With ``return_indices=True``, also return indices into prior + new rows.

    Returns:
        ``(keys, values)`` — ``[BH, n_kept, D]`` each.
    """
    bh, s, d = new_keys.shape
    n_prior = 0 if keys is None else keys.shape[1]
    indices = mx.broadcast_to(mx.arange(n_prior)[None], (bh, n_prior))
    if s == 0:
        return (keys, values, indices) if return_indices else (keys, values)
    if n_sink >= budget:
        raise ValueError("kvzip: n_sink must be < budget — no evictable positions remain")

    k_dtype = new_keys.dtype
    v_dtype = new_values.dtype

    for i in range(s):
        if return_indices:
            indices = mx.concatenate([indices, mx.full((bh, 1), n_prior + i)], axis=1)
        if keys is None:
            keys = new_keys[:, i : i + 1].astype(k_dtype)  # [BH, 1, D]
            values = new_values[:, i : i + 1].astype(v_dtype)
            continue

        keys_cat = mx.concatenate([keys, new_keys[:, i : i + 1].astype(k_dtype)], axis=1)
        values_cat = mx.concatenate([values, new_values[:, i : i + 1].astype(v_dtype)], axis=1)

        n_total = keys_cat.shape[1]
        if n_total > budget:
            importance = _reconstruction_importance_batched(keys_cat, probe)  # [BH, n_total]

            n_sink_eff = min(n_sink, n_total)
            protect = mx.zeros((bh, n_total), dtype=mx.float32)
            if n_sink_eff > 0:
                sink_inf = mx.full((bh, n_sink_eff), float("inf"), dtype=mx.float32)
                protect = mx.concatenate([sink_inf, protect[:, n_sink_eff:]], axis=1)
            sel = importance + protect

            evict_idx = mx.argmin(sel, axis=-1, keepdims=True)  # [BH, 1]
            rows = mx.arange(n_total - 1)[None]  # [1, n_total-1]
            source = rows + (rows >= evict_idx)  # [BH, n_total-1]

            keys_cat = mx.take_along_axis(keys_cat, source[..., None], axis=1)
            values_cat = mx.take_along_axis(values_cat, source[..., None], axis=1)
            if return_indices:
                indices = mx.take_along_axis(indices, source, axis=1)

        keys, values = keys_cat, values_cat

        if (i + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(keys, values, indices)

    return (keys, values, indices) if return_indices else (keys, values)


def kvzip_get_kv(state: KVzipState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update
    (same contract as ``morphkv_get_kv`` / ``tova_get_kv``).
    """
    return get_kv(state.keys, state.values)


def kvzip_fp16_bytes(state: KVzipState) -> int:
    """Bytes currently stored for K + V in fp16.

    The reconstruction probe reuses ``keys`` (not extra payload), so only K + V
    are counted — same accounting as H2O / TOVA / MorphKV.
    """
    return fp16_kv_bytes(state.keys)


def full_kvzip_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return full_fp16_kv_bytes(tokens_seen, head_dim)


__all__ = [
    "KVzipState",
    "init_kvzip_state",
    "kvzip_update",
    "kvzip_update_batched",
    "kvzip_get_kv",
    "kvzip_fp16_bytes",
    "full_kvzip_fp16_bytes",
]
