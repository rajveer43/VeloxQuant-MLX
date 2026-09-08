"""TOVA (Token Omission Via Attention) KV eviction primitives.

Inspired by "Transformers are Multi-State RNNs" (Oren et al., 2024,
arXiv:2401.06104), whose TOVA policy retains a fixed-size cache by dropping, at
each step, the single token with the lowest attention weight in the *current*
step. Documented as "TOVA-adapted (VeloxQuant-MLX implementation)" — not a
faithful port. See adaptation limitations below and in cache/tova_cache.py.

The distinguishing feature versus H2O-adapted: TOVA is *memoryless*. It scores
tokens by the attention weight received at the **current** step only, with no
running accumulation. H2O keeps a cumulative sum (inertial — a token that was a
heavy hitter long ago survives); TOVA reacts instantly to the present context (a
token that stops being attended to is evicted even if it dominated earlier).

Adaptation limitations (stated plainly):
  - Key-as-query proxy: the paper reads the actual attention distribution of the
    most recent query row from the forward pass. At cache level the query is not
    visible, so we use the incoming key vector as a proxy query to approximate
    the current-step attention distribution. Same approximation as SnapKV-adapted
    and H2O-adapted.
  - No RoPE position-ID *renumbering* after eviction — the evicted row is
    dropped and the rest kept in temporal order, so surviving tokens keep
    their original absolute positions and the cache layer reports the true
    token position for RoPE rather than the retained row count (see
    ``cache/tova_cache.py`` and :issue:`171`, :issue:`175`).
  - Uniform budget across all heads.
  - When a multi-token chunk (S > 1) arrives, tokens are absorbed one at a time
    and the last incoming key of each step is the proxy query. This differs from
    the paper's single-step decode framing but keeps budget semantics identical.

Public API
----------
TovaState          — immutable per-head state dataclass
init_tova_state    — construct empty state
tova_update        — absorb S new tokens, evict lowest current-step-weight token
tova_get_kv        — extract current (keys, values) arrays
tova_fp16_bytes    — bytes stored in current state
full_tova_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TovaState:
    """Per-head sliding TOVA state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens to keep at any time (including sinks).

    Note:
        Unlike H2OState there is no ``scores`` field — TOVA is memoryless and
        recomputes the current-step attention weights fresh on every update,
        discarding them after the eviction decision.
    """

    keys: mx.array | None
    values: mx.array | None
    n_sink: int
    budget: int


def init_tova_state(n_sink: int, budget: int, head_dim: int) -> TovaState:  # noqa: ARG001
    """Create an empty TovaState before any tokens arrive.

    Args:
        n_sink:   Number of initial sink positions to protect from eviction.
        budget:   Maximum total tokens kept (sinks + non-sinks).
        head_dim: Head dimension D (unused here; accepted for API symmetry
                  with H2O's init_h2o_state and StreamingLLM's init).

    Raises:
        ValueError: if there are sink positions to protect but they leave no
            evictable room within ``budget`` (``n_sink=0, budget=0`` remains
            a valid "disabled cache" configuration).
    """
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"tova: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable positions remain, so sinks would be evicted once "
            "the cache fills"
        )
    return TovaState(keys=None, values=None, n_sink=n_sink, budget=budget)


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


def _tova_update_reference(
    state: TovaState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> TovaState:
    """Absorb S new tokens into state, evicting the lowest current-step-weight token if over budget.

    For each of the S incoming tokens:
      1. Append the new token to the cache.
      2. If total tokens > budget: compute the current-step attention weights of
         the newly appended key (as proxy query) over *all* rows including
         itself, protect the first ``n_sink`` positions with +inf, and evict the
         non-sink token with the lowest current-step weight.

    Unlike H2O, no per-token score is carried across steps — the weight vector is
    computed fresh each step for the eviction decision and then discarded.

    Args:
        state:      Current TovaState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated TovaState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = TovaState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                n_sink=state.n_sink,
                budget=state.budget,
            )
            continue

        # --- append new token ----------------------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)

        n_total = keys_cat.shape[0]

        if n_total > state.budget:
            # Current-step attention weights of the new key over ALL rows.
            weights = _attention_scores(
                k_i.astype(mx.float32), keys_cat.astype(mx.float32)
            )  # [n_total]

            # Build eviction-protected weight view: sinks get +inf.
            n_sink_eff = min(state.n_sink, n_total)
            if n_sink_eff > 0:
                inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
                protected = mx.concatenate([inf_block, weights[n_sink_eff:]], axis=0)
            else:
                protected = weights

            evict_idx = int(mx.argmin(protected).item())
            keep_indices = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep_indices]
            values_cat = values_cat[keep_indices]

        state = TovaState(
            keys=keys_cat,
            values=values_cat,
            n_sink=state.n_sink,
            budget=state.budget,
        )

    return state


_EVAL_FLUSH_INTERVAL = 32
_BACKENDS = ("auto", "mlx", "metal", "reference")


def _resolve_backend(backend, *, n_tokens=1):
    if backend not in _BACKENDS:
        raise ValueError(f"tova: backend must be one of {_BACKENDS}")
    # Multi-token graphs consistently benefit from the two-dispatch path on
    # the measured M4. Synchronized single-token gains are smaller/noisy.
    if backend == "auto":
        if n_tokens > 1 and mx.default_device() == mx.gpu:
            from veloxquant_mlx.metal import metal_available

            if metal_available():
                return "metal"
        return "mlx"
    if backend == "metal":
        from veloxquant_mlx.metal import metal_available

        if mx.default_device() != mx.gpu or not metal_available():
            raise ValueError("tova: forced Metal requires an available default GPU device")
    return backend


def _evict_mlx(keys, values, weights, n_sink):
    """GPU-only argmin and fixed-size ordered gather on [BH,N,D]."""
    n = keys.shape[1]
    protected = mx.where(mx.arange(n)[None] < n_sink, float("inf"), weights)
    evicted = mx.argmin(protected, axis=-1, keepdims=True)
    rows = mx.arange(n - 1)[None]
    source = (rows + (rows >= evicted))[..., None]
    return (
        mx.take_along_axis(keys, source, axis=1),
        mx.take_along_axis(values, source, axis=1),
    )


def _evict_mlx_virtual_values(keys, values_old, values_new, weights, n_sink):
    """GPU-only selection with V read from retained/new virtual sources."""
    n = keys.shape[1]
    protected = mx.where(mx.arange(n)[None] < n_sink, float("inf"), weights)
    evicted = mx.argmin(protected, axis=-1, keepdims=True)
    rows = mx.arange(n - 1)[None]
    source = rows + (rows >= evicted)
    source_k = source[..., None]
    keys_out = mx.take_along_axis(keys, source_k, axis=1)
    old_n = n - 1
    old_source = mx.minimum(source, old_n - 1)
    old_v = mx.take_along_axis(values_old, old_source[..., None], axis=1)
    new_v = values_new[:, None, :]
    return keys_out, mx.where((source == old_n)[..., None], new_v, old_v)


def _evict_mlx_indices(keys, lineage, weights, n_sink):
    """GPU-only K and lineage compaction for deferred V."""
    n = keys.shape[1]
    protected = mx.where(mx.arange(n)[None] < n_sink, float("inf"), weights)
    evicted = mx.argmin(protected, axis=-1, keepdims=True)
    rows = mx.arange(n - 1)[None]
    source = rows + (rows >= evicted)
    return (
        mx.take_along_axis(keys, source[..., None], axis=1),
        mx.take_along_axis(lineage, source, axis=1),
    )


def _tova_update_batched(keys, values, new_keys, new_values, n_sink, budget, *, backend="auto"):
    """Internal [BH,S,D] update; token steps remain sequential after fill.

    The incoming proxy keeps its original precision, matching direct quantizer
    callers. K/V storage is FP16. No device-derived Python control flow occurs.
    """
    if new_keys.ndim != 3 or new_keys.shape != new_values.shape:
        raise ValueError("tova: new K/V must have matching [BH,S,D] shapes")
    bh, s, d = new_keys.shape
    backend = _resolve_backend(backend, n_tokens=s)
    if bh < 1 or d < 1:
        raise ValueError("tova: batch-head count and head dimension must be positive")
    if keys is not None and (keys.shape != values.shape or keys.shape[::2] != (bh, d)):
        raise ValueError("tova: stored K/V shape does not match incoming batch/heads/dim")
    if s == 0:
        return keys, values
    # Retain the original zero/negative-budget bootstrap behavior and support
    # explicitly overfull states without inventing a new eviction policy.
    if backend == "reference" or budget <= 0 or n_sink < 0:
        states = [
            _tova_update_reference(
                TovaState(
                    None if keys is None else keys[h],
                    None if values is None else values[h],
                    n_sink,
                    budget,
                ),
                new_keys[h],
                new_values[h],
            )
            for h in range(bh)
        ]
        return mx.stack([st.keys for st in states]), mx.stack([st.values for st in states])
    if n_sink >= budget:
        raise ValueError("tova: sinks must leave at least one evictable position")
    n = 0 if keys is None else keys.shape[1]
    prefix = min(s, max(0, budget - n))
    # Deferred lineage pays one final V gather and is beneficial once enough
    # independent KV groups amortize that work. Keep the virtual-V path for
    # small G, where the extra map traffic is measurable overhead.
    deferred = backend in ("mlx", "metal") and prefix < s and bh >= 4
    source_values = None
    lineage = None
    if deferred:
        source_values = new_values if keys is None else mx.concatenate([values, new_values], axis=1)
        lineage = mx.broadcast_to(mx.arange(n, dtype=mx.int32)[None], (bh, n))
    if prefix:
        k = new_keys[:, :prefix].astype(mx.float16)
        v = new_values[:, :prefix].astype(mx.float16)
        keys = k if keys is None else mx.concatenate([keys, k], axis=1)
        values = v if values is None else mx.concatenate([values, v], axis=1)
        if deferred:
            prefix_ids = mx.broadcast_to(
                mx.arange(n, n + prefix, dtype=mx.int32)[None], (bh, prefix)
            )
            lineage = mx.concatenate([lineage, prefix_ids], axis=1)
    if deferred:
        # Values are not touched during the dependent eviction chain. The
        # immutable source buffer is gathered exactly once after all decisions.
        values = None
    for i in range(prefix, s):
        keys = mx.concatenate([keys, new_keys[:, i : i + 1].astype(mx.float16)], axis=1)
        proxy = new_keys[:, i].astype(mx.float32)
        logits = (keys.astype(mx.float32) @ proxy[..., None])[..., 0] * (1.0 / math.sqrt(float(d)))
        weights = mx.softmax(logits, axis=-1)
        if deferred:
            incoming_id = mx.full((bh, 1), n + i, dtype=mx.int32)
            lineage = mx.concatenate([lineage, incoming_id], axis=1)
            if backend == "metal":
                from veloxquant_mlx.metal import tova_fused_evict_indices

                keys, lineage = tova_fused_evict_indices(keys, lineage, weights, n_sink)
            else:
                keys, lineage = _evict_mlx_indices(keys, lineage, weights, n_sink)
        else:
            incoming_v = new_values[:, i].astype(mx.float16)
            if backend == "metal":
                from veloxquant_mlx.metal import tova_fused_evict_virtual_values

                keys, values = tova_fused_evict_virtual_values(
                    keys, values, incoming_v, weights, n_sink
                )
            else:
                keys, values = _evict_mlx_virtual_values(keys, values, incoming_v, weights, n_sink)
        if (i - prefix + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(keys, lineage if deferred else values)
    if deferred:
        values = mx.take_along_axis(source_values, lineage[..., None], axis=1)
    return keys, values


def tova_update(
    state: TovaState,
    new_keys: mx.array,
    new_values: mx.array,
    *,
    backend: str = "auto",
) -> TovaState:
    """Absorb [S,D] K/V using auto, GPU-only mlx, metal, or reference eviction.

    Surviving rows retain their original positions and FP16 values. The
    reference backend retains the original Python-driven algorithm for parity
    and benchmarking. Auto uses the measured default (see TOVA_METAL_FINDINGS).
    """
    if new_keys.ndim != 2 or new_keys.shape != new_values.shape:
        raise ValueError("tova: new K/V must have matching [S,D] shapes")
    _resolve_backend(backend)
    if new_keys.shape[0] == 0:
        return state
    keys, values = _tova_update_batched(
        None if state.keys is None else state.keys[None],
        None if state.values is None else state.values[None],
        new_keys[None],
        new_values[None],
        state.n_sink,
        state.budget,
        backend=backend,
    )
    return TovaState(keys[0], values[0], state.n_sink, state.budget)


def tova_get_kv(state: TovaState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def tova_fp16_bytes(state: TovaState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_tova_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "TovaState",
    "init_tova_state",
    "tova_update",
    "tova_get_kv",
    "tova_fp16_bytes",
    "full_tova_fp16_bytes",
]
