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
    positions: mx.array | None = None,  # [n] int32, parallel to state.keys
    new_positions: mx.array | None = None,  # [S] int32, parallel to new_keys
) -> tuple[TovaState, mx.array | None]:
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
        positions:  Optional ``[n]`` int32 true absolute positions parallel to
            ``state.keys``. Must be ``None`` iff ``state.keys`` is ``None``
            (mirrors K/V's own bootstrap contract) while ``new_positions``
            is given. Gathered under the exact same ``evict_idx``/
            ``keep_indices`` decision as K/V (see VeloxQuant-MLX#370), so
            it is always consistent with whichever rows this function
            actually kept — never independently re-derived.
        new_positions: Optional ``[S]`` int32 true absolute positions
            parallel to ``new_keys``. ``None`` disables position tracking
            entirely (the default).

    Returns:
        ``(state, positions_out)`` — ``positions_out`` is ``None`` iff
        ``new_positions`` was ``None``.
    """
    S = new_keys.shape[0]
    track = new_positions is not None
    if track and (positions is None) != (state.keys is None):
        raise ValueError("tova: positions must be given iff state.keys is given")

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]
        p_i = new_positions[i : i + 1] if track else None

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = TovaState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                n_sink=state.n_sink,
                budget=state.budget,
            )
            if track:
                positions = p_i
            continue

        # --- append new token ----------------------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        if track:
            positions = mx.concatenate([positions, p_i], axis=0)

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
            if track:
                positions = positions[keep_indices]

        state = TovaState(
            keys=keys_cat,
            values=values_cat,
            n_sink=state.n_sink,
            budget=state.budget,
        )

    return state, (positions if track else None)


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


def _evict_mlx_virtual_values(
    keys, values_old, values_new, weights, n_sink, positions_old=None, position_new=None
):
    """GPU-only selection with V read from retained/new virtual sources.

    ``positions_old``/``position_new`` (see VeloxQuant-MLX#370): optional
    parallel true-absolute-position bookkeeping, gathered under the exact
    same ``source``/``evicted`` indices as K/V so the surviving rows' true
    positions stay in lockstep with the surviving K/V rows. ``None`` for
    either disables position tracking (returns ``None``) — used by callers
    that don't need it (e.g. quantizer-only unit tests / ``tova_update``'s
    single-head convenience wrapper).
    """
    n = keys.shape[1]
    source = _eviction_source_row(weights, n_sink)
    source_k = source[..., None]
    keys_out = mx.take_along_axis(keys, source_k, axis=1)
    old_n = n - 1
    old_source = mx.minimum(source, old_n - 1)
    old_v = mx.take_along_axis(values_old, old_source[..., None], axis=1)
    new_v = values_new[:, None, :]
    values_out = mx.where((source == old_n)[..., None], new_v, old_v)
    positions_out = None
    if positions_old is not None and position_new is not None:
        old_p = mx.take_along_axis(positions_old, old_source, axis=1)
        new_p = position_new[:, None]
        positions_out = mx.where(source == old_n, new_p, old_p)
    return keys_out, values_out, positions_out


def _eviction_source_row(weights, n_sink):
    """The ``[BH, N-1]`` source-row gather index shared by every eviction
    step, regardless of which kernel (mlx or metal) is used to compact K/V.

    Computing this once in pure MLX — cheap, an argmin over ``[BH, N]`` — and
    reusing it to gather ``positions`` independently of the K/V compaction
    kernel is what lets position tracking (#370) stay bit-for-bit consistent
    with the metal backend's eviction decisions without needing the metal
    kernels themselves to expose ``evicted`` or know about positions at all:
    the metal kernels compute this exact same value internally from the same
    ``weights``/``n_sink`` inputs (see ``src/tova_evict_reduce.metal``).
    """
    n = weights.shape[1]
    protected = mx.where(mx.arange(n)[None] < n_sink, float("inf"), weights)
    evicted = mx.argmin(protected, axis=-1, keepdims=True)
    rows = mx.arange(n - 1)[None]
    return rows + (rows >= evicted)


def _evict_mlx_indices(keys, lineage, weights, n_sink):
    """GPU-only K and lineage compaction for deferred V."""
    source = _eviction_source_row(weights, n_sink)
    return (
        mx.take_along_axis(keys, source[..., None], axis=1),
        mx.take_along_axis(lineage, source, axis=1),
    )


def _tova_update_batched(
    keys,
    values,
    new_keys,
    new_values,
    n_sink,
    budget,
    *,
    backend="auto",
    positions=None,
    new_positions=None,
):
    """Internal [BH,S,D] update; token steps remain sequential after fill.

    The incoming proxy keeps its original precision, matching direct quantizer
    callers. K/V storage is FP16. No device-derived Python control flow occurs.

    Args:
        positions: ``[BH, n]`` int32 true absolute positions of the
            currently-stored rows (parallel to ``keys``/``values``), or
            ``None`` iff ``keys`` is also ``None`` (nothing stored yet —
            mirrors K/V's own bootstrap contract) while still tracking
            positions via ``new_positions``. Ignored (may be omitted) when
            ``new_positions`` is ``None`` — the default, paying no extra
            cost for existing callers that don't need it.
        new_positions: ``[BH, S]`` int32 true absolute positions of the
            incoming rows. ``None`` disables position tracking entirely
            (the default). Non-``None`` is what actually switches tracking
            on — used by ``TOVAKVCache`` to build an explicit attention
            mask post-eviction instead of relying on mlx_lm's
            ``mask="causal"`` shortcut (see VeloxQuant-MLX#370) — TOVA's
            surviving rows are a non-contiguous subset of original
            positions (no renumbering), so their true positions must be
            tracked through eviction exactly like K/V are.

    Returns:
        ``(keys, values)`` if ``new_positions`` is ``None``, else
        ``(keys, values, positions_out)`` with ``positions_out`` the true
        absolute positions of the surviving rows, gathered under the same
        eviction indices as K/V at every step.
    """
    track_positions = new_positions is not None
    if track_positions and (positions is None) != (keys is None):
        raise ValueError("tova: positions must be given iff keys is given (both None or both set)")
    if new_keys.ndim != 3 or new_keys.shape != new_values.shape:
        raise ValueError("tova: new K/V must have matching [BH,S,D] shapes")
    bh, s, d = new_keys.shape
    backend = _resolve_backend(backend, n_tokens=s)
    if bh < 1 or d < 1:
        raise ValueError("tova: batch-head count and head dimension must be positive")
    if keys is not None and (keys.shape != values.shape or keys.shape[::2] != (bh, d)):
        raise ValueError("tova: stored K/V shape does not match incoming batch/heads/dim")
    if s == 0:
        if track_positions:
            return keys, values, positions
        return keys, values
    # Retain the original zero/negative-budget bootstrap behavior and support
    # explicitly overfull states without inventing a new eviction policy.
    if backend == "reference" or budget <= 0 or n_sink < 0:
        results = [
            _tova_update_reference(
                TovaState(
                    None if keys is None else keys[h],
                    None if values is None else values[h],
                    n_sink,
                    budget,
                ),
                new_keys[h],
                new_values[h],
                None if positions is None else positions[h],
                None if new_positions is None else new_positions[h],
            )
            for h in range(bh)
        ]
        out_k = mx.stack([st.keys for st, _ in results])
        out_v = mx.stack([st.values for st, _ in results])
        if not track_positions:
            return out_k, out_v
        return out_k, out_v, mx.stack([pos for _, pos in results])
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
    if track_positions and prefix:
        positions = (
            new_positions[:, :prefix]
            if positions is None
            else mx.concatenate([positions, new_positions[:, :prefix]], axis=1)
        )
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
    # Positions ride the exact same "append then compact under the eviction
    # argmin" chain as K, whether or not the deferred lineage path is active
    # — a plain parallel array, gathered with the same `source`/`evicted`
    # indices at every step (see _evict_mlx_virtual_values's positions_old/
    # position_new and _evict_mlx_indices's lineage, which position tracking
    # here reuses unmodified for the deferred path).
    for i in range(prefix, s):
        keys = mx.concatenate([keys, new_keys[:, i : i + 1].astype(mx.float16)], axis=1)
        proxy = new_keys[:, i].astype(mx.float32)
        logits = (keys.astype(mx.float32) @ proxy[..., None])[..., 0] * (1.0 / math.sqrt(float(d)))
        weights = mx.softmax(logits, axis=-1)
        if deferred:
            incoming_id = mx.full((bh, 1), n + i, dtype=mx.int32)
            lineage = mx.concatenate([lineage, incoming_id], axis=1)
            if track_positions:
                positions = mx.concatenate([positions, new_positions[:, i : i + 1]], axis=1)
                # Computed once, in pure MLX, from `weights`/`n_sink` alone —
                # bit-for-bit the same decision the metal kernel below makes
                # internally (see _eviction_source_row's docstring) — so
                # positions stay consistent with the metal path's K/V without
                # needing the kernel to expose its internal `evicted` index.
                source = _eviction_source_row(weights, n_sink)
                positions = mx.take_along_axis(positions, source, axis=1)
            if backend == "metal":
                from veloxquant_mlx.metal import tova_fused_evict_indices

                keys, lineage = tova_fused_evict_indices(keys, lineage, weights, n_sink)
            else:
                keys, lineage = _evict_mlx_indices(keys, lineage, weights, n_sink)
        else:
            incoming_v = new_values[:, i].astype(mx.float16)
            if track_positions:
                source = _eviction_source_row(weights, n_sink)
                old_n = source.shape[1]
                old_source = mx.minimum(source, old_n - 1)
                old_p = mx.take_along_axis(positions, old_source, axis=1)
                new_p = new_positions[:, i][:, None]
                positions = mx.where(source == old_n, new_p, old_p)
            if backend == "metal":
                from veloxquant_mlx.metal import tova_fused_evict_virtual_values

                keys, values = tova_fused_evict_virtual_values(
                    keys, values, incoming_v, weights, n_sink
                )
            else:
                keys, values, _ = _evict_mlx_virtual_values(
                    keys, values, incoming_v, weights, n_sink
                )
        if (i - prefix + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(keys, lineage if deferred else values)
    if deferred:
        values = mx.take_along_axis(source_values, lineage[..., None], axis=1)
    if track_positions:
        return keys, values, positions
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
