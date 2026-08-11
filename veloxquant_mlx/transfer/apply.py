"""Apply a fitted mapper to a source KV cache (paper §3, inference path).

This is the path that replaces the target's prefill. Per target layer it does

    K̂_t = (strip_rope(K_s) · W_K + b_K) · target_rope
    V̂_t =  V_s            · W_V + b_V

as one batched matmul per layer over all heads, then hands the result back as
arrays the caller can install into the target model's cache.

The paper measures 2.7–25× versus re-prefill. That ratio comes from replacing
the target's whole transformer forward pass with a matmul, so it widens as the
target grows and as the sequence lengthens. It is a *compute* saving only —
the mapped cache is exactly as large as the target's own would have been.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from veloxquant_mlx.transfer.mapper import CrossModelMapper
from veloxquant_mlx.transfer.rope import apply_rope, strip_rope

__all__ = ["load_mapper", "transfer_layer", "transfer_cache"]


def load_mapper(path: str | Path) -> CrossModelMapper:
    """Load a mapper saved by :meth:`CrossModelMapper.save`.

    Weights are read lazily per layer on first use, so this is cheap even for
    a multi-GB artifact.
    """
    return CrossModelMapper.load(path)


def _stack_sources(
    source_kv: dict[int, tuple[mx.array, mx.array]],
    source_layers: list,
    positions: mx.array,
    rope_theta: float,
    content_space: bool,
) -> tuple[mx.array, mx.array]:
    """Build the ``[N, k*n_heads*head_dim]`` design rows for one target layer.

    Concatenation follows ``source_layers`` order exactly, matching how
    :func:`~veloxquant_mlx.transfer.fit._fit_layer` built its design matrix —
    the fitted weight blocks are position-dependent, so a different order here
    would quietly produce wrong keys rather than an error.
    """
    ks, vs = [], []
    for li in source_layers:
        k, v = source_kv[li]
        if content_space:
            k = strip_rope(k, positions, rope_theta)
        # [n_heads, N, d] -> [N, n_heads*d]
        ks.append(k.transpose(1, 0, 2).reshape(k.shape[1], -1))
        vs.append(v.transpose(1, 0, 2).reshape(v.shape[1], -1))
    return mx.concatenate(ks, axis=1), mx.concatenate(vs, axis=1)


def transfer_layer(
    mapper: CrossModelMapper,
    target_layer: int,
    source_kv: dict[int, tuple[mx.array, mx.array]],
    positions: mx.array,
) -> tuple[mx.array, mx.array]:
    """Map the source cache into one target layer's keys and values.

    Args:
        mapper: Fitted mapper.
        target_layer: Which target layer to produce.
        source_kv: ``{layer: (keys, values)}``, ``[n_kv_heads, N, head_dim]``,
            keys post-RoPE as stored by the source model.
        positions: ``[N]`` absolute positions of the cached tokens.

    Returns:
        ``(keys, values)`` for the target layer, keys carrying the *target's*
        RoPE, ready to install into the target's cache.
    """
    lm = mapper.load_layer(target_layer)
    missing = [li for li in lm.source_layers if li not in source_kv]
    if missing:
        raise KeyError(
            f"Target layer {target_layer} was fit against source layers "
            f"{lm.source_layers}, but the supplied cache is missing {missing}."
        )

    cfg = mapper.config
    x_k, x_v = _stack_sources(
        source_kv, lm.source_layers, positions, mapper.source.rope_theta, cfg.content_space
    )

    # One batched matmul over heads: [H,N,in] @ [H,in,d] -> [H,N,d]. Accumulate
    # in float32 — in_dim is k*n_heads*head_dim (thousands), long enough for
    # fp16 accumulation error to be visible in the mapped keys.
    n_heads = lm.w_k.shape[0]
    xk_b = mx.broadcast_to(x_k[None].astype(mx.float32), (n_heads, *x_k.shape))
    xv_b = mx.broadcast_to(x_v[None].astype(mx.float32), (n_heads, *x_v.shape))

    keys = xk_b @ lm.w_k.astype(mx.float32) + lm.b_k.astype(mx.float32)[:, None, :]
    values = xv_b @ lm.w_v.astype(mx.float32) + lm.b_v.astype(mx.float32)[:, None, :]

    if cfg.content_space:
        keys = apply_rope(keys, positions, mapper.target.rope_theta)

    out_dtype = source_kv[lm.source_layers[0]][0].dtype
    return keys.astype(out_dtype), values.astype(out_dtype)


def transfer_cache(
    source_kv: dict[int, tuple[mx.array, mx.array]],
    mapper: CrossModelMapper,
    positions: mx.array | None = None,
    unload_after: bool = True,
) -> dict[int, tuple[mx.array, mx.array]]:
    """Map a whole source KV cache into the target model's layout.

    Args:
        source_kv: ``{layer: (keys, values)}`` from the source model, each
            ``[n_kv_heads, N, head_dim]``, keys post-RoPE.
        mapper: Fitted mapper for this exact model pair.
        positions: ``[N]`` absolute positions. Defaults to ``0..N-1``, correct
            for a cache prefilled from the start of a sequence.
        unload_after: Drop each layer's weights once it has been applied.
            Keeps peak memory at one layer's weights instead of the whole
            multi-GB mapper — the reason this subsystem is usable on Apple
            Silicon at all. Set False when mapping many caches in a row.

    Returns:
        ``{target_layer: (keys, values)}`` ready for the target model.
    """
    if not source_kv:
        raise ValueError("source_kv is empty; nothing to transfer.")

    n_tokens = next(iter(source_kv.values()))[0].shape[1]
    if positions is None:
        positions = mx.arange(n_tokens)
    elif int(positions.shape[0]) != n_tokens:
        raise ValueError(
            f"positions has {int(positions.shape[0])} entries but the cache "
            f"holds {n_tokens} tokens."
        )

    out: dict[int, tuple[mx.array, mx.array]] = {}
    for lm in mapper.layers:
        out[lm.target_layer] = transfer_layer(mapper, lm.target_layer, source_kv, positions)
        if unload_after:
            lm.unload()
    return out
