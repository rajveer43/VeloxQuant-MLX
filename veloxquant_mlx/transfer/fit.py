"""Offline fitting of a cross-model KV mapper (paper §3.1–§3.2).

Three pieces, in the order the paper composes them:

  1. :func:`fit_ridge` — the closed-form per-head solve
     ``W = (XᵀX + λI)⁻¹ XᵀY`` on centered data.
  2. :func:`layer_r2` — single-source ``R²`` probe used to rank source layers.
  3. :func:`select_source_layers` — top-``k`` selection per target layer.

:func:`fit_mapper` drives all three over a calibration corpus.

**Cost warning.** The paper reports ~47–87 min per pair on an 8×H100 node with
500×1024-token sequences. On Apple Silicon, with both models resident, expect
this to be the dominant cost of using this subsystem — it is an offline,
once-per-pair job, not something to run at serve time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import mlx.core as mx

from veloxquant_mlx.transfer.mapper import (
    CrossModelMapper,
    LayerMap,
    MapperConfig,
    ModelKVSpec,
)
from veloxquant_mlx.transfer.rope import strip_rope

__all__ = ["fit_ridge", "layer_r2", "select_source_layers", "fit_mapper"]


def fit_ridge(
    x: mx.array,
    y: mx.array,
    ridge_lambda: float = 0.01,
) -> tuple[mx.array, mx.array]:
    """Closed-form ridge regression, solved on centered data.

    Implements paper eq. (4), ``W* = (XᵀX + λI)⁻¹ XᵀY``, with the bias
    recovered as ``b = ȳ - x̄ W*``. Centering is what makes the intercept
    exempt from the ridge penalty — penalizing it would bias predictions
    toward zero rather than toward the target's mean.

    Solved via :func:`mx.linalg.solve` rather than forming the explicit
    inverse: same result, better conditioning. ``XᵀX`` is accumulated in
    float32 because the feature dimension reaches tens of thousands and the
    selected source layers are correlated by construction (they were chosen
    for being jointly predictive), leaving the Gram matrix near-singular —
    the regime where fp16 accumulation visibly degrades the solve.

    Args:
        x: ``[N, in_dim]`` design matrix.
        y: ``[N, out_dim]`` targets.
        ridge_lambda: Tikhonov coefficient λ.

    Returns:
        ``(W, b)`` of shapes ``[in_dim, out_dim]`` and ``[out_dim]``, float32.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x has {x.shape[0]} rows but y has {y.shape[0]}.")
    if x.shape[0] == 0:
        raise ValueError("Cannot fit ridge on zero samples.")

    x32, y32 = x.astype(mx.float32), y.astype(mx.float32)
    x_mean, y_mean = mx.mean(x32, axis=0), mx.mean(y32, axis=0)
    xc, yc = x32 - x_mean, y32 - y_mean

    in_dim = xc.shape[1]
    gram = xc.T @ xc + ridge_lambda * mx.eye(in_dim, dtype=mx.float32)
    w = mx.linalg.solve(gram, xc.T @ yc, stream=mx.cpu)
    b = y_mean - x_mean @ w
    return w, b


def _r2(y_true: mx.array, y_pred: mx.array) -> float:
    """Coefficient of determination, pooled across output dimensions.

    ``1 - SS_res / SS_tot`` with both sums taken over every element, matching
    the paper's head-averaged reporting. Returns 0.0 when the target has no
    variance at all (a constant target is perfectly predicted by the bias
    alone, so no linear structure is being credited).
    """
    yt, yp = y_true.astype(mx.float32), y_pred.astype(mx.float32)
    ss_res = mx.sum((yt - yp) ** 2)
    ss_tot = mx.sum((yt - mx.mean(yt, axis=0)) ** 2)
    if float(ss_tot) <= 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def layer_r2(
    source_feats: mx.array,
    target_feats: mx.array,
    ridge_lambda: float = 0.01,
) -> float:
    """In-sample ``R²`` of predicting one target layer from one source layer.

    This is the single-source probe of paper §2.3, used only to *rank* source
    layers. Note the paper's own §4.5 finding: ``R²`` does not predict
    downstream retention across model pairs (r = −0.20, versus +0.57 for
    attention-output cosine). It remains useful *within* a pair for source
    selection, which is exactly and only what it is used for here.

    Args:
        source_feats: ``[N, in_dim]`` source-layer features.
        target_feats: ``[N, out_dim]`` target-layer features.
        ridge_lambda: Tikhonov coefficient.

    Returns:
        In-sample ``R²``.
    """
    w, b = fit_ridge(source_feats, target_feats, ridge_lambda)
    return _r2(target_feats, source_feats.astype(mx.float32) @ w + b)


def select_source_layers(
    source_by_layer: Sequence[mx.array],
    target_feats: mx.array,
    k: int,
    ridge_lambda: float = 0.01,
) -> list[int]:
    """Pick the ``k`` source layers that best predict one target layer.

    Paper §3.2 uses fixed top-``k`` by single-source ``R²`` (rather than greedy
    forward selection, which its Appendix B uses for analysis) for
    tractability: single-source scoring is ``L_s`` solves, greedy is
    ``O(k·L_s)``. Both support the same conclusion that predictive information
    is spread across several source layers.

    Returned indices are sorted ascending, which fixes the concatenation order
    for both fitting and applying — the order is then recorded in the artifact,
    since applying a permuted order would misalign the weight blocks.

    Args:
        source_by_layer: One ``[N, in_dim]`` array per source layer.
        target_feats: ``[N, out_dim]`` features of the target layer.
        k: Number of source layers to keep.
        ridge_lambda: Tikhonov coefficient.

    Returns:
        Ascending list of ``min(k, len(source_by_layer))`` source layer indices.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if not source_by_layer:
        raise ValueError("No source layers supplied.")
    if k >= len(source_by_layer):
        return list(range(len(source_by_layer)))

    scores = [
        (layer_r2(feats, target_feats, ridge_lambda), idx)
        for idx, feats in enumerate(source_by_layer)
    ]
    # Rank by R² descending; ties break toward the lower layer index so the
    # selection is deterministic across runs.
    scores.sort(key=lambda pair: (-pair[0], pair[1]))
    return sorted(idx for _, idx in scores[:k])


def _flatten_heads(kv: mx.array) -> mx.array:
    """``[n_heads, N, head_dim]`` → ``[N, n_heads * head_dim]``."""
    return kv.transpose(1, 0, 2).reshape(kv.shape[1], -1)


def _fit_layer(
    target_layer: int,
    source_k: Sequence[mx.array],
    source_v: Sequence[mx.array],
    tgt_k: mx.array,
    tgt_v: mx.array,
    config: MapperConfig,
) -> LayerMap:
    """Fit one target layer's K and V maps across all of its heads.

    Source layers are selected once per target layer using the *concatenated*
    key features, then that same selection feeds both the K and V solves —
    paper §3.2 selects on the mean of RoPE-stripped-key and value ``R²`` and
    shares the choice across every head in the layer, which is what lets heads
    exchange information through the shared input block.

    Args:
        target_layer: Index of the layer being fit.
        source_k: Per-source-layer ``[n_heads, N, head_dim]`` content-space keys.
        source_v: Per-source-layer values, same shape.
        tgt_k: ``[n_heads, N, head_dim]`` target content-space keys.
        tgt_v: ``[n_heads, N, head_dim]`` target values.
        config: Fit hyperparameters.

    Returns:
        A fully-populated :class:`LayerMap`.
    """
    flat_k = [_flatten_heads(s) for s in source_k]
    flat_v = [_flatten_heads(s) for s in source_v]

    # Rank source layers on the layer's pooled key+value signal rather than a
    # single head, so one atypical head cannot drive the whole layer's choice.
    pooled_target = mx.concatenate([_flatten_heads(tgt_k), _flatten_heads(tgt_v)], axis=1)
    # strict=True: a key/value length mismatch means the caller's two caches
    # disagree on layer count, which would otherwise silently truncate the
    # candidate set and skip real source layers.
    pooled_source = [mx.concatenate([a, b], axis=1) for a, b in zip(flat_k, flat_v, strict=True)]
    chosen = select_source_layers(pooled_source, pooled_target, config.k, config.ridge_lambda)

    x_k = mx.concatenate([flat_k[i] for i in chosen], axis=1)
    x_v = mx.concatenate([flat_v[i] for i in chosen], axis=1)

    n_heads = tgt_k.shape[0]
    w_k, b_k, w_v, b_v = [], [], [], []
    r2_k_acc, r2_v_acc = 0.0, 0.0

    for h in range(n_heads):
        wk, bk = fit_ridge(x_k, tgt_k[h], config.ridge_lambda)
        wv, bv = fit_ridge(x_v, tgt_v[h], config.ridge_lambda)
        r2_k_acc += _r2(tgt_k[h], x_k @ wk + bk)
        r2_v_acc += _r2(tgt_v[h], x_v @ wv + bv)
        w_k.append(wk)
        b_k.append(bk)
        w_v.append(wv)
        b_v.append(bv)

    dtype = getattr(mx, config.dtype)
    return LayerMap(
        target_layer=target_layer,
        source_layers=chosen,
        w_k=mx.stack(w_k).astype(dtype),
        b_k=mx.stack(b_k).astype(dtype),
        w_v=mx.stack(w_v).astype(dtype),
        b_v=mx.stack(b_v).astype(dtype),
        r2_k=r2_k_acc / n_heads,
        r2_v=r2_v_acc / n_heads,
    )


def fit_mapper(
    source_kv: dict[int, tuple[mx.array, mx.array]],
    target_kv: dict[int, tuple[mx.array, mx.array]],
    source_spec: ModelKVSpec,
    target_spec: ModelKVSpec,
    positions: mx.array,
    config: MapperConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CrossModelMapper:
    """Fit a full source→target mapper from collected calibration activations.

    Takes already-collected KV rather than running the models itself: capturing
    per-layer KV from an ``mlx_lm`` model requires hooking its attention
    modules, which differs by architecture and belongs in the caller. See
    ``docs-site/docs/algorithms/cross-model-transfer.md`` for a worked example.

    Keys are stripped to content space here (paper §3.3) so the fit is
    position-free; values are used as-is, carrying no positional encoding.

    Args:
        source_kv: ``{layer: (keys, values)}`` with ``[n_kv_heads, N, head_dim]``
            arrays of post-RoPE keys and raw values from the source model.
        target_kv: Same structure for the target model, over the same tokens.
        source_spec: Source model shape/RoPE spec.
        target_spec: Target model shape/RoPE spec.
        positions: ``[N]`` absolute position of each calibration token, needed
            to strip RoPE correctly.
        config: Fit hyperparameters (defaults to :class:`MapperConfig`).
        progress: Optional ``(done, total)`` callback, called per target layer.

    Returns:
        A fitted :class:`CrossModelMapper` with all layers resident.

    Raises:
        MatchedKVError: If the pair violates the matched-KV precondition.
    """
    source_spec.validate_transfer_to(target_spec)
    cfg = config or MapperConfig()

    src_layers = sorted(source_kv)
    tgt_layers = sorted(target_kv)
    if not src_layers or not tgt_layers:
        raise ValueError("Both source_kv and target_kv must be non-empty.")

    n_tokens = next(iter(source_kv.values()))[0].shape[1]
    if int(positions.shape[0]) != n_tokens:
        raise ValueError(
            f"positions has {int(positions.shape[0])} entries but the "
            f"calibration activations have {n_tokens} tokens."
        )

    # Strip RoPE once per source layer, not once per target layer — with k
    # source layers feeding each of L_t targets this is the difference between
    # L_s and k*L_t strips.
    if cfg.content_space:
        src_k = [strip_rope(source_kv[i][0], positions, source_spec.rope_theta) for i in src_layers]
        tgt_k_all = [
            strip_rope(target_kv[i][0], positions, target_spec.rope_theta) for i in tgt_layers
        ]
    else:
        src_k = [source_kv[i][0] for i in src_layers]
        tgt_k_all = [target_kv[i][0] for i in tgt_layers]
    src_v = [source_kv[i][1] for i in src_layers]

    layers: list[LayerMap] = []
    for pos, tl in enumerate(tgt_layers):
        layers.append(_fit_layer(tl, src_k, src_v, tgt_k_all[pos], target_kv[tl][1], cfg))
        if progress is not None:
            progress(pos + 1, len(tgt_layers))

    return CrossModelMapper(source=source_spec, target=target_spec, config=cfg, layers=layers)
