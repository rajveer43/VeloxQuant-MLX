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


def _fit_ridge_batched(
    xs: mx.array,
    y: mx.array,
    ridge_lambda: float,
) -> tuple[mx.array, mx.array]:
    """Ridge regression for a batch of independent design matrices, one shared ``y``.

    Equivalent to calling :func:`fit_ridge` once per slice of ``xs`` against
    the same ``y``, but solved as a single batched ``mx.linalg.solve`` — each
    candidate's ``XᵀX + λI`` becomes one entry of the batch dimension rather
    than a separate Python-level call, which is what lets the whole ranking
    pass in :func:`select_source_layers` pay for one CPU sync instead of one
    per candidate.

    Args:
        xs: ``[B, N, in_dim]`` — ``B`` independent design matrices, same
            ``N`` and ``in_dim`` (candidates are all built by the same
            flattening logic upstream, so this always holds here).
        y: ``[N, out_dim]`` targets, shared across the batch.
        ridge_lambda: Tikhonov coefficient, shared across the batch.

    Returns:
        ``(w, b)`` of shapes ``[B, in_dim, out_dim]`` and ``[B, out_dim]``,
        float32.
    """
    xs32, y32 = xs.astype(mx.float32), y.astype(mx.float32)
    x_mean = mx.mean(xs32, axis=1, keepdims=True)
    y_mean = mx.mean(y32, axis=0)
    xc = xs32 - x_mean
    yc = y32 - y_mean

    b_dim, in_dim = xs32.shape[0], xs32.shape[2]
    gram = mx.matmul(xc.transpose(0, 2, 1), xc) + ridge_lambda * mx.eye(in_dim, dtype=mx.float32)
    xty = mx.matmul(xc.transpose(0, 2, 1), mx.broadcast_to(yc[None], (b_dim, *yc.shape)))
    w = mx.linalg.solve(gram, xty, stream=mx.cpu)
    b = y_mean[None, :] - mx.matmul(x_mean, w)[:, 0, :]
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


def _r2_batched(y_true: mx.array, y_pred: mx.array) -> list[float]:
    """Per-batch-entry :func:`_r2`, for a shared ``y_true`` broadcast over a batch.

    Args:
        y_true: ``[N, out_dim]``, shared across the batch (as in the
            single-target ranking pass — every candidate is scored against
            the same target layer).
        y_pred: ``[B, N, out_dim]`` per-candidate predictions.

    Returns:
        ``B`` pooled ``R²`` values, in the same convention as :func:`_r2`.
    """
    yt = y_true.astype(mx.float32)
    yp = y_pred.astype(mx.float32)
    ss_res = mx.sum((yt[None] - yp) ** 2, axis=(1, 2))
    ss_tot = float(mx.sum((yt - mx.mean(yt, axis=0)) ** 2))
    if ss_tot <= 0.0:
        return [0.0] * yp.shape[0]
    return (1.0 - ss_res / ss_tot).tolist()


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
    is spread across several source layers. When every candidate shares the
    same ``in_dim`` (always true for the calibration features this is called
    with — see :func:`_fit_layer`), the ``L_s`` solves are issued as a single
    batched :func:`mx.linalg.solve` instead of one Python call each, which is
    what actually determines the wall-clock cost on MLX's lazy scheduler:
    ``L_s`` separate calls each force their own CPU sync, one batched call
    forces exactly one.

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

    same_shape = all(feats.shape == source_by_layer[0].shape for feats in source_by_layer[1:])
    if same_shape:
        # One batched solve for all L_s candidates instead of L_s separate
        # CPU-stream solves — see the docstring above.
        xs = mx.stack(list(source_by_layer))
        w, b = _fit_ridge_batched(xs, target_feats, ridge_lambda)
        preds = mx.matmul(xs.astype(mx.float32), w) + b[:, None, :]
        mx.eval(preds)
        r2_values = _r2_batched(target_feats, preds)
        scores = list(enumerate(r2_values))
    else:
        scores = [
            (idx, layer_r2(feats, target_feats, ridge_lambda))
            for idx, feats in enumerate(source_by_layer)
        ]
    # Rank by R² descending; ties break toward the lower layer index so the
    # selection is deterministic across runs.
    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return sorted(idx for idx, _ in scores[:k])


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

    The K solve (and, separately, the V solve) is done once for the whole
    layer rather than once per head: ``x_k``/``x_v`` is the same design matrix
    for every head, only the target column-block differs, so all heads'
    targets are concatenated into one ``[N, n_heads*head_dim]`` RHS and fit
    with a single :func:`fit_ridge` call. This reuses one ``XᵀX`` solve
    instead of redoing an identical one ``n_heads`` times.

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

    n_heads, head_dim = tgt_k.shape[0], tgt_k.shape[2]
    # tgt_k/tgt_v are [n_heads, N, head_dim]; flatten to [N, n_heads*head_dim]
    # so every head's target rides as extra RHS columns of one ridge solve —
    # x_k/x_v (and therefore XᵀX and its factorization) is identical across
    # heads, so solving once per layer instead of once per head skips
    # redundant, identical work rather than approximating it.
    y_k = _flatten_heads(tgt_k)
    y_v = _flatten_heads(tgt_v)
    w_k_flat, b_k_flat = fit_ridge(x_k, y_k, config.ridge_lambda)
    w_v_flat, b_v_flat = fit_ridge(x_v, y_v, config.ridge_lambda)

    pred_k = x_k @ w_k_flat + b_k_flat
    pred_v = x_v @ w_v_flat + b_v_flat
    # Per-head R², matching the original per-head-then-averaged semantics
    # exactly (not a single R² pooled across all heads at once): sums are
    # taken within each head's own [N, head_dim] block before averaging.
    yk_r = y_k.reshape(y_k.shape[0], n_heads, head_dim)
    pk_r = pred_k.reshape(pred_k.shape[0], n_heads, head_dim)
    yv_r = y_v.reshape(y_v.shape[0], n_heads, head_dim)
    pv_r = pred_v.reshape(pred_v.shape[0], n_heads, head_dim)

    def _per_head_r2_mean(yt: mx.array, yp: mx.array) -> float:
        ss_res = mx.sum((yt - yp) ** 2, axis=(0, 2))
        ss_tot = mx.sum((yt - mx.mean(yt, axis=0, keepdims=True)) ** 2, axis=(0, 2))
        mx.eval(ss_res, ss_tot)
        per_head = [
            0.0 if float(tot) <= 0.0 else 1.0 - float(res) / float(tot)
            for res, tot in zip(ss_res.tolist(), ss_tot.tolist(), strict=True)
        ]
        return sum(per_head) / n_heads

    r2_k = _per_head_r2_mean(yk_r, pk_r)
    r2_v = _per_head_r2_mean(yv_r, pv_r)

    dtype = getattr(mx, config.dtype)
    # w_*_flat is [in_dim, n_heads*head_dim]; split back into per-head blocks
    # and stack to the documented [n_heads, in_dim, head_dim] contract.
    w_k = mx.stack([w_k_flat[:, h * head_dim : (h + 1) * head_dim] for h in range(n_heads)])
    w_v = mx.stack([w_v_flat[:, h * head_dim : (h + 1) * head_dim] for h in range(n_heads)])
    b_k = b_k_flat.reshape(n_heads, head_dim)
    b_v = b_v_flat.reshape(n_heads, head_dim)

    result = LayerMap(
        target_layer=target_layer,
        source_layers=chosen,
        w_k=w_k.astype(dtype),
        b_k=b_k.astype(dtype),
        w_v=w_v.astype(dtype),
        b_v=b_v.astype(dtype),
        r2_k=r2_k,
        r2_v=r2_v,
    )
    # Retire this layer's graph now rather than letting it accumulate across
    # every target layer in fit_mapper's loop — with L_t layers unevaluated
    # at once, MLX holds every layer's whole computation graph (and the
    # buffers it depends on) resident simultaneously instead of retiring each
    # layer's intermediates as it finishes.
    mx.eval(result.w_k, result.b_k, result.w_v, result.b_v)
    return result


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
        # _fit_layer evaluates its own output before returning, so each
        # iteration's graph is retired here rather than accumulating across
        # all L_t layers — peak memory stays bounded to one layer's working
        # set instead of the whole calibration run's.
        layers.append(_fit_layer(tl, src_k, src_v, tgt_k_all[pos], target_kv[tl][1], cfg))
        if progress is not None:
            progress(pos + 1, len(tgt_layers))

    return CrossModelMapper(source=source_spec, target=target_spec, config=cfg, layers=layers)
