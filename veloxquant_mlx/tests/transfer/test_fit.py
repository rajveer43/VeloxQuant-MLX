"""Ridge fitting and source-layer selection tests (paper §3.1–§3.2).

The strategy throughout is to construct data whose correct answer is known in
closed form — a target that is an exact linear function of a *specific* source
layer — so a fit that is subtly wrong (unpenalized intercept, wrong
concatenation order, R² computed over the wrong axis) fails visibly rather
than merely scoring slightly worse.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.transfer.fit import (
    _r2,
    fit_mapper,
    fit_ridge,
    layer_r2,
    select_source_layers,
)
from veloxquant_mlx.transfer.mapper import MapperConfig, MatchedKVError, ModelKVSpec

N_TOKENS = 160
HEAD_DIM = 16
N_HEADS = 2


def _spec(n_layers: int, base: float = 10000.0, **kw) -> ModelKVSpec:
    return ModelKVSpec(
        n_layers=n_layers,
        n_kv_heads=N_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=base,
        **kw,
    )


# --- ridge core --------------------------------------------------------


def test_ridge_recovers_exact_linear_relationship():
    """With a noiseless linear target and tiny λ, the fit must recover W and b."""
    mx.random.seed(0)
    x = mx.random.normal((N_TOKENS, 8))
    w_true = mx.random.normal((8, 4))
    b_true = mx.random.normal((4,))
    w, b = fit_ridge(x, x @ w_true + b_true, ridge_lambda=1e-8)
    assert float(mx.max(mx.abs(w - w_true))) < 1e-3
    assert float(mx.max(mx.abs(b - b_true))) < 1e-3


def test_ridge_fits_intercept_without_penalizing_it():
    """A large constant offset must be absorbed by the bias, not shrunk by λ.

    Centering is what buys this. Without it, a heavy λ would drag predictions
    toward zero instead of toward the target's mean.
    """
    mx.random.seed(1)
    x = mx.random.normal((N_TOKENS, 6))
    y = x @ mx.random.normal((6, 3)) + 50.0
    _, b = fit_ridge(x, y, ridge_lambda=10.0)
    assert float(mx.min(b)) > 40.0


def test_ridge_shrinks_weights_as_lambda_grows():
    """Larger λ must produce a smaller-norm solution — the point of the penalty."""
    mx.random.seed(2)
    x = mx.random.normal((N_TOKENS, 6))
    y = mx.random.normal((N_TOKENS, 3))
    small, _ = fit_ridge(x, y, ridge_lambda=1e-6)
    large, _ = fit_ridge(x, y, ridge_lambda=1e4)
    assert float(mx.sum(large**2)) < float(mx.sum(small**2))


def test_ridge_handles_rank_deficient_input():
    """Duplicated columns make XᵀX singular; the λ term must keep it solvable."""
    mx.random.seed(3)
    col = mx.random.normal((N_TOKENS, 1))
    x = mx.concatenate([col, col, col], axis=1)
    w, b = fit_ridge(x, mx.random.normal((N_TOKENS, 2)), ridge_lambda=0.01)
    assert bool(mx.all(mx.isfinite(w))) and bool(mx.all(mx.isfinite(b)))


def test_ridge_rejects_mismatched_and_empty_input():
    with pytest.raises(ValueError, match="rows"):
        fit_ridge(mx.zeros((10, 4)), mx.zeros((9, 2)))
    with pytest.raises(ValueError, match="zero samples"):
        fit_ridge(mx.zeros((0, 4)), mx.zeros((0, 2)))


# --- R² ----------------------------------------------------------------


def test_r2_is_one_for_perfect_and_zero_for_mean_prediction():
    mx.random.seed(4)
    y = mx.random.normal((50, 3))
    assert _r2(y, y) == pytest.approx(1.0, abs=1e-5)
    mean_only = mx.broadcast_to(mx.mean(y, axis=0), y.shape)
    assert _r2(y, mean_only) == pytest.approx(0.0, abs=1e-5)


def test_r2_of_constant_target_is_zero():
    """A zero-variance target has nothing to explain; must not divide by zero."""
    const = mx.ones((20, 2))
    assert _r2(const, const) == 0.0


def test_layer_r2_ranks_predictive_source_above_noise():
    mx.random.seed(5)
    signal = mx.random.normal((N_TOKENS, 8))
    noise = mx.random.normal((N_TOKENS, 8))
    target = signal @ mx.random.normal((8, 4))
    assert layer_r2(signal, target) > 0.99
    assert layer_r2(noise, target) < layer_r2(signal, target)


# --- source selection --------------------------------------------------


def test_select_source_layers_picks_the_predictive_layer():
    """The layer the target was actually built from must be selected."""
    mx.random.seed(6)
    layers = [mx.random.normal((N_TOKENS, 8)) for _ in range(5)]
    target = layers[3] @ mx.random.normal((8, 4))
    assert 3 in select_source_layers(layers, target, k=1)


def test_select_source_layers_returns_sorted_indices():
    """Ascending order is part of the artifact contract: it fixes concat order."""
    mx.random.seed(7)
    layers = [mx.random.normal((N_TOKENS, 8)) for _ in range(6)]
    chosen = select_source_layers(layers, mx.random.normal((N_TOKENS, 4)), k=3)
    assert chosen == sorted(chosen) and len(chosen) == 3


def test_select_source_layers_clamps_k_to_available():
    mx.random.seed(8)
    layers = [mx.random.normal((N_TOKENS, 8)) for _ in range(3)]
    assert select_source_layers(layers, mx.random.normal((N_TOKENS, 4)), k=99) == [0, 1, 2]


def test_select_source_layers_is_deterministic():
    """Repeat calls must agree — a saved mapper's source list has to be stable."""
    mx.random.seed(9)
    layers = [mx.random.normal((N_TOKENS, 8)) for _ in range(5)]
    target = mx.random.normal((N_TOKENS, 4))
    assert select_source_layers(layers, target, k=2) == select_source_layers(layers, target, k=2)


def test_select_source_layers_rejects_bad_k():
    with pytest.raises(ValueError, match="k must be positive"):
        select_source_layers([mx.zeros((10, 4))], mx.zeros((10, 2)), k=0)


# --- full fit ----------------------------------------------------------


def _linear_pair(n_src_layers: int, n_tgt_layers: int, driver: int, tgt_base: float):
    """Build a source cache and a target that is an exact linear map of ``driver``."""
    mx.random.seed(10)
    pos = mx.arange(N_TOKENS)
    src_spec = _spec(n_src_layers)
    tgt_spec = _spec(n_tgt_layers, base=tgt_base)
    source = {
        i: (
            mx.random.normal((N_HEADS, N_TOKENS, HEAD_DIM)),
            mx.random.normal((N_HEADS, N_TOKENS, HEAD_DIM)),
        )
        for i in range(n_src_layers)
    }
    from veloxquant_mlx.transfer.rope import apply_rope, strip_rope

    w = mx.random.normal((N_HEADS * HEAD_DIM, HEAD_DIM)) * 0.1
    content = strip_rope(source[driver][0], pos, src_spec.rope_theta)
    flat_k = content.transpose(1, 0, 2).reshape(N_TOKENS, -1)
    flat_v = source[driver][1].transpose(1, 0, 2).reshape(N_TOKENS, -1)
    target = {
        li: (
            apply_rope(mx.stack([flat_k @ w] * N_HEADS), pos, tgt_spec.rope_theta),
            mx.stack([flat_v @ w] * N_HEADS),
        )
        for li in range(n_tgt_layers)
    }
    return source, target, src_spec, tgt_spec, pos


def test_fit_mapper_recovers_synthetic_linear_map():
    """End-to-end: a target built linearly from one source layer must fit at R²≈1."""
    source, target, src_spec, tgt_spec, pos = _linear_pair(4, 3, driver=1, tgt_base=1e6)
    mapper = fit_mapper(source, target, src_spec, tgt_spec, pos, MapperConfig(k=2))
    assert len(mapper.layers) == 3
    for lm in mapper.layers:
        assert lm.r2_k > 0.99 and lm.r2_v > 0.99
        assert 1 in lm.source_layers


def test_fit_mapper_records_hyperparameters_and_shapes():
    source, target, src_spec, tgt_spec, pos = _linear_pair(4, 2, driver=0, tgt_base=1e4)
    cfg = MapperConfig(k=3, ridge_lambda=0.05)
    mapper = fit_mapper(source, target, src_spec, tgt_spec, pos, cfg)
    lm = mapper.layers[0]
    assert mapper.config.ridge_lambda == 0.05
    assert lm.w_k.shape == (N_HEADS, 3 * N_HEADS * HEAD_DIM, HEAD_DIM)
    assert lm.b_k.shape == (N_HEADS, HEAD_DIM)


def test_fit_mapper_enforces_matched_kv():
    source, target, src_spec, _, pos = _linear_pair(2, 2, driver=0, tgt_base=1e4)
    mismatched = ModelKVSpec(n_layers=2, n_kv_heads=N_HEADS * 2, head_dim=HEAD_DIM, rope_theta=1e4)
    with pytest.raises(MatchedKVError, match="matched KV"):
        fit_mapper(source, target, src_spec, mismatched, pos)


def test_fit_mapper_rejects_rope_scaling():
    """rope_scaling would break the strip; refuse rather than silently corrupt."""
    source, target, src_spec, _, pos = _linear_pair(2, 2, driver=0, tgt_base=1e4)
    scaled = _spec(2, rope_scaling={"rope_type": "llama3", "factor": 8.0})
    with pytest.raises(MatchedKVError, match="rope_scaling"):
        fit_mapper(source, target, src_spec, scaled, pos)


def test_fit_mapper_rejects_traditional_rope():
    source, target, src_spec, _, pos = _linear_pair(2, 2, driver=0, tgt_base=1e4)
    trad = _spec(2, rope_traditional=True)
    with pytest.raises(MatchedKVError, match="traditional"):
        fit_mapper(source, target, src_spec, trad, pos)


def test_fit_mapper_rejects_position_count_mismatch():
    source, target, src_spec, tgt_spec, _ = _linear_pair(2, 2, driver=0, tgt_base=1e4)
    with pytest.raises(ValueError, match="positions has"):
        fit_mapper(source, target, src_spec, tgt_spec, mx.arange(N_TOKENS - 1))


def test_fit_mapper_reports_progress_per_layer():
    source, target, src_spec, tgt_spec, pos = _linear_pair(3, 3, driver=0, tgt_base=1e4)
    seen: list = []
    fit_mapper(
        source,
        target,
        src_spec,
        tgt_spec,
        pos,
        MapperConfig(k=1),
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]
