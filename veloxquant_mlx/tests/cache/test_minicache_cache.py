"""Tests for MiniCacheKVCache — cross-layer depth-dimension SLERP merging.

MiniCache merges adjacent middle-to-deep layers into a shared direction +
per-layer magnitudes, retaining high-divergence token pairs. These tests cover
role assignment via the builder, the shared-coordinator merge path, the
retention set, byte accounting, and the degenerate (no-coordinator) passthrough.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder, KVCacheFactory
from veloxquant_mlx.cache.minicache_cache import MiniCacheKVCache
from veloxquant_mlx.cache.minicache_coordinator import MiniCacheCoordinator


# --- fake model for for_model ---------------------------------------------
class _Attn:
    head_dim = 64


class _Layer:
    def __init__(self):
        self.self_attn = _Attn()


class _Args:
    hidden_size = 256
    num_attention_heads = 4


class _Inner:
    def __init__(self, n):
        self.layers = [_Layer() for _ in range(n)]


class _Model:
    def __init__(self, n=8):
        self.model = _Inner(n)
        self.args = _Args()


def _kv(S, H=4, D=64, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((1, H, S, D)).astype(np.float16)
    V = rng.standard_normal((1, H, S, D)).astype(np.float16)
    return mx.array(K), mx.array(V)


def _build(n=8, **cfg):
    base = dict(method="minicache", head_dim=64, minicache_start_frac=0.5, minicache_group_size=2)
    base.update(cfg)
    return KVCacheBuilder.for_model(_Model(n), KVCacheConfig(**base))


# ------------------------------------------------------------------
# Role assignment
# ------------------------------------------------------------------


def test_factory_degenerate_is_primary() -> None:
    c = KVCacheFactory.create(KVCacheConfig(method="minicache", head_dim=64))
    assert isinstance(c, MiniCacheKVCache)
    assert c.role == "primary"  # no coordinator → degenerate primary


def test_for_model_assigns_primary_and_merge() -> None:
    caches = _build(8, minicache_start_frac=0.5)
    roles = [(c.role, c.group_id) for c in caches]
    # early layers (below 0.5 depth) are all primary
    assert all(r[0] == "primary" for r in roles[:4])
    # middle-to-deep has merge layers
    assert any(r[0] == "merge" for r in roles[4:])


def test_early_layers_never_merged() -> None:
    caches = _build(8, minicache_start_frac=0.5)
    for c in caches[:4]:
        assert c.role == "primary"


# ------------------------------------------------------------------
# Shapes through a forward pass (primaries before merges)
# ------------------------------------------------------------------


def test_forward_pass_shapes_preserved() -> None:
    caches = _build(8)
    K, V = _kv(32)
    for c in caches:
        ko, vo = c.update_and_fetch(K, V)
        mx.eval(ko, vo)
        assert ko.shape == (1, 4, 32, 64)
        assert ko.dtype == mx.float16


# ------------------------------------------------------------------
# Merge quality: similar layers reconstruct well
# ------------------------------------------------------------------


def test_merge_layer_reconstructs_similar_primary() -> None:
    """When primary and merge layers are near-identical, the merge reconstructs
    the merge layer with low error (the regime MiniCache targets)."""
    caches = _build(4, minicache_start_frac=0.0)  # all eligible to merge
    # find a primary/merge pair in the same group
    pairs = {}
    for c in caches:
        pairs.setdefault(c.group_id, []).append(c)
    grp = next(v for v in pairs.values() if len(v) == 2)
    primary = next(c for c in grp if c.role == "primary")
    merge = next(c for c in grp if c.role == "merge")

    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, 4, 16, 64)).astype(np.float32)
    Kp = mx.array(base.astype(np.float16))
    Km = mx.array(
        (base + rng.standard_normal(base.shape).astype(np.float32) * 0.02).astype(np.float16)
    )
    V = mx.zeros((1, 4, 16, 64), dtype=mx.float16)

    primary.update_and_fetch(Kp, V)
    ko, _ = merge.update_and_fetch(Km, V)
    mx.eval(ko)
    mse = float(mx.mean((ko.astype(mx.float32) - Km.astype(mx.float32)) ** 2).item())
    assert mse < 0.05, f"merge reconstruction MSE too high for similar layers: {mse}"


# ------------------------------------------------------------------
# Retention: dissimilar token pairs kept unmerged
# ------------------------------------------------------------------


def test_dissimilar_tokens_retained() -> None:
    caches = _build(4, minicache_start_frac=0.0, minicache_retention_threshold=0.95)
    pairs = {}
    for c in caches:
        pairs.setdefault(c.group_id, []).append(c)
    grp = next(v for v in pairs.values() if len(v) == 2)
    primary = next(c for c in grp if c.role == "primary")
    merge = next(c for c in grp if c.role == "merge")

    rng = np.random.default_rng(1)
    Kp = mx.array(rng.standard_normal((1, 4, 8, 64)).astype(np.float16))
    Km = mx.array((-np.array(Kp)).astype(np.float16))  # opposite direction → retained
    V = mx.zeros((1, 4, 8, 64), dtype=mx.float16)
    primary.update_and_fetch(Kp, V)
    ko, _ = merge.update_and_fetch(Km, V)
    mx.eval(ko)
    # all opposite-direction tokens retained → reconstructed exactly
    assert merge.n_retained == 4 * 8
    assert merge.retention_rate == 1.0
    mse = float(mx.mean((ko.astype(mx.float32) - Km.astype(mx.float32)) ** 2).item())
    assert mse < 1e-3


# ------------------------------------------------------------------
# Byte accounting: merge layer compresses
# ------------------------------------------------------------------


def test_merge_layer_compresses() -> None:
    caches = _build(8)
    K, V = _kv(32, seed=2)
    for c in caches:
        c.update_and_fetch(K, V)
    merges = [c for c in caches if c.role == "merge"]
    assert merges, "expected at least one merge layer"
    for mc in merges:
        assert mc.compressed_key_bytes <= mc.fp16_key_bytes


def test_n_retained_plus_merged_equals_total() -> None:
    caches = _build(4, minicache_start_frac=0.0)
    K, V = _kv(16, seed=3)
    for c in caches:
        c.update_and_fetch(K, V)
    for mc in [c for c in caches if c.role == "merge"]:
        assert mc.n_retained + mc.n_merged == 4 * 16


# ------------------------------------------------------------------
# Degenerate passthrough is lossless
# ------------------------------------------------------------------


def test_degenerate_passthrough_lossless() -> None:
    c = KVCacheFactory.create(KVCacheConfig(method="minicache", head_dim=64))
    K, V = _kv(16, seed=5)
    ko, vo = c.update_and_fetch(K, V)
    mx.eval(ko, vo)
    assert np.allclose(np.array(ko), np.array(K), atol=1e-3)


# ------------------------------------------------------------------
# Coordinator
# ------------------------------------------------------------------


def test_coordinator_max_ctx_guard() -> None:
    coord = MiniCacheCoordinator(max_ctx=8)
    K, V = _kv(16)
    with pytest.raises(RuntimeError, match="max_ctx"):
        coord.publish_primary(0, 0, 16, K, V)


def test_determinism() -> None:
    c1 = _build(8, seed=11)
    c2 = _build(8, seed=11)
    K, V = _kv(32, seed=9)
    out1 = [c.update_and_fetch(K, V)[0] for c in c1]
    out2 = [c.update_and_fetch(K, V)[0] for c in c2]
    mx.eval(*out1, *out2)
    for a, b in zip(out1, out2):
        assert np.allclose(np.array(a), np.array(b), atol=1e-4)
