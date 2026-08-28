"""Tests for ahead-of-time Metal kernel warmup (issue #250).

``warmup_for_config`` exists purely to move a config's first-call shader
compile from mid-generation (inside ``update_and_fetch``) to cache-build
time. The tests here check that:

  * warmup actually populates the same kernel cache a real call would use,
    keyed identically (so it isn't compiling a variant that then goes
    unused and gets recompiled anyway),
  * ``KVCacheBuilder.build()`` / ``.for_model()`` trigger it automatically,
  * it degrades to a silent no-op for unregistered methods, missing Metal,
    or a warmer that raises — warmup is a latency optimization, never a
    correctness gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.cache.kivi_cache import KIVIKVCache
from veloxquant_mlx.metal import _kivi_quant, metal_available
from veloxquant_mlx.metal._warmup import register_warmer, warmup_for_config

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


def _make_fake_model(n_layers: int = 2, n_heads: int = 2, head_dim: int = 32) -> SimpleNamespace:
    hidden_size = n_heads * head_dim
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(head_dim=head_dim)) for _ in range(n_layers)
    ]
    args = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=n_heads)
    return SimpleNamespace(layers=layers, args=args)


def test_warmup_matches_real_call_cache_key():
    _kivi_quant._cache.clear()
    cfg = KVCacheConfig(method="kivi", head_dim=128, bit_width_inlier=2)
    warmup_for_config(cfg)
    warm_keys = set(_kivi_quant._cache.keys())
    assert warm_keys, "warmup did not compile any KIVI kernel variant"

    cache = KIVIKVCache(cfg)
    x = mx.random.normal((1, 1, 32, 128)).astype(mx.float16)
    cache._quant_dequant_along(x, axis=-2)
    cache._quant_dequant_along(x, axis=-1)

    assert set(_kivi_quant._cache.keys()) == warm_keys


def test_warmup_noop_for_unregistered_method():
    cfg = KVCacheConfig(method="turboquant_prod", head_dim=128, bit_width_inlier=2)
    # Must not raise even though no warmer is registered for this method.
    warmup_for_config(cfg)


def test_warmup_swallows_warmer_exceptions():
    def _broken(_config):
        raise RuntimeError("boom")

    register_warmer("kivi", _broken)
    try:
        cfg = KVCacheConfig(method="kivi", head_dim=128, bit_width_inlier=2)
        warmup_for_config(cfg)  # must not propagate
    finally:
        from veloxquant_mlx.metal._warmup import _warm_kivi

        register_warmer("kivi", _warm_kivi)


def test_builder_build_warms_kivi_kernels():
    _kivi_quant._cache.clear()
    KVCacheBuilder().with_method("kivi").with_head_dim(64).with_bit_width(4).build()
    assert ("kivi_group_quant", -2, 32, 15, 1e-08, 64, "half") in _kivi_quant._cache
    assert ("kivi_group_quant", -1, 32, 15, 1e-08, 64, "half") in _kivi_quant._cache


def test_for_model_warms_kivi_kernels_once_per_distinct_head_dim():
    _kivi_quant._cache.clear()
    model = _make_fake_model(n_layers=3, n_heads=2, head_dim=32)
    config = KVCacheConfig(method="kivi", bit_width_inlier=2, seed=42)

    caches = KVCacheBuilder.for_model(model, config)

    assert len(caches) == 3
    assert ("kivi_group_quant", -2, 32, 3, 1e-08, 32, "half") in _kivi_quant._cache
    assert ("kivi_group_quant", -1, 32, 3, 1e-08, 32, "half") in _kivi_quant._cache
    # All three layers share head_dim=32, so warmup should compile each
    # variant exactly once rather than once per layer.
    assert len(_kivi_quant._cache) == 2
