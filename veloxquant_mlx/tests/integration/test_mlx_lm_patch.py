"""Integration test for patch_model_kv_cache.

Regression coverage for a real bug: the previous implementation looked for a
pre-existing ``.cache`` attribute on attention sub-modules and overwrote it.
mlx_lm never creates such an attribute — caches are only ever built lazily
through ``model.make_cache()`` (see mlx_lm.models.cache.make_prompt_cache).
So the old code silently patched zero layers on every real model and only
emitted a warning, while ``run_turboquant_method`` in the benchmark script
kept measuring the *unpatched* fp16 model and mislabeling the result.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache


def _make_fake_model(n_layers: int = 4, n_heads: int = 4, head_dim: int = 32) -> SimpleNamespace:
    """A minimal object shaped like the mlx_lm attributes KVCacheBuilder.for_model reads.

    Matches the real convention: model.layers[i].self_attn.head_dim,
    model.args.hidden_size / num_attention_heads for the fallback path.
    """
    hidden_size = n_heads * head_dim
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(head_dim=head_dim)) for _ in range(n_layers)
    ]
    args = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=n_heads)
    return SimpleNamespace(layers=layers, args=args)


def test_patch_model_kv_cache_wires_make_cache() -> None:
    model = _make_fake_model(n_layers=4)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)

    caches = patch_model_kv_cache(model, config)

    assert len(caches) == 4
    assert hasattr(model, "make_cache")
    # make_cache must be callable with mlx_lm's (args, kwargs) convention
    # and must return the same cache list every time (persistent, not rebuilt).
    assert model.make_cache() is caches
    assert model.make_cache(some_arg=1) is caches


def test_patch_model_kv_cache_refuses_standalone_method() -> None:
    """spectral does not implement the mlx_lm KVCache serving contract (#27) --
    patching it in must raise before model.make_cache is ever touched, not
    fail later inside mlx_lm.generate().
    """
    model = _make_fake_model(n_layers=2)
    config = KVCacheConfig(method="spectral", bit_width_inlier=1, seed=42)

    with pytest.raises(QuantizerConfigError, match="spectral"):
        patch_model_kv_cache(model, config)

    assert not hasattr(model, "make_cache")


def test_patch_model_kv_cache_returns_correct_method() -> None:
    model = _make_fake_model(n_layers=2)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)

    caches = patch_model_kv_cache(model, config)

    for c in caches:
        assert type(c).__name__ == "TurboQuantRVQKVCache"


def test_patch_model_kv_cache_caches_are_independently_usable() -> None:
    """Each patched cache should be an empty, usable KVCache — not shared state."""
    model = _make_fake_model(n_layers=2, n_heads=2, head_dim=32)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)

    caches = patch_model_kv_cache(model, config)
    assert caches[0] is not caches[1]

    k = mx.zeros((1, 2, 5, 32), dtype=mx.float16)
    v = mx.zeros((1, 2, 5, 32), dtype=mx.float16)
    caches[0].update_and_fetch(k, v)
    # A fresh layer's cache must remain untouched by another layer's writes.
    assert caches[1].offset == 0
