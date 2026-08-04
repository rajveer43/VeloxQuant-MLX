"""Integration test for patch_model_kv_cache.

Regression coverage for two real bugs:

1. The original implementation looked for a pre-existing ``.cache``
   attribute on attention sub-modules and overwrote it. mlx_lm never
   creates such an attribute — caches are only ever built lazily through
   ``model.make_cache()`` (see mlx_lm.models.cache.make_prompt_cache). So
   the old code silently patched zero layers on every real model and only
   emitted a warning, while ``run_turboquant_method`` in the benchmark
   script kept measuring the *unpatched* fp16 model and mislabeling the
   result.
2. A later implementation fixed (1) but built the cache list once at
   patch time and returned that same list from every subsequent
   ``model.make_cache()`` call. ``mlx_lm.generate()`` calls
   ``make_cache()`` fresh at the start of each generation to get a clean
   cache — reusing one list meant a second ``generate()`` call on the
   same patched model silently leaked KV state (offsets, cached
   keys/values) from the first call. ``patch_vlm_kv_cache`` never had
   this bug (it always rebuilds fresh); ``patch_model_kv_cache`` now
   matches that behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

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
    # and must return a *fresh* cache list on every call — reusing one list
    # across generations would leak KV state between unrelated generate() calls.
    second = model.make_cache()
    third = model.make_cache(some_arg=1)
    assert second is not caches
    assert third is not caches
    assert second is not third
    assert len(second) == len(caches) == len(third) == 4


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


def test_patch_model_kv_cache_does_not_leak_across_generate_calls() -> None:
    """Writing to one make_cache() call's caches must not affect the next call's.

    Regression test for the sticky-cache bug: model.make_cache() used to
    close over a single cache list built once at patch time, so a second
    generate() call would inherit whatever offset/state the first call left
    behind instead of starting from an empty cache.
    """
    model = _make_fake_model(n_layers=2, n_heads=2, head_dim=32)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
    patch_model_kv_cache(model, config)

    first_call_caches = model.make_cache()
    k = mx.zeros((1, 2, 5, 32), dtype=mx.float16)
    v = mx.zeros((1, 2, 5, 32), dtype=mx.float16)
    first_call_caches[0].update_and_fetch(k, v)
    assert first_call_caches[0].offset == 5

    second_call_caches = model.make_cache()
    assert second_call_caches[0].offset == 0
    assert second_call_caches[1].offset == 0
