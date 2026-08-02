"""Integration tests for patch_vlm_kv_cache.

mlx-vlm's single-prompt path (verified against 0.6.5) builds its cache
via ``mlx_vlm.models.cache.make_prompt_cache(model.language_model)``,
which defers to ``language_model.make_cache()`` when defined. These
tests check the patch installs exactly that hook, builds fresh caches
per call (no KV state leaking between generations), warns on
token-eviction methods, and — when mlx-vlm is installed — that the real
``make_prompt_cache`` actually returns our caches.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.integration.mlx_vlm_patch import patch_vlm_kv_cache


def _make_language_model(n_layers: int, n_heads: int, head_dim: int) -> SimpleNamespace:
    hidden_size = n_heads * head_dim
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(head_dim=head_dim)) for _ in range(n_layers)
    ]
    args = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=n_heads)
    return SimpleNamespace(layers=layers, args=args)


def _make_fake_vlm(
    n_layers: int = 4,
    n_heads: int = 4,
    head_dim: int = 32,
    wrapper_exposes_layers: bool = False,
) -> SimpleNamespace:
    """A minimal object shaped like an mlx-vlm model.

    Two real wrapper shapes exist: Qwen2-VL style re-exposes the decoder
    layers as ``model.layers`` (and its args lack hidden_size), others
    only carry ``model.language_model``.
    """
    lm = _make_language_model(n_layers, n_heads, head_dim)
    wrapper = SimpleNamespace(
        language_model=lm,
        vision_tower=SimpleNamespace(),
        args=SimpleNamespace(text_config={}),  # no hidden_size on the wrapper
    )
    if wrapper_exposes_layers:
        wrapper.layers = lm.layers
    return wrapper


@pytest.mark.parametrize("wrapper_exposes_layers", [False, True])
def test_patch_wires_language_model_make_cache(wrapper_exposes_layers):
    model = _make_fake_vlm(n_layers=4, wrapper_exposes_layers=wrapper_exposes_layers)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)

    caches = patch_vlm_kv_cache(model, config)

    assert len(caches) == 4
    # The hook must land on the language model — the top-level model must
    # stay unpatched, or mlx-vlm's batch path would crash in to_batch_cache.
    assert hasattr(model.language_model, "make_cache")
    assert not callable(getattr(model, "make_cache", None))
    assert len(model.language_model.make_cache()) == 4


def test_make_cache_returns_fresh_caches_per_call():
    """Each generate() must get its own caches — no cross-call KV leaks."""
    model = _make_fake_vlm(n_layers=2, n_heads=2, head_dim=32)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)

    patch_vlm_kv_cache(model, config)
    first = model.language_model.make_cache()
    second = model.language_model.make_cache()

    assert first is not second
    assert first[0] is not second[0]

    # Writing into the first generation's cache must not touch the second's.
    k = mx.zeros((1, 2, 5, 32), dtype=mx.float16)
    v = mx.zeros((1, 2, 5, 32), dtype=mx.float16)
    first[0].update_and_fetch(k, v)
    assert second[0].offset == 0


def test_patch_builds_requested_method():
    model = _make_fake_vlm(n_layers=3)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
    caches = patch_vlm_kv_cache(model, config)
    for c in caches:
        assert type(c).__name__ == "TurboQuantRVQKVCache"


def test_rejects_text_only_model():
    text_model = _make_language_model(n_layers=2, n_heads=2, head_dim=32)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
    with pytest.raises(ValueError, match="language_model"):
        patch_vlm_kv_cache(text_model, config)


def test_eviction_method_warns():
    model = _make_fake_vlm(n_layers=2)
    config = KVCacheConfig(method="snapkv", seed=42)
    with pytest.warns(UserWarning, match="image tokens"):
        patch_vlm_kv_cache(model, config)


def test_quantization_method_does_not_warn():
    model = _make_fake_vlm(n_layers=2)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        patch_vlm_kv_cache(model, config)


def test_real_mlx_vlm_make_prompt_cache_uses_our_caches():
    """With mlx-vlm installed, its real make_prompt_cache must defer to
    the patched hook and hand back VeloxQuant caches."""
    cache_mod = pytest.importorskip("mlx_vlm.models.cache")

    model = _make_fake_vlm(n_layers=3)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
    patch_vlm_kv_cache(model, config)

    prompt_cache = cache_mod.make_prompt_cache(model.language_model, max_kv_size=None)
    assert len(prompt_cache) == 3
    for c in prompt_cache:
        assert type(c).__name__ == "TurboQuantRVQKVCache"
