"""#612: never compute attention over memory-bound sentinel K/V."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import llama

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.metal import fused_sdpa as fs
from veloxquant_mlx.metal import metal_available

pytestmark = [pytest.mark.metal, pytest.mark.skipif(not metal_available(), reason="Requires Metal")]


@pytest.fixture
def dispatcher():
    import mlx_lm.models.base as base

    fs.unpatch_mlx_lm()
    fs.patch_mlx_lm_for_fused_sdpa()
    try:
        yield base.scaled_dot_product_attention
    finally:
        fs.unpatch_mlx_lm()


def _cache(memory_bound):
    return KVCacheFactory.create(
        KVCacheConfig(
            method="vecinfer",
            head_dim=64,
            key_sub_dim=4,
            value_sub_dim=8,
            key_codebook_bits=8,
            value_codebook_bits=8,
            seed=612,
            fused_sdpa=True,
            fused_sdpa_memory_bound=memory_bound,
            fused_sdpa_max_ctx=128,
        )
    )


@pytest.mark.parametrize("kind", ["causal", "window", "additive", "additive-window", "all-visible"])
@pytest.mark.parametrize("expanded", [False, True])
def test_array_masks_match_materialized_attention(dispatcher, kind, expanded):
    """Compare prefill/chunk/decode output to ordinary SDPA over real K/V."""
    compressed, dense = _cache(True), _cache(False)
    rng = np.random.default_rng(612)
    for N in (16, 4, 1):
        # Models construct masks BEFORE updating the cache.
        mask = compressed.make_mask(
            N, return_array=True, window_size=8 if kind in ("window", "additive-window") else None
        )
        if mask is None:  # mlx_lm's single-query, no-window fast path
            mask = mx.ones((N, compressed.offset + N), dtype=mx.bool_)
        if kind in ("additive", "additive-window"):
            mask = mx.where(mask, 0.0, float("-inf")).astype(mx.float16)
        elif kind == "all-visible":
            mask = mx.ones_like(mask)
        if expanded:
            mask = mx.broadcast_to(mask, (2, 4, N, compressed.offset + N))
        shape = (2, 2, N, 64)
        k = mx.array(rng.standard_normal(shape).astype(np.float16))
        v = mx.array(rng.standard_normal(shape).astype(np.float16))
        q = mx.array(rng.standard_normal((2, 4, N, 64)).astype(np.float16))
        sentinel_k, sentinel_v = compressed.update_and_fetch(k, v)
        real_k, real_v = dense.update_and_fetch(k, v)
        expected = mx.fast.scaled_dot_product_attention(q, real_k, real_v, scale=0.125, mask=mask)
        actual = dispatcher(q, sentinel_k, sentinel_v, cache=compressed, scale=0.125, mask=mask)
        assert float(mx.max(mx.abs(expected))) > 0.1
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=5e-3, rtol=5e-3)
        assert compressed.keys is None and compressed.values is None


@pytest.mark.parametrize(
    "unsupported", ["hole", "padding", "head-specific", "bias", "empty", "shape", "string", "sinks"]
)
def test_memory_bound_rejects_unsupported_attention_without_fallback(unsupported):
    """A mask cannot be accepted based only on its shape or final row."""
    calls = []

    def original(*args, **kwargs):
        pytest.fail("Ordinary SDPA would read sentinel K/V")

    def fused(*args, **kwargs):
        calls.append(kwargs)
        pytest.fail("Unsupported attention must not reach the fused kernel")

    cache = SimpleNamespace(_memory_bound=True, fused_sdpa=fused)
    q = mx.zeros((2, 4, 4, 64))
    kv = mx.zeros((2, 2, 8, 64))
    mask = mx.arange(8)[None, :] <= mx.arange(4, 8)[:, None]
    sinks = None
    if unsupported == "hole":
        mask[0, 1] = False
    elif unsupported == "padding":
        mask[:, 0] = False
    elif unsupported == "head-specific":
        mask = mx.broadcast_to(mask, (2, 4, 4, 8))
        mask[1, 2, 0, 1] = False
    elif unsupported == "bias":
        mask = mx.where(mask, 0.25, float("-inf"))
    elif unsupported == "empty":
        mask = mx.zeros_like(mask)
    elif unsupported == "shape":
        mask = mx.ones((4, 7), dtype=mx.bool_)
    elif unsupported == "string":
        mask = "custom"
    elif unsupported == "sinks":
        mask, sinks = "causal", mx.zeros((4,))
    with pytest.raises(ValueError, match="fused_sdpa_memory_bound=False"):
        fs._make_patched_sdpa(original)(q, kv, kv, cache=cache, scale=0.125, mask=mask, sinks=sinks)
    assert not calls


def test_memory_bound_missing_fused_method_never_falls_back():
    def original(*args, **kwargs):
        pytest.fail("Must not use sentinel K/V")

    q = mx.zeros((1, 1, 1, 64))
    with pytest.raises(RuntimeError, match="callable fused_sdpa"):
        fs._make_patched_sdpa(original)(
            q, q, q, cache=SimpleNamespace(_memory_bound=True), scale=1, mask=None
        )


def test_standard_cache_preserves_arbitrary_masks_and_sinks():
    q = mx.zeros((1, 1, 1, 64))
    mask, sinks = mx.array([[0.5]]), mx.array([1.0])
    cache = SimpleNamespace(_memory_bound=False)
    result = object()

    def original(queries, keys, values, **kwargs):
        assert queries is q and keys is q and values is q
        assert kwargs == {"cache": cache, "scale": 0.125, "mask": mask, "sinks": sinks}
        return result

    assert (
        fs._make_patched_sdpa(original)(q, q, q, cache=cache, scale=0.125, mask=mask, sinks=sinks)
        is result
    )


def test_sliding_window_model_logits_match_materialized_cache(dispatcher):
    """Exercise the real patched model module, including single-token decode."""
    mx.random.seed(612)
    model = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=256,
            num_hidden_layers=2,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=128,
            head_dim=64,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4,
        )
    )
    dense, compressed = [_cache(False) for _ in range(2)], [_cache(True) for _ in range(2)]
    for tokens in ([5, 17, 42, 9, 77, 3, 64, 21, 11, 99], [7, 8, 9], [4], [2]):
        tokens = mx.array([tokens])
        expected = model(tokens, cache=dense)
        actual = model(tokens, cache=compressed)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-2, rtol=1e-2)
