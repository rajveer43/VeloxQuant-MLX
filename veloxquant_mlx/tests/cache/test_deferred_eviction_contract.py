"""#610: make_mask runs before eviction, but must describe returned K/V.

PyramidKV/Squeeze remain excluded: their per-layer budgets require the
separate shared-model-mask fix tracked in #390.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import llama
from mlx_lm.models.cache import KVCache

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory

CASES = {
    "knorm": {"knorm_budget": 16, "knorm_n_sink": 2},
    "qfilters": {
        "qfilters_budget": 16,
        "qfilters_n_sink": 2,
        "qfilters_calib_tokens": 8,
        "qfilters_min_retention": None,
    },
    "morphkv": {"morphkv_budget": 16, "morphkv_n_sink": 2, "morphkv_window": 2},
    "kvzip": {"kvzip_budget": 16, "kvzip_n_sink": 2},
    "keyformer": {"keyformer_budget": 16, "keyformer_n_sink": 2},
    "curdkv": {"curdkv_budget": 16, "curdkv_n_sink": 2, "curdkv_rank_cap": 4},
    "nestedkv": {"nestedkv_budget": 16, "nestedkv_n_sink": 2, "nestedkv_window": 4},
    "rocketkv": {},
}


@pytest.mark.parametrize("method", CASES)
@pytest.mark.parametrize("window", [None, 8])
@pytest.mark.parametrize("chunks", [(32, 4, 24, 1), (1, 32, 1, 4)])
def test_mask_matches_returned_rows_and_token_positions(method, window, chunks):
    cache = KVCacheFactory.create(KVCacheConfig(method=method, head_dim=32, **CASES[method]))
    rng = np.random.default_rng(61)
    seen = 0
    for N in chunks:
        prior_k, prior_v = (None, None) if cache.keys is None else cache.state
        # Values carry original token IDs, giving an independent mask oracle.
        keys = mx.array(rng.normal(size=(2, 2, N, 32)).astype(np.float16))
        ids = np.arange(seen, seen + N)
        values = mx.broadcast_to(mx.array(ids.astype(np.float16))[None, None, :, None], keys.shape)
        mask = cache.make_mask(N, window_size=window)
        k, v = cache.update_and_fetch(keys, values)
        expected_k = keys if prior_k is None else mx.concatenate([prior_k, keys], axis=2)
        expected_v = values if prior_v is None else mx.concatenate([prior_v, values], axis=2)
        np.testing.assert_array_equal(np.array(k), np.array(expected_k))
        np.testing.assert_array_equal(np.array(v), np.array(expected_v))
        kv_ids = np.array(v[:, 0, :, 0]).astype(int)
        expected = kv_ids[:, None, :] <= ids[None, :, None]
        if window is not None:
            expected &= ids[None, :, None] < kv_ids[:, None, :] + window
        if isinstance(mask, str):
            assert mask == "causal"
            actual = np.arange(k.shape[2])[None, :] <= (np.arange(N)[:, None] + k.shape[2] - N)
            np.testing.assert_array_equal(np.broadcast_to(actual, expected.shape), expected)
        elif mask is None:
            assert expected.all()
        else:
            assert mask.shape[-2:] == (N, k.shape[2])
            np.testing.assert_array_equal(
                np.broadcast_to(np.array(mask), (2, 1, N, k.shape[2])), expected[:, None]
            )
        # Exercise SDPA broadcasting with GQA, as model attention does.
        q = mx.zeros((2, 4, N, 32), dtype=mx.float16)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=32**-0.5, mask=mask)
        assert bool(mx.all(mx.isfinite(out)))
        seen += N
        if method not in ("nestedkv", "rocketkv"):
            assert cache.state[0].shape[2] <= 16


@pytest.mark.parametrize("method", CASES)
def test_prefill_logits_match_uncompressed_model(method):
    mx.random.seed(0)
    model = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=100,
            head_dim=32,
            num_key_value_heads=2,
        )
    )
    model.set_dtype(mx.float16)
    tokens = mx.random.randint(0, 100, (1, 32))
    reference = model(tokens, cache=[KVCache(), KVCache()])
    caches = KVCacheBuilder.for_model(
        model, KVCacheConfig(method=method, head_dim=32, **CASES[method])
    )
    actual = model(tokens, cache=caches)
    np.testing.assert_allclose(np.array(actual), np.array(reference), atol=1e-3, rtol=0)
    assert all(c.state[0].shape[2] < 32 for c in caches)
    # The model reuses layer zero's mask, so exercise subsequent chunked calls too.
    for N in (4, 24, 1):
        output = model(mx.random.randint(0, 100, (1, N)), cache=caches)
        assert bool(mx.all(mx.isfinite(output)))


@pytest.mark.parametrize("metal", [False, True])
def test_calibrated_qfilters_tracks_actual_survivors(metal):
    cfg = KVCacheConfig(
        method="qfilters",
        head_dim=32,
        qfilters_budget=16,
        qfilters_n_sink=2,
        qfilters_min_retention=None,
        use_metal_kernels=metal,
    )
    from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache

    cache = QFiltersKVCache(cfg, filters=mx.ones((2, 32), dtype=mx.float32))
    rng = np.random.default_rng(2)
    for start, N in [(0, 32), (32, 4), (36, 24)]:
        mask = cache.make_mask(N, window_size=8)
        k = mx.array(rng.normal(size=(2, 2, N, 32)).astype(np.float16))
        v = mx.broadcast_to(mx.arange(start, start + N)[None, None, :, None], k.shape).astype(
            mx.float16
        )
        ko, vo = cache.update_and_fetch(k, v)
        ids = np.array(vo[:, 0, :, 0])
        q = np.arange(start, start + N)[None, :, None]
        expected = ((ids[:, None] <= q) & (q < ids[:, None] + 8))[:, None]
        np.testing.assert_array_equal(np.broadcast_to(np.array(mask), expected.shape), expected)


@pytest.mark.parametrize("window", [None, 8])
def test_keyformer_mlx_survivor_indices(monkeypatch, window):
    import veloxquant_mlx.quantizers.keyformer as keyformer

    monkeypatch.setattr(keyformer, "_metal_evict_available", lambda: False)
    test_mask_matches_returned_rows_and_token_positions("keyformer", window, (32, 4, 24, 1))
