"""Tests for KVTCKVCache (cache/kvtc_cache.py).

Covers: factory dispatch, construction guards, config propagation via
for_model, the basis/allocation being fixed after prefill and reused
unchanged across decode steps (the "not path-dependent" contrast with the
eviction family — H2O/TOVA/MorphKV/KVzip), byte-accounting properties, and
compression_ratio > 1 at a reasonable budget on structured (low-rank) data.
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kvtc_cache import KVTCKVCache


def _kv(B, H, S, D, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    k = mx.array((rng.standard_normal((B, H, S, D)) * scale).astype(np.float16))
    v = mx.array((rng.standard_normal((B, H, S, D)) * scale).astype(np.float16))
    return k, v


def _low_rank_kv(B, H, S, D, r_true=4, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.array([20.0 / (i + 1) for i in range(r_true)])
    k = np.zeros((B, H, S, D), dtype=np.float32)
    v = np.zeros((B, H, S, D), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            U = rng.standard_normal((S, r_true))
            Wk = rng.standard_normal((r_true, D)) * scale[:, None]
            Wv = rng.standard_normal((r_true, D)) * scale[:, None]
            k[b, h] = U @ Wk + rng.standard_normal((S, D)) * 0.05
            v[b, h] = U @ Wv + rng.standard_normal((S, D)) * 0.05
    return mx.array(k.astype(np.float16)), mx.array(v.astype(np.float16))


def _make(**kw):
    cfg = KVCacheConfig(method="kvtc", **kw)
    return KVCacheFactory.create(cfg)


# ---------------------------------------------------------------------------
# factory / construction
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = _make(head_dim=16, kvtc_bit_budget=32)
    assert isinstance(cache, KVTCKVCache)


def test_no_bits_attribute():
    cache = _make(head_dim=16)
    assert not hasattr(cache, "bits")


def test_construction_guard_negative_budget():
    with pytest.raises(ValueError, match="kvtc_bit_budget must be >= 0"):
        _make(head_dim=16, kvtc_bit_budget=-4)


def test_config_defaults_propagate():
    cache = _make(head_dim=32)
    assert cache.bit_budget == 512  # KVCacheConfig default


def test_config_overrides_propagate():
    cache = _make(head_dim=16, kvtc_bit_budget=64, kvtc_beta=4.0)
    assert cache.bit_budget == 64
    assert cache._beta == 4.0


def test_for_model_propagates_kvtc_fields():
    class _MockAttn:
        head_dim = 16

    class _MockLayer:
        def __init__(self):
            self.self_attn = _MockAttn()

    class _MockArgs:
        hidden_size = 64
        num_attention_heads = 4

    class _MockModel:
        args = _MockArgs()
        layers = [_MockLayer() for _ in range(3)]

    model = _MockModel()
    cfg = KVCacheConfig(method="kvtc", kvtc_bit_budget=48, seed=0)
    caches = KVCacheBuilder.for_model(model, cfg)
    assert len(caches) == 3
    assert all(isinstance(c, KVTCKVCache) for c in caches)
    assert all(c.bit_budget == 48 for c in caches)


# ---------------------------------------------------------------------------
# output shape / dtype sanity
# ---------------------------------------------------------------------------
def test_prefill_output_shape():
    cache = _make(head_dim=16, kvtc_bit_budget=32)
    k, v = _kv(1, 2, 20, 16, seed=0)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape == (1, 2, 20, 16)
    assert V.shape == (1, 2, 20, 16)
    assert K.dtype == mx.float16 and V.dtype == mx.float16


def test_decode_accumulates_offset():
    cache = _make(head_dim=8, kvtc_bit_budget=16)
    k0, v0 = _kv(1, 1, 10, 8, seed=0)
    cache.update_and_fetch(k0, v0)
    for i in range(5):
        k1, v1 = _kv(1, 1, 1, 8, seed=100 + i)
        K, V = cache.update_and_fetch(k1, v1)
    assert cache.offset == 15
    assert K.shape == (1, 1, 15, 8)


# ---------------------------------------------------------------------------
# basis/allocation fixed after prefill — NOT path-dependent (contrast with
# the eviction family H2O/TOVA/MorphKV/KVzip)
# ---------------------------------------------------------------------------
def test_basis_and_allocation_frozen_after_prefill():
    cache = _make(head_dim=16, kvtc_bit_budget=32)
    k0, v0 = _kv(1, 1, 20, 16, seed=1)
    cache.update_and_fetch(k0, v0)

    key_state = cache._keys_states[0]
    V_after_prefill = key_state._artifact.V
    bits_after_prefill = key_state._artifact.bit_allocation.copy()

    for i in range(15):
        k1, v1 = _kv(1, 1, 1, 16, seed=200 + i)
        cache.update_and_fetch(k1, v1)

    V_after_decode = key_state._artifact.V
    bits_after_decode = key_state._artifact.bit_allocation
    assert bool(mx.all(V_after_prefill == V_after_decode).item())
    assert np.array_equal(bits_after_prefill, bits_after_decode)


def test_deterministic_same_call_pattern_same_output():
    """Same prefill-then-decode call sequence, run twice independently,
    gives bit-for-bit identical stored basis/allocation/output — the
    determinism pin required because KVTC is explicitly NOT path-dependent
    like H2O/TOVA/MorphKV/KVzip.
    """

    def run():
        cache = _make(head_dim=16, kvtc_bit_budget=32)
        rng = np.random.default_rng(3)
        k_all = rng.standard_normal((1, 1, 20, 16)).astype(np.float16)
        v_all = rng.standard_normal((1, 1, 20, 16)).astype(np.float16)
        cache.update_and_fetch(mx.array(k_all[:, :, :10]), mx.array(v_all[:, :, :10]))
        K = V = None
        for t in range(10, 20):
            K, V = cache.update_and_fetch(
                mx.array(k_all[:, :, t : t + 1]), mx.array(v_all[:, :, t : t + 1])
            )
        return K, V

    K1, V1 = run()
    K2, V2 = run()
    assert bool(mx.all(K1 == K2).item())
    assert bool(mx.all(V1 == V2).item())


# ---------------------------------------------------------------------------
# byte-accounting properties
# ---------------------------------------------------------------------------
def test_byte_accounting_properties_exist_and_sane():
    cache = _make(head_dim=16, kvtc_bit_budget=32)
    k, v = _kv(1, 2, 40, 16, seed=5)
    cache.update_and_fetch(k, v)

    assert cache.kvtc_bytes > 0
    assert cache.pre_entropy_bytes >= 0
    assert cache.full_seq_bytes == 1 * 2 * 40 * 16 * 2 * 2
    assert cache.entropy_coding_gain > 0


def test_compression_ratio_defaults_to_one_when_empty():
    cache = _make(head_dim=16, kvtc_bit_budget=32)
    assert cache.compression_ratio == 1.0


def test_compression_ratio_above_one_on_structured_long_sequence():
    """On low-rank (structured) data at long enough sequence length that the
    fixed projection-basis overhead is amortised, compression_ratio should
    exceed 1 for both K and V (both compressed, mirroring Palu's scope).
    """
    cache = _make(head_dim=32, kvtc_bit_budget=48)
    k, v = _low_rank_kv(1, 1, 512, 32, r_true=4, seed=7)
    cache.update_and_fetch(k, v)
    assert cache.compression_ratio > 1.0


# ---------------------------------------------------------------------------
# Regression for #77: the realized entropy-coded payload+table must never
# exceed fixed-width packing, on both the prefill path (kvtc_compress) and
# the decode/requantize path (_TensorKVTC._requantize in kvtc_cache.py,
# which independently built the artifact and previously duplicated the
# always-Huffman bug). Note: cache.entropy_coding_gain (pre_entropy_bytes /
# kvtc_bytes) is a *different*, wider ratio that also carries the fixed
# projection-basis (V) overhead unrelated to entropy coding, and can
# legitimately stay < 1 even after this fix -- so we assert the actual
# per-tensor-state guarantee the fix provides, not that wider ratio.
# ---------------------------------------------------------------------------
def _payload_never_exceeds_fixed_width(cache):
    from veloxquant_mlx.quantizers._entropy_coding import table_nbytes

    for state in [*cache._keys_states, *cache._vals_states]:
        art = state._artifact
        if art is None:
            continue
        stored = len(art.entropy_payload) + table_nbytes(art.entropy_table)
        assert stored <= state.pre_entropy_bytes


def test_entropy_payload_never_worse_than_fixed_width_after_prefill():
    cache = _make(head_dim=128, kvtc_bit_budget=128 * 3)
    k, v = _kv(1, 1, 512, 128, seed=11)
    cache.update_and_fetch(k, v)
    _payload_never_exceeds_fixed_width(cache)


def test_entropy_payload_never_worse_than_fixed_width_after_decode_requantize():
    """Prefill then several decode steps exercise _requantize -- the
    guarantee must hold there too, not just at prefill."""
    cache = _make(head_dim=64, kvtc_bit_budget=64 * 3)
    k0, v0 = _kv(1, 1, 64, 64, seed=12)
    cache.update_and_fetch(k0, v0)
    for i in range(10):
        k, v = _kv(1, 1, 1, 64, seed=100 + i)
        cache.update_and_fetch(k, v)
    _payload_never_exceeds_fixed_width(cache)


# ---------------------------------------------------------------------------
# multi-head / multi-batch independence
# ---------------------------------------------------------------------------
def test_multi_head_batch_independent_shapes():
    cache = _make(head_dim=8, kvtc_bit_budget=16)
    k, v = _kv(2, 3, 12, 8, seed=9)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape == (2, 3, 12, 8)
    assert V.shape == (2, 3, 12, 8)
