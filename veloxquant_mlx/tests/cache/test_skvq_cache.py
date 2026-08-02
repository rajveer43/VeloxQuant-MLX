"""Tests for SKVQKVCache — SKVQ-adapted sliding-window quantization wrapper.

Covers mlx_lm protocol shape/dtype preservation, chunk-flush sliding-window
semantics (prefill/decode bit-for-bit path independence, frozen first-chunk
channel permutations), the attention-sink filter, reorder-on vs reorder-off
under heterogeneous channels, byte accounting against the closed form,
build-time validation, the max_ctx guard, and for_model wiring incl. the
fallback path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.skvq_cache import SKVQKVCache
from veloxquant_mlx.quantizers.skvq import skvq_compressed_bytes


def _kv(B, H, S, D, seed=0, channel_scales=None):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((B, H, S, D)).astype(np.float32)
    v = rng.standard_normal((B, H, S, D)).astype(np.float32)
    if channel_scales is not None:
        cs = channel_scales.astype(np.float32)[None, None, None, :]
        k, v = k * cs, v * cs
    return mx.array(k.astype(np.float16)), mx.array(v.astype(np.float16))


def _het_scales(d, seed=3):
    rng = np.random.default_rng(seed)
    scales = np.logspace(-2, 1, d)
    rng.shuffle(scales)
    return scales


def _make(**cfg):
    base = dict(method="skvq", head_dim=64, skvq_window=16, skvq_n_sink=4, skvq_group_size=16)
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ------------------------------------------------------------------
# Factory / protocol basics
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), SKVQKVCache)


def test_short_sequence_passthrough_exact() -> None:
    """Everything inside the sliding window stays bit-exact fp16."""
    cache = _make()
    k, v = _kv(1, 2, 10, 64, seed=1)  # 10 < window=16 -> no flush
    ko, vo = cache.update_and_fetch(k, v)
    assert ko.shape == k.shape and vo.shape == v.shape
    assert ko.dtype == mx.float16
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))
    assert cache.quantized_tokens == 0
    assert cache.compressed_key_bytes == 0


def test_no_bits_leak() -> None:
    cache = _make()
    assert not hasattr(cache, "bits")
    assert cache.assigned_avg_bits == 16.0  # nothing flushed yet


def test_frontier_advances_in_whole_chunks() -> None:
    r = 16
    cache = _make()
    k, v = _kv(1, 2, 3 * r + 5, 64, seed=2)
    cache.update_and_fetch(k, v)
    assert cache.quantized_tokens == 3 * r
    # ... and the un-flushed tail is exact fp16
    ko = cache.keys[..., : cache.offset, :]
    assert np.array_equal(np.array(ko[:, :, 3 * r :, :]), np.array(k[:, :, 3 * r :, :]))


def test_sink_rows_fp16_exact_after_flush() -> None:
    n_sink = 4
    cache = _make(skvq_n_sink=n_sink)
    k, v = _kv(1, 2, 40, 64, seed=3)  # 2 chunks flushed
    ko, vo = cache.update_and_fetch(k, v)
    assert cache.quantized_tokens == 32
    assert np.array_equal(np.array(ko[:, :, :n_sink, :]), np.array(k[:, :, :n_sink, :]))
    assert np.array_equal(np.array(vo[:, :, :n_sink, :]), np.array(v[:, :, :n_sink, :]))


def test_quantized_region_actually_quantized() -> None:
    """Beyond the sinks, both K and V in flushed chunks differ from the
    fp16 input at 2 bits (the round-trip is lossy)."""
    cache = _make()
    k, v = _kv(1, 2, 32, 64, seed=4)
    ko, vo = cache.update_and_fetch(k, v)
    q = slice(4, 32)  # past sinks, inside flushed chunks
    assert not np.array_equal(np.array(ko[:, :, q, :]), np.array(k[:, :, q, :]))
    assert not np.array_equal(np.array(vo[:, :, q, :]), np.array(v[:, :, q, :]))


# ------------------------------------------------------------------
# Sliding-window / path-independence semantics
# ------------------------------------------------------------------


def test_prefill_decode_bit_for_bit_equivalence() -> None:
    """Same tokens as one prefill block vs token-by-token decode produce
    bit-for-bit identical caches: chunk boundaries, first-chunk permutation
    statistics, clip search, and sink restore are all functions of the same
    chunk contents."""
    r = 16
    S = 3 * r + 5
    k, v = _kv(1, 2, S, 64, seed=5, channel_scales=_het_scales(64))

    prefill = _make()
    ko_a, vo_a = prefill.update_and_fetch(k, v)

    decode = _make()
    for t in range(S):
        ko_b, vo_b = decode.update_and_fetch(k[:, :, t : t + 1, :], v[:, :, t : t + 1, :])
    mx.eval(ko_a, vo_a, ko_b, vo_b)
    assert np.array_equal(np.array(ko_a), np.array(ko_b))
    assert np.array_equal(np.array(vo_a), np.array(vo_b))
    assert prefill.quantized_tokens == decode.quantized_tokens


def test_chunks_frozen_after_flush() -> None:
    """A flushed chunk's stored reconstruction never changes afterwards."""
    r = 16
    cache = _make()
    k, v = _kv(1, 2, r, 64, seed=6)
    ko1, _ = cache.update_and_fetch(k, v)
    first = np.array(ko1[:, :, :r, :]).copy()
    for step in range(2 * r):
        k2, v2 = _kv(1, 2, 1, 64, seed=300 + step)
        ko, _ = cache.update_and_fetch(k2, v2)
    assert np.array_equal(np.array(ko[:, :, :r, :]), first)


def test_permutations_frozen_from_first_chunk() -> None:
    cache = _make()
    assert cache.key_perms is None and cache.value_perms is None
    k, v = _kv(1, 2, 16, 64, seed=7, channel_scales=_het_scales(64))
    cache.update_and_fetch(k, v)
    pk = np.array(cache.key_perms).copy()
    pv = np.array(cache.value_perms).copy()
    assert pk.shape == (2, 64) and pv.shape == (2, 64)
    for h in range(2):  # each row is a valid permutation
        assert sorted(pk[h].tolist()) == list(range(64))
        assert sorted(pv[h].tolist()) == list(range(64))
    # Later chunks (with different statistics) must not move the perms.
    k2, v2 = _kv(1, 2, 32, 64, seed=8)
    cache.update_and_fetch(k2, v2)
    assert cache.quantized_tokens == 48
    assert np.array_equal(np.array(cache.key_perms), pk)
    assert np.array_equal(np.array(cache.value_perms), pv)


def test_reorder_off_uses_identity() -> None:
    cache = _make(skvq_reorder=False)
    k, v = _kv(1, 2, 32, 64, seed=9)
    cache.update_and_fetch(k, v)
    assert cache.key_perms is None and cache.value_perms is None
    assert cache.perm_bytes == 0


def test_reorder_beats_identity_on_heterogeneous_channels() -> None:
    k, v = _kv(1, 2, 64, 64, seed=10, channel_scales=_het_scales(64))

    def mse(reorder):
        cache = _make(skvq_reorder=reorder, skvq_n_sink=0)
        ko, _ = cache.update_and_fetch(k, v)
        q = cache.quantized_tokens
        diff = np.array(ko[:, :, :q, :], dtype=np.float32) - np.array(
            k[:, :, :q, :], dtype=np.float32
        )
        return float(np.mean(diff**2))

    assert mse(True) < mse(False)


def test_clip_search_off_fixed_alpha_runs() -> None:
    cache = _make(skvq_clip_search=False, skvq_clip_alpha=0.9)
    k, v = _kv(1, 2, 32, 64, seed=11)
    ko, vo = cache.update_and_fetch(k, v)
    assert cache.quantized_tokens == 32
    assert ko.shape == k.shape


# ------------------------------------------------------------------
# Byte accounting / reporting
# ------------------------------------------------------------------


def test_byte_accounting_closed_form() -> None:
    B, H, D, r, n_sink, gs = 1, 2, 64, 16, 4, 16
    cache = _make()
    k, v = _kv(B, H, 2 * r + 3, D, seed=12)
    cache.update_and_fetch(k, v)
    # chunk 0 quantizes r - n_sink tokens, chunk 1 quantizes r
    expect_k = (
        (skvq_compressed_bytes(r - n_sink, D, 2, gs) + skvq_compressed_bytes(r, D, 2, gs)) * B * H
    )
    assert cache.compressed_key_bytes == expect_k
    assert cache.compressed_value_bytes == expect_k  # same bits for values
    assert cache.fp16_key_bytes == B * H * (2 * r + 3) * D * 2
    # residual snapshot: 3-token tail + 4 sink rows, K+V
    assert cache.residual_fp16_bytes == (3 + n_sink) * D * 2 * 2 * B * H
    assert cache.perm_bytes == 2 * H * D * 4
    assert 2.0 < cache.assigned_avg_bits < 16.0
    assert cache.tokens_seen == 2 * r + 3


def test_compression_ratio_beats_fp16_at_long_context() -> None:
    cache = _make(skvq_window=32, skvq_n_sink=4, skvq_max_ctx=4096)
    k, v = _kv(1, 2, 1024, 64, seed=13)
    cache.update_and_fetch(k, v)
    end_to_end = (
        cache.compressed_key_bytes
        + cache.compressed_value_bytes
        + cache.residual_fp16_bytes
        + cache.perm_bytes
    ) / (cache.fp16_key_bytes + cache.fp16_value_bytes)
    assert end_to_end < 0.35  # 2-bit codes + metadata + fp16 window


def test_determinism() -> None:
    k, v = _kv(1, 2, 80, 64, seed=14, channel_scales=_het_scales(64))
    a = _make().update_and_fetch(k, v)
    b = _make().update_and_fetch(k, v)
    assert np.array_equal(np.array(a[0]), np.array(b[0]))
    assert np.array_equal(np.array(a[1]), np.array(b[1]))


# ------------------------------------------------------------------
# Guards / validation
# ------------------------------------------------------------------


def test_max_ctx_guard_raises() -> None:
    cache = _make(skvq_max_ctx=32)
    k, v = _kv(1, 2, 33, 64, seed=15)
    with pytest.raises(ValueError, match="skvq_max_ctx"):
        cache.update_and_fetch(k, v)


def test_build_time_validation() -> None:
    with pytest.raises(ValueError, match="skvq_bits_key"):
        _make(skvq_bits_key=0)
    with pytest.raises(ValueError, match="skvq_bits_value"):
        _make(skvq_bits_value=9)
    with pytest.raises(ValueError, match="skvq_group_size"):
        _make(skvq_group_size=0)
    with pytest.raises(ValueError, match="skvq_window"):
        _make(skvq_window=1)
    with pytest.raises(ValueError, match="skvq_n_sink"):
        _make(skvq_n_sink=16)  # == window
    with pytest.raises(ValueError, match="skvq_clip_alpha"):
        _make(skvq_clip_search=False, skvq_clip_alpha=0.0)
    with pytest.raises(ValueError, match="skvq_clip_alpha"):
        _make(skvq_clip_search=False, skvq_clip_alpha=1.5)


# ------------------------------------------------------------------
# for_model wiring
# ------------------------------------------------------------------


class _ToyAttn:
    def __init__(self, head_dim):
        self.head_dim = head_dim


class _ToyLayer:
    def __init__(self, head_dim=64):
        self.self_attn = _ToyAttn(head_dim)


class _ToyNorm:
    """Layer without attention — must get the fallback cache."""

    pass


class _ToyInner:
    def __init__(self):
        self.layers = [_ToyLayer(), _ToyLayer(), _ToyNorm(), _ToyLayer()]


class _ToyModel:
    def __init__(self):
        self.model = _ToyInner()
        self.args = None


def test_for_model_wiring_and_fallback() -> None:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    caches = KVCacheBuilder.for_model(_ToyModel(), KVCacheConfig(method="skvq", head_dim=64))
    assert len(caches) == 4
    assert isinstance(caches[0], SKVQKVCache)
    assert isinstance(caches[1], SKVQKVCache)
    assert type(caches[2]) is _FallbackCache  # non-attention layer
    assert isinstance(caches[3], SKVQKVCache)

    k, v = _kv(1, 2, 8, 64, seed=16)
    ko, vo = caches[2].update_and_fetch(k, v)
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))
