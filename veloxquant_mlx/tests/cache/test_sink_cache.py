"""Tests for SinkProtectedKVCache — KVSink-adapted key-norm sink protection.

Covers factory dispatch, planted-sink selection correctness, fp16
preservation of selected tokens, residual/sink byte-accounting (no double
count), determinism, and the core quality claims: sink-protected KIVI must
beat plain KIVI on data with planted sinks, and we compare honestly against
a Preserve-First-N (PFN) baseline at equal fp16 budget.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kivi_cache import KIVIKVCache
from veloxquant_mlx.cache.sink_cache import SinkProtectedKVCache


def _make(**cfg):
    base = dict(
        method="kivi_sink",
        head_dim=128,
        bit_width_inlier=2,
        residual_length=8,
        kivi_group_size=32,
        n_sink_tokens=5,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _kv_with_sinks(S=64, H=4, D=128, sink_pos=(0, 7, 20), scale=25.0, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((1, H, S, D)).astype(np.float16)
    V = rng.standard_normal((1, H, S, D)).astype(np.float16)
    for p in sink_pos:
        K[:, :, p, :] *= scale
    return K, V


def test_factory_dispatch_and_no_bits_leak() -> None:
    c = _make()
    assert isinstance(c, SinkProtectedKVCache)
    assert isinstance(c, KIVIKVCache)  # inherits the KIVI codec
    assert not hasattr(c, "bits")
    assert hasattr(c, "assigned_avg_bits")


def test_rejects_negative_sink_count() -> None:
    with pytest.raises(ValueError):
        _make(n_sink_tokens=-1)


def test_planted_sinks_detected_and_preserved() -> None:
    c = _make(n_sink_tokens=3)
    K, V = _kv_with_sinks(sink_pos=(0, 7, 20))
    ko, vo = c.update_and_fetch(mx.array(K), mx.array(V))
    mx.eval(ko, vo)
    assert set(c.sink_positions) == {0, 7, 20}
    ko_np, vo_np = np.array(ko), np.array(vo)
    for p in (0, 7, 20):
        assert np.array_equal(ko_np[:, :, p, :], K[:, :, p, :]), f"key sink {p}"
        assert np.array_equal(vo_np[:, :, p, :], V[:, :, p, :]), f"val sink {p}"
    # A non-sink, non-residual token must have been quantized (≠ input).
    assert not np.array_equal(ko_np[:, :, 30, :], K[:, :, 30, :])


def test_zero_sinks_equals_plain_kivi() -> None:
    """n_sink_tokens=0 must reproduce plain KIVI bit-for-bit."""
    K, V = _kv_with_sinks(sink_pos=())
    sink0 = _make(n_sink_tokens=0)
    kivi = KVCacheFactory.create(
        KVCacheConfig(
            method="kivi", head_dim=128, bit_width_inlier=2, residual_length=8, kivi_group_size=32
        )
    )
    a, _ = sink0.update_and_fetch(mx.array(K), mx.array(V))
    b, _ = kivi.update_and_fetch(mx.array(K), mx.array(V))
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


def test_byte_accounting_no_double_count() -> None:
    """compressed + sink + residual pools must partition the S tokens."""
    c = _make(n_sink_tokens=3, residual_length=8)
    K, V = _kv_with_sinks(S=64, sink_pos=(0, 7, 20))
    c.update_and_fetch(mx.array(K), mx.array(V))
    B, H, D = 1, 4, 128
    fp16_tok = D * 2 * 2 * H * B  # K+V bytes per token at fp16
    # 64 tokens: 8 residual, 3 sinks (all in quantized region), 53 compressed.
    assert c.residual_fp16_bytes == 8 * fp16_tok
    assert c.sink_fp16_bytes == 3 * fp16_tok
    assert c.fp16_key_bytes == H * B * 64 * D * 2
    total = (
        c.compressed_key_bytes
        + c.compressed_value_bytes
        + c.sink_fp16_bytes
        + c.residual_fp16_bytes
    )
    assert total < c.fp16_key_bytes + c.fp16_value_bytes  # still compresses


def test_determinism() -> None:
    K, V = _kv_with_sinks()
    a, _ = _make().update_and_fetch(mx.array(K), mx.array(V))
    b, _ = _make().update_and_fetch(mx.array(K), mx.array(V))
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


def test_decode_steps_after_prefill() -> None:
    c = _make(residual_length=4)
    K, V = _kv_with_sinks(S=40)
    c.update_and_fetch(mx.array(K), mx.array(V))
    rng = np.random.default_rng(9)
    for _ in range(5):
        k1 = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float16))
        v1 = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float16))
        ko, vo = c.update_and_fetch(k1, v1)
        mx.eval(ko, vo)
    assert c.offset == 45


def test_decode_tokens_age_out_of_residual_window() -> None:
    """Regression for #86 (inherited via KIVIKVCache.update_and_fetch):
    with S==1 every decode step, tokens must still age out of the fp16
    residual window once cumulative length exceeds residual_length."""
    c = _make(residual_length=4, n_sink_tokens=1, kivi_group_size=4)
    K, V = _kv_with_sinks(S=10, H=1, D=128, sink_pos=(0,))
    c.update_and_fetch(mx.array(K), mx.array(V))

    rng = np.random.default_rng(11)
    for _ in range(50):
        k1 = mx.array(rng.standard_normal((1, 1, 1, 128)).astype(np.float16))
        v1 = mx.array(rng.standard_normal((1, 1, 1, 128)).astype(np.float16))
        c.update_and_fetch(k1, v1)

    assert c.offset == 60
    D, H, B = 128, 1, 1
    expected_residual = 4 * D * 2 * 2 * H * B
    assert c.residual_fp16_bytes == expected_residual, (
        f"residual_fp16_bytes={c.residual_fp16_bytes} did not plateau at "
        f"{expected_residual} after 50 decode steps past the residual window"
    )
    assert c.compressed_key_bytes > 0


def _recon_err(cache, K, V):
    ko, _ = cache.update_and_fetch(mx.array(K), mx.array(V))
    mx.eval(ko)
    diff = np.array(ko).astype(np.float32) - K.astype(np.float32)
    return float(np.mean(diff**2))


def test_sink_protection_beats_plain_kivi_on_planted_sinks() -> None:
    """Core quality claim: protecting high-norm tokens must reduce key
    reconstruction MSE vs plain KIVI at the same bit-width, because the
    large-magnitude sinks dominate the squared error when quantized."""
    K, V = _kv_with_sinks(S=128, sink_pos=(0, 7, 20, 41, 90), scale=25.0)
    err_sink = _recon_err(_make(n_sink_tokens=5, residual_length=8), K, V)
    kivi = KVCacheFactory.create(
        KVCacheConfig(
            method="kivi", head_dim=128, bit_width_inlier=2, residual_length=8, kivi_group_size=32
        )
    )
    err_plain = _recon_err(kivi, K, V)
    assert err_sink < err_plain, (
        f"sink-protected MSE {err_sink:.5f} not < plain KIVI {err_plain:.5f}"
    )


def test_dynamic_selection_vs_pfn_equal_budget() -> None:
    """KVSink's actual claim: dynamic selection beats Preserve-First-N at
    equal fp16 budget when sinks are NOT all at the front.  PFN-5 here is
    sink protection that would pick positions 0-4; our planted sinks sit at
    {0, 7, 20, 41, 90}, so PFN protects only 1 of 5.  If this assertion
    ever fails, that is a reportable negative result for the key-norm
    proxy — do not delete the test; report it."""
    K, V = _kv_with_sinks(S=128, sink_pos=(0, 7, 20, 41, 90), scale=25.0)
    err_dyn = _recon_err(_make(n_sink_tokens=5, residual_length=8), K, V)

    # PFN-5 emulation: force the sink set to positions 0..4 by planting
    # nothing and protecting first-5 via a cache whose selection we bypass.
    pfn = _make(n_sink_tokens=5, residual_length=8)
    pfn._sink_norms = {i: float("inf") for i in range(5)}
    pfn._update_sinks = lambda keys, start: set(pfn._sink_norms)  # freeze
    err_pfn = _recon_err(pfn, K, V)
    assert err_dyn < err_pfn, f"dynamic MSE {err_dyn:.5f} not < PFN-5 {err_pfn:.5f}"
