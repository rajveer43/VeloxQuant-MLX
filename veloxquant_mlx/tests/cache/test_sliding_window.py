"""Tests for SlidingWindowKVCache."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def base_cache():
    from veloxquant_mlx.cache.base import KVCacheBuilder

    return KVCacheBuilder().with_method("qjl").with_head_dim(64).with_jl_dim(64).build()


def test_sliding_window_evicts(base_cache) -> None:
    import mlx.core as mx

    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    sw = SlidingWindowKVCache(base_cache, window_size=5)
    rng = np.random.default_rng(0)
    for _ in range(10):
        k = mx.array(rng.standard_normal(64).astype(np.float16))
        v = mx.array(rng.standard_normal(64).astype(np.float16))
        sw.append(k, v)

    assert len(sw) == 5  # window of 5


def test_sliding_window_attend(base_cache) -> None:
    import mlx.core as mx

    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    sw = SlidingWindowKVCache(base_cache, window_size=10)
    rng = np.random.default_rng(1)
    for _ in range(20):
        k = mx.array(rng.standard_normal(64).astype(np.float16))
        v = mx.array(rng.standard_normal(64).astype(np.float16))
        sw.append(k, v)

    q = mx.array(rng.standard_normal(64).astype(np.float16))
    out = sw.attend(q)
    mx.eval(out)
    assert out.shape == (64,)


def test_sliding_window_invalid_size() -> None:
    from veloxquant_mlx.cache.base import KVCacheBuilder
    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    cache = KVCacheBuilder().with_method("qjl").with_head_dim(64).with_jl_dim(64).build()
    with pytest.raises(ValueError):
        SlidingWindowKVCache(cache, window_size=0)


# ---------------------------------------------------------------------------
# Regression for #81: KVCacheConfig.sliding_window + a non-standalone method
# (mlx_lm-protocol, i.e. update_and_fetch-based) must raise a clear config
# error at construction time, not produce a cache broken at first use.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,extra",
    [
        ("h2o", {}),
        ("tova", {}),
        ("kivi", {}),
        ("snapkv", {}),
        ("streaming_llm", {}),
    ],
)
def test_sliding_window_rejected_for_mlx_lm_protocol_methods(method, extra) -> None:
    """Regression for #81's exact repro: sliding_window combined with any
    mlx_lm-protocol (update_and_fetch-based) method must raise
    QuantizerConfigError at KVCacheFactory.create(), not silently produce a
    cache with neither update_and_fetch (not on the wrapper) nor
    append_key/append_value (not on the inner cache) working."""
    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.core.exceptions import QuantizerConfigError

    cfg = KVCacheConfig(method=method, head_dim=16, sliding_window=8, **extra)
    with pytest.raises(QuantizerConfigError, match="sliding_window"):
        KVCacheFactory.create(cfg)


def test_sliding_window_rejected_via_builder() -> None:
    """Same guard must apply through the KVCacheBuilder.build() entry point,
    since it delegates to KVCacheFactory.create()."""
    from veloxquant_mlx.cache.base import KVCacheBuilder
    from veloxquant_mlx.core.exceptions import QuantizerConfigError

    with pytest.raises(QuantizerConfigError, match="sliding_window"):
        (KVCacheBuilder().with_method("h2o").with_head_dim(16).with_sliding_window(8).build())


@pytest.mark.parametrize("method,extra", [("qjl", {"jl_dim": 8}), ("polar", {})])
def test_sliding_window_accepted_for_standalone_methods(method, extra) -> None:
    """sliding_window must still work end-to-end for STANDALONE_METHODS —
    the only family SlidingWindowKVCache's append_key/append_value/attend
    wrapping is actually compatible with."""
    import mlx.core as mx

    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    d = 16
    cfg = KVCacheConfig(method=method, head_dim=d, sliding_window=4, bit_width_inlier=2, **extra)
    cache = KVCacheFactory.create(cfg)
    assert isinstance(cache, SlidingWindowKVCache)

    rng = np.random.default_rng(0)
    for _ in range(6):
        k = mx.array(rng.standard_normal(d).astype(np.float16))
        v = mx.array(rng.standard_normal(d).astype(np.float16))
        cache.append(k, v)
    assert len(cache) == 4

    q = mx.array(rng.standard_normal(d).astype(np.float16))
    out = cache.attend(q)
    mx.eval(out)
    assert out.shape == (d,)


# ---------------------------------------------------------------------------
# Regression for #274: _rebuild_inner()'s attribute-guessing reset
# (hasattr(fresh, "_k_indices")) matched no registered cache class, so
# eviction never actually happened -- the inner cache grew unbounded and
# attend() was computed over stale/evicted-in-name-only tokens instead of
# the real window. Fixed by giving every concrete standalone cache class a
# reset() (KVCache ABC) that SlidingWindowKVCache calls directly instead of
# reflectively guessing internal attribute names.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,extra", [("qjl", {"jl_dim": 64}), ("polar", {})])
def test_sliding_window_memory_and_attend_match_ground_truth(method, extra) -> None:
    """The issue's own repro: memory_bytes() and attend() for a
    sliding-window cache fed 12 tokens at window=4 must exactly match a
    fresh cache fed only the true last 4 tokens -- not just report a capped
    len() while silently still holding (and attending over) every token."""
    import mlx.core as mx

    from veloxquant_mlx.cache.base import KVCacheBuilder

    def build(window=None):
        b = KVCacheBuilder().with_method(method).with_head_dim(64).with_bit_width(inlier=2)
        for k, v in extra.items():
            b = getattr(b, f"with_{k}")(v)
        if window:
            b = b.with_sliding_window(window)
        return b.build()

    rng = np.random.default_rng(2)
    vec = lambda: mx.array(rng.standard_normal(64).astype(np.float16))  # noqa: E731

    windowed = build(window=4)
    kvs = [(vec(), vec()) for _ in range(12)]
    for k, v in kvs:
        windowed.append(k, v)

    ground_truth = build(window=None)
    for k, v in kvs[-4:]:
        ground_truth.append(k, v)

    q = vec()
    out_windowed = np.array(windowed.attend(q))
    out_truth = np.array(ground_truth.attend(q))
    mx.eval()

    assert len(windowed) == 4
    assert windowed.memory_bytes() == ground_truth.memory_bytes()
    assert np.max(np.abs(out_windowed - out_truth)) < 1e-3


def test_sliding_window_reset_clears_window_and_inner(base_cache) -> None:
    """SlidingWindowKVCache.reset() must empty both its own window buffers
    and the inner cache, not just one or the other."""
    import mlx.core as mx

    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    sw = SlidingWindowKVCache(base_cache, window_size=5)
    rng = np.random.default_rng(0)
    for _ in range(8):
        sw.append(
            mx.array(rng.standard_normal(64).astype(np.float16)),
            mx.array(rng.standard_normal(64).astype(np.float16)),
        )
    assert len(sw) == 5

    sw.reset()
    assert len(sw) == 0
    assert sw.memory_bytes() == 0


def test_spectral_calibration_survives_sliding_window_eviction() -> None:
    """Regression for the calibration-loss trap a naive fix could introduce:
    SpectralQuantKVCache.calibrate() injects externally-computed rotation
    matrices *after* construction, not derived from the token stream itself
    (unlike TurboQuant's online outlier detector). A rebuild that
    reconstructs the inner cache from scratch (e.g. via its constructor)
    would silently drop that calibration on the first window eviction and
    regress to random-rotation mode. reset() must clear only token storage,
    leaving self._key_q/self._val_q (and any injected calibration) intact.
    """
    import mlx.core as mx

    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache
    from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache

    d = 64
    cfg = KVCacheConfig(method="spectral", head_dim=d, bit_width_inlier=2, sliding_window=4)
    cache = KVCacheFactory.create(cfg)
    assert isinstance(cache, SlidingWindowKVCache)
    inner = cache._inner
    assert isinstance(inner, SpectralQuantKVCache)

    rng = np.random.default_rng(5)
    key_U = mx.array(np.linalg.qr(rng.standard_normal((d, d)))[0].astype(np.float32))
    val_U = mx.array(np.linalg.qr(rng.standard_normal((d, d)))[0].astype(np.float32))
    inner.calibrate((key_U, val_U, None, None, 4, 50))
    key_q_before = inner._key_q
    val_q_before = inner._val_q

    for _ in range(10):  # more than window_size=4 -> forces at least one eviction/rebuild
        cache.append(
            mx.array(rng.standard_normal(d).astype(np.float16)),
            mx.array(rng.standard_normal(d).astype(np.float16)),
        )

    assert len(cache) == 4
    assert inner._key_q is key_q_before
    assert inner._val_q is val_q_before


def test_turboquant_outlier_detector_recalibrates_after_eviction() -> None:
    """TurboQuant's outlier-channel detector calibrates online from the
    token stream itself (unlike spectral's externally-injected rotation),
    so unlike spectral it is correct -- not merely acceptable -- for reset()
    to rebuild a fresh detector rather than preserve the old one: stale
    channel picks calibrated on now-evicted tokens would misrepresent the
    current window's statistics. This just confirms reset() leaves the
    cache in a working, re-calibratable state instead of crashing or
    reusing a detector wired to arrays sized for the old capacity."""
    import mlx.core as mx

    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    d = 16
    cfg = KVCacheConfig(
        method="turboquant_prod",
        head_dim=d,
        bit_width_inlier=2,
        sliding_window=4,
        enable_outlier_two_stream=True,
        n_outlier_channels=2,
        n_calib_tokens=2,
    )
    cache = KVCacheFactory.create(cfg)
    assert isinstance(cache, SlidingWindowKVCache)

    rng = np.random.default_rng(7)
    for _ in range(9):  # forces multiple window rebuilds past n_calib_tokens=2
        cache.append(
            mx.array(rng.standard_normal(d).astype(np.float16)),
            mx.array(rng.standard_normal(d).astype(np.float16)),
        )
    assert len(cache) == 4

    q = mx.array(rng.standard_normal(d).astype(np.float16))
    out = cache.attend(q)
    mx.eval(out)
    assert out.shape == (d,)
