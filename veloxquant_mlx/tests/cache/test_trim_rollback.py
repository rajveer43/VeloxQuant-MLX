"""trim() must roll back quantization frontiers / running statistics (#623).

After ``trim()`` the cache has to behave like one fed only the kept tokens:
rows past the trimmed offset must be quantized when they age out again, and
byte accounting / running stats must not include the discarded tokens.
"""

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

D = 64


def _feeder(seed=0, H=2):
    rng = np.random.default_rng(seed)
    return lambda n: mx.array(rng.standard_normal((1, H, n, D)).astype(np.float16))


def _cache(method, **kw):
    return KVCacheFactory.create(KVCacheConfig(method=method, head_dim=D, **kw))


@pytest.mark.parametrize("method", ["kivi", "kivi_sink"])
def test_kivi_trim_rolls_back_frontier_and_bytes(method):
    kw = {"bit_width_inlier": 2, "residual_length": 32}
    f = _feeder()
    c = _cache(method, **kw)
    c.update_and_fetch(f(256), f(256))
    c.trim(200)
    assert c._n_quantized <= c.offset
    new = f(400)
    k, _ = c.update_and_fetch(new, new)
    # Rows that were never quantized before the trim are now quantized.
    err = float(
        mx.abs(k[:, :, 56:224].astype(mx.float32) - new[:, :, :168].astype(mx.float32)).max()
    )
    assert err > 0.0

    fresh = _cache(method, **kw)
    x = mx.concatenate([f(56), new], 2)
    fresh.update_and_fetch(x, x)
    assert c._n_quantized == fresh._n_quantized
    assert c.compressed_key_bytes == fresh.compressed_key_bytes
    assert c.compressed_value_bytes == fresh.compressed_value_bytes
    assert c.fp16_key_bytes == fresh.fp16_key_bytes


def test_nsnquant_trim_rolls_back_frontier_and_bytes():
    f = _feeder()
    c = _cache("nsnquant", bit_width_inlier=4)
    r = c._residual_length
    c.update_and_fetch(f(2 * r), f(2 * r))
    c.update_and_fetch(f(3 * r), f(3 * r))
    c.trim(3 * r)
    assert c._q_end == 2 * r
    ref = _cache("nsnquant", bit_width_inlier=4)
    ref.update_and_fetch(f(2 * r), f(2 * r))
    assert c.compressed_key_bytes == ref.compressed_key_bytes
    assert c.fp16_key_bytes == ref.fp16_key_bytes
    c.trim(r // 2)  # unaligned trim snaps the frontier down to a chunk edge
    assert c._q_end == r and c._q_end <= c.offset


def _stats(c, names):
    return [np.array(getattr(c, n)) for n in names]


@pytest.mark.parametrize(
    "method,names,counter",
    [
        ("kitty", ["_key_sum", "_key_sq_sum"], "_n_keys"),
        ("adakv", ["_norm_sum", "_norm_sq_sum"], "_n_tokens"),
    ],
)
def test_running_stats_exclude_trimmed_tokens(method, names, counter):
    f = _feeder()
    P, E = 64, 40
    kp, vp = f(P), f(P)
    c = _cache(method, bit_width_inlier=4)
    ref = _cache(method, bit_width_inlier=4)
    c.update_and_fetch(kp, vp)
    ref.update_and_fetch(kp, vp)
    c.update_and_fetch(f(E), f(E))
    assert c.trim(E) == E
    assert getattr(c, counter) == getattr(ref, counter) == P
    for a, b in zip(_stats(c, names), _stats(ref, names), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)
    assert c.compressed_key_bytes == ref.compressed_key_bytes
    assert c.fp16_key_bytes == ref.fp16_key_bytes


@pytest.mark.parametrize("method", ["kitty", "adakv"])
def test_trim_to_empty_resets_stats(method):
    f = _feeder()
    c = _cache(method, bit_width_inlier=4)
    c.update_and_fetch(f(32), f(32))
    c.trim(32)
    k, _ = c.update_and_fetch(f(8), f(8))
    assert k.shape[2] == 8
