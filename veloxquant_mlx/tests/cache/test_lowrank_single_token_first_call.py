"""Low-rank caches must not freeze a basis fit on a single row (#840).

svdq, palu, kvtc and xkv fit a projection basis once and reuse it for every
later row. When the first ``update_and_fetch`` carried one token (a one-token
prompt, or an empty cache that starts with decode), the basis was rank-1 and
every later key/value was projected through it, collapsing output quality
(cosine ~0.05 against the input).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

D, H, TOTAL, TAIL = 64, 4, 205, 10
METHODS = ["svdq", "palu", "kvtc", "xkv"]


def _data(S: int, seed: int) -> mx.array:
    """Rank-8 structure plus small noise, roughly what real K/V looks like."""
    rng = np.random.RandomState(seed)
    basis = rng.randn(8, D).astype(np.float32)
    z = rng.randn(1, H, S, 8).astype(np.float32)
    return mx.array(z @ basis + 0.02 * rng.randn(1, H, S, D).astype(np.float32)).astype(mx.float16)


def _cos(a: mx.array, b: mx.array) -> float:
    a = np.array(a.astype(mx.float32)).ravel()
    b = np.array(b.astype(mx.float32)).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def _run(method: str, steps: list[int]):
    keys, values = _data(TOTAL, 1), _data(TOTAL, 2)
    cache = KVCacheFactory.create(KVCacheConfig(method=method, head_dim=D))
    i = 0
    for s in steps:
        out_k, out_v = cache.update_and_fetch(keys[:, :, i : i + s], values[:, :, i : i + s])
        i += s
    mx.eval(out_k, out_v)
    return cache, out_k, out_v, keys[:, :, :i], values[:, :, :i]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("first", [[1], [1, 1, 1]])
def test_single_token_first_matches_prefill_first(method, first):
    rest = TOTAL - sum(first) - TAIL
    _, k_ref, v_ref, kt, vt = _run(method, [rest + sum(first)] + [1] * TAIL)
    _, k_out, v_out, _, _ = _run(method, first + [rest] + [1] * TAIL)

    assert k_out.shape == k_ref.shape == kt.shape
    assert _cos(k_out, kt) > _cos(k_ref, kt) - 0.05
    assert _cos(v_out, vt) > _cos(v_ref, vt) - 0.05
    assert _cos(k_out, kt) > 0.5, f"{method}: keys collapsed (cos={_cos(k_out, kt):.3f})"


@pytest.mark.parametrize("method", METHODS)
def test_single_token_stream_is_lossless_and_counts_rows(method):
    cache, k_out, v_out, kt, vt = _run(method, [1] * 20)
    assert k_out.shape == kt.shape == (1, H, 20, D)
    assert _cos(k_out, kt) > 0.9999
    assert _cos(v_out, vt) > 0.9999
    assert cache.offset == 20


@pytest.mark.parametrize("method", METHODS)
def test_byte_accessors_work_before_the_first_fit(method):
    cache, *_ = _run(method, [1] * 5)
    assert cache.nbytes >= 0
    for attr in ("compressed_key_bytes", "fp16_key_bytes"):
        if hasattr(cache, attr):
            assert getattr(cache, attr) >= 0


@pytest.mark.parametrize("method", ["palu", "kvtc"])
def test_trim_before_the_first_fit(method):
    cache, *_ = _run(method, [1] * 6)
    assert cache.trim(2) == 2
    assert cache.offset == 4
    keys = _data(40, 3)
    out_k, _ = cache.update_and_fetch(keys, keys)
    mx.eval(out_k)
    assert out_k.shape[2] == 44
    assert cache.offset == 44
