"""#967: memory-bound (fused_sdpa_memory_bound=True) ``state`` round trip.

``state`` returned the whole pre-allocated index buffers (capacity >= 256 rows)
but the setter read ``k.shape[2]`` as the live length, so a restored cache held
``capacity`` tokens instead of the real count: ``fused_sdpa`` attended over
phantom centroid-0 tokens and RoPE positions were shifted by the same amount.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.metal import fused_sdpa as fs
from veloxquant_mlx.metal import metal_available

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(not metal_available(), reason="Metal kernels not available."),
]

D = 32
H = 2


@pytest.fixture(autouse=True)
def _patched():
    # The memory-bound constructor requires the mlx_lm SDPA patch to be active.
    fs.unpatch_mlx_lm()
    fs.patch_mlx_lm_for_fused_sdpa()
    try:
        yield
    finally:
        fs.unpatch_mlx_lm()


def _cache(**kw):
    return KVCacheFactory.create(
        KVCacheConfig(
            method="vecinfer",
            head_dim=D,
            key_sub_dim=4,
            value_sub_dim=4,
            key_codebook_bits=6,
            value_codebook_bits=6,
            seed=967,
            fused_sdpa=True,
            fused_sdpa_memory_bound=True,
            **kw,
        )
    )


def _kv(s: int, seed: int):
    mx.random.seed(seed)
    return mx.random.normal((1, H, s, D)), mx.random.normal((1, H, s, D))


def _q(seed: int = 9):
    mx.random.seed(seed)
    return mx.random.normal((1, H, 1, D))


def _restore(src):
    dst = _cache()
    dst.state = src.state
    return dst


@pytest.mark.parametrize("n", [1, 8, 255, 256, 300])
def test_round_trip_keeps_live_length(n) -> None:
    c = _cache()
    c.update_and_fetch(*_kv(n, 0))
    assert (c.offset, c._stored_S_kv) == (n, n)

    c2 = _restore(c)
    assert (c2.offset, c2._stored_S_kv) == (n, n)
    assert c2.size() == n
    assert not c2.empty()


def test_state_returns_only_live_rows() -> None:
    c = _cache()
    c.update_and_fetch(*_kv(8, 0))
    k, v = c.state
    assert k.shape[2] == 8 and v.shape[2] == 8
    # The backing buffer is still over-allocated; only the view is trimmed.
    assert c._stored_k_indices.shape[2] >= 256


def test_fused_sdpa_matches_before_round_trip() -> None:
    c = _cache()
    c.update_and_fetch(*_kv(8, 0))
    q = _q()
    before = c.fused_sdpa(q, scale=D**-0.5)

    after = _restore(c).fused_sdpa(q, scale=D**-0.5)
    mx.eval(before, after)
    assert mx.allclose(before, after, atol=1e-5).item()


def test_restored_cache_matches_uninterrupted_cache() -> None:
    """Restore, then keep decoding: same indices and attention as never restoring."""
    a = _cache()
    a.update_and_fetch(*_kv(8, 0))
    a.update_and_fetch(*_kv(1, 1))

    b = _cache()
    b.update_and_fetch(*_kv(8, 0))
    c = _restore(b)
    c.update_and_fetch(*_kv(1, 1))

    assert (c.offset, c._stored_S_kv) == (9, 9)
    n = a._stored_S_kv
    assert mx.array_equal(a._stored_k_indices[:, :, :n], c._stored_k_indices[:, :, :n])
    assert mx.array_equal(a._stored_v_indices[:, :, :n], c._stored_v_indices[:, :, :n])
    q = _q()
    out_a = a.fused_sdpa(q, scale=D**-0.5)
    out_c = c.fused_sdpa(q, scale=D**-0.5)
    mx.eval(out_a, out_c)
    assert mx.allclose(out_a, out_c, atol=1e-5).item()


def test_restored_cache_grows_past_its_trimmed_capacity() -> None:
    c = _restore(_filled(8))
    c.update_and_fetch(*_kv(300, 2))
    assert c.offset == 308 == c._stored_S_kv
    assert c._stored_k_indices.shape[2] >= 308


def _filled(n):
    c = _cache()
    c.update_and_fetch(*_kv(n, 0))
    return c


def test_empty_round_trip_allows_first_update_with_real_shape() -> None:
    """An empty cache saves (1, 1, 0, n_sub) placeholders; they must not become the buffers."""
    c = _restore(_cache())
    assert c.empty() and c.offset == 0
    c.update_and_fetch(*_kv(4, 0))  # H=2: would fail to concatenate with a (1, 1, ...) buffer
    assert (c.offset, c._stored_S_kv) == (4, 4)


def test_max_ctx_still_enforced_after_restore() -> None:
    src = _cache(fused_sdpa_max_ctx=16)
    src.update_and_fetch(*_kv(8, 0))
    dst = _cache(fused_sdpa_max_ctx=16)
    dst.state = src.state
    assert dst.offset == 8
    with pytest.raises(RuntimeError, match="fused_sdpa_max_ctx"):
        dst.update_and_fetch(*_kv(9, 1))
