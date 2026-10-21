"""age_tiered must re-quantize only groups that changed, not the whole history (#846).

``_requantize`` was meant to splice unchanged groups from the previous output
(#397) but returned early whenever the buffer had grown, i.e. on every call, so
each decode step re-quantized all tokens for K and for V. This pins the work
done per decode step and checks the incremental output still equals a full
recompute from the raw tokens at every step, across tier boundaries.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache import age_tiered_cache
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.age_tiered import (
    age_tier_quantize_batched,
    assign_age_tiers,
)

H, D, GS = 2, 16, 8
RECENT_B, MID_B = 16, 48


def _make():
    return KVCacheFactory.create(
        KVCacheConfig(
            method="age_tiered",
            head_dim=D,
            age_recent_boundary=RECENT_B,
            age_mid_boundary=MID_B,
            age_group_size=GS,
        )
    )


def _full_recompute(cache, raw):
    n = raw.shape[1]
    ages = [cache._current_position - (i + 1) for i in range(n)]
    tiers = assign_age_tiers(ages, RECENT_B, MID_B)
    gtier = [tiers[g * GS] for g in range((n + GS - 1) // GS)]
    return age_tier_quantize_batched(raw, gtier, cache._tiers, GS)


def test_incremental_output_equals_full_recompute_every_step():
    rng = np.random.default_rng(0)
    cache = _make()
    chunks = [37] + [1] * 90 + [5, 1, 1, 12] + [1] * 20
    for i, s in enumerate(chunks):
        k = mx.array(rng.standard_normal((1, H, s, D)).astype(np.float16))
        cache.update_and_fetch(k, k)
        expected = _full_recompute(cache, cache._raw_keys)
        np.testing.assert_array_equal(
            np.array(cache._keys_batched), np.array(expected), err_msg=f"step {i}"
        )


def test_decode_quantizes_only_changed_groups(monkeypatch):
    real, widths = age_tiered_cache.age_tier_quantize_batched, []

    def spy(raw, *a, **kw):
        widths.append(int(raw.shape[1]))
        return real(raw, *a, **kw)

    monkeypatch.setattr(age_tiered_cache, "age_tier_quantize_batched", spy)
    cache = _make()
    k = mx.random.normal((1, H, 200, D)).astype(mx.float16)
    cache.update_and_fetch(k, k)
    widths.clear()
    for _ in range(20):
        t = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        cache.update_and_fetch(t, t)
    # Per decode step and per K/V: the partial last group plus at most one
    # group crossing each tier boundary -- never the ~200-token history.
    assert max(widths) <= 4 * GS, f"re-quantized {max(widths)} tokens in one call"
