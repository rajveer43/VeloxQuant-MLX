"""snapkv / nestedkv budget bounds the prompt, not total length (#844).

The README used to call snapkv a "hard token cap". The budget is applied once
at prefill; decode tokens are always appended. This pins the documented
behavior so the docs and the code cannot drift apart again.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

D, H, BUDGET, PROMPT, DECODE = 64, 4, 64, 200, 50

ARMS = [("snapkv", "snap_budget"), ("nestedkv", "nestedkv_budget")]


@pytest.mark.parametrize("method,budget_field", ARMS)
def test_budget_caps_prompt_and_decode_appends(method, budget_field):
    cache = KVCacheFactory.create(
        KVCacheConfig(method=method, head_dim=D, **{budget_field: BUDGET})
    )
    k = mx.random.normal((1, H, PROMPT, D)).astype(mx.float16)
    cache.update_and_fetch(k, k)

    for _ in range(DECODE):
        k = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        k_out, _ = cache.update_and_fetch(k, k)
    # The prompt was cut to the budget, then every decode token was appended.
    assert k_out.shape[2] == BUDGET + DECODE
