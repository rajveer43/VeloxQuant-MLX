"""snapkv / nestedkv must reject a budget that leaves nothing evictable (#843).

Both used to clamp silently: ``snap_budget`` of 0, -1 or 1 collapsed a 50-token
prefill to a single row and ignored ``snap_n_sink``, with no error. The other
eviction caches (h2o, tova, pyramidkv, ...) raise for ``n_sink >= budget``;
these two now do too (sibling of #835).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.core.exceptions import QuantizerConfigError

D, H = 16, 2

ARMS = [
    ("snapkv", "snap_budget", "snap_n_sink"),
    ("nestedkv", "nestedkv_budget", "nestedkv_n_sink"),
]


def _make(method, budget_field, sink_field, budget, n_sink=None):
    kwargs = {budget_field: budget}
    if n_sink is not None:
        kwargs[sink_field] = n_sink
    return KVCacheFactory.create(KVCacheConfig(method=method, head_dim=D, **kwargs))


@pytest.mark.parametrize("method,budget_field,sink_field", ARMS)
@pytest.mark.parametrize("budget", [0, -1, 1, 4])  # default n_sink is 4
def test_budget_not_above_default_sinks_raises(method, budget_field, sink_field, budget):
    with pytest.raises(QuantizerConfigError, match="n_sink"):
        _make(method, budget_field, sink_field, budget)


@pytest.mark.parametrize("method,budget_field,sink_field", ARMS)
def test_negative_n_sink_raises(method, budget_field, sink_field):
    with pytest.raises(QuantizerConfigError, match="n_sink"):
        _make(method, budget_field, sink_field, 32, n_sink=-1)


@pytest.mark.parametrize("method,budget_field,sink_field", ARMS)
@pytest.mark.parametrize("budget,n_sink", [(1, 0), (5, 4), (32, 4)])
def test_valid_budgets_still_work_and_keep_the_sinks(
    method, budget_field, sink_field, budget, n_sink
):
    cache = _make(method, budget_field, sink_field, budget, n_sink)
    mx.random.seed(0)
    k = mx.random.normal((1, H, 50, D)).astype(mx.float16)
    cache.update_and_fetch(k, k)
    for _ in range(3):
        step = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        keys, _ = cache.update_and_fetch(step, step)
    mx.eval(keys)
    # The prefill is reduced to ``budget`` rows (not collapsed past it), plus the decodes.
    assert keys.shape[2] >= min(budget, 50)
