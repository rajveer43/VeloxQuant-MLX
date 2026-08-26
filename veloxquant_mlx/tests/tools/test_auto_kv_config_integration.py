"""Integration coverage for the auto KV-config selector (#253) against real caches.

Pure-heuristic behaviour of ``select_kv_config`` is covered without MLX in
``tests/non_metal/test_auto_kv_config.py``; this file only checks that
``to_kv_cache_config()`` produces a ``KVCacheConfig`` that
``KVCacheFactory`` actually accepts and can round-trip real tensors
through, for each method the selector can choose.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache.base import KVCacheFactory
from veloxquant_mlx.tools.auto_kv_config import (
    HardwareProfile,
    WorkloadProfile,
    select_kv_config,
    to_kv_cache_config,
)


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


@pytest.mark.parametrize(
    "workload,hardware",
    [
        (
            WorkloadProfile(seq_len=2048, head_dim=128, n_layers=32, n_kv_heads=8),
            HardwareProfile(ram_gb=64),
        ),  # short context -> kivi
        (
            WorkloadProfile(seq_len=65536, head_dim=128, n_layers=32, n_kv_heads=8),
            HardwareProfile(ram_gb=64),
        ),  # long context -> turboquant_rvq
        (
            WorkloadProfile(seq_len=200000, head_dim=128, n_layers=80, n_kv_heads=8),
            HardwareProfile(ram_gb=8),
        ),  # extreme pressure -> streaming_llm
    ],
)
def test_auto_selected_config_builds_and_round_trips(workload, hardware) -> None:
    result = select_kv_config(workload, hardware)
    config = to_kv_cache_config(result, workload)

    assert config.method == result.method
    assert config.head_dim == workload.head_dim

    cache = KVCacheFactory.create(config)
    assert isinstance(cache, _MLXKVCache)

    k, v = _kv(1, workload.n_kv_heads, 16, workload.head_dim)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, workload.n_kv_heads, 16, workload.head_dim)
    assert ko.dtype == mx.float16


def test_short_context_pure_mlx_packing_still_builds() -> None:
    workload = WorkloadProfile(seq_len=2048, head_dim=128, n_layers=32, n_kv_heads=8)
    result = select_kv_config(workload, HardwareProfile(ram_gb=64, metal_available=False))
    config = to_kv_cache_config(result, workload)

    assert config.use_metal_kernels is False
    cache = KVCacheFactory.create(config)

    k, v = _kv(1, workload.n_kv_heads, 8, workload.head_dim)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, workload.n_kv_heads, 8, workload.head_dim)


@pytest.mark.parametrize("head_dim", [48, 96, 100, 192, 384])
def test_odd_head_dims_produce_buildable_configs(head_dim: int) -> None:
    workload = WorkloadProfile(seq_len=1024, head_dim=head_dim, n_layers=16, n_kv_heads=4)
    result = select_kv_config(workload, HardwareProfile(ram_gb=32))
    config = to_kv_cache_config(result, workload)

    cache = KVCacheFactory.create(config)
    k, v = _kv(1, workload.n_kv_heads, 8, head_dim)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, workload.n_kv_heads, 8, head_dim)
