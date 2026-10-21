"""RocketKV's retained set must not depend on how prefill is chunked (#845)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

D, H, S = 64, 4, 400


def _prefill(x: mx.array, chunks: list[int]):
    cache = KVCacheFactory.create(KVCacheConfig(method="rocketkv", head_dim=D))
    i = 0
    for n in chunks:
        cache.update_and_fetch(x[:, :, i : i + n], x[:, :, i : i + n])
        i += n
    return cache


@pytest.mark.parametrize("chunks", [[200, 200], [100] * 4, [50] * 8])
def test_retained_rows_independent_of_chunking(chunks) -> None:
    mx.random.seed(0)
    x = mx.random.normal((1, H, S, D)).astype(mx.float16)
    ref = _prefill(x, [S])
    got = _prefill(x, chunks)

    assert got.state[0].shape[2] == ref.state[0].shape[2]
    assert got.tokens_kept == ref.tokens_kept
    assert got.tokens_total == ref.tokens_total
    assert got.offset == S
