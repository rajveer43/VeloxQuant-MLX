"""NestedKV's prompt budget must not depend on how prefill is chunked (#878)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

D, H, S, BUDGET = 64, 4, 2048, 512


def _prefill(x: mx.array, chunks: list[int]):
    cache = KVCacheFactory.create(
        KVCacheConfig(method="nestedkv", head_dim=D, nestedkv_budget=BUDGET)
    )
    i = 0
    for n in chunks:
        cache.update_and_fetch(x[:, :, i : i + n], x[:, :, i : i + n])
        i += n
    return cache


@pytest.mark.parametrize("chunks", [[512] * 4, [256] * 8, [1000, 1000, 48]])
def test_retained_rows_independent_of_chunking(chunks) -> None:
    mx.random.seed(0)
    x = mx.random.normal((1, H, S, D)).astype(mx.float16)
    ref = _prefill(x, [S])
    got = _prefill(x, chunks)

    assert ref.tokens_kept == BUDGET
    assert got.tokens_kept == ref.tokens_kept
    assert got.nbytes == ref.nbytes
    assert got.compression_ratio == ref.compression_ratio
    assert got.offset == S


def test_chunked_attention_sees_every_new_row_and_keeps_positions() -> None:
    mx.random.seed(1)
    x = mx.random.normal((1, H, S, D)).astype(mx.float16)
    cache = KVCacheFactory.create(
        KVCacheConfig(method="nestedkv", head_dim=D, nestedkv_budget=BUDGET)
    )
    for i in range(4):
        k, _ = cache.update_and_fetch(
            x[:, :, i * 512 : (i + 1) * 512], x[:, :, i * 512 : (i + 1) * 512]
        )
        # attention sees prior kept rows plus the whole new chunk (#610)
        assert k.shape[2] == (512 if i == 0 else BUDGET + 512)
    pos = cache._positions
    assert pos.shape[-1] == BUDGET
    assert mx.all(pos[..., 1:] > pos[..., :-1]).item()  # ascending original positions
    assert pos[0, 0, :4].tolist() == [0, 1, 2, 3]  # sinks survive every chunk


def test_decode_after_chunked_prefill_still_appends_past_budget() -> None:
    mx.random.seed(2)
    x = mx.random.normal((1, H, S + 3, D)).astype(mx.float16)
    cache = _prefill(x[:, :, :S], [512] * 4)
    for i in range(3):
        cache.update_and_fetch(x[:, :, S + i : S + i + 1], x[:, :, S + i : S + i + 1])
    assert cache.tokens_kept == BUDGET + 3
    # a later multi-token call (e.g. speculative verify) appends, never re-evicts
    cache.update_and_fetch(x[:, :, :8], x[:, :, :8])
    assert cache.tokens_kept == BUDGET + 3 + 8
