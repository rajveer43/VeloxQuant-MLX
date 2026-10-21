"""PyramidKV / SqueezeAttention must never evict the row they just received (#839).

The row appended on a step carries score 0 (it has had no step to accumulate
attention mass yet) while every older row has a positive score, so a plain
argmin over the scores picks it every time once the cache is at budget. The
returned cache then lacks the current token, which therefore cannot attend to
itself.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

HEAD_DIM = 16
HEADS = 3
BUDGET = 16
N_SINK = 4
PREFILL = 40
DECODE = 30

ARMS = [
    (
        "pyramidkv-reference",
        {"method": "pyramidkv", "pyramid_budget": BUDGET, "pyramid_backend": "reference"},
    ),
    ("pyramidkv-mlx", {"method": "pyramidkv", "pyramid_budget": BUDGET, "pyramid_backend": "mlx"}),
    (
        "pyramidkv-metal",
        {"method": "pyramidkv", "pyramid_budget": BUDGET, "pyramid_backend": "metal"},
    ),
    ("squeeze", {"method": "squeeze", "squeeze_budget": BUDGET}),
]


def _kv(first_pos: int, steps: int, batch: int = 1):
    """K/V whose channel 0 of K carries the token's absolute position."""
    k = mx.random.normal((batch, HEADS, steps, HEAD_DIM))
    pos = mx.arange(first_pos, first_pos + steps).astype(mx.float32)[None, None, :, None]
    k = mx.concatenate([mx.broadcast_to(pos, (batch, HEADS, steps, 1)), k[..., 1:]], axis=-1)
    return k.astype(mx.float16), mx.random.normal((batch, HEADS, steps, HEAD_DIM)).astype(
        mx.float16
    )


def _positions(keys: mx.array) -> np.ndarray:
    return np.array(keys[..., 0].astype(mx.float32)).round().astype(int)  # [B, H, T]


@pytest.mark.parametrize("name,cfg", ARMS, ids=[a[0] for a in ARMS])
@pytest.mark.parametrize("batch", [1, 2])
def test_newest_token_is_kept_every_decode_step(name, cfg, batch):
    mx.random.seed(0)
    cache = KVCacheFactory.create(KVCacheConfig(head_dim=HEAD_DIM, **cfg))
    k, v = _kv(0, PREFILL, batch)
    cache.update_and_fetch(k, v)

    for step in range(DECODE):
        pos = PREFILL + step
        k, v = _kv(pos, 1, batch)
        keys, _ = cache.update_and_fetch(k, v)
        mx.eval(keys)
        kept = _positions(keys)
        assert (kept == pos).any(axis=-1).all(), f"{name}: token {pos} missing from returned cache"
        assert keys.shape[2] <= BUDGET


@pytest.mark.parametrize("name,cfg", ARMS, ids=[a[0] for a in ARMS])
def test_sinks_survive_and_budget_holds(name, cfg):
    mx.random.seed(1)
    cache = KVCacheFactory.create(KVCacheConfig(head_dim=HEAD_DIM, **cfg))
    k, v = _kv(0, PREFILL)
    cache.update_and_fetch(k, v)
    for step in range(DECODE):
        k, v = _kv(PREFILL + step, 1)
        keys, _ = cache.update_and_fetch(k, v)
    mx.eval(keys)
    assert keys.shape[2] == BUDGET
    for h in range(HEADS):
        assert set(range(N_SINK)) <= set(_positions(keys)[0, h].tolist())


def test_pyramid_backends_agree_after_the_fix():
    """The reference, batched-MLX and fused-Metal paths must make the same decisions."""
    outs = []
    for backend in ("reference", "mlx", "metal"):
        mx.random.seed(2)
        cache = KVCacheFactory.create(
            KVCacheConfig(
                method="pyramidkv",
                head_dim=HEAD_DIM,
                pyramid_budget=BUDGET,
                pyramid_backend=backend,
            )
        )
        k, v = _kv(0, PREFILL)
        cache.update_and_fetch(k, v)
        for step in range(DECODE):
            k, v = _kv(PREFILL + step, 1)
            keys, _ = cache.update_and_fetch(k, v)
        mx.eval(keys)
        outs.append(np.sort(_positions(keys), axis=-1))
    np.testing.assert_array_equal(outs[0], outs[1])
    np.testing.assert_array_equal(outs[0], outs[2])
