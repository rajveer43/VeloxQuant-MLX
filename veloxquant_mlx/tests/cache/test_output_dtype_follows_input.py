"""#622: every mlx_lm-protocol cache returns K/V in the input dtype.

A bf16 model must not get fp16 K/V back: values above 65504 overflow, and
SDPA with bf16 queries and fp16 keys silently promotes attention to fp32.
"""

from __future__ import annotations

import typing

import mlx.core as mx
import pytest

from veloxquant_mlx.cache import base
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

_METHODS = list(typing.get_args(base.MethodName))


def _create(method: str):
    cache = KVCacheFactory.create(KVCacheConfig(method=method, head_dim=64, bit_width_inlier=4))
    if not hasattr(cache, "update_and_fetch"):
        pytest.skip(f"{method} is a standalone cache without update_and_fetch")
    return cache


@pytest.mark.parametrize("method", _METHODS)
def test_bf16_input_returns_bf16(method):
    cache = _create(method)
    mx.random.seed(0)
    k = mx.random.normal((1, 2, 40, 64)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, 40, 64)).astype(mx.bfloat16)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.dtype == mx.bfloat16
    assert v_out.dtype == mx.bfloat16
