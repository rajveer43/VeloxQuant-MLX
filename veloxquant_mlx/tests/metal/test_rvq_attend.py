"""Input guards for the prototype fused RVQ decode+attend kernel."""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.metal._rvq_attend import turboquant_fused_rvq_decode_attend


def test_empty_kv_cache_raises_instead_of_dividing_by_zero() -> None:
    q = mx.zeros((1, 2, 1, 64), dtype=mx.float16)
    idx = mx.zeros((1, 2, 0, 64), dtype=mx.uint8)
    v_idx = mx.zeros((1, 2, 0, 8), dtype=mx.uint8)
    cents = mx.zeros((4,), dtype=mx.float32)
    vcb = mx.zeros((4, 8), dtype=mx.float16)
    with pytest.raises(ValueError, match="S_kv == 0"):
        turboquant_fused_rvq_decode_attend(q, idx, idx, cents, cents, v_idx, vcb, 2, 2, 2)
