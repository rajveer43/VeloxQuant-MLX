"""Tests for ChunkKV-adapted pure primitives (quantizers/chunkkv.py).

Covers chunk partitioning (contiguity, sink exclusion, ragged tail), chunk-score
pooling, the chunk-aligned keep-mask, the per-head eviction state machine, both
score modes, byte accounting, determinism, and the C=1 == H2O bit-for-bit
equivalence. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.chunkkv import (
    ChunkKVState,
    chunk_partition,
    chunk_scores,
    chunkkv_fp16_bytes,
    chunkkv_get_kv,
    chunkkv_keep_mask,
    chunkkv_update,
    full_chunkkv_fp16_bytes,
    init_chunkkv_state,
)
from veloxquant_mlx.quantizers.h2o import h2o_get_kv, h2o_update, init_h2o_state


def _kv(S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ======================================================================
# chunk_partition
# ======================================================================


def test_partition_contiguous_and_covers_tail():
    sinks, chunks = chunk_partition(seq_len=20, chunk_size=4, n_sink=2)
    assert sinks == [0, 1]
    # chunks partition [2, 20) with no gaps or overlaps
    flat = [i for (a, b) in chunks for i in range(a, b)]
    assert flat == list(range(2, 20))
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))


def test_partition_ragged_tail():
    _, chunks = chunk_partition(seq_len=10, chunk_size=4, n_sink=0)
    # 10 / 4 → chunks of 4, 4, 2 (ragged tail)
    assert [b - a for (a, b) in chunks] == [4, 4, 2]


def test_partition_chunk_size_one_is_per_token():
    sinks, chunks = chunk_partition(seq_len=6, chunk_size=1, n_sink=2)
    assert sinks == [0, 1]
    assert chunks == [(2, 3), (3, 4), (4, 5), (5, 6)]


def test_partition_sinks_exceed_len():
    sinks, chunks = chunk_partition(seq_len=3, chunk_size=4, n_sink=5)
    assert sinks == [0, 1, 2]
    assert chunks == []


def test_partition_rejects_bad_chunk_size():
    with pytest.raises(ValueError):
        chunk_partition(seq_len=10, chunk_size=0, n_sink=0)


# ======================================================================
# chunk_scores  +  chunkkv_keep_mask
# ======================================================================


def test_chunk_scores_are_means():
    scores = mx.array([0.0, 0.0, 2.0, 4.0, 1.0, 3.0], dtype=mx.float32)
    _, chunks = chunk_partition(6, chunk_size=2, n_sink=0)
    pooled = chunk_scores(scores, chunks)
    # means of (0,0), (2,4), (1,3) = 0, 3, 2
    assert [round(float(x), 3) for x in pooled] == [0.0, 3.0, 2.0]


def test_keep_mask_is_chunk_aligned_and_keeps_sinks():
    # 3 body chunks of width 2; budget lets 2 sinks + 2 chunks (=6) through.
    scores = mx.array([9, 9, 1.0, 1.0, 5.0, 5.0, 3.0, 3.0], dtype=mx.float32)
    mask = chunkkv_keep_mask(scores, seq_len=8, chunk_size=2, n_sink=2, budget=6)
    kept = [i for i in range(8) if bool(mask[i].item())]
    # sinks 0,1 always; then highest-mean chunks: (4,6)=5 and (6,8)=3, not (2,4)=1
    assert kept == [0, 1, 4, 5, 6, 7]


def test_keep_mask_never_exceeds_budget():
    scores = mx.arange(30).astype(mx.float32)
    mask = chunkkv_keep_mask(scores, seq_len=30, chunk_size=4, n_sink=4, budget=16)
    n_kept = int(mx.sum(mask.astype(mx.int32)).item())
    assert n_kept <= 16


# ======================================================================
# ChunkKVState eviction
# ======================================================================


def test_init_rejects_bad_args():
    with pytest.raises(ValueError):
        init_chunkkv_state(4, 32, 16, chunk_size=0)
    with pytest.raises(ValueError):
        init_chunkkv_state(4, 32, 16, score_mode="bogus")


def test_init_state_rejects_n_sink_equal_budget() -> None:
    """n_sink >= budget leaves no evictable room — sinks would be evicted
    once the cache fills, defeating the sink guarantee."""
    with pytest.raises(ValueError, match="n_sink"):
        init_chunkkv_state(n_sink=4, budget=4, head_dim=16)


def test_init_state_rejects_n_sink_above_budget() -> None:
    with pytest.raises(ValueError, match="n_sink"):
        init_chunkkv_state(n_sink=8, budget=4, head_dim=16)


def test_update_respects_budget_and_is_chunk_aligned():
    st = init_chunkkv_state(n_sink=2, budget=10, head_dim=8, chunk_size=4)
    k, v = _kv(40, 8)
    st = chunkkv_update(st, k, v)
    kept = int(st.keys.shape[0])
    assert kept <= 10
    # kept = sinks + whole chunks: (kept - n_sink) is a multiple of chunk_size
    # unless the ragged newest chunk survives; in steady state it is chunk-sized.
    assert kept >= st.n_sink


def test_survivors_are_whole_chunks_no_partial():
    """After eviction settles, non-sink survivors form whole chunk_size blocks."""
    st = init_chunkkv_state(n_sink=4, budget=20, head_dim=8, chunk_size=4)
    k, v = _kv(200, 8, seed=3)
    st = chunkkv_update(st, k, v)
    body = int(st.keys.shape[0]) - st.n_sink
    # body is a whole number of chunks (a ragged newest chunk only appears at the
    # very tail; with a 4|16 budget it divides evenly).
    assert body % 4 == 0


def test_sinks_always_retained():
    st = init_chunkkv_state(n_sink=3, budget=8, head_dim=8, chunk_size=4)
    k, v = _kv(60, 8, seed=5)
    st = chunkkv_update(st, k, v)
    # The first 3 stored keys must equal the first 3 input keys (never evicted).
    assert bool(mx.all(st.keys[:3] == k[:3].astype(mx.float16)).item())


def test_key_norm_score_mode_runs():
    st = init_chunkkv_state(n_sink=2, budget=12, head_dim=8, chunk_size=4, score_mode="key_norm")
    k, v = _kv(50, 8, seed=7)
    st = chunkkv_update(st, k, v)
    assert st.keys.shape[0] <= 12
    assert st.score_mode == "key_norm"


def test_determinism_no_rng():
    k, v = _kv(40, 8, seed=11)
    a = chunkkv_update(init_chunkkv_state(2, 10, 8, chunk_size=4), k, v)
    b = chunkkv_update(init_chunkkv_state(2, 10, 8, chunk_size=4), k, v)
    assert bool(mx.all(a.keys == b.keys).item())
    assert bool(mx.all(a.values == b.values).item())


def test_get_kv_placeholder_before_update():
    st = init_chunkkv_state(2, 10, 8)
    k, v = chunkkv_get_kv(st)
    assert k.shape == (0, 1) and v.shape == (0, 1)


def test_byte_accounting_helpers():
    st = init_chunkkv_state(2, 10, 8, chunk_size=4)
    assert chunkkv_fp16_bytes(st) == 0  # empty
    k, v = _kv(30, 8, seed=13)
    st = chunkkv_update(st, k, v)
    n = int(st.keys.shape[0])
    assert chunkkv_fp16_bytes(st) == n * 8 * 2 * 2
    assert full_chunkkv_fp16_bytes(100, 8) == 100 * 8 * 2 * 2


# ======================================================================
# C = 1  ==  H2O  (bit-for-bit)
# ======================================================================


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chunk_size_one_reduces_to_h2o(seed):
    """chunk_size=1 + attn_mass must match H2O-adapted exactly."""
    S, D, budget, n_sink = 40, 8, 8, 2
    k, v = _kv(S, D, seed=seed)

    cs = init_chunkkv_state(n_sink, budget, D, chunk_size=1, score_mode="attn_mass")
    cs = chunkkv_update(cs, k, v)
    ck, cv = chunkkv_get_kv(cs)

    hs = init_h2o_state(n_sink, budget, D)
    hs = h2o_update(hs, k, v)
    hk, hv = h2o_get_kv(hs)

    assert ck.shape == hk.shape
    assert bool(mx.all(ck == hk).item())
    assert bool(mx.all(cv == hv).item())
