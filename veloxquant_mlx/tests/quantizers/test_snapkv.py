"""Tests for SnapKV-adapted quantizer primitives — prefill obs-window token eviction.

SnapKV-adapted (arXiv:2404.14469, ICLR 2025) uses the last obs_window key rows
as proxy queries to score all prefix tokens via softmax attention, then retains
only the top-budget token positions (plus sink positions). These tests cover:
observation-window scoring shape/range/clamp, snap_select_indices exact count,
sorted order, sink guarantee, high-score preference, snapkv_compress shape/dtype,
no-eviction edge case, and byte accounting. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.snapkv import (
    SnapKVState,
    full_fp16_bytes,
    obs_window_attention_scores,
    snap_select_indices,
    snapkv_compress,
    snapkv_fp16_bytes,
)


def _rand_kv(S: int = 64, D: int = 128, seed: int = 0) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    V = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    return K, V


# ---------------------------------------------------------------------------
# obs_window_attention_scores
# ---------------------------------------------------------------------------


def test_obs_window_scores_shape() -> None:
    K, _ = _rand_kv(S=64, D=128)
    scores = obs_window_attention_scores(K, obs_window=8)
    assert scores.shape == (64,)


def test_obs_window_scores_dtype_fp32() -> None:
    K, _ = _rand_kv(S=32, D=64)
    scores = obs_window_attention_scores(K, obs_window=4)
    assert scores.dtype == mx.float32


def test_obs_window_scores_in_range() -> None:
    """Scores are in [0, 1] (they are mean of softmax rows)."""
    K, _ = _rand_kv(S=64, D=128)
    scores = obs_window_attention_scores(K, obs_window=8)
    s = scores.tolist()
    assert all(0.0 <= v <= 1.0 + 1e-5 for v in s)


def test_obs_window_clamp_large_window() -> None:
    """obs_window > S must not raise — clamped to S internally."""
    K, _ = _rand_kv(S=16, D=32)
    scores = obs_window_attention_scores(K, obs_window=999)
    assert scores.shape == (16,)


def test_obs_window_single_token() -> None:
    """S == 1 edge case: single token scores without error."""
    K, _ = _rand_kv(S=1, D=32)
    scores = obs_window_attention_scores(K, obs_window=4)
    assert scores.shape == (1,)


# ---------------------------------------------------------------------------
# snap_select_indices
# ---------------------------------------------------------------------------


def test_select_count_exact() -> None:
    scores = mx.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7], dtype=mx.float32)
    idx = snap_select_indices(scores, budget=3, n_sink=1)
    assert idx.shape[0] == 3


def test_select_sorted_ascending() -> None:
    scores = mx.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7], dtype=mx.float32)
    idx = snap_select_indices(scores, budget=4, n_sink=1)
    idx_list = idx.tolist()
    assert idx_list == sorted(idx_list)


def test_select_sink_always_included() -> None:
    """First n_sink positions must always appear in the result."""
    scores = mx.zeros((20,), dtype=mx.float32)  # equal scores — sinks must win
    idx = snap_select_indices(scores, budget=6, n_sink=3)
    idx_list = idx.tolist()
    assert 0 in idx_list
    assert 1 in idx_list
    assert 2 in idx_list


def test_select_budget_ge_S_keeps_all() -> None:
    scores = mx.array([0.1, 0.5, 0.3], dtype=mx.float32)
    idx = snap_select_indices(scores, budget=10, n_sink=1)
    assert idx.shape[0] == 3
    assert sorted(idx.tolist()) == [0, 1, 2]


def test_select_high_score_tokens_preferred() -> None:
    """Tokens at positions 3 and 4 have the highest scores — must be selected."""
    scores = mx.array([0.01, 0.02, 0.03, 0.90, 0.85, 0.04, 0.05], dtype=mx.float32)
    # budget=3, n_sink=1 → sink keeps pos 0, then top-2 non-sink → pos 3 and 4
    idx = snap_select_indices(scores, budget=3, n_sink=1)
    idx_list = idx.tolist()
    assert 3 in idx_list
    assert 4 in idx_list


def test_select_n_sink_zero() -> None:
    """n_sink=0 runs without error and returns budget indices."""
    scores = mx.array([0.1, 0.9, 0.3, 0.8], dtype=mx.float32)
    idx = snap_select_indices(scores, budget=2, n_sink=0)
    assert idx.shape[0] == 2


# ---------------------------------------------------------------------------
# snapkv_compress
# ---------------------------------------------------------------------------


def test_compress_output_shape() -> None:
    K, V = _rand_kv(S=64, D=128)
    state = snapkv_compress(K, V, budget=16, obs_window=8, n_sink=2)
    assert state.kept_keys.shape == (16, 128)
    assert state.kept_values.shape == (16, 128)


def test_compress_output_dtype_fp16() -> None:
    K, V = _rand_kv(S=32, D=64)
    state = snapkv_compress(K, V, budget=8, obs_window=4)
    assert state.kept_keys.dtype == mx.float16
    assert state.kept_values.dtype == mx.float16


def test_compress_n_kept_matches_budget() -> None:
    K, V = _rand_kv(S=64, D=128)
    state = snapkv_compress(K, V, budget=20, obs_window=8)
    assert state.n_kept == 20
    assert state.n_original == 64


def test_compress_no_eviction_short_seq() -> None:
    """budget >= S: all tokens kept, n_kept == S."""
    K, V = _rand_kv(S=10, D=32)
    state = snapkv_compress(K, V, budget=50, obs_window=4, n_sink=2)
    assert state.n_kept == 10
    assert state.kept_keys.shape[0] == 10


def test_compress_edge_n_sink_zero_obs1() -> None:
    """n_sink=0, obs_window=1 edge-case runs without error."""
    K, V = _rand_kv(S=32, D=64)
    state = snapkv_compress(K, V, budget=8, obs_window=1, n_sink=0)
    assert state.n_kept == 8


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_snapkv_fp16_bytes_formula() -> None:
    K, V = _rand_kv(S=32, D=128)
    state = snapkv_compress(K, V, budget=16, obs_window=4)
    expected = 16 * 128 * 2 * 2  # n_kept * D * 2 (K+V) * 2 (fp16)
    assert snapkv_fp16_bytes(state) == expected


def test_full_fp16_bytes_formula() -> None:
    assert full_fp16_bytes(64, 128) == 64 * 128 * 2 * 2
