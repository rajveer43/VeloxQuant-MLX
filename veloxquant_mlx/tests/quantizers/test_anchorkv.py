"""Tests for AnchorKV-adapted quantizer primitives — anchor-residual compression.

AnchorKV-adapted (arXiv:2608.02901v1, no verified peer-reviewed venue as of
2026-08-20 — a one-time, user-directed exception; see
paper/research/surveys/NEW_METHOD_SURVEY_V22.md) selects a small set of
anchor positions per head, assigns every other token to its nearest anchor
(index + scalar coefficient), and spends a byte budget on quantized
residuals for the tokens whose approximation error costs the most
attention-output error. Unlike every eviction method in this repo, no token
is ever dropped — test_assign_and_project_covers_every_token and
test_anchors_reconstruct_exactly_without_residual prove this directly at
the primitive level. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.anchorkv import (
    RESIDUAL_BITS,
    ResidualCodec,
    allocate_residual_budget,
    anchorkv_budget_slots,
    assign_and_project,
    key_value_utility,
    pack_codes,
    select_anchors,
    unpack_codes,
)


def _rand_kv(S: int, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    return k, v


# ---------------------------------------------------------------------------
# select_anchors
# ---------------------------------------------------------------------------


def test_select_anchors_includes_trailing_window() -> None:
    k, _ = _rand_kv(S=40)
    anchors = select_anchors(k, k=10, window=5, rho=0.7, seed=0)
    anchor_set = set(anchors.tolist())
    assert set(range(35, 40)).issubset(anchor_set)


def test_select_anchors_respects_budget() -> None:
    k, _ = _rand_kv(S=100)
    anchors = select_anchors(k, k=12, window=4, rho=0.5, seed=0)
    assert int(anchors.shape[0]) == 12


def test_select_anchors_budget_capped_at_seq_len() -> None:
    k, _ = _rand_kv(S=8)
    anchors = select_anchors(k, k=100, window=4, rho=0.7, seed=0)
    assert int(anchors.shape[0]) == 8


def test_select_anchors_deterministic() -> None:
    k, _ = _rand_kv(S=50)
    a1 = select_anchors(k, k=10, window=4, rho=0.6, seed=7)
    a2 = select_anchors(k, k=10, window=4, rho=0.6, seed=7)
    assert a1.tolist() == a2.tolist()


def test_select_anchors_sorted_ascending() -> None:
    k, _ = _rand_kv(S=60)
    anchors = select_anchors(k, k=15, window=4, rho=0.7, seed=3)
    idx = anchors.tolist()
    assert idx == sorted(idx)


# ---------------------------------------------------------------------------
# assign_and_project
# ---------------------------------------------------------------------------


def test_assign_and_project_covers_every_token() -> None:
    """Every token gets an assignment/coefficient/residual — none dropped."""
    S = 30
    k, _ = _rand_kv(S=S)
    anchors = select_anchors(k, k=6, window=2, rho=0.7, seed=0)
    result = assign_and_project(k, anchors)
    assert int(result.assign_idx.shape[0]) == S
    assert int(result.gamma.shape[0]) == S
    assert result.residual.shape == (S, 32)


def test_anchors_reconstruct_exactly_without_residual() -> None:
    """An anchor projected onto itself has gamma=1 and zero residual."""
    S = 20
    k, _ = _rand_kv(S=S)
    anchors = select_anchors(k, k=5, window=2, rho=0.7, seed=1)
    result = assign_and_project(k, anchors)
    for local_i, orig_pos in enumerate(anchors.tolist()):
        assigned_anchor_local = int(result.assign_idx[orig_pos].item())
        assert assigned_anchor_local == local_i
        gamma = float(result.gamma[orig_pos].item())
        assert gamma == pytest.approx(1.0, abs=1e-4)
        residual_norm = float(mx.sum(mx.abs(result.residual[orig_pos])).item())
        assert residual_norm == pytest.approx(0.0, abs=1e-3)


def test_projection_reduces_norm_or_matches() -> None:
    """x_tilde + residual == x exactly (orthogonal decomposition, Eq. 2)."""
    S = 25
    k, _ = _rand_kv(S=S, seed=2)
    anchors = select_anchors(k, k=6, window=2, rho=0.7, seed=2)
    result = assign_and_project(k, anchors)
    anchor_vecs = k.astype(mx.float32)[anchors][result.assign_idx]
    x_tilde = result.gamma[:, None] * anchor_vecs
    reconstructed = x_tilde + result.residual
    mse = float(mx.mean((reconstructed - k.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# ResidualCodec
# ---------------------------------------------------------------------------


def test_residual_codec_roundtrip_reduces_error() -> None:
    """Quantized residual should be closer to the true residual than zero is."""
    S = 16
    k, _ = _rand_kv(S=S, seed=3)
    anchors = select_anchors(k, k=4, window=2, rho=0.7, seed=3)
    result = assign_and_project(k, anchors)

    codec = ResidualCodec(head_dim=32, seed=3)
    codes, scale = codec.encode(result.residual)
    decoded = codec.decode(codes, scale)

    mse_with_residual = float(mx.mean((decoded - result.residual) ** 2).item())
    mse_zero = float(mx.mean(result.residual**2).item())
    assert mse_with_residual < mse_zero


def test_residual_codec_bytes_per_residual_matches_bits() -> None:
    import math

    codec = ResidualCodec(head_dim=32, seed=0, bits=2)
    expected = math.ceil(32 * 2 / 8) + 2  # packed codes + fp16 scale
    assert codec.bytes_per_residual == expected


def test_residual_codec_deterministic() -> None:
    S = 10
    k, _ = _rand_kv(S=S, seed=4)
    anchors = select_anchors(k, k=3, window=2, rho=0.7, seed=4)
    result = assign_and_project(k, anchors)

    codec1 = ResidualCodec(head_dim=32, seed=9)
    codec2 = ResidualCodec(head_dim=32, seed=9)
    c1, s1 = codec1.encode(result.residual)
    c2, s2 = codec2.encode(result.residual)
    assert c1.tolist() == c2.tolist()
    assert s1.tolist() == s2.tolist()


def test_residual_codec_non_hadamard_dim_falls_back() -> None:
    """head_dim=17 is not Hadamard-compatible — codec must still work (identity rotation)."""
    S = 8
    k, _ = _rand_kv(S=S, D=17, seed=5)
    anchors = select_anchors(k, k=3, window=2, rho=0.7, seed=5)
    result = assign_and_project(k, anchors)
    codec = ResidualCodec(head_dim=17, seed=5)
    assert codec._rotation is None
    codes, scale = codec.encode(result.residual)
    decoded = codec.decode(codes, scale)
    assert decoded.shape == (S, 17)


# ---------------------------------------------------------------------------
# pack_codes / unpack_codes
# ---------------------------------------------------------------------------


def test_pack_unpack_roundtrip_2bit() -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=(12, 32)).astype(np.uint8)
    packed = pack_codes(codes, bits=2)
    assert packed.shape == (12, 8)  # 32 * 2 bits / 8 = 8 bytes/row
    unpacked = unpack_codes(packed, d=32, bits=2)
    assert np.array_equal(codes, unpacked)


def test_pack_unpack_roundtrip_nondivisible_dim() -> None:
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 4, size=(5, 17), dtype=np.uint8)
    packed = pack_codes(codes, bits=2)
    unpacked = unpack_codes(packed, d=17, bits=2)
    assert np.array_equal(codes, unpacked)


# ---------------------------------------------------------------------------
# key_value_utility
# ---------------------------------------------------------------------------


def test_utility_nonnegative() -> None:
    S = 20
    k, v = _rand_kv(S=S, seed=6)
    anchors = select_anchors(k, k=5, window=3, rho=0.7, seed=6)
    k_assign = assign_and_project(k, anchors)
    v_assign = assign_and_project(v, anchors)
    proxy_q = k[-3:]
    u_k, u_v = key_value_utility(proxy_q, k, v, k_assign.residual, v_assign.residual)
    assert bool(mx.all(u_k >= -1e-6).item())
    assert bool(mx.all(u_v >= -1e-6).item())


def test_utility_zero_for_zero_residual() -> None:
    """A token with zero residual (e.g. an anchor) contributes zero utility."""
    S = 15
    k, v = _rand_kv(S=S, seed=7)
    anchors = select_anchors(k, k=4, window=2, rho=0.7, seed=7)
    k_assign = assign_and_project(k, anchors)
    v_assign = assign_and_project(v, anchors)
    proxy_q = k[-2:]
    u_k, u_v = key_value_utility(proxy_q, k, v, k_assign.residual, v_assign.residual)
    for pos in anchors.tolist():
        assert float(u_k[pos].item()) == pytest.approx(0.0, abs=1e-5)
        assert float(u_v[pos].item()) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# allocate_residual_budget
# ---------------------------------------------------------------------------


def test_allocate_residual_budget_picks_highest_utility() -> None:
    u = mx.array([0.1, 0.9, 0.3, 0.05, 0.7], dtype=mx.float32)
    masks = allocate_residual_budget([u], n_slots=2)
    mask = masks[0]
    chosen = set(i for i in range(5) if bool(mask[i].item()))
    assert chosen == {1, 4}


def test_allocate_residual_budget_pools_across_heads() -> None:
    u_head0 = mx.array([0.9, 0.1], dtype=mx.float32)
    u_head1 = mx.array([0.2, 0.8], dtype=mx.float32)
    masks = allocate_residual_budget([u_head0, u_head1], n_slots=2)
    assert bool(masks[0][0].item()) is True
    assert bool(masks[1][1].item()) is True
    total = sum(int(mx.sum(m.astype(mx.int32)).item()) for m in masks)
    assert total == 2


def test_allocate_residual_budget_zero_slots() -> None:
    u = mx.array([0.5, 0.5], dtype=mx.float32)
    masks = allocate_residual_budget([u], n_slots=0)
    assert int(mx.sum(masks[0].astype(mx.int32)).item()) == 0


def test_allocate_residual_budget_more_slots_than_candidates() -> None:
    u = mx.array([0.5, 0.5, 0.1], dtype=mx.float32)
    masks = allocate_residual_budget([u], n_slots=100)
    assert int(mx.sum(masks[0].astype(mx.int32)).item()) == 3


# ---------------------------------------------------------------------------
# anchorkv_budget_slots
# ---------------------------------------------------------------------------


def test_budget_slots_zero_at_theta_zero() -> None:
    slots = anchorkv_budget_slots(
        seq_len=100, head_dim=64, n_anchor=10, theta=0.0, residual_codec_bytes=18
    )
    assert slots == 0


def test_budget_slots_increase_with_theta() -> None:
    kwargs = dict(seq_len=1000, head_dim=64, n_anchor=20, residual_codec_bytes=18)
    low = anchorkv_budget_slots(theta=0.05, **kwargs)
    high = anchorkv_budget_slots(theta=0.5, **kwargs)
    assert high > low


def test_budget_slots_never_negative() -> None:
    slots = anchorkv_budget_slots(
        seq_len=10, head_dim=128, n_anchor=8, theta=0.001, residual_codec_bytes=18
    )
    assert slots >= 0


def test_residual_bits_default() -> None:
    assert RESIDUAL_BITS == 2
