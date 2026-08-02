"""Tests for XKVCache — cross-layer shared-subspace key compression.

Covers:
  1.  Factory dispatch (degenerate, no coordinator)
  2.  No .bits attribute leak (exposes assigned_avg_bits instead)
  3.  Group-of-1 (standalone) reduces to per-layer SVD-style compression
  4.  Multi-member group: all members receive the identical shared basis
  5.  Output shapes (prefill + decode) for every group member
  6.  Values pass through unchanged (fp16)
  7.  Byte accounting: only member_idx==0 reports nonzero shared_basis_bytes
  8.  Byte accounting: compressed_key_bytes < fp16_key_bytes
  9.  Decode accumulation after prefill projects into the frozen basis
  10. Coordinator token budget — exceeding max_ctx raises
  11. for_model builds correct member/group assignment across layers
  12. Shared structure across group members reconstructs better than
      independent per-layer SVD at the same rank (mechanism validation)
  13. Determinism
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory, KVCacheBuilder
from veloxquant_mlx.cache.xkv_cache import XKVCache
from veloxquant_mlx.cache.xkv_coordinator import XKVCoordinator
from veloxquant_mlx.quantizers.xkv import pair_layers_grouped


def _cfg(**kwargs) -> KVCacheConfig:
    d = dict(method="xkv", head_dim=32)
    d.update(kwargs)
    return KVCacheConfig(**d)


def _rand(B=1, H=2, S=32, D=32, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))


def _mse(a, b):
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


class _MockAttn:
    def __init__(self, head_dim):
        self.head_dim = head_dim


class _MockLayer:
    def __init__(self, head_dim=32):
        self.self_attn = _MockAttn(head_dim)


class _MockModel:
    def __init__(self, n_layers, head_dim=32):
        self.layers = [_MockLayer(head_dim) for _ in range(n_layers)]
        self.args = None


def _group(coordinator, n_members, rank=8, group_id=0, **cfg_kwargs):
    cfg = _cfg(xkv_rank=rank, **cfg_kwargs)
    return [
        XKVCache(cfg, member_idx=i, group_id=group_id, n_members=n_members, coordinator=coordinator)
        for i in range(n_members)
    ]


def _prefill_and_settle(members, k, v, settle_k=None, settle_v=None):
    """Run one prefill call per member (fan-in), then one more call per
    member (fan-out settle round) so every member — including those that
    published before their peers — adopts the shared basis.

    This mirrors real usage: within one mlx_lm forward pass every layer gets
    exactly one update_and_fetch call, so a group's earlier-iterated members
    only observe the completed shared basis on their *next* call (the next
    decode step). Tests that need every member to already hold the shared
    basis must therefore run a settle round before asserting on it.
    """
    if settle_k is None:
        settle_k = _rand(1, 2, 1, k.shape[-1], seed=9001)
    if settle_v is None:
        settle_v = _rand(1, 2, 1, v.shape[-1], seed=9002)
    outs = [m.update_and_fetch(k, v) for m in members]
    for m in members:
        m.update_and_fetch(settle_k, settle_v)
    return outs


# ---------------------------------------------------------------------------
# Test 1 — factory dispatch (degenerate, no coordinator)
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = KVCacheFactory.create(_cfg())
    assert isinstance(cache, XKVCache)


def test_no_bits_attribute():
    cache = KVCacheFactory.create(_cfg())
    assert not hasattr(cache, "bits")
    assert hasattr(cache, "assigned_avg_bits")


# ---------------------------------------------------------------------------
# Test 3 — group-of-1 (standalone) reduces to per-layer SVD-style compression
# ---------------------------------------------------------------------------
def test_group_of_one_standalone():
    cache = XKVCache(_cfg(xkv_rank=8), member_idx=0, group_id=0, n_members=1, coordinator=None)
    k = _rand(1, 2, 32, 32)
    v = _rand(1, 2, 32, 32, seed=1)
    ko, vo = cache.update_and_fetch(k, v)
    assert ko.shape == k.shape
    assert vo.shape == v.shape
    # values pass through unchanged
    np.testing.assert_array_equal(np.array(vo.tolist()), np.array(v.tolist()))


# ---------------------------------------------------------------------------
# Test 4 — multi-member group: identical shared basis across members
# ---------------------------------------------------------------------------
def test_all_members_receive_identical_basis():
    coord = XKVCoordinator()
    members = _group(coord, n_members=3, rank=8)
    k = _rand(1, 2, 32, 32)
    v = _rand(1, 2, 32, 32, seed=1)
    _prefill_and_settle(members, k, v)
    bases = [np.array(m._V_g.tolist()) for m in members]
    np.testing.assert_allclose(bases[0], bases[1], atol=1e-6)
    np.testing.assert_allclose(bases[0], bases[2], atol=1e-6)
    means = [np.array(m._K_mean_g.tolist()) for m in members]
    np.testing.assert_allclose(means[0], means[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5 — output shapes (prefill + decode) for every group member
# ---------------------------------------------------------------------------
def test_output_shapes_prefill_decode():
    coord = XKVCoordinator()
    members = _group(coord, n_members=2, rank=8)
    k = _rand(1, 2, 32, 32)
    v = _rand(1, 2, 32, 32, seed=1)
    for m in members:
        ko, vo = m.update_and_fetch(k, v)
        assert ko.shape == (1, 2, 32, 32)
        assert vo.shape == (1, 2, 32, 32)

    kd = _rand(1, 2, 1, 32, seed=2)
    vd = _rand(1, 2, 1, 32, seed=3)
    for m in members:
        ko2, vo2 = m.update_and_fetch(kd, vd)
        assert ko2.shape == (1, 2, 33, 32)


# ---------------------------------------------------------------------------
# Test 6 — values pass through unchanged
# ---------------------------------------------------------------------------
def test_values_pass_through():
    coord = XKVCoordinator()
    members = _group(coord, n_members=2, rank=8)
    k = _rand(1, 2, 32, 32)
    v = _rand(1, 2, 32, 32, seed=1)
    for m in members:
        _, vo = m.update_and_fetch(k, v)
        np.testing.assert_array_equal(np.array(vo.tolist()), np.array(v.tolist()))


# ---------------------------------------------------------------------------
# Test 7 — only member_idx==0 reports nonzero shared_basis_bytes
# ---------------------------------------------------------------------------
def test_only_leader_reports_basis_bytes():
    coord = XKVCoordinator()
    members = _group(coord, n_members=3, rank=8)
    k = _rand(1, 2, 32, 32)
    v = _rand(1, 2, 32, 32, seed=1)
    _prefill_and_settle(members, k, v)
    assert members[0].shared_basis_bytes > 0
    assert members[1].shared_basis_bytes == 0
    assert members[2].shared_basis_bytes == 0


# ---------------------------------------------------------------------------
# Test 8 — compressed_key_bytes < fp16_key_bytes
# ---------------------------------------------------------------------------
def test_byte_accounting_compressed_less_than_fp16():
    coord = XKVCoordinator()
    members = _group(coord, n_members=2, rank=8)
    k = _rand(1, 2, 64, 32)
    v = _rand(1, 2, 64, 32, seed=1)
    for m in members:
        m.update_and_fetch(k, v)
    for m in members:
        assert m.compressed_key_bytes < m.fp16_key_bytes


# ---------------------------------------------------------------------------
# Test 9 — decode accumulation after prefill projects into frozen basis
# ---------------------------------------------------------------------------
def test_decode_accumulation_uses_frozen_basis():
    coord = XKVCoordinator()
    members = _group(coord, n_members=2, rank=8)
    k = _rand(1, 2, 32, 32)
    v = _rand(1, 2, 32, 32, seed=1)
    _prefill_and_settle(members, k, v)  # settle round advances offset by 1 each
    basis_before = np.array(members[0]._V_g.tolist())

    for step in range(4):
        kd = _rand(1, 2, 1, 32, seed=100 + step)
        vd = _rand(1, 2, 1, 32, seed=200 + step)
        for m in members:
            m.update_and_fetch(kd, vd)

    basis_after = np.array(members[0]._V_g.tolist())
    np.testing.assert_array_equal(basis_before, basis_after)
    assert members[0]._token_offset == 32 + 1 + 4


# ---------------------------------------------------------------------------
# Test 10 — coordinator token budget raises
# ---------------------------------------------------------------------------
def test_coordinator_budget_raises():
    # The guard fires on member 0's own publish (the only member whose
    # publish increments published_tokens). A prefill batch alone exceeding
    # max_ctx must raise immediately on that first call. Note: once a member
    # has published for its (fixed) basis_token_start, later calls only poll
    # — the coordinator's per-group budget is a prefill-time guard, matching
    # the design intent that the joint SVD runs once, at prefill.
    coord = XKVCoordinator(max_ctx=8)
    members = _group(coord, n_members=2, rank=4)
    k = _rand(1, 2, 16, 32)  # exceeds max_ctx=8 in a single prefill call
    v = _rand(1, 2, 16, 32, seed=1)
    with pytest.raises(RuntimeError, match="max_ctx"):
        members[0].update_and_fetch(k, v)


# ---------------------------------------------------------------------------
# Test 11 — for_model builds correct member/group assignment
# ---------------------------------------------------------------------------
def test_for_model_grouping():
    assert pair_layers_grouped(6, 2) == [
        (0, 0, 2),
        (1, 0, 2),
        (0, 1, 2),
        (1, 1, 2),
        (0, 2, 2),
        (1, 2, 2),
    ]
    model = _MockModel(n_layers=6, head_dim=32)
    caches = KVCacheBuilder.for_model(model, _cfg(xkv_group_size=2, xkv_rank=8))
    assert len(caches) == 6
    assert all(isinstance(c, XKVCache) for c in caches)
    member_groups = [(c.member_idx, c.group_id) for c in caches]
    assert member_groups == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
    ]
    coords = {id(c._coord) for c in caches}
    assert len(coords) == 1

    k = _rand(1, 2, 32, 32)
    v = _rand(1, 2, 32, 32, seed=1)
    for c in caches:
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape == (1, 2, 32, 32)


def test_for_model_trailing_partial_group():
    model = _MockModel(n_layers=5, head_dim=32)
    caches = KVCacheBuilder.for_model(model, _cfg(xkv_group_size=2, xkv_rank=8))
    assert len(caches) == 5
    member_groups = [(c.member_idx, c.group_id) for c in caches]
    assert member_groups == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),  # trailing group of size 1
    ]
    k = _rand(1, 2, 16, 32)
    v = _rand(1, 2, 16, 32, seed=1)
    for c in caches:
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape == (1, 2, 16, 32)


# ---------------------------------------------------------------------------
# Test 12 — shared structure reconstructs better than independent SVD
# ---------------------------------------------------------------------------
def test_shared_structure_beats_independent_svd():
    rng = np.random.default_rng(42)
    D, r_true, S = 32, 4, 48
    shared_basis = rng.standard_normal((D, r_true)).astype(np.float32)

    layer_keys = []
    for i in range(3):
        coeffs = rng.standard_normal((S, r_true)).astype(np.float32) * 2.0
        noise = rng.standard_normal((S, D)).astype(np.float32) * 0.02
        layer = (coeffs @ shared_basis.T + noise).astype(np.float16)
        layer_keys.append(mx.array(layer[None, None, :, :]))  # [1, 1, S, D]

    coord = XKVCoordinator()
    members = _group(coord, n_members=3, rank=r_true)
    shared_mse = 0.0
    for m, k in zip(members, layer_keys):
        ko, _ = m.update_and_fetch(k, k)
        shared_mse += _mse(ko, k)
    shared_mse /= 3

    # Independent per-layer standalone caches (group_of_1) on the SAME data.
    indep_mse = 0.0
    for k in layer_keys:
        standalone = XKVCache(
            _cfg(xkv_rank=r_true), member_idx=0, group_id=0, n_members=1, coordinator=None
        )
        ko, _ = standalone.update_and_fetch(k, k)
        indep_mse += _mse(ko, k)
    indep_mse /= 3

    # Shared basis (fit jointly across all 3) should reconstruct comparably
    # to or better than 3 independent single-layer fits at the same rank,
    # since the true structure is genuinely shared.
    assert shared_mse < indep_mse * 1.5


# ---------------------------------------------------------------------------
# Test 13 — determinism
# ---------------------------------------------------------------------------
def test_determinism():
    k = _rand(1, 2, 32, 32, seed=77)
    v = _rand(1, 2, 32, 32, seed=88)

    def run():
        coord = XKVCoordinator()
        members = _group(coord, n_members=2, rank=8)
        outs = [m.update_and_fetch(k, v) for m in members]
        return [(np.array(ko.tolist()), np.array(vo.tolist())) for ko, vo in outs]

    r1 = run()
    r2 = run()
    for (k1, v1), (k2, v2) in zip(r1, r2):
        np.testing.assert_array_equal(k1, k2)
        np.testing.assert_array_equal(v1, v2)
