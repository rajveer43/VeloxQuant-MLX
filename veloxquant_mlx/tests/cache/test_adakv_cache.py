"""Tests for AdaKVCache — per-head adaptive bit allocation over KIVI.

22 tests covering:
  1.  Factory dispatch via KVCacheFactory
  2.  Output shape preserved after prefill
  3.  Output shape preserved after decode
  4.  Values unchanged (AdaKV-proxy compresses keys only)
  5.  High-importance heads get more bits than low-importance heads
  6.  Average bits equals target (within ±0.5 due to rounding)
  7.  Equal importance degrades to uniform target allocation
  8.  MSE lower on the high-importance head than if it had been given lo_bit
  9.  Running norm accumulator correctness vs ground-truth variance
  10. Decode after prefill — sequential accumulation produces correct shape
  11. Byte accounting — compressed_key_bytes < fp16_key_bytes
  12. assigned_avg_bits within [lo_bit, hi_bit]
  13. Single-head model — trivially assigns target_avg_bits (snapped)
  14. Determinism — identical inputs produce identical outputs

Regression tests for issue #31 (allocator paper-fidelity fixes):
  15. Default config is actually adaptive; degenerate target warns
  16. Allocation is monotone in importance (no saturation)
  17. Allocation is permutation-equivariant over distinct importances
  18. Budget met exactly where the allowed set permits
  19. attention_entropy carries the paper's sign; norm_variance is
      anti-correlated with it (pinned as a documented distinction)
  20. attention_entropy mode wired end-to-end; invalid mode rejected
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.adakv_cache import AdaKVCache
from veloxquant_mlx.quantizers.adakv import (
    allocate_head_bits,
    compute_head_attention_entropy,
    compute_head_norm_variance,
    quantize_head,
)
from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cfg(**kwargs) -> KVCacheConfig:
    defaults = dict(method="adakv", head_dim=64)
    defaults.update(kwargs)
    return KVCacheConfig(**defaults)


def _keys(B=1, H=4, S=32, D=64, seed=0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))


def _values(B=1, H=4, S=32, D=64, seed=1) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))


def _heterogeneous_keys(B=1, H=4, S=64, D=64, seed=3) -> mx.array:
    """Keys where head importance (inter-token norm variance) increases with h.

    Head h gets its per-token norm scaled by an h-dependent jittered factor,
    so higher heads have larger inter-token norm variance.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((B, H, S, D)).astype(np.float32)
    for h in range(H):
        # Per-token magnitude scale; spread grows with h → norm variance grows.
        spread = 0.05 + h * 0.6
        scale = (1.0 + spread * rng.standard_normal((B, S, 1))).astype(np.float32)
        data[:, h, :, :] = data[:, h, :, :] * np.abs(scale)
    return mx.array(data.astype(np.float16))


# ---------------------------------------------------------------------------
# Test 1 — factory dispatch
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = KVCacheFactory.create(_make_cfg())
    assert isinstance(cache, AdaKVCache)


# ---------------------------------------------------------------------------
# Test 2 — output shape after prefill
# ---------------------------------------------------------------------------
def test_output_shape_prefill():
    cache = AdaKVCache(_make_cfg())
    k = _keys(B=1, H=4, S=32, D=64)
    v = _values(B=1, H=4, S=32, D=64)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == (1, 4, 32, 64)
    assert v_out.shape == (1, 4, 32, 64)


# ---------------------------------------------------------------------------
# Test 3 — output shape after decode
# ---------------------------------------------------------------------------
def test_output_shape_decode():
    cache = AdaKVCache(_make_cfg())
    cache.update_and_fetch(_keys(B=1, H=4, S=16, D=64), _values(B=1, H=4, S=16, D=64))
    k_dec = _keys(B=1, H=4, S=1, D=64, seed=99)
    v_dec = _values(B=1, H=4, S=1, D=64, seed=100)
    k_out, v_out = cache.update_and_fetch(k_dec, v_dec)
    assert k_out.shape == (1, 4, 17, 64)
    assert v_out.shape == (1, 4, 17, 64)


# ---------------------------------------------------------------------------
# Test 4 — values unchanged
# ---------------------------------------------------------------------------
def test_values_unchanged():
    cache = AdaKVCache(_make_cfg())
    k = _keys()
    v = _values()
    _, v_out = cache.update_and_fetch(k, v)
    assert np.allclose(
        np.array(v_out[0, 0, :, :].tolist()),
        np.array(v[0, 0, :, :].tolist()),
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Test 5 — high-importance heads get more bits than low-importance heads
# ---------------------------------------------------------------------------
def test_high_importance_heads_get_more_bits():
    cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=3.0, head_dim=64))
    k = _heterogeneous_keys(B=1, H=4, S=64, D=64)
    v = _values(B=1, H=4, S=64, D=64)
    cache.update_and_fetch(k, v)
    bits = cache.head_bits
    # Head 0 (lowest norm variance) should not exceed head 3 (highest).
    assert bits[3] >= bits[0], f"head_bits={bits} — high-importance head got fewer bits"
    assert bits[3] > bits[0], (
        f"head_bits={bits} — expected strictly more bits on the high-importance head"
    )


# ---------------------------------------------------------------------------
# Test 6 — average bits ≈ target (within ±0.5)
# ---------------------------------------------------------------------------
def test_average_bits_matches_target():
    for target in (2.0, 2.5, 3.0):
        cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=target, head_dim=64))
        k = _heterogeneous_keys(B=1, H=8, S=64, D=64, seed=11)
        v = _values(B=1, H=8, S=64, D=64, seed=12)
        cache.update_and_fetch(k, v)
        assert abs(cache.assigned_avg_bits - target) <= 0.5, (
            f"target={target}, assigned={cache.assigned_avg_bits}, bits={cache.head_bits}"
        )


# ---------------------------------------------------------------------------
# Test 7 — equal importance degrades to uniform target allocation
# ---------------------------------------------------------------------------
def test_equal_importance_uniform_allocation():
    """Equal importance → uniform allocation at the target.

    Uses target=3.0, strictly inside (lo=2, hi=4), so the uniform result
    reflects the *absence of an importance signal* rather than the degenerate
    endpoint case. (This test previously asserted [2,2,2,2] at target=2.0,
    which pinned the target==lo_bit degeneracy as correct behaviour and
    masked it — see test_degenerate_target_warns_and_is_uniform.)
    """
    bits = allocate_head_bits(
        head_importance=[1.0, 1.0, 1.0, 1.0],
        target_avg_bits=3.0,
        allowed_bits=[2, 3, 4],
        n_heads=4,
    )
    assert bits == [3, 3, 3, 3], f"expected uniform at target, got {bits}"

    # All-zero importance → also uniform.
    bits_zero = allocate_head_bits(
        head_importance=[0.0, 0.0, 0.0, 0.0],
        target_avg_bits=3.0,
        allowed_bits=[2, 3, 4],
        n_heads=4,
    )
    assert bits_zero == [3, 3, 3, 3], f"expected uniform at target, got {bits_zero}"


# ---------------------------------------------------------------------------
# Test 8 — high-importance head: assigned bits give lower MSE than lo_bit
# ---------------------------------------------------------------------------
def test_high_importance_head_lower_mse_than_lo_bit():
    cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=3.0, head_dim=64))
    k = _heterogeneous_keys(B=1, H=4, S=64, D=64, seed=21)
    v = _values(B=1, H=4, S=64, D=64, seed=22)
    cache.update_and_fetch(k, v)

    bits = cache.head_bits
    hi_head = int(np.argmax(cache.head_importance))
    assert bits[hi_head] > cache._lo_bit, (
        f"high-importance head {hi_head} got only {bits[hi_head]} bits; bits={bits}"
    )

    orig = np.array(k[0, hi_head].tolist())
    recon_assigned = np.array(quantize_head(k[0, hi_head], bits[hi_head], 32).tolist())
    recon_lo = np.array(_group_quant_dequant(k[0, hi_head], cache._lo_bit, 32).tolist())

    mse_assigned = float(np.mean((orig - recon_assigned) ** 2))
    mse_lo = float(np.mean((orig - recon_lo) ** 2))
    assert mse_assigned < mse_lo, (
        f"assigned-bit MSE {mse_assigned:.6f} should be < lo_bit MSE {mse_lo:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 9 — running norm accumulator correctness vs ground truth
# ---------------------------------------------------------------------------
def test_running_norm_accumulator_correctness():
    H, S, D = 3, 50, 64
    rng = np.random.default_rng(33)
    data = rng.standard_normal((1, H, S, D)).astype(np.float32)
    keys = mx.array(data)

    # Ground-truth inter-token norm variance per head.
    norms = np.sqrt((data[0] ** 2).sum(axis=-1))  # [H, S]
    gt_var = norms.var(axis=1)  # [H]

    cache = AdaKVCache(_make_cfg(head_dim=D))
    cache._update_norm_accumulators(keys)
    acc_var = np.array(cache.head_importance)

    np.testing.assert_allclose(acc_var, gt_var, rtol=1e-3, atol=1e-3)

    # And the standalone quantizer helper agrees.
    direct = np.array(compute_head_norm_variance(keys).tolist())
    np.testing.assert_allclose(direct, gt_var, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 10 — decode after prefill accumulates shapes correctly
# ---------------------------------------------------------------------------
def test_decode_after_prefill_accumulates():
    cache = AdaKVCache(_make_cfg())
    cache.update_and_fetch(_keys(B=1, H=4, S=20, D=64), _values(B=1, H=4, S=20, D=64))
    for step in range(5):
        k_dec = _keys(B=1, H=4, S=1, D=64, seed=200 + step)
        v_dec = _values(B=1, H=4, S=1, D=64, seed=300 + step)
        k_out, v_out = cache.update_and_fetch(k_dec, v_dec)
        expected_S = 20 + step + 1
        assert k_out.shape[2] == expected_S
        assert v_out.shape[2] == expected_S
    # Accumulated token count reflects prefill + decode steps.
    assert cache._n_tokens == 25


# ---------------------------------------------------------------------------
# Test 11 — byte accounting: compressed < fp16
# ---------------------------------------------------------------------------
def test_byte_accounting_compressed_less_than_fp16():
    cache = AdaKVCache(_make_cfg())
    cache.update_and_fetch(_keys(B=1, H=4, S=64, D=64), _values(B=1, H=4, S=64, D=64))
    assert cache.compressed_key_bytes < cache.fp16_key_bytes, (
        f"compressed={cache.compressed_key_bytes} should be < fp16={cache.fp16_key_bytes}"
    )


# ---------------------------------------------------------------------------
# Test 12 — assigned_avg_bits within [lo_bit, hi_bit]
# ---------------------------------------------------------------------------
def test_assigned_avg_bits_in_range():
    cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=3.0))
    cache.update_and_fetch(_heterogeneous_keys(B=1, H=4, S=64, D=64), _values(B=1, H=4, S=64, D=64))
    avg = cache.assigned_avg_bits
    assert cache._lo_bit <= avg <= cache._hi_bit, (
        f"assigned_avg_bits={avg} out of range [{cache._lo_bit}, {cache._hi_bit}]"
    )


# ---------------------------------------------------------------------------
# Test 13 — single-head model trivially assigns target (snapped)
# ---------------------------------------------------------------------------
def test_single_head_assigns_target():
    cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=3.0, head_dim=64))
    cache.update_and_fetch(_keys(B=1, H=1, S=32, D=64), _values(B=1, H=1, S=32, D=64))
    # Target 3.0 snaps exactly to allowed mid_bit=3.
    assert cache.head_bits == [3], f"single-head bits={cache.head_bits}"

    # A non-allowed target snaps to the nearest allowed value.
    bits = allocate_head_bits([5.0], target_avg_bits=2.4, allowed_bits=[2, 3, 4], n_heads=1)
    assert bits == [2], f"expected [2], got {bits}"


# ---------------------------------------------------------------------------
# Test 14 — determinism
# ---------------------------------------------------------------------------
def test_determinism():
    k = _heterogeneous_keys(B=1, H=4, S=32, D=64, seed=77)
    v = _values(B=1, H=4, S=32, D=64, seed=88)

    cache1 = AdaKVCache(_make_cfg())
    k_out1, v_out1 = cache1.update_and_fetch(k, v)
    cache2 = AdaKVCache(_make_cfg())
    k_out2, v_out2 = cache2.update_and_fetch(k, v)

    np.testing.assert_array_equal(np.array(k_out1.tolist()), np.array(k_out2.tolist()))
    np.testing.assert_array_equal(np.array(v_out1.tolist()), np.array(v_out2.tolist()))
    assert cache1.head_bits == cache2.head_bits


# ===========================================================================
# Regression tests for issue #31 — allocator paper-fidelity fixes
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 15 — adaptation is non-trivial at the DEFAULT config
# ---------------------------------------------------------------------------
def test_default_config_is_adaptive():
    """The shipped default must actually adapt per head.

    Regression for issue #31 finding 1: the previous default
    (adakv_target_avg_bits=2.0 with lo_bit=2) sat exactly on the floor of the
    allowed set, so every head was forced to lo_bit for *every* importance
    vector — silently identical to plain KIVI while the docs advertised
    per-head adaptation. No prior test exercised the default.
    """
    cache = AdaKVCache(_make_cfg())          # defaults only — no overrides
    k = _heterogeneous_keys(B=1, H=8, S=64, D=64, seed=31)
    v = _values(B=1, H=8, S=64, D=64, seed=32)
    cache.update_and_fetch(k, v)

    bits = cache.head_bits
    assert len(set(bits)) > 1, (
        f"default config produced a uniform allocation {bits} — no per-head "
        f"adaptation, equivalent to plain KIVI"
    )


def test_degenerate_target_warns_and_is_uniform():
    """target at an endpoint of allowed_bits forces uniform — and must warn.

    Adaptation requires headroom on both sides: raising one head must be
    payable by lowering another. At an endpoint no such pair exists. This is
    unavoidable, so the contract is that it is *loud*, never silent.
    """
    import pytest

    for target in (2.0, 4.0):               # lo and hi of {2, 3, 4}
        with pytest.warns(UserWarning, match="no per-head"):
            bits = allocate_head_bits(
                head_importance=[100.0, 1.0, 1.0, 1.0],
                target_avg_bits=target,
                allowed_bits=[2, 3, 4],
                n_heads=4,
            )
        assert len(set(bits)) == 1, f"expected uniform at target={target}, got {bits}"


# ---------------------------------------------------------------------------
# Test 16 — allocation is monotone in importance
# ---------------------------------------------------------------------------
def test_allocation_monotone_in_importance():
    """Raising one head's importance must never lower its bit-width.

    Regression for issue #31 finding 2: clamp-before-normalize saturated the
    real-valued budget, so importance ratios spanning 4 orders of magnitude
    produced byte-identical allocations — the allocator could not distinguish
    "somewhat important" from "overwhelmingly important".
    """
    prev = -1
    seen = set()
    for imp0 in (0.1, 1.0, 2.0, 5.0, 10.0, 100.0, 1e4):
        bits = allocate_head_bits(
            head_importance=[imp0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            target_avg_bits=2.5,
            allowed_bits=[2, 3, 4],
            n_heads=8,
        )
        assert bits[0] >= prev, (
            f"imp0={imp0} lowered head 0 from {prev} to {bits[0]} bits "
            f"despite higher importance"
        )
        prev = bits[0]
        seen.add(tuple(bits))

    assert len(seen) > 1, (
        "allocation never changed across a 5-order-of-magnitude importance "
        "sweep — the importance signal is being discarded"
    )


# ---------------------------------------------------------------------------
# Test 17 — distinct importances give a permutation-equivariant allocation
# ---------------------------------------------------------------------------
def test_allocation_permutation_equivariant():
    """Permuting the head axis must permute the allocation identically.

    Regression for issue #31 finding 4: greedy correction scanned heads in
    index order and kept the first strict minimum, so [10,1,1,1] -> [4,3,3,2]
    but [1,1,1,10] -> [3,3,2,4] — which head was starved depended on position.

    Uses distinct importances: with exact ties and an indivisible budget some
    tied head must lose a bit regardless, which is an integer-allocation fact
    rather than an ordering bug (documented in allocate_head_bits).
    """
    import itertools

    base = [7.0, 3.0, 2.0, 1.0]
    ref = allocate_head_bits(base, 2.75, [2, 3, 4], 4)

    for perm in itertools.permutations(range(4)):
        permuted = [base[i] for i in perm]
        got = allocate_head_bits(permuted, 2.75, [2, 3, 4], 4)
        expected = [ref[i] for i in perm]
        assert got == expected, (
            f"perm={perm}: got {got}, expected {expected} — allocation depends "
            f"on head ordering, not importance"
        )


# ---------------------------------------------------------------------------
# Test 18 — budget is met exactly where the allowed set permits
# ---------------------------------------------------------------------------
def test_budget_met_exactly():
    """Total assigned bits must land on H * target when representable."""
    for target in (2.25, 2.5, 2.75, 3.0, 3.5):
        bits = allocate_head_bits(
            head_importance=[5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            target_avg_bits=target,
            allowed_bits=[2, 3, 4],
            n_heads=8,
        )
        assert sum(bits) == 8 * target, (
            f"target={target}: total {sum(bits)} != budget {8 * target}, bits={bits}"
        )


# ---------------------------------------------------------------------------
# Test 19 — attention-entropy proxy carries the PAPER's sign
# ---------------------------------------------------------------------------
def test_attention_entropy_ranks_dispersed_above_sparse():
    """Ada-KV gives MORE budget to attention-DISPERSED heads (§3.3, Fig. 1b).

    Regression for issue #31 finding 3. The norm-variance signal is
    anti-correlated with that criterion (see the companion test below), so
    compute_head_attention_entropy exists to provide a signal with the
    paper's sign. Head 0 is attention-sparse (a few keys aligned with the
    observation window dominate the logits); head 1 is dispersed (unit-norm
    isotropic keys).
    """
    rng = np.random.default_rng(3)
    S, D = 512, 64

    u = rng.standard_normal(D).astype(np.float32)
    u /= np.linalg.norm(u)
    sparse = rng.standard_normal((S, D)).astype(np.float32) * 0.3
    sparse[:5] = u * 8.0            # a few dominant keys
    sparse[-32:] = u * 3.0          # obs window points at them

    dispersed = rng.standard_normal((S, D)).astype(np.float32)
    dispersed /= np.linalg.norm(dispersed, axis=1, keepdims=True)

    keys = mx.array(np.stack([sparse, dispersed]))[None]   # [1, 2, S, D]
    ent = compute_head_attention_entropy(keys, obs_window=32)

    assert ent[1].item() > ent[0].item(), (
        f"entropy ranked sparse head above dispersed one: {ent.tolist()} — "
        f"signal does not carry the paper's sign"
    )
    # Normalised to [0, 1].
    assert 0.0 <= ent[0].item() <= 1.0 and 0.0 <= ent[1].item() <= 1.0


def test_norm_variance_is_anticorrelated_with_paper_criterion():
    """Pins the documented caveat: norm-variance has the OPPOSITE sign.

    This is not a bug being tolerated — it is a *different, sound* criterion
    (quantization sensitivity: wider ||k_t|| spread means wider dynamic range
    per min/max group, so extra bits buy more). The test exists so the
    distinction stays true as the code changes, and so nobody re-adds the
    claim that norm-variance approximates attention entropy.
    """
    rng = np.random.default_rng(3)
    S, D = 512, 64

    u = rng.standard_normal(D).astype(np.float32)
    u /= np.linalg.norm(u)
    sparse = rng.standard_normal((S, D)).astype(np.float32) * 0.3
    sparse[:5] = u * 8.0
    sparse[-32:] = u * 3.0

    dispersed = rng.standard_normal((S, D)).astype(np.float32)
    dispersed /= np.linalg.norm(dispersed, axis=1, keepdims=True)

    keys = mx.array(np.stack([sparse, dispersed]))[None]
    var = compute_head_norm_variance(keys)
    ent = compute_head_attention_entropy(keys, obs_window=32)

    # norm-variance favours the sparse head; entropy favours the dispersed one.
    assert var[0].item() > var[1].item()
    assert ent[1].item() > ent[0].item()


# ---------------------------------------------------------------------------
# Test 20 — entropy mode is wired end-to-end through the cache
# ---------------------------------------------------------------------------
def test_entropy_importance_mode_end_to_end():
    """adakv_importance='attention_entropy' drives allocation and preserves shape."""
    cfg = _make_cfg(adakv_importance="attention_entropy", adakv_target_avg_bits=2.5)
    cache = AdaKVCache(cfg)
    k = _heterogeneous_keys(B=1, H=8, S=64, D=64, seed=41)
    v = _values(B=1, H=8, S=64, D=64, seed=42)

    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape
    assert cache.importance_mode == "attention_entropy"
    assert len(cache.head_bits) == 8
    assert all(b in cache.allowed_bits for b in cache.head_bits)

    # Decode step: S == 1 carries no attention distribution, so the prefill
    # entropy estimate must be retained rather than reset to zeros.
    before = list(cache.head_bits)
    k1 = _keys(B=1, H=8, S=1, D=64, seed=43)
    v1 = _values(B=1, H=8, S=1, D=64, seed=44)
    cache.update_and_fetch(k1, v1)
    assert cache.head_bits == before, (
        "decode step discarded the prefill entropy estimate"
    )


def test_invalid_importance_mode_rejected():
    import pytest

    with pytest.raises(ValueError, match="adakv_importance"):
        AdaKVCache(_make_cfg(adakv_importance="nonsense"))
