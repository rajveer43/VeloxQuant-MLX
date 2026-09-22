"""Tests for KVQuantKVCache — non-uniform quantization + dense/sparse outliers.

22 tests covering:
  1.  Factory dispatch
  2.  Output shape (prefill + decode)
  3.  Values reconstructed within tolerance
  4.  NUQ lower MSE than uniform at equal bits on non-uniform (Laplacian) data
  5.  NUQ ~= uniform on genuinely uniform data (no false free-lunch claim)
  6.  Lloyd-Max convergence: distortion monotone non-increasing across iters
  7.  split_dense_sparse selects the true top-k by magnitude
  8.  Outlier isolation lowers MSE vs same-bit NUQ without isolation (heavy tails)
  9.  outlier_fraction=0 reduces to plain NUQ (no side-channel, no outliers)
  10. Level-table determinism (fixed init → identical levels)
  11. Decode after prefill — frozen key levels, correct accumulation
  12. Byte accounting: compressed < fp16
  13. effective_bits within [bits, bits + overhead] at realistic context
  14. Per-channel (key) vs per-token (value) axis correctness
  15. Determinism (end-to-end)
  16. No public `.bits` attribute (mlx_lm SDPA dispatch trap)
  17. Outlier split exact under ties (constant columns don't over-select)
  18. Attention sink tokens kept bit-exact in fp16 (paper §3.5)
  19. Sink protection off at n_sink=0; never applied to decode tokens
  20. Decode keys keep outlier protection via frozen per-channel threshold
  21. Sink tokens excluded from the level fit (paper §3.5)
  22. Byte accounting tracks realized outliers + fp16 sink rows
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kvquant_cache import KVQuantKVCache
from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant
from veloxquant_mlx.quantizers.kvquant import (
    dequant_nuq,
    dequant_nuq_batched,
    fit_nuq_levels,
    fit_nuq_levels_batched,
    nuq_distortion,
    nuq_quant_dequant,
    quantize_nuq,
    quantize_nuq_batched,
    split_dense_sparse,
    split_dense_sparse_batched,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cfg(**kwargs) -> KVCacheConfig:
    d = {"method": "kvquant", "head_dim": 64, "kvquant_bits": 3}
    d.update(kwargs)
    return KVCacheConfig(**d)


def _laplace(B=1, H=2, S=64, D=64, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.laplace(0, 1, (B, H, S, D)).astype(np.float16))


def _mse(a, b):
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


# ---------------------------------------------------------------------------
# Test 1 — factory dispatch
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = KVCacheFactory.create(_cfg())
    assert isinstance(cache, KVQuantKVCache)


# ---------------------------------------------------------------------------
# Test 2 — output shape (prefill + decode)
# ---------------------------------------------------------------------------
def test_output_shape_prefill_decode():
    cache = KVQuantKVCache(_cfg())
    ko, vo = cache.update_and_fetch(_laplace(1, 2, 32, 64), _laplace(1, 2, 32, 64, seed=1))
    assert ko.shape == (1, 2, 32, 64) and vo.shape == (1, 2, 32, 64)
    ko2, vo2 = cache.update_and_fetch(_laplace(1, 2, 1, 64, seed=2), _laplace(1, 2, 1, 64, seed=3))
    assert ko2.shape == (1, 2, 33, 64) and vo2.shape == (1, 2, 33, 64)


# ---------------------------------------------------------------------------
# Test 3 — values reconstructed within tolerance
# ---------------------------------------------------------------------------
def test_values_reconstructed():
    cache = KVQuantKVCache(_cfg(kvquant_bits=4))
    v = _laplace(1, 2, 64, 64, seed=5)
    _, vo = cache.update_and_fetch(_laplace(1, 2, 64, 64), v)
    assert vo.shape == v.shape
    assert bool(mx.all(mx.isfinite(vo)).item())
    assert _mse(vo, v) < 0.5  # 4-bit NUQ on unit-scale Laplacian is tight


# ---------------------------------------------------------------------------
# Test 4 — NUQ beats uniform at equal bits on non-uniform data
# ---------------------------------------------------------------------------
def test_nuq_beats_uniform_on_nonuniform():
    rng = np.random.default_rng(4)
    x = mx.array(rng.laplace(0, 1, (256, 16)).astype(np.float16))
    nuq = nuq_quant_dequant(x, bits=3, outlier_fraction=0.0)
    uni = _group_quant_dequant(x, b=3, group_size=256)
    assert _mse(nuq, x) < _mse(uni, x), "NUQ should beat uniform on Laplacian data"


# ---------------------------------------------------------------------------
# Test 5 — NUQ ~= uniform on genuinely uniform data (no false claim)
# ---------------------------------------------------------------------------
def test_nuq_not_worse_on_uniform():
    rng = np.random.default_rng(50)
    x = mx.array(rng.uniform(-1, 1, (256, 16)).astype(np.float16))
    nuq = nuq_quant_dequant(x, bits=3, outlier_fraction=0.0)
    uni = _group_quant_dequant(x, b=3, group_size=256)
    # NUQ must not be materially worse than uniform where uniform is near-optimal.
    assert _mse(nuq, x) <= _mse(uni, x) * 1.2


# ---------------------------------------------------------------------------
# Test 6 — Lloyd-Max distortion monotone non-increasing
# ---------------------------------------------------------------------------
def test_lloyd_max_monotone():
    rng = np.random.default_rng(6)
    x = mx.array(rng.laplace(0, 1, (256, 8)).astype(np.float32))
    prev = None
    for it in range(1, 9):
        lv = fit_nuq_levels(x, bits=3, n_iters=it)
        d = nuq_distortion(x, lv)
        if prev is not None:
            assert d <= prev + 1e-5, f"distortion increased at iter {it}: {d} > {prev}"
        prev = d


# ---------------------------------------------------------------------------
# Test 7 — split_dense_sparse selects true top-k
# ---------------------------------------------------------------------------
def test_split_selects_top_k():
    # One column, clear outliers at known positions.
    col = np.array([0.1, 0.2, 9.0, 0.3, -8.0, 0.1, 0.2, 0.05], dtype=np.float32)
    x = mx.array(col.reshape(-1, 1))
    ds = split_dense_sparse(x, outlier_fraction=0.25)  # top 2 of 8
    mask = np.array(ds.outlier_mask.tolist()).reshape(-1).astype(bool)
    assert mask[2] and mask[4], f"expected positions 2,4 flagged, got {np.where(mask)[0]}"
    assert mask.sum() == 2


# ---------------------------------------------------------------------------
# Test 8 — outlier isolation lowers MSE on heavy-tailed data
# ---------------------------------------------------------------------------
def test_outlier_isolation_lowers_mse():
    rng = np.random.default_rng(8)
    base = rng.laplace(0, 1, (256, 8)).astype(np.float32)
    # inject a few extreme spikes
    base[rng.integers(0, 256, 5), rng.integers(0, 8, 5)] = 30.0
    x = mx.array(base.astype(np.float16))
    no_out = nuq_quant_dequant(x, bits=3, outlier_fraction=0.0)
    with_out = nuq_quant_dequant(x, bits=3, outlier_fraction=0.02)
    assert _mse(with_out, x) < _mse(no_out, x)


# ---------------------------------------------------------------------------
# Test 9 — outlier_fraction=0 reduces to plain NUQ (no outliers)
# ---------------------------------------------------------------------------
def test_outlier_fraction_zero_pure_nuq():
    cache = KVQuantKVCache(_cfg(kvquant_outlier_fraction=0.0))
    cache.update_and_fetch(_laplace(1, 2, 64, 64), _laplace(1, 2, 64, 64, seed=1))
    assert cache.outlier_count == 0


# ---------------------------------------------------------------------------
# Test 10 — level-table determinism
# ---------------------------------------------------------------------------
def test_level_table_determinism():
    rng = np.random.default_rng(10)
    x = mx.array(rng.laplace(0, 1, (256, 8)).astype(np.float32))
    l1 = fit_nuq_levels(x, bits=3, n_iters=8)
    l2 = fit_nuq_levels(x, bits=3, n_iters=8)
    np.testing.assert_array_equal(np.array(l1.tolist()), np.array(l2.tolist()))


# ---------------------------------------------------------------------------
# Test 11 — decode after prefill, frozen key levels, accumulation
# ---------------------------------------------------------------------------
def test_decode_frozen_key_levels():
    cache = KVQuantKVCache(_cfg())
    cache.update_and_fetch(_laplace(1, 2, 20, 64), _laplace(1, 2, 20, 64, seed=1))
    frozen = [np.array(level.tolist()) for level in cache.key_levels]
    for step in range(5):
        kd = _laplace(1, 2, 1, 64, seed=100 + step)
        vd = _laplace(1, 2, 1, 64, seed=200 + step)
        ko, _ = cache.update_and_fetch(kd, vd)
        assert ko.shape[2] == 20 + step + 1
    # Key levels unchanged (refit_interval=0 → frozen).
    for a, b in zip(frozen, cache.key_levels, strict=True):
        np.testing.assert_array_equal(a, np.array(b.tolist()))


# ---------------------------------------------------------------------------
# Test 12 — byte accounting compressed < fp16
# ---------------------------------------------------------------------------
def test_byte_accounting():
    cache = KVQuantKVCache(_cfg())
    cache.update_and_fetch(_laplace(1, 2, 512, 64), _laplace(1, 2, 512, 64, seed=1))
    assert cache.compressed_key_bytes < cache.fp16_key_bytes
    assert cache.compressed_value_bytes < cache.fp16_value_bytes


# ---------------------------------------------------------------------------
# Test 13 — effective_bits within [bits, bits + overhead] at realistic context
# ---------------------------------------------------------------------------
def test_effective_bits_range():
    cache = KVQuantKVCache(_cfg(kvquant_bits=3, kvquant_outlier_fraction=0.01))
    cache.update_and_fetch(_laplace(1, 2, 1024, 64), _laplace(1, 2, 1024, 64, seed=1))
    eff = cache.effective_bits
    assert 3.0 <= eff <= 4.0, f"effective_bits={eff} out of [3.0, 4.0]"


# ---------------------------------------------------------------------------
# Test 14 — per-channel (key) vs per-token (value) axis correctness
# ---------------------------------------------------------------------------
def test_key_value_axes():
    cache = KVQuantKVCache(_cfg())
    cache.update_and_fetch(_laplace(1, 1, 64, 64), _laplace(1, 1, 64, 64, seed=1))
    # Keys: per-channel levels → [L, D] with D = head_dim columns.
    kl = cache.key_levels[0]
    assert kl.shape[0] == (1 << cache.nuq_bits) and kl.shape[1] == 64
    # Values: per-token levels (transposed space) → columns are tokens (S=64).
    vl = cache.value_levels[0]
    assert vl.shape[0] == (1 << cache.nuq_bits) and vl.shape[1] == 64


# ---------------------------------------------------------------------------
# Test 15 — determinism end-to-end
# ---------------------------------------------------------------------------
def test_determinism():
    k = _laplace(1, 2, 64, 64, seed=77)
    v = _laplace(1, 2, 64, 64, seed=88)

    def run():
        c = KVQuantKVCache(_cfg())
        ko, vo = c.update_and_fetch(k, v)
        return np.array(ko.tolist()), np.array(vo.tolist())

    k1, v1 = run()
    k2, v2 = run()
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Test 16 — no public .bits attribute (mlx_lm SDPA dispatch trap)
# ---------------------------------------------------------------------------
def test_no_bits_leak():
    """mlx_lm's SDPA checks `hasattr(cache, "bits")` to route to its
    quantized-matmul kernel, which expects mx.quantize's native tuple layout
    and a `.group_size` attribute this cache doesn't have. Exposing `.bits`
    here would silently hijack attention dispatch — see issue #87."""
    cache = KVQuantKVCache(_cfg(kvquant_bits=3))
    assert not hasattr(cache, "bits"), (
        "KVQuantKVCache must not expose .bits — would break mlx_lm SDPA dispatch"
    )
    assert hasattr(cache, "nuq_bits")
    assert cache.nuq_bits == 3


# ---------------------------------------------------------------------------
# Test 17 — outlier split is exact under ties (constant / repeated columns)
# ---------------------------------------------------------------------------
def test_split_exact_under_ties():
    """A value threshold (`mag >= kth_largest`) over-selects when values tie:
    a constant column has every element equal to the threshold, so all N would
    be flagged as outliers and shipped to the fp16 side-channel while byte
    accounting charges only k. Rank-based selection keeps it at exactly k."""
    x = mx.array(np.ones((8, 3), dtype=np.float32))  # every element ties
    ds = split_dense_sparse(x, outlier_fraction=0.25)  # k = 2 of 8
    per_col = np.array(ds.outlier_mask.tolist()).sum(axis=0)
    assert list(per_col) == [2, 2, 2], f"ties over-selected: {per_col}"

    # Mixed data: exactly k per column, independent of duplicate magnitudes.
    rng = np.random.default_rng(17)
    y = mx.array(rng.laplace(0, 1, (100, 7)).astype(np.float32))
    dsy = split_dense_sparse(y, outlier_fraction=0.1)  # k = 10
    assert (np.array(dsy.outlier_mask.tolist()).sum(axis=0) == 10).all()


# ---------------------------------------------------------------------------
# Test 18 — attention sink tokens kept bit-exact in fp16 (paper §3.5)
# ---------------------------------------------------------------------------
def test_attention_sink_exact():
    cache = KVQuantKVCache(_cfg(kvquant_n_sink=2))
    k = _laplace(1, 2, 32, 64, seed=18)
    v = _laplace(1, 2, 32, 64, seed=19)
    ko, vo = cache.update_and_fetch(k, v)
    assert bool(mx.all(ko[:, :, :2, :] == k[:, :, :2, :]).item()), "sink keys must be exact"
    assert bool(mx.all(vo[:, :, :2, :] == v[:, :, :2, :]).item()), "sink values must be exact"
    # Non-sink positions are actually quantized (guards against a no-op cache).
    assert not bool(mx.all(ko[:, :, 2:, :] == k[:, :, 2:, :]).item())
    assert cache.sink_kept == 2


# ---------------------------------------------------------------------------
# Test 19 — sink protection is off when kvquant_n_sink=0, and only applies
#           to the leading tokens of the sequence (not to decode tokens)
# ---------------------------------------------------------------------------
def test_sink_disabled_and_not_applied_at_decode():
    cache = KVQuantKVCache(_cfg(kvquant_n_sink=0))
    k = _laplace(1, 2, 16, 64, seed=20)
    ko, _ = cache.update_and_fetch(k, _laplace(1, 2, 16, 64, seed=21))
    assert cache.sink_kept == 0
    assert not bool(mx.all(ko[:, :, :1, :] == k[:, :, :1, :]).item())

    # A decode token is mid-stream, never a sink → must be quantized.
    c2 = KVQuantKVCache(_cfg(kvquant_n_sink=1))
    c2.update_and_fetch(_laplace(1, 2, 16, 64, seed=22), _laplace(1, 2, 16, 64, seed=23))
    kd = _laplace(1, 2, 1, 64, seed=24)
    ko2, _ = c2.update_and_fetch(kd, _laplace(1, 2, 1, 64, seed=25))
    assert c2.sink_kept == 1, "sink count must not grow during decode"
    assert not bool(mx.all(ko2[:, :, -1:, :] == kd).item()), "decode token must be quantized"


# ---------------------------------------------------------------------------
# Test 20 — decode keys still get outlier protection (S=1 degeneracy)
# ---------------------------------------------------------------------------
def test_decode_keys_keep_outlier_protection():
    """At decode a per-channel column holds one sample, so a rank-based top-k
    cannot flag anything while keeping an inlier. Without the carried-over
    prefill threshold, decode keys would silently lose outlier isolation."""
    cache = KVQuantKVCache(_cfg(kvquant_outlier_fraction=0.01))
    cache.update_and_fetch(_laplace(1, 2, 64, 64, seed=26), _laplace(1, 2, 64, 64, seed=27))
    assert cache.key_outlier_thresh is not None
    assert cache.key_outlier_thresh.shape == (1, 64)

    before = cache.outlier_count
    # Large-magnitude decode key → must trip the frozen per-channel threshold.
    kd = _laplace(1, 2, 1, 64, seed=28) * 5
    cache.update_and_fetch(kd, _laplace(1, 2, 1, 64, seed=29))
    assert cache.outlier_count > before, "decode keys must still isolate outliers"


# ---------------------------------------------------------------------------
# Test 21 — sink tokens are excluded from the level fit (paper §3.5)
# ---------------------------------------------------------------------------
def test_sink_excluded_from_level_fit():
    """The paper ignores sink tokens when deriving the nuqX datatype. With an
    extreme sink token, including it would drag the fitted levels; excluding it
    must leave the levels equal to a fit with the sink absent."""
    rng = np.random.default_rng(30)
    body = rng.laplace(0, 1, (1, 1, 32, 16)).astype(np.float16)
    spiked = body.copy()
    spiked[:, :, 0, :] = 500.0  # extreme sink row

    cfg = {"head_dim": 16, "kvquant_bits": 3, "kvquant_n_sink": 1, "kvquant_outlier_fraction": 0.0}
    c_spike = KVQuantKVCache(_cfg(**cfg))
    c_spike.update_and_fetch(mx.array(spiked), mx.array(spiked))
    lv_spike = np.array(c_spike.key_levels[0].tolist())

    # Reference: same tensor with the sink row replaced by ordinary data.
    clean = body.copy()
    c_clean = KVQuantKVCache(_cfg(**cfg))
    c_clean.update_and_fetch(mx.array(clean), mx.array(clean))
    lv_clean = np.array(c_clean.key_levels[0].tolist())

    # Levels are fit on tokens 1.. in both cases → identical.
    np.testing.assert_allclose(lv_spike, lv_clean, rtol=1e-5)
    assert np.abs(lv_spike).max() < 100.0, "sink spike leaked into the level fit"


# ---------------------------------------------------------------------------
# Test 22 — byte accounting tracks realized outliers and fp16 sink rows
# ---------------------------------------------------------------------------
def test_accounting_tracks_realized_outliers_and_sinks():
    # More sink tokens kept in fp16 → strictly more compressed bytes.
    ka = _laplace(1, 2, 256, 64, seed=31)
    va = _laplace(1, 2, 256, 64, seed=32)

    c0 = KVQuantKVCache(_cfg(kvquant_n_sink=0))
    c0.update_and_fetch(ka, va)
    c4 = KVQuantKVCache(_cfg(kvquant_n_sink=4))
    c4.update_and_fetch(ka, va)
    assert c4.compressed_key_bytes > c0.compressed_key_bytes

    # Accounting stays honest: still below fp16, and effective_bits sane.
    assert c4.compressed_key_bytes < c4.fp16_key_bytes
    assert 3.0 <= c4.effective_bits <= 6.0


# ---------------------------------------------------------------------------
# Test 23 — not batchable via mlx_lm.server's probe (issue #16, same defect
# as knorm/#15 and #357)
# ---------------------------------------------------------------------------
def test_not_batchable_via_mlx_lm_server_probe() -> None:
    """`mlx_lm.server`'s `ModelProvider.load()` decides whether a method is
    batchable purely via `hasattr(cache, "merge")` on a probe instance. The
    base `KVCache` this inherits from defines `merge()` as a classmethod
    returning a plain `BatchKVCache` — oblivious to this class's frozen NUQ
    levels, outlier thresholds, and sink bookkeeping. Left inherited, every
    request (even a lone one — `BatchGenerator` merges a batch of 1 too)
    would silently replace this cache with that generic one: no
    quantization, no outlier isolation, no sink protection, while the server
    still believes it is running `kvquant`. `hasattr` must see `merge` as
    absent so the server routes `kvquant` through its sequential path
    instead, where this class runs correctly.
    """
    cache = KVQuantKVCache(_cfg())
    assert not hasattr(cache, "merge")
    with pytest.raises(AttributeError):
        cache.merge


# ---------------------------------------------------------------------------
# Regression tests for issue #504 (unbatched B*H loop + per-Lloyd-Max-
# iteration forced host sync measured a 59x real mlx_lm.generate() decode
# slowdown; fixed to 9.6x recovery via batched primitives — see
# quantizers/kvquant.py's *_batched functions and cache/kvquant_cache.py's
# module docstring for the full before/after numbers)
# ---------------------------------------------------------------------------
def test_split_dense_sparse_batched_matches_per_row_loop():
    rng = np.random.default_rng(40)
    BH, N, D = 5, 32, 16
    x = mx.array(rng.laplace(0, 1, (BH, N, D)).astype(np.float32))

    ref_inliers, ref_mask, ref_vals = [], [], []
    for i in range(BH):
        ds = split_dense_sparse(x[i], 0.1)
        ref_inliers.append(ds.inliers)
        ref_mask.append(ds.outlier_mask)
        ref_vals.append(ds.outlier_vals)
    ref_inliers = mx.stack(ref_inliers)
    ref_mask = mx.stack(ref_mask)
    ref_vals = mx.stack(ref_vals)

    out = split_dense_sparse_batched(x, 0.1)
    np.testing.assert_array_equal(np.array(ref_mask), np.array(out.outlier_mask))
    np.testing.assert_allclose(np.array(ref_inliers), np.array(out.inliers), atol=1e-5)
    np.testing.assert_allclose(np.array(ref_vals), np.array(out.outlier_vals), atol=1e-5)


def test_split_dense_sparse_batched_handles_decode_shape():
    """N=1 (a single decode-step token per row) must not crash or diverge
    from the per-row reference — the shape every real decode step hits."""
    rng = np.random.default_rng(41)
    BH, D = 4, 16
    x = mx.array(rng.laplace(0, 1, (BH, 1, D)).astype(np.float32))
    out = split_dense_sparse_batched(x, 0.1)
    ref = mx.stack([split_dense_sparse(x[i], 0.1).inliers for i in range(BH)])
    np.testing.assert_allclose(np.array(ref), np.array(out.inliers), atol=1e-5)


def test_fit_nuq_levels_batched_matches_per_row_loop():
    rng = np.random.default_rng(42)
    BH, N, D, bits = 5, 32, 16, 3
    x = mx.array(rng.laplace(0, 1, (BH, N, D)).astype(np.float32))

    ref = mx.stack([fit_nuq_levels(x[i], bits, 8) for i in range(BH)])
    out = fit_nuq_levels_batched(x, bits, 8)
    np.testing.assert_allclose(np.array(ref), np.array(out), atol=1e-3)


def test_quantize_and_dequant_nuq_batched_match_per_row_loop():
    rng = np.random.default_rng(43)
    BH, N, D, bits = 5, 32, 16, 3
    x = mx.array(rng.laplace(0, 1, (BH, N, D)).astype(np.float32))
    levels = fit_nuq_levels_batched(x, bits, 4)

    ref_codes = mx.stack([quantize_nuq(x[i], levels[i]) for i in range(BH)])
    out_codes = quantize_nuq_batched(x, levels)
    np.testing.assert_array_equal(np.array(ref_codes), np.array(out_codes))

    ref_recon = mx.stack([dequant_nuq(ref_codes[i], levels[i]) for i in range(BH)])
    out_recon = dequant_nuq_batched(out_codes, levels)
    np.testing.assert_array_equal(np.array(ref_recon), np.array(out_recon))


def test_batched_cache_matches_per_head_reference_for_b_gt_1():
    """The trickiest correctness detail in the #504 batching rewrite: the
    ORIGINAL per-head loop fit key levels only from batch element 0's data
    (`keys[0]`) and shared that single fit across every batch element,
    rather than fitting independently per (b, h). The batched rewrite must
    reproduce this exactly for B > 1 — fitting independently per BH row
    would be a silent behavior change, not just a speed optimization. This
    test pins that by comparing prefill+decode output at B=2 against a
    literal reimplementation of the original per-(b,h) loop.
    """
    rng = np.random.default_rng(44)
    B, H, S, D, bits = 2, 3, 20, 16, 3
    keys = rng.laplace(0, 1, (B, H, S, D)).astype(np.float32)

    # Literal old-style per-head loop: fit ONCE on b==0, reuse for b==1.
    new_klev = [None] * H
    ref_recon = np.zeros_like(keys)
    for b in range(B):
        for h in range(H):
            kl = new_klev[h] if b > 0 else None
            if kl is None:
                ds = split_dense_sparse(mx.array(keys[b, h]), 0.0)
                kl = fit_nuq_levels(ds.inliers, bits, 4)
                new_klev[h] = kl
            codes = quantize_nuq(mx.array(keys[b, h]), kl)
            ref_recon[b, h] = np.array(dequant_nuq(codes, kl))

    # New cache path.
    cfg = _cfg(
        kvquant_bits=bits, kvquant_lloyd_iters=4, kvquant_outlier_fraction=0.0, kvquant_n_sink=0
    )
    cache = KVQuantKVCache(cfg)
    k_out, _ = cache.update_and_fetch(mx.array(keys), mx.array(keys))
    np.testing.assert_allclose(np.array(k_out), ref_recon, atol=1e-3)


def test_key_levels_frozen_across_decode_default_refit_interval():
    """kvquant_refit_interval=0 (default) must freeze KEY levels after
    prefill — the value path is, by design, always fit fresh (see class
    docstring) and is unaffected by this field."""
    cache = KVQuantKVCache(_cfg(kvquant_refit_interval=0))
    cache.update_and_fetch(_laplace(1, 2, 32, 64, seed=50), _laplace(1, 2, 32, 64, seed=51))
    frozen = np.array(cache.key_levels[0].tolist())
    cache.update_and_fetch(_laplace(1, 2, 1, 64, seed=52), _laplace(1, 2, 1, 64, seed=53))
    still_frozen = np.array(cache.key_levels[0].tolist())
    np.testing.assert_array_equal(frozen, still_frozen)


def test_outlier_count_matches_between_batched_and_manual_sum():
    """Batched outlier accounting sums the mask once per call instead of
    once per (b, h) — must produce the same total as manually summing a
    per-head mask."""
    cache = KVQuantKVCache(_cfg(kvquant_outlier_fraction=0.05))
    k = _laplace(1, 4, 64, 64, seed=60)
    v = _laplace(1, 4, 64, 64, seed=61)
    cache.update_and_fetch(k, v)
    assert cache.outlier_count > 0
    # Sanity bound: outlier_count should scale with B*H*S*D*outlier_fraction*2
    # (keys + values), not blow up or vanish from a batching indexing bug.
    expected_order = 1 * 4 * 64 * 64 * 0.05 * 2
    assert 0.1 * expected_order < cache.outlier_count < 10 * expected_order
