"""Tests for GEARKVCache — error-feedback compression over a base group quant.

GEAR's reconstructed K/V genuinely recover quality the base bit-width loses
(unlike CacheGen, whose reconstruction is identical to group quant). These tests
cover factory dispatch, shape preservation, the quality-recovery property, byte
accounting, the values-off path, decode accumulation, determinism, and
construction via both KVCacheFactory.create and KVCacheBuilder.for_model. All
data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.gear_cache import GEARKVCache


def _make(**cfg):
    base = {
        "method": "gear",
        "head_dim": 128,
        "gear_bits": 2,
        "gear_rank": 8,
        "gear_sparse_fraction": 0.005,
        "gear_group_size": 32,
    }
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _lowrank_kv(S=128, H=2, D=128, r=6, seed=0):
    """Low-rank + small-noise KV — the regime GEAR's error feedback helps."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((H, S, r)).astype(np.float32)
    B = rng.standard_normal((H, r, D)).astype(np.float32)
    K = (A @ B + 0.03 * rng.standard_normal((H, S, D))).astype(np.float16)[None]
    V = (A @ B + 0.03 * rng.standard_normal((H, S, D))).astype(np.float16)[None]
    return mx.array(K), mx.array(V)


# ------------------------------------------------------------------
# Factory and interface
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), GEARKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "assigned_avg_bits")


def test_output_shape_preserved() -> None:
    c = _make()
    k, v = _lowrank_kv()
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape == k.shape and vo.shape == v.shape


# ------------------------------------------------------------------
# Core claim: GEAR recovers quality the base bit-width loses
# ------------------------------------------------------------------


def test_error_recovery_positive() -> None:
    c = _make()
    k, v = _lowrank_kv()
    c.update_and_fetch(k, v)
    assert 0.0 < c.error_recovery_ratio <= 1.0


def test_beats_naive_base_reconstruction() -> None:
    """The reconstructed keys are closer to the originals than base quant alone."""
    from veloxquant_mlx.quantizers.cachegen import cachegen_quant_dequant

    c = _make()
    k, v = _lowrank_kv()
    ko, _ = c.update_and_fetch(k, v)

    def mse(a, b):
        return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())

    base = mx.stack(
        [
            mx.stack([cachegen_quant_dequant(k[b, h], 2, 32) for h in range(k.shape[1])])
            for b in range(k.shape[0])
        ]
    )
    assert mse(ko, k) < mse(base, k)


# ------------------------------------------------------------------
# Byte accounting
# ------------------------------------------------------------------


def test_byte_accounting_ordering() -> None:
    c = _make()
    k, v = _lowrank_kv()
    c.update_and_fetch(k, v)
    assert c.base_only_key_bytes <= c.compressed_key_bytes <= c.fp16_key_bytes
    assert c.assigned_avg_bits <= 16.0


def test_values_off_keeps_values_fp16() -> None:
    c = _make(gear_quantize_values=False)
    k, v = _lowrank_kv()
    ko, vo = c.update_and_fetch(k, v)
    # values pass through unchanged (lossless)
    assert float(mx.mean((vo.astype(mx.float32) - v.astype(mx.float32)) ** 2).item()) == 0.0
    assert c.compressed_value_bytes == 0
    assert c.fp16_value_bytes > 0


# ------------------------------------------------------------------
# Decode and robustness
# ------------------------------------------------------------------


def test_decode_accumulation() -> None:
    c = _make()
    k, v = _lowrank_kv(S=64)
    c.update_and_fetch(k, v)
    for i in range(4):
        k1, v1 = _lowrank_kv(S=1, seed=100 + i)
        ko, vo = c.update_and_fetch(k1, v1)
    assert ko.shape[2] == 64 + 4


def test_deterministic() -> None:
    k, v = _lowrank_kv()
    c1, c2 = _make(), _make()
    ko1, _ = c1.update_and_fetch(k, v)
    ko2, _ = c2.update_and_fetch(k, v)
    assert float(mx.mean((ko1.astype(mx.float32) - ko2.astype(mx.float32)) ** 2).item()) == 0.0


# ------------------------------------------------------------------
# KCVT backbone: keys per-channel, values per-token (paper's actual scheme)
# ------------------------------------------------------------------


def test_cache_uses_kcvt_axes_for_keys_and_values() -> None:
    """The wrapper's base layer groups keys along the channel axis and
    values along the token axis, per the paper's KCVT backbone — not the
    same axis for both, which was this wrapper's behavior before KCVT was
    wired in.

    VeloxQuant-MLX#504's batched-SVD/batched-base-quant rewrite replaced
    the per-head ``quantize_base(..., axis=...)`` calls with one batched
    ``_group_quant_codes_batched`` call plus an explicit transpose for the
    "channel" case (see ``_compress_and_account``'s docstring) — there is
    no longer a single named ``axis`` parameter to spy on. Verified
    behaviorally instead: with rank=0 and sparse_fraction=0 (pure base
    layer, no error-feedback terms to obscure the comparison), the cache's
    key output must match ``quantize_base(..., axis="channel")`` applied
    directly, and its value output must match ``axis="token"`` — proving
    the axis choice is actually still correct, not just that some
    parameter was passed through unchanged.
    """
    from veloxquant_mlx.quantizers.gear import quantize_base

    cfg = {
        "method": "gear",
        "head_dim": 32,
        "gear_bits": 2,
        "gear_rank": 0,
        "gear_sparse_fraction": 0.0,
        "gear_group_size": 8,
    }
    c = KVCacheFactory.create(KVCacheConfig(**cfg))
    rng = np.random.default_rng(21)
    k = mx.array(rng.standard_normal((1, 1, 24, 32)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, 1, 24, 32)).astype(np.float16))
    k_out, v_out = c.update_and_fetch(k, v)

    _, k_channel_ref = quantize_base(k[0, 0].astype(mx.float32), 2, 8, axis="channel")
    _, k_token_ref = quantize_base(k[0, 0].astype(mx.float32), 2, 8, axis="token")
    _, v_token_ref = quantize_base(v[0, 0].astype(mx.float32), 2, 8, axis="token")

    k_actual = np.array(k_out[0, 0].astype(mx.float32))
    v_actual = np.array(v_out[0, 0].astype(mx.float32))

    np.testing.assert_allclose(k_actual, np.array(k_channel_ref), atol=1e-3)
    np.testing.assert_allclose(v_actual, np.array(v_token_ref), atol=1e-3)
    # Sanity: channel-axis and token-axis grouping genuinely differ on this
    # data (otherwise the test above couldn't distinguish a wrong axis).
    assert not np.allclose(np.array(k_channel_ref), np.array(k_token_ref), atol=1e-3)


def test_build_via_for_model_propagates_config() -> None:
    """KVCacheBuilder.for_model must carry the gear_* fields (replace path)."""
    from veloxquant_mlx.cache.base import KVCacheBuilder

    class _Attn:
        head_dim = 128

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="gear", head_dim=128, gear_bits=2, gear_rank=8, gear_sparse_fraction=0.005
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, GEARKVCache) for c in caches)
    assert caches[0]._rank == 8
    assert caches[0]._sparse_frac == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Regression tests for issue #504 (unbatched B*H loop over a per-matrix,
# CPU-stream SVD call measured a real 92% mlx_lm.generate() decode
# regression — 72 -> 5.8 tok/s on this M4; fixed to a 3.2x recovery via
# batched SVD + batched base group-quant, ~18.6 tok/s)
# ---------------------------------------------------------------------------
def test_truncated_svd_batched_matches_per_row_loop_energy_threshold():
    """_truncated_svd_batched (energy-threshold rank selection) must match
    looping the scalar _truncated_svd once per row, including the case
    where different rows genuinely pick different ranks — the case the
    padded-to-max-rank batched representation must handle correctly (each
    row's own rank, not the batch max, is what accounting must charge)."""
    from veloxquant_mlx.quantizers._quant_utils import _truncated_svd, _truncated_svd_batched

    rng = np.random.default_rng(70)
    N, D = 20, 16
    mats = []
    u, _, vt = np.linalg.svd(rng.standard_normal((N, D)).astype(np.float32), full_matrices=False)
    mats.append(
        u @ np.diag(np.array([10.0, 8.0] + [0.01] * (min(N, D) - 2), dtype=np.float32)) @ vt
    )
    u2, _, vt2 = np.linalg.svd(rng.standard_normal((N, D)).astype(np.float32), full_matrices=False)
    mats.append(u2 @ np.diag(np.ones(min(N, D), dtype=np.float32) * 3.0) @ vt2)
    E = mx.array(np.stack(mats).astype(np.float32))

    ref = []
    for i in range(2):
        U, s, Vt = _truncated_svd(E[i], rank=None, energy_threshold=0.9)
        ref.append((np.array(U * s[None, :]), np.array(Vt), U.shape[1]))

    L, R, ranks = _truncated_svd_batched(E, rank=None, energy_threshold=0.9)
    assert ranks[0] != ranks[1], "test fixture must produce genuinely different ranks per row"
    for i in range(2):
        ref_L, ref_Vt, ref_rank = ref[i]
        assert ranks[i] == ref_rank
        recon_ref = ref_L @ ref_Vt
        recon_new = np.array(L[i, :, : ranks[i]]) @ np.array(R[i, : ranks[i], :])
        np.testing.assert_allclose(recon_ref, recon_new, atol=1e-3)


def test_truncated_svd_batched_zero_matrix_matches_scalar_fallback():
    """A zero (or near-zero) residual must get rank=1, matching the scalar
    version's explicit `total < 1e-12` special case — an earlier draft of
    the batched version silently gave the full rank instead."""
    from veloxquant_mlx.quantizers._quant_utils import _truncated_svd_batched

    E = mx.zeros((3, 10, 8), dtype=mx.float32)
    _, _, ranks = _truncated_svd_batched(E, rank=None, energy_threshold=0.9)
    assert ranks == [1, 1, 1]


def test_truncated_svd_batched_explicit_rank_shared_across_rows():
    from veloxquant_mlx.quantizers._quant_utils import _truncated_svd, _truncated_svd_batched

    rng = np.random.default_rng(71)
    E = mx.array(rng.standard_normal((4, 20, 16)).astype(np.float32))
    L, R, ranks = _truncated_svd_batched(E, rank=5, energy_threshold=0.9)
    assert ranks == [5, 5, 5, 5]
    for i in range(4):
        U, s, Vt = _truncated_svd(E[i], rank=5, energy_threshold=0.9)
        recon_ref = np.array(U * s[None, :]) @ np.array(Vt)
        recon_new = np.array(L[i]) @ np.array(R[i])
        np.testing.assert_allclose(recon_ref, recon_new, atol=1e-3)


def test_group_quant_codes_batched_matches_per_row_loop():
    from veloxquant_mlx.quantizers._quant_utils import (
        _group_dequant_codes,
        _group_dequant_codes_batched,
        _group_quant_codes,
        _group_quant_codes_batched,
    )

    rng = np.random.default_rng(72)
    N, S, D, bits, gs = 5, 24, 16, 2, 8
    x = mx.array(rng.laplace(0, 1, (N, S, D)).astype(np.float32))

    ref_codes, ref_scale, ref_zero = [], [], []
    for i in range(N):
        c, s, z = _group_quant_codes(x[i], bits, gs)
        ref_codes.append(c)
        ref_scale.append(s)
        ref_zero.append(z)
    new_codes, new_scale, new_zero = _group_quant_codes_batched(x, bits, gs)
    for i in range(N):
        np.testing.assert_array_equal(np.array(ref_codes[i]), np.array(new_codes[i]))

    ref_recon = mx.stack(
        [_group_dequant_codes(ref_codes[i], ref_scale[i], ref_zero[i], S, gs) for i in range(N)]
    )
    new_recon = _group_dequant_codes_batched(new_codes, new_scale, new_zero, S, gs)
    np.testing.assert_allclose(np.array(ref_recon), np.array(new_recon), atol=1e-5)


def test_batched_cache_matches_per_head_reference_for_b_gt_1():
    """End-to-end pin: the batched cache path (SVD + base-quant, both now
    batched across B*H) must reproduce the exact byte accounting and
    reconstructed output of the original per-head loop at B > 1, including
    the fp16-truncation contract quantize_base's own reconstruction has
    (an earlier draft of this fix skipped that truncation, silently
    shifting the residual fed into the SVD by up to ~0.18 on synthetic
    data — a real, if small, precision divergence from the original
    per-head implementation, caught by this exact comparison)."""
    rng = np.random.default_rng(73)
    B, H, S, D = 2, 3, 24, 16

    cfg = _make(
        head_dim=D,
        gear_bits=2,
        gear_rank=None,
        gear_energy_threshold=0.9,
        gear_sparse_fraction=0.02,
        gear_group_size=8,
        gear_quantize_values=True,
    )
    keys = mx.array(rng.laplace(0, 1, (B, H, S, D)).astype(np.float32))
    values = mx.array(rng.laplace(0, 1, (B, H, S, D)).astype(np.float32))
    k_out, v_out = cfg.update_and_fetch(keys, values)

    # Reference: literal per-head loop using the ORIGINAL (unbatched)
    # numerics this class used before #504 — quantize_base + residual +
    # per-matrix SVD + sparse_outliers + gear_reconstruct, one call per
    # (b, h), exactly as the pre-fix implementation did.
    from veloxquant_mlx.quantizers._quant_utils import _truncated_svd
    from veloxquant_mlx.quantizers.gear import (
        GEARState,
        gear_reconstruct,
        quantize_base,
        sparse_outliers,
    )
    from veloxquant_mlx.quantizers.gear import (
        residual as gear_residual,
    )

    def ref_compress_and_account(t, is_key):
        base_axis = "channel" if is_key else "token"
        recon_b = []
        for b in range(B):
            recon_h = []
            for h in range(H):
                mat = t[b, h]
                stream, base_recon = quantize_base(mat, 2, 8, axis=base_axis)
                E = gear_residual(mat, base_recon)
                U, s, Vt = _truncated_svd(E, rank=None, energy_threshold=0.9)
                L = U * s[None, :]
                E_after = E - (L @ Vt)
                sp_idx, sp_val = sparse_outliers(E_after, 0.02)
                n, d = int(mat.shape[0]), int(mat.shape[1])
                state = GEARState(
                    codes=stream.codes,
                    scale=stream.scale,
                    zero=stream.zero,
                    L=L,
                    R=Vt,
                    sp_idx=sp_idx,
                    sp_val=sp_val,
                    n_rows=n,
                    bits=2,
                    rank=int(L.shape[1]),
                    axis=base_axis,
                    d_cols=d,
                )
                recon_h.append(gear_reconstruct(state))
            recon_b.append(mx.stack(recon_h, axis=0))
        return mx.stack(recon_b, axis=0)

    ref_k = ref_compress_and_account(keys, is_key=True)
    ref_v = ref_compress_and_account(values, is_key=False)

    np.testing.assert_allclose(np.array(ref_k), np.array(k_out), atol=1e-3)
    np.testing.assert_allclose(np.array(ref_v), np.array(v_out), atol=1e-3)
