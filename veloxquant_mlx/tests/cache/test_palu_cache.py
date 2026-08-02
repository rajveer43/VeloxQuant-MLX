"""Tests for PALUKVCache — true low-rank latent storage for keys *and* values.

PALU's distinguishing property vs SVDq is that the cache stores the latent
codes ``[S, r]`` directly and never holds full fp16 ``[S, D]`` for storage.
These tests assert that latent-storage property explicitly, plus the usual
projection-correctness, decode-accumulation, byte-accounting, and determinism
checks.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.palu_cache import PALUKVCache


def _make(**cfg):
    base = dict(
        method="palu",
        head_dim=64,
        palu_rank=16,  # explicit rank for deterministic tests
        palu_n_head_groups=2,
        palu_hi_bit=4,
        palu_lo_bit=2,
        palu_hi_fraction=0.25,
        palu_group_size=16,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S=128, H=4, D=64, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((1, H, S, D)).astype(np.float16)
    V = rng.standard_normal((1, H, S, D)).astype(np.float16)
    return mx.array(K), mx.array(V)


# ------------------------------------------------------------------
# Factory and interface
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    c = _make()
    assert isinstance(c, PALUKVCache)


def test_no_bits_attribute() -> None:
    """No .bits attribute — keeps mlx_lm SDPA on the clean fp16 path."""
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "assigned_avg_bits")


# ------------------------------------------------------------------
# Group-head SVD fit
# ------------------------------------------------------------------


def test_group_projections_stored_after_prefill() -> None:
    c = _make(palu_rank=16, palu_n_head_groups=2)
    K, V = _rand_kv(S=64, H=4, D=64)
    ko, vo = c.update_and_fetch(K, V)
    mx.eval(ko, vo)
    # Two head groups → two projection matrices for keys and for values.
    assert len(c._keys_lr._V) == 2
    assert len(c._vals_lr._V) == 2
    assert c._keys_lr._V[0].shape == (64, 16)
    assert c._keys_lr._mu[0].shape == (64,)
    assert c.rank == 16


def test_output_shape_preserved() -> None:
    c = _make()
    K, V = _rand_kv(S=64, H=4, D=64)
    ko, vo = c.update_and_fetch(K, V)
    mx.eval(ko, vo)
    assert ko.shape == (1, 4, 64, 64)
    assert vo.shape == (1, 4, 64, 64)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


# ------------------------------------------------------------------
# THE differentiator: latent storage stays [S, r], never [S, D]
# ------------------------------------------------------------------


def test_storage_is_latent_not_full_fp16() -> None:
    """The cache must store [S, r] latents, not [S, D] fp16 keys/values."""
    c = _make(head_dim=64, palu_rank=16)
    K, V = _rand_kv(S=64, H=4, D=64)
    c.update_and_fetch(K, V)
    # Per-head latent buffers hold the latent dimension r, not D.
    assert c._keys_lr._latents is not None
    assert c._vals_lr._latents is not None
    assert c._keys_lr._latents[0].shape[-1] == 16  # r, not 64
    assert c._vals_lr._latents[0].shape[-1] == 16
    # The parent fp16 ring buffer is bypassed — it never gets populated.
    assert c.keys is None and c.values is None


# ------------------------------------------------------------------
# Reconstruction quality on low-rank data — both K and V
# ------------------------------------------------------------------


def test_reconstruction_lower_mse_than_raw_2bit_both_tensors() -> None:
    """PALU beats naive 2-bit on low-rank data for BOTH keys and values."""
    from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

    rng = np.random.default_rng(42)
    S, D, true_rank = 128, 64, 8
    U = rng.standard_normal((S, true_rank)).astype(np.float32)
    Wk = rng.standard_normal((true_rank, D)).astype(np.float32)
    Wv = rng.standard_normal((true_rank, D)).astype(np.float32)
    K_np = U @ Wk + rng.standard_normal((S, D)).astype(np.float32) * 0.05
    V_np = U @ Wv + rng.standard_normal((S, D)).astype(np.float32) * 0.05
    K_mx, V_mx = mx.array(K_np), mx.array(V_np)

    c = _make(head_dim=D, palu_rank=true_rank, palu_n_head_groups=1)
    K_in = mx.array(K_np[None, None])
    V_in = mx.array(V_np[None, None])
    ko, vo = c.update_and_fetch(K_in, V_in)
    mx.eval(ko, vo)

    palu_k_mse = float(mx.mean((ko[0, 0].astype(mx.float32) - K_mx) ** 2).item())
    palu_v_mse = float(mx.mean((vo[0, 0].astype(mx.float32) - V_mx) ** 2).item())

    naive_k = _group_quant_dequant(K_mx, b=2, group_size=16)
    naive_v = _group_quant_dequant(V_mx, b=2, group_size=16)
    mx.eval(naive_k, naive_v)
    naive_k_mse = float(mx.mean((naive_k.astype(mx.float32) - K_mx) ** 2).item())
    naive_v_mse = float(mx.mean((naive_v.astype(mx.float32) - V_mx) ** 2).item())

    assert palu_k_mse < naive_k_mse, f"key MSE {palu_k_mse:.5f} !< {naive_k_mse:.5f}"
    assert palu_v_mse < naive_v_mse, f"val MSE {palu_v_mse:.5f} !< {naive_v_mse:.5f}"


# ------------------------------------------------------------------
# Decode accumulation
# ------------------------------------------------------------------


def test_decode_after_prefill() -> None:
    c = _make(palu_rank=16)
    K_pre, V_pre = _rand_kv(S=32, H=4, D=64, seed=0)
    c.update_and_fetch(K_pre, V_pre)
    for step in range(4):
        K_dec, V_dec = _rand_kv(S=1, H=4, D=64, seed=step + 10)
        ko, vo = c.update_and_fetch(K_dec, V_dec)
        mx.eval(ko, vo)
        assert ko.dtype == mx.float16
        assert not mx.any(mx.isnan(ko)).item()
        assert not mx.any(mx.isnan(vo)).item()
    # Sequence length grew by exactly the decode steps.
    assert c.offset == 32 + 4
    assert ko.shape == (1, 4, 36, 64)


# ------------------------------------------------------------------
# Byte accounting — BOTH tensors compress (unlike SVDq's fp16 values)
# ------------------------------------------------------------------


def test_both_tensors_compressed() -> None:
    c = _make(palu_rank=16)
    K, V = _rand_kv(S=128, H=4, D=64)
    c.update_and_fetch(K, V)
    assert 0 < c.compressed_key_bytes < c.fp16_key_bytes
    assert 0 < c.compressed_value_bytes < c.fp16_value_bytes


def test_low_rank_only_values_still_compress() -> None:
    """palu_quantize_values=False keeps latents fp16 but still wins via rank."""
    c = _make(palu_rank=16, palu_quantize_values=False)
    K, V = _rand_kv(S=128, H=4, D=64)
    c.update_and_fetch(K, V)
    assert c.compressed_value_bytes < c.fp16_value_bytes
    # fp16 latent rate is 16 * r/D bits.
    assert abs(c._vals_lr.assigned_avg_bits - 16.0 * c.rank / 64) < 1e-3


# ------------------------------------------------------------------
# Effective bit-width
# ------------------------------------------------------------------


def test_assigned_avg_bits_sub_2() -> None:
    """Low-rank + mixed-bit gives a deeply sub-2-bit effective rate."""
    c = _make(head_dim=128, palu_rank=32)
    K, V = _rand_kv(S=64, H=4, D=128)
    c.update_and_fetch(K, V)
    assert c.assigned_avg_bits < 2.0, f"got {c.assigned_avg_bits:.3f}"


# ------------------------------------------------------------------
# Energy-threshold rank selection
# ------------------------------------------------------------------


def test_energy_threshold_rank_selection() -> None:
    c = _make(palu_rank=None, palu_energy_threshold=0.90, head_dim=64)
    K, V = _rand_kv(S=128, H=4, D=64)
    c.update_and_fetch(K, V)
    assert 1 <= c.rank <= 64


# ------------------------------------------------------------------
# Head grouping
# ------------------------------------------------------------------


def test_single_group_vs_multi_group() -> None:
    """1 group → one shared projection; 4 groups → four projections."""
    c1 = _make(palu_n_head_groups=1)
    c4 = _make(palu_n_head_groups=4)
    K, V = _rand_kv(S=64, H=4, D=64)
    c1.update_and_fetch(K, V)
    c4.update_and_fetch(K, V)
    assert len(c1._keys_lr._V) == 1
    assert len(c4._keys_lr._V) == 4


# ------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------


def test_determinism() -> None:
    K, V = _rand_kv(S=64, H=4, D=64, seed=7)
    c1, c2 = _make(), _make()
    ko1, vo1 = c1.update_and_fetch(K, V)
    ko2, vo2 = c2.update_and_fetch(K, V)
    mx.eval(ko1, ko2, vo1, vo2)
    assert np.allclose(np.array(ko1), np.array(ko2), atol=1e-4)
    assert np.allclose(np.array(vo1), np.array(vo2), atol=1e-4)


# ---------------------------------------------------------------------------
# Config validation — palu_hi_fraction must be in [0, 1]
# ---------------------------------------------------------------------------


def test_hi_fraction_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="palu_hi_fraction"):
        _make(palu_hi_fraction=1.5)


def test_hi_fraction_negative_rejected() -> None:
    with pytest.raises(ValueError, match="palu_hi_fraction"):
        _make(palu_hi_fraction=-0.2)
