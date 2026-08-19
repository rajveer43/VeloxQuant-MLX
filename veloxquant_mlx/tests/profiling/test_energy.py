"""Tests for the unprivileged energy/throughput metrics.

These run as a normal user in CI. Two of them exist specifically to guard the
failure mode that would fabricate a measurement: energy that is missing must
surface as ``None``, never as ``0.0``.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.profiling.energy import (
    RunMetrics,
    compute_j_per_token,
    kv_bytes_per_token,
    measure_generation,
)

N_LAYERS = 32
N_KV_HEADS = 8
HEAD_DIM = 128


def test_kv_bytes_per_token_scales_with_bit_width():
    """4-bit KV approaches a quarter of fp16 as the fp16 residual amortises.

    It does not reach exactly 1/4: KIVI keeps the most recent
    ``residual_length`` tokens in fp16, and group scale/zero add a little on
    top. At a long sequence the residual amortises away and the ratio tends
    toward the bit ratio.
    """
    seq = 32768
    fp16 = kv_bytes_per_token(None, N_LAYERS, N_KV_HEADS, HEAD_DIM, seq)
    q4 = kv_bytes_per_token(
        KVCacheConfig(method="kivi", bit_width_inlier=4),
        N_LAYERS,
        N_KV_HEADS,
        HEAD_DIM,
        seq,
    )
    ratio = q4 / fp16
    assert 0.25 <= ratio < 0.35, f"4-bit/fp16 ratio was {ratio}"


def test_kv_bytes_per_token_residual_window_is_billed_at_fp16():
    """The fp16 residual tail must be charged at fp16, not the quantized width.

    Charging every resident token at 4 bits would understate KIVI's real
    traffic. Below the residual length nothing is quantized yet, so a 4-bit
    arm must read exactly what fp16 reads.
    """
    cfg = KVCacheConfig(method="kivi", bit_width_inlier=4, residual_length=128)
    short = 64  # entirely inside the fp16 residual window
    fp16 = kv_bytes_per_token(None, N_LAYERS, N_KV_HEADS, HEAD_DIM, short)
    q4 = kv_bytes_per_token(cfg, N_LAYERS, N_KV_HEADS, HEAD_DIM, short)
    assert q4 == fp16

    # And the ratio must improve monotonically as the residual amortises.
    r_short = q4 / fp16
    long_seq = 32768
    r_long = kv_bytes_per_token(cfg, N_LAYERS, N_KV_HEADS, HEAD_DIM, long_seq) / (
        kv_bytes_per_token(None, N_LAYERS, N_KV_HEADS, HEAD_DIM, long_seq)
    )
    assert r_long < r_short


def test_kv_bytes_per_token_2bit_is_smaller_than_4bit():
    """Bit-width ordering must be monotonic."""
    seq = 4096
    q2 = kv_bytes_per_token(
        KVCacheConfig(method="kivi", bit_width_inlier=2), N_LAYERS, N_KV_HEADS, HEAD_DIM, seq
    )
    q4 = kv_bytes_per_token(
        KVCacheConfig(method="kivi", bit_width_inlier=4), N_LAYERS, N_KV_HEADS, HEAD_DIM, seq
    )
    assert q2 < q4


def test_kv_bytes_per_token_is_capped_by_budget_for_eviction():
    """Past the budget, eviction stops traffic growing with sequence length.

    This is the asymmetry the harness exists to demonstrate: quantization
    scales traffic, eviction caps it.
    """
    cfg = KVCacheConfig(method="qfilters", qfilters_budget=512)
    at_budget = kv_bytes_per_token(cfg, N_LAYERS, N_KV_HEADS, HEAD_DIM, 512)
    way_past = kv_bytes_per_token(cfg, N_LAYERS, N_KV_HEADS, HEAD_DIM, 32768)
    assert at_budget == way_past, "eviction traffic must plateau at the budget"

    # And it must genuinely be below an uncapped fp16 read at that length.
    fp16 = kv_bytes_per_token(None, N_LAYERS, N_KV_HEADS, HEAD_DIM, 32768)
    assert way_past < fp16


def test_kv_bytes_per_token_grows_with_seq_len_below_budget():
    """Below the budget nothing is evicted yet, so traffic still grows."""
    cfg = KVCacheConfig(method="qfilters", qfilters_budget=4096)
    small = kv_bytes_per_token(cfg, N_LAYERS, N_KV_HEADS, HEAD_DIM, 256)
    bigger = kv_bytes_per_token(cfg, N_LAYERS, N_KV_HEADS, HEAD_DIM, 1024)
    assert bigger > small


def test_j_per_token_is_none_not_zero_when_energy_missing():
    """A silent 0.0 would propagate downstream as 'inference is free'."""
    assert compute_j_per_token(None, 128) is None
    assert compute_j_per_token(None, 128) != 0.0


def test_j_per_token_is_none_for_zero_tokens():
    """No division-by-zero, and no fabricated zero either."""
    assert compute_j_per_token(12.5, 0) is None


def test_j_per_token_computes_when_energy_present():
    assert compute_j_per_token(10.0, 100) == pytest.approx(0.1)


def test_metrics_degrade_to_none_without_power():
    """With no sampler, throughput and memory still populate; energy does not."""

    def prefill():
        return mx.ones((8, 8))

    def decode_step(_i):
        return mx.ones((4, 4)) * 2

    m = measure_generation(
        prefill=prefill,
        decode_step=decode_step,
        n_tokens=4,
        kv_bytes=1234,
        sampler=None,
        label="no-power",
    )
    assert isinstance(m, RunMetrics)
    # Energy path degrades to None -- not 0.0.
    assert m.energy_j is None
    assert m.j_per_token is None
    assert m.mean_gpu_mw is None
    assert m.mean_cpu_mw is None
    # Non-energy metrics are still real.
    assert m.tokens_generated == 4
    assert m.wall_s > 0
    assert m.decode_s > 0
    assert m.tokens_per_s > 0
    assert m.kv_bytes_per_token == 1234


def test_prefill_and_decode_are_timed_separately():
    """Folding prefill into a J/token average is a real confound; keep them apart."""

    def prefill():
        return mx.ones((256, 256)) @ mx.ones((256, 256))

    def decode_step(_i):
        return mx.ones((4, 4))

    m = measure_generation(prefill, decode_step, n_tokens=3, kv_bytes=0)
    assert m.prefill_s > 0
    assert m.decode_s > 0
    # wall is the sum of the two phases (within timer noise).
    assert m.wall_s == pytest.approx(m.prefill_s + m.decode_s, abs=1e-3)


def test_run_metrics_serializes_none_energy():
    """JSON output must carry null, not a stand-in number."""
    m = measure_generation(lambda: mx.ones((2, 2)), lambda i: mx.ones((2, 2)), 2, 99)
    d = m.to_dict()
    assert d["energy_j"] is None
    assert d["j_per_token"] is None
    assert d["kv_bytes_per_token"] == 99
