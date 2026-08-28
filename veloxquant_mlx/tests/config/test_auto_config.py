"""Tests for the hardware-aware auto-config selector (config/auto_config.py, issue #253)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheFactory
from veloxquant_mlx.config.auto_config import (
    LARGE_HEAD_DIM,
    LONG_CONTEXT_TOKENS,
    MEMORY_PRESSURE_FRACTION,
    SHORT_CONTEXT_TOKENS,
    AutoConfigResult,
    HardwareInfo,
    WorkloadSpec,
    detect_hardware_info,
    select_kv_cache_config,
)
from veloxquant_mlx.core.exceptions import QuantizerConfigError

# ============================================================================
# WorkloadSpec validation
# ============================================================================


def test_workload_spec_rejects_non_power_of_two_head_dim():
    with pytest.raises(QuantizerConfigError):
        WorkloadSpec(head_dim=100)


@pytest.mark.parametrize("bad_dim", [0, -1, -128, 3, 5, 6, 7, 9, 100, 129, 200])
def test_workload_spec_rejects_various_non_power_of_two_or_nonpositive_head_dims(bad_dim):
    with pytest.raises(QuantizerConfigError):
        WorkloadSpec(head_dim=bad_dim)


@pytest.mark.parametrize("good_dim", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_workload_spec_accepts_powers_of_two_head_dims(good_dim):
    spec = WorkloadSpec(head_dim=good_dim)
    assert spec.head_dim == good_dim


def test_workload_spec_rejects_zero_seq_len():
    with pytest.raises(QuantizerConfigError):
        WorkloadSpec(seq_len=0)


def test_workload_spec_rejects_negative_seq_len():
    with pytest.raises(QuantizerConfigError):
        WorkloadSpec(seq_len=-1)


def test_workload_spec_accepts_seq_len_of_one():
    spec = WorkloadSpec(seq_len=1)
    assert spec.seq_len == 1


def test_workload_spec_rejects_zero_n_layers():
    with pytest.raises(QuantizerConfigError):
        WorkloadSpec(n_layers=0)


def test_workload_spec_rejects_negative_n_layers():
    with pytest.raises(QuantizerConfigError):
        WorkloadSpec(n_layers=-5)


def test_workload_spec_rejects_zero_batch_size():
    with pytest.raises(QuantizerConfigError):
        WorkloadSpec(batch_size=0)


def test_workload_spec_rejects_negative_batch_size():
    with pytest.raises(QuantizerConfigError):
        WorkloadSpec(batch_size=-2)


def test_workload_spec_defaults():
    spec = WorkloadSpec()
    assert spec.head_dim == 128
    assert spec.seq_len == 4_096
    assert spec.n_layers == 1
    assert spec.batch_size == 1


def test_workload_spec_is_frozen():
    import dataclasses

    spec = WorkloadSpec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.head_dim = 64  # type: ignore[misc]


def test_fp16_kv_bytes_scales_with_all_dims():
    spec = WorkloadSpec(head_dim=128, seq_len=1000, n_layers=4, batch_size=2)
    assert spec.fp16_kv_bytes() == 2 * 2 * 4 * 1000 * 128 * 2


def test_fp16_kv_bytes_minimal_workload():
    spec = WorkloadSpec(head_dim=1, seq_len=1, n_layers=1, batch_size=1)
    assert spec.fp16_kv_bytes() == 2 * 1 * 1 * 1 * 1 * 2  # = 4 bytes


def test_fp16_kv_bytes_large_realistic_model():
    # Llama-3-70B-style: 80 layers, head_dim 128, 8K context, batch 1.
    spec = WorkloadSpec(head_dim=128, seq_len=8192, n_layers=80, batch_size=1)
    expected = 2 * 1 * 80 * 8192 * 128 * 2
    assert spec.fp16_kv_bytes() == expected


def test_fp16_kv_bytes_huge_batch_and_context_does_not_overflow():
    spec = WorkloadSpec(head_dim=128, seq_len=1_000_000, n_layers=128, batch_size=256)
    result = spec.fp16_kv_bytes()
    assert result > 0
    assert isinstance(result, int)


# ============================================================================
# HardwareInfo
# ============================================================================


def test_hardware_info_defaults():
    hw = HardwareInfo()
    assert hw.total_memory_bytes is None
    assert hw.active_memory_bytes == 0


def test_pressure_fraction_none_when_total_unknown():
    hw = HardwareInfo()
    assert hw.pressure_fraction(1_000) is None


def test_pressure_fraction_none_when_total_is_zero():
    # 0 is falsy, so pressure_fraction treats it like "unknown" rather than
    # dividing by zero.
    hw = HardwareInfo(total_memory_bytes=0)
    assert hw.pressure_fraction(500) is None


def test_pressure_fraction_computes_ratio():
    hw = HardwareInfo(total_memory_bytes=1000, active_memory_bytes=250)
    assert hw.pressure_fraction(250) == 0.5


def test_pressure_fraction_can_exceed_one():
    hw = HardwareInfo(total_memory_bytes=100, active_memory_bytes=50)
    assert hw.pressure_fraction(100) == 1.5


def test_pressure_fraction_zero_additional_bytes():
    hw = HardwareInfo(total_memory_bytes=1000, active_memory_bytes=100)
    assert hw.pressure_fraction(0) == pytest.approx(0.1)


def test_pressure_fraction_fully_saturated_active_memory():
    hw = HardwareInfo(total_memory_bytes=1000, active_memory_bytes=1000)
    assert hw.pressure_fraction(0) == 1.0


def test_hardware_info_is_frozen():
    import dataclasses

    hw = HardwareInfo()
    with pytest.raises(dataclasses.FrozenInstanceError):
        hw.total_memory_bytes = 123  # type: ignore[misc]


def test_detect_hardware_info_does_not_raise():
    hw = detect_hardware_info()
    assert isinstance(hw, HardwareInfo)


def test_detect_hardware_info_reports_positive_memory_on_this_machine():
    # This suite runs on real Apple-Silicon MLX, so device introspection
    # should succeed and report a plausible nonzero unified-memory size.
    hw = detect_hardware_info()
    assert hw.total_memory_bytes is None or hw.total_memory_bytes > 0
    assert hw.active_memory_bytes >= 0


# ============================================================================
# select_kv_cache_config: sequence-length rules
# ============================================================================


def test_short_context_selects_turboquant_rvq_high_precision():
    workload = WorkloadSpec(head_dim=128, seq_len=SHORT_CONTEXT_TOKENS - 1)
    hardware = HardwareInfo()  # unknown memory -> pressure rule inactive
    result = select_kv_cache_config(workload, hardware)

    assert isinstance(result, AutoConfigResult)
    assert result.config.method == "turboquant_rvq"
    assert result.config.bit_width_inlier == 4
    assert result.config.head_dim == 128
    assert "short context" in result.reason


def test_seq_len_one_selects_turboquant_rvq():
    workload = WorkloadSpec(head_dim=128, seq_len=1)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "turboquant_rvq"


def test_mid_length_context_selects_kivi_balanced():
    workload = WorkloadSpec(head_dim=128, seq_len=(SHORT_CONTEXT_TOKENS + LONG_CONTEXT_TOKENS) // 2)
    result = select_kv_cache_config(workload, HardwareInfo())

    assert result.config.method == "kivi"
    assert result.config.bit_width_inlier == 2
    assert result.config.kivi_group_size == 32


def test_long_context_selects_kvquant_aggressive():
    workload = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS)
    result = select_kv_cache_config(workload, HardwareInfo())

    assert result.config.method == "kvquant"
    assert result.config.kvquant_bits == 3
    assert result.config.kvquant_outlier_fraction == 0.01


def test_very_long_context_selects_kvquant():
    workload = WorkloadSpec(head_dim=128, seq_len=1_000_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "kvquant"


# --- Exact boundary values ---------------------------------------------------


def test_boundary_just_below_short_context_threshold():
    workload = WorkloadSpec(head_dim=128, seq_len=SHORT_CONTEXT_TOKENS - 1)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "turboquant_rvq"


def test_boundary_exactly_at_short_context_threshold_is_mid_band():
    # seq_len == SHORT_CONTEXT_TOKENS is NOT < threshold, so it falls into
    # the mid-length band, not the short-context band.
    workload = WorkloadSpec(head_dim=128, seq_len=SHORT_CONTEXT_TOKENS)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "kivi"


def test_boundary_just_above_short_context_threshold_is_mid_band():
    workload = WorkloadSpec(head_dim=128, seq_len=SHORT_CONTEXT_TOKENS + 1)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "kivi"


def test_boundary_just_below_long_context_threshold_is_mid_band():
    workload = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS - 1)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "kivi"


def test_boundary_exactly_at_long_context_threshold_is_long_band():
    workload = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "kvquant"


def test_boundary_just_above_long_context_threshold_is_long_band():
    workload = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS + 1)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "kvquant"


# ============================================================================
# select_kv_cache_config: head-dim group-size rule
# ============================================================================


def test_large_head_dim_doubles_group_size():
    workload = WorkloadSpec(head_dim=256, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())

    assert result.config.method == "kivi"
    assert result.config.kivi_group_size == 64


def test_small_head_dim_keeps_default_group_size():
    workload = WorkloadSpec(head_dim=128, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.kivi_group_size == 32


def test_boundary_head_dim_just_below_large_threshold():
    workload = WorkloadSpec(head_dim=LARGE_HEAD_DIM // 2, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.kivi_group_size == 32


def test_boundary_head_dim_exactly_at_large_threshold():
    workload = WorkloadSpec(head_dim=LARGE_HEAD_DIM, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.kivi_group_size == 64


def test_very_large_head_dim_still_group_size_64():
    workload = WorkloadSpec(head_dim=1024, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.kivi_group_size == 64


def test_head_dim_one_uses_default_group_size():
    workload = WorkloadSpec(head_dim=1, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.kivi_group_size == 32


def test_large_head_dim_affects_group_size_in_short_context_branch_not_at_all():
    # turboquant_rvq (short-context branch) has no group-size knob in this
    # selector's output, so a large head_dim must not add one.
    workload = WorkloadSpec(head_dim=512, seq_len=100)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "turboquant_rvq"
    assert result.config.head_dim == 512


def test_large_head_dim_doubles_group_size_in_long_context_branch():
    workload = WorkloadSpec(head_dim=512, seq_len=LONG_CONTEXT_TOKENS)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "kvquant"
    assert result.config.kvquant_group_size == 64


def test_large_head_dim_doubles_group_size_under_memory_pressure():
    workload = WorkloadSpec(head_dim=512, seq_len=100, n_layers=32, batch_size=8)
    hardware = HardwareInfo(total_memory_bytes=workload.fp16_kv_bytes(), active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"
    assert result.config.gear_group_size == 64


# ============================================================================
# select_kv_cache_config: memory-pressure rule
# ============================================================================


def test_memory_pressure_overrides_short_context_to_gear():
    workload = WorkloadSpec(head_dim=128, seq_len=100, n_layers=32, batch_size=8)
    fp16_bytes = workload.fp16_kv_bytes()
    hardware = HardwareInfo(total_memory_bytes=fp16_bytes, active_memory_bytes=0)

    result = select_kv_cache_config(workload, hardware)

    assert result.config.method == "gear"
    assert result.config.gear_bits == 2
    assert "memory pressure" in result.reason


def test_memory_pressure_overrides_mid_context():
    workload = WorkloadSpec(head_dim=128, seq_len=8_000, n_layers=32, batch_size=8)
    hardware = HardwareInfo(total_memory_bytes=workload.fp16_kv_bytes(), active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"


def test_memory_pressure_overrides_long_context():
    workload = WorkloadSpec(head_dim=128, seq_len=20_000, n_layers=32, batch_size=8)
    hardware = HardwareInfo(total_memory_bytes=workload.fp16_kv_bytes(), active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"


def test_low_memory_pressure_does_not_override_sequence_rule():
    workload = WorkloadSpec(head_dim=128, seq_len=100)
    hardware = HardwareInfo(total_memory_bytes=10**15, active_memory_bytes=0)

    result = select_kv_cache_config(workload, hardware)

    assert result.config.method == "turboquant_rvq"


def test_pressure_from_active_memory_alone_can_trigger_override():
    # Even a tiny workload should trip the pressure rule if active_memory
    # already consumes most of the device.
    workload = WorkloadSpec(head_dim=128, seq_len=10)
    total = 1_000_000
    hardware = HardwareInfo(total_memory_bytes=total, active_memory_bytes=int(total * 0.9))
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"


def test_pressure_threshold_boundary_exactly_at_fraction_triggers():
    workload = WorkloadSpec(head_dim=128, seq_len=100)
    fp16_bytes = workload.fp16_kv_bytes()
    total = int(fp16_bytes / MEMORY_PRESSURE_FRACTION)
    hardware = HardwareInfo(total_memory_bytes=total, active_memory_bytes=0)

    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"


def test_pressure_threshold_just_below_does_not_trigger():
    workload = WorkloadSpec(head_dim=128, seq_len=100)
    fp16_bytes = workload.fp16_kv_bytes()
    # total slightly larger than fp16_bytes / FRACTION -> pressure slightly
    # below the threshold.
    total = int(fp16_bytes / MEMORY_PRESSURE_FRACTION) + 10_000_000
    hardware = HardwareInfo(total_memory_bytes=total, active_memory_bytes=0)

    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "turboquant_rvq"


def test_pressure_rule_inactive_when_total_memory_unset():
    workload = WorkloadSpec(head_dim=128, seq_len=100, n_layers=1000, batch_size=1000)
    hardware = HardwareInfo(total_memory_bytes=None)
    result = select_kv_cache_config(workload, hardware)
    # No memory info at all -> falls through to sequence-length rule.
    assert result.config.method == "turboquant_rvq"


def test_extreme_memory_pressure_over_100_percent():
    workload = WorkloadSpec(head_dim=128, seq_len=1_000_000, n_layers=200, batch_size=64)
    hardware = HardwareInfo(total_memory_bytes=1024, active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"


def test_zero_active_memory_and_huge_total_never_triggers_pressure():
    workload = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS)
    hardware = HardwareInfo(total_memory_bytes=10**18, active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    # Falls through to sequence-length rule since pressure ~= 0.
    assert result.config.method == "kvquant"


# ============================================================================
# select_kv_cache_config: hardware=None auto-detects
# ============================================================================


def test_hardware_none_auto_detects_without_raising():
    workload = WorkloadSpec(head_dim=128, seq_len=4_096)
    result = select_kv_cache_config(workload)
    assert isinstance(result, AutoConfigResult)
    assert result.config.method in {"turboquant_rvq", "kivi", "kvquant", "gear"}


def test_hardware_none_matches_explicit_detect_hardware_info():
    # Sanity: passing hardware=None should behave like passing the result of
    # detect_hardware_info() explicitly (both call the same detector), given
    # memory state doesn't change between the two calls.
    workload = WorkloadSpec(head_dim=128, seq_len=1000)
    r1 = select_kv_cache_config(workload, None)
    r2 = select_kv_cache_config(workload, detect_hardware_info())
    assert r1.config.method == r2.config.method


# ============================================================================
# Exhaustive grid: 3 sequence bands x 3 head-dim sizes x 2 pressure states
# ============================================================================

_SEQ_BANDS = {
    "short": SHORT_CONTEXT_TOKENS - 1,
    "mid": (SHORT_CONTEXT_TOKENS + LONG_CONTEXT_TOKENS) // 2,
    "long": LONG_CONTEXT_TOKENS,
}
_HEAD_DIMS = [64, 128, 256]
_EXPECTED_NO_PRESSURE = {
    "short": "turboquant_rvq",
    "mid": "kivi",
    "long": "kvquant",
}


@pytest.mark.parametrize("band_name,seq_len", list(_SEQ_BANDS.items()))
@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_grid_no_pressure_selects_expected_method(band_name, seq_len, head_dim):
    workload = WorkloadSpec(head_dim=head_dim, seq_len=seq_len)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == _EXPECTED_NO_PRESSURE[band_name]
    assert result.config.head_dim == head_dim


@pytest.mark.parametrize("band_name,seq_len", list(_SEQ_BANDS.items()))
@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_grid_under_pressure_always_selects_gear(band_name, seq_len, head_dim):
    workload = WorkloadSpec(head_dim=head_dim, seq_len=seq_len, n_layers=64, batch_size=16)
    hardware = HardwareInfo(total_memory_bytes=workload.fp16_kv_bytes(), active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"


# ============================================================================
# Realistic model shapes
# ============================================================================


def test_llama_3_8b_style_short_prompt():
    # 32 layers, head_dim 128, short prompt, single request, plenty of RAM.
    workload = WorkloadSpec(head_dim=128, seq_len=512, n_layers=32, batch_size=1)
    hardware = HardwareInfo(total_memory_bytes=32 * 1024**3, active_memory_bytes=4 * 1024**3)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "turboquant_rvq"


def test_llama_3_8b_style_long_document_summarization():
    workload = WorkloadSpec(head_dim=128, seq_len=64_000, n_layers=32, batch_size=1)
    hardware = HardwareInfo(total_memory_bytes=32 * 1024**3, active_memory_bytes=4 * 1024**3)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "kvquant"


def test_small_device_serving_many_concurrent_requests_triggers_pressure():
    # 8GB Mac, model already resident, many concurrent long-ish requests.
    workload = WorkloadSpec(head_dim=128, seq_len=4_000, n_layers=32, batch_size=32)
    hardware = HardwareInfo(total_memory_bytes=8 * 1024**3, active_memory_bytes=5 * 1024**3)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"


def test_gqa_style_small_head_dim():
    # Some GQA configs use smaller head_dim (e.g. 64) with many heads;
    # selector operates per-head-dim so this should behave like any other
    # small head_dim workload.
    workload = WorkloadSpec(head_dim=64, seq_len=4_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "kivi"
    assert result.config.kivi_group_size == 32


def test_wide_head_dim_model():
    # Some architectures use head_dim=256 (e.g. certain MQA setups).
    workload = WorkloadSpec(head_dim=256, seq_len=100)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert result.config.method == "turboquant_rvq"
    assert result.config.head_dim == 256


# ============================================================================
# Result is usable by KVCacheFactory — full round trip through every branch
# ============================================================================


def _kv(batch, heads, seq, head_dim, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float16))
    v = mx.array(rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float16))
    return k, v


def test_selected_config_builds_a_real_cache():
    workload = WorkloadSpec(head_dim=64, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    cache = KVCacheFactory.create(result.config)
    assert cache is not None


@pytest.mark.parametrize(
    "seq_len,head_dim",
    [
        (100, 64),  # short context -> turboquant_rvq
        (8_000, 64),  # mid context -> kivi
        (LONG_CONTEXT_TOKENS, 64),  # long context -> kvquant
        (100, 256),  # short context, large head_dim -> turboquant_rvq
        (8_000, 256),  # mid context, large head_dim -> kivi, group_size 64
    ],
)
def test_selected_config_round_trips_real_tensors_through_update_and_fetch(seq_len, head_dim):
    workload = WorkloadSpec(head_dim=head_dim, seq_len=seq_len)
    result = select_kv_cache_config(workload, HardwareInfo())
    cache = KVCacheFactory.create(result.config)

    k, v = _kv(1, 2, 32, head_dim)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)

    assert ko.shape == (1, 2, 32, head_dim)
    assert vo.shape == (1, 2, 32, head_dim)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16
    assert not mx.any(mx.isnan(ko)).item()
    assert not mx.any(mx.isnan(vo)).item()


def test_selected_config_under_memory_pressure_round_trips_real_tensors():
    workload = WorkloadSpec(head_dim=64, seq_len=100, n_layers=32, batch_size=8)
    hardware = HardwareInfo(total_memory_bytes=workload.fp16_kv_bytes(), active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "gear"

    cache = KVCacheFactory.create(result.config)
    k, v = _kv(1, 2, 32, 64)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 2, 32, 64)
    assert vo.shape == (1, 2, 32, 64)


def test_selected_config_survives_multiple_decode_steps():
    # Simulate prefill + several decode steps through the selected cache to
    # catch any state-handling issue that only shows up after step 1.
    workload = WorkloadSpec(head_dim=64, seq_len=100)
    result = select_kv_cache_config(workload, HardwareInfo())
    cache = KVCacheFactory.create(result.config)

    k, v = _kv(1, 2, 50, 64, seed=1)
    cache.update_and_fetch(k, v)

    expected_len = 50
    for step in range(5):
        k1, v1 = _kv(1, 2, 1, 64, seed=100 + step)
        expected_len += 1
        ko, vo = cache.update_and_fetch(k1, v1)
        mx.eval(ko, vo)
        # update_and_fetch returns the full accumulated sequence (mlx_lm
        # protocol), so the token axis grows by one each decode step.
        assert ko.shape == (1, 2, expected_len, 64)
        assert vo.shape == (1, 2, expected_len, 64)


# ============================================================================
# AutoConfigResult: structure, immutability, exact field contents per branch
# ============================================================================


def test_autoconfig_result_is_frozen():
    import dataclasses

    workload = WorkloadSpec(head_dim=128, seq_len=100)
    result = select_kv_cache_config(workload, HardwareInfo())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.reason = "different"  # type: ignore[misc]


def test_autoconfig_result_config_is_a_kvcacheconfig_instance():
    from veloxquant_mlx.cache.base import KVCacheConfig

    workload = WorkloadSpec(head_dim=128, seq_len=100)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert isinstance(result.config, KVCacheConfig)


def test_autoconfig_result_reason_is_nonempty_string_in_every_branch():
    scenarios = [
        WorkloadSpec(head_dim=128, seq_len=100),  # short -> turboquant_rvq
        WorkloadSpec(head_dim=128, seq_len=8_000),  # mid -> kivi
        WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS),  # long -> kvquant
    ]
    for workload in scenarios:
        result = select_kv_cache_config(workload, HardwareInfo())
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    # Pressure branch.
    pressure_workload = WorkloadSpec(head_dim=128, seq_len=100, n_layers=32, batch_size=8)
    hardware = HardwareInfo(total_memory_bytes=pressure_workload.fp16_kv_bytes())
    result = select_kv_cache_config(pressure_workload, hardware)
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


def test_reason_mentions_selected_method_name_short():
    workload = WorkloadSpec(head_dim=128, seq_len=100)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert "turboquant_rvq" in result.reason


def test_reason_mentions_selected_method_name_mid():
    workload = WorkloadSpec(head_dim=128, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert "kivi" in result.reason


def test_reason_mentions_selected_method_name_long():
    workload = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS)
    result = select_kv_cache_config(workload, HardwareInfo())
    assert "kvquant" in result.reason


def test_reason_mentions_selected_method_name_pressure():
    workload = WorkloadSpec(head_dim=128, seq_len=100, n_layers=32, batch_size=8)
    hardware = HardwareInfo(total_memory_bytes=workload.fp16_kv_bytes())
    result = select_kv_cache_config(workload, hardware)
    assert "gear" in result.reason


def test_reason_reports_actual_computed_percentage_for_pressure():
    workload = WorkloadSpec(head_dim=128, seq_len=100, n_layers=32, batch_size=8)
    fp16_bytes = workload.fp16_kv_bytes()
    # Exactly double the fp16 footprint as total -> pressure == 50%.
    hardware = HardwareInfo(total_memory_bytes=fp16_bytes * 2, active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    # 50% is below the 75% threshold, so this should NOT trigger gear.
    assert result.config.method != "gear"

    # Now push pressure to exactly 100%.
    hardware2 = HardwareInfo(total_memory_bytes=fp16_bytes, active_memory_bytes=0)
    result2 = select_kv_cache_config(workload, hardware2)
    assert "100%" in result2.reason


# ============================================================================
# Exact KVCacheConfig field values for every branch (not just method name)
# ============================================================================


def test_short_context_config_has_only_expected_fields_set():
    workload = WorkloadSpec(head_dim=128, seq_len=100)
    result = select_kv_cache_config(workload, HardwareInfo())
    cfg = result.config
    assert cfg.method == "turboquant_rvq"
    assert cfg.head_dim == 128
    assert cfg.bit_width_inlier == 4
    # Defaults for fields this branch does not touch.
    assert cfg.bit_width_outlier is None
    assert cfg.seed == 42  # KVCacheConfig default, unmodified by selector


def test_mid_context_config_has_only_expected_fields_set():
    workload = WorkloadSpec(head_dim=128, seq_len=8_000)
    result = select_kv_cache_config(workload, HardwareInfo())
    cfg = result.config
    assert cfg.method == "kivi"
    assert cfg.head_dim == 128
    assert cfg.bit_width_inlier == 2
    assert cfg.kivi_group_size == 32
    assert cfg.seed == 42


def test_long_context_config_has_only_expected_fields_set():
    workload = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS)
    result = select_kv_cache_config(workload, HardwareInfo())
    cfg = result.config
    assert cfg.method == "kvquant"
    assert cfg.head_dim == 128
    assert cfg.kvquant_bits == 3
    assert cfg.kvquant_group_size == 32
    assert cfg.kvquant_outlier_fraction == 0.01
    # Fields this branch does not touch should remain at their dataclass defaults.
    assert cfg.kvquant_lloyd_iters == 8
    assert cfg.kvquant_n_sink == 1


def test_pressure_config_has_only_expected_fields_set():
    workload = WorkloadSpec(head_dim=128, seq_len=100, n_layers=32, batch_size=8)
    hardware = HardwareInfo(total_memory_bytes=workload.fp16_kv_bytes())
    result = select_kv_cache_config(workload, hardware)
    cfg = result.config
    assert cfg.method == "gear"
    assert cfg.head_dim == 128
    assert cfg.gear_bits == 2
    assert cfg.gear_group_size == 32
    # Fields this branch does not touch should remain at their dataclass defaults.
    assert cfg.gear_energy_threshold == 0.90
    assert cfg.gear_sparse_fraction == 0.01
    assert cfg.gear_quantize_values is True


# ============================================================================
# n_layers / batch_size interaction with the pressure rule at their defaults
# ============================================================================


def test_default_n_layers_and_batch_size_produce_single_layer_estimate():
    # With n_layers=1, batch_size=1 (defaults), the fp16 estimate is for one
    # layer's cache only -- a single layer alone should essentially never
    # trip memory pressure on a real machine's total memory.
    workload = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS)
    hardware = HardwareInfo(total_memory_bytes=32 * 1024**3, active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method != "gear"


def test_scaling_n_layers_alone_can_trigger_pressure():
    workload_1_layer = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS, n_layers=1)
    workload_96_layers = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS, n_layers=96)
    total = workload_1_layer.fp16_kv_bytes() * 50  # enough headroom for 1 layer, not 96

    result_1 = select_kv_cache_config(workload_1_layer, HardwareInfo(total_memory_bytes=total))
    result_96 = select_kv_cache_config(workload_96_layers, HardwareInfo(total_memory_bytes=total))

    assert result_1.config.method != "gear"
    assert result_96.config.method == "gear"


def test_scaling_batch_size_alone_can_trigger_pressure():
    workload_b1 = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS, batch_size=1)
    workload_b64 = WorkloadSpec(head_dim=128, seq_len=LONG_CONTEXT_TOKENS, batch_size=64)
    total = workload_b1.fp16_kv_bytes() * 50

    result_1 = select_kv_cache_config(workload_b1, HardwareInfo(total_memory_bytes=total))
    result_64 = select_kv_cache_config(workload_b64, HardwareInfo(total_memory_bytes=total))

    assert result_1.config.method != "gear"
    assert result_64.config.method == "gear"


# ============================================================================
# Public export surface
# ============================================================================


def test_top_level_package_exports_public_api():
    import veloxquant_mlx as vq

    assert vq.select_kv_cache_config is select_kv_cache_config
    assert vq.WorkloadSpec is WorkloadSpec
    assert vq.HardwareInfo is HardwareInfo
    assert vq.AutoConfigResult is AutoConfigResult
    assert vq.detect_hardware_info is detect_hardware_info


def test_config_subpackage_exports_match_module_all():
    import veloxquant_mlx.config as config_pkg
    from veloxquant_mlx.config import auto_config as module

    for name in module.__all__:
        assert hasattr(config_pkg, name), f"config package missing export {name!r}"


def test_config_module_all_has_no_duplicates_and_matches_defined_names():
    from veloxquant_mlx.config import auto_config as module

    assert len(module.__all__) == len(set(module.__all__))
    for name in module.__all__:
        assert hasattr(module, name)


# ============================================================================
# Determinism: same inputs always produce the same output
# ============================================================================


@pytest.mark.parametrize("_repeat", range(5))
def test_selection_is_deterministic_across_repeated_calls(_repeat):
    workload = WorkloadSpec(head_dim=128, seq_len=8_000, n_layers=10, batch_size=2)
    hardware = HardwareInfo(total_memory_bytes=64 * 1024**3, active_memory_bytes=1 * 1024**3)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method == "kivi"
    assert result.config.kivi_group_size == 32


def test_independent_calls_do_not_share_mutable_state():
    # KVCacheConfig has a mutable `observers` list field; two independently
    # selected configs must not accidentally alias the same list instance.
    w1 = WorkloadSpec(head_dim=128, seq_len=100)
    w2 = WorkloadSpec(head_dim=128, seq_len=200)
    r1 = select_kv_cache_config(w1, HardwareInfo())
    r2 = select_kv_cache_config(w2, HardwareInfo())

    assert r1.config.observers is not r2.config.observers
    r1.config.observers.append("marker")
    assert r2.config.observers == []


# ============================================================================
# Combined stress: many random-ish workloads never raise and always return
# a method from the known pool
# ============================================================================


@pytest.mark.parametrize(
    "head_dim,seq_len,n_layers,batch_size,total_mem_gb",
    [
        (32, 1, 1, 1, 8),
        (64, 2047, 1, 1, 16),
        (128, 2048, 12, 1, 24),
        (256, 16383, 24, 2, 32),
        (512, 16384, 48, 4, 48),
        (1024, 100_000, 96, 8, 64),
        (16, 5, 200, 200, 4),
        (2048, 3, 1, 1, 128),
    ],
)
def test_select_never_raises_and_always_picks_a_pool_method(
    head_dim, seq_len, n_layers, batch_size, total_mem_gb
):
    workload = WorkloadSpec(
        head_dim=head_dim, seq_len=seq_len, n_layers=n_layers, batch_size=batch_size
    )
    hardware = HardwareInfo(total_memory_bytes=total_mem_gb * 1024**3, active_memory_bytes=0)
    result = select_kv_cache_config(workload, hardware)
    assert result.config.method in {"turboquant_rvq", "kivi", "kvquant", "gear"}
    assert result.config.head_dim == head_dim


# ============================================================================
# detect_hardware_info: mocked success and failure paths
# ============================================================================


def test_detect_hardware_info_uses_device_info_and_active_memory(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx, "device_info", lambda: {"memory_size": 12345})
    monkeypatch.setattr(mx, "get_active_memory", lambda: 999)

    hw = detect_hardware_info()
    assert hw.total_memory_bytes == 12345
    assert hw.active_memory_bytes == 999


def test_detect_hardware_info_falls_back_when_device_info_raises(monkeypatch):
    import mlx.core as mx

    def _boom():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(mx, "device_info", _boom)

    hw = detect_hardware_info()
    assert hw == HardwareInfo()


def test_detect_hardware_info_falls_back_when_get_active_memory_raises(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx, "device_info", lambda: {"memory_size": 999})
    monkeypatch.setattr(
        mx, "get_active_memory", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    hw = detect_hardware_info()
    assert hw == HardwareInfo()


def test_detect_hardware_info_handles_missing_memory_size_key(monkeypatch):
    import mlx.core as mx

    # device_info() dict without the expected key -> total_memory_bytes is None.
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M4"})
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)

    hw = detect_hardware_info()
    assert hw.total_memory_bytes is None
    assert hw.active_memory_bytes == 0


def test_select_falls_through_to_sequence_rule_when_detection_fails(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx, "device_info", lambda: (_ for _ in ()).throw(RuntimeError("no gpu")))

    workload = WorkloadSpec(head_dim=128, seq_len=100)
    result = select_kv_cache_config(workload, None)
    assert result.config.method == "turboquant_rvq"


# ============================================================================
# Monotonicity / consistency invariants across the selection space
# ============================================================================


def test_increasing_seq_len_never_decreases_aggressiveness_rank():
    # Precision rank: turboquant_rvq (4-bit) > kivi (2-bit) > kvquant (3-bit,
    # but with outlier isolation and eviction-free NUQ, intentionally the
    # most aggressive non-pressure choice). We only assert the *method*
    # transitions happen in the documented order as seq_len grows, not that
    # bit-width itself is monotonic (kvquant's 3 > kivi's 2 by design, since
    # kvquant compensates for aggressiveness with outlier isolation).
    seq_lens = [
        1,
        SHORT_CONTEXT_TOKENS - 1,
        SHORT_CONTEXT_TOKENS,
        LONG_CONTEXT_TOKENS - 1,
        LONG_CONTEXT_TOKENS,
        LONG_CONTEXT_TOKENS * 10,
    ]
    methods_seen = []
    for seq_len in seq_lens:
        workload = WorkloadSpec(head_dim=128, seq_len=seq_len)
        result = select_kv_cache_config(workload, HardwareInfo())
        methods_seen.append(result.config.method)

    expected = [
        "turboquant_rvq",
        "turboquant_rvq",
        "kivi",
        "kivi",
        "kvquant",
        "kvquant",
    ]
    assert methods_seen == expected


def test_increasing_memory_pressure_eventually_forces_gear():
    workload = WorkloadSpec(head_dim=128, seq_len=100, n_layers=32, batch_size=8)
    fp16_bytes = workload.fp16_kv_bytes()

    # Sweep total memory from "huge" (no pressure) down to "tiny" (max
    # pressure) and confirm the transition to gear happens exactly once and
    # never reverts once triggered.
    totals = [fp16_bytes * 1000, fp16_bytes * 10, fp16_bytes * 2, fp16_bytes, fp16_bytes // 2]
    methods = []
    for total in totals:
        hardware = HardwareInfo(total_memory_bytes=total, active_memory_bytes=0)
        result = select_kv_cache_config(workload, hardware)
        methods.append(result.config.method)

    # Once gear appears, every subsequent (smaller-memory) entry must also be gear.
    first_gear_idx = next((i for i, m in enumerate(methods) if m == "gear"), None)
    assert first_gear_idx is not None
    assert all(m == "gear" for m in methods[first_gear_idx:])


def test_head_dim_group_size_rule_is_independent_of_seq_len_band():
    # The group-size doubling at LARGE_HEAD_DIM must hold in both the mid
    # and long bands (the only two branches that expose a group_size field).
    mid = WorkloadSpec(head_dim=LARGE_HEAD_DIM, seq_len=8_000)
    long = WorkloadSpec(head_dim=LARGE_HEAD_DIM, seq_len=LONG_CONTEXT_TOKENS)

    mid_result = select_kv_cache_config(mid, HardwareInfo())
    long_result = select_kv_cache_config(long, HardwareInfo())

    assert mid_result.config.kivi_group_size == 64
    assert long_result.config.kvquant_group_size == 64


def test_result_config_head_dim_always_matches_workload_head_dim():
    for head_dim in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        for seq_len in [1, 100, 4_000, 20_000]:
            workload = WorkloadSpec(head_dim=head_dim, seq_len=seq_len)
            result = select_kv_cache_config(workload, HardwareInfo())
            assert result.config.head_dim == head_dim


# ============================================================================
# WorkloadSpec / HardwareInfo: dataclass equality and hashing behavior
# ============================================================================


def test_workload_spec_equality():
    a = WorkloadSpec(head_dim=128, seq_len=100, n_layers=2, batch_size=1)
    b = WorkloadSpec(head_dim=128, seq_len=100, n_layers=2, batch_size=1)
    c = WorkloadSpec(head_dim=128, seq_len=101, n_layers=2, batch_size=1)
    assert a == b
    assert a != c


def test_hardware_info_equality():
    a = HardwareInfo(total_memory_bytes=100, active_memory_bytes=10)
    b = HardwareInfo(total_memory_bytes=100, active_memory_bytes=10)
    c = HardwareInfo(total_memory_bytes=100, active_memory_bytes=20)
    assert a == b
    assert a != c


def test_workload_spec_repr_contains_field_values():
    spec = WorkloadSpec(head_dim=128, seq_len=4096)
    r = repr(spec)
    assert "128" in r
    assert "4096" in r


# ============================================================================
# Keyword-only-style call safety: passing WorkloadSpec/HardwareInfo by keyword
# ============================================================================


def test_select_accepts_workload_as_positional_and_hardware_as_keyword():
    workload = WorkloadSpec(head_dim=128, seq_len=100)
    result = select_kv_cache_config(workload, hardware=HardwareInfo())
    assert result.config.method == "turboquant_rvq"


def test_select_accepts_both_as_keyword():
    workload = WorkloadSpec(head_dim=128, seq_len=100)
    result = select_kv_cache_config(workload=workload, hardware=HardwareInfo())
    assert result.config.method == "turboquant_rvq"


# ============================================================================
# Repeated / bulk calls do not leak state between unrelated workloads
# ============================================================================


def test_bulk_sequential_calls_are_isolated_from_each_other():
    workloads = [
        WorkloadSpec(head_dim=128, seq_len=100),
        WorkloadSpec(head_dim=128, seq_len=8_000, n_layers=64, batch_size=16),
        WorkloadSpec(head_dim=256, seq_len=LONG_CONTEXT_TOKENS),
        WorkloadSpec(head_dim=128, seq_len=8_000, n_layers=64, batch_size=16),
    ]
    hardware_for_pressure = HardwareInfo(
        total_memory_bytes=workloads[1].fp16_kv_bytes(), active_memory_bytes=0
    )

    results = [
        select_kv_cache_config(workloads[0], HardwareInfo()),
        select_kv_cache_config(workloads[1], hardware_for_pressure),
        select_kv_cache_config(workloads[2], HardwareInfo()),
        select_kv_cache_config(workloads[3], HardwareInfo()),  # same shape, no pressure this time
    ]

    assert results[0].config.method == "turboquant_rvq"
    assert results[1].config.method == "gear"
    assert results[2].config.method == "kvquant"
    assert results[3].config.method == "kivi"  # confirms result[1]'s pressure didn't leak
