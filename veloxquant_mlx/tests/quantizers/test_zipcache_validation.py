"""Argument validation, non-finite rejection and trim accounting for ZipCache.

Covers ``validate_zipcache_config`` (shared by ``KVCacheConfig`` and
``ZipCacheKVCache``), the standalone quantizer functions, prefill NaN/Inf rejection
(and that a rejected chunk changes nothing), and ``trim()`` keeping the byte counters
honest. Valid inputs must behave exactly as before; controls below pin that.
"""

from __future__ import annotations

import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.zipcache_cache import ZipCacheKVCache
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers.zipcache import (
    base_only_bytes,
    channel_dequant,
    channel_quant,
    saliency_mask,
    token_key_norms,
    validate_zipcache_config,
    zipcache_bytes,
    zipcache_compress,
    zipcache_quant_dequant,
)

NAN = float("nan")
INF = float("inf")


def _cfg(**kw) -> KVCacheConfig:
    return KVCacheConfig(method="zipcache", head_dim=64, **kw)


def _kv(S: int = 64, H: int = 4, D: int = 64, seed: int = 0, dtype=np.float16) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, H, S, D)).astype(dtype))


def _f32(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


# ---------------------------------------------------------------------------
# Config: every bad value is rejected, by name, with QuantizerConfigError
# ---------------------------------------------------------------------------
BAD_BITS = [0, -1, 9, 16, 2.5, True, "4", None, NAN, INF, [4]]
BAD_GROUP = [0, -4, 1.5, True, "32", None, NAN, INF]
BAD_FRACTION = [NAN, INF, -INF, True, "0.2", None, 2.0, -0.1, 1.0000001, [0.2]]
BAD_FLAG = ["false", "no", "True", 0, 1, None, 0.0, [True]]


@pytest.mark.parametrize("field", ["zipcache_hi_bits", "zipcache_lo_bits"])
@pytest.mark.parametrize("bad", BAD_BITS, ids=repr)
def test_bad_bits_rejected_at_config(field: str, bad) -> None:
    with pytest.raises(QuantizerConfigError, match=field):
        _cfg(**{field: bad})


@pytest.mark.parametrize("bad", BAD_GROUP, ids=repr)
def test_bad_group_size_rejected_at_config(bad) -> None:
    with pytest.raises(QuantizerConfigError, match="zipcache_group_size"):
        _cfg(zipcache_group_size=bad)


@pytest.mark.parametrize("bad", BAD_FRACTION, ids=repr)
def test_bad_hi_fraction_rejected_at_config(bad) -> None:
    with pytest.raises(QuantizerConfigError, match="zipcache_hi_fraction"):
        _cfg(zipcache_hi_fraction=bad)


@pytest.mark.parametrize("bad", BAD_FLAG, ids=repr)
def test_bad_quantize_values_rejected_at_config(bad) -> None:
    with pytest.raises(QuantizerConfigError, match="zipcache_quantize_values"):
        _cfg(zipcache_quantize_values=bad)


@pytest.mark.parametrize("hi,lo", [(2, 5), (3, 4), (1, 8)])
def test_lo_bits_above_hi_bits_rejected(hi: int, lo: int) -> None:
    with pytest.raises(QuantizerConfigError, match="zipcache_lo_bits"):
        _cfg(zipcache_hi_bits=hi, zipcache_lo_bits=lo)


def test_error_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        _cfg(zipcache_hi_bits=0)


def test_bits_nine_or_more_no_longer_corrupt_silently() -> None:
    """On master ``hi_bits=9`` was accepted and codes wrapped through uint8 (max err ~3.3 vs 0.014)."""
    with pytest.raises(QuantizerConfigError, match="zipcache_hi_bits"):
        ZipCacheKVCache(_cfg(zipcache_hi_bits=9))
    c = ZipCacheKVCache(_cfg(zipcache_hi_bits=8, zipcache_lo_bits=2, zipcache_hi_fraction=1.0))
    x = _kv(128, 2)
    k, _ = c.update_and_fetch(x, x)
    assert float(np.abs(_f32(k) - _f32(x)).max()) < 0.05


@pytest.mark.parametrize(
    "kw",
    [
        {"zipcache_hi_bits": 4.0},
        {"zipcache_group_size": 32.0},
        {"zipcache_hi_bits": np.int64(4), "zipcache_lo_bits": np.int32(2)},
        {"zipcache_hi_fraction": np.float32(0.25)},
        {"zipcache_hi_fraction": 1},
        {"zipcache_hi_fraction": 0},
        {"zipcache_quantize_values": np.bool_(False)},
        {"zipcache_hi_bits": 1, "zipcache_lo_bits": 1},
        {"zipcache_hi_bits": 8, "zipcache_lo_bits": 8},
        {"zipcache_hi_bits": 4, "zipcache_lo_bits": 4},
        {"zipcache_group_size": 1},
        {"zipcache_group_size": 4096},
        {"zipcache_hi_fraction": 0.0},
        {"zipcache_hi_fraction": 1.0},
    ],
    ids=repr,
)
def test_valid_values_accepted_and_work(kw) -> None:
    """Whole-number floats, numpy scalars and the legal extremes all construct and run."""
    c = ZipCacheKVCache(_cfg(**kw))
    x = _kv(40, 2)
    k, v = c.update_and_fetch(x, x)
    assert k.shape == x.shape and v.shape == x.shape
    assert np.isfinite(_f32(k)).all()


def test_integral_floats_behave_exactly_like_ints() -> None:
    x = _kv(96, 2)
    a = ZipCacheKVCache(_cfg(zipcache_hi_bits=4.0, zipcache_lo_bits=2.0, zipcache_group_size=16.0))
    b = ZipCacheKVCache(_cfg(zipcache_hi_bits=4, zipcache_lo_bits=2, zipcache_group_size=16))
    ka, va = a.update_and_fetch(x, x)
    kb, vb = b.update_and_fetch(x, x)
    assert np.array_equal(_f32(ka), _f32(kb)) and np.array_equal(_f32(va), _f32(vb))
    assert a.compressed_key_bytes == b.compressed_key_bytes


def test_validate_zipcache_config_defaults_and_types() -> None:
    cfg = validate_zipcache_config(_cfg())
    assert cfg == {
        "hi_bits": 4,
        "lo_bits": 2,
        "hi_fraction": 0.20,
        "group_size": 32,
        "quantize_values": True,
    }
    coerced = validate_zipcache_config(
        _cfg(zipcache_hi_bits=np.int64(6), zipcache_group_size=16.0, zipcache_hi_fraction=1)
    )
    assert type(coerced["hi_bits"]) is int and type(coerced["group_size"]) is int
    assert type(coerced["hi_fraction"]) is float and coerced["hi_fraction"] == 1.0


def test_validate_accepts_a_bare_namespace_with_defaults() -> None:
    """A look-alike config with no zipcache_* fields gets the documented defaults."""

    class Bare:
        pass

    assert validate_zipcache_config(Bare())["hi_bits"] == 4


def test_other_methods_ignore_zipcache_fields() -> None:
    """Every config carries zipcache_*; only method='zipcache' reads them."""
    KVCacheConfig(method="kivi", head_dim=64, zipcache_hi_bits=0, zipcache_group_size=-1)


def test_cache_constructor_revalidates_an_edited_config() -> None:
    cfg = _cfg()
    with pytest.warns(DeprecationWarning):
        cfg.zipcache_group_size = 0  # edited after __post_init__
    with pytest.raises(QuantizerConfigError, match="zipcache_group_size"):
        ZipCacheKVCache(cfg)
    cfg2 = _cfg()
    with pytest.warns(DeprecationWarning):
        cfg2.zipcache_hi_fraction = 2.0
    with pytest.raises(QuantizerConfigError, match="zipcache_hi_fraction"):
        ZipCacheKVCache(cfg2)


def test_no_warnings_on_valid_use() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = ZipCacheKVCache(_cfg())
        x = _kv(64, 2)
        c.update_and_fetch(x, x)
        c.update_and_fetch(_kv(1, 2, seed=1), _kv(1, 2, seed=2))


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------
def _mat(n: int = 64, d: int = 32, seed: int = 0) -> mx.array:
    return mx.array(np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32))


@pytest.mark.parametrize("bad", [0, -1, 9, 2.5, True, "4", None])
def test_channel_quant_rejects_bad_bits(bad) -> None:
    with pytest.raises(QuantizerConfigError, match="bits"):
        channel_quant(_mat(), bad, 32)


@pytest.mark.parametrize("bad", [0, -4, 1.5, True, "32", None])
def test_group_size_rejected_by_every_standalone_function(bad) -> None:
    x = _mat()
    codes, scales, zeros = channel_quant(x, 4, 32)
    st = zipcache_compress(x)
    for call in (
        lambda: channel_quant(x, 4, bad),
        lambda: channel_dequant(codes, scales, zeros, bad),
        lambda: zipcache_compress(x, 4, 2, 0.2, bad),
        lambda: base_only_bytes(64, 32, 4, bad),
        lambda: zipcache_quant_dequant(x, 4, 2, 0.2, bad),
    ):
        with pytest.raises(QuantizerConfigError, match="group_size"):
            call()
    if bad is not None:  # zipcache_bytes(state, None) means "use the state's group size"
        with pytest.raises(QuantizerConfigError, match="group_size"):
            zipcache_bytes(st, bad)


@pytest.mark.parametrize("bad", [NAN, INF, -0.1, 1.5, True, "0.2", None])
def test_hi_fraction_rejected_by_standalone_functions(bad) -> None:
    with pytest.raises(QuantizerConfigError, match="hi_fraction"):
        zipcache_compress(_mat(), 4, 2, bad, 32)
    with pytest.raises(QuantizerConfigError, match="hi_fraction"):
        saliency_mask(mx.ones((8,)), bad)


def test_standalone_bits_rejected() -> None:
    x = _mat()
    for hi, lo in [(0, 2), (4, 0), (9, 2), (4, 9), (2.5, 2), (True, 1), (2, 4)]:
        with pytest.raises(QuantizerConfigError):
            zipcache_compress(x, hi, lo, 0.2, 32)
    with pytest.raises(QuantizerConfigError, match="bits"):
        base_only_bytes(64, 32, 0, 32)
    with pytest.raises(QuantizerConfigError, match="bits"):
        base_only_bytes(64, 32, 9, 32)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_standalone_rejects_non_finite(bad: float) -> None:
    x = np.array(_mat())
    x[3, 5] = bad
    x = mx.array(x)
    with pytest.raises(QuantizerConfigError, match="finite"):
        zipcache_compress(x)
    with pytest.raises(QuantizerConfigError, match="finite"):
        channel_quant(x, 4, 32)
    with pytest.raises(QuantizerConfigError, match="finite"):
        saliency_mask(token_key_norms(x), 0.2)


def test_standalone_rejects_non_2d_input() -> None:
    for call in (
        lambda: zipcache_compress(mx.zeros((64,))),
        lambda: zipcache_compress(mx.zeros((1, 64, 32))),
        lambda: channel_quant(mx.zeros((64,)), 4, 32),
        lambda: token_key_norms(mx.zeros((64,))),
        lambda: saliency_mask(mx.zeros((2, 8)), 0.2),
    ):
        with pytest.raises(QuantizerConfigError, match="2-D|1-D"):
            call()


def test_standalone_controls_still_accept_valid_input() -> None:
    """Empty buckets, S=0 and the legal extremes keep working."""
    codes, scales, zeros = channel_quant(mx.zeros((0, 8)), 4, 32)
    assert codes.shape == (0, 8)
    assert zipcache_compress(mx.zeros((0, 8))).seq_len == 0
    for hi, lo, frac in [(8, 8, 1.0), (1, 1, 0.0), (4, 4, 0.5), (4.0, 2.0, 0.2)]:
        out = zipcache_quant_dequant(_mat(40, 16), hi, lo, frac, 8)
        assert out.shape == (40, 16) and np.isfinite(_f32(out)).all()


# ---------------------------------------------------------------------------
# Cache: a non-finite prefill chunk is reported by the next update and taken back out
# ---------------------------------------------------------------------------
def _poison(x: mx.array, bad: float, head: int = 0, tok: int = 5) -> mx.array:
    a = np.array(x.astype(mx.float32))
    a[0, head, tok, :] = bad
    return mx.array(a.astype(np.float16))


def _step(c: ZipCacheKVCache, seed: int = 7):
    """One decode step: the call that resolves a pending prefill check."""
    one = _kv(1, 4, seed=seed)
    return c.update_and_fetch(one, one)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_prefill_keys_non_finite_reported_by_the_next_update(bad: float) -> None:
    c = ZipCacheKVCache(_cfg())
    x = _kv()
    c.update_and_fetch(_poison(x, bad), x)  # the chunk itself returns: no host sync mid-forward
    with pytest.raises(QuantizerConfigError, match="NaN or Inf"):
        _step(c)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_prefill_values_non_finite_reported_by_the_next_update(bad: float) -> None:
    c = ZipCacheKVCache(_cfg())
    x = _kv()
    c.update_and_fetch(x, _poison(x, bad))
    with pytest.raises(QuantizerConfigError, match="NaN or Inf"):
        _step(c)


def test_reported_by_the_next_prefill_chunk_too() -> None:
    c = ZipCacheKVCache(_cfg())
    x = _kv()
    c.update_and_fetch(_poison(x, NAN), x)
    with pytest.raises(QuantizerConfigError, match="NaN or Inf"):
        c.update_and_fetch(x, x)


def test_error_names_the_removed_chunk() -> None:
    c = ZipCacheKVCache(_cfg())
    x = _kv(64)
    c.update_and_fetch(_poison(x, NAN), x)
    with pytest.raises(QuantizerConfigError, match="64 tokens were removed"):
        _step(c)


def test_a_rejected_first_chunk_leaves_the_cache_as_it_was() -> None:
    c = ZipCacheKVCache(_cfg())
    x = _kv()
    c.update_and_fetch(_poison(x, NAN), x)
    with pytest.raises(QuantizerConfigError):
        _step(c)
    assert c.offset == 0
    assert c._ledger == []
    assert c._pending is None
    assert _counters(c) == (0, 0, 0, 0, 0, 0)


def test_a_rejected_chunk_after_a_good_one_changes_nothing_and_the_cache_recovers() -> None:
    """The chunk is taken back out; the next clean update equals a cache that never saw it."""
    good, bad_chunk, nxt = _kv(64, seed=0), _kv(64, seed=1), _kv(64, seed=2)
    c = ZipCacheKVCache(_cfg())
    c.update_and_fetch(good, good)
    before = (c.offset, *_counters(c))
    c.update_and_fetch(bad_chunk, _poison(bad_chunk, INF))
    with pytest.raises(QuantizerConfigError):
        c.update_and_fetch(nxt, nxt)  # reports it, and does not store `nxt`
    assert (c.offset, *_counters(c)) == before

    ref = ZipCacheKVCache(_cfg())
    ref.update_and_fetch(good, good)
    k1, v1 = c.update_and_fetch(nxt, nxt)
    k2, v2 = ref.update_and_fetch(nxt, nxt)
    assert np.array_equal(_f32(k1), _f32(k2)) and np.array_equal(_f32(v1), _f32(v2))
    assert _counters(c) == _counters(ref)
    assert c.offset == ref.offset == 128


def test_each_chunk_is_checked_once_and_clean_chunks_never_raise() -> None:
    c = ZipCacheKVCache(_cfg())
    for seed in range(4):
        c.update_and_fetch(_kv(32, seed=seed), _kv(32, seed=seed + 10))
    for seed in range(3):
        _step(c, seed)
    assert c._pending is None  # a decode step resolved the last chunk's flag


def test_trim_clears_the_pending_check() -> None:
    """Editing the cache after a chunk means the flag no longer describes what is stored."""
    c = ZipCacheKVCache(_cfg())
    x = _kv()
    c.update_and_fetch(_poison(x, NAN), x)
    c.trim(64)
    _step(c)  # must not raise


def test_values_not_quantized_are_not_checked() -> None:
    """With quantize_values=False values pass through untouched (as in an fp16 cache)."""
    c = ZipCacheKVCache(_cfg(zipcache_quantize_values=False))
    x = _kv()
    _, v = c.update_and_fetch(x, _poison(x, NAN))
    _step(c)  # no raise
    assert np.isnan(_f32(v)).any()


def test_decode_step_is_not_checked_and_damages_only_its_own_token() -> None:
    """No check per decode token, so a bad decode row is not rejected; it stays local."""
    c = ZipCacheKVCache(_cfg())
    x = _kv(64)
    c.update_and_fetch(x, x)
    bad = mx.array(np.full((1, 4, 1, 64), np.nan, dtype=np.float16))
    k, _ = c.update_and_fetch(bad, bad)  # must not raise
    _step(c)  # nor does the step after it
    kf = _f32(k)
    assert np.isnan(kf[:, :, -1, :]).all()
    assert np.isfinite(kf[:, :, :-1, :]).all()
    assert c._pending is None


def test_non_4d_input_has_a_clear_error() -> None:
    c = ZipCacheKVCache(_cfg())
    with pytest.raises(ValueError, match=r"\[B, H, S, D\]"):
        c.update_and_fetch(mx.zeros((4, 64, 64)), mx.zeros((4, 64, 64)))


def test_chunk_of_one_token_is_the_decode_path() -> None:
    """S == 1 is not checked, and is still stored."""
    c = ZipCacheKVCache(_cfg())
    one = mx.array(np.full((1, 2, 1, 64), np.inf, dtype=np.float16))
    c.update_and_fetch(one, one)
    _step_small = _kv(1, 2, seed=3)
    c.update_and_fetch(_step_small, _step_small)  # no raise
    assert c.offset == 2


# ---------------------------------------------------------------------------
# Controls: valid input is unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hf", [0.0, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("gs", [1, 8, 32])
def test_cache_prefill_matches_standalone_quant_dequant(hf: float, gs: int) -> None:
    x = _kv(65, 1)
    c = ZipCacheKVCache(_cfg(zipcache_hi_fraction=hf, zipcache_group_size=gs))
    k, _ = c.update_and_fetch(x, x)
    ref = zipcache_quant_dequant(x[0, 0], 4, 2, hf, gs)
    assert np.array_equal(_f32(k[0, 0]), _f32(ref))


def test_bf16_prefill_passes_the_finite_check() -> None:
    x = mx.array(np.random.default_rng(0).standard_normal((1, 2, 64, 64)).astype(np.float32))
    x = x.astype(mx.bfloat16)
    c = ZipCacheKVCache(_cfg())
    k, _ = c.update_and_fetch(x, x)
    assert k.dtype == mx.bfloat16


def test_fp16_extreme_finite_values_are_not_mistaken_for_non_finite() -> None:
    """65504 squared and summed over D stays finite in the float32 norm."""
    x = mx.full((1, 2, 64, 128), 65504.0, dtype=mx.float16)
    c = ZipCacheKVCache(_cfg())
    k, _ = c.update_and_fetch(x, x)
    assert np.isfinite(_f32(k)).all()


# ---------------------------------------------------------------------------
# trim() keeps the byte counters honest
# ---------------------------------------------------------------------------
def _counters(c: ZipCacheKVCache) -> tuple[int, ...]:
    return (
        c.compressed_key_bytes,
        c.baseline_key_bytes,
        c.fp16_key_bytes,
        c.compressed_value_bytes,
        c.baseline_value_bytes,
        c.fp16_value_bytes,
    )


def _prefill_then_decode(n_dec: int, S: int = 64) -> ZipCacheKVCache:
    c = ZipCacheKVCache(_cfg())
    x = _kv(S, 2)
    c.update_and_fetch(x, x)
    for i in range(n_dec):
        t = _kv(1, 2, seed=100 + i)
        c.update_and_fetch(t, t)
    return c


def test_trim_decode_tokens_restores_the_prefill_counters_exactly() -> None:
    base = _prefill_then_decode(0)
    c = _prefill_then_decode(7)
    assert _counters(c) != _counters(base)
    assert c.trim(7) == 7
    assert c.offset == 64
    assert _counters(c) == _counters(base)


def test_trim_part_of_a_decode_run_is_exact() -> None:
    a = _prefill_then_decode(10)
    b = _prefill_then_decode(6)
    a.trim(4)
    assert _counters(a) == _counters(b)


def test_trim_then_decode_again_matches_never_having_decoded_the_trimmed_tokens() -> None:
    a = _prefill_then_decode(5)
    a.trim(3)
    t = _kv(1, 2, seed=900)
    a.update_and_fetch(t, t)
    b = _prefill_then_decode(2)
    b.update_and_fetch(t, t)
    assert _counters(a) == _counters(b)


def test_trim_everything_zeroes_the_counters() -> None:
    c = _prefill_then_decode(3)
    c.trim(10_000)
    assert c.offset == 0
    assert _counters(c) == (0, 0, 0, 0, 0, 0)
    assert c.compression_ratio == 1.0


def test_trim_zero_changes_nothing() -> None:
    c = _prefill_then_decode(3)
    before = _counters(c)
    assert c.trim(0) == 0
    assert _counters(c) == before


def test_trim_into_a_multi_token_chunk_is_prorated_by_tokens_kept() -> None:
    c = ZipCacheKVCache(_cfg())
    x = _kv(128, 2)
    c.update_and_fetch(x, x)
    full = _counters(c)
    c.trim(64)
    half = _counters(c)
    assert c.offset == 64
    for f, h in zip(full, half, strict=True):
        assert h == round(f * 64 / 128)
    c.trim(64)
    assert _counters(c) == (0, 0, 0, 0, 0, 0)


def test_trim_across_chunk_and_decode_boundaries() -> None:
    c = ZipCacheKVCache(_cfg())
    p1, p2 = _kv(64, 2, seed=1), _kv(32, 2, seed=2)
    c.update_and_fetch(p1, p1)
    after_p1 = _counters(c)
    c.update_and_fetch(p2, p2)
    for i in range(4):
        t = _kv(1, 2, seed=50 + i)
        c.update_and_fetch(t, t)
    c.trim(4 + 32)  # the 4 decode tokens and the whole second chunk
    assert _counters(c) == after_p1


def test_trim_never_makes_a_counter_negative() -> None:
    c = _prefill_then_decode(5)
    for n in (1, 2, 7, 3, 100):
        c.trim(n)
        assert min(_counters(c)) >= 0


def test_counters_unchanged_for_valid_use_without_trim() -> None:
    """Control: the ledger must not perturb the totals a plain prefill reports."""
    c = ZipCacheKVCache(_cfg())
    x = _kv(256, 4)
    c.update_and_fetch(x, x)
    S, D, BH = 256, 64, 4
    n_hi = math.ceil(S * 0.2)
    per_slab_key = (
        math.ceil(n_hi * D * 4 / 8)
        + math.ceil((S - n_hi) * D * 2 / 8)
        + math.ceil(n_hi / 32) * D * 4
        + math.ceil((S - n_hi) / 32) * D * 4
        + S
    )
    assert c.compressed_key_bytes == per_slab_key * BH
    assert c.fp16_key_bytes == BH * S * D * 2
