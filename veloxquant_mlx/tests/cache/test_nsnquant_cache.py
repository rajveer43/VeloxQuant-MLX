"""Tests for NSNQuantKVCache — calibration-free universal-codebook VQ wrapper.

Covers mlx_lm protocol shape/dtype preservation, chunk-flush residual buffer
semantics (prefill/decode path-independence, per-chunk statistics), byte
accounting against the closed form, both K and V quantization, build-time
validation, the max_ctx guard, and for_model wiring incl. the fallback path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.nsnquant_cache import NSNQuantKVCache

# Fast codebook for tests: seed the module cache once so every wrapper below
# reuses it instead of building the default-size codebook.
from veloxquant_mlx.quantizers import nsnquant as _nsn_mod

_TEST_SEED = 1234
for _kind in ("signed", "magnitude"):
    cb = _nsn_mod.build_universal_codebook(seed=_TEST_SEED, n_samples=131_072, iters=15, kind=_kind)
    _nsn_mod._CODEBOOK_CACHE[(256, 8, _TEST_SEED, _kind)] = cb


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(**cfg):
    base = dict(
        method="nsnquant", head_dim=128, nsn_bits=2, nsn_residual_length=16, nsn_seed=_TEST_SEED
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _mean_cosine(a, b) -> float:
    an = np.array(a, dtype=np.float64).reshape(-1, a.shape[-1])
    bn = np.array(b, dtype=np.float64).reshape(-1, b.shape[-1])
    num = np.sum(an * bn, axis=1)
    den = np.linalg.norm(an, axis=1) * np.linalg.norm(bn, axis=1) + 1e-9
    return float(np.mean(num / den))


# ------------------------------------------------------------------
# Factory / protocol basics
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), NSNQuantKVCache)


def test_shape_dtype_preserved() -> None:
    cache = _make()
    k, v = _kv(1, 4, 64, 128)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 4, 64, 128)
    assert vo.shape == (1, 4, 64, 128)
    assert ko.dtype == mx.float16


def test_no_bits_leak() -> None:
    cache = _make()
    assert not hasattr(cache, "bits")
    assert hasattr(cache, "assigned_avg_bits")


# ------------------------------------------------------------------
# Reconstruction quality
# ------------------------------------------------------------------


@pytest.mark.parametrize("bits,floor", [(2, 0.90), (1, 0.75)])
def test_prefill_reconstruction_cosine_floor(bits: int, floor: float) -> None:
    cache = _make(nsn_bits=bits)
    k, v = _kv(1, 2, 128, 128, seed=1)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert _mean_cosine(ko, k) > floor
    assert _mean_cosine(vo, v) > floor  # values quantized too, not just keys


def test_residual_window_kept_fp16() -> None:
    """Tokens short of a full chunk pass through untouched."""
    cache = _make(nsn_residual_length=64)
    k, v = _kv(1, 2, 32, 128, seed=2)  # S=32 < chunk 64 — no flush yet
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))
    assert cache.compressed_key_bytes == 0
    assert cache.quantized_tokens == 0


# ------------------------------------------------------------------
# Chunk-flush semantics
# ------------------------------------------------------------------


@pytest.mark.parametrize("S", [47, 48, 49])
def test_chunking_arithmetic_edges(S: int) -> None:
    """T = k*residual_length ± 1: frontier advances in whole chunks only."""
    r = 16
    cache = _make(nsn_residual_length=r)
    k, v = _kv(1, 2, S, 128, seed=3)
    cache.update_and_fetch(k, v)
    assert cache.quantized_tokens == (S // r) * r
    n_res = S - cache.quantized_tokens
    assert cache.residual_fp16_bytes == n_res * 128 * 2 * 2 * 1 * 2


def test_decode_accumulation_across_flushes() -> None:
    """Token-by-token decode crosses >= 2 flush boundaries; fetch length
    always equals total tokens pushed, and aged tokens DO get quantized."""
    r = 8
    cache = _make(nsn_residual_length=r)
    total = 0
    for step in range(3 * r):
        k, v = _kv(1, 2, 1, 128, seed=100 + step)
        ko, vo = cache.update_and_fetch(k, v)
        total += 1
        assert ko.shape[2] == total
    assert cache.quantized_tokens == 3 * r  # decode tokens age into chunks


def test_prefill_decode_path_independence() -> None:
    """Same tokens pushed as one prefill block vs token-by-token decode yield
    an identical quantized state (chunk boundaries are identical by
    construction)."""
    r = 8
    S = 3 * r + 5
    k, v = _kv(1, 2, S, 128, seed=4)

    prefill = _make(nsn_residual_length=r)
    ko_a, vo_a = prefill.update_and_fetch(k, v)

    decode = _make(nsn_residual_length=r)
    for t in range(S):
        ko_b, vo_b = decode.update_and_fetch(k[:, :, t : t + 1, :], v[:, :, t : t + 1, :])
    mx.eval(ko_a, vo_a, ko_b, vo_b)
    assert np.allclose(np.array(ko_a), np.array(ko_b), atol=1e-3)
    assert np.allclose(np.array(vo_a), np.array(vo_b), atol=1e-3)
    assert prefill.quantized_tokens == decode.quantized_tokens


def test_per_chunk_independence() -> None:
    """Chunk i's stored reconstruction never changes after later pushes."""
    r = 16
    cache = _make(nsn_residual_length=r)
    k, v = _kv(1, 2, r, 128, seed=5)
    ko1, _ = cache.update_and_fetch(k, v)
    first_chunk = np.array(ko1[:, :, :r, :]).copy()
    bytes_after_first = cache.compressed_key_bytes
    for step in range(2 * r):
        k2, v2 = _kv(1, 2, 1, 128, seed=200 + step)
        ko, _ = cache.update_and_fetch(k2, v2)
    assert np.array_equal(np.array(ko[:, :, :r, :]), first_chunk)
    assert cache.compressed_key_bytes > bytes_after_first


def test_determinism() -> None:
    k, v = _kv(1, 2, 80, 128, seed=6)
    a = _make()
    b = _make()
    ka, va = a.update_and_fetch(k, v)
    kb, vb = b.update_and_fetch(k, v)
    mx.eval(ka, va, kb, vb)
    assert np.array_equal(np.array(ka), np.array(kb))
    assert np.array_equal(np.array(va), np.array(vb))


# ------------------------------------------------------------------
# Byte accounting
# ------------------------------------------------------------------


def test_byte_accounting_closed_form() -> None:
    B, H, D, r = 1, 2, 128, 16
    n_chunks = 4
    cache = _make(nsn_residual_length=r)
    k, v = _kv(B, H, n_chunks * r, D, seed=7)
    cache.update_and_fetch(k, v)
    n_sub = D // 8
    payload = r * n_sub * 2  # 2-bit: sign mask + index per subvector
    metadata = r * 2 * 2 + D * 2  # fp16 s1+s2 per token, fp16 o per chunk
    expected = (payload + metadata) * B * H * n_chunks
    assert cache.compressed_key_bytes == expected
    assert cache.compressed_value_bytes == expected
    assert cache.fp16_key_bytes == B * H * n_chunks * r * D * 2


def test_2bit_payload_double_of_1bit() -> None:
    r, S = 16, 64
    c2 = _make(nsn_bits=2, nsn_residual_length=r)
    c1 = _make(nsn_bits=1, nsn_residual_length=r)
    k, v = _kv(1, 2, S, 128, seed=8)
    c2.update_and_fetch(k, v)
    c1.update_and_fetch(k, v)
    # Payload doubles; the shared fp16 metadata term keeps the total below 2x.
    assert c1.compressed_key_bytes < c2.compressed_key_bytes < 2 * c1.compressed_key_bytes
    assert c2.assigned_avg_bits < 3.5  # ~2 bits payload + fp16 metadata
    assert c1.assigned_avg_bits < 2.5


def test_compression_ratio_beats_fp16_at_long_context() -> None:
    r = 16
    cache = _make(nsn_residual_length=r)
    k, v = _kv(1, 2, 512, 128, seed=9)
    cache.update_and_fetch(k, v)
    total = cache.compressed_key_bytes + cache.compressed_value_bytes + cache.residual_fp16_bytes
    fp16_total = cache.fp16_key_bytes + cache.fp16_value_bytes
    assert total < fp16_total / 4  # comfortably past 4x at T >> r


# ------------------------------------------------------------------
# Guards and validation
# ------------------------------------------------------------------


def test_max_ctx_guard_raises() -> None:
    cache = _make(nsn_max_ctx=32)
    k, v = _kv(1, 2, 32, 128, seed=10)
    cache.update_and_fetch(k, v)
    k1, v1 = _kv(1, 2, 1, 128, seed=11)
    with pytest.raises(ValueError, match="nsn_max_ctx"):
        cache.update_and_fetch(k1, v1)


def test_build_time_validation() -> None:
    with pytest.raises(ValueError, match="nsn_bits"):
        _make(nsn_bits=3)
    with pytest.raises(ValueError, match="divisible"):
        _make(head_dim=100)  # 100 % 8 != 0
    with pytest.raises(ValueError, match="hadamard"):
        _make(head_dim=72)  # divisible by 8 but not Hadamard-compatible
    with pytest.raises(ValueError, match="nsn_residual_length"):
        _make(nsn_residual_length=1)


# ------------------------------------------------------------------
# for_model wiring
# ------------------------------------------------------------------


class _ToyAttn:
    def __init__(self, head_dim):
        self.head_dim = head_dim


class _ToyLayer:
    def __init__(self, head_dim=128):
        self.self_attn = _ToyAttn(head_dim)


class _ToyNorm:
    """Layer without attention — must get the fallback cache."""

    pass


class _ToyInner:
    def __init__(self):
        self.layers = [_ToyLayer(), _ToyLayer(), _ToyNorm(), _ToyLayer()]


class _ToyModel:
    def __init__(self):
        self.model = _ToyInner()
        self.args = None


def test_for_model_wiring_and_fallback() -> None:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    caches = KVCacheBuilder.for_model(
        _ToyModel(),
        KVCacheConfig(method="nsnquant", head_dim=128, nsn_seed=_TEST_SEED),
    )
    assert len(caches) == 4
    assert isinstance(caches[0], NSNQuantKVCache)
    assert isinstance(caches[1], NSNQuantKVCache)
    assert type(caches[2]) is _FallbackCache  # non-attention layer
    assert isinstance(caches[3], NSNQuantKVCache)

    # Fallback path unaffected: plain passthrough for the non-attention slot.
    k, v = _kv(1, 2, 8, 128, seed=12)
    ko, vo = caches[2].update_and_fetch(k, v)
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))
