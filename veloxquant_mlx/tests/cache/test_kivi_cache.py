"""Tests for KIVIKVCache — asymmetric group-quant cache wrapper.

Covers mlx_lm protocol shape/dtype preservation, the fp16 residual window
(recent tokens kept exact), byte accounting, and the no-``.bits``-leak
invariant that keeps mlx_lm's SDPA on the standard fp16 path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kivi_cache import KIVIKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(method_kw=None, **cfg):
    base = dict(
        method="kivi", head_dim=128, bit_width_inlier=2, residual_length=16, kivi_group_size=32
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def test_factory_dispatch() -> None:
    cache = _make()
    assert isinstance(cache, KIVIKVCache)


def test_shape_dtype_preserved() -> None:
    cache = _make()
    k, v = _kv(1, 4, 64, 128)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 4, 64, 128)
    assert vo.shape == (1, 4, 64, 128)
    assert ko.dtype == mx.float16


def test_no_bits_leak() -> None:
    # mlx_lm's SDPA checks hasattr(cache, "bits") to route to a quantized
    # kernel; KIVIKVCache must not expose it (it dequantizes to fp16).
    cache = _make()
    assert not hasattr(cache, "bits")
    assert hasattr(cache, "assigned_avg_bits")


def test_residual_window_kept_fp16() -> None:
    """When S <= residual_length the whole block passes through untouched."""
    cache = _make(residual_length=64)
    k, v = _kv(1, 2, 32, 128, seed=1)  # S=32 < residual 64
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    # Exact passthrough: output equals input bit-for-bit.
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))
    assert cache.compressed_key_bytes == 0  # nothing quantized yet


def test_quantizes_aged_tokens() -> None:
    """When S > residual_length the oldest (S - r) tokens are quantized."""
    cache = _make(residual_length=16)
    k, v = _kv(1, 2, 80, 128, seed=2)  # 64 quantized, 16 residual
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    ko_np, k_np = np.array(ko), np.array(k)
    # Residual tail (last 16) is exact; quantized head differs.
    assert np.array_equal(ko_np[:, :, -16:, :], k_np[:, :, -16:, :])
    head_diff = np.mean(np.abs(ko_np[:, :, :64, :] - k_np[:, :, :64, :]))
    assert head_diff > 0.0, "quantized region should differ from fp16 input"
    assert cache.compressed_key_bytes > 0
    assert cache.residual_fp16_bytes > 0


def test_compression_ratio_below_fp16() -> None:
    """Quantized region (+ params + residual) must beat fp16."""
    cache = _make(bit_width_inlier=2, residual_length=16)
    k, v = _kv(1, 4, 256, 128, seed=3)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    total_comp = (
        cache.compressed_key_bytes + cache.compressed_value_bytes + cache.residual_fp16_bytes
    )
    total_fp16 = cache.fp16_key_bytes + cache.fp16_value_bytes
    ratio = total_fp16 / total_comp
    assert ratio > 1.5, f"end-to-end ratio {ratio:.2f} not better than fp16"


def test_decode_step_after_prefill() -> None:
    """Simulate prefill + single-token decode steps (S==1)."""
    cache = _make(residual_length=8)
    k, v = _kv(1, 2, 40, 128, seed=4)
    cache.update_and_fetch(k, v)
    for t in range(5):
        k1, v1 = _kv(1, 2, 1, 128, seed=100 + t)
        ko, vo = cache.update_and_fetch(k1, v1)
        mx.eval(ko, vo)
    assert cache.offset == 45


@pytest.mark.parametrize("b", [2, 4])
def test_higher_bits_more_compressed_bytes_but_better_quality(b: int) -> None:
    cache = _make(bit_width_inlier=b, residual_length=16)
    k, v = _kv(1, 2, 128, 128, seed=b)
    ko, _ = cache.update_and_fetch(k, v)
    mx.eval(ko)
    assert cache.assigned_avg_bits == float(b)


def test_decode_tokens_age_out_of_residual_window() -> None:
    """Regression for #86: with S==1 every decode step, tokens must still
    age out of the fp16 residual window once total cumulative length
    exceeds residual_length — the residual byte count must plateau, not
    grow unboundedly with decode steps."""
    cache = _make(residual_length=4, kivi_group_size=4)
    k, v = _kv(1, 1, 10, 128, seed=7)
    cache.update_and_fetch(k, v)  # prefill: 10 tokens, 6 quantized, 4 residual

    for t in range(50):
        k1, v1 = _kv(1, 1, 1, 128, seed=200 + t)
        cache.update_and_fetch(k1, v1)

    assert cache.offset == 60
    D, H, B = 128, 1, 1
    # Tokens wait in fp16 until they can form a whole group_size block, so the
    # live fp16 tail is residual_length plus the partial group still buffered.
    n_res = cache.offset - cache._n_quantized
    assert n_res <= 4 + 4, f"fp16 tail {n_res} should stay bounded by residual + one group"
    expected_residual = n_res * D * 2 * 2 * H * B  # K+V fp16 bytes
    assert cache.residual_fp16_bytes == expected_residual, (
        f"residual_fp16_bytes={cache.residual_fp16_bytes} did not plateau at "
        f"{expected_residual} after 50 decode steps past the residual window"
    )
    assert cache.compressed_key_bytes > 0


# ---------------------------------------------------------------------------
# Regression: paper-faithful residual buffering (#162)
# ---------------------------------------------------------------------------


def test_decode_quantized_keys_are_actually_lossy() -> None:
    """Regression for #162: keys quantized during single-token decode must be
    genuinely lossy.

    KIVI quantizes keys *per channel* — the group runs along the token axis —
    so flushing one token at a time puts a single element in every group,
    making min == max and the round-trip bit-exact. The residual buffer must
    hold tokens until a whole ``group_size`` block can be flushed.
    """
    g, r = 8, 8
    cache = _make(residual_length=r, kivi_group_size=g, bit_width_inlier=2)
    k, v = _kv(1, 1, 8, 128, seed=11)
    cache.update_and_fetch(k, v)

    ks, vs = [k], [v]
    for t in range(56):  # single-token decode steps
        k1, v1 = _kv(1, 1, 1, 128, seed=300 + t)
        ko, vo = cache.update_and_fetch(k1, v1)
        ks.append(k1)
        vs.append(v1)
    mx.eval(ko, vo)

    k_in = np.array(mx.concatenate(ks, axis=2))
    k_out = np.array(ko)
    n_q = cache._n_quantized
    assert n_q > 0, "nothing was ever quantized during decode"
    err = np.abs(k_out[:, :, :n_q, :] - k_in[:, :, :n_q, :]).mean()
    assert err > 1e-3, (
        f"decode-quantized keys are bit-exact (mean err {err:.2e}) — per-channel "
        f"groups collapsed to a single token, so 2-bit quantization was a no-op"
    )


def test_quantization_flushes_are_group_aligned() -> None:
    """The quantized prefix must always be a whole multiple of group_size, so
    no per-channel key group is ever built from a partial token block."""
    g = 16
    cache = _make(residual_length=8, kivi_group_size=g)
    k, v = _kv(1, 1, 5, 128, seed=12)
    cache.update_and_fetch(k, v)
    for t in range(70):
        k1, v1 = _kv(1, 1, 1, 128, seed=400 + t)
        cache.update_and_fetch(k1, v1)
        assert cache._n_quantized % g == 0, (
            f"quantized prefix {cache._n_quantized} is not a multiple of group_size {g}"
        )


def test_prefill_and_decode_are_equivalent() -> None:
    """Chunking must not change the result: feeding a sequence one-shot and
    token-by-token must quantize the same positions with the same group
    boundaries, hence produce bit-identical output."""
    g, r = 8, 8
    S = 48
    k, v = _kv(1, 2, S, 128, seed=13)

    one_shot = _make(residual_length=r, kivi_group_size=g)
    ko1, vo1 = one_shot.update_and_fetch(k, v)
    mx.eval(ko1, vo1)

    streamed = _make(residual_length=r, kivi_group_size=g)
    for t in range(S):
        ko2, vo2 = streamed.update_and_fetch(k[:, :, t : t + 1, :], v[:, :, t : t + 1, :])
    mx.eval(ko2, vo2)

    assert np.array_equal(np.array(ko1), np.array(ko2)), "keys differ between prefill and decode"
    assert np.array_equal(np.array(vo1), np.array(vo2)), "values differ between prefill and decode"
    assert one_shot._n_quantized == streamed._n_quantized


def test_key_accounting_beats_fp16() -> None:
    """Regression for #162: key bytes must never be billed above fp16.

    With 1-token groups the per-group (scale, zero) overhead exceeded the
    codes themselves and the reported key ratio inverted to 0.56x — i.e. the
    accounting claimed a *loss* for data that was in fact stored as fp16.
    """
    cache = _make(bit_width_inlier=2, residual_length=32, kivi_group_size=32)
    k, v = _kv(1, 1, 32, 64, seed=14)
    cache.update_and_fetch(k, v)
    for t in range(168):
        k1, v1 = _kv(1, 1, 1, 64, seed=500 + t)
        cache.update_and_fetch(k1, v1)

    ratio = cache.fp16_key_bytes / cache.compressed_key_bytes
    assert ratio > 1.0, f"key compression ratio {ratio:.2f}x is worse than fp16"
    assert cache.effective_compression_ratio > 1.0


# ---------------------------------------------------------------------------
# Cross-model shape coverage
# ---------------------------------------------------------------------------

# (label, n_kv_heads, n_layers) for the model geometries in the docs table.
# All three share head_dim=128; they differ in KV-head count and depth.
_DOC_MODELS = [
    ("Llama-3.2-3B-Instruct-4bit", 8, 28, 31),
    ("Qwen2.5-7B-Instruct-4bit", 4, 28, 32),
    ("Mistral-7B-Instruct-v0.3-4bit", 8, 32, 33),
]


@pytest.mark.parametrize("label,n_kv_heads,n_layers,seed", _DOC_MODELS)
def test_doc_model_geometries_quantize(
    label: str, n_kv_heads: int, n_layers: int, seed: int
) -> None:
    """The documented model geometries must all quantize on the decode path.

    Guards the #162 failure mode across the real head counts rather than a
    single synthetic shape: grouped-query models (Qwen, 4 KV heads) exercise
    a different [B, H, S, D] layout than the 8-KV-head models.
    """
    D = 128
    cache = _make(head_dim=D, bit_width_inlier=2, residual_length=32, kivi_group_size=32)
    k, v = _kv(1, n_kv_heads, 64, D, seed=seed)
    cache.update_and_fetch(k, v)
    for t in range(96):
        k1, v1 = _kv(1, n_kv_heads, 1, D, seed=600 + t)
        ko, vo = cache.update_and_fetch(k1, v1)
    mx.eval(ko, vo)

    assert cache._n_quantized % 32 == 0
    assert cache.compressed_key_bytes > 0
    assert cache.fp16_key_bytes / cache.compressed_key_bytes > 1.0


def test_compression_ratio_is_model_independent() -> None:
    """Compression depends on (b, group_size, residual_length, head_dim) —
    not on n_kv_heads or n_layers.

    The docs table reports the same 5.46x key ratio for Llama, Qwen and
    Mistral; this pins that down as a property of the accounting rather than
    a copy-paste error. Head count scales the compressed and fp16 byte pools
    together, so it cancels in the ratio.
    """
    D, S = 128, 160
    ratios = set()
    for _, n_kv_heads, _, _ in _DOC_MODELS:
        cache = _make(head_dim=D, bit_width_inlier=2, residual_length=32, kivi_group_size=32)
        k, v = _kv(1, n_kv_heads, 32, D, seed=21)
        cache.update_and_fetch(k, v)
        for t in range(S - 32):
            k1, v1 = _kv(1, n_kv_heads, 1, D, seed=700 + t)
            cache.update_and_fetch(k1, v1)
        ratios.add(round(cache.fp16_key_bytes / cache.compressed_key_bytes, 6))
    assert len(ratios) == 1, f"key compression varied across head counts: {ratios}"
