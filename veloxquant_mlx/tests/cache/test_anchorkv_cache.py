"""Tests for AnchorKVKVCache — anchor-residual compression, no eviction.

AnchorKV-adapted (arXiv:2608.02901v1, no verified peer-reviewed venue as of
2026-08-20 — a one-time, user-directed exception; see
paper/research/surveys/NEW_METHOD_SURVEY_V22.md) compresses the KV cache
ONCE at the end of prefill: each head selects anchor positions, projects
every other token onto its nearest anchor, and spends a byte budget on
quantized residuals for the highest-utility tokens. UNLIKE every eviction
method in this repo (H2O, SnapKV, PyramidKV, NestedKV, ...), no token is
EVER dropped — test_prefill_never_drops_tokens and
test_decode_tokens_always_appended prove this directly, the core structural
difference from the rest of the eviction family. Tests also cover: factory
dispatch, interface attributes, shape/dtype, byte accounting, determinism,
and for_model config propagation. All data is synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.anchorkv_cache import AnchorKVKVCache
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory


def _make(**cfg):
    base = dict(
        method="anchorkv",
        head_dim=32,
        anchorkv_theta=0.3,
        anchorkv_window=4,
        anchorkv_anchor_frac=0.2,
        anchorkv_seed=0,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 20, H: int = 2, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), AnchorKVKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "anchorkv_bytes")


def test_not_trimmable() -> None:
    assert _make().is_trimmable() is False


# ---------------------------------------------------------------------------
# No-eviction invariant — the core structural difference from H2O/SnapKV/etc.
# ---------------------------------------------------------------------------


def test_prefill_never_drops_tokens() -> None:
    """Output token count at prefill must equal S exactly, at ANY budget."""
    for theta in (0.02, 0.1, 0.5, 1.0):
        c = _make(anchorkv_theta=theta)
        k, v = _rand_kv(S=30, H=2, D=32)
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape[2] == 30, f"theta={theta} dropped tokens"
        assert vo.shape[2] == 30


def test_decode_tokens_always_appended() -> None:
    c = _make(anchorkv_theta=0.1)
    k, v = _rand_kv(S=20, H=2, D=32)
    ko, _ = c.update_and_fetch(k, v)
    prefill_size = ko.shape[2]

    for i in range(10):
        kd, vd = _rand_kv(S=1, H=2, D=32, seed=50 + i)
        ko, vo = c.update_and_fetch(kd, vd)

    assert ko.shape[2] == prefill_size + 10
    assert vo.shape[2] == prefill_size + 10


def test_tokens_kept_equals_tokens_total_always() -> None:
    c = _make(anchorkv_theta=0.05)
    k, v = _rand_kv(S=40, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_kept == c.tokens_total

    kd, vd = _rand_kv(S=1, H=2, D=32, seed=99)
    c.update_and_fetch(kd, vd)
    assert c.tokens_kept == c.tokens_total


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=10)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_output_batch_head_dims_preserved() -> None:
    c = _make()
    k, v = _rand_kv(S=16, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[0] == 1
    assert ko.shape[1] == 4
    assert ko.shape[3] == 32


def test_single_token_prefill_below_window() -> None:
    """S <= window: no meaningful compression possible, must still not crash or drop."""
    c = _make(anchorkv_window=8, anchorkv_anchor_frac=0.5)
    k, v = _rand_kv(S=3, H=1, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 3


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_gt_1_at_aggressive_budget() -> None:
    c = _make(anchorkv_theta=0.05, anchorkv_anchor_frac=0.1, anchorkv_window=4)
    k, v = _rand_kv(S=200, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_compression_ratio_lower_at_larger_theta() -> None:
    k, v = _rand_kv(S=200, H=2, D=32, seed=1)
    c_low = _make(anchorkv_theta=0.05, anchorkv_anchor_frac=0.1, anchorkv_window=4)
    c_low.update_and_fetch(k, v)

    k2, v2 = _rand_kv(S=200, H=2, D=32, seed=1)
    c_high = _make(anchorkv_theta=0.6, anchorkv_anchor_frac=0.1, anchorkv_window=4)
    c_high.update_and_fetch(k2, v2)

    assert c_low.compression_ratio > c_high.compression_ratio


def test_anchorkv_bytes_positive_after_update() -> None:
    c = _make()
    k, v = _rand_kv(S=30, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.anchorkv_bytes > 0


def test_full_seq_bytes_matches_fp16_accounting() -> None:
    c = _make()
    B, H, S, D = 1, 2, 20, 32
    k, v = _rand_kv(S=S, H=H, D=D)
    c.update_and_fetch(k, v)
    assert c.full_seq_bytes == B * H * S * D * 2 * 2


def test_n_anchor_and_residual_totals_positive() -> None:
    c = _make(anchorkv_theta=0.3)
    k, v = _rand_kv(S=40, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.n_anchor_total > 0
    assert c.n_residual_total >= 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=25, H=2, D=32)
    c1 = _make(anchorkv_seed=11)
    c2 = _make(anchorkv_seed=11)
    ko1, _ = c1.update_and_fetch(k, v)
    ko2, _ = c2.update_and_fetch(k, v)
    mse = float(mx.mean((ko1.astype(mx.float32) - ko2.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)


# ---------------------------------------------------------------------------
# for_model construction
# ---------------------------------------------------------------------------


def test_build_via_for_model_propagates_config() -> None:
    from veloxquant_mlx.cache.base import KVCacheBuilder

    class _Attn:
        head_dim = 32

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="anchorkv",
        head_dim=32,
        anchorkv_theta=0.15,
        anchorkv_window=16,
        anchorkv_rho=0.6,
        anchorkv_anchor_frac=0.1,
        anchorkv_residual_bits=2,
        anchorkv_seed=5,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, AnchorKVKVCache) for c in caches)
    assert caches[0]._theta == 0.15
    assert caches[0]._window == 16
    assert caches[0]._rho == 0.6
    assert caches[0]._anchor_frac == 0.1
    assert caches[0]._residual_bits == 2
    assert caches[0]._seed == 5


def test_factory_smoke_compression_ratio_positive_both_kv() -> None:
    """End-to-end factory smoke test exercising both K and V through the full path."""
    c = _make(anchorkv_theta=0.1, anchorkv_anchor_frac=0.1, anchorkv_window=4)
    k, v = _rand_kv(S=64, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 64
    assert vo.shape[2] == 64
    assert c.compression_ratio > 1.0
