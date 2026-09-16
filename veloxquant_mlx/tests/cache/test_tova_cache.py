"""Tests for TOVAKVCache — current-step attention-weight token eviction.

TOVA-adapted (arXiv:2401.06104) scores each token by the attention weight it
receives at the current step (memoryless — no accumulation) and evicts the
lowest-weight non-sink token whenever the cache exceeds tova_budget. Tests cover:
factory dispatch, interface attributes, output shape bounded by budget, output
dtype fp16, sink protection, decode budget enforcement across many steps, byte
accounting (compression_ratio, tova_kept_bytes), tokens_kept, n_sink=0 edge case,
determinism, and for_model config propagation. All data is synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.tova_cache import TOVAKVCache


def _make(**cfg):
    base = {"method": "tova", "head_dim": 32, "tova_budget": 8, "tova_n_sink": 2}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 4, H: int = 2, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), TOVAKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "tova_kept_bytes")


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_below_budget() -> None:
    """S < budget → all tokens returned."""
    c = _make(tova_budget=16, tova_n_sink=2)
    k, v = _rand_kv(S=6, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 6
    assert vo.shape[2] == 6


def test_output_shape_bounded_by_budget() -> None:
    """S > budget → STORED seq dim <= budget.

    The call's own RETURN value is deliberately NOT capped at budget (see
    #370): mlx_lm's attention mask for this call is fixed before eviction
    can run, so update_and_fetch defers eviction to storage only and
    returns the full pre-eviction set for this call's own (already
    correctly masked) attention.
    """
    budget = 8
    c = _make(tova_budget=budget, tova_n_sink=2)
    k, v = _rand_kv(S=20, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 20
    assert c.keys.shape[2] <= budget


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=4)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_output_batch_head_dims_preserved() -> None:
    """B and H dims pass through unchanged."""
    c = _make(tova_budget=16, tova_n_sink=0)
    k, v = _rand_kv(S=4, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[0] == 1  # B
    assert ko.shape[1] == 4  # H
    assert ko.shape[3] == 32  # D


# ---------------------------------------------------------------------------
# Budget enforcement across steps
# ---------------------------------------------------------------------------


def test_budget_enforced_after_many_steps() -> None:
    """30 decode steps — STORED seq dim never exceeds budget.

    Each step's own return is one token larger than what's stored (the
    previous step's kept set + this step's 1 new token, un-evicted — see
    #370's deferred-eviction fix in update_and_fetch's docstring).
    """
    budget = 10
    c = _make(tova_budget=budget, tova_n_sink=3)
    for i in range(30):
        k, v = _rand_kv(S=1, H=2, D=32, seed=i)
        c.update_and_fetch(k, v)
        assert c.keys.shape[2] <= budget, f"step {i}: stored={c.keys.shape[2]} > {budget}"


def test_tokens_kept_bounded_by_budget() -> None:
    budget = 8
    c = _make(tova_budget=budget, tova_n_sink=2)
    k, v = _rand_kv(S=20, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_kept <= budget


# ---------------------------------------------------------------------------
# Sink protection
# ---------------------------------------------------------------------------


def test_n_sink_zero_still_enforces_budget() -> None:
    """With n_sink=0, all tokens may be evicted; STORED budget still
    respected (see #370: the call's own return is deliberately un-evicted)."""
    budget = 4
    c = _make(tova_budget=budget, tova_n_sink=0)
    k, v = _rand_kv(S=20, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 20
    assert c.keys.shape[2] <= budget


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_equals_1_below_budget() -> None:
    """When tokens < budget, no eviction → ratio == 1."""
    c = _make(tova_budget=32, tova_n_sink=0)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio == pytest.approx(1.0, rel=1e-3)


def test_compression_ratio_gt_1_after_evictions() -> None:
    """After many evictions, ratio > 1."""
    c = _make(tova_budget=8, tova_n_sink=2)
    k, v = _rand_kv(S=100, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_tokens_seen_accumulates() -> None:
    """tokens_seen grows by B * H * S per call."""
    c = _make(tova_budget=32)
    k, v = _rand_kv(S=6, H=2, D=32)
    c.update_and_fetch(k, v)
    # B=1, H=2, S=6 → 12
    assert c.tokens_seen == 12


def test_tova_kept_bytes_positive_after_update() -> None:
    c = _make(tova_budget=16)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tova_kept_bytes > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=12, H=2, D=32)
    c1 = _make()
    c2 = _make()
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
        method="tova",
        head_dim=32,
        tova_budget=64,
        tova_n_sink=8,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, TOVAKVCache) for c in caches)
    assert caches[0]._budget == 64
    assert caches[0]._n_sink == 8


# ==================================================================
# RoPE position bookkeeping (#171, #175)
# ==================================================================


def test_offset_tracks_true_position_after_eviction() -> None:
    """``cache.offset`` must be the true token position, not the retained count.

    mlx_lm rotates both the query and the incoming key at ``offset=cache.offset``
    *before* calling ``update_and_fetch``. Before this fix, ``self.offset`` was
    left at whatever the base ``KVCache.update_and_fetch`` set it to — the
    number of RETAINED rows — so once eviction pinned the kept count at the
    budget, the offset stopped advancing and every later token was rotated at
    the wrong (stale) position while its true position kept climbing. This
    reproduces that drift without the fix: without ``_true_offset`` tracking,
    ``cache.offset`` would stall at ``budget`` instead of tracking ``t + 1``.
    """
    budget = 8
    cache = _make(tova_budget=budget, tova_n_sink=2)

    n_steps = 5 * budget
    for t in range(n_steps):
        k, v = _rand_kv(S=1, H=2, D=32, seed=100 + t)
        cache.update_and_fetch(k, v)
        assert cache.offset == t + 1, (
            f"offset {cache.offset} != true position {t + 1} — RoPE would be wrong"
        )

    assert cache.tokens_kept <= budget


def test_offset_advances_by_block_size_on_prefill() -> None:
    """A multi-token block advances the offset by S, not by rows retained."""
    S, budget = 100, 16
    cache = _make(tova_budget=budget, tova_n_sink=2)

    k, v = _rand_kv(S=S, H=2, D=32, seed=7)
    cache.update_and_fetch(k, v)
    assert cache.offset == S
    assert cache.tokens_kept <= budget

    k2, v2 = _rand_kv(S=1, H=2, D=32, seed=8)
    cache.update_and_fetch(k2, v2)
    assert cache.offset == S + 1


def test_offset_survives_prefill_then_decode_mix() -> None:
    """Offset stays the true position across a prefill block followed by
    many decode steps, even though eviction is active throughout."""
    S, budget = 40, 12
    cache = _make(tova_budget=budget, tova_n_sink=2)

    k, v = _rand_kv(S=S, H=2, D=32, seed=9)
    cache.update_and_fetch(k, v)
    assert cache.offset == S

    for t in range(30):
        kd, vd = _rand_kv(S=1, H=2, D=32, seed=200 + t)
        cache.update_and_fetch(kd, vd)
        assert cache.offset == S + t + 1
    assert cache.tokens_kept <= budget


@pytest.mark.parametrize("backend", ["mlx", "metal", "reference"])
def test_batched_cache_state_and_accounting(backend):
    cache = _make(tova_backend=backend)
    reference = _make(tova_backend="reference")
    rng = np.random.default_rng(61)
    total = 0
    for s in [3, 11, 1, 17]:
        k, v = [mx.array(rng.normal(size=(2, 3, s, 33)).astype(np.float16)) for _ in range(2)]
        actual = cache.update_and_fetch(k, v)
        expected = reference.update_and_fetch(k, v)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(np.array(a), np.array(e))
        total += s
        assert cache.offset == total
        assert cache.size() == min(total, 8)
        # `state` (STORED) is capped at budget; `actual` (this call's own
        # RETURN) deliberately is not (see #370's deferred-eviction fix).
        assert cache.state[0].shape[2] == min(total, 8)
        assert cache.tokens_seen == 6 * total
        assert cache.tova_kept_bytes == 6 * min(total, 8) * 33 * 4
        assert cache.nbytes == cache.tova_kept_bytes
        assert not cache.is_trimmable()


def test_restore_rebuilds_compatibility_states():
    cache = _make(tova_backend="mlx")
    k, v = _rand_kv(S=4)
    cache.state = (k, v)
    cache.update_and_fetch(*_rand_kv(S=1))
    assert cache.offset == 5
    assert cache.tokens_kept == 5
    assert cache.state[0].shape[2] == 5


def test_empty_update_is_noop_and_default_mask_contract():
    cache = _make()
    empty = mx.zeros((1, 2, 0, 32))
    assert cache.update_and_fetch(empty, empty)[0].shape == empty.shape
    assert cache.offset == cache.size() == cache.tokens_seen == 0
    assert cache.make_mask(1, return_array=False, window_size=None) is None
    assert cache.make_mask(4, return_array=False, window_size=None) == "causal"
    k, v = cache.update_and_fetch(*_rand_kv(S=12))
    a, b = cache.update_and_fetch(empty, empty)
    assert a is k and b is v
    assert cache.offset == 12


def test_shape_change_rejected_without_mutating_state():
    cache = _make()
    cache.update_and_fetch(*_rand_kv(S=2))
    before = cache.state
    with pytest.raises(ValueError, match="cannot change"):
        cache.update_and_fetch(*_rand_kv(S=1, H=3))
    assert cache.state == before
    assert cache.offset == 2


# ==================================================================
# merge() batching guard (VeloxQuant-MLX#358)
# ==================================================================


def test_not_batchable_via_mlx_lm_server_probe():
    """mlx_lm.server decides batchability via hasattr(cache, "merge").

    Without an explicit guard, TOVAKVCache inherits the base mlx_lm
    KVCache.merge() classmethod unchanged, so hasattr(cache, "merge") is
    True and the server treats it as batchable -- silently substituting a
    plain BatchKVCache (see the next test) instead of ever running real
    TOVA eviction. Found verifying VeloxQuant-Studio issue #31.
    """
    cache = _make()
    assert not hasattr(cache, "merge")


def test_merge_on_empty_cache_would_silently_substitute_if_inherited():
    """Reproduces the substitution mlx_lm.server would perform if
    TOVAKVCache had no merge() guard: calling the inherited base-class
    merge() classmethod on a fresh cache list returns a plain BatchKVCache
    with none of TOVA's eviction behavior, and no error is raised."""
    from mlx_lm.models.cache import BatchKVCache
    from mlx_lm.models.cache import KVCache as _MLXKVCache

    cache = _make()
    substituted = _MLXKVCache.merge([cache])
    assert isinstance(substituted, BatchKVCache)
    assert not isinstance(substituted, TOVAKVCache)


def test_merge_raises_attribute_error():
    cache = _make()
    with pytest.raises(AttributeError, match="does not support merge"):
        cache.merge


def test_real_mlx_lm_llama_gqa_forward_parity():
    """Actual mlx-lm model code with tiny random weights, not a quality test."""
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(72)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=96,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
        )
    )
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    paths = [
        [_make(tova_backend=b, tova_budget=8) for _ in range(2)]
        for b in ["reference", "mlx", "metal"]
    ]
    for step in range(129):
        tokens = mx.array([[1, 2, 3, 4]]) if step == 0 else mx.array([[step % 64]])
        outputs = [model(tokens, cache=caches) for caches in paths]
        mx.eval(*outputs)
        for out in outputs[1:]:
            np.testing.assert_array_equal(np.array(out), np.array(outputs[0]))
    assert all(c.offset == 132 and c.size() == 8 for caches in paths for c in caches)


# ---------------------------------------------------------------------------
# Attention mask correctness (#370)
# ---------------------------------------------------------------------------


def test_make_mask_before_any_call_falls_back_to_base() -> None:
    from mlx_lm.models.base import create_attention_mask

    c = _make(tova_budget=8, tova_n_sink=2)
    h_fake = mx.zeros((1, 5, 4))
    assert create_attention_mask(h_fake, c) == "causal"


def test_every_call_returns_full_unevicted_set_for_own_attention() -> None:
    """Neither the first nor any later multi-token call may shrink what it
    RETURNS below its own pre-eviction count — mlx_lm's mask for that call
    is fixed (based on the previous call's true kept positions) before this
    call's own eviction can run, and only what's returned matches that
    fixed mask's shape."""
    budget = 6
    c = _make(tova_budget=budget, tova_n_sink=1)
    k1, v1 = _rand_kv(S=5, H=1, D=8, seed=1)
    ko1, _ = c.update_and_fetch(k1, v1)
    assert ko1.shape[2] == 5  # first call: nothing stored yet to concat onto
    assert c.keys.shape[2] <= budget

    k2, v2 = _rand_kv(S=5, H=1, D=8, seed=2)
    ko2, _ = c.update_and_fetch(k2, v2)
    # returned == (previously stored, <= budget) ++ (this call's 5 new)
    prev_stored = min(5, budget)
    assert ko2.shape[2] == prev_stored + 5
    assert c.keys.shape[2] <= budget


def test_make_mask_after_eviction_is_position_correct_explicit_array() -> None:
    from mlx_lm.models.base import create_attention_mask

    budget = 4
    c = _make(tova_budget=budget, tova_n_sink=1)
    k1, v1 = _rand_kv(S=15, H=1, D=8, seed=3)
    c.update_and_fetch(k1, v1)
    assert c.keys.shape[2] <= budget
    kept_positions = c._kept_positions[0].tolist()

    h_fake = mx.zeros((1, 3, 4))
    mask = create_attention_mask(h_fake, c)
    assert isinstance(mask, mx.array)
    n_stored = c.keys.shape[2]
    assert mask.shape == (1, 1, 3, n_stored + 3)

    query_positions = [c.offset + i for i in range(3)]
    key_positions = kept_positions + [c.offset + i for i in range(3)]
    expected = [[kj <= qi for kj in key_positions] for qi in query_positions]
    assert mask[0, 0].tolist() == expected


def test_make_mask_single_query_returns_none() -> None:
    from mlx_lm.models.base import create_attention_mask

    c = _make(tova_budget=4, tova_n_sink=1)
    k, v = _rand_kv(S=15, H=1, D=8, seed=3)
    c.update_and_fetch(k, v)

    h_fake = mx.zeros((1, 1, 4))
    assert create_attention_mask(h_fake, c) is None
