"""Tests for QFiltersKVCache — query-agnostic projection eviction wrapper.

Covers mlx_lm protocol shape/dtype preservation, budget enforcement,
pre-calibration passthrough, sink protection, the honest prefill-vs-decode
NON-equivalence (both valid, not bit-for-bit — Q-Filters is path-dependent),
the sign ablation knob, the mechanism test under paper-like geometry, byte
accounting, build-time validation, and for_model wiring.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache
from veloxquant_mlx.quantizers.qfilters import qfilters_fp16_bytes


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(**cfg):
    base = dict(
        method="qfilters",
        head_dim=64,
        qfilters_budget=16,
        qfilters_n_sink=2,
        qfilters_calib_tokens=16,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ------------------------------------------------------------------
# Protocol basics
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), QFiltersKVCache)


def test_shape_dtype_preserved() -> None:
    cache = _make(qfilters_budget=128)
    k, v = _kv(1, 4, 64, 64)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 4, 64, 64)  # under budget: everything kept
    assert vo.shape == (1, 4, 64, 64)
    assert ko.dtype == mx.float16


def test_no_bits_leak() -> None:
    assert not hasattr(_make(), "bits")


def test_under_budget_bit_for_bit_passthrough() -> None:
    cache = _make(qfilters_budget=128)
    k, v = _kv(1, 2, 32, 64, seed=1)
    ko, vo = cache.update_and_fetch(k, v)
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))


def test_pre_calibration_passthrough() -> None:
    """Below calib_tokens, no filter yet — nothing evicted even over budget."""
    cache = _make(qfilters_budget=8, qfilters_calib_tokens=200)
    k, v = _kv(1, 2, 50, 64, seed=2)  # 50 > budget 8, but < calib 200
    ko, _ = cache.update_and_fetch(k, v)
    assert ko.shape[2] == 50


# ------------------------------------------------------------------
# Eviction behavior
# ------------------------------------------------------------------


def test_budget_enforced_after_long_prefill() -> None:
    cache = _make(qfilters_budget=16)
    k, v = _kv(1, 2, 200, 64, seed=3)
    ko, _ = cache.update_and_fetch(k, v)
    assert ko.shape[2] == 16
    assert cache.tokens_kept == 16


def test_sinks_retained_across_heavy_eviction() -> None:
    B, H, S, D, n_sink = 1, 1, 120, 64, 3
    k, v = _kv(B, H, S, D, seed=4)
    cache = _make(qfilters_budget=8, qfilters_n_sink=n_sink, head_dim=D)
    ko, _ = cache.update_and_fetch(k, v)
    assert np.array_equal(np.array(ko[0, 0, :n_sink]), np.array(k[0, 0, :n_sink]))


def test_decode_accumulation_caps_at_budget() -> None:
    cache = _make(qfilters_budget=8, qfilters_n_sink=2, qfilters_calib_tokens=8)
    for t in range(30):
        k, v = _kv(1, 2, 1, 64, seed=100 + t)
        ko, _ = cache.update_and_fetch(k, v)
    assert ko.shape[2] == 8
    assert cache.tokens_seen == 30 * 2


def test_prefill_decode_both_valid_not_equivalent() -> None:
    """Q-Filters is path-DEPENDENT: prefill and decode may freeze different
    filters. We assert both stay within budget and both freeze a valid
    unit-norm filter — NOT that they are bit-for-bit equal."""
    S = 80
    k, v = _kv(1, 1, S, 64, seed=5)

    block = _make(qfilters_budget=20, qfilters_n_sink=2, head_dim=64)
    ka, _ = block.update_and_fetch(k, v)

    stream = _make(qfilters_budget=20, qfilters_n_sink=2, head_dim=64)
    for t in range(S):
        kb, _ = stream.update_and_fetch(k[:, :, t : t + 1, :], v[:, :, t : t + 1, :])
    mx.eval(ka, kb)

    assert ka.shape[2] <= 20 and kb.shape[2] <= 20
    for st in (block._states[0], stream._states[0]):
        d = np.array(st.filter_dir)
        assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-4


def test_sign_differs_and_respects_budget() -> None:
    k, v = _kv(1, 2, 80, 64, seed=6)
    pos = _make(qfilters_budget=16, qfilters_sign=1)
    neg = _make(qfilters_budget=16, qfilters_sign=-1)
    kp, _ = pos.update_and_fetch(k, v)
    kn, _ = neg.update_and_fetch(k, v)
    assert kp.shape[2] == 16 and kn.shape[2] == 16
    assert not np.array_equal(np.array(kp), np.array(kn))


# ------------------------------------------------------------------
# Mechanism test under paper-like geometry
# ------------------------------------------------------------------


def _attn_out(q, k, v):
    scale = 1.0 / math.sqrt(float(k.shape[-1]))
    w = mx.softmax((q @ k.T) * scale, axis=-1)
    return w @ v


def test_projection_scorer_beats_random_under_paper_geometry() -> None:
    """Construct the QK anisotropy the paper (arXiv:2503.02812) reports:
    important tokens carry a large projection onto the dominant axis and align
    with the query cluster; the rest are near-orthogonal noise.

    The honest claim (given a KEY-derived filter): the key-SVD recovers the
    dominant *axis* but not which *end* is important — that sign is exactly
    what a query would disambiguate and the cache never sees one. So the
    ``sign`` knob is a real ablation, and the correct-sign cache's attention
    output must clearly beat random eviction. We assert the *better of the two
    signs* beats random by a wide margin (not that +1 specifically wins) —
    that is the truthful statement for a query-agnostic key-only estimator."""
    rng = np.random.default_rng(7)
    D, S, n_imp = 64, 256, 24
    mu = rng.standard_normal(D).astype(np.float32)
    mu /= np.linalg.norm(mu)

    k = np.zeros((S, D), dtype=np.float32)
    imp_idx = rng.choice(S, size=n_imp, replace=False)
    imp_mask = np.zeros(S, dtype=bool)
    imp_mask[imp_idx] = True
    k[imp_mask] = 4.0 * mu + 0.2 * rng.standard_normal((n_imp, D))
    k[~imp_mask] = 0.3 * rng.standard_normal((S - n_imp, D))
    v = rng.standard_normal((S, D)).astype(np.float32)
    q = (mu + 0.1 * rng.standard_normal((16, D))).astype(np.float32)

    kk = mx.array(k[None, None].astype(np.float16))
    vv = mx.array(v[None, None].astype(np.float16))
    ref = _attn_out(mx.array(q), mx.array(k), mx.array(v))
    budget = n_imp + 4

    def _pert(out) -> float:
        rn = ref / (mx.sqrt(mx.sum(ref * ref, -1, keepdims=True)) + 1e-8)
        on = out / (mx.sqrt(mx.sum(out * out, -1, keepdims=True)) + 1e-8)
        return float(mx.mean(1.0 - mx.sum(rn * on, -1)).item())

    def sign_pert(sign: int) -> float:
        cache = _make(
            qfilters_budget=budget,
            qfilters_n_sink=0,
            qfilters_sign=sign,
            qfilters_calib_tokens=32,
            head_dim=D,
        )
        ko, vo = cache.update_and_fetch(kk, vv)
        return _pert(
            _attn_out(mx.array(q), ko[0, 0].astype(mx.float32), vo[0, 0].astype(mx.float32))
        )

    # Random-eviction baseline at the same budget.
    rperts = []
    for s in range(8):
        idx = np.sort(np.random.default_rng(s).choice(S, budget, replace=False))
        rperts.append(_pert(_attn_out(mx.array(q), mx.array(k[idx]), mx.array(v[idx]))))
    random_pert = float(np.mean(rperts))

    best_sign = min(sign_pert(1), sign_pert(-1))
    assert best_sign < 0.5 * random_pert


# ------------------------------------------------------------------
# Accounting / validation / wiring
# ------------------------------------------------------------------


def test_compression_ratio_math() -> None:
    D = 64
    cache = _make(qfilters_budget=16, head_dim=D)
    k, v = _kv(1, 2, 128, D, seed=8)
    cache.update_and_fetch(k, v)
    # 2 heads: K+V fp16 for 16 tokens each, plus D float32 filter per head.
    expected = 2 * (16 * D * 2 * 2 + D * 4)
    assert cache.qfilters_kept_bytes == expected
    assert cache.full_seq_bytes == 2 * 128 * D * 2 * 2


def test_determinism() -> None:
    k, v = _kv(1, 2, 80, 64, seed=9)
    a, b = _make(), _make()
    ka, va = a.update_and_fetch(k, v)
    kb, vb = b.update_and_fetch(k, v)
    assert np.array_equal(np.array(ka), np.array(kb))
    assert np.array_equal(np.array(va), np.array(vb))


def test_build_time_validation() -> None:
    with pytest.raises(ValueError, match="sign"):
        _make(qfilters_sign=3)
    with pytest.raises(ValueError, match="evictable"):
        _make(qfilters_budget=8, qfilters_n_sink=4, qfilters_recent=4)


class _ToyAttn:
    def __init__(self, head_dim):
        self.head_dim = head_dim


class _ToyLayer:
    def __init__(self, head_dim=64):
        self.self_attn = _ToyAttn(head_dim)


class _ToyNorm:
    pass


class _ToyInner:
    def __init__(self):
        self.layers = [_ToyLayer(), _ToyNorm(), _ToyLayer()]


class _ToyModel:
    def __init__(self):
        self.model = _ToyInner()
        self.args = None


def test_for_model_wiring_and_fallback() -> None:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    caches = KVCacheBuilder.for_model(_ToyModel(), KVCacheConfig(method="qfilters", head_dim=64))
    assert len(caches) == 3
    assert isinstance(caches[0], QFiltersKVCache)
    assert type(caches[1]) is _FallbackCache
    assert isinstance(caches[2], QFiltersKVCache)


# ==================================================================
# Calibrated (paper-faithful) filters — see quantizers/qfilters_calibration.py
# ==================================================================


def _calibrated_filters(H: int, D: int, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((H, D))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return mx.array(f.astype(np.float32))


def _kv_block(B: int, H: int, S: int, D: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def test_calibrated_filter_evicts_without_warmup() -> None:
    """A calibrated filter is frozen before token 0 — no calibration window.

    The key-SVD fallback must first observe `calib_tokens` keys, during which
    the cache grows past its budget. A supplied filter removes that window
    entirely.
    """
    B, H, S, D = 1, 4, 80, 16
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=32, qfilters_n_sink=2)
    k, v = _kv_block(B, H, S, D, seed=1)

    calibrated = QFiltersKVCache(cfg, filters=_calibrated_filters(H, D))
    k_out, _ = calibrated.update_and_fetch(k, v)
    assert k_out.shape[2] == 32

    # Fallback: still inside its calibration window, so nothing is evicted.
    fallback = QFiltersKVCache(cfg)
    k_fb, _ = fallback.update_and_fetch(k, v)
    assert k_fb.shape[2] == S


def test_calibrated_filter_is_path_independent() -> None:
    """Prefill-in-one-block and token-by-token decode agree bit-for-bit.

    This is the guarantee the key-SVD fallback cannot make: its direction
    depends on which chunk first crosses the calibration threshold.
    """
    B, H, S, D = 1, 4, 80, 16
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=32, qfilters_n_sink=2)
    filters = _calibrated_filters(H, D, seed=2)
    k, v = _kv_block(B, H, S, D, seed=3)

    prefill = QFiltersKVCache(cfg, filters=filters)
    k_prefill, v_prefill = prefill.update_and_fetch(k, v)

    decode = QFiltersKVCache(cfg, filters=filters)
    k_decode = v_decode = None
    for t in range(S):
        k_decode, v_decode = decode.update_and_fetch(k[:, :, t : t + 1], v[:, :, t : t + 1])

    assert np.array_equal(np.array(k_prefill), np.array(k_decode))
    assert np.array_equal(np.array(v_prefill), np.array(v_decode))


@pytest.mark.parametrize("chunk", [1073, 512, 256, 64, 16, 1])
def test_calibrated_kept_set_is_invariant_to_prefill_chunk_size(chunk: int) -> None:
    """Same prompt, any chunking — bit-identical retained K/V and offset (#172).

    Q-Filters scores a token once, against a filter frozen before token 0, and
    a token's score never changes afterwards. Top-k under a fixed key is
    order-invariant, so absorbing S tokens then evicting once and evicting
    incrementally select the *same* rows. This pins that: the chunk-size
    sensitivity reported in #172 is not an eviction-ordering effect.
    """
    B, H, S, D = 1, 4, 1073, 16
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=128, qfilters_n_sink=4)
    filters = _calibrated_filters(H, D, seed=7)
    k, v = _kv_block(B, H, S, D, seed=8)

    one_shot = QFiltersKVCache(cfg, filters=filters)
    k_ref, v_ref = one_shot.update_and_fetch(k, v)

    chunked = QFiltersKVCache(cfg, filters=filters)
    k_out = v_out = None
    for i in range(0, S, chunk):
        k_out, v_out = chunked.update_and_fetch(k[:, :, i : i + chunk], v[:, :, i : i + chunk])

    assert np.array_equal(np.array(k_ref), np.array(k_out))
    assert np.array_equal(np.array(v_ref), np.array(v_out))
    # Budget is respected regardless of how the prompt was fed in.
    assert k_out.shape[2] == 128
    # True absolute position must not depend on chunking, or RoPE drifts.
    assert chunked.offset == one_shot.offset == S


@pytest.mark.parametrize("chunk", [1073, 256, 64])
def test_recent_window_protects_prompt_tail_at_every_chunk_size(chunk: int) -> None:
    """``qfilters_recent`` pins the trailing window regardless of chunking (#172).

    With ``recent=0`` the tail of a long prompt — where the actual question
    lives — is scored like any other token and is almost entirely evicted,
    which is what produces degenerate generation. The trailing guard is the
    knob that prevents it, and it must behave identically under one-shot and
    chunked prefill.
    """
    B, H, S, D, recent = 1, 2, 1073, 16, 32
    cfg = KVCacheConfig(
        method="qfilters",
        head_dim=D,
        qfilters_budget=128,
        qfilters_n_sink=4,
        qfilters_recent=recent,
    )
    filters = _calibrated_filters(H, D, seed=9)
    k, v = _kv_block(B, H, S, D, seed=10)

    cache = QFiltersKVCache(cfg, filters=filters)
    k_out = None
    for i in range(0, S, chunk):
        k_out, _ = cache.update_and_fetch(k[:, :, i : i + chunk], v[:, :, i : i + chunk])

    assert k_out.shape[2] == 128
    # The final `recent` rows of the window are exactly the prompt's last
    # `recent` tokens, in order — never evicted, never reordered.
    tail_expected = np.array(k)[:, :, S - recent :]
    assert np.array_equal(np.array(k_out)[:, :, -recent:], tail_expected)


def _reference_evict(k, v, filters, budget, n_sink, recent, sign):
    """Per-(b, h) numpy oracle for the vectorized path (#173).

    Deliberately an independent reimplementation rather than a call into
    ``qfilters_update``: a parity test that shares code with the thing it
    checks can only catch wiring mistakes, not selection mistakes.
    """
    k = np.array(k, dtype=np.float16)
    v = np.array(v, dtype=np.float16)
    f = np.array(filters, dtype=np.float32)
    B, H, S, _ = k.shape
    ks, vs = [], []
    for b in range(B):
        kh, vh = [], []
        for h in range(H):
            scores = k[b, h].astype(np.float32) @ f[h] * sign
            if S > budget:
                sel = scores.copy()
                sel[: min(n_sink, S)] = np.inf
                if recent > 0:
                    sel[S - min(recent, S - min(n_sink, S)) :] = np.inf
                keep = np.sort(np.argsort(sel, kind="stable")[S - budget :])
            else:
                keep = np.arange(S)
            kh.append(k[b, h][keep])
            vh.append(v[b, h][keep])
        ks.append(np.stack(kh))
        vs.append(np.stack(vh))
    return np.stack(ks), np.stack(vs)


@pytest.mark.parametrize(
    ("B", "H", "S", "budget", "n_sink", "recent", "sign"),
    [
        (1, 4, 300, 64, 4, 0, 1),  # single batch
        (3, 4, 300, 64, 4, 0, 1),  # batched — the loop's worst case
        (2, 3, 257, 48, 4, 8, 1),  # recent window + ragged S
        (1, 4, 300, 64, 2, 0, -1),  # inverted sign ablation
        (2, 2, 20, 64, 4, 0, 1),  # under budget — no eviction at all
        (2, 8, 512, 128, 4, 16, 1),  # GQA-shaped head count
    ],
)
def test_batched_path_matches_per_head_reference(
    B: int, H: int, S: int, budget: int, n_sink: int, recent: int, sign: int
) -> None:
    """Vectorizing the B×H loop must not move a single bit (#173)."""
    D = 16
    cfg = KVCacheConfig(
        method="qfilters",
        head_dim=D,
        qfilters_budget=budget,
        qfilters_n_sink=n_sink,
        qfilters_recent=recent,
        qfilters_sign=sign,
    )
    filters = _calibrated_filters(H, D, seed=11)
    k, v = _kv_block(B, H, S, D, seed=12)

    cache = QFiltersKVCache(cfg, filters=filters)
    k_out, v_out = cache.update_and_fetch(k, v)

    k_ref, v_ref = _reference_evict(k, v, filters, budget, n_sink, recent, sign)
    assert np.array_equal(np.array(k_out), k_ref)
    assert np.array_equal(np.array(v_out), v_ref)


def test_batched_path_state_mirror_stays_consistent() -> None:
    """Lazily-rebuilt per-head states must match the batched buffers (#173).

    The fast path leaves ``_states`` stale and refreshes it on demand; if that
    sync drifts, byte accounting and ``tokens_kept`` silently go wrong.
    """
    B, H, S, D, budget = 2, 4, 300, 16, 64
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=budget, qfilters_n_sink=4)
    filters = _calibrated_filters(H, D, seed=13)
    k, v = _kv_block(B, H, S, D, seed=14)

    cache = QFiltersKVCache(cfg, filters=filters)
    k_out, _ = cache.update_and_fetch(k, v)

    assert cache.tokens_kept == budget
    states = cache.states
    assert len(states) == B * H
    for g, st in enumerate(states):
        b, h = divmod(g, H)
        assert np.array_equal(np.array(st.keys), np.array(k_out)[b, h])
        assert st.scores.shape[0] == budget
    # Closed-form byte accounting must equal the per-state sum it replaced.
    assert cache.qfilters_kept_bytes == sum(qfilters_fp16_bytes(st) for st in states)


def test_fallback_path_still_uses_per_head_loop() -> None:
    """The key-SVD fallback must not be silently routed to the batched path.

    Its filters freeze per group at different times, which the shared-filter
    vectorization cannot represent — so it must keep the loop (#173).
    """
    cfg = KVCacheConfig(method="qfilters", head_dim=16, qfilters_budget=32)
    fallback = QFiltersKVCache(cfg)
    assert fallback._can_batch() is False
    calibrated = QFiltersKVCache(cfg, filters=_calibrated_filters(4, 16))
    assert calibrated._can_batch() is True


def test_calibrated_selection_matches_projection_ranking() -> None:
    """The kept set is the budget highest-projection rows against the filter."""
    B, H, S, D, budget = 1, 2, 40, 8, 12
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=budget, qfilters_n_sink=0)
    filters = _calibrated_filters(H, D, seed=4)
    k, v = _kv_block(B, H, S, D, seed=5)

    cache = QFiltersKVCache(cfg, filters=filters)
    k_out, _ = cache.update_and_fetch(k, v)

    for h in range(H):
        scores = np.array(k, dtype=np.float32)[0, h] @ np.array(filters, dtype=np.float32)[h]
        keep = np.sort(np.argsort(scores, kind="stable")[S - budget :])
        assert np.array_equal(np.array(k)[0, h][keep], np.array(k_out)[0, h])


def test_calibrated_filters_reject_head_layout_mismatch() -> None:
    """Filters from another model must fail loudly, not silently mis-evict."""
    cfg = KVCacheConfig(method="qfilters", head_dim=16, qfilters_budget=16)
    cache = QFiltersKVCache(cfg, filters=_calibrated_filters(8, 16))
    k, v = _kv_block(1, 4, 20, 16, seed=6)  # 4 heads, filters have 8

    with pytest.raises(ValueError, match="calibrated on a different model"):
        cache.update_and_fetch(k, v)


def test_calibrated_filters_reject_wrong_rank() -> None:
    cfg = KVCacheConfig(method="qfilters", head_dim=16, qfilters_budget=16)
    with pytest.raises(ValueError, match=r"2-D \[H_kv, D\]"):
        QFiltersKVCache(cfg, filters=mx.zeros((4,), dtype=mx.float32))


# ==================================================================
# RoPE position bookkeeping (#171)
# ==================================================================


def test_offset_tracks_true_position_after_eviction() -> None:
    """``cache.offset`` must be the true token position, not the retained count.

    mlx_lm rotates both the query and the incoming key at ``offset=cache.offset``
    *before* calling ``update_and_fetch``. If the offset stalls at the budget
    once eviction begins, every later token is rotated at the wrong position and
    attention degrades without bound (measured: ppl 598 -> 17.6 on
    Llama-3.2-1B once this is correct).
    """
    B, H, D, budget = 1, 2, 8, 16
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=budget, qfilters_n_sink=2)
    cache = QFiltersKVCache(cfg, filters=_calibrated_filters(H, D))

    n_steps = 5 * budget
    for t in range(n_steps):
        k, v = _kv_block(B, H, 1, D, seed=100 + t)
        cache.update_and_fetch(k, v)
        assert cache.offset == t + 1, (
            f"offset {cache.offset} != true position {t + 1} — RoPE would be wrong"
        )

    # Eviction still bounds the cache; the offset just no longer conflates the
    # two quantities.
    assert cache.tokens_kept <= budget


def test_offset_advances_by_block_size_on_prefill() -> None:
    """A multi-token block advances the offset by S, not by rows retained."""
    B, H, S, D = 1, 2, 100, 8
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=32, qfilters_n_sink=2)
    cache = QFiltersKVCache(cfg, filters=_calibrated_filters(H, D))

    k, v = _kv_block(B, H, S, D, seed=7)
    cache.update_and_fetch(k, v)
    assert cache.offset == S
    assert cache.tokens_kept <= 32

    k2, v2 = _kv_block(B, H, 1, D, seed=8)
    cache.update_and_fetch(k2, v2)
    assert cache.offset == S + 1
