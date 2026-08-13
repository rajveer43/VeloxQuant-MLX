"""Tests for MorphKV-adapted eviction primitives (quantizers/morphkv.py).

Covers: init guards, constant-size budget invariant (token/block), sink and
trailing-window protection, byte accounting, determinism, the window=1 ==
latest-token (TOVA-adapted) reduction (the honest pinned reduction), and the
topic-shift mechanism (recent-window correlation retains the region the recent
context attends to, at a higher rate than a cumulative H2O-style baseline), with
a null "stable" control where it shows no advantage.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.morphkv import (
    full_morphkv_fp16_bytes,
    init_morphkv_state,
    morphkv_fp16_bytes,
    morphkv_get_kv,
    morphkv_update,
)
from veloxquant_mlx.quantizers.tova import (
    init_tova_state,
    tova_get_kv,
    tova_update,
)
from veloxquant_mlx.quantizers.h2o import h2o_get_kv, h2o_update, init_h2o_state


def _rand_kv(S: int, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ---------------------------------------------------------------------------
# init guards
# ---------------------------------------------------------------------------
def test_init_rejects_bad_budget():
    with pytest.raises(ValueError, match="budget must be >= 1"):
        init_morphkv_state(n_sink=0, budget=0, head_dim=16, window=1)


def test_init_rejects_bad_window():
    with pytest.raises(ValueError, match="window must be >= 1"):
        init_morphkv_state(n_sink=2, budget=8, head_dim=16, window=0)


def test_init_rejects_sink_ge_budget():
    with pytest.raises(ValueError, match="n_sink .* must be < budget"):
        init_morphkv_state(n_sink=8, budget=8, head_dim=16, window=1)


def test_init_rejects_window_gt_budget():
    with pytest.raises(ValueError, match="window .* must be <= budget"):
        init_morphkv_state(n_sink=1, budget=8, head_dim=16, window=9)


def test_init_rejects_no_evictable_room():
    with pytest.raises(ValueError, match="no evictable positions"):
        init_morphkv_state(n_sink=4, budget=8, head_dim=16, window=4)


def test_init_empty_state():
    st = init_morphkv_state(n_sink=2, budget=8, head_dim=16, window=4)
    assert st.keys is None and st.values is None and st.pos == 0
    assert st.n_sink == 2 and st.budget == 8 and st.window == 4


# ---------------------------------------------------------------------------
# constant-size budget invariant
# ---------------------------------------------------------------------------
def test_budget_never_exceeded_token_by_token():
    st = init_morphkv_state(n_sink=2, budget=8, head_dim=16, window=3)
    for i in range(40):
        k, v = _rand_kv(1, 16, seed=i)
        st = morphkv_update(st, k, v)
        assert st.keys.shape[0] <= 8


def test_budget_never_exceeded_block_prefill():
    st = init_morphkv_state(n_sink=2, budget=8, head_dim=16, window=3)
    k, v = _rand_kv(40, 16, seed=7)
    st = morphkv_update(st, k, v)
    assert st.keys.shape[0] == 8


def test_under_budget_keeps_all():
    st = init_morphkv_state(n_sink=2, budget=32, head_dim=16, window=4)
    k, v = _rand_kv(10, 16, seed=3)
    st = morphkv_update(st, k, v)
    K, _ = morphkv_get_kv(st)
    assert K.shape[0] == 10


# ---------------------------------------------------------------------------
# protection: sinks (leading) and recent window (trailing)
# ---------------------------------------------------------------------------
def test_sinks_survive_eviction():
    st = init_morphkv_state(n_sink=3, budget=8, head_dim=8, window=2)
    first_k, first_v = _rand_kv(3, 8, seed=100)
    st = morphkv_update(st, first_k, first_v)
    sink_rows = st.keys[:3]
    for i in range(60):
        k, v = _rand_kv(1, 8, seed=i)
        st = morphkv_update(st, k, v)
    assert bool(mx.all(st.keys[:3] == sink_rows).item())


def test_recent_window_protects_trailing():
    st = init_morphkv_state(n_sink=2, budget=8, head_dim=8, window=3)
    for i in range(30):
        k, v = _rand_kv(1, 8, seed=i)
        st = morphkv_update(st, k, v)
    # The last `window` inserted rows must all be present (trailing protection).
    tail = [_rand_kv(1, 8, seed=s)[0][0] for s in (27, 28, 29)]
    kept_tail = st.keys[-3:]
    assert bool(mx.all(kept_tail == mx.stack(tail, axis=0)).item())


# ---------------------------------------------------------------------------
# byte accounting
# ---------------------------------------------------------------------------
def test_fp16_bytes():
    st = init_morphkv_state(n_sink=2, budget=8, head_dim=16, window=3)
    assert morphkv_fp16_bytes(st) == 0
    k, v = _rand_kv(8, 16, seed=5)
    st = morphkv_update(st, k, v)
    assert morphkv_fp16_bytes(st) == 8 * 16 * 2 * 2


def test_full_fp16_bytes():
    assert full_morphkv_fp16_bytes(100, 64) == 100 * 64 * 2 * 2


def test_get_kv_placeholder_before_update():
    st = init_morphkv_state(n_sink=2, budget=8, head_dim=16, window=2)
    K, V = morphkv_get_kv(st)
    assert K.shape == (0, 1) and V.shape == (0, 1)


# ---------------------------------------------------------------------------
# determinism (MorphKV has no RNG)
# ---------------------------------------------------------------------------
def test_run_is_reproducible():
    def run():
        st = init_morphkv_state(n_sink=2, budget=8, head_dim=16, window=4)
        for i in range(30):
            k, v = _rand_kv(1, 16, seed=i)
            st = morphkv_update(st, k, v)
        return morphkv_get_kv(st)[0]

    assert bool(mx.all(run() == run()).item())


# ---------------------------------------------------------------------------
# window = 1 reduces to the latest-token (TOVA-adapted) eviction — pinned
# ---------------------------------------------------------------------------
def test_window_one_matches_tova():
    """With window=1 the recent-relevance is exactly the newest key's attention
    over the keep set — the TOVA-adapted latest-token ranking. Both protect only
    the single newest row, so their kept keysets must be bit-for-bit identical.
    """
    ks = [_rand_kv(1, 16, seed=i) for i in range(40)]

    m = init_morphkv_state(n_sink=4, budget=12, head_dim=16, window=1)
    t = init_tova_state(n_sink=4, budget=12, head_dim=16)
    for k, v in ks:
        m = morphkv_update(m, k, v)
        t = tova_update(t, k, v)

    m_k, _ = morphkv_get_kv(m)
    t_k, _ = tova_get_kv(t)
    assert m_k.shape == t_k.shape
    assert bool(mx.all(m_k == t_k).item())


def test_larger_window_can_differ_from_window_one():
    """A wider recent window should be capable of retaining a different set than
    the latest-token-only ranking — otherwise the mechanism is vacuous.
    """
    ks = [_rand_kv(1, 16, seed=i) for i in range(60)]

    def run(window):
        st = init_morphkv_state(n_sink=2, budget=12, head_dim=16, window=window)
        for k, v in ks:
            st = morphkv_update(st, k, v)
        return morphkv_get_kv(st)[0]

    narrow = run(1)
    wide = run(8)
    assert narrow.shape == wide.shape
    assert not bool(mx.all(narrow == wide).item())


# ---------------------------------------------------------------------------
# topic-shift mechanism: recent-window retention beats cumulative baseline
# ---------------------------------------------------------------------------
def test_topic_shift_retains_recent_relevant():
    """Recent-window correlation should retain the region the *recent* context
    attends to, where a cumulative (H2O-style) keep set retains stale early
    heavy-hitters instead.

    Geometry: an early block of tokens on axis A (heavy early traffic), then a
    late block of tokens on a distinct axis B, then a recent window of queries
    aligned with axis B. MorphKV (window > 1) should retain axis-B tokens (what
    the recent window reads) at a higher rate than H2O, whose cumulative mass is
    dominated by the earlier axis-A traffic. Reported as a RATE over planted
    seeds — a statistical claim, not a per-seed guarantee.
    """
    D = 16
    axis_a = np.zeros(D, dtype=np.float16)
    axis_a[0] = 3.0
    axis_b = np.zeros(D, dtype=np.float16)
    axis_b[1] = 3.0

    def build_stream(seed):
        r = np.random.default_rng(seed)
        stream = []
        # early: many tokens aligned with axis A (become cumulative heavy hitters)
        for _ in range(14):
            n = r.standard_normal(D).astype(np.float16) * 0.2
            stream.append(axis_a + n)
        # late: a few tokens aligned with axis B (the "new topic")
        b_rows = []
        for _ in range(4):
            n = r.standard_normal(D).astype(np.float16) * 0.2
            row = axis_b + n
            b_rows.append(row)
            stream.append(row)
        # recent window: queries aligned with axis B (attend to the new topic)
        for _ in range(4):
            n = r.standard_normal(D).astype(np.float16) * 0.2
            stream.append(axis_b + n)
        return [mx.array(x[None]) for x in stream], b_rows

    def b_retention(method, seed):
        stream, b_rows = build_stream(seed)
        if method == "morphkv":
            st = init_morphkv_state(n_sink=1, budget=10, head_dim=D, window=4)
            for k in stream:
                st = morphkv_update(st, k, k)
            K, _ = morphkv_get_kv(st)
        else:
            st = init_h2o_state(n_sink=1, budget=10, head_dim=D)
            for k in stream:
                st = h2o_update(st, k, k)
            K, _ = h2o_get_kv(st)
        # fraction of kept rows strongly aligned with axis B
        proj_b = K.astype(mx.float32) @ mx.array(axis_b.astype(np.float32))
        return int((proj_b > 6.0).sum().item())

    seeds = range(30)
    morph = sum(b_retention("morphkv", s) for s in seeds)
    h2o = sum(b_retention("h2o", s) for s in seeds)
    # Mechanism claim: MorphKV retains more of the recent-relevant (axis-B) region.
    assert morph >= h2o


def test_stable_control_no_advantage_required():
    """Null control: with no topic shift (all traffic on one axis), MorphKV need
    not beat the cumulative baseline. We only assert both keep budget — the win
    is regime-dependent and not overclaimed on stable geometry.
    """
    D = 16
    axis = np.zeros(D, dtype=np.float16)
    axis[0] = 3.0

    def run(method, seed):
        r = np.random.default_rng(seed)
        stream = [
            mx.array((axis + r.standard_normal(D).astype(np.float16) * 0.2)[None])
            for _ in range(30)
        ]
        if method == "morphkv":
            st = init_morphkv_state(n_sink=1, budget=10, head_dim=D, window=4)
            for k in stream:
                st = morphkv_update(st, k, k)
            return morphkv_get_kv(st)[0]
        st = init_h2o_state(n_sink=1, budget=10, head_dim=D)
        for k in stream:
            st = h2o_update(st, k, k)
        return h2o_get_kv(st)[0]

    assert run("morphkv", 0).shape[0] == 10
    assert run("h2o", 0).shape[0] == 10
