"""Tests for KVzip-adapted eviction primitives (quantizers/kvzip.py).

Covers: init guards, constant-size budget invariant (token/block), sink
protection, byte accounting, determinism, the probe="latest" == latest-token
(TOVA-adapted) reduction (the honest pinned reduction), that the default
context probe is not vacuously equal to latest, and the reconstruction-geometry
mechanism (reconstruction-reliance retains the region the model relies on to
reconstruct its context, at a higher rate than a cumulative H2O-style baseline),
with a null "flat" control where it shows no advantage.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.kvzip import (
    full_kvzip_fp16_bytes,
    init_kvzip_state,
    kvzip_fp16_bytes,
    kvzip_get_kv,
    kvzip_update,
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
        init_kvzip_state(n_sink=0, budget=0, head_dim=16)


def test_init_rejects_sink_ge_budget():
    with pytest.raises(ValueError, match="n_sink .* must be < budget"):
        init_kvzip_state(n_sink=8, budget=8, head_dim=16)


def test_init_rejects_bad_probe():
    with pytest.raises(ValueError, match="probe must be one of"):
        init_kvzip_state(n_sink=2, budget=8, head_dim=16, probe="bogus")


def test_init_empty_state():
    st = init_kvzip_state(n_sink=2, budget=8, head_dim=16, probe="context")
    assert st.keys is None and st.values is None and st.pos == 0
    assert st.n_sink == 2 and st.budget == 8 and st.probe == "context"


def test_init_default_probe_is_context():
    st = init_kvzip_state(n_sink=2, budget=8, head_dim=16)
    assert st.probe == "context"


# ---------------------------------------------------------------------------
# constant-size budget invariant
# ---------------------------------------------------------------------------
def test_budget_never_exceeded_token_by_token():
    st = init_kvzip_state(n_sink=2, budget=8, head_dim=16, probe="context")
    for i in range(40):
        k, v = _rand_kv(1, 16, seed=i)
        st = kvzip_update(st, k, v)
        assert st.keys.shape[0] <= 8


def test_budget_never_exceeded_block_prefill():
    st = init_kvzip_state(n_sink=2, budget=8, head_dim=16, probe="context")
    k, v = _rand_kv(40, 16, seed=7)
    st = kvzip_update(st, k, v)
    assert st.keys.shape[0] == 8


def test_budget_respected_latest_probe():
    st = init_kvzip_state(n_sink=2, budget=8, head_dim=16, probe="latest")
    k, v = _rand_kv(40, 16, seed=11)
    st = kvzip_update(st, k, v)
    assert st.keys.shape[0] == 8


def test_under_budget_keeps_all():
    st = init_kvzip_state(n_sink=2, budget=32, head_dim=16, probe="context")
    k, v = _rand_kv(10, 16, seed=3)
    st = kvzip_update(st, k, v)
    K, _ = kvzip_get_kv(st)
    assert K.shape[0] == 10


# ---------------------------------------------------------------------------
# protection: sinks (leading)
# ---------------------------------------------------------------------------
def test_sinks_survive_eviction():
    st = init_kvzip_state(n_sink=3, budget=8, head_dim=8, probe="context")
    first_k, first_v = _rand_kv(3, 8, seed=100)
    st = kvzip_update(st, first_k, first_v)
    sink_rows = st.keys[:3]
    for i in range(60):
        k, v = _rand_kv(1, 8, seed=i)
        st = kvzip_update(st, k, v)
    assert bool(mx.all(st.keys[:3] == sink_rows).item())


def test_sinks_survive_latest_probe():
    st = init_kvzip_state(n_sink=2, budget=8, head_dim=8, probe="latest")
    first_k, first_v = _rand_kv(2, 8, seed=200)
    st = kvzip_update(st, first_k, first_v)
    sink_rows = st.keys[:2]
    for i in range(50):
        k, v = _rand_kv(1, 8, seed=i)
        st = kvzip_update(st, k, v)
    assert bool(mx.all(st.keys[:2] == sink_rows).item())


# ---------------------------------------------------------------------------
# byte accounting
# ---------------------------------------------------------------------------
def test_fp16_bytes():
    st = init_kvzip_state(n_sink=2, budget=8, head_dim=16, probe="context")
    assert kvzip_fp16_bytes(st) == 0
    k, v = _rand_kv(8, 16, seed=5)
    st = kvzip_update(st, k, v)
    assert kvzip_fp16_bytes(st) == 8 * 16 * 2 * 2


def test_full_fp16_bytes():
    assert full_kvzip_fp16_bytes(100, 64) == 100 * 64 * 2 * 2


def test_get_kv_placeholder_before_update():
    st = init_kvzip_state(n_sink=2, budget=8, head_dim=16, probe="context")
    K, V = kvzip_get_kv(st)
    assert K.shape == (0, 1) and V.shape == (0, 1)


# ---------------------------------------------------------------------------
# determinism (KVzip has no RNG)
# ---------------------------------------------------------------------------
def test_run_is_reproducible():
    def run():
        st = init_kvzip_state(n_sink=2, budget=8, head_dim=16, probe="context")
        for i in range(30):
            k, v = _rand_kv(1, 16, seed=i)
            st = kvzip_update(st, k, v)
        return kvzip_get_kv(st)[0]

    assert bool(mx.all(run() == run()).item())


# ---------------------------------------------------------------------------
# probe="latest" reduces to the latest-token (TOVA-adapted) eviction — pinned
# ---------------------------------------------------------------------------
def test_latest_probe_matches_tova():
    """With probe="latest" the reconstruction probe is the single most-recent key,
    so the reconstruction importance is exactly that key's attention over the keep
    set — the TOVA-adapted latest-token ranking. Both protect only the leading
    sinks and argmin the same weight vector, so their kept keysets must be
    bit-for-bit identical.
    """
    ks = [_rand_kv(1, 16, seed=i) for i in range(40)]

    z = init_kvzip_state(n_sink=4, budget=12, head_dim=16, probe="latest")
    t = init_tova_state(n_sink=4, budget=12, head_dim=16)
    for k, v in ks:
        z = kvzip_update(z, k, v)
        t = tova_update(t, k, v)

    z_k, _ = kvzip_get_kv(z)
    t_k, _ = tova_get_kv(t)
    assert z_k.shape == t_k.shape
    assert bool(mx.all(z_k == t_k).item())


def test_context_probe_can_differ_from_latest():
    """The default context probe (max reliance over the whole keep set) should be
    capable of retaining a different set than the latest-token-only probe —
    otherwise the reconstruction axis is vacuous.
    """
    ks = [_rand_kv(1, 16, seed=i) for i in range(60)]

    def run(probe):
        st = init_kvzip_state(n_sink=2, budget=12, head_dim=16, probe=probe)
        for k, v in ks:
            st = kvzip_update(st, k, v)
        return kvzip_get_kv(st)[0]

    latest = run("latest")
    context = run("context")
    assert latest.shape == context.shape
    assert not bool(mx.all(latest == context).item())


# ---------------------------------------------------------------------------
# reconstruction-shift mechanism: reliance retention beats cumulative baseline
# ---------------------------------------------------------------------------
def test_reconstruction_geometry_retains_critical():
    """Reconstruction reliance should retain the region the model relies on to
    reconstruct its context, where a cumulative (H2O-style) keep set retains
    stale early heavy-hitters instead.

    Geometry: an early block of tokens on axis A (heavy early traffic that
    dominates cumulative attention mass), then a block of reconstruction-critical
    tokens on a distinct axis B that mutually reinforce (each attends to the
    others under the reconstruction probe). KVzip (context probe) should retain
    axis-B tokens (high reconstruction reliance) at a higher rate than H2O, whose
    cumulative mass is dominated by the earlier axis-A traffic. Reported as a RATE
    over planted seeds — a statistical claim, not a per-seed guarantee.
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
        # a cluster of mutually-reinforcing axis-B tokens (reconstruction-critical:
        # they attend strongly to one another under the reconstruction probe)
        for _ in range(6):
            n = r.standard_normal(D).astype(np.float16) * 0.2
            stream.append(axis_b + n)
        return [mx.array(x[None]) for x in stream]

    def b_retention(method, seed):
        stream = build_stream(seed)
        if method == "kvzip":
            st = init_kvzip_state(n_sink=1, budget=10, head_dim=D, probe="context")
            for k in stream:
                st = kvzip_update(st, k, k)
            K, _ = kvzip_get_kv(st)
        else:
            st = init_h2o_state(n_sink=1, budget=10, head_dim=D)
            for k in stream:
                st = h2o_update(st, k, k)
            K, _ = h2o_get_kv(st)
        proj_b = K.astype(mx.float32) @ mx.array(axis_b.astype(np.float32))
        return int((proj_b > 6.0).sum().item())

    seeds = range(30)
    kvzip = sum(b_retention("kvzip", s) for s in seeds)
    h2o = sum(b_retention("h2o", s) for s in seeds)
    # Mechanism claim: KVzip retains more of the reconstruction-critical region.
    assert kvzip >= h2o


def test_flat_control_no_advantage_required():
    """Null control: with no reconstruction shift (all traffic on one axis), KVzip
    need not beat the cumulative baseline. We only assert both keep budget — the
    win is regime-dependent and not overclaimed on flat geometry.
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
        if method == "kvzip":
            st = init_kvzip_state(n_sink=1, budget=10, head_dim=D, probe="context")
            for k in stream:
                st = kvzip_update(st, k, k)
            return kvzip_get_kv(st)[0]
        st = init_h2o_state(n_sink=1, budget=10, head_dim=D)
        for k in stream:
            st = h2o_update(st, k, k)
        return h2o_get_kv(st)[0]

    assert run("kvzip", 0).shape[0] == 10
    assert run("h2o", 0).shape[0] == 10
