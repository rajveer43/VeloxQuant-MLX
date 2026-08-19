"""Cross-cache RoPE contract for eviction caches (#171, #174, #183).

``mlx_lm``'s attention module rotates BOTH the query and the incoming key with
``self.rope(x, offset=cache.offset)`` *before* ``update_and_fetch`` is called.
An eviction cache therefore cannot correct that rotation after the fact: the
only way it can be right is for ``cache.offset`` to equal the true absolute
token position at all times, never the number of rows the cache happens to
still be holding.

The base ``mlx_lm`` ``KVCache`` sets ``offset`` to the stored row count, which
is the same number right up until the first eviction and wrong forever after —
once ``n_kept`` pins at the budget, ``offset`` stops advancing while the true
position keeps climbing, and the drift grows without bound.

Each cache already tests this for itself. This module pins it as a *shared*
contract across every eviction cache used as a benchmark comparison arm, so a
cache cannot be added to those comparisons while silently drifting. That
matters for benchmark integrity specifically: an arm with a broken offset
measures position drift rather than eviction quality, which would make
whichever method is correct look good for the wrong reason (#183).

How each cache satisfies the contract differs, and both ways are valid:
  - Q-Filters / TOVA / L2Norm PRESERVE original positions (eviction drops rows
    but never renumbers), so stored keys already carry the right rotation and
    reporting the true offset is sufficient.
  - H2O RENUMBERS to a gap-free layout, so it additionally de-rotates and
    re-rotates survivors; ``h2o_rope_base`` must match the model's RoPE base
    for that correction to cancel.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

# Large enough that every arm has evictable room: H2O protects `h2o_grace`
# (default 16) trailing rows on top of its sinks, and a budget that leaves
# nothing evictable is rejected at construction.
BUDGET = 64
HEAD_DIM = 32

# (id, config kwargs) — every eviction cache used as a Q-Filters benchmark arm.
ARMS = [
    ("qfilters", dict(method="qfilters", qfilters_budget=BUDGET, qfilters_n_sink=4)),
    ("h2o", dict(method="h2o", h2o_budget=BUDGET, h2o_n_sink=4)),
    ("tova", dict(method="tova", tova_budget=BUDGET, tova_n_sink=4)),
    ("knorm", dict(method="knorm", knorm_budget=BUDGET, knorm_n_sink=4)),
]


def _kv(S, H=2, D=HEAD_DIM, seed=0):
    rng = np.random.default_rng(seed)
    return (
        mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16)),
        mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16)),
    )


def _make(cfg_kwargs):
    return KVCacheFactory.create(KVCacheConfig(head_dim=HEAD_DIM, **cfg_kwargs))


@pytest.mark.parametrize("name,cfg", ARMS, ids=[a[0] for a in ARMS])
def test_offset_never_drifts_from_true_position_during_decode(name, cfg) -> None:
    """Token-by-token decode well past the budget: offset == true position."""
    cache = _make(cfg)
    n_steps = 6 * BUDGET
    for t in range(n_steps):
        k, v = _kv(S=1, seed=t)
        cache.update_and_fetch(k, v)
        assert cache.offset == t + 1, (
            f"{name}: offset {cache.offset} != true position {t + 1} at step {t} — "
            "RoPE would rotate this token at the wrong position"
        )


@pytest.mark.parametrize("name,cfg", ARMS, ids=[a[0] for a in ARMS])
def test_offset_tracks_true_position_across_prefill_then_decode(name, cfg) -> None:
    """A long prefill (forcing eviction) followed by decode keeps offset exact.

    This is the shape the benchmarks actually run, and the case where a
    row-count offset diverges fastest: after prefill the cache holds ``budget``
    rows but has consumed ``S_pre`` positions.
    """
    S_pre, n_dec = 8 * BUDGET, 3 * BUDGET
    cache = _make(cfg)

    k, v = _kv(S=S_pre, seed=99)
    k_out, _ = cache.update_and_fetch(k, v)
    mx.eval(k_out)

    assert cache.offset == S_pre, f"{name}: offset stalled at {cache.offset} after prefill"
    # Eviction genuinely happened — otherwise this test proves nothing.
    assert k_out.shape[2] <= BUDGET < S_pre

    for t in range(n_dec):
        k1, v1 = _kv(S=1, seed=1000 + t)
        cache.update_and_fetch(k1, v1)
        assert cache.offset == S_pre + t + 1, (
            f"{name}: offset {cache.offset} != true position {S_pre + t + 1} "
            f"{t + 1} tokens into decode"
        )


@pytest.mark.parametrize("name,cfg", ARMS, ids=[a[0] for a in ARMS])
def test_offset_is_independent_of_retained_row_count(name, cfg) -> None:
    """The invariant that fails first: offset must decouple from ``n_kept``.

    Two caches fed the same number of positions at different budgets retain
    different row counts but must report the *same* offset. A cache reporting
    a row count would report the two budgets differently.
    """
    S = 8 * BUDGET
    small = _make(cfg)
    wide = dict(cfg)
    wide[next(kk for kk in wide if kk.endswith("_budget"))] = BUDGET * 4
    large = _make(wide)

    k, v = _kv(S=S, seed=7)
    ks, _ = small.update_and_fetch(k, v)
    kl, _ = large.update_and_fetch(k, v)
    mx.eval(ks, kl)

    assert ks.shape[2] != kl.shape[2], f"{name}: budgets did not change the retained count"
    assert small.offset == large.offset == S
