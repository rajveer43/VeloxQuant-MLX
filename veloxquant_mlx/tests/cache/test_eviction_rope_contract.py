"""Cross-cache RoPE contract for eviction caches (#171, #174, #183, #611).

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

Every cache here satisfies the contract the same way (#609): eviction only
drops a row. Q-Filters / TOVA / L2Norm / H2O / CurDKV all PRESERVE the
original position and rotation of every survivor — none of them renumber or
re-rotate. An earlier version of H2O (and CurDKV) additionally renumbered
survivors to a gap-free layout and re-rotated their keys to match, on the
theory that the model's position bookkeeping assumed a contiguous cache.
That theory was wrong given the fix directly above (``offset`` == true
position, not row count): renumbering a survivor changes its true distance
from every future query, corrupting the RoPE relative angle the offset fix
was meant to protect. See ``veloxquant_mlx/quantizers/h2o.py`` and
``curdkv.py`` module docstrings for the full incident writeup.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

# Large enough that every arm has evictable room: H2O protects `h2o_grace`
# (default 16) trailing rows on top of its sinks, and a budget that leaves
# nothing evictable is rejected at construction.
BUDGET = 64
HEAD_DIM = 32

# (id, config kwargs): benchmark arms plus the #611 regression methods.
ARMS = [
    ("qfilters", {"method": "qfilters", "qfilters_budget": BUDGET, "qfilters_n_sink": 4}),
    ("h2o", {"method": "h2o", "h2o_budget": BUDGET, "h2o_n_sink": 4}),
    ("tova", {"method": "tova", "tova_budget": BUDGET, "tova_n_sink": 4}),
    ("knorm", {"method": "knorm", "knorm_budget": BUDGET, "knorm_n_sink": 4}),
    ("curdkv", {"method": "curdkv", "curdkv_budget": BUDGET, "curdkv_n_sink": 4}),
    ("morphkv", {"method": "morphkv", "morphkv_budget": BUDGET, "morphkv_n_sink": 4}),
    ("kvzip", {"method": "kvzip", "kvzip_budget": BUDGET, "kvzip_n_sink": 4}),
    ("nestedkv", {"method": "nestedkv", "nestedkv_budget": BUDGET, "nestedkv_n_sink": 4}),
]


def _stored_or_returned_count(name: str, cache, k_out) -> int:
    """Budgets constrain retained storage, not current attention inputs (#610)."""
    return cache.keys.shape[2]


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
    assert _stored_or_returned_count(name, cache, k_out) <= BUDGET < S_pre

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

    n_small = _stored_or_returned_count(name, small, ks)
    n_large = _stored_or_returned_count(name, large, kl)
    assert n_small != n_large, f"{name}: budgets did not change the retained count"
    assert small.offset == large.offset == S


@pytest.mark.parametrize("method", ["morphkv", "kvzip", "nestedkv"])
@pytest.mark.parametrize("batch,heads", [(1, 1), (2, 3)])
@pytest.mark.parametrize(
    "chunks",
    [[1] * 48, [32] + [1] * 16, [16, 16, 1, 7, 8]],
    ids=["decode", "prefill-decode", "chunked-prefill"],
)
def test_retained_keys_preserve_true_rope_positions(method, batch, heads, chunks):
    """Reproduce model call order and verify rotations, not just offset metadata.

    Value rows encode token IDs independently of cache position tracking.
    Every retained key must match the original token rotated at its absolute
    position, including tokens admitted after eviction. NestedKV compresses
    only its first multi-token prefill, so decode-only input need not evict.
    """
    budget = 12
    options = {f"{method}_budget": budget, f"{method}_n_sink": 2}
    if method == "morphkv":
        options["morphkv_window"] = 2
    cache = _make({"method": method, **options})
    rng = np.random.default_rng(611)
    total = sum(chunks)
    raw = mx.array(rng.standard_normal((batch, heads, total, HEAD_DIM)).astype(np.float32))
    rope = nn.RoPE(HEAD_DIM, traditional=False, base=10000.0)
    expected_keys = np.array(rope(raw, offset=0).astype(mx.float16))
    seen = 0
    evicted = False
    for length in chunks:
        # mlx_lm rotates new keys before update_and_fetch, using this offset.
        incoming = rope(raw[:, :, seen : seen + length], offset=cache.offset).astype(mx.float16)
        values = mx.broadcast_to(
            mx.arange(seen, seen + length)[None, None, :, None], incoming.shape
        ).astype(mx.float16)
        cache.update_and_fetch(incoming, values)
        seen += length
        kept_k, kept_v = cache.state[:2]
        ids = np.array(kept_v[..., 0]).astype(np.int32)
        reference = np.take_along_axis(expected_keys, ids[..., None], axis=2)
        np.testing.assert_allclose(np.array(kept_k), reference, atol=1e-3, rtol=0)
        assert np.all(ids < seen)
        evicted |= kept_k.shape[2] < seen

    assert cache.offset == total
    if method != "nestedkv" or max(chunks) > 1:
        assert evicted, "Regression must exercise eviction, not just append-only storage"
    assert np.any(ids >= budget), "Check rotations of rows admitted after reaching the budget"
