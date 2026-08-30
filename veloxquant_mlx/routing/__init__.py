"""Rate-aware session admission and shard placement for a shared KV pool (issue #278).

Adapts "CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM
Serving" (Cheng, Meta, Aug 2026, arXiv:2608.19677) — originally a routing
plan across many servers in a distributed serving fleet — to a single
VeloxQuant-MLX process: many concurrent session owners sharing one
:class:`~veloxquant_mlx.memory.block_pool.BlockPoolAllocator`, optionally
partitioned into shards.

Core pieces:

- :class:`RateEstimator` — tracks a smoothed per-owner request rate from
  live traffic (or skip this and supply your own ``{owner: rate}`` source,
  e.g. offline aggregate statistics).
- :class:`CacheRoutePlanner` — builds a :class:`RoutingTable` once per
  control interval: sizes each hot owner's shard count from its rate
  (Eq. 1 in the paper), admits owners into a warm set up to a slot budget
  in decreasing-rate order, and places admitted owners with
  longest-processing-time-first (LPT) list scheduling so shard load stays
  balanced (Algorithm 1).
- :class:`RoutingTable` — the resulting fixed plan: which owners are
  admitted, which shard(s) each should prefer, and the expected load per
  shard the plan was built from.

This module only plans admission and placement — it does not allocate or
evict any block itself. Combine it with
:class:`~veloxquant_mlx.memory.block_pool.BlockPoolAllocator` /
:class:`~veloxquant_mlx.memory.pooled_cache.PooledKVCache`: use
``RoutingTable.is_admitted(owner)`` to decide which owners' blocks to
protect from eviction under pressure, and ``RoutingTable.preferred_shard``
to pick which pool partition a session's blocks should live in.

Typical usage::

    from veloxquant_mlx.routing import CacheRoutePlanner, RateEstimator

    estimator = RateEstimator()
    planner = CacheRoutePlanner(n_shards=4, qcap=50.0, warm_slots_per_shard=200.0)

    # on every incoming request:
    estimator.record(owner=session_id)

    # once per control interval:
    table = planner.plan(estimator.rates())
    shard = table.preferred_shard(session_id, fallback=session_id % planner.n_shards)
    if table.is_admitted(session_id):
        ...  # protect this session's blocks from eviction
"""

from __future__ import annotations

from veloxquant_mlx.routing.cacheroute import (
    CacheRoutePlanner,
    RateEstimator,
    RoutingTable,
    SessionRate,
)

__all__ = [
    "SessionRate",
    "RoutingTable",
    "CacheRoutePlanner",
    "RateEstimator",
]
