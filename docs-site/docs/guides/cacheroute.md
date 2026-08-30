---
id: cacheroute
title: CacheRoute — Rate-Aware Session Admission and Placement
sidebar_label: CacheRoute
slug: /guides/cacheroute
---

# CacheRoute: rate-aware session admission and shard placement

Multi-tenant serving shares one KV block pool across many concurrent conversation sessions. A plain LRU pool disperses which sessions stay warm somewhat arbitrarily under pressure, and a naive fixed (hash-based) assignment of sessions to pool shards can let one shard inherit far more traffic than the others even though every shard has the same capacity.

CacheRoute adapts the routing plan from ["CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving"](https://arxiv.org/abs/2608.19677) (Cheng, Meta, Aug 2026) — originally a plan for routing requests across many *servers* in a distributed fleet — to a single VeloxQuant-MLX process with one or more pool *shards*: it periodically computes which sessions are worth keeping warm and which shard each should prefer, based on measured request rate, instead of reacting to cache/load state per request.

```python
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
```

## What it maps from the paper

| Paper concept | This module |
|---|---|
| Business key / recurring request key | `owner` — a session id, matching `BlockPoolAllocator` owner ids |
| Destination (a routable server) | A pool *shard* — a caller-defined partition of the block pool (e.g. a hash range, or `1` shard if the pool isn't partitioned) |
| Warm-prefix slot | A session's admitted place in the warm set, sized in rate units (equal-slot admission, Section 3.1 of the paper) |
| Routing table `T(b)` | `RoutingTable`, recomputed periodically by `CacheRoutePlanner.plan`, not per-request |

CacheRoute only **plans** admission and placement — it doesn't allocate, evict, or move any block itself. Pair it with [`BlockPoolAllocator`](/guides/mlx-lm-integration) / `PooledKVCache`: use `RoutingTable.is_admitted(owner)` to decide which sessions' blocks to protect from eviction under pressure, and `RoutingTable.preferred_shard()` to pick which pool partition a session's blocks should live in.

## The mechanism

1. **Load-based assignment count.** A session with observed rate `λ` needs `kb = max(1, ceil(λ / qcap))` shards to keep its expected load per shard at or below `qcap` (Eq. 1 in the paper). Most workloads never exceed `qcap` on a single session, so `kb = 1` almost everywhere — replication only kicks in for a session hot enough to overload one shard on its own.
2. **Warm-set admission.** Sessions are considered in decreasing-rate order and admitted while the cumulative `kb` stays within the slot budget (`n_shards * warm_slots_per_shard`). A session that doesn't fit falls back to a plain cold-tail path (e.g. LRU) — CacheRoute never guarantees residency for unadmitted traffic.
3. **LPT placement.** Admitted sessions are assigned to shards using longest-processing-time-first list scheduling: process sessions in decreasing rate, always placing on the currently least-loaded eligible shard set. This is what keeps shard load balanced even under a heavily skewed session population.

## Choosing `qcap` and `warm_slots_per_shard`

- **`qcap`** should reflect a single shard's real load ceiling — e.g. a request rate near where that shard's own latency starts to climb. The paper calibrates this from a measured latency/load knee; this module does not calibrate it for you.
- **`warm_slots_per_shard`** should match your pool's actual physical capacity for that shard, in the same rate units as `qcap`. If the plan admits more sessions than the pool can physically hold resident, physical eviction pressure will silently override the plan — size them consistently.

## Periodic replanning, not per-request

Call `planner.plan()` once per control interval and hold onto the `RoutingTable` for the whole interval — LPT re-sorts from scratch every call, so replanning too often churns most sessions' shard assignments even when only a few rates actually moved, each churn forcing a cold restart for that session. This mirrors the tradeoff the paper measures directly: a stale table loses a little hit rate/latency to drift, while a freshly recomputed one pays a rewarming cost from unnecessary reassignment. There's no built-in hysteresis — if you need it, diff two `RoutingTable`s yourself before switching.

## `RateEstimator`: an optional rate source

If you don't already track per-session request rates, `RateEstimator` provides a minimal exponentially-weighted estimate:

```python
from veloxquant_mlx.routing import RateEstimator

estimator = RateEstimator(half_life=20.0)  # smoothing window, in number of record() calls
estimator.record(owner=session_id)         # call on every request
estimator.rate(session_id)                 # current smoothed estimate
estimator.rates()                          # -> list[SessionRate], feed straight into plan()
```

Any other source of `{owner: rate}` works too — including offline aggregate statistics, as in the paper's telemetry-derived workload.

## What doesn't generalize (inherited from the paper)

- **Equal-slot admission.** The warm-set budget assumes sessions have roughly uniform reusable-context size. Heterogeneous prefix sizes need a byte-aware admission model this module doesn't provide.
- **No residency guarantee.** CacheRoute plans placement; it does not reserve or guarantee that a block survives to the session's next turn. The paper found analytic residency prediction unreliable (its Appendix H) and recommends measuring served hit rate directly instead — see the benchmarks below.
- **Not always a win.** When there's little recoverable reuse to begin with, or shard load is already well balanced, affinity planning can *reduce* capacity relative to a cache-blind baseline (the paper's Section 5.3 negative regimes). Measure before enabling in production — a short shadow-replay comparing served hit rate and load against your current policy is the paper's own recommended gate.

## Benchmarks

- `python -m veloxquant_mlx.benchmarks.cacheroute_benchmark` — a fast, pure-Python semi-synthetic simulation (Zipf-skewed session population, tens of thousands of arrivals) comparing CacheRoute against LRU and sticky-hash baselines on served hit rate and shard load imbalance.
- `python benchmark_scripts/benchmark_cacheroute.py` — drives a real local MLX model through interleaved multi-turn sessions across pool shards, measuring the same metrics on actual prefill/decode traffic. See [Benchmarking](/guides/benchmarking) for the full harness this pairs with.

On the real-model harness, CacheRoute's hit rate can run a few points below a sticky-hash baseline (periodic replanning occasionally moves a session and pays a cold-start cost fixed affinity never incurs), while its shard-load imbalance is consistently and substantially better. This matches the paper's own framing: hit rate alone understates the benefit of balanced placement, which shows up as sustained throughput under load rather than as a raw hit-rate delta.
