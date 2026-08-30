---
id: cacheroute-api
title: CacheRoute API
sidebar_label: CacheRoute
slug: /api/cacheroute-api
---

# CacheRoute API

`veloxquant_mlx.routing`

Rate-aware session admission and shard placement for a shared KV block pool, adapted from ["CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving"](https://arxiv.org/abs/2608.19677) (Cheng, Meta, Aug 2026).

---

## SessionRate

```python
from veloxquant_mlx import SessionRate
```

```python
@dataclass(frozen=True)
class SessionRate:
    owner: int
    rate: float
```

A session's estimated request rate.

| Field | Type | Description |
|---|---|---|
| `owner` | `int` | Opaque session/owner id, matching `BlockPoolAllocator` owner ids. |
| `rate` | `float` | Estimated request rate (`lambda_b` in the paper). Any consistent unit works as long as `qcap` uses the same unit. |

Raises `QuantizerConfigError` at construction time if `rate < 0`.

---

## RoutingTable

```python
from veloxquant_mlx import RoutingTable
```

A fixed admission-and-placement plan for one control interval, returned by `CacheRoutePlanner.plan()`.

| Field | Type | Description |
|---|---|---|
| `shards` | `dict[int, tuple[int, ...]]` | Maps each admitted owner to the tuple of shard ids it may use. |
| `expected_load` | `list[float]` | Per-shard expected load after placement, indexed by shard id. Diagnostic only — not enforced at dispatch time. |
| `n_shards` | `int` | Total number of shards this plan was computed for. |
| `admitted_rate_total` | `float` | Sum of `rate` over admitted owners. |
| `unadmitted_rate_total` | `float` | Sum of `rate` over owners considered but not admitted (cold-tail traffic). |

### Methods

**`is_admitted(owner: int) -> bool`** — Whether `owner` has a warm-set placement in this plan.

**`preferred_shard(owner: int, fallback: int = 0) -> int`** — The single best shard for `owner`, or `fallback` if not admitted. For an owner with `kb > 1` shards, returns the currently least-loaded of its assigned shards (ties broken by lowest shard id).

**`imbalance() -> float`** — Ratio of the most-loaded shard's expected load to the mean (1.0 is perfectly balanced). Returns `1.0` for zero total load.

---

## CacheRoutePlanner

```python
from veloxquant_mlx import CacheRoutePlanner
```

```python
CacheRoutePlanner(
    n_shards: int,
    qcap: float,
    warm_slots_per_shard: float,
)
```

Builds a periodic `RoutingTable` from measured session rates: sizes each admitted session's shard count from its rate relative to `qcap` (Eq. 1 in the paper), admits sessions in decreasing-rate order up to the warm-slot budget, then places admitted sessions with longest-processing-time-first (LPT) list scheduling.

| Parameter | Type | Description |
|---|---|---|
| `n_shards` | `int` | Number of pool shards to plan across (`R` in the paper). Use `1` if the pool isn't partitioned. Must be `>= 1`. |
| `qcap` | `float` | Load cap per shard, in the same rate unit as `SessionRate.rate`. Calibrate from a single shard's measured latency/load knee. Must be `> 0`. |
| `warm_slots_per_shard` | `float` | Warm-set capacity per shard (`W` in the paper). Total admission budget is `n_shards * warm_slots_per_shard`. Must be `> 0`. |

Raises `QuantizerConfigError` at construction time if any bound is violated.

### Methods

**`plan(rates: list[SessionRate]) -> RoutingTable`** — Computes a fresh routing table from the given session rates for the next control interval. Stateless between calls: always recomputes from scratch, so call once per control interval rather than per request, and hold onto the result for the whole interval.

```python
planner = CacheRoutePlanner(n_shards=4, qcap=50.0, warm_slots_per_shard=200.0)
table = planner.plan([SessionRate(owner=1, rate=10.0), SessionRate(owner=2, rate=80.0)])
```

---

## RateEstimator

```python
from veloxquant_mlx import RateEstimator
```

```python
RateEstimator(
    half_life: float = 20.0,
    window: int = 256,
)
```

Tracks a per-owner exponentially-weighted request rate estimate. Optional — any source of `{owner: rate}` works as a `CacheRoutePlanner.plan()` input, including offline aggregate statistics.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `half_life` | `float` | `20.0` | Number of `record()` calls for an owner's rate contribution to decay by half. Must be `> 0`. |
| `window` | `int` | `256` | Maximum timestamps retained per owner, bounding memory for long-lived owners. Must be `>= 1`. |

Raises `QuantizerConfigError` at construction time if either bound is violated.

### Methods

**`record(owner: int) -> None`** — Record one request arrival for `owner`.

**`rate(owner: int) -> float`** — Current smoothed rate estimate for `owner` (`0.0` if never seen).

**`rates() -> list[SessionRate]`** — Snapshot every tracked owner's current rate.

**`forget(owner: int) -> None`** — Drop all tracked state for `owner` (e.g. on session close).

---

## Good to know

- **Plan, don't guarantee.** `CacheRoutePlanner` and `RoutingTable` only decide admission and preferred placement — they never allocate, evict, or move a block. Pair them with `BlockPoolAllocator` / `PooledKVCache` for actual bookkeeping.
- **No residency guarantee.** There's no analytic guarantee an admitted session's blocks survive to its next request — the source paper found analytic residency prediction unreliable and recommends measuring served hit rate directly instead.
- **Deterministic given its input.** `plan()` always produces the same `RoutingTable` for the same `rates` list — there's no randomness or hidden state between calls.
