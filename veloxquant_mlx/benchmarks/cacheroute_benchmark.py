"""CacheRoute semi-synthetic admission/placement benchmark (issue #278).

The paper this module adapts (Cheng, "CacheRoute: Planned Prefix-Affinity
Routing for Large-Scale LLM Serving", Meta, Aug 2026, arXiv:2608.19677)
evaluates its routing plan against a semi-synthetic aggregate workload with
a heavily skewed per-key rate distribution (Gini coefficient 0.756: ~4% of
keys account for 47% of requests) and reports served KV-cache hit rate and
per-destination load imbalance. This module reproduces that evaluation
shape — a synthetic Zipf-distributed session-rate population, replayed as a
discrete-event arrival stream against a simulated shared pool of warm
slots — to compare :class:`~veloxquant_mlx.routing.cacheroute.CacheRoutePlanner`
against two baselines from the same paper's comparison set:

  * ``lru``: cache-blind — no admission plan; every shard's warm set is
    whichever sessions arrived most recently, evicted LRU-first once full
    (the paper's "Flat-LB" analogue for a single-process pool: no affinity
    planning at all).
  * ``sticky_hash``: fixed affinity — every session is pinned to
    ``hash(owner) % n_shards`` for the whole run, with no load awareness
    (the paper's "Sticky" consistent-hashing baseline).

All three policies share the same arrival stream and the same total warm
budget (``n_shards * warm_slots_per_shard`` sessions resident at once), so
the comparison isolates the routing/admission policy, not cache size. This
is a discrete-event simulation over synthetic session arrivals, not a real
inference engine — no tokens are generated and no model is loaded; for that,
see ``benchmark_scripts/benchmark_cacheroute.py``, which drives the same
idea through a real local model.

Usage::

    python -m veloxquant_mlx.benchmarks.cacheroute_benchmark
    python -m veloxquant_mlx.benchmarks.cacheroute_benchmark --gini 0.756 --n-sessions 2000
    python -m veloxquant_mlx.benchmarks.cacheroute_benchmark --json-out figures/cacheroute/results.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parents[2] / "figures" / "cacheroute"

DEFAULT_N_SESSIONS = 2000
DEFAULT_N_SHARDS = 8
DEFAULT_WARM_SLOTS_PER_SHARD = 40
DEFAULT_N_ARRIVALS = 40_000
DEFAULT_ZIPF_S = 1.1  # skew exponent; higher = more concentrated, mimics the paper's Gini~0.76
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic workload generation
# ---------------------------------------------------------------------------


def _zipf_weights(n: int, s: float) -> list[float]:
    """Un-normalized Zipf(s) weights for ``n`` ranked sessions (rank 1 hottest)."""
    return [1.0 / ((rank + 1) ** s) for rank in range(n)]


def gini_coefficient(weights: list[float]) -> float:
    """Gini coefficient of a rate/weight distribution, for reporting workload skew."""
    if not weights:
        return 0.0
    values = sorted(weights)
    n = len(values)
    cum = 0.0
    weighted_sum = 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    for v in values:
        cum += v
        weighted_sum += cum
    return (n + 1 - 2 * weighted_sum / total) / n


@dataclass
class SyntheticWorkload:
    """A generated session population plus a request arrival stream.

    Attributes:
        n_sessions: Number of distinct session ids.
        session_rates: True underlying rate (arrivals per unit time) used
            to generate arrivals, indexed by session id.
        arrivals: Ordered stream of session ids, one per arrival event.
        gini: Gini coefficient of ``session_rates`` (workload skew summary,
            comparable to the paper's Table 6).
    """

    n_sessions: int
    session_rates: list[float]
    arrivals: list[int]
    gini: float


def generate_workload(
    n_sessions: int = DEFAULT_N_SESSIONS,
    n_arrivals: int = DEFAULT_N_ARRIVALS,
    zipf_s: float = DEFAULT_ZIPF_S,
    seed: int = DEFAULT_SEED,
) -> SyntheticWorkload:
    """Generate a Zipf-skewed session population and an interleaved arrival stream.

    Session ids are shuffled after ranking so "hot" sessions aren't
    trivially the low-numbered ids — a routing/admission policy has to
    infer heat from arrival history, not from the id itself.
    """
    rng = random.Random(seed)
    weights = _zipf_weights(n_sessions, zipf_s)
    order = list(range(n_sessions))
    rng.shuffle(order)
    session_rates = [0.0] * n_sessions
    for rank, owner in enumerate(order):
        session_rates[owner] = weights[rank]

    arrivals = rng.choices(range(n_sessions), weights=session_rates, k=n_arrivals)
    return SyntheticWorkload(
        n_sessions=n_sessions,
        session_rates=session_rates,
        arrivals=arrivals,
        gini=gini_coefficient(session_rates),
    )


# ---------------------------------------------------------------------------
# Policies under simulation
# ---------------------------------------------------------------------------


class _LruShardPool:
    """Cache-blind baseline: LRU eviction, no affinity, no admission plan."""

    def __init__(self, n_shards: int, warm_slots_per_shard: int) -> None:
        self.n_shards = n_shards
        self.capacity = n_shards * warm_slots_per_shard
        self._resident: dict[int, int] = {}  # owner -> last-seen arrival index
        self._shard_of: dict[int, int] = {}
        self._next_shard = 0
        self._arrivals_per_shard = [0] * n_shards

    def on_arrival(self, owner: int, t: int) -> tuple[bool, int]:
        """Returns (was_hit, shard_used)."""
        if owner in self._resident:
            self._resident[owner] = t
            shard = self._shard_of[owner]
            self._arrivals_per_shard[shard] += 1
            return True, shard

        if len(self._resident) >= self.capacity:
            evict_owner = min(self._resident, key=lambda o: self._resident[o])
            del self._resident[evict_owner]
            shard = self._shard_of.pop(evict_owner)
        else:
            shard = self._next_shard
            self._next_shard = (self._next_shard + 1) % self.n_shards

        self._resident[owner] = t
        self._shard_of[owner] = shard
        self._arrivals_per_shard[shard] += 1
        return False, shard

    def load_per_shard(self) -> list[int]:
        """Arrival count actually routed to each shard over the whole replay.

        This is the traffic/queue-load signal the paper's imbalance metric
        is about — not the resident-slot count, which stays even by
        construction under round-robin admission regardless of how skewed
        the arrival stream is.
        """
        return list(self._arrivals_per_shard)


class _StickyHashPool:
    """Fixed-affinity baseline: owner always maps to hash(owner) % n_shards.

    Admission is still capacity-limited per shard (LRU within the shard the
    hash pins it to), so a hot shard's own sessions compete only among
    themselves — modeling the paper's "busiest hash bucket inherits the
    workload skew" failure mode (Section 2.2).
    """

    def __init__(self, n_shards: int, warm_slots_per_shard: int) -> None:
        self.n_shards = n_shards
        self.warm_slots_per_shard = warm_slots_per_shard
        self._resident: dict[int, dict[int, int]] = {s: {} for s in range(n_shards)}
        self._arrivals_per_shard = [0] * n_shards

    def on_arrival(self, owner: int, t: int) -> tuple[bool, int]:
        shard = owner % self.n_shards
        self._arrivals_per_shard[shard] += 1
        bucket = self._resident[shard]
        if owner in bucket:
            bucket[owner] = t
            return True, shard

        if len(bucket) >= self.warm_slots_per_shard:
            evict_owner = min(bucket, key=lambda o: bucket[o])
            del bucket[evict_owner]

        bucket[owner] = t
        return False, shard

    def load_per_shard(self) -> list[int]:
        """Arrival count actually routed to each shard — see _LruShardPool.load_per_shard."""
        return list(self._arrivals_per_shard)


def _run_cacheroute(
    workload: SyntheticWorkload,
    n_shards: int,
    warm_slots_per_shard: int,
    qcap: float,
    replan_interval: int,
) -> tuple[int, list[float]]:
    """CacheRoute needs its own loop: admitted-session hits are membership-based, not LRU."""
    from veloxquant_mlx.routing.cacheroute import CacheRoutePlanner, RateEstimator

    planner = CacheRoutePlanner(n_shards=n_shards, qcap=qcap, warm_slots_per_shard=warm_slots_per_shard)
    estimator = RateEstimator(half_life=replan_interval / 2)
    table = planner.plan([])
    admitted_ever_seen: set[int] = set()
    cold_tail: dict[int, int] = {}
    cold_capacity = max(1, n_shards)

    hits = 0
    for t, owner in enumerate(workload.arrivals):
        estimator.record(owner)
        if t % replan_interval == 0:
            table = planner.plan(estimator.rates())

        if table.is_admitted(owner):
            if owner in admitted_ever_seen or owner in cold_tail:
                hits += 1
            admitted_ever_seen.add(owner)
            cold_tail.pop(owner, None)
            continue

        if owner in cold_tail:
            hits += 1
            cold_tail[owner] = t
        else:
            if len(cold_tail) >= cold_capacity:
                evict = min(cold_tail, key=lambda o: cold_tail[o])
                del cold_tail[evict]
            cold_tail[owner] = t

    return hits, list(table.expected_load)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class PolicyResult:
    """Aggregated outcome for one policy over one workload replay."""

    policy: str
    n_arrivals: int
    hits: int
    hit_rate: float
    load_per_shard: list[float]
    imbalance: float


def _imbalance(load: list[float]) -> float:
    if not load:
        return 1.0
    mean_load = sum(load) / len(load)
    if mean_load <= 0:
        return 1.0
    return max(load) / mean_load


def run_benchmark(
    n_sessions: int = DEFAULT_N_SESSIONS,
    n_arrivals: int = DEFAULT_N_ARRIVALS,
    n_shards: int = DEFAULT_N_SHARDS,
    warm_slots_per_shard: int = DEFAULT_WARM_SLOTS_PER_SHARD,
    zipf_s: float = DEFAULT_ZIPF_S,
    qcap: float | None = None,
    replan_interval: int = 500,
    seed: int = DEFAULT_SEED,
) -> dict[str, PolicyResult]:
    """Replay the same synthetic arrival stream through all three policies.

    Args:
        qcap: CacheRoute's per-shard load cap. Defaults to a value that
            keeps the hottest synthetic session's ``kb`` at 1 (mirrors the
            paper's primary 70B workload, where every key uses exactly one
            destination) unless overridden to exercise replication.
    """
    workload = generate_workload(n_sessions, n_arrivals, zipf_s, seed)
    if qcap is None:
        qcap = max(workload.session_rates) * n_arrivals + 1.0  # never triggers kb > 1 by default

    results: dict[str, PolicyResult] = {}

    lru = _LruShardPool(n_shards, warm_slots_per_shard)
    hits = 0
    for t, owner in enumerate(workload.arrivals):
        was_hit, _ = lru.on_arrival(owner, t)
        hits += was_hit
    results["lru"] = PolicyResult(
        policy="lru",
        n_arrivals=n_arrivals,
        hits=hits,
        hit_rate=hits / n_arrivals,
        load_per_shard=[float(x) for x in lru.load_per_shard()],
        imbalance=_imbalance(lru.load_per_shard()),
    )

    sticky = _StickyHashPool(n_shards, warm_slots_per_shard)
    hits = 0
    for t, owner in enumerate(workload.arrivals):
        was_hit, _ = sticky.on_arrival(owner, t)
        hits += was_hit
    results["sticky_hash"] = PolicyResult(
        policy="sticky_hash",
        n_arrivals=n_arrivals,
        hits=hits,
        hit_rate=hits / n_arrivals,
        load_per_shard=[float(x) for x in sticky.load_per_shard()],
        imbalance=_imbalance(sticky.load_per_shard()),
    )

    cr_hits, cr_load = _run_cacheroute(workload, n_shards, warm_slots_per_shard, qcap, replan_interval)
    results["cacheroute"] = PolicyResult(
        policy="cacheroute",
        n_arrivals=n_arrivals,
        hits=cr_hits,
        hit_rate=cr_hits / n_arrivals,
        load_per_shard=cr_load,
        imbalance=_imbalance(cr_load),
    )

    return results


def format_summary_table(results: dict[str, PolicyResult], gini: float) -> str:
    header = f"{'Policy':<14}{'Hit rate':>10}{'Imbalance':>12}{'Max shard load':>16}"
    sep = "-" * len(header)
    lines = [f"Workload Gini coefficient: {gini:.3f}", header, sep]
    for name, r in results.items():
        max_load = max(r.load_per_shard) if r.load_per_shard else 0.0
        lines.append(f"{name:<14}{r.hit_rate:>10.1%}{r.imbalance:>12.2f}x{max_load:>15.1f} ")
    return "\n".join(lines)


def results_to_json(results: dict[str, PolicyResult], gini: float) -> dict[str, Any]:
    return {"gini": gini, "policies": {name: asdict(r) for name, r in results.items()}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CacheRoute semi-synthetic admission/placement benchmark")
    parser.add_argument("--n-sessions", type=int, default=DEFAULT_N_SESSIONS)
    parser.add_argument("--n-arrivals", type=int, default=DEFAULT_N_ARRIVALS)
    parser.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS)
    parser.add_argument("--warm-slots-per-shard", type=int, default=DEFAULT_WARM_SLOTS_PER_SHARD)
    parser.add_argument("--zipf-s", type=float, default=DEFAULT_ZIPF_S)
    parser.add_argument("--qcap", type=float, default=None)
    parser.add_argument("--replan-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    workload = generate_workload(args.n_sessions, args.n_arrivals, args.zipf_s, args.seed)
    results = run_benchmark(
        n_sessions=args.n_sessions,
        n_arrivals=args.n_arrivals,
        n_shards=args.n_shards,
        warm_slots_per_shard=args.warm_slots_per_shard,
        zipf_s=args.zipf_s,
        qcap=args.qcap,
        replan_interval=args.replan_interval,
        seed=args.seed,
    )

    print()
    print(format_summary_table(results, workload.gini))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results_to_json(results, workload.gini), f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
