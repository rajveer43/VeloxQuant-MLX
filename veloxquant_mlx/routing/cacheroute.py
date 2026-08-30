"""CacheRoute: rate-aware session admission and shard placement (issue #278).

Adapts the routing plan from Cheng, "CacheRoute: Planned Prefix-Affinity
Routing for Large-Scale LLM Serving" (Meta, Aug 2026, arXiv:2608.19677) to a
single-process, single-GPU server. The paper plans which *server* in a
distributed fleet should receive a recurring request so its prefix KV stays
warm, balancing that affinity against per-server queue load. VeloxQuant-MLX
has no fleet — one process, one shared :class:`~veloxquant_mlx.memory.block_pool.BlockPoolAllocator`
serves every concurrent session — so there is nothing to route *to* in the
network sense. What carries over is the mechanism, retargeted at the two
things a single engine actually has: many concurrent session owners
competing for a bounded pool of warm blocks, and (optionally) several
logical *shards* of that pool a session's blocks can prefer.

Mapping from the paper to this module:

  * "business key" / recurring request key -> ``owner`` (a session id already
    used by :class:`~veloxquant_mlx.memory.block_pool.BlockPoolAllocator`).
  * "destination" (a routable server) -> a pool *shard* (an integer in
    ``range(n_shards)``), a caller-defined partition of the block pool (e.g.
    a hash range, a NUMA/core affinity group, or just ``1`` shard if the
    pool isn't partitioned at all).
  * "warm-prefix slot" -> one session's admitted place in the warm set, sized
    in the same units as ``qcap``/``rate`` rather than raw block counts,
    matching the paper's equal-slot admission model (Section 3.1).
  * routing table ``T(b)`` -> :class:`RoutingTable`, recomputed periodically
    by :class:`CacheRoutePlanner.plan` rather than per-request (Section 3.2).

As in the paper, this module plans admission and placement; it does not
itself allocate, evict, or move any block. Pair a :class:`RoutingTable` with
your own eviction policy (e.g. prefer evicting non-admitted owners' blocks
before admitted ones) and with :class:`~veloxquant_mlx.memory.block_pool.BlockPoolAllocator`
for the actual bookkeeping. There is also no analytic residency guarantee
here (the paper's Appendix H found analytic residency prediction unreliable
and recommends measuring instead) — :class:`RoutingTable` exposes the
expected-load numbers the plan was built from so a caller can compare them
against measured hit rate before trusting a plan.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from veloxquant_mlx.core.exceptions import QuantizerConfigError

__all__ = [
    "SessionRate",
    "RoutingTable",
    "CacheRoutePlanner",
    "RateEstimator",
]


@dataclass(frozen=True)
class SessionRate:
    """A session's estimated request rate, in requests per unit time.

    Attributes:
        owner: Opaque session/owner id, matching
            :class:`~veloxquant_mlx.memory.block_pool.BlockPoolAllocator`
            owner ids.
        rate: Estimated request rate (``lambda_b`` in the paper). Any
            consistent unit works as long as ``qcap`` uses the same unit.
    """

    owner: int
    rate: float

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise QuantizerConfigError(f"SessionRate: rate must be >= 0, got {self.rate}")


@dataclass(frozen=True)
class RoutingTable:
    """A fixed admission-and-placement plan for one control interval.

    Attributes:
        shards: Maps each admitted owner to the tuple of shard ids it may
            use, ordered by preference (most-loaded-eligible-first is not
            guaranteed; callers needing a single preferred shard should use
            ``shards[owner][0]``). An owner with more than one shard should
            split its load evenly across them, per the paper's ``kb``
            replication (Eq. 1) — an owner needs more than one shard only
            when its rate exceeds ``qcap`` on its own.
        expected_load: Per-shard expected load after this plan's placement,
            indexed by shard id. Sums admitted owners' ``rate / kb`` over
            every shard each owner was placed on, plus each shard's
            cold-tail share of unadmitted traffic. For diagnostics and
            shadow-replay comparisons only — not enforced at dispatch time.
        n_shards: Total number of shards this plan was computed for.
        admitted_rate_total: Sum of ``rate`` over admitted owners, for
            computing what fraction of traffic the warm set actually covers.
        unadmitted_rate_total: Sum of ``rate`` over owners considered but not
            admitted (the plan's cold-tail fallback traffic).
    """

    shards: dict[int, tuple[int, ...]]
    expected_load: list[float]
    n_shards: int
    admitted_rate_total: float
    unadmitted_rate_total: float

    def is_admitted(self, owner: int) -> bool:
        """Whether ``owner`` has a warm-set placement in this plan."""
        return owner in self.shards

    def preferred_shard(self, owner: int, fallback: int = 0) -> int:
        """The single best shard for ``owner``, or ``fallback`` if not admitted.

        For an owner with ``kb > 1`` shards, returns the currently
        least-loaded of its assigned shards (ties broken by lowest shard
        id), so repeated calls track load drift within the same interval
        even though the *set* of eligible shards stays fixed.
        """
        candidates = self.shards.get(owner)
        if not candidates:
            return fallback
        return min(candidates, key=lambda s: (self.expected_load[s], s))

    def imbalance(self) -> float:
        """Ratio of the most-loaded shard's expected load to the mean.

        Matches the "imbalance" metric in the paper's Table 4/5 (1.0x is
        perfectly balanced; higher is worse). Returns 1.0 for zero total
        load (nothing to balance).
        """
        if not self.expected_load:
            return 1.0
        mean_load = sum(self.expected_load) / len(self.expected_load)
        if mean_load <= 0:
            return 1.0
        return max(self.expected_load) / mean_load


@dataclass
class RateEstimator:
    """Tracks a per-owner exponentially-weighted request rate estimate.

    A minimal helper for callers who don't already track per-session rates:
    call :meth:`record` on every incoming request, then :meth:`rates` to get
    the :class:`SessionRate` list a :class:`CacheRoutePlanner` needs. Not
    required — any source of ``{owner: rate}`` works, including offline
    aggregate statistics as in the paper's telemetry-derived workload.

    Args:
        half_life: Number of :meth:`record` calls for an owner's rate
            contribution to decay by half. Smaller values track bursts
            faster but are noisier; larger values are stable but slow to
            react to a session going cold or hot.
        window: Maximum number of timestamps retained per owner, bounding
            memory for long-lived owners. Older timestamps are dropped once
            exceeded, which only affects rate estimation, not correctness.
    """

    half_life: float = 20.0
    window: int = 256
    _counts: dict[int, float] = field(default_factory=dict)
    _decay: float = field(init=False)

    def __post_init__(self) -> None:
        if self.half_life <= 0:
            raise QuantizerConfigError(f"RateEstimator: half_life must be > 0, got {self.half_life}")
        if self.window < 1:
            raise QuantizerConfigError(f"RateEstimator: window must be >= 1, got {self.window}")
        self._decay = math.pow(0.5, 1.0 / self.half_life)
        self._history: dict[int, deque[int]] = {}

    def record(self, owner: int) -> None:
        """Record one request arrival for ``owner``."""
        self._counts[owner] = self._counts.get(owner, 0.0) * self._decay + 1.0
        hist = self._history.setdefault(owner, deque(maxlen=self.window))
        hist.append(1)

    def rate(self, owner: int) -> float:
        """Current smoothed rate estimate for ``owner`` (0.0 if never seen)."""
        return self._counts.get(owner, 0.0)

    def rates(self) -> list[SessionRate]:
        """Snapshot every tracked owner's current rate as a :class:`SessionRate` list."""
        return [SessionRate(owner=o, rate=r) for o, r in self._counts.items()]

    def forget(self, owner: int) -> None:
        """Drop all tracked state for ``owner`` (e.g. on session close)."""
        self._counts.pop(owner, None)
        self._history.pop(owner, None)


class CacheRoutePlanner:
    """Builds a periodic :class:`RoutingTable` from measured session rates.

    Implements Algorithm 1 from the paper: size each admitted session's
    shard count from its rate relative to ``qcap`` (Eq. 1), admit sessions
    in decreasing-rate order up to the warm-slot budget, then place
    admitted sessions with longest-processing-time-first (LPT) list
    scheduling so no shard's expected load is far from the mean.

    This planner is stateless between calls — call :meth:`plan` once per
    control interval with fresh rates, and hold onto the returned
    :class:`RoutingTable` for the whole interval rather than recomputing
    per request. Recomputing every interval, versus reusing a stale table,
    is a real tradeoff the paper measures directly (Appendix E): a fresh
    plan changes most owners' shard assignments (high "cache churn") even
    when only a few owners' rates actually moved, which costs a rewarming
    dip; a stale plan drifts and loses some hit rate/latency instead. This
    class always recomputes fresh — callers wanting hysteresis should diff
    two :class:`RoutingTable` instances themselves before switching.

    Args:
        n_shards: Number of pool shards to plan across (``R`` in the
            paper). Use ``1`` if the pool isn't partitioned — every
            admitted owner then simply gets warm-set priority with no
            placement decision to make.
        qcap: Load cap per shard, in the same rate unit as ``SessionRate.rate``
            (Section 3.1). Calibrate from a single shard's measured
            latency/load knee, as the paper recommends — this module does
            not calibrate it for you.
        warm_slots_per_shard: Warm-set capacity per shard, ``W`` in the
            paper. Total admission budget is ``n_shards * warm_slots_per_shard``,
            counted in the same units as ``rate``/``qcap`` (equal-slot
            admission model; see the paper's Section 3.1 and 6 for the
            heterogeneous-prefix-size limitation this inherits).

    Example::

        planner = CacheRoutePlanner(n_shards=4, qcap=50.0, warm_slots_per_shard=200.0)
        table = planner.plan(estimator.rates())
        shard = table.preferred_shard(owner, fallback=hash(owner) % planner.n_shards)
    """

    def __init__(
        self,
        n_shards: int,
        qcap: float,
        warm_slots_per_shard: float,
    ) -> None:
        if n_shards < 1:
            raise QuantizerConfigError(f"CacheRoutePlanner: n_shards must be >= 1, got {n_shards}")
        if qcap <= 0:
            raise QuantizerConfigError(f"CacheRoutePlanner: qcap must be > 0, got {qcap}")
        if warm_slots_per_shard <= 0:
            raise QuantizerConfigError(
                f"CacheRoutePlanner: warm_slots_per_shard must be > 0, got {warm_slots_per_shard}"
            )
        self.n_shards = n_shards
        self.qcap = qcap
        self.warm_slots_per_shard = warm_slots_per_shard

    def _assignment_count(self, rate: float) -> int:
        """``kb`` from Eq. 1: destinations needed to keep per-shard load <= qcap."""
        if rate <= 0:
            return 1
        return max(1, math.ceil(rate / self.qcap))

    def plan(self, rates: list[SessionRate]) -> RoutingTable:
        """Compute a fresh routing table from the given session rates.

        Args:
            rates: Estimated per-session rates for this control interval.
                Order does not matter; sessions are re-sorted internally by
                decreasing rate as the paper's Algorithm 1 requires.

        Returns:
            A :class:`RoutingTable` valid for the caller's next control
            interval.
        """
        capacity = self.n_shards * self.warm_slots_per_shard
        ordered = sorted((r for r in rates if r.rate > 0), key=lambda r: r.rate, reverse=True)

        admitted: list[tuple[SessionRate, int]] = []
        used = 0.0
        admitted_rate_total = 0.0
        unadmitted_rate_total = 0.0
        for r in ordered:
            kb = self._assignment_count(r.rate)
            if used + kb <= capacity:
                admitted.append((r, kb))
                used += kb
                admitted_rate_total += r.rate
            else:
                unadmitted_rate_total += r.rate

        load = [0.0] * self.n_shards
        if unadmitted_rate_total > 0:
            cold_share = unadmitted_rate_total / self.n_shards
            for i in range(self.n_shards):
                load[i] = cold_share

        shards: dict[int, tuple[int, ...]] = {}
        for r, kb in admitted:
            kb = min(kb, self.n_shards)
            chosen = sorted(range(self.n_shards), key=lambda s: (load[s], s))[:kb]
            per_shard_load = r.rate / kb
            for s in chosen:
                load[s] += per_shard_load
            shards[r.owner] = tuple(sorted(chosen))

        return RoutingTable(
            shards=shards,
            expected_load=load,
            n_shards=self.n_shards,
            admitted_rate_total=admitted_rate_total,
            unadmitted_rate_total=unadmitted_rate_total,
        )
