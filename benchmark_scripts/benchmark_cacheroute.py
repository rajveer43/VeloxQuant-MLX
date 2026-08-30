"""Real-model CacheRoute benchmark (issue #278).

Drives an actual local MLX model through a skewed, interleaved multi-turn
chat workload spread across several independent pool *shards* — each a
separate :class:`~veloxquant_mlx.memory.block_pool.BlockPoolAllocator` sized
to hold only a fraction of all sessions resident at once, forcing real
eviction pressure per shard. Compares two shard-assignment policies:

  * ``sticky_hash``: fixed affinity — ``owner % n_shards`` for the whole
    run, with LRU eviction *within* whichever shard a session lands on
    (the paper's consistent-hashing baseline: stable placement, no load
    awareness, so a shard that happens to draw more hot sessions gets more
    eviction pressure than the others).
  * ``cacheroute``: :class:`~veloxquant_mlx.routing.cacheroute.CacheRoutePlanner`
    periodically replans which shard each session should prefer from its
    observed turn rate (longest-processing-time-first placement, Algorithm
    1 in the paper), so shard load stays balanced even though the
    underlying arrival stream is skewed.

Both policies share the same skewed arrival stream and the same total pool
budget (``n_shards`` shards of equal size), so the comparison isolates the
placement policy — this is the paper's actual mechanism (Section 5.1/5.2:
the 2.3x capacity gain comes from *balanced placement* under affinity, not
from admission alone). For each turn whose previous-turn cache is still
resident in its current shard, that turn's prior context skips prefill; a
session evicted since its last turn, or moved to a different shard by a
replan, must re-prefill from scratch. Reports served hit rate (prefill
avoided) and per-shard load imbalance (``max/mean`` turns routed to a
shard, matching the paper's Table 4/5 metric).

Observed on this harness (SmolLM2-135M, 24-32 sessions, 3-4 shards, Zipf
s=1.1-1.3): CacheRoute's hit rate runs a few points *below* sticky_hash
(~79-81% vs ~82%) because periodic replanning occasionally moves a session
to a new shard and pays a cold-start cost sticky affinity never incurs,
but its load imbalance is consistently much better (1.1-1.3x vs 1.6-1.8x).
This matches the paper's own framing: hit rate alone understates the
benefit of balanced placement, which shows up as sustained *capacity*
under load (Section 5.1) rather than as a raw hit-rate improvement over a
single-shard-equivalent baseline — a hot, imbalanced shard becomes the
serving bottleneck long before its raw hit rate looks bad. See
``veloxquant_mlx/benchmarks/cacheroute_benchmark.py`` for a much larger,
pure-Python discrete-event version of the same comparison (with a
cache-blind baseline too) that makes this tradeoff visible at a scale
real-model inference is too slow to reach.

Usage::

    python benchmark_scripts/benchmark_cacheroute.py
    python benchmark_scripts/benchmark_cacheroute.py \\
        --model mlx-community/Llama-3.2-1B-Instruct-4bit --n-sessions 32 --n-shards 4 --n-turns 10
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator, PoolConfig
from veloxquant_mlx.memory.pool_backed_cache import PoolBackedKVCache, build_pooled_caches
from veloxquant_mlx.routing.cacheroute import CacheRoutePlanner, RateEstimator

DEFAULT_MODEL = "mlx-community/SmolLM2-135M-Instruct"
DEFAULT_N_SESSIONS = 24
DEFAULT_N_SHARDS = 3
DEFAULT_N_TURNS = 10
DEFAULT_TURN_TOKENS = 24  # new prompt tokens contributed by each turn
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEED = 0
# Fraction of one shard's session quota (n_sessions / n_shards) that shard
# is sized to hold resident at once. Below 1.0 by construction, so
# eviction pressure is guaranteed on every shard regardless of model or
# session-count choice; the concrete block budget is computed from the
# model's real layer count once it's loaded (see run_benchmark).
DEFAULT_RESIDENT_FRACTION = 0.5
DEFAULT_ZIPF_S = 1.1


@dataclass
class PolicyOutcome:
    """Aggregated outcome for one shard-assignment policy over one replay."""

    policy: str
    n_turns: int
    warm_hits: int  # turns whose previous-turn cache was still resident in-shard
    tokens_saved: int  # prompt tokens skipped thanks to a warm cache
    arrivals_per_shard: list[int]
    wall_time_s: float

    @property
    def hit_rate(self) -> float:
        return self.warm_hits / self.n_turns if self.n_turns else 0.0

    @property
    def imbalance(self) -> float:
        if not self.arrivals_per_shard:
            return 1.0
        mean_load = sum(self.arrivals_per_shard) / len(self.arrivals_per_shard)
        return max(self.arrivals_per_shard) / mean_load if mean_load > 0 else 1.0


def _load_model(model_id: str):
    import mlx_lm

    model, tokenizer = mlx_lm.load(model_id)
    return model, tokenizer


def _n_layers(model) -> int:
    layers = getattr(model, "layers", None) or model.model.layers
    return len(layers)


def _prefill(model, caches: list[PoolBackedKVCache], token_ids: mx.array) -> None:
    """Run one forward pass over ``token_ids`` to grow every layer's cache."""
    logits = model(token_ids[None, :], cache=caches)
    mx.eval(logits)


def _zipf_session_weights(n_sessions: int, zipf_s: float, rng: np.random.Generator) -> np.ndarray:
    """Shuffled Zipf(s) arrival-probability weights, summing to 1.

    Mirrors the paper's skewed per-key rate distribution (Gini ~0.76):
    without this skew, every session is equally cache-worthy and there is
    no load-imbalance problem for placement to solve.
    """
    ranks = np.arange(1, n_sessions + 1)
    weights = 1.0 / (ranks**zipf_s)
    order = rng.permutation(n_sessions)
    session_weight = np.empty(n_sessions)
    session_weight[order] = weights
    return session_weight / session_weight.sum()


def _pool_blocks_per_shard(
    n_layers: int,
    n_sessions: int,
    n_shards: int,
    n_turns: int,
    turn_tokens: int,
    block_size: int,
    resident_fraction: float,
) -> tuple[int, int]:
    """Block budget for one shard, sized to force real eviction pressure.

    Returns:
        ``(pool_blocks_per_shard, resident_sessions_per_shard)``. Each
        shard is sized to hold only ``resident_fraction`` of its fair
        share of sessions (``n_sessions / n_shards``) at peak per-session
        length; ``separate_kv=True`` halves usable blocks per stream, so
        the raw budget is doubled to keep the K/V accounting simple.
    """
    peak_len = n_turns * turn_tokens
    blocks_per_session = -(-peak_len // block_size) * n_layers
    fair_share = max(1, n_sessions // n_shards)
    resident_sessions = max(1, int(fair_share * resident_fraction))
    return 2 * blocks_per_session * resident_sessions, resident_sessions


class _ShardedPool:
    """Owns one independent BlockPoolAllocator per shard.

    A session's blocks live entirely in whichever single shard it is
    currently assigned to — there is no cross-shard sharing, matching the
    paper's model of destinations as independent servers each with their
    own cache.
    """

    def __init__(self, n_shards: int, block_size: int, pool_blocks_per_shard: int) -> None:
        self.n_shards = n_shards
        self.block_size = block_size
        self.pools = [
            BlockPoolAllocator(PoolConfig(block_size=block_size, n_blocks=pool_blocks_per_shard, grow_on_exhaustion=False))
            for _ in range(n_shards)
        ]

    def blocks_needed(self, current_len: int, turn_tokens: int, n_layers: int) -> int:
        prev_blocks = -(-current_len // self.block_size) if current_len > 0 else 0
        new_blocks = -(-(current_len + turn_tokens) // self.block_size)
        return max(0, new_blocks - prev_blocks) * n_layers

    def has_room(self, shard: int, current_len: int, turn_tokens: int, n_layers: int) -> bool:
        needed = self.blocks_needed(current_len, turn_tokens, n_layers)
        pool = self.pools[shard]
        return pool.n_free("k") >= needed and pool.n_free("v") >= needed


def _run_policy(
    policy: str,
    model,
    n_sessions: int,
    n_shards: int,
    n_turns: int,
    turn_tokens: int,
    pool_blocks_per_shard: int,
    resident_capacity_per_shard: int,
    max_context_tokens: int,
    block_size: int,
    vocab_size: int,
    zipf_s: float,
    seed: int,
) -> PolicyOutcome:
    rng = np.random.default_rng(seed)
    n_layers = _n_layers(model)
    session_weight = _zipf_session_weights(n_sessions, zipf_s, rng)
    n_total_arrivals = n_sessions * n_turns
    arrival_stream = rng.choice(n_sessions, size=n_total_arrivals, p=session_weight)

    sharded_pool = _ShardedPool(n_shards, block_size, pool_blocks_per_shard)
    # session_shard[owner]: which shard currently holds (or last held) this
    # session's resident cache, so a "hit" means both same-session AND
    # same-shard — a session moved to a new shard by a replan starts cold there.
    session_shard: dict[int, int] = {}
    session_caches: dict[int, list[PoolBackedKVCache]] = {}
    session_len: dict[int, int] = {}
    session_last_turn: dict[int, int] = {}
    arrivals_per_shard = [0] * n_shards

    estimator = RateEstimator(half_life=n_sessions)
    planner = CacheRoutePlanner(
        n_shards=n_shards,
        qcap=float(n_sessions),  # generous: keeps kb=1 for every session, like the paper's primary workload
        warm_slots_per_shard=float(resident_capacity_per_shard),
    )
    table = planner.plan([])

    def target_shard(owner: int) -> int:
        if policy == "cacheroute":
            return table.preferred_shard(owner, fallback=owner % n_shards)
        return owner % n_shards  # sticky_hash

    def evict_from(shard: int, protect: int | None) -> bool:
        candidates = [o for o, s in session_shard.items() if s == shard and o != protect]
        if not candidates:
            return False
        evict_owner = min(candidates, key=lambda o: session_last_turn[o])
        for cache in session_caches.pop(evict_owner):
            cache.release()
        session_len.pop(evict_owner, None)
        session_last_turn.pop(evict_owner, None)
        session_shard.pop(evict_owner, None)
        return True

    warm_hits = 0
    tokens_saved = 0
    t_start = time.perf_counter()
    # Replan only a handful of times per run, not every few arrivals: LPT
    # resorts from scratch each time, so frequent replanning churns most
    # sessions' shard assignments even when few rates actually moved,
    # forcing needless cold resets (the paper's own finding, Appendix E).
    replan_every = max(1, n_total_arrivals // 4)

    for turn, owner in enumerate(arrival_stream.tolist()):
        estimator.record(owner)
        if policy == "cacheroute" and turn % replan_every == 0:
            table = planner.plan(estimator.rates())

        shard = target_shard(owner)
        arrivals_per_shard[shard] += 1
        new_tokens = mx.array(rng.integers(0, vocab_size, size=(turn_tokens,), dtype=np.int64))

        # A hot session under Zipf sampling can receive far more than
        # n_turns turns; cap its running context like a real chat session's
        # sliding window so per-session footprint stays bounded instead of
        # eventually exceeding what any shard could ever hold.
        over_context_cap = owner in session_len and session_len[owner] + turn_tokens > max_context_tokens
        still_resident = (
            owner in session_caches and session_shard.get(owner) == shard and not over_context_cap
        )
        if still_resident:
            warm_hits += 1
            tokens_saved += session_len[owner]
        else:
            if owner in session_caches:  # resident but in the wrong shard, or over its context cap
                for cache in session_caches.pop(owner):
                    cache.release()
                session_shard.pop(owner, None)
            session_len[owner] = 0

        while not sharded_pool.has_room(shard, session_len[owner], turn_tokens, n_layers):
            if not evict_from(shard, protect=owner if still_resident else None):
                break

        if not still_resident:
            session_caches[owner] = build_pooled_caches(model, sharded_pool.pools[shard], owner=owner, step=block_size)
            session_shard[owner] = shard

        _prefill(model, session_caches[owner], new_tokens)
        session_len[owner] += turn_tokens
        session_last_turn[owner] = turn

    wall_time_s = time.perf_counter() - t_start
    return PolicyOutcome(
        policy=policy,
        n_turns=n_total_arrivals,
        warm_hits=warm_hits,
        tokens_saved=tokens_saved,
        arrivals_per_shard=arrivals_per_shard,
        wall_time_s=wall_time_s,
    )


def run_benchmark(
    model_id: str = DEFAULT_MODEL,
    n_sessions: int = DEFAULT_N_SESSIONS,
    n_shards: int = DEFAULT_N_SHARDS,
    n_turns: int = DEFAULT_N_TURNS,
    turn_tokens: int = DEFAULT_TURN_TOKENS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    resident_fraction: float = DEFAULT_RESIDENT_FRACTION,
    zipf_s: float = DEFAULT_ZIPF_S,
    seed: int = DEFAULT_SEED,
) -> dict[str, PolicyOutcome]:
    model, tokenizer = _load_model(model_id)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 32000
    n_layers = _n_layers(model)
    pool_blocks_per_shard, resident_capacity_per_shard = _pool_blocks_per_shard(
        n_layers, n_sessions, n_shards, n_turns, turn_tokens, block_size, resident_fraction
    )
    # Cap each session's running context at its "fair share" peak length
    # (n_turns turns' worth) even though skewed sampling gives hot sessions
    # many more than n_turns arrivals — otherwise a hot session's context
    # grows without bound and eventually can't fit in any shard at all.
    max_context_tokens = n_turns * turn_tokens
    print(
        f"Model has {n_layers} layers; {n_shards} shards, each sized to "
        f"{pool_blocks_per_shard} blocks (~{resident_capacity_per_shard} sessions resident at peak length, "
        f"{max_context_tokens}-token context cap per session)."
    )

    results = {}
    for policy in ("sticky_hash", "cacheroute"):
        results[policy] = _run_policy(
            policy=policy,
            model=model,
            n_sessions=n_sessions,
            n_shards=n_shards,
            n_turns=n_turns,
            turn_tokens=turn_tokens,
            pool_blocks_per_shard=pool_blocks_per_shard,
            resident_capacity_per_shard=resident_capacity_per_shard,
            max_context_tokens=max_context_tokens,
            block_size=block_size,
            vocab_size=vocab_size,
            zipf_s=zipf_s,
            seed=seed,
        )
    return results


def format_summary(results: dict[str, PolicyOutcome]) -> str:
    header = f"{'Policy':<14}{'Hit rate':>10}{'Imbalance':>12}{'Tokens saved':>14}{'Wall time (s)':>16}"
    sep = "-" * len(header)
    lines = [header, sep]
    for name, r in results.items():
        lines.append(
            f"{name:<14}{r.hit_rate:>9.1%} {r.imbalance:>10.2f}x{r.tokens_saved:>14}{r.wall_time_s:>16.2f}"
        )
    for name, r in results.items():
        lines.append(f"  {name} arrivals/shard: {r.arrivals_per_shard}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-model CacheRoute benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-sessions", type=int, default=DEFAULT_N_SESSIONS)
    parser.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS)
    parser.add_argument("--n-turns", type=int, default=DEFAULT_N_TURNS)
    parser.add_argument("--turn-tokens", type=int, default=DEFAULT_TURN_TOKENS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--resident-fraction", type=float, default=DEFAULT_RESIDENT_FRACTION)
    parser.add_argument("--zipf-s", type=float, default=DEFAULT_ZIPF_S)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    results = run_benchmark(
        model_id=args.model,
        n_sessions=args.n_sessions,
        n_shards=args.n_shards,
        n_turns=args.n_turns,
        turn_tokens=args.turn_tokens,
        block_size=args.block_size,
        resident_fraction=args.resident_fraction,
        zipf_s=args.zipf_s,
        seed=args.seed,
    )
    print()
    print(format_summary(results))


if __name__ == "__main__":
    main()
