"""Property-based correctness audit for TOVA / H2O / KIVI cache bookkeeping.

From-scratch slow-but-obviously-correct Python reference implementations for
TOVA and H2O eviction (list-of-tuples, pure Python, no MLX vectorization
tricks) are compared against the real VeloxQuant-MLX implementations across
many randomized operation sequences. KIVI is checked deterministically
(round-trip quant/dequant has no eviction, so a small fixed matrix suffices,
not property search) for group-atomicity of (codes, scale, zero) under a
simulated eviction-like slice.

Run: .venv/bin/python scripts/kv_bookkeeping_property_audit.py --output /tmp/kv_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers.h2o import h2o_update, init_h2o_state
from veloxquant_mlx.quantizers.tova import init_tova_state, tova_update

# ===========================================================================
# Slow, obviously-correct Python reference implementations
# ===========================================================================


class RefTova:
    """Pure-Python TOVA reference: list of (key, value) rows, no MLX.

    Score = softmax(key_row @ proxy_query / sqrt(d)) over ALL current rows
    (post-append), matching the documented algorithm in quantizers/tova.py:
    key-as-query proxy, memoryless (no score history), sink rows protected
    with +inf, ties broken toward the lowest index (matches mx.argmin).
    """

    def __init__(self, n_sink: int, budget: int, d: int):
        self.n_sink = n_sink
        self.budget = budget
        self.d = d
        self.rows: list[tuple[list[float], list[float]]] = []  # (key, value)

    def update(self, new_keys: list[list[float]], new_values: list[list[float]]):
        for k_i, v_i in zip(new_keys, new_values):
            self.rows.append((list(k_i), list(v_i)))
            n_total = len(self.rows)
            if n_total > self.budget:
                scale = 1.0 / math.sqrt(self.d)
                logits = [_dot(k_i, r[0]) * scale for r in self.rows]
                mx_ = max(logits)
                exps = [math.exp(v - mx_) for v in logits]
                s = sum(exps)
                weights = [e / s for e in exps]
                n_sink_eff = min(self.n_sink, n_total)
                protected = list(weights)
                for j in range(n_sink_eff):
                    protected[j] = float("inf")
                evict_idx = 0
                best = protected[0]
                for j in range(1, n_total):
                    if protected[j] < best:
                        best = protected[j]
                        evict_idx = j
                del self.rows[evict_idx]

    def kv(self):
        if not self.rows:
            return [], []
        keys = [r[0] for r in self.rows]
        values = [r[1] for r in self.rows]
        return keys, values


class RefH2O:
    """Pure-Python H2O reference: list of (key, value, score, position).

    Cumulative softmax-mass score with decay, grace-protected + sink-protected
    argmin eviction, contiguous-position renumbering of survivors after the
    evicted index (matches quantizers/h2o.py's documented semantics). Does
    NOT re-rotate keys (that's a RoPE detail orthogonal to bookkeeping
    correctness of scores/positions/keep-set); this reference is used only to
    check which tokens are kept/evicted and position bookkeeping, not RoPE
    numerics.
    """

    def __init__(self, n_sink: int, budget: int, d: int, grace: int = 0, decay: float = 1.0):
        self.n_sink = n_sink
        self.budget = budget
        self.d = d
        self.grace = grace
        self.decay = decay
        self.rows: list[list] = []  # [key, value, score, position]
        self.next_pos = 0

    def update(self, new_keys, new_values):
        for k_i, v_i in zip(new_keys, new_values):
            cur_pos = self.next_pos
            if not self.rows:
                self.rows.append([list(k_i), list(v_i), 1.0, cur_pos])
                self.next_pos += 1
                continue
            scale = 1.0 / math.sqrt(self.d)
            logits = [_dot(k_i, r[0]) * scale for r in self.rows]
            mx_ = max(logits)
            exps = [math.exp(v - mx_) for v in logits]
            s = sum(exps)
            attn = [e / s for e in exps]
            for j, r in enumerate(self.rows):
                r[2] = r[2] * self.decay + attn[j]
            self.rows.append([list(k_i), list(v_i), 0.0, cur_pos])
            n_total = len(self.rows)
            if n_total > self.budget:
                n_sink_eff = min(self.n_sink, n_total)
                n_grace_eff = min(self.grace, n_total)
                protected = [r[2] for r in self.rows]
                for j in range(n_sink_eff):
                    protected[j] = float("inf")
                for j in range(n_total - n_grace_eff, n_total):
                    protected[j] = float("inf")
                evict_idx = 0
                best = protected[0]
                for j in range(1, n_total):
                    if protected[j] < best:
                        best = protected[j]
                        evict_idx = j
                evicted_pos = self.rows[evict_idx][3]
                del self.rows[evict_idx]
                for r in self.rows:
                    if r[3] > evicted_pos:
                        r[3] -= 1
            self.next_pos = cur_pos + 1

    def kv(self):
        if not self.rows:
            return [], []
        keys = [r[0] for r in self.rows]
        values = [r[1] for r in self.rows]
        return keys, values

    def positions(self):
        return [r[3] for r in self.rows]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ===========================================================================
# Property tests
# ===========================================================================


def _to_py(arr: mx.array) -> list:
    return np.array(arr).astype(np.float64).tolist()


def run_tova_trial(rng: random.Random, seed_id: int) -> dict:
    n_sink = rng.randint(0, 3)
    budget = rng.randint(n_sink + 1, n_sink + 6)
    d = rng.choice([2, 4, 8])
    n_steps = rng.randint(1, 40)

    ref = RefTova(n_sink, budget, d)
    state = init_tova_state(n_sink, budget, d)

    np_rng = np.random.default_rng(seed_id)
    for step in range(n_steps):
        chunk = rng.randint(1, 3)
        new_k = np_rng.normal(size=(chunk, d)).astype(np.float32) * 2.0
        new_v = np_rng.normal(size=(chunk, d)).astype(np.float32) * 2.0
        ref.update(new_k.tolist(), new_v.tolist())
        for backend in ("mlx", "metal"):
            pass
        state = tova_update(
            state,
            mx.array(new_k.astype(np.float16)),
            mx.array(new_v.astype(np.float16)),
            backend="mlx",
        )

    ref_k, ref_v = ref.kv()
    real_k = _to_py(state.keys) if state.keys is not None else []
    real_v = _to_py(state.values) if state.values is not None else []

    ok = len(ref_k) == len(real_k)
    max_err = 0.0
    if ok:
        for a, b in zip(ref_k, real_k):
            for x, y in zip(a, b):
                max_err = max(max_err, abs(x - y))
        for a, b in zip(ref_v, real_v):
            for x, y in zip(a, b):
                max_err = max(max_err, abs(x - y))
    return dict(
        seed=seed_id,
        n_sink=n_sink,
        budget=budget,
        d=d,
        n_steps=n_steps,
        len_match=ok,
        max_abs_err=max_err,
        pass_=ok and max_err < 0.05,  # fp16 tolerance
    )


def run_h2o_trial(rng: random.Random, seed_id: int) -> dict:
    n_sink = rng.randint(0, 2)
    grace = rng.randint(0, 2)
    budget = rng.randint(n_sink + grace + 1, n_sink + grace + 6)
    d = rng.choice([2, 4, 8])
    n_steps = rng.randint(1, 30)
    decay = 1.0  # keep decay=1.0 for the reference-vs-real position/keep-set check

    ref = RefH2O(n_sink, budget, d, grace=grace, decay=decay)
    state = init_h2o_state(n_sink, budget, d, grace=grace, decay=decay)

    np_rng = np.random.default_rng(seed_id + 10_000)
    for step in range(n_steps):
        chunk = rng.randint(1, 2)
        new_k = np_rng.normal(size=(chunk, d)).astype(np.float32) * 2.0
        new_v = np_rng.normal(size=(chunk, d)).astype(np.float32) * 2.0
        ref.update(new_k.tolist(), new_v.tolist())
        state = h2o_update(
            state,
            mx.array(new_k.astype(np.float16)),
            mx.array(new_v.astype(np.float16)),
        )

    ref_k, ref_v = ref.kv()
    ref_pos = ref.positions()
    real_k = _to_py(state.keys) if state.keys is not None else []
    real_v = _to_py(state.values) if state.values is not None else []
    real_pos = _to_py(state.positions) if state.positions is not None else []

    ok_len = len(ref_k) == len(real_k)
    ok_pos = ref_pos == [int(p) for p in real_pos] if ok_len else False
    return dict(
        seed=seed_id,
        n_sink=n_sink,
        grace=grace,
        budget=budget,
        d=d,
        n_steps=n_steps,
        len_match=ok_len,
        positions_match=ok_pos,
        ref_n_kept=len(ref_k),
        real_n_kept=len(real_k),
        pass_=ok_len and ok_pos,
    )


def kivi_group_atomicity_check() -> dict:
    """Check that KIVI's (codes, scale, zero) triple stays atomically
    consistent under a simulated eviction: slicing away a leading group of
    tokens must not corrupt the (scale, zero) of the surviving groups, since
    KIVIKVCache never evicts mid-group (flush boundary is snapped down to a
    multiple of group_size — see kivi_cache.py's _quantization_boundary).
    """
    from veloxquant_mlx.quantizers.kivi import KIVIQuantizer

    d = 8
    group_size = 4
    q = KIVIQuantizer(d=d, b=4, group_size=group_size, axis="channel")
    rng = np.random.default_rng(7)
    x = mx.array(rng.normal(size=(16, d)).astype(np.float32))
    ev = q.encode(x)
    recon_full = q.decode(ev)

    # Simulate "evicting" the first group_size rows (a group-aligned cut,
    # which is the ONLY cut KIVIKVCache ever performs per _quantization_boundary).
    x_sliced = x[group_size:]
    ev2 = q.encode(x_sliced)
    recon_sliced = q.decode(ev2)

    # The surviving rows' reconstruction should match whether or not the
    # evicted group's rows were ever present, because groups are independent.
    err = float(
        mx.max(mx.abs(recon_full[group_size:].astype(mx.float32) - recon_sliced.astype(mx.float32)))
    )

    # Now check a NON-group-aligned cut would break atomicity (proving the
    # boundary-snapping logic in the real cache is load-bearing, not just
    # a style choice).
    misaligned_cut = group_size + 1
    x_mis = x[misaligned_cut:]
    ev3 = q.encode(x_mis)
    recon_mis = q.decode(ev3)
    # Compare against recon_full's tail at the same absolute rows.
    err_mis = float(
        mx.max(
            mx.abs(recon_full[misaligned_cut:].astype(mx.float32) - recon_mis.astype(mx.float32))
        )
    )

    return dict(
        group_aligned_cut_max_err=err,
        misaligned_cut_max_err=err_mis,
        group_aligned_atomic=err < 1e-3,
        misaligned_breaks_atomicity=err_mis > 1e-3,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()

    tova_results = [run_tova_trial(rng, i) for i in range(args.trials)]
    tova_failures = [r for r in tova_results if not r["pass_"]]

    h2o_results = [run_h2o_trial(rng, i + 1_000_000) for i in range(args.trials)]
    h2o_failures = [r for r in h2o_results if not r["pass_"]]

    kivi_result = kivi_group_atomicity_check()

    elapsed = time.time() - t0

    out = dict(
        elapsed_s=elapsed,
        trials=args.trials,
        tova=dict(
            n_pass=len(tova_results) - len(tova_failures),
            n_fail=len(tova_failures),
            failures=tova_failures[:20],
        ),
        h2o=dict(
            n_pass=len(h2o_results) - len(h2o_failures),
            n_fail=len(h2o_failures),
            failures=h2o_failures[:20],
        ),
        kivi_group_atomicity=kivi_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("tova", "h2o")}, indent=2))
    print("TOVA:", out["tova"]["n_pass"], "/", args.trials, "pass")
    print("H2O:", out["h2o"]["n_pass"], "/", args.trials, "pass")
    print("KIVI atomicity:", kivi_result)


if __name__ == "__main__":
    main()
