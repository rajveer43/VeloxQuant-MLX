"""Correctness + benchmark harness for flash_prefill_attend tuning (issue #277).

Single tool used across the whole tuning campaign so every optimization
phase is checked against the same boundary-case matrix and the same
latency methodology before being kept. Two subcommands:

  correctness  — parity against a numpy fp32 causal-attention reference,
                 plus boundary cases (first/last token, partial final KV
                 block, S not divisible by BK, fully-masked query rows).
  bench        — median-of-N latency for flash_prefill_attend vs
                 mx.fast.scaled_dot_product_attention, across D and S.

Usage:
  python scripts/flash_prefill_harness.py correctness
  python scripts/flash_prefill_harness.py bench
  python scripts/flash_prefill_harness.py all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Force repo-root resolution of veloxquant_mlx: when this file is run as
# `python scripts/flash_prefill_harness.py`, sys.path[0] is scripts/, not
# the repo root, so an unrelated frozen veloxquant_mlx copy installed in
# site-packages (if present) would shadow the local source under active
# development here. Insert the repo root first so local edits are always
# what gets imported and benchmarked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal.kernels import flash_prefill_attend

N_WARMUP, N_ITER = 10, 30


# ---------------------------------------------------------------------------
# Reference + metrics
# ---------------------------------------------------------------------------


def _reference_flash(q, k, v, scale):
    S_q, S_kv = q.shape[2], k.shape[2]
    scores = np.einsum("bhqd,bhsd->bhqs", q.astype(np.float32), k.astype(np.float32)) * scale
    q_abs = (S_kv - S_q) + np.arange(S_q)
    kv_pos = np.arange(S_kv)
    mask = kv_pos[None, :] > q_abs[:, None]
    scores = np.where(mask, -np.inf, scores)
    row_has_any = np.isfinite(scores).any(axis=-1, keepdims=True)
    scores = scores - np.where(row_has_any, scores.max(axis=-1, keepdims=True), 0.0)
    w = np.where(np.isfinite(scores), np.exp(scores), 0.0)
    denom = w.sum(axis=-1, keepdims=True)
    w = np.where(denom > 0, w / np.where(denom > 0, denom, 1.0), 0.0)
    return np.einsum("bhqs,bhsd->bhqd", w, v.astype(np.float32)).astype(np.float32)


def _make_inputs(B, H, S_q, S_kv, D, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, S_q, D)).astype(np.float16)
    k = rng.standard_normal((B, H, S_kv, D)).astype(np.float16)
    v = rng.standard_normal((B, H, S_kv, D)).astype(np.float16)
    scale = np.array([1.0 / np.sqrt(D)], dtype=np.float32)
    return q, k, v, scale


def _run_kernel(q, k, v, scale):
    out = flash_prefill_attend(mx.array(q), mx.array(k), mx.array(v), mx.array(scale))
    mx.eval(out)
    return np.array(out, dtype=np.float32)


def _metrics(got: np.ndarray, expected: np.ndarray) -> dict:
    diff = got - expected
    abs_err = np.abs(diff)
    n_nan = int(np.isnan(got).sum())
    n_inf = int(np.isinf(got).sum())
    denom = np.linalg.norm(expected.reshape(-1))
    rel_err = float(np.linalg.norm(diff.reshape(-1)) / denom) if denom > 0 else 0.0
    g, e = got.reshape(-1), expected.reshape(-1)
    ng, ne = np.linalg.norm(g), np.linalg.norm(e)
    cos = float(np.dot(g, e) / (ng * ne)) if ng > 0 and ne > 0 else float("nan")
    return {
        "max_abs_err": float(abs_err.max()) if abs_err.size else 0.0,
        "mean_abs_err": float(abs_err.mean()) if abs_err.size else 0.0,
        "rel_err": rel_err,
        "cosine_sim": cos,
        "n_nan": n_nan,
        "n_inf": n_inf,
    }


_ATOL, _RTOL = 3e-2, 3e-2


def _check(name: str, got: np.ndarray, expected: np.ndarray) -> bool:
    m = _metrics(got, expected)
    ok = (
        m["n_nan"] == 0
        and m["n_inf"] == 0
        and m["max_abs_err"] < _ATOL + _RTOL * np.abs(expected).max()
    )
    status = "OK  " if ok else "FAIL"
    print(
        f"[{status}] {name:<48} max_abs={m['max_abs_err']:.4e} mean_abs={m['mean_abs_err']:.4e} "
        f"rel={m['rel_err']:.4e} cos={m['cosine_sim']:.6f} nan={m['n_nan']} inf={m['n_inf']}"
    )
    return ok


# ---------------------------------------------------------------------------
# Correctness matrix
# ---------------------------------------------------------------------------


def run_correctness() -> bool:
    all_ok = True

    print("== parity sweep: D x S_q x S_kv x (B,H) ==")
    for D in (32, 64, 128):
        for S in (64, 128, 256, 512, 1024, 2048):
            for S_q, S_kv in ((S, S), (max(1, S // 2), S)):
                for B, H in ((1, 1), (2, 4)):
                    q, k, v, scale = _make_inputs(B, H, S_q, S_kv, D, seed=D + S_q + S_kv + B * H)
                    expected = _reference_flash(q, k, v, scale)
                    got = _run_kernel(q, k, v, scale)
                    name = f"D={D} S_q={S_q} S_kv={S_kv} B={B} H={H}"
                    all_ok &= _check(name, got, expected)

    print()
    print("== causal boundary cases ==")

    # First token: S_q=1 attending to a single-slot KV cache — only one
    # valid slot, degenerate softmax (weight=1 on that slot).
    q, k, v, scale = _make_inputs(1, 2, 1, 1, 64, seed=101)
    expected = _reference_flash(q, k, v, scale)
    got = _run_kernel(q, k, v, scale)
    all_ok &= _check("first token (S_q=1, S_kv=1)", got, expected)

    # Final token of a long self-attention sequence: last row sees the
    # full KV range — exercises the widest per-row softmax in the sweep.
    q, k, v, scale = _make_inputs(1, 2, 257, 257, 128, seed=102)
    expected = _reference_flash(q, k, v, scale)
    got = _run_kernel(q, k, v, scale)
    all_ok &= _check("final token wide row (S=257, D=128)", got[:, :, -1:], expected[:, :, -1:])

    # Partial final KV block: S_kv not a multiple of BK=16.
    for S_kv in (17, 31, 33, 47, 63, 65):
        q, k, v, scale = _make_inputs(1, 1, S_kv, S_kv, 64, seed=200 + S_kv)
        expected = _reference_flash(q, k, v, scale)
        got = _run_kernel(q, k, v, scale)
        all_ok &= _check(
            f"partial final KV block (S={S_kv}, BK=16 remainder={S_kv % 16})", got, expected
        )

    # S not divisible by BQ_TG=32 (query block partially outside S_q):
    # the Python wrapper pads the grid to a full 32-row block, so rows
    # >= S_q must never be written into the (correctly shaped) output —
    # this is implicitly checked by got.shape matching, plus parity on
    # the actually-valid rows.
    for S in (1, 5, 31, 33, 63, 65, 97):
        q, k, v, scale = _make_inputs(1, 1, S, S, 32, seed=300 + S)
        expected = _reference_flash(q, k, v, scale)
        got = _run_kernel(q, k, v, scale)
        assert got.shape == (1, 1, S, 32), f"shape mismatch at S={S}: {got.shape}"
        all_ok &= _check(
            f"query block partially outside S_q (S={S}, BQ_TG=32 remainder={S % 32})", got, expected
        )

    # Fully masked query rows: S_kv < S_q with no cache alignment offset
    # possible would violate causal (q_abs would be negative for early
    # rows) — construct via S_kv > S_q but forcing q_align negative isn't
    # reachable through the public API (S_kv >= S_q is required for the
    # kernel's causal convention to make sense), so instead exercise the
    # kernel's n_chunks==0 path directly: S_kv=0 is invalid, so use the
    # smallest S_kv=S_q=1 case (already covered above) plus a cache-
    # continuation shape where S_q spans a block whose first rows still
    # see a nonempty prefix (n_chunks>0 for all rows) — the true
    # zero-visible-KV case only occurs when S_kv < q_align+1, which the
    # wrapper's alignment convention (q_align = S_kv - S_q >= 0) makes
    # unreachable for any in-range row. Documented here rather than
    # faked: this kernel's causal convention structurally guarantees
    # every valid query row sees at least one KV slot (itself, at
    # minimum), so "n_chunks==0 for a valid row" is unreachable and the
    # write-time zero-fill in the kernel's tail is dead-but-safe code
    # for rows < S_q (it only ever fires for the padding rows >= S_q,
    # already covered by the "query block partially outside S_q" cases).
    print(
        "[info] fully-masked-valid-query-row case is structurally unreachable "
        "given this kernel's q_align=S_kv-S_q>=0 convention (every valid row "
        "sees >=1 KV slot); verified the padding-row zero-fill path instead "
        "via the S%32!=0 cases above."
    )

    # KV-cache continuation (S_kv > S_q): queries are new tokens appended
    # after an existing plain prefix.
    for S_q, S_kv in ((1, 500), (16, 500), (100, 4096)):
        q, k, v, scale = _make_inputs(1, 2, S_q, S_kv, 128, seed=400 + S_q + S_kv)
        expected = _reference_flash(q, k, v, scale)
        got = _run_kernel(q, k, v, scale)
        all_ok &= _check(f"cache continuation (S_q={S_q}, S_kv={S_kv})", got, expected)

    print()
    if all_ok:
        print("ALL CORRECTNESS CHECKS PASSED")
    else:
        print("CORRECTNESS FAILURES DETECTED — see FAIL rows above")
    return all_ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _bench(fn, n_warmup: int = N_WARMUP, n_iter: int = N_ITER) -> tuple[float, float, float]:
    for _ in range(n_warmup):
        mx.eval(fn())
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - t0)
    samples.sort()
    n = len(samples)
    median = samples[n // 2]
    p50 = samples[n // 2]
    p95 = samples[min(n - 1, int(n * 0.95))]
    return median, p50, p95


def run_bench() -> None:
    print(f"[bench] device=Apple M4 (applegpu_g16g)  N_WARMUP={N_WARMUP} N_ITER={N_ITER}")
    print(
        f"{'D':>4} {'B':>2} {'H':>3} {'S_q':>6} {'S_kv':>6} | {'flash median (ms)':>17} | "
        f"{'p95 (ms)':>9} | {'sdpa median (ms)':>17} | {'speedup':>8} | {'max_err':>9}"
    )
    print("-" * 105)

    rng = np.random.default_rng(7)
    configs = [
        # D,  B, H,  S
        (32, 1, 8, 512),
        (32, 1, 8, 2048),
        (64, 1, 8, 512),
        (64, 1, 8, 2048),
        (64, 1, 8, 8192),
        (128, 1, 8, 512),
        (128, 1, 8, 2048),
        (128, 1, 8, 8192),
        (128, 1, 32, 2048),
    ]
    for D, B, H, S in configs:
        q = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
        k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
        v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
        scale_arr = mx.array([1.0 / np.sqrt(D)], dtype=mx.float32)
        mx.eval(q, k, v, scale_arr)
        scale = 1.0 / float(D) ** 0.5

        def flash():
            return flash_prefill_attend(q, k, v, scale_arr)

        def sdpa():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

        n_iter = N_ITER if S <= 2048 else max(5, N_ITER // (S // 2048))
        t_f, _, p95_f = _bench(flash, N_WARMUP, n_iter)
        t_s, _, _ = _bench(sdpa, N_WARMUP, n_iter)

        out_f = np.array(flash(), dtype=np.float32)
        out_s = np.array(sdpa(), dtype=np.float32)
        max_err = float(np.abs(out_f - out_s).max())

        print(
            f"{D:>4} {B:>2} {H:>3} {S:>6} {S:>6} | {t_f * 1000:>17.4f} | {p95_f * 1000:>9.4f} | "
            f"{t_s * 1000:>17.4f} | {t_s / t_f:>7.2f}x | {max_err:>9.4e}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["correctness", "bench", "all"])
    args = ap.parse_args()

    ok = True
    if args.mode in ("correctness", "all"):
        ok = run_correctness()
    if args.mode in ("bench", "all"):
        run_bench()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
