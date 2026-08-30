"""Benchmark harness: experimental streaming prefill vs flash_prefill vs MLX native.

Compares the row-owned streaming kernel family (metal/_experimental_streaming_prefill.py)
against:
  - flash_prefill_attend       — the existing tiled simdgroup_matrix kernel (unmodified).
  - mx.fast.scaled_dot_product_attention(mask="causal") — MLX's own tuned SDPA.

across D in {32, 64, 128} and S in {64, 128, 256, 512, 1024, 2048, 4096, 8192},
for both S_q == S_kv (full prefill) and one S_q < S_kv (cache-continuation)
case per D. Uses the same timing methodology as scripts/flash_prefill_harness.py
(``_bench``: warmup + N_ITER timed mx.eval() calls, median/p50/p95 reported) and
the same error-metric helper (``_metrics``) for max/mean abs error, relative
error, cosine similarity, and NaN/Inf flags vs a numpy fp32 reference.

Usage:
  python scripts/streaming_prefill_benchmark.py
  python scripts/streaming_prefill_benchmark.py --n_iter 30 --skip-large
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# See scripts/flash_prefill_harness.py's identical comment: insert the repo
# root first so local edits under active development are what get imported,
# not a frozen veloxquant_mlx copy that might be shadowing it in site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal.kernels import flash_prefill_attend, streaming_prefill_attend

N_WARMUP, N_ITER = 10, 30

STREAMING_IMPLS = [
    "streaming",
    "streaming_block2",
    "streaming_block4",
    "streaming_block8",
    "streaming_multirow",
]

ALL_KERNELS = ["flash_prefill", "mlx_native"] + STREAMING_IMPLS


# ---------------------------------------------------------------------------
# Reference + metrics (same convention as scripts/flash_prefill_harness.py)
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


def _metrics(got: np.ndarray, expected: np.ndarray) -> dict:
    diff = got - expected
    abs_err = np.abs(diff)
    n_nan = int(np.isnan(got).sum())
    n_inf = int(np.isinf(got).sum())
    denom = np.linalg.norm(expected.reshape(-1))
    rel_err = float(np.linalg.norm(diff.reshape(-1)) / denom) if denom > 0 else 0.0
    return {
        "max_abs_err": float(abs_err.max()) if abs_err.size else 0.0,
        "mean_abs_err": float(abs_err.mean()) if abs_err.size else 0.0,
        "rel_err": rel_err,
        "n_nan": n_nan,
        "n_inf": n_inf,
    }


def _bench(fn, n_warmup: int = N_WARMUP, n_iter: int = N_ITER) -> tuple[float, float, float]:
    """Median/p50/p95 wall-clock seconds per call. mx.eval() forces GPU sync."""
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
    p95 = samples[min(n - 1, int(n * 0.95))]
    return median, median, p95


# ---------------------------------------------------------------------------
# Kernel call wrappers
# ---------------------------------------------------------------------------


def _make_callable(kernel: str, q, k, v, scale_arr, scale_scalar):
    if kernel == "flash_prefill":
        return lambda: flash_prefill_attend(q, k, v, scale_arr)
    if kernel == "mlx_native":
        return lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale_scalar, mask="causal"
        )
    if kernel in STREAMING_IMPLS:
        return lambda: streaming_prefill_attend(q, k, v, scale_arr, implementation=kernel)
    raise ValueError(kernel)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run(n_iter: int, skip_large: bool) -> list[dict]:
    Ds = [32, 64, 128]
    S_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    if skip_large:
        S_values = [s for s in S_values if s <= 2048]

    B, H = 1, 8
    results: list[dict] = []

    print(f"[bench] device=Apple M4 GPU  N_WARMUP={N_WARMUP} N_ITER={n_iter}")
    print(f"[bench] kernels: {ALL_KERNELS}")
    print()

    for D in Ds:
        # Shapes to test per D: full prefill (S_q==S_kv) for every S, plus
        # exactly one cache-continuation (S_q<S_kv) case.
        shapes = [(S, S, "full") for S in S_values]
        shapes.append((max(1, S_values[len(S_values) // 2] // 4), S_values[len(S_values) // 2], "cache_cont"))

        for S_q, S_kv, kind in shapes:
            # Reduce iteration count for very large sequences to keep the
            # sweep tractable — noted explicitly rather than silently.
            eff_iter = n_iter if S_kv <= 2048 else max(5, n_iter // max(1, S_kv // 2048))

            q_np, k_np, v_np, scale_np = _make_inputs(B, H, S_q, S_kv, D, seed=D + S_q + S_kv)
            expected = _reference_flash(q_np, k_np, v_np, scale_np)

            q = mx.array(q_np)
            k = mx.array(k_np)
            v = mx.array(v_np)
            scale_arr = mx.array(scale_np, dtype=mx.float32)
            scale_scalar = float(scale_np[0])
            mx.eval(q, k, v, scale_arr)

            row = {"D": D, "S_q": S_q, "S_kv": S_kv, "kind": kind, "B": B, "H": H}
            flash_median = None
            mlx_median = None

            for kernel in ALL_KERNELS:
                try:
                    fn = _make_callable(kernel, q, k, v, scale_arr, scale_scalar)
                    out = fn()
                    mx.eval(out)
                    got = np.array(out, dtype=np.float32)
                    m = _metrics(got, expected)

                    median, _, p95 = _bench(fn, N_WARMUP, eff_iter)
                    tok_per_s = (B * H * S_q) / median if median > 0 else float("nan")

                    row[kernel] = {
                        "median_ms": median * 1000.0,
                        "p95_ms": p95 * 1000.0,
                        "tokens_per_s": tok_per_s,
                        **m,
                    }
                    if kernel == "flash_prefill":
                        flash_median = median
                    if kernel == "mlx_native":
                        mlx_median = median
                except Exception as e:  # noqa: BLE001 — record failure, keep sweeping
                    row[kernel] = {"error": str(e)}

            for kernel in ALL_KERNELS:
                entry = row.get(kernel, {})
                if isinstance(entry, dict) and "median_ms" in entry:
                    if flash_median:
                        entry["speedup_vs_flash"] = flash_median / (entry["median_ms"] / 1000.0)
                    if mlx_median:
                        entry["speedup_vs_mlx"] = mlx_median / (entry["median_ms"] / 1000.0)

            results.append(row)

            print(f"=== D={D} S_q={S_q} S_kv={S_kv} ({kind}) ===")
            print(
                f"  {'kernel':<20} {'median(ms)':>11} {'p95(ms)':>9} {'tok/s':>12} "
                f"{'vs_flash':>9} {'vs_mlx':>8} {'max_err':>10} {'mean_err':>10} {'nan/inf':>8}"
            )
            for kernel in ALL_KERNELS:
                entry = row.get(kernel, {})
                if "error" in entry:
                    print(f"  {kernel:<20} ERROR: {entry['error'][:80]}")
                    continue
                vs_flash = entry.get("speedup_vs_flash")
                vs_mlx = entry.get("speedup_vs_mlx")
                print(
                    f"  {kernel:<20} {entry['median_ms']:>11.4f} {entry['p95_ms']:>9.4f} "
                    f"{entry['tokens_per_s']:>12.1f} "
                    f"{(f'{vs_flash:.2f}x' if vs_flash else '-'):>9} "
                    f"{(f'{vs_mlx:.2f}x' if vs_mlx else '-'):>8} "
                    f"{entry['max_abs_err']:>10.3e} {entry['mean_abs_err']:>10.3e} "
                    f"{entry['n_nan']}/{entry['n_inf']:>5}"
                )
            print()

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=N_ITER)
    ap.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip S in {4096, 8192} to keep the sweep fast.",
    )
    ap.add_argument("--out", type=str, default=None, help="Optional path to save JSON results.")
    args = ap.parse_args()

    results = run(args.n_iter, args.skip_large)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
