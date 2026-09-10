"""Peak-memory check for KIVI: does the 'compressed' cache actually reduce
live process/GPU memory, or does it store fp16 the whole time and merely
report a compression ratio?

Also checks: does long-running decode (repeated evictions) leak memory /
plateau, for TOVA and H2O (proxy for the 20k-op long-loop requirement).

Run: .venv/bin/python scripts/kv_bookkeeping_memory_check.py --output /tmp/kv_memcheck.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory


def _peak_mb():
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except Exception:
        return None


def _active_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        return None


def kivi_dtype_check():
    """Inspect the actual dtype/shape of self.keys/self.values inside a live
    KIVIKVCache after quantization has run, to check if it stores compressed
    bytes or fp16 round-tripped values."""
    cfg = KVCacheConfig(
        method="kivi", head_dim=64, bit_width_inlier=2, kivi_group_size=32, residual_length=32
    )
    cache = KVCacheFactory.create(cfg)
    B, H, D = 1, 4, 64
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = mx.array(rng.normal(size=(B, H, 1, D)).astype(np.float16))
        v = mx.array(rng.normal(size=(B, H, 1, D)).astype(np.float16))
        cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)
    return dict(
        keys_dtype=str(cache.keys.dtype),
        values_dtype=str(cache.values.dtype),
        keys_shape=list(cache.keys.shape),
        keys_nbytes_live=int(cache.keys.nbytes),
        values_nbytes_live=int(cache.values.nbytes),
        reported_compressed_key_bytes=cache.compressed_key_bytes,
        reported_fp16_key_bytes=cache.fp16_key_bytes,
        reported_effective_compression_ratio=cache.effective_compression_ratio,
        note=(
            "keys_nbytes_live is the ACTUAL live tensor size backing "
            "cache.keys (what SDPA reads and what occupies GPU/unified "
            "memory at this instant) -- compare against "
            "reported_compressed_key_bytes, which is a separate byte-"
            "accounting estimate that does NOT correspond to any tensor "
            "actually stored."
        ),
    )


def long_loop_plateau_check(method, n_ops, budget, B=1, H=4, D=64):
    cfg_kwargs = dict(head_dim=D)
    if method == "tova":
        cfg_kwargs.update(method="tova", tova_budget=budget, tova_n_sink=4, tova_backend="mlx")
    elif method == "h2o":
        cfg_kwargs.update(
            method="h2o", h2o_budget=budget, h2o_n_sink=4, h2o_grace=16, h2o_decay=0.98
        )
    cfg = KVCacheConfig(**cfg_kwargs)
    cache = KVCacheFactory.create(cfg)
    rng = np.random.default_rng(1)

    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    samples = []
    checkpoint_every = max(1, n_ops // 20)
    for i in range(n_ops):
        k = mx.array(rng.normal(size=(B, H, 1, D)).astype(np.float16))
        v = mx.array(rng.normal(size=(B, H, 1, D)).astype(np.float16))
        ko, vo = cache.update_and_fetch(k, v)
        if i % checkpoint_every == 0:
            mx.eval(ko, vo)
            gc.collect()
            samples.append(dict(step=i, active_mb=_active_mb(), peak_mb=_peak_mb()))
    mx.eval(cache.keys, cache.values)
    samples.append(dict(step=n_ops, active_mb=_active_mb(), peak_mb=_peak_mb()))
    # Plateau check: compare last 25% of samples' active_mb range
    tail = samples[len(samples) * 3 // 4 :]
    tail_vals = [s["active_mb"] for s in tail if s["active_mb"] is not None]
    plateaued = (max(tail_vals) - min(tail_vals) < 2.0) if len(tail_vals) > 1 else None
    return dict(
        method=method,
        n_ops=n_ops,
        budget=budget,
        final_n_kept=int(cache.keys.shape[2]) if cache.keys is not None else 0,
        samples=samples,
        tail_plateaued_within_2mb=plateaued,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-loop-ops", type=int, default=20000)
    args = parser.parse_args()

    out = {}
    print("KIVI dtype/live-tensor check...")
    out["kivi_dtype_check"] = kivi_dtype_check()
    print(json.dumps(out["kivi_dtype_check"], indent=2))

    print(f"TOVA long-loop plateau check ({args.long_loop_ops} ops)...")
    out["tova_long_loop"] = long_loop_plateau_check("tova", args.long_loop_ops, budget=256)
    print("plateaued:", out["tova_long_loop"]["tail_plateaued_within_2mb"])

    print(f"H2O long-loop plateau check ({args.long_loop_ops} ops)...")
    out["h2o_long_loop"] = long_loop_plateau_check("h2o", args.long_loop_ops, budget=256)
    print("plateaued:", out["h2o_long_loop"]["tail_plateaued_within_2mb"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
