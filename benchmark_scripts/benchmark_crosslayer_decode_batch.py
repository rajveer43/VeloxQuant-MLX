"""Cross-layer batched decode-attend dispatch benchmark (issue #307 part 1).

Extends ``scripts/kv_kernel_roofline_bench.py``'s self-calibrating
bandwidth-peak methodology (never trust a spec-sheet number — measure the
machine's own achieved ceiling on the same run) to answer the one question
``docs/KV_KERNEL_ROOFLINE_FINDINGS.md``'s occupancy sweep left open:
does batching the independent per-layer
:func:`scalar_fused_decode_attend` calls a transformer decode step already
makes (one per layer, typically 28-80 layers) into a single
:func:`scalar_fused_decode_attend_batched` dispatch raise achieved
bandwidth by raising threadgroup count, without changing total bytes moved
or total FLOPs?

This benchmark's output IS the deliverable finding, not a side artifact —
see the third addendum in ``docs/KV_KERNEL_ROOFLINE_FINDINGS.md`` for the
write-up built from these numbers.

Reports, as separate line items (a real win must survive all of them):
  1. NL sequential single-layer calls vs. one batched call, at realistic
     decode shapes spanning current open-weight model depths.
  2. Achieved bandwidth against a live-recalibrated peak (this run's
     machine, not a hardcoded historical figure).
  3. The ``mx.stack`` cost of producing the batched layout from NL
     independent per-layer arrays, timed standalone.
  4. Whether ``mlx_lm``'s stock ``KVCache`` naturally produces a
     layer-stacked buffer (it does not — see the printed finding — each
     layer gets its own independent ``KVCache`` instance in a Python list,
     confirmed by reading ``mlx_lm.models.cache``), which determines
     whether the stacking cost is a one-time layout change or a per-step
     tax on every decode step in a real integration.

Usage: python benchmark_scripts/benchmark_crosslayer_decode_batch.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal._scalar_attend import (
    scalar_fused_decode_attend,
    scalar_fused_decode_attend_batched,
)

N_WARMUP, N_ITER = 10, 20


def _bench(fn, n_warmup: int = N_WARMUP, n_iter: int = N_ITER) -> float:
    for _ in range(n_warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / n_iter


def _calibrate_bandwidth_peak() -> float:
    """Achieved GB/s on a large elementwise op — mirrors
    ``kv_kernel_roofline_bench.py``'s ``_calibrate_bandwidth_peak`` exactly,
    recalibrated live rather than reusing that script's historical number,
    since this must be comparable to *this* run's kernel measurements."""
    best = 0.0
    for n in (100_000_000, 200_000_000, 300_000_000):
        a = mx.random.normal((n,)).astype(mx.float16)
        mx.eval(a)
        t = _bench(lambda a=a: a * 2.0, n_warmup=3, n_iter=8)
        bytes_moved = n * 2 * 2  # fp16 read + fp16 write
        best = max(best, bytes_moved / t / 1e9)
    return best


def _make_layer(B, H_q, H_kv, S_kv, D, group, seed):
    rng = np.random.default_rng(seed)
    GK = (S_kv + group - 1) // group
    GV = (D + group - 1) // group
    q = mx.array(rng.standard_normal((B, H_q, 1, D)).astype(np.float16))
    k_codes = mx.array(rng.integers(0, 16, size=(B, H_kv, S_kv, D)).astype(np.uint8))
    k_scale = mx.array(rng.uniform(0.01, 0.1, size=(B, H_kv, GK, D)).astype(np.float32))
    k_zero = mx.array(rng.uniform(-1, 1, size=(B, H_kv, GK, D)).astype(np.float32))
    v_codes = mx.array(rng.integers(0, 16, size=(B, H_kv, S_kv, D)).astype(np.uint8))
    v_scale = mx.array(rng.uniform(0.01, 0.1, size=(B, H_kv, S_kv, GV)).astype(np.float32))
    v_zero = mx.array(rng.uniform(-1, 1, size=(B, H_kv, S_kv, GV)).astype(np.float32))
    mx.eval(q, k_codes, k_scale, k_zero, v_codes, v_scale, v_zero)
    return q, k_codes, k_scale, k_zero, v_codes, v_scale, v_zero


def bench_primary_comparison(peak_gbs: float) -> None:
    """NL sequential calls vs. one batched call, across realistic decode
    shapes. Layer counts cite actual open-weight model depths: Llama-3-8B
    (32), Llama-3-70B/Qwen2-72B-class (80), Mistral-7B/Llama-2-7B-class
    (32), a shallow 28-layer class (Qwen2-7B), and a 48-layer mid-size
    class — not invented round numbers."""
    print("=" * 100)
    print("Primary comparison: NL sequential scalar_fused_decode_attend calls vs.")
    print("one scalar_fused_decode_attend_batched call")
    print("=" * 100)
    D, group = 128, 32
    scale = 1.0 / float(D) ** 0.5
    B = 1
    print(
        f"{'H_kv':>4} {'H_q/H_kv':>8} {'S_kv':>6} {'NL':>3} "
        f"{'seq ms':>9} {'batch ms':>9} {'speedup':>8} "
        f"{'seq GB/s':>9} {'seq %pk':>8} {'batch GB/s':>10} {'batch %pk':>9}"
    )
    for H_kv in (2, 4, 8):
        for ratio in (1, 4, 8):
            H_q = H_kv * ratio
            for S_kv in (128, 2048, 16384):
                GK = (S_kv + group - 1) // group
                GV = (D + group - 1) // group
                bytes_per_layer = (
                    2 * B * H_kv * S_kv * D  # k_codes + v_codes, 1B each
                    + (B * H_kv * GK * D + B * H_kv * S_kv * GV) * 4 * 2  # scale+zero fp32
                )
                for NL in (28, 32, 48, 80):
                    layers = [
                        _make_layer(B, H_q, H_kv, S_kv, D, group, seed=1000 * NL + layer_idx)
                        for layer_idx in range(NL)
                    ]

                    def _sequential(layers=layers):
                        outs = [
                            scalar_fused_decode_attend(*layer, group, scale, nsg=2)
                            for layer in layers
                        ]
                        return mx.stack(outs, axis=0)

                    q_b = mx.stack([layer[0] for layer in layers], axis=0)
                    kc_b = mx.stack([layer[1] for layer in layers], axis=0)
                    ks_b = mx.stack([layer[2] for layer in layers], axis=0)
                    kz_b = mx.stack([layer[3] for layer in layers], axis=0)
                    vc_b = mx.stack([layer[4] for layer in layers], axis=0)
                    vs_b = mx.stack([layer[5] for layer in layers], axis=0)
                    vz_b = mx.stack([layer[6] for layer in layers], axis=0)
                    mx.eval(q_b, kc_b, ks_b, kz_b, vc_b, vs_b, vz_b)

                    def _batched(
                        q_b=q_b, kc_b=kc_b, ks_b=ks_b, kz_b=kz_b, vc_b=vc_b, vs_b=vs_b, vz_b=vz_b
                    ):
                        return scalar_fused_decode_attend_batched(
                            q_b, kc_b, ks_b, kz_b, vc_b, vs_b, vz_b, group, scale, nsg=2
                        )

                    ts = _bench(_sequential)
                    tb = _bench(_batched)
                    total_bytes = bytes_per_layer * NL
                    seq_gbs = total_bytes / ts / 1e9
                    batch_gbs = total_bytes / tb / 1e9
                    print(
                        f"{H_kv:>4} {ratio:>8} {S_kv:>6} {NL:>3} "
                        f"{ts * 1e3:>9.3f} {tb * 1e3:>9.3f} {ts / tb:>7.2f}x "
                        f"{seq_gbs:>9.1f} {100 * seq_gbs / peak_gbs:>7.1f}% "
                        f"{batch_gbs:>10.1f} {100 * batch_gbs / peak_gbs:>8.1f}%"
                    )
    print()


def bench_stacking_cost() -> None:
    """mx.stack cost over NL per-layer arrays, timed standalone — NOT folded
    into the batched kernel's own number above, so the win/loss reported
    there isn't accidentally flattering itself by ignoring what it costs to
    produce its own input layout."""
    print("=" * 100)
    print("mx.stack cost (standalone) — the tax a real integration must pay to")
    print("produce the batched layout from independent per-layer arrays")
    print("=" * 100)
    D, group = 128, 32
    B, H_kv, ratio = 1, 4, 8
    H_q = H_kv * ratio
    print(f"{'S_kv':>6} {'NL':>3} {'stack ms':>10}")
    for S_kv in (128, 2048, 16384):
        for NL in (28, 32, 48, 80):
            layers = [
                _make_layer(B, H_q, H_kv, S_kv, D, group, seed=2000 * NL + layer_idx)
                for layer_idx in range(NL)
            ]

            def _stack_all(layers=layers):
                q_b = mx.stack([layer[0] for layer in layers], axis=0)
                kc_b = mx.stack([layer[1] for layer in layers], axis=0)
                ks_b = mx.stack([layer[2] for layer in layers], axis=0)
                kz_b = mx.stack([layer[3] for layer in layers], axis=0)
                vc_b = mx.stack([layer[4] for layer in layers], axis=0)
                vs_b = mx.stack([layer[5] for layer in layers], axis=0)
                vz_b = mx.stack([layer[6] for layer in layers], axis=0)
                return q_b, kc_b, ks_b, kz_b, vc_b, vs_b, vz_b

            def _stack_eval():
                arrs = _stack_all()
                mx.eval(*arrs)
                return arrs[0]

            t = _bench(_stack_eval)
            print(f"{S_kv:>6} {NL:>3} {t * 1e3:>10.3f}")
    print()


def report_mlx_lm_kvcache_layout() -> None:
    """Whether mlx_lm's stock KVCache naturally produces a layer-stacked
    buffer — verified by reading the source, not assumed. This determines
    whether the mx.stack cost above is a one-time layout change or a
    per-decode-step tax in a real end-to-end integration."""
    print("=" * 100)
    print("mlx_lm KVCache layout finding")
    print("=" * 100)
    try:
        import inspect

        from mlx_lm.models import cache as mlx_lm_cache

        make_cache_src = inspect.getsource(mlx_lm_cache)
        per_layer_list = "[KVCache() for _ in range(num_layers)]" in make_cache_src.replace(
            " ", ""
        ).replace("\n", "")
    except Exception as e:  # pragma: no cover - diagnostic path only
        print(f"Could not introspect mlx_lm.models.cache: {e}")
        return
    print(
        "mlx_lm.models.cache.make_prompt_cache / model make_cache() instantiate\n"
        "one independent KVCache() object PER LAYER, held in a plain Python list\n"
        "(e.g. `[KVCache() for _ in range(len(self.model.layers))]` per-model in\n"
        "models/*.py, and cache.py's own default). Each KVCache owns its own\n"
        "`.keys`/`.values` arrays with independent step-growth buffers — there is\n"
        "no shared, layer-stacked backing buffer anywhere in the stock cache.\n"
    )
    print(
        f"[introspection check] per-layer KVCache list pattern found: {per_layer_list}\n"
        if per_layer_list
        else "[introspection check] expected per-layer list pattern NOT found — "
        "re-verify against the installed mlx_lm version before trusting the "
        "conclusion above.\n"
    )
    print(
        "Consequence: a real end-to-end integration of the batched kernel would\n"
        "need to mx.stack() each layer's .keys/.values (and the corresponding\n"
        "quantized codes/scale/zero, if quantized) fresh on EVERY decode step,\n"
        "since nothing in the stock cache keeps them contiguously stacked between\n"
        "steps. The mx.stack cost measured above is therefore a PER-STEP TAX in\n"
        "any integration built on the stock KVCache, not a one-time layout change\n"
        "paid once at cache-construction time. This is a real integration cost —\n"
        "reported here as required future-work context per the implementation\n"
        "prompt's Phase 4, not resolved by this benchmark."
    )
    print()


def main() -> None:
    print("[roofline] calibrating achieved memory-bandwidth peak on this GPU...")
    peak_gbs = _calibrate_bandwidth_peak()
    print(f"[roofline] calibrated peak: {peak_gbs:.1f} GB/s (achieved, not spec-sheet)\n")

    bench_primary_comparison(peak_gbs)
    bench_stacking_cost()
    report_mlx_lm_kvcache_layout()

    print(
        "Reading this benchmark: the primary comparison table is the deliverable\n"
        "finding for issue #307 part 1 (cross-layer batched decode-attend\n"
        "dispatch) — whatever it shows for a given (H_kv, ratio, S_kv, NL) cell,\n"
        "win or null, is what should be written into\n"
        "docs/KV_KERNEL_ROOFLINE_FINDINGS.md's third addendum, together with the\n"
        "stacking-cost table and the mlx_lm KVCache layout finding above. Do NOT\n"
        "read a kernel-only speedup as a net win without netting out the stacking\n"
        "cost per decode step, since the KVCache finding above shows that cost is\n"
        "paid on every step in a real integration, not once."
    )


if __name__ == "__main__":
    main()
