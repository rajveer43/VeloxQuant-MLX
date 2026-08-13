"""End-to-end benchmark: KVSink-adapted sink protection vs plain KIVI vs fp16.

Measures throughput, peak memory, and realized compression for KIVI-2bit
with and without key-norm sink protection (``method="kivi_sink"``), against
a real-timed fp16 baseline, on a long prompt (so the quantized path actually
exercises — see benchmark_kivi.py).  Writes ``results.json`` in the
established schema plus a summary plot under ``figures/kivi_sink/<model>/``.

The k-sweep (k=5, k=20) shows the protection-cost curve: each protected
token costs fp16 storage, so compression decreases slightly as k grows.

Honest scope: without a perplexity harness, quality evidence here is limited
to tokens-generated (a coherence proxy); the reconstruction-quality claims
are covered by the unit tests on planted-sink data
(tests/cache/test_sink_cache.py), not by this script.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_sink.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx

# Same long-context prompt construction as benchmark_kivi.py.
_PASSAGE = (
    "The key-value cache stores the attention keys and values of every past "
    "token so the model need not recompute them. Its size grows linearly with "
    "context length and, on Apple Silicon unified memory, it competes with the "
    "model weights and the operating system for the same pool. "
)
PROMPT = (_PASSAGE * 40) + (
    "\n\nGiven the passage above, explain in simple terms why the KV cache is "
    "the binding memory constraint for long-context inference on Apple Silicon, "
    "covering both the linear growth and the unified-memory contention."
)


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _peak_mb() -> float:
    try:
        return float(mx.metal.get_peak_memory()) / (1024**2)
    except Exception:
        return float("nan")


def _reset_peak() -> None:
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def _hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        import subprocess

        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if chip:
            info["chip"] = chip
        if mem:
            info["ram_gb"] = round(int(mem) / (1024**3), 1)
    except Exception:
        pass
    return info


def _build_caches(
    model, method: str, b: int, group_size: int, residual_length: int, n_sink: int
) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache
    from veloxquant_mlx import KVCacheConfig, KVCacheFactory

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            caches.append(_FallbackCache())
            continue
        hd = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        if hd is None:
            caches.append(_FallbackCache())
            continue
        cfg = KVCacheConfig(
            method=method,
            head_dim=hd,
            bit_width_inlier=b,
            kivi_group_size=group_size,
            residual_length=residual_length,
            n_sink_tokens=n_sink,
            seed=42 + i,
        )
        caches.append(KVCacheFactory.create(cfg))
    return caches


def _build_fp16_caches(model) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    layers = getattr(model, "layers", None) or model.model.layers
    return [_FallbackCache() for _ in layers]


def _generate(model, tokenizer, max_tokens: int, caches: list) -> tuple:
    from mlx_lm import generate

    t0 = time.time()
    out = generate(
        model, tokenizer, prompt=PROMPT, max_tokens=max_tokens, verbose=False, prompt_cache=caches
    )
    elapsed = time.time() - t0
    n_tok = len(tokenizer.encode(out)) if out else 0
    return n_tok, elapsed


def _run_config(model, tokenizer, name: str, build_fn, max_tokens: int) -> dict:
    print(f"\n--- {name} ---", flush=True)
    _reset_peak()
    caches = build_fn()
    n_tok, elapsed = _generate(model, tokenizer, max_tokens, caches)
    throughput = n_tok / max(elapsed, 1e-6)
    peak_mb = _peak_mb()

    key_c = key_f = val_c = val_f = resid = sink = 0
    for c in caches:
        if hasattr(c, "compressed_key_bytes"):
            key_c += c.compressed_key_bytes
            key_f += c.fp16_key_bytes
            val_c += getattr(c, "compressed_value_bytes", 0)
            val_f += getattr(c, "fp16_value_bytes", 0)
            resid += getattr(c, "residual_fp16_bytes", 0)
            sink += getattr(c, "sink_fp16_bytes", 0)

    key_ratio = (key_f / key_c) if key_c else 1.0
    total_c = key_c + val_c + resid + sink
    full_kv = ((key_f + val_f) / total_c) if total_c else 1.0

    print(
        f"  {n_tok} tok in {elapsed:.2f}s ({throughput:.1f} tok/s)  "
        f"peak={peak_mb:.0f}MB  key_x={key_ratio:.2f}  fullKV_x={full_kv:.2f}  "
        f"sink_fp16={sink}B"
    )
    return {
        "name": name,
        "throughput_tok_s": throughput,
        "peak_mb": peak_mb,
        "key_compression": key_ratio,
        "full_kv_compression": full_kv,
        "sink_fp16_bytes": sink,
        "tokens_generated": n_tok,
        "elapsed_s": elapsed,
    }


def _plot(results: list, out_path: Path, model_label: str, hw: dict) -> None:
    names = [r["name"] for r in results]
    colors = ["#666", "#00d4ff", "#22c55e", "#7c3aed"][: len(names)]
    chip = hw.get("chip", "Apple Silicon")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"KVSink-adapted sink protection — {model_label} ({chip})", fontsize=13)
    for ax, key, title in [
        (axes[0], "throughput_tok_s", "Throughput (tok/s)"),
        (axes[1], "full_kv_compression", "Full-KV compression (×)"),
        (axes[2], "key_compression", "Key compression (×)"),
    ]:
        ax.bar(names, [r[key] for r in results], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> int:
    _ensure_path()
    p = argparse.ArgumentParser(description="KVSink-adapted benchmark")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--residual-length", type=int, default=32)
    args = p.parse_args()

    from mlx_lm import load

    model_stem = args.model.split("/")[-1]
    out_dir = Path("figures/kivi_sink") / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer = load(args.model)
    layers = getattr(model, "layers", None) or model.model.layers
    margs = getattr(model, "args", None) or model.model.args
    first_attn = next(
        (
            getattr(L, "self_attn", None) or getattr(L, "attn", None)
            for L in layers
            if (getattr(L, "self_attn", None) or getattr(L, "attn", None))
        ),
        None,
    )
    head_dim = getattr(first_attn, "head_dim", None) or (
        margs.hidden_size // margs.num_attention_heads
    )
    n_kv = getattr(margs, "num_key_value_heads", None) or getattr(margs, "num_attention_heads", 1)
    hw = _hardware()
    prompt_tokens = len(tokenizer.encode(PROMPT))
    print(
        f"  head_dim={head_dim} kv_heads={n_kv} layers={len(layers)} "
        f"prompt_tok={prompt_tokens} hw={hw}"
    )

    runs = [
        ("fp16-baseline", lambda: _build_fp16_caches(model)),
        (
            "KIVI-2bit",
            lambda: _build_caches(model, "kivi", 2, args.group_size, args.residual_length, 0),
        ),
        (
            "KIVI-2bit+sink-k5",
            lambda: _build_caches(model, "kivi_sink", 2, args.group_size, args.residual_length, 5),
        ),
        (
            "KIVI-2bit+sink-k20",
            lambda: _build_caches(model, "kivi_sink", 2, args.group_size, args.residual_length, 20),
        ),
    ]
    results = [_run_config(model, tokenizer, n, f, args.max_tokens) for n, f in runs]

    _plot(results, out_dir / "sink_summary.png", model_stem, hw)
    payload = {
        "model": args.model,
        "head_dim": head_dim,
        "n_kv_heads": n_kv,
        "n_layers": len(layers),
        "max_tokens": args.max_tokens,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "prompt_tokens": prompt_tokens,
        "prompt": PROMPT[:200] + "...",
        "hardware": hw,
        "results": results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults: {out_dir / 'results.json'}")
    for r in results:
        print(
            f"  {r['name']:<22s} {r['throughput_tok_s']:6.1f} tok/s  "
            f"key_x={r['key_compression']:.2f}  fullKV_x={r['full_kv_compression']:.2f}  "
            f"toks={r['tokens_generated']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
