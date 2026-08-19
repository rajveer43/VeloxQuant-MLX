"""Energy / throughput harness: FP16 KV vs. VeloxQuant-compressed KV.

Step 1 (experiments A and B) of the energy-aware inference investigation. This
script measures; it does not optimise. No Metal kernel work happens here.

What it measures
----------------
* Decode throughput (tokens/s), with prefill timed separately.
* Peak GPU memory per arm.
* Whole-package energy (J) and J/token -- **only under ``sudo``**, because
  ``powermetrics`` requires root. Unprivileged runs report ``n/a (requires
  sudo)`` for every energy field and stay useful for everything else.

What it does NOT measure
------------------------
* **KV traffic is DERIVED**, computed from cache geometry, not observed. MLX
  exposes no bytes-moved counter. See ``veloxquant_mlx.profiling.energy``.
* Energy is a sampled integration of package power, so other processes on the
  machine contribute, and the sampling interval bounds resolution.

Confound controls
-----------------
* A warm-up run is discarded before every measured arm (first-run Metal
  compilation and page-in would otherwise be charged to whichever arm ran
  first).
* Arms are **interleaved** across repetitions (A,B,C,A,B,C,...) rather than run
  in blocks, so thermal drift on a sustained-load M4 cannot be confounded with
  "this arm used more energy". Medians and spread are reported, not means.

Usage:
    python benchmark_scripts/benchmark_energy.py [MODEL] [--reps N] [--max-tokens N]
    sudo python benchmark_scripts/benchmark_energy.py    # adds J/token
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import KVCache as _FallbackCache

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig  # noqa: E402
from veloxquant_mlx.profiling.energy import (  # noqa: E402
    kv_bytes_per_token,
    measure_generation,
)
from veloxquant_mlx.profiling.power_sampler import PowerSampler  # noqa: E402

# Reuse the established prompt/length convention from benchmark_core.
PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)
MAX_TOKENS = 200
DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
SEED = 42
OUT_JSON = _REPO_ROOT / "energy_benchmark_results.json"

NA = "n/a (requires sudo)"


# ---------------------------------------------------------------------------
# Model geometry
# ---------------------------------------------------------------------------
def _model_head_info(model):
    """Return (head_dim, n_kv_heads, n_heads, n_layers) by inspecting layers."""
    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    head_dim = n_kv = n_heads = None
    for L in layers:
        attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
        if attn is None:
            continue
        head_dim = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        n_kv = getattr(attn, "n_kv_heads", None) or getattr(
            args, "num_key_value_heads", getattr(args, "num_attention_heads", 1)
        )
        n_heads = getattr(attn, "n_heads", None) or getattr(args, "num_attention_heads", n_kv)
        break
    return head_dim, n_kv, n_heads, len(layers)


def _guard_model_size(model) -> None:
    """Refuse to run if weights approach the GPU working-set ceiling.

    Swap thrash would quietly corrupt the very energy numbers being collected,
    so this fails loudly instead.
    """
    try:
        info = mx.device_info()
        ceiling = int(info.get("max_recommended_working_set_size", 0))
    except Exception:
        return
    if not ceiling:
        return
    try:
        active = int(mx.get_active_memory())
    except Exception:
        return
    if active > 0.85 * ceiling:
        raise SystemExit(
            f"ERROR: model resident size {active / 1024**3:.2f} GB exceeds 85% of the "
            f"GPU working-set ceiling ({ceiling / 1024**3:.2f} GB). Swap thrash would "
            f"corrupt the energy measurement. Use a smaller model."
        )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def _build_fp16(model):
    """Arm A: stock mlx_lm cache, FP16 KV."""
    layers = getattr(model, "layers", None) or model.model.layers
    return [_FallbackCache() for _ in layers]


def _arm_specs(head_dim: int):
    """(label, config-or-None, description) for each measured arm.

    Arm C (Metal kernel) is deliberately absent. Its precondition is a profiled
    bottleneck identified from arms A and B; building it before this harness
    reports would be optimising a bottleneck nobody has demonstrated.
    """
    return [
        ("A: fp16 baseline", None, "stock mlx_lm KVCache, FP16 KV"),
        (
            "B1: KIVI 4-bit",
            KVCacheConfig(method="kivi", bit_width_inlier=4, head_dim=head_dim, seed=SEED),
            "quantization -- scales bytes/token by bit ratio",
        ),
        (
            "B2: Q-Filters budget=512",
            KVCacheConfig(method="qfilters", qfilters_budget=512, head_dim=head_dim, seed=SEED),
            "eviction -- caps bytes/token at the budget",
        ),
    ]


def _build_caches(model, config):
    if config is None:
        return _build_fp16(model)
    return KVCacheBuilder.for_model(model, config)


# ---------------------------------------------------------------------------
# One measured run
# ---------------------------------------------------------------------------
def _run_arm(model, ids, config, n_tokens: int, kv_bytes: int, label: str, privileged: bool):
    """Warm up (discarded), then measure one arm."""
    # --- warm-up, discarded: absorbs Metal compilation and page-in ---------
    warm_caches = _build_caches(model, config)
    logits = model(mx.array([ids[:32]]), cache=warm_caches)
    mx.eval(logits)
    del warm_caches
    mx.clear_cache()

    # --- measured run ------------------------------------------------------
    # Held in `state` rather than a bare local: the closures below capture it,
    # and the `del` at the end of this function would otherwise make it an
    # unbound local for them.
    state = {"caches": _build_caches(model, config)}

    def prefill():
        logits = model(mx.array([ids]), cache=state["caches"])
        tok = mx.argmax(logits[0, -1]).reshape(1, 1)
        state["cur"] = tok
        return tok

    def decode_step(_i):
        logits = model(state["cur"], cache=state["caches"])
        state["cur"] = mx.argmax(logits[0, -1]).reshape(1, 1)
        return state["cur"]

    sampler = PowerSampler(interval_ms=100) if privileged else None
    if sampler is not None:
        with sampler:
            m = measure_generation(
                prefill, decode_step, n_tokens, kv_bytes, sampler=sampler, label=label
            )
        # Recompute energy after __exit__ parsed the samples.
        from veloxquant_mlx.profiling.energy import RunMetrics, compute_j_per_token

        energy = sampler.energy_joules()
        power = sampler.mean_power_mw()
        m = RunMetrics(
            tokens_generated=m.tokens_generated,
            wall_s=m.wall_s,
            tokens_per_s=m.tokens_per_s,
            prefill_s=m.prefill_s,
            decode_s=m.decode_s,
            peak_memory_mb=m.peak_memory_mb,
            kv_bytes_per_token=m.kv_bytes_per_token,
            energy_j=energy,
            j_per_token=compute_j_per_token(energy, m.tokens_generated),
            mean_gpu_mw=power.get("gpu"),
            mean_cpu_mw=power.get("cpu"),
            label=label,
        )
    else:
        m = measure_generation(prefill, decode_step, n_tokens, kv_bytes, label=label)

    state.clear()
    mx.clear_cache()
    return m


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value, spec: str = "{:.3f}") -> str:
    """Format a value, or the explicit n/a marker. Never prints a bare dash."""
    if value is None:
        return NA
    return spec.format(value)


def _spread(values: list[float]) -> str:
    if len(values) < 2:
        return "n=1"
    return f"±{(max(values) - min(values)) / 2:.3f}"


def _report(results: dict, privileged: bool, meta: dict) -> None:
    print("\n" + "=" * 92)
    print("ENERGY BENCHMARK -- experiments A and B")
    print("=" * 92)
    print(f"model        : {meta['model']}")
    print(f"hardware     : {meta['device']}  |  MLX {meta['mlx_version']}")
    print(f"reps         : {meta['reps']} (interleaved)   decode tokens: {meta['max_tokens']}")
    print(f"privileged   : {privileged}")
    if not privileged:
        print("\nNOTE: energy fields require root. Re-run with:")
        print("      sudo python benchmark_scripts/benchmark_energy.py")
    print("-" * 92)
    header = f"{'arm':<26} {'tok/s':>9} {'peak MB':>9} {'KV B/tok':>12} {'J':>10} {'J/token':>22}"
    print(header)
    print("-" * 92)

    for label, runs in results.items():
        tps = [r["tokens_per_s"] for r in runs]
        peak = [r["peak_memory_mb"] for r in runs]
        kvb = runs[0]["kv_bytes_per_token"]
        js = [r["energy_j"] for r in runs if r["energy_j"] is not None]
        jpt = [r["j_per_token"] for r in runs if r["j_per_token"] is not None]

        j_str = f"{statistics.median(js):.2f}" if js else NA
        jpt_str = f"{statistics.median(jpt):.5f} {_spread(jpt)}" if jpt else NA
        print(
            f"{label:<26} {statistics.median(tps):>9.2f} "
            f"{statistics.median(peak):>9.1f} {kvb:>12,} {j_str:>10} {jpt_str:>22}"
        )
    print("=" * 92)
    print("KV B/tok is DERIVED from cache geometry, not measured -- MLX exposes")
    print("no bytes-moved counter. Energy is sampled package power integrated over")
    print("wall time, not a hardware energy counter.")
    print("=" * 92)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("--reps", type=int, default=3, help="interleaved repetitions per arm")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help=(
            "Pad the prompt to at least this many tokens. Eviction arms only "
            "differ from the baseline once the sequence exceeds their budget, "
            "so a short prompt makes them look like no-ops."
        ),
    )
    args = ap.parse_args()

    import os

    privileged = os.geteuid() == 0

    mx.random.seed(SEED)
    print(f"Loading {args.model} ...")
    model, tokenizer = mlx_lm.load(args.model)
    _guard_model_size(model)

    head_dim, n_kv, _n_heads, n_layers = _model_head_info(model)
    print(f"geometry: layers={n_layers} kv_heads={n_kv} head_dim={head_dim}")

    try:
        messages = [{"role": "user", "content": PROMPT}]
        prompt_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_txt = PROMPT
    ids = tokenizer.encode(prompt_txt)
    if args.context_tokens and len(ids) < args.context_tokens:
        # Repeat the prompt body to reach the requested context length. The
        # content is irrelevant to a power measurement -- what matters is that
        # every arm sees the identical token sequence at a length where the
        # eviction budget actually binds.
        filler = tokenizer.encode(PROMPT)
        while len(ids) < args.context_tokens:
            ids = ids + filler
        ids = ids[: args.context_tokens]
    seq_len = len(ids) + args.max_tokens
    print(f"prompt tokens={len(ids)}  modelled seq_len={seq_len}")

    specs = _arm_specs(head_dim)
    results: dict[str, list[dict]] = {label: [] for label, _, _ in specs}

    # Interleaved: A,B1,B2, A,B1,B2, ... -- never blocked, so thermal drift
    # cannot be confounded with the arm identity.
    for rep in range(args.reps):
        for label, config, _desc in specs:
            kv_bytes = kv_bytes_per_token(config, n_layers, n_kv, head_dim, seq_len)
            print(f"  rep {rep + 1}/{args.reps}  {label} ...", flush=True)
            m = _run_arm(model, ids, config, args.max_tokens, kv_bytes, label, privileged)
            results[label].append(m.to_dict())

    info = mx.device_info()
    meta = {
        "model": args.model,
        "device": info.get("device_name", "unknown"),
        "mlx_version": _mlx_version(),
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "n_layers": n_layers,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "seq_len_modelled": seq_len,
        "privileged": privileged,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    OUT_JSON.write_text(json.dumps({"meta": meta, "arms": results}, indent=2))
    _report(results, privileged, meta)
    print(f"\nwrote {OUT_JSON}")


def _mlx_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("mlx")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
