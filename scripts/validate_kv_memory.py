#!/usr/bin/env python3
"""End-to-end KV-cache validation: fp16 vs RVQ-1bit vs VecInfer-1bit.

Reports tokens in cache, key/value byte accounting, MLX peak memory,
throughput, and a short output preview. Writes
``figures/validation/<model_stem>/results.json``.

Accounting vs resident memory
-----------------------------
``key_compression`` is ``fp16_key_bytes / compressed_key_bytes`` on the
cache objects. Default RVQ and VecInfer paths quantize then dequantize
into the parent mlx_lm fp16 KVCache, so large accounting ratios can
appear while process RSS barely moves at short context. MLX peak MB
includes weights and activations.

Usage
-----
::

    source .venv/bin/activate
    PYTHONPATH=. python scripts/validate_kv_memory.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
        --max-tokens 128
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)


def _peak_mb() -> float:
    import mlx.core as mx

    try:
        return float(mx.get_peak_memory()) / (1024**2)
    except Exception:
        try:
            return float(mx.metal.get_peak_memory()) / (1024**2)
        except Exception:
            return float("nan")


def _reset_peak() -> None:
    import mlx.core as mx

    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def _model_head_info(model) -> tuple:
    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    head_dim = n_kv = n_heads = None
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
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


def _cache_offsets(caches: list) -> dict[str, Any]:
    offsets = []
    for c in caches:
        off = getattr(c, "offset", None)
        if off is None:
            continue
        try:
            offsets.append(int(off))
        except Exception:
            continue
    if not offsets:
        return {"tokens_in_cache_per_layer": [], "tokens_in_cache_max": 0}
    return {
        "tokens_in_cache_per_layer": offsets,
        "tokens_in_cache_max": max(offsets),
        "tokens_in_cache_mean": float(sum(offsets) / len(offsets)),
    }


def _byte_stats(caches: list) -> dict[str, Any]:
    key_c = key_f = val_c = val_f = residual = 0
    for c in caches:
        if hasattr(c, "compressed_key_bytes"):
            key_c += int(c.compressed_key_bytes)
            key_f += int(c.fp16_key_bytes)
        if hasattr(c, "compressed_value_bytes"):
            val_c += int(getattr(c, "compressed_value_bytes", 0) or 0)
            val_f += int(getattr(c, "fp16_value_bytes", 0) or 0)
        residual += int(getattr(c, "residual_fp16_bytes", 0) or 0)

    key_ratio = (key_f / key_c) if key_c else 1.0
    val_ratio = (val_f / val_c) if val_c else 1.0
    total_fp16 = key_f + val_f
    total_comp = key_c + val_c + residual
    full_kv_ratio = (total_fp16 / total_comp) if total_comp else 1.0
    return {
        "fp16_key_bytes": key_f,
        "compressed_key_bytes": key_c,
        "fp16_key_mb": round(key_f / (1024**2), 3),
        "compressed_key_mb": round(key_c / (1024**2), 3),
        "key_compression": key_ratio,
        "fp16_value_bytes": val_f,
        "compressed_value_bytes": val_c,
        "fp16_value_mb": round(val_f / (1024**2), 3),
        "compressed_value_mb": round(val_c / (1024**2), 3),
        "value_compression": val_ratio,
        "residual_fp16_bytes": residual,
        "full_kv_compression": full_kv_ratio,
        "metric_type": "key_byte_accounting",
        "storage_note": (
            "Default RVQ/VecInfer dequantize into parent fp16 KVCache; "
            "ratios are packed-format accounting unless a fused/packed path is active."
        ),
    }


def _build_fp16(model):
    from mlx_lm.models.cache import KVCache as _FB

    layers = getattr(model, "layers", None) or model.model.layers
    return [_FB() for _ in layers]


def _build_rvq(model, bits: int = 1):
    from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

    cfg = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=bits, seed=42)
    return KVCacheBuilder.for_model(model, cfg)


def _vecinfer_artifacts(
    head_dim: int,
    n_heads: int,
    key_bits: int,
    value_bits: int,
    key_sub_dim: int,
    value_sub_dim: int,
    cache_dir: Path,
    seed: int = 42,
) -> dict:
    import mlx.core as mx
    from veloxquant_mlx.allocators.vecinfer import (
        calibrate_smooth_factors,
        train_codebook,
    )

    sig = f"hd{head_dim}_h{n_heads}_kb{key_bits}_vb{value_bits}_ks{key_sub_dim}_vs{value_sub_dim}"
    path = cache_dir / f"{sig}.npz"
    if path.exists():
        data = np.load(path)
        return {
            "smooth": mx.array(data["smooth"]),
            "key_codebook": mx.array(data["key_cb"]),
            "value_codebook": mx.array(data["value_cb"]),
        }

    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((4096, n_heads, head_dim)).astype(np.float32))
    V = mx.array(rng.standard_normal((4096, n_heads, head_dim)).astype(np.float32))
    smooth = calibrate_smooth_factors(K)
    k_subs = mx.array(np.asarray(K).reshape(-1, key_sub_dim))
    v_subs = mx.array(np.asarray(V).reshape(-1, value_sub_dim))
    n_train = min(8000, k_subs.shape[0])
    key_cb = train_codebook(k_subs[:n_train], 2**key_bits, max_iter=15, seed=seed)
    val_cb = train_codebook(v_subs[:n_train], 2**value_bits, max_iter=15, seed=seed + 1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        smooth=np.asarray(smooth),
        key_cb=np.asarray(key_cb),
        value_cb=np.asarray(val_cb),
    )
    return {"smooth": smooth, "key_codebook": key_cb, "value_codebook": val_cb}


def _build_vecinfer_1bit(model, model_stem: str):
    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
    from mlx_lm.models.cache import KVCache as _FB

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    head_dim, _n_kv, n_heads, _ = _model_head_info(model)
    if head_dim is None or n_heads is None:
        raise RuntimeError("Could not resolve head_dim / n_heads for VecInfer")

    # 1-bit product VQ at sub_dim=8 => 16x key accounting when head_dim % 8 == 0
    key_sub_dim = 8 if head_dim % 8 == 0 else (4 if head_dim % 4 == 0 else 2)
    value_sub_dim = key_sub_dim
    key_bits = 8
    value_bits = 8

    cache_root = Path(os.path.expanduser("~/.cache/veloxquant/vecinfer")) / model_stem
    art = _vecinfer_artifacts(
        head_dim, n_heads, key_bits, value_bits, key_sub_dim, value_sub_dim, cache_root
    )

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            caches.append(_FB())
            continue
        hd = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        if hd is None or hd % key_sub_dim != 0 or hd % value_sub_dim != 0:
            caches.append(_FB())
            continue
        cfg = KVCacheConfig(
            method="vecinfer",
            head_dim=hd,
            key_sub_dim=key_sub_dim,
            value_sub_dim=value_sub_dim,
            key_codebook_bits=key_bits,
            value_codebook_bits=value_bits,
            smooth_factors=art["smooth"],
            key_codebook=art["key_codebook"],
            value_codebook=art["value_codebook"],
            seed=42 + i,
            use_metal_kernels=None,
        )
        caches.append(VecInferKVCache(cfg))
    return caches


def _run_one(
    model,
    tokenizer,
    label: str,
    builder: Callable,
    max_tokens: int,
    prompt: str,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx_lm

    print(f"  [{label}] generating...", flush=True)
    try:
        caches = builder()
    except Exception as e:
        traceback.print_exc()
        return {
            "name": label,
            "error": f"builder: {e}",
            "throughput_tok_s": 0.0,
            "peak_mb": float("nan"),
            "tokens_generated": 0,
            "elapsed_s": 0.0,
        }

    messages = [{"role": "user", "content": prompt}]
    try:
        prompt_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_txt = prompt

    _reset_peak()
    mx.clear_cache()

    t0 = time.perf_counter()
    try:
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt_txt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=caches,
        )
    except Exception as e:
        traceback.print_exc()
        return {
            "name": label,
            "error": str(e),
            "throughput_tok_s": 0.0,
            "peak_mb": float("nan"),
            "tokens_generated": 0,
            "elapsed_s": 0.0,
        }
    elapsed = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(response)) if response else 0
    preview = (response or "")[:240]

    out: dict[str, Any] = {
        "name": label,
        "throughput_tok_s": n_tok / max(elapsed, 1e-6),
        "peak_mb": _peak_mb(),
        "tokens_generated": n_tok,
        "elapsed_s": elapsed,
        "output_preview": preview,
    }
    out.update(_cache_offsets(caches))
    out.update(_byte_stats(caches))

    k_fp = out.get("fp16_key_mb", 0.0)
    k_c = out.get("compressed_key_mb", 0.0)
    k_x = out.get("key_compression", 1.0)
    print(
        f"    {n_tok} tok in {elapsed:.1f}s ({out['throughput_tok_s']:.1f} tok/s) "
        f"peak={out['peak_mb']:.0f}MB "
        f"cache_tokens={out.get('tokens_in_cache_max', 0)} "
        f"keys={k_fp:.3f}MB(fp16)->{k_c:.3f}MB(acct) "
        f"key_claim={k_x:.2f}x",
        flush=True,
    )
    return out


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        import subprocess

        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["chip"] = brand
        info["unified_ram_gb"] = round(mem / (1024**3), 1)
    except Exception:
        pass
    try:
        import mlx

        info["mlx_version"] = getattr(mlx, "__version__", "unknown")
    except Exception:
        info["mlx_version"] = None
    try:
        import mlx_lm

        info["mlx_lm_version"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        info["mlx_lm_version"] = None
    try:
        import veloxquant_mlx

        info["veloxquant_version"] = getattr(veloxquant_mlx, "__version__", "unknown")
    except Exception:
        info["veloxquant_version"] = None
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="Repeat the base prompt this many times to grow prefill length",
    )
    parser.add_argument(
        "--skip-vecinfer",
        action="store_true",
        help="Only run fp16 and RVQ (faster smoke test)",
    )
    args = parser.parse_args()

    from mlx_lm import load
    import mlx.core as mx

    model_id = args.model
    model_stem = model_id.split("/")[-1]
    out_dir = REPO_ROOT / "figures" / "validation" / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = " ".join([PROMPT] * max(1, args.prompt_repeat))

    print(f"Loading {model_id}...", flush=True)
    model, tokenizer = load(model_id)
    head_dim, n_kv, n_heads, n_layers = _model_head_info(model)
    print(
        f"  head_dim={head_dim} n_kv_heads={n_kv} n_q_heads={n_heads} n_layers={n_layers}",
        flush=True,
    )

    configs: list[tuple[str, Callable]] = [
        ("fp16-baseline", lambda: _build_fp16(model)),
        ("RVQ-1bit", lambda: _build_rvq(model, 1)),
    ]
    if not args.skip_vecinfer:
        configs.append(("VecInfer-1bit", lambda: _build_vecinfer_1bit(model, model_stem)))

    results = []
    for label, builder in configs:
        results.append(_run_one(model, tokenizer, label, builder, args.max_tokens, prompt))
        mx.clear_cache()

    payload = {
        "schema": "veloxquant_validation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "head_dim": head_dim,
        "n_kv_heads": n_kv,
        "n_q_heads": n_heads,
        "n_layers": n_layers,
        "max_tokens": args.max_tokens,
        "prompt_repeat": args.prompt_repeat,
        "prompt": prompt,
        "hardware": _hardware_info(),
        "honesty": {
            "key_compression": "byte accounting on cache counters",
            "peak_mb": "mx.get_peak_memory (includes weights + activations)",
            "resident_rss": "not measured by this script",
        },
        "results": results,
    }

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
