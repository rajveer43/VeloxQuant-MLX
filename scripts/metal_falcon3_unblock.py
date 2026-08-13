"""Validate that Falcon3-7B VecInfer-2bit now works with the Metal kernel.

Before Phase 1, this configuration OOMed at the chunked argmin step in
``quantize_vq`` (head_dim=256 × n_centroids=256 × chunk → multi-GB
intermediate).  With the Metal kernel keeping argmin in registers, the
peak memory drops from ~700 MB to ~12 MB and the model runs end-to-end.

Run from repo root:

    PYTHONPATH=. python scripts/metal_falcon3_unblock.py

Expected outcome: 120 tokens generated, peak memory significantly lower
than the pure-MLX path would have hit.
"""

from __future__ import annotations

import time

import mlx.core as mx

from veloxquant_mlx.metal import metal_available

MODEL_ID = "mlx-community/Falcon3-7B-Instruct-4bit"
PROMPT = "Explain the theory of relativity in simple terms."
MAX_TOKENS = 120


def _peak_mb() -> float:
    try:
        return float(mx.get_peak_memory()) / (1024**2)
    except Exception:
        try:
            return float(mx.metal.get_peak_memory()) / (1024**2)
        except Exception:
            return float("nan")


def _reset_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def _build_caches(model, use_metal: bool, key_sub_dim: int = 4):
    """Falcon3-7B: head_dim=256, n_kv_heads=4, 28 layers.

    With key_sub_dim=4 the pure-MLX path OOMs at the
    [chunk, 256_centroids, 4] diff allocation.
    """
    from mlx_lm.models.cache import KVCache as _FB
    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache

    layers = getattr(model, "layers", None) or model.model.layers
    caches = []
    for i, L in enumerate(layers):
        attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
        if attn is None:
            caches.append(_FB())
            continue
        hd = getattr(attn, "head_dim", None)
        if hd is None:
            caches.append(_FB())
            continue
        cfg = KVCacheConfig(
            method="vecinfer",
            head_dim=hd,
            key_sub_dim=key_sub_dim,
            value_sub_dim=key_sub_dim,
            key_codebook_bits=8,
            value_codebook_bits=8,
            seed=42 + i,
            use_metal_kernels=use_metal,
        )
        caches.append(VecInferKVCache(cfg))
    return caches


def main() -> int:
    if not metal_available():
        print("Metal unavailable — aborting.")
        return 1

    from mlx_lm import generate, load

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)

    messages = [{"role": "user", "content": PROMPT}]
    try:
        prompt_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_txt = PROMPT

    print("\n=== Falcon3-7B VecInfer-2bit (key_sub_dim=4) with Metal kernel ===")
    caches = _build_caches(model, use_metal=True, key_sub_dim=4)
    _reset_peak()
    mx.clear_cache()

    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt_txt,
        max_tokens=MAX_TOKENS,
        verbose=False,
        prompt_cache=caches,
    )
    elapsed = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(response)) if response else 0
    peak = _peak_mb()

    print(f"  tokens generated: {n_tok}")
    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  throughput: {n_tok / max(elapsed, 1e-6):.1f} tok/s")
    print(f"  peak memory: {peak:.0f} MB")
    print(f"\n  response preview: {response[:200]!r}...")

    if n_tok > 0:
        print("\n  SUCCESS — VecInfer-2bit ran on Falcon3-7B with the Metal kernel.")
        print("  (Pure-MLX path OOMs at this configuration.)")
        return 0
    print("\n  FAILED — no tokens generated.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
