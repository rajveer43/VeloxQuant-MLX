"""End-to-end real-model benchmark for the fused packed-RVQ key decode.

Compares decode tokens/sec, prefill tokens/sec and peak memory for
``turboquant_rvq`` with the fused ``rvq_unpack_decode`` kernel off and on
(``use_metal_kernels`` False/True, everything else identical), against the
baselines that actually run in production: plain fp16 ``KVCache`` and
mlx_lm's ``QuantizedKVCache(bits=4)``. Sweeps prompt length, because a
single short-context point hides the effect.

    python scripts/rvq_decode_e2e_bench.py
    python scripts/rvq_decode_e2e_bench.py --prompt-lens 256 4096 --bits 2

Hardware note: numbers are only meaningful for the machine they ran on
(dev machine: Apple M4 MacBook Air, 10-core GPU, 24 GB).
"""

from __future__ import annotations

import argparse
import gc
import statistics

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from veloxquant_mlx import KVCacheConfig
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
FILLER = (
    "The quick brown fox jumps over the lazy dog while the river runs past the old mill "
    "and the farmers count the harvest before the first frost arrives. "
)


def make_prompt_tokens(tokenizer, n_tokens: int) -> list[int]:
    ids = tokenizer.encode(FILLER * (n_tokens // 20 + 8))
    return ids[:n_tokens]


def setup(model, variant: str) -> None:
    n_layers = len(model.layers)
    if variant == "fp16":
        model.make_cache = lambda: [KVCache() for _ in range(n_layers)]
    elif variant == "qkv4":
        model.make_cache = lambda: [
            QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)
        ]
    else:
        # variant like "rvq2-off" / "rvq2-on"
        bits = int(variant[3])
        use_metal = variant.endswith("-on")
        cfg = KVCacheConfig(
            method="turboquant_rvq",
            head_dim=128,
            bit_width_inlier=bits,
            seed=42,
            use_metal_kernels=use_metal,
        )
        patch_model_kv_cache(model, cfg)


def run_once(model, tokenizer, prompt_ids: list[int], max_tokens: int):
    mx.reset_peak_memory()
    text, last = [], None
    for resp in stream_generate(
        model, tokenizer, prompt=mx.array(prompt_ids), max_tokens=max_tokens
    ):
        text.append(resp.text)
        last = resp
    return "".join(text), last.prompt_tps, last.generation_tps, mx.get_peak_memory() / 1e6


def measure(variant: str, prompt_len: int, max_tokens: int, repeats: int):
    mx.clear_cache()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    setup(model, variant)
    ids = make_prompt_tokens(tokenizer, prompt_len)
    run_once(model, tokenizer, ids[:256], 8)  # warm up: kernel compile, allocator
    runs = [run_once(model, tokenizer, ids, max_tokens) for _ in range(repeats)]
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return {
        "text": runs[0][0],
        "prefill_tps": statistics.median(r[1] for r in runs),
        "decode_tps": statistics.median(r[2] for r in runs),
        "peak_mb": statistics.median(r[3] for r in runs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192])
    ap.add_argument("--bits", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    variants = ["fp16", "qkv4"]
    for b in args.bits:
        variants += [f"rvq{b}-off", f"rvq{b}-on"]

    print(f"model={MODEL_ID} device={mx.default_device()} max_tokens={args.max_tokens}")
    print(
        f"{'prompt':>6} {'variant':>9} {'prefill t/s':>12} {'decode t/s':>11} {'peak MB':>9}  text==off"
    )
    for plen in args.prompt_lens:
        off_text = {}
        for v in variants:
            r = measure(v, plen, args.max_tokens, args.repeats)
            same = ""
            if v.endswith("-off"):
                off_text[v[:4]] = r["text"]
            elif v.endswith("-on"):
                same = "identical" if r["text"] == off_text[v[:4]] else "DIFFERS"
            print(
                f"{plen:>6} {v:>9} {r['prefill_tps']:>12.1f} {r['decode_tps']:>11.2f} "
                f"{r['peak_mb']:>9.1f}  {same}",
                flush=True,
            )


if __name__ == "__main__":
    main()
