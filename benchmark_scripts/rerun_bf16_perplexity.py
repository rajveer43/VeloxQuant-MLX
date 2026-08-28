"""Perplexity-only rerun for bfloat16 models that hit a numpy buffer-protocol
bug in the main multi-model sweep (see benchmark_kivi_multi_model.py history:
`np.array(logits[0], dtype=np.float32)` fails via the PEP 3118 buffer
protocol on mlx bfloat16 arrays -- fixed there to
`np.array(logits[0].astype(mx.float32))`, but that fix doesn't apply
retroactively to the already-completed sweep).

Covers: SmolLM2-135M-Instruct, gemma-3-4b-it-4bit, Qwen3-4B-4bit, Qwen3-8B-4bit.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

from benchmark_kivi_multi_model import LONG_SAMPLE, compute_perplexity_stable, load_model

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache
from veloxquant_mlx.metal import _kivi_quant

RESULTS_PATH = Path(__file__).parents[1] / "kivi_bf16_perplexity_rerun.json"

MODELS = [
    "mlx-community/SmolLM2-135M-Instruct",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3-8B-4bit",
]


def run_model(model_id: str) -> dict:
    print(f"\n=== {model_id} ===")
    out: dict = {"model": model_id, "results": {}}

    # fp16 baseline -- only if make_cache exists on the unpatched model.
    model, tokenizer = load_model(model_id)
    if hasattr(model, "make_cache"):
        try:
            ppl = compute_perplexity_stable(model, tokenizer, LONG_SAMPLE)
            out["results"]["fp16 (baseline)"] = {"perplexity": ppl}
            print(f"  fp16 (baseline): ppl={ppl:.3f}")
        except Exception as e:
            out["results"]["fp16 (baseline)"] = {"error": str(e)}
            print(f"  fp16 (baseline): FAILED {e}")
    else:
        out["results"]["fp16 (baseline)"] = {"error": "no make_cache on unpatched model"}
        print("  fp16 (baseline): SKIPPED (no make_cache)")
    del model, tokenizer
    gc.collect()

    for label, bits in (("KIVI b=2", 2), ("KIVI b=4", 4)):
        _kivi_quant._cache.clear()
        model, tokenizer = load_model(model_id)
        config = KVCacheConfig(method="kivi", bit_width_inlier=bits, seed=42)
        try:
            patch_model_kv_cache(model, config)
            ppl = compute_perplexity_stable(model, tokenizer, LONG_SAMPLE)
            out["results"][label] = {"perplexity": ppl}
            print(f"  {label}: ppl={ppl:.3f}")
        except Exception as e:
            out["results"][label] = {"error": str(e)}
            print(f"  {label}: FAILED {e}")
        del model, tokenizer
        gc.collect()

    return out


def main() -> None:
    results = [run_model(m) for m in MODELS]
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
