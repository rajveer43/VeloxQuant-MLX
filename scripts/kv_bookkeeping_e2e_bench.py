"""End-to-end real-model benchmark: baseline fp16 cache vs TOVA vs H2O vs KIVI
on Llama-3.2-1B-Instruct (local mlx-community 4bit weights, already cached —
no download performed by this script). Measures prefill/decode tok/s and a
logit-agreement check between baseline and bounded caches at a budget larger
than the prompt (so no eviction should occur and logits should match closely
except for the bounded caches' own approximations, e.g. quantization).

Run: .venv/bin/python scripts/kv_bookkeeping_e2e_bench.py --output /tmp/kv_e2e.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

MODEL_PATH = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def load():
    from mlx_lm import load

    model, tokenizer = load(MODEL_PATH)
    return model, tokenizer


def make_caches(model, method, **kwargs):
    from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig

    cfg = KVCacheConfig(method=method, **kwargs)
    return KVCacheBuilder.for_model(model, cfg)


def run_generate(model, tokenizer, prompt_ids, caches, n_decode):
    """Manual prefill+decode loop (not mlx_lm.generate) so we can swap in
    VeloxQuant caches and measure prefill vs decode timing separately."""
    prompt = mx.array(prompt_ids)[None]
    t0 = time.perf_counter()
    logits = model(prompt, cache=caches)
    mx.eval(logits)
    prefill_s = time.perf_counter() - t0
    next_tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_tok)

    tokens = [int(next_tok.item())]
    t1 = time.perf_counter()
    for _ in range(n_decode - 1):
        logits = model(next_tok[:, None], cache=caches)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tok)
        tokens.append(int(next_tok.item()))
    decode_s = time.perf_counter() - t1

    return dict(
        tokens=tokens,
        prefill_s=prefill_s,
        prefill_tok_s=prompt.shape[1] / prefill_s if prefill_s > 0 else None,
        decode_s=decode_s,
        decode_tok_s=(n_decode - 1) / decode_s if decode_s > 0 and n_decode > 1 else None,
        ms_per_token=(decode_s / (n_decode - 1) * 1000) if n_decode > 1 else None,
        first_logits=logits,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--n-decode", type=int, default=64)
    parser.add_argument("--budget", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading {MODEL_PATH} (local cache)...")
    model, tokenizer = load()

    rng = np.random.default_rng(0)
    vocab_size = model.args.vocab_size if hasattr(model.args, "vocab_size") else 128256
    prompt_ids = rng.integers(0, min(vocab_size, 100000), size=args.prompt_len).tolist()

    from mlx_lm.models.cache import KVCache as PlainCache

    print("Baseline (plain fp16 KVCache)...")
    plain_caches = [PlainCache() for _ in model.layers]
    base = run_generate(model, tokenizer, prompt_ids, plain_caches, args.n_decode)
    base_logits = base.pop("first_logits")

    results = dict(
        prompt_len=args.prompt_len,
        n_decode=args.n_decode,
        budget=args.budget,
        baseline=base,
    )

    for method, kwargs in [
        ("tova", dict(tova_budget=args.budget, tova_n_sink=4)),
        ("h2o", dict(h2o_budget=args.budget, h2o_n_sink=4, h2o_grace=16, h2o_decay=0.98)),
        ("kivi", dict(bit_width_inlier=2, kivi_group_size=32, residual_length=32)),
    ]:
        print(f"{method}...")
        caches = make_caches(model, method, head_dim=64, **kwargs)
        r = run_generate(model, tokenizer, prompt_ids, caches, args.n_decode)
        logits = r.pop("first_logits")

        # Logit agreement at the LAST prefill position (before any decode-time
        # divergence from different sampled tokens)
        b32 = base_logits[:, -1, :].astype(mx.float32)
        c32 = logits[:, -1, :].astype(mx.float32)
        max_err = float(mx.max(mx.abs(b32 - c32)))
        mean_err = float(mx.mean(mx.abs(b32 - c32)))
        cos = float(
            mx.sum(b32 * c32) / (mx.sqrt(mx.sum(b32 * b32)) * mx.sqrt(mx.sum(c32 * c32)) + 1e-8)
        )
        b_top = int(mx.argmax(b32, axis=-1).item())
        c_top = int(mx.argmax(c32, axis=-1).item())

        r["logit_max_abs_err"] = max_err
        r["logit_mean_abs_err"] = mean_err
        r["logit_cosine_sim"] = cos
        r["top_token_agreement"] = b_top == c_top
        r["token_agreement_fraction"] = sum(
            a == b for a, b in zip(r["tokens"], base["tokens"])
        ) / len(r["tokens"])
        results[method] = r
        print(
            f"  prefill {r['prefill_tok_s']:.1f} tok/s, decode {r['decode_tok_s']:.1f} tok/s, "
            f"cos_sim={cos:.5f}, token_agree_frac={r['token_agreement_fraction']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print("Baseline:", base["prefill_tok_s"], "prefill tok/s,", base["decode_tok_s"], "decode tok/s")


if __name__ == "__main__":
    main()
