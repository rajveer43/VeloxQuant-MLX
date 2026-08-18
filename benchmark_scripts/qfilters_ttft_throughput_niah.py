"""TTFT, throughput and needle-in-a-haystack for Q-Filters vs SnapKV / StreamingLLM.

Fills part of the "Not measured" gap listed in the Q-Filters docs: latency
(paper Figure 10) and retrieval quality, with the two eviction baselines the
repo actually implements.

WHAT THIS IS NOT
----------------
* **Not RULER.** The retrieval arm here is the NIAH family only (single-key,
  multi-key, multi-value), generated synthetically in-process. RULER's other
  ten task categories (variable tracking, common/frequent-word extraction, QA)
  are not covered.
* **Not Expected Attention.** That method is not implemented in this repo, so
  it cannot be a comparison arm.
* **Not a speedup claim over fp16.** These caches run a per-(B, H) Python loop
  in ``update_and_fetch``; wall-clock decode is therefore *slower* than the
  fp16 baseline despite storing far less. The throughput column measures this
  implementation, not the algorithm's potential. Treat cross-method
  comparisons as meaningful and absolute tok/s as an artifact of the loop.

A NOTE ON PREFILL CHUNK SIZE
----------------------------
Q-Filters' output quality on this harness depends strongly on how the prompt
is fed. ``qfilters_update`` absorbs a whole block then evicts to budget **in
one shot**, so a single 1073-token prefill makes one 1073 -> 256 decision
against a filter frozen on that same block. Measured on Llama-3.2-1B at
budget 256, identical prompt and identical 256 tokens retained:

    one-shot (1073 tok)   " the  the  the  the ..."      (degenerate)
    256-token chunks      partially coherent
    <=64-token chunks     "...is not explicitly stated"  (coherent)

This affects the CALIBRATED path too, not just the key-SVD fallback — both
produce the same degenerate output one-shot. The cache docstring calls the
fallback "path-DEPENDENT"; this is that path-dependence, and its practical
cost is larger than that phrasing suggests. This harness prefills in one
shot (what ``model(prompt, cache=...)`` does by default), so the Q-Filters
rows below reflect the worst case for this method.

BASELINE CORRECTNESS
--------------------
SnapKV and StreamingLLM carried the same RoPE ``offset`` defect that #171
fixed for Q-Filters: ``offset`` reported the retained ROW COUNT rather than
the true absolute token position, so once eviction started every subsequent
token was rotated at the wrong position (StreamingLLM's froze entirely once
its window saturated; SnapKV's stayed short by exactly the number evicted).
Both are fixed in this branch. Without that fix every number below would be
dominated by position drift in the baselines rather than by eviction quality,
and Q-Filters would appear to win for entirely the wrong reason.

Usage:
    python benchmark_scripts/qfilters_ttft_throughput_niah.py [MODEL ...]
"""

from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import KVCache

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache
from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache
from veloxquant_mlx.cache.streaming_llm_cache import StreamingLLMKVCache
from veloxquant_mlx.quantizers.qfilters_calibration import (
    average_gqa_filters,
    collect_query_activations,
    compute_qfilters,
)

# NIAH is swept across budgets. At 256 against a 1-2k context every eviction
# method scores 0%, which measures only that the needle was lost — not how
# gracefully each degrades. Larger budgets separate them.
NIAH_BUDGETS = [256, 512, 1024]

DEFAULT_MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
]

# Calibration corpus for the query-SVD filters (kept distinct from eval text).
CALIB = [
    "The Roman Republic was established after the overthrow of the Roman Kingdom, "
    "traditionally dated to 509 BC. Power was held by annually elected magistrates "
    "and the Senate, an assembly drawn largely from the patrician class. ",
    "Photosynthesis is the process by which green plants, algae and some bacteria "
    "convert light energy into chemical energy stored in carbohydrate molecules. "
    "In plants the reaction takes place in chloroplasts. ",
    "Modern cryptography relies on problems believed to be computationally hard, "
    "such as factoring large integers or computing discrete logarithms in finite "
    "groups. Public-key systems let two parties agree on a shared secret. ",
    "Plate tectonics describes the movement of large sections of the Earth's "
    "lithosphere over the underlying asthenosphere. Where plates converge one may "
    "subduct beneath another, producing deep ocean trenches and volcanic arcs. ",
]

# Filler prose for the haystack. Deliberately bland and repetitive in topic but
# not literally looped, so the needle is the only distinctive span.
HAYSTACK_SENTENCES = [
    "The committee reviewed the quarterly figures and found them consistent with prior guidance.",
    "Rainfall across the northern districts remained close to the seasonal average this year.",
    "The library extended its opening hours during the examination period as usual.",
    "Maintenance work on the eastern bridge is scheduled to continue through the autumn.",
    "Attendance at the regional conference was slightly higher than the organisers expected.",
    "The survey team completed its mapping of the coastal path ahead of the deadline.",
    "Local suppliers reported steady demand throughout the second half of the period.",
    "The archive catalogued several hundred additional documents over the winter months.",
    "Bus timetables were adjusted to reflect the revised school opening times.",
    "The laboratory replaced two ageing spectrometers during the scheduled shutdown.",
]


# ── cache construction ───────────────────────────────────────────────────────
def make_caches(method: str, n_layers: int, head_dim: int, budget: int, filters=None):
    """One cache instance per layer for the named method at a matched budget."""
    if method == "fp16":
        return [KVCache() for _ in range(n_layers)]

    if method == "qfilters":
        cfg = KVCacheConfig(
            method="qfilters",
            head_dim=head_dim,
            qfilters_budget=budget,
            qfilters_n_sink=4,
            qfilters_recent=budget // 4,
            qfilters_calib_tokens=min(128, budget // 2),
        )
        return [
            QFiltersKVCache(cfg, filters=None if filters is None else filters[i])
            for i in range(n_layers)
        ]

    if method == "snapkv":
        cfg = KVCacheConfig(
            method="snapkv",
            head_dim=head_dim,
            snap_budget=budget,
            snap_obs_window=min(32, budget // 2),
            snap_n_sink=4,
        )
        return [SnapKVKVCache(cfg) for _ in range(n_layers)]

    if method == "streaming_llm":
        # Matched total footprint: sinks + window == budget.
        cfg = KVCacheConfig(
            method="streaming_llm",
            head_dim=head_dim,
            stream_n_sink=4,
            stream_window_size=budget - 4,
        )
        return [StreamingLLMKVCache(cfg) for _ in range(n_layers)]

    raise ValueError(f"unknown method {method!r}")


# ── latency ─────────────────────────────────────────────────────────────────
def measure_ttft_throughput(model, ids, caches, n_gen: int = 64) -> tuple[float, float]:
    """Return (TTFT seconds, decode tokens/sec).

    TTFT is the prefill forward pass over the whole prompt plus the first
    sampled token. Throughput is measured over the following ``n_gen`` decode
    steps, excluding prefill. ``mx.eval`` forces completion before each timer
    stop — MLX is lazy, so without it the timers would measure graph
    construction rather than compute.
    """
    prompt = mx.array([ids])

    t0 = time.perf_counter()
    logits = model(prompt, cache=caches)
    tok = mx.argmax(logits[0, -1])
    mx.eval(tok)
    ttft = time.perf_counter() - t0

    cur = tok.reshape(1, 1)
    t1 = time.perf_counter()
    for _ in range(n_gen):
        logits = model(cur, cache=caches)
        cur = mx.argmax(logits[0, -1]).reshape(1, 1)
        mx.eval(cur)
    decode_s = time.perf_counter() - t1

    return ttft, n_gen / decode_s if decode_s > 0 else float("nan")


# ── needle in a haystack ────────────────────────────────────────────────────
def build_niah(tok, ctx_tokens: int, depth: float, variant: str, seed: int):
    """Construct one NIAH prompt.

    Args:
        ctx_tokens: approximate haystack length in tokens.
        depth: fractional insertion point of the needle (0.0 = start, 1.0 = end).
        variant: ``single`` (one key, one value), ``multikey`` (three distinct
            keys, only one queried — distractors share the surface form), or
            ``multivalue`` (one key carrying three values, all required).
        seed: RNG seed for filler ordering and secret values.

    Returns:
        ``(prompt_text, expected_strings)``.
    """
    rng = random.Random(seed)

    if variant == "single":
        secret = f"{rng.randint(1000, 9999)}"
        needles = [f"The access code for the Aurora project is {secret}."]
        question = "What is the access code for the Aurora project?"
        expected = [secret]
    elif variant == "multikey":
        vals = [f"{rng.randint(1000, 9999)}" for _ in range(3)]
        needles = [
            f"The access code for the Aurora project is {vals[0]}.",
            f"The access code for the Borealis project is {vals[1]}.",
            f"The access code for the Cascade project is {vals[2]}.",
        ]
        question = "What is the access code for the Borealis project?"
        expected = [vals[1]]
    elif variant == "multivalue":
        vals = [f"{rng.randint(1000, 9999)}" for _ in range(3)]
        needles = [
            f"The Aurora project has three access codes: {vals[0]}, {vals[1]} and {vals[2]}."
        ]
        question = "List all three access codes for the Aurora project."
        expected = vals
    else:
        raise ValueError(variant)

    # Grow filler until the token budget is met.
    filler: list[str] = []
    while len(tok.encode(" ".join(filler))) < ctx_tokens:
        filler.append(rng.choice(HAYSTACK_SENTENCES))

    # Insert needles at the requested depth (spread slightly for multikey).
    for i, needle in enumerate(needles):
        pos = int(len(filler) * min(depth + 0.05 * i, 0.99))
        filler.insert(pos, needle)

    body = " ".join(filler)
    prompt = (
        "Read the following text carefully and then answer the question.\n\n"
        f"{body}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt, expected


def run_niah_case(model, tok, caches, prompt: str, expected: list[str], max_new: int = 24) -> bool:
    """Greedy-decode an answer and report whether every expected string appears."""
    ids = tok.encode(prompt)
    logits = model(mx.array([ids]), cache=caches)
    cur = mx.argmax(logits[0, -1]).reshape(1, 1)
    out = [int(cur.item())]
    for _ in range(max_new - 1):
        logits = model(cur, cache=caches)
        cur = mx.argmax(logits[0, -1]).reshape(1, 1)
        out.append(int(cur.item()))
    text = tok.decode(out)
    return all(e in text for e in expected)


# ── driver ──────────────────────────────────────────────────────────────────
def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    methods = ["fp16", "qfilters", "snapkv", "streaming_llm"]
    budget = 256
    results: dict = {}

    for model_id in models:
        print(f"\n{'=' * 78}\n{model_id}\n{'=' * 78}")
        model, tok = mlx_lm.load(model_id)
        inner = getattr(model, "model", model)
        n_layers = len(inner.layers)
        kv_heads = model.args.num_key_value_heads
        # Qwen2's ModelArgs omits head_dim; Llama's carries it. Derive when absent.
        head_dim = (
            getattr(model.args, "head_dim", None)
            or model.args.hidden_size // model.args.num_attention_heads
        )

        print("calibrating query-SVD filters ...", flush=True)
        acts = collect_query_activations(
            model, tok, CALIB, max_length=512, max_samples_per_head=2000
        )
        filters = [average_gqa_filters(compute_qfilters(a), kv_heads) for a in acts]
        del acts
        gc.collect()

        mrec: dict = {"latency": {}, "niah": {}}

        # ---- TTFT / throughput ----
        print(f"\n--- TTFT & throughput (prompt 2048 tok, 64 decode, budget {budget}) ---")
        print(f"{'method':<16}{'TTFT (s)':>12}{'decode tok/s':>16}")
        prompt_ids = tok.encode(" ".join(HAYSTACK_SENTENCES * 40))[:2048]
        for method in methods:
            caches = make_caches(method, n_layers, head_dim, budget, filters)
            ttft, tps = measure_ttft_throughput(model, prompt_ids, caches)
            mrec["latency"][method] = {"ttft_s": ttft, "decode_tok_s": tps}
            print(f"{method:<16}{ttft:>12.3f}{tps:>16.1f}")
            del caches
            gc.collect()

        # ---- NIAH ----
        # Swept across budgets: at a single aggressive budget every eviction
        # method floors at 0%, which cannot distinguish "somewhat worse" from
        # "catastrophically worse". The sweep shows where each one breaks.
        depths = [0.1, 0.5, 0.9]
        ctx_lens = [1024, 2048]
        variants = ["single", "multikey", "multivalue"]
        for nb in NIAH_BUDGETS:
            print(f"\n--- NIAH (budget {nb}) ---")
            print(f"{'method':<16}{'variant':<14}{'accuracy':>10}   (per-depth)")
            for method in methods:
                # fp16 ignores the budget entirely — measure it once, at the
                # first budget, and carry that row as the ceiling.
                if method == "fp16" and nb != NIAH_BUDGETS[0]:
                    continue
                mrec["niah"].setdefault(method, {})
                for variant in variants:
                    hits, total, per_depth = 0, 0, []
                    for ctx in ctx_lens:
                        for depth in depths:
                            prompt, expected = build_niah(
                                tok, ctx, depth, variant, seed=ctx + int(depth * 100)
                            )
                            caches = make_caches(method, n_layers, head_dim, nb, filters)
                            ok = run_niah_case(model, tok, caches, prompt, expected)
                            hits += int(ok)
                            total += 1
                            per_depth.append(f"{depth:.1f}:{'Y' if ok else '.'}")
                            del caches
                            gc.collect()
                    acc = hits / total
                    mrec["niah"].setdefault(method, {}).setdefault(variant, {})[nb] = acc
                    print(f"{method:<16}{variant:<14}{acc:>9.0%}   {' '.join(per_depth)}")

        results[model_id] = mrec
        del model, tok, filters
        gc.collect()

    out = _repo_root / "benchmark_scripts" / "results_qfilters_ttft_niah.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
