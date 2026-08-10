"""Real-model validation for Keyformer-adapted: prefill, decode, correctness,
and Metal-kernel speedup.

Unlike benchmark_keyformer.py (offline-synthetic survival-rate/perturbation
numbers only), this script runs actual mlx_lm models end-to-end and answers
the questions the docs page explicitly flags as untested:

  1. Does a long prefill (thousands of prompt tokens) survive without
     crashing, at various budgets, matching the same scalability fix H2O
     needed?
  2. Does prefill (one big batched call) vs. decode (token-by-token) produce
     coherent, budget-respecting output on a real model, for both a constant
     tau and an annealed tau schedule?
  3. Does generation stay coherent (no repetition-loop freeze, no mid-word
     corruption) across a real multi-hundred-token generation, at a tight
     budget where the eviction mechanism is actually exercised?
  4. What's the actual wall-clock speedup from the fused Metal kernel vs.
     pure-MLX, end-to-end through mlx_lm.generate (not just the isolated
     kernel microbenchmark in tests/metal/test_keyformer_evict.py)?
  5. Does the tau=0 config produce output indistinguishable from running
     H2O directly, on a real model (not just synthetic state)?

Usage
-----
    python benchmark_scripts/benchmark_keyformer_real_model.py
    python benchmark_scripts/benchmark_keyformer_real_model.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
    python benchmark_scripts/benchmark_keyformer_real_model.py --skip-large-prefill

Prints tables; saves a JSON summary to figures/keyformer/real_model_results.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.metal import metal_available

_repo_root = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
LONG_PROMPT = (
    "Write a detailed, factual explanation of how photosynthesis converts "
    "light energy into chemical energy in plants. Cover the light-dependent "
    "reactions, the Calvin cycle, the role of chlorophyll, and why this "
    "process is essential for life on Earth. Be thorough and precise."
)


def _inject_cache(model, config: KVCacheConfig):
    """Patch model.make_cache to build our cache and return the built list."""
    injected = []
    original = model.make_cache

    def _patch(*_a, **_k):
        caches = KVCacheBuilder.for_model(model, config)
        injected.clear()
        injected.extend(caches)
        return caches

    model.make_cache = _patch
    return original, injected


def _restore(model, original):
    model.make_cache = original


def _timed_generate(model, tokenizer, prompt_txt, max_tokens):
    mx.clear_cache()
    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt_txt, max_tokens=max_tokens, verbose=False
    )
    elapsed = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(response))
    return response, elapsed, n_tok


def _chat_prompt(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# 1. Long-prefill scalability
# ---------------------------------------------------------------------------


def bench_long_prefill(model, tokenizer, results: dict):
    print(f"\n{'=' * 72}\n1. LONG-PREFILL SCALABILITY (crash regression check)\n{'=' * 72}")
    long_context = (LONG_PROMPT + " ") * 60  # ~thousands of tokens once tokenized
    prompt_txt = _chat_prompt(tokenizer, long_context)
    n_prompt_tokens = len(tokenizer.encode(prompt_txt))
    print(f"  prompt tokens: {n_prompt_tokens}")

    rows = []
    for budget, tau_mode in [(128, "constant"), (128, "annealed"), (256, "constant")]:
        kw = dict(
            method="keyformer",
            keyformer_budget=budget,
            keyformer_n_sink=4,
            keyformer_recent=16,
        )
        if tau_mode == "constant":
            kw["keyformer_tau"] = 1.0
        else:
            kw["keyformer_tau_init"] = 1.0
            kw["keyformer_tau_end"] = 2.0
            kw["keyformer_anneal_steps"] = 128
        config = KVCacheConfig(**kw)
        original, injected = _inject_cache(model, config)
        label = f"budget={budget} tau={tau_mode}"
        try:
            response, elapsed, n_tok = _timed_generate(model, tokenizer, prompt_txt, max_tokens=40)
            kept = injected[0].tokens_kept if injected else -1
            status = "OK"
            print(f"  [{label}] OK — kept={kept} tokens, {n_tok} generated in {elapsed:.1f}s")
            print(f"    -> {response[:150]!r}")
        except Exception as e:  # noqa: BLE001 — we want to record the failure, not crash the sweep
            status = f"CRASH: {type(e).__name__}: {e}"
            elapsed, n_tok, kept = -1.0, 0, -1
            print(f"  [{label}] {status}")
        finally:
            _restore(model, original)
        rows.append(
            dict(
                label=label,
                budget=budget,
                tau_mode=tau_mode,
                status=status,
                elapsed_s=elapsed,
                n_generated=n_tok,
                kept=kept,
                n_prompt_tokens=n_prompt_tokens,
            )
        )
    results["long_prefill"] = rows


# ---------------------------------------------------------------------------
# 2. Prefill-path vs decode-path: both produce coherent, budget-respecting output
# ---------------------------------------------------------------------------


def bench_prefill_vs_decode(model, tokenizer, results: dict):
    print(
        f"\n{'=' * 72}\n2. PREFILL vs DECODE PATH (budget respected, output coherent)\n{'=' * 72}"
    )
    prompt_txt = _chat_prompt(tokenizer, LONG_PROMPT)
    n_prompt_tokens = len(tokenizer.encode(prompt_txt))
    print(f"  prompt tokens: {n_prompt_tokens}")

    rows = []
    for tau_mode in ["constant", "annealed"]:
        kw = dict(method="keyformer", keyformer_budget=128, keyformer_n_sink=4, keyformer_recent=16)
        if tau_mode == "constant":
            kw["keyformer_tau"] = 1.0
        else:
            kw["keyformer_tau_init"] = 1.0
            kw["keyformer_tau_end"] = 2.0
            kw["keyformer_anneal_steps"] = 64
        config = KVCacheConfig(**kw)

        # "Prefill path" here just means: this IS how mlx_lm.generate works —
        # the whole prompt goes through update_and_fetch in one (or few,
        # chunked) calls before decode starts. We record budget compliance
        # and output coherence (no crash, no empty/garbage output).
        original, injected = _inject_cache(model, config)
        try:
            response, elapsed, n_tok = _timed_generate(model, tokenizer, prompt_txt, max_tokens=120)
            kept = injected[0].tokens_kept if injected else -1
            budget_ok = kept <= 128
            coherent = n_tok > 5 and len(response.strip()) > 10
            print(
                f"  [{tau_mode}] kept={kept} budget_ok={budget_ok} tokens={n_tok} "
                f"coherent={coherent} time={elapsed:.1f}s"
            )
            print(f"    -> {response[:200]!r}")
        finally:
            _restore(model, original)
        rows.append(
            dict(
                tau_mode=tau_mode,
                kept=kept,
                budget_ok=budget_ok,
                n_generated=n_tok,
                coherent=coherent,
                elapsed_s=elapsed,
                response_preview=response[:300],
            )
        )
    results["prefill_vs_decode"] = rows


# ---------------------------------------------------------------------------
# 3. Generation coherence at a tight budget (freeze/gibberish regression check)
# ---------------------------------------------------------------------------


def bench_coherence_tight_budget(model, tokenizer, results: dict):
    print(
        f"\n{'=' * 72}\n3. COHERENCE ACROSS BUDGETS (freeze / repetition / mid-word corruption)\n{'=' * 72}"
    )
    print("  NOTE: budgets <64 exercise the documented open issue (single eviction")
    print("  can still corrupt output at tight budgets, independent of grace/decay/")
    print("  tau) — see docs-site/docs/algorithms/h2o.md. These rows are expected")
    print("  to degrade; they are reported for visibility, not as pass/fail.")
    prompt_txt = _chat_prompt(tokenizer, "Tell me a short story about a lighthouse keeper.")

    rows = []
    for budget, n_sink, recent, tau_kw, label in [
        (24, 2, 8, dict(keyformer_tau=1.0), "TIGHT (known-degradation zone), constant tau"),
        (
            24,
            2,
            8,
            dict(keyformer_tau_init=1.0, keyformer_tau_end=2.0, keyformer_anneal_steps=64),
            "TIGHT (known-degradation zone), annealed tau",
        ),
        (128, 4, 16, dict(keyformer_tau=1.0), "realistic budget, constant tau"),
        (
            128,
            4,
            16,
            dict(keyformer_tau_init=1.0, keyformer_tau_end=2.0, keyformer_anneal_steps=64),
            "realistic budget, annealed tau",
        ),
        (128, 4, 16, dict(keyformer_tau=0.0), "realistic budget, tau=0 (H2O-adapted collapse)"),
    ]:
        config = KVCacheConfig(
            method="keyformer",
            keyformer_budget=budget,
            keyformer_n_sink=n_sink,
            keyformer_recent=recent,
            **tau_kw,
        )
        original, injected = _inject_cache(model, config)
        try:
            response, elapsed, n_tok = _timed_generate(model, tokenizer, prompt_txt, max_tokens=200)
            kept = injected[0].tokens_kept if injected else -1
            # crude repetition-loop detector: any 4-gram repeated >= 6 times
            words = response.split()
            grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
            max_repeat = max((grams.count(g) for g in set(grams)), default=0)
            frozen_loop = max_repeat >= 6
            print(
                f"  [{label}] kept={kept} tokens={n_tok} max_4gram_repeat={max_repeat} "
                f"frozen_loop={frozen_loop} time={elapsed:.1f}s"
            )
            print(f"    -> {response[:250]!r}")
        finally:
            _restore(model, original)
        rows.append(
            dict(
                label=label,
                budget=budget,
                kept=kept,
                n_generated=n_tok,
                max_4gram_repeat=max_repeat,
                frozen_loop=frozen_loop,
                response_preview=response[:400],
            )
        )
    results["coherence_tight_budget"] = rows


# ---------------------------------------------------------------------------
# 4. tau=0 real-model parity: Keyformer(tau=0) vs H2O directly
# ---------------------------------------------------------------------------


def bench_tau_zero_matches_h2o(model, tokenizer, results: dict):
    print(f"\n{'=' * 72}\n4. tau=0 REAL-MODEL PARITY vs H2O-adapted directly\n{'=' * 72}")
    prompt_txt = _chat_prompt(tokenizer, "Explain what a hash table is in two sentences.")

    kf_config = KVCacheConfig(
        method="keyformer",
        keyformer_budget=128,
        keyformer_n_sink=4,
        keyformer_tau=0.0,
    )
    h2o_config = KVCacheConfig(
        method="h2o",
        h2o_budget=128,
        h2o_n_sink=4,
        h2o_grace=0,
        h2o_decay=1.0,
    )

    original, kf_injected = _inject_cache(model, kf_config)
    kf_response, kf_elapsed, kf_tok = _timed_generate(model, tokenizer, prompt_txt, max_tokens=60)
    kf_kept = kf_injected[0].tokens_kept
    _restore(model, original)

    original, h2o_injected = _inject_cache(model, h2o_config)
    h2o_response, h2o_elapsed, h2o_tok = _timed_generate(
        model, tokenizer, prompt_txt, max_tokens=60
    )
    h2o_kept = h2o_injected[0].tokens_kept
    _restore(model, original)

    exact_match = kf_response == h2o_response
    print(f"  Keyformer(tau=0): kept={kf_kept} tokens={kf_tok} -> {kf_response[:200]!r}")
    print(f"  H2O-adapted:      kept={h2o_kept} tokens={h2o_tok} -> {h2o_response[:200]!r}")
    print(f"  EXACT TEXT MATCH: {exact_match}")
    results["tau_zero_vs_h2o"] = dict(
        keyformer_response=kf_response,
        h2o_response=h2o_response,
        exact_match=exact_match,
        keyformer_kept=kf_kept,
        h2o_kept=h2o_kept,
    )


# ---------------------------------------------------------------------------
# 5. Metal kernel speedup end-to-end through mlx_lm.generate
# ---------------------------------------------------------------------------


def bench_metal_speedup(model, tokenizer, results: dict):
    print(f"\n{'=' * 72}\n5. METAL KERNEL SPEEDUP (end-to-end through mlx_lm.generate)\n{'=' * 72}")
    if not metal_available():
        print("  Metal not available on this build — skipping.")
        results["metal_speedup"] = {"skipped": True, "reason": "metal_available() is False"}
        return

    prompt_txt = _chat_prompt(tokenizer, LONG_PROMPT)
    n_prompt_tokens = len(tokenizer.encode(prompt_txt))
    rows = []
    # max_tokens is chosen so budget is guaranteed to be exceeded during
    # decode (prompt + generated > budget with margin) — otherwise eviction
    # (and therefore the kernel this benchmarks) never actually triggers and
    # "speedup" measures two identical no-eviction code paths.
    for budget, max_tokens in [(128, 220), (192, 260)]:
        assert n_prompt_tokens + max_tokens > budget * 1.5, (
            f"budget={budget} not guaranteed to be exceeded: "
            f"{n_prompt_tokens} prompt + {max_tokens} max_tokens"
        )
        config = KVCacheConfig(
            method="keyformer",
            keyformer_budget=budget,
            keyformer_n_sink=4,
            keyformer_recent=8,
            keyformer_tau=1.0,
        )

        # Metal path (default when available)
        original, kf_injected_metal = _inject_cache(model, config)
        _, t_metal, n_metal = _timed_generate(model, tokenizer, prompt_txt, max_tokens=max_tokens)
        kept_metal = kf_injected_metal[0].tokens_kept if kf_injected_metal else -1
        _restore(model, original)

        # Force pure-MLX path by monkeypatching metal_available to False for
        # the quantizer module's own check.
        import veloxquant_mlx.quantizers.keyformer as kf_mod

        real_check = kf_mod._metal_evict_available
        kf_mod._metal_evict_available = lambda: False
        try:
            original, kf_injected_mlx = _inject_cache(model, config)
            _, t_mlx, n_mlx = _timed_generate(model, tokenizer, prompt_txt, max_tokens=max_tokens)
            kept_mlx = kf_injected_mlx[0].tokens_kept if kf_injected_mlx else -1
            _restore(model, original)
        finally:
            kf_mod._metal_evict_available = real_check

        evictions_happened = kept_metal >= budget and kept_mlx >= budget
        speedup = t_mlx / t_metal if t_metal > 0 else float("nan")
        print(
            f"  budget={budget}: kept(metal)={kept_metal} kept(mlx)={kept_mlx} "
            f"evictions_happened={evictions_happened}"
        )
        print(
            f"    MLX={t_mlx:.2f}s ({n_mlx} tok, {n_mlx / t_mlx:.1f} tok/s)  "
            f"Metal={t_metal:.2f}s ({n_metal} tok, {n_metal / t_metal:.1f} tok/s)  "
            f"speedup={speedup:.2f}x"
        )
        rows.append(
            dict(
                budget=budget,
                t_mlx_s=t_mlx,
                t_metal_s=t_metal,
                speedup=speedup,
                mlx_tok_per_s=n_mlx / t_mlx,
                metal_tok_per_s=n_metal / t_metal,
                kept_metal=kept_metal,
                kept_mlx=kept_mlx,
                evictions_happened=evictions_happened,
            )
        )
    results["metal_speedup"] = rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-large-prefill", action="store_true")
    parser.add_argument(
        "--out", default=str(_repo_root / "figures" / "keyformer" / "real_model_results.json")
    )
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = mlx_lm.load(args.model)
    if not hasattr(model, "make_cache"):

        def _default_make_cache():
            return [None] * len(model.layers)

        model.make_cache = _default_make_cache

    results: dict = {"model": args.model, "metal_available": metal_available()}

    if not args.skip_large_prefill:
        bench_long_prefill(model, tokenizer, results)
    bench_prefill_vs_decode(model, tokenizer, results)
    bench_coherence_tight_budget(model, tokenizer, results)
    bench_tau_zero_matches_h2o(model, tokenizer, results)
    bench_metal_speedup(model, tokenizer, results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'=' * 72}\nSaved results to {out_path}\n{'=' * 72}")


if __name__ == "__main__":
    main()
