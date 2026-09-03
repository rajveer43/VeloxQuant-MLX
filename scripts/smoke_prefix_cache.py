"""Manual smoke test for PrefixCache (#310, Phase 2): proves a second call
sharing a long prompt prefix actually reuses cached KV instead of
re-prefilling it end to end -- the literal rebuttal to the Ollama MLX
prefix-caching gap (ollama/ollama#17829) cited in issue #310.

No pytest marker convention exists in this repo for "requires a real
downloaded model" (see test_serve_cli.py's own docstring) -- this is run
manually and its output captured in the PR description per CONTRIBUTING.md's
"no number claims without a reproducible script" rule.

Run from repo root:

    PYTHONPATH=. python scripts/smoke_prefix_cache.py
"""

from __future__ import annotations

import time

from mlx_lm import load

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.integration.prefix_cache import PrefixCache

MODEL_ID = "mlx-community/SmolLM2-135M-Instruct"

SHARED_PREFIX = (
    "You are a careful, concise assistant helping with software engineering "
    "tasks. Follow the user's instructions precisely, prefer minimal correct "
    "changes over large rewrites, and always explain trade-offs when more "
    "than one reasonable approach exists. " * 20
)


def _timed_generate(pc: PrefixCache, model, tokenizer, prompt: str, max_tokens: int = 40):
    start = time.perf_counter()
    text = pc.generate(model, tokenizer, prompt, max_tokens=max_tokens)
    elapsed = time.perf_counter() - start
    return text, elapsed


def main() -> None:
    model, tokenizer = load(MODEL_ID)
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
    pc = PrefixCache(config)

    prompt_a = SHARED_PREFIX + "Question: what is a KV cache?"
    prompt_b = SHARED_PREFIX + "Question: why does prefix reuse matter for agents?"

    print(f"Shared prefix length: {len(tokenizer.encode(SHARED_PREFIX))} tokens")

    _, t_cold = _timed_generate(pc, model, tokenizer, prompt_a)
    print(f"Call 1 (cold, full prefill): {t_cold:.3f}s")

    _, t_warm = _timed_generate(pc, model, tokenizer, prompt_b)
    print(f"Call 2 (shares prefix, should reuse KV): {t_warm:.3f}s")

    speedup = t_cold / t_warm if t_warm > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x")
    if t_warm >= t_cold:
        print(
            "WARNING: second call was not faster -- prefix reuse may not be "
            "taking effect. Investigate before citing this as a fix."
        )


if __name__ == "__main__":
    main()
