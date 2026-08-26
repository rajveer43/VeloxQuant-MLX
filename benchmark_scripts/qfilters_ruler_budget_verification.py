"""Verify the RULER arms compress equally, so score gaps are selection not budget (#177).

On common-word extraction the RULER harness reports a 0%-to-93% spread across
eviction methods at the same nominal budget. Before treating that as a
selection-quality result it has to be ruled out that some arm simply keeps
more tokens, or reports RoPE positions wrongly -- the ``offset`` defect that
:issue:`171` fixed for Q-Filters and :issue:`174` for L2Norm would produce
exactly this kind of spread for entirely the wrong reason.

This measures the retained cache rows and the reported offset directly, after
a real prefill, for every arm on one identical prompt.

Result on Qwen2.5-7B, CWE prompt of 1272 tokens, budget 512:

    method          cache.keys.shape        offset   CWE mean
    snapkv          (1, 4, 512, 128)        1272        0%
    qfilters        (1, 4, 512, 128)        1272       33%
    knorm           (1, 4, 512, 128)        1272       38%
    streaming_llm   (1, 4, 512, 128)        1272       93%

Every arm retains exactly 512 of 1272 tokens and reports the true final
position. No arm is advantaged by keeping more, and none carries a position
defect, so the spread is entirely *which* tokens each method selected.

WHY SNAPKV SCORES ZERO HERE
---------------------------
SnapKV picks tokens by attention from a trailing ``snap_obs_window`` (32
tokens) used as proxy queries. In the CWE construction that window lands on
the question text rather than on the word list it asks about, so the proxy is
unrepresentative: SnapKV scores the question region as important and evicts
the list it needs to count. Decoded answers on the same prompt make the
failure mode legible:

    snapkv b512    ' 10000000000000000000000000000000000000000000000'
    snapkv b1024   ': "opportunity", "innovation", and "inspiration".'
    stream b1024   ' opal, juniper, and birch. Each of these words appears 9 times ...'

The b1024 answer is fluent, well-formed and entirely fabricated -- none of
those three words appear anywhere in the prompt. Having lost the list, the
model answers from priors. StreamingLLM on the identical prompt cites the
counts, showing it still held the data.

This is a statement about a proxy-query method meeting a task whose relevant
span is not where its proxy looks. It is not a general claim about SnapKV.

Usage:
    python benchmark_scripts/qfilters_ruler_budget_verification.py [MODEL]
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from benchmark_scripts.qfilters_ruler_beyond_niah import (  # noqa: E402
    BUILDERS,
    make_caches,
    run_case,
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
BUDGET = 512
CTX = 1024
METHODS = ["snapkv", "qfilters", "knorm", "streaming_llm"]


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model, tok = mlx_lm.load(model_id)
    inner = getattr(model, "model", model)
    n_layers = len(inner.layers)
    head_dim = (
        getattr(model.args, "head_dim", None)
        or model.args.hidden_size // model.args.num_attention_heads
    )

    prompt, expected, pool = BUILDERS["cwe"](tok, CTX, CTX * 100 + 10_000)
    ids = tok.encode(prompt)
    print(f"CWE prompt = {len(ids)} tokens, budget {BUDGET}, expected {expected}\n")
    print(f"{'method':<16}{'cache.keys.shape':<26}{'offset':>8}{'score':>8}")

    for method in METHODS:
        # Shape after a real prefill -- the retained rows, not a counter.
        # `tokens_kept` on these caches is a cumulative sum across (B, H) and
        # is NOT the current cache size; reading it as one is misleading.
        caches = make_caches(method, n_layers, head_dim, BUDGET, None)
        logits = model(mx.array([ids]), cache=caches)
        mx.eval(logits)
        shape = tuple(caches[0].keys.shape)
        offset = caches[0].offset
        del caches
        gc.collect()

        caches = make_caches(method, n_layers, head_dim, BUDGET, None)
        score, text = run_case(model, tok, caches, prompt, expected, pool)
        del caches
        gc.collect()

        print(f"{method:<16}{str(shape):<26}{offset:>8}{score:>7.0%}")
        print(f"    {text[:100]!r}")


if __name__ == "__main__":
    main()
