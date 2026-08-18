"""Does chunked prefill rescue Q-Filters on variable tracking? (#177)

The NIAH harness documents that Q-Filters' output quality depends strongly on
how the prompt is fed: ``qfilters_update`` absorbs a block then evicts to
budget in one shot, so a single large prefill makes one eviction decision
against a filter frozen on that same block. Measured there on Llama-3.2-1B,
identical prompt and budget, output went from degenerate at one shot to
coherent at <=64-token chunks.

The RULER harness prefills in one shot (what ``model(prompt, cache=...)``
does by default), so its Q-Filters rows are that method's worst case. Before
reporting "Q-Filters scores 0% on variable tracking" it is worth establishing
whether the collapse is inherent to the task or an artifact of that path.

Result on Qwen2.5-7B, budget 512, ctx 1024, 3 seeds (fp16 ceiling 69%):

    one-shot          0.00   [0.00, 0.00, 0.00]
    256-token chunks  0.00   [0.00, 0.00, 0.00]
    64-token chunks   0.11   [0.00, 0.00, 0.33]

Chunking does not rescue it. On NIAH the 256-token setting already produced
partial recovery; here it produces none, and the <=64 setting that restored
coherence there yields a single partial hit across three seeds. Against a 69%
ceiling, 0.11 is still collapse -- but it is not zero, and the run is 3 seeds
at one budget, so this establishes "chunking does not rescue it" rather than a
precise chunked score.

The likely reason is structural rather than about prefill: VT requires every
hop of a scattered chain to survive eviction, and a projection-based
importance score has no way to mark a bare ``VAR X502 = X684.`` as
load-bearing. Retaining most of a chain still scores zero, because a broken
chain yields no correct answer. NIAH differs -- retaining the single needle
suffices.

Usage:
    python benchmark_scripts/qfilters_ruler_prefill_chunking.py [MODEL]
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
    CALIB,
    _first_segment,
    make_caches,
    run_case,
    score_answer,
)
from veloxquant_mlx.quantizers.qfilters_calibration import (  # noqa: E402
    average_gqa_filters,
    collect_query_activations,
    compute_qfilters,
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
BUDGET = 512
CTX = 1024
SEEDS = 3
CHUNKS = [None, 256, 64]  # None == one-shot, the harness default


def run_chunked(model, tok, caches, prompt, expected, pool, chunk, max_new=48):
    """Prefill in ``chunk``-token blocks, then greedy-decode and score.

    Each block is a separate ``update_and_fetch``, so Q-Filters evicts once
    per block against a filter frozen on that block rather than making one
    decision over the whole prompt.
    """
    ids = tok.encode(prompt)
    logits = None
    for i in range(0, len(ids), chunk):
        logits = model(mx.array([ids[i : i + chunk]]), cache=caches)
        mx.eval(logits)
    cur = mx.argmax(logits[0, -1]).reshape(1, 1)
    out = [int(cur.item())]
    for _ in range(max_new - 1):
        logits = model(cur, cache=caches)
        cur = mx.argmax(logits[0, -1]).reshape(1, 1)
        out.append(int(cur.item()))
    return score_answer(_first_segment(tok.decode(out)), expected, pool)


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model, tok = mlx_lm.load(model_id)
    inner = getattr(model, "model", model)
    n_layers = len(inner.layers)
    kv_heads = model.args.num_key_value_heads
    head_dim = (
        getattr(model.args, "head_dim", None)
        or model.args.hidden_size // model.args.num_attention_heads
    )

    print("calibrating query-SVD filters ...", flush=True)
    acts = collect_query_activations(model, tok, CALIB, max_length=512, max_samples_per_head=2000)
    filters = [average_gqa_filters(compute_qfilters(a), kv_heads) for a in acts]
    del acts
    gc.collect()

    print(f"\nVT, {model_id}, budget {BUDGET}, ctx {CTX} -- prefill chunk sweep ({SEEDS} seeds)")
    for chunk in CHUNKS:
        scores = []
        for s in range(SEEDS):
            prompt, expected, pool = BUILDERS["vt"](tok, CTX, CTX * 100 + s)
            caches = make_caches("qfilters", n_layers, head_dim, BUDGET, filters)
            if chunk is None:
                sc = run_case(model, tok, caches, prompt, expected, pool)[0]
            else:
                sc = run_chunked(model, tok, caches, prompt, expected, pool, chunk)
            scores.append(sc)
            del caches
            gc.collect()
        label = "one-shot" if chunk is None else f"{chunk}-tok chunks"
        mean = sum(scores) / len(scores)
        print(f"  {label:<16} mean={mean:.2f}  {[f'{x:.2f}' for x in scores]}", flush=True)


if __name__ == "__main__":
    main()
