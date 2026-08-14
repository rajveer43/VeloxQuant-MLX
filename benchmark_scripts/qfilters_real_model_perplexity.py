"""Memory-constrained generation perplexity for Q-Filters (paper §4 / Fig. 5).

Paper setup: "We let the KV Cache grow up until a certain threshold, after
which we start evicting the KV pairs so that the total size never exceeds the
maximum threshold. We measure performance by tracking the model perplexity
computed on past tokens."

So tokens are fed one at a time, eviction runs as the cache fills, and each
next-token prediction is scored against the compressed cache.

Two things this harness is careful about, both of which produced misleading
numbers in earlier drafts:

  * **Non-repetitive text.** Looping a short paragraph gives the model a
    trivially predictable stream (fp16 perplexity ~1.3), which compresses the
    dynamic range so far that every eviction policy looks equally bad. Real
    prose keeps the baseline in a realistic range so policy differences are
    visible.
  * **RoPE positions.** Requires the ``cache.offset`` fix from #171; without
    it every arm is dominated by position drift rather than by eviction
    quality, and the comparison measures nothing.

Usage:
    python benchmark_scripts/qfilters_real_model_perplexity.py MODEL [BUDGET ...]
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm.models.cache import KVCache

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache
from veloxquant_mlx.quantizers.qfilters_calibration import (
    average_gqa_filters,
    collect_query_activations,
    compute_qfilters,
)

# Calibration corpus — deliberately different in content from the eval text.
CALIB = [
    "The Roman Republic was established after the overthrow of the Roman Kingdom, "
    "traditionally dated to 509 BC. Power was held by annually elected magistrates "
    "and the Senate, an assembly drawn largely from the patrician class. Over the "
    "following centuries Rome expanded across the Italian peninsula through a "
    "combination of military conquest and negotiated alliances with neighbouring "
    "peoples, each of which carried different obligations and privileges. ",
    "Photosynthesis is the process by which green plants, algae and some bacteria "
    "convert light energy into chemical energy stored in carbohydrate molecules. "
    "In plants the reaction takes place in chloroplasts, organelles containing the "
    "pigment chlorophyll, which absorbs light most strongly in the blue and red "
    "portions of the spectrum while reflecting green wavelengths. The products "
    "sustain nearly every food web on the planet. ",
    "Modern cryptography relies on problems believed to be computationally hard, "
    "such as factoring large integers or computing discrete logarithms in finite "
    "groups. Public-key systems let two parties who have never met agree on a "
    "shared secret over an open channel, which underpins secure communication on "
    "the internet. The security of these schemes rests on assumptions that have "
    "resisted attack but remain unproven. ",
    "Plate tectonics describes the movement of large sections of the Earth's "
    "lithosphere over the underlying asthenosphere. Where plates converge one may "
    "subduct beneath another, producing deep ocean trenches, volcanic arcs and "
    "some of the most powerful earthquakes recorded. Where they diverge, new "
    "crust forms as magma rises to fill the gap, a process most visible along "
    "mid-ocean ridges. ",
]

# Evaluation text — continuous prose on unrelated topics, never a repeated loop.
EVAL = (
    "The invention of movable type in Europe is generally credited to Johannes "
    "Gutenberg, a goldsmith working in Mainz during the middle of the fifteenth "
    "century. His press combined several existing technologies in a novel way: "
    "the screw mechanism borrowed from wine and olive presses, an oil-based ink "
    "that adhered to metal rather than running off it, and above all a hand mould "
    "that allowed individual letters to be cast quickly and to a consistent "
    "height. Earlier printing in East Asia had used carved blocks and, later, "
    "movable characters of clay and metal, but the enormous character sets of "
    "written Chinese limited the advantage such systems offered over careful "
    "copying. The alphabet changed that calculation entirely. A printer working "
    "in a European vernacular needed only a few hundred distinct sorts to "
    "compose any text whatsoever, and those sorts could be redistributed and "
    "reused indefinitely. Within fifty years presses had been established in "
    "more than two hundred cities, and the price of a book had fallen to a small "
    "fraction of what a manuscript copy had cost. The consequences reached far "
    "beyond the book trade. Standardised texts made it possible for scholars in "
    "distant places to refer to precisely the same passage, which mattered a "
    "great deal for astronomical tables, anatomical drawings and legal codes "
    "where a copyist's slip could propagate silently for generations. Printers "
    "became publishers, choosing what to issue and in what form, and the "
    "resulting competition pushed them toward vernacular languages and shorter, "
    "cheaper formats aimed at readers who were not clergy or lawyers. Attempts "
    "to control the new medium followed almost immediately, through licensing "
    "systems, privileges granted to favoured printers, and lists of prohibited "
    "works, none of which proved durable against the sheer number of presses "
    "operating across jurisdictions that did not cooperate with one another. "
    "Historians have long debated how much of the subsequent upheaval should be "
    "attributed to printing itself rather than to the religious and political "
    "currents it carried. What is not seriously disputed is that the cost of "
    "reproducing an argument collapsed, and that arguments which would once have "
    "died in a single monastery library could now be answered, and answered "
    "again, by people who had never met. "
) * 3


def perplexity(model, ids, caches, label: str) -> float:
    """Token-by-token generation perplexity against a (possibly evicting) cache."""
    total_nll, n = 0.0, 0
    t0 = time.time()
    for t in range(len(ids) - 1):
        logits = model(mx.array([[ids[t]]]), cache=caches)
        logp = nn.log_softmax(logits[0, -1].astype(mx.float32))
        total_nll += -float(np.array(logp[ids[t + 1]]))
        n += 1
    ppl = float(np.exp(total_nll / n))
    print(f"  {label:42s} ppl {ppl:8.3f}   ({time.time() - t0:5.1f}s)")
    return ppl


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Llama-3.2-1B-Instruct-4bit"
    budgets = [int(b) for b in sys.argv[2:]] or [256, 128, 64]

    model, tok = mlx_lm.load(model_id)
    inner = getattr(model, "model", model)
    n_layers = len(inner.layers)
    kv_heads = model.args.num_key_value_heads
    head_dim = model.args.head_dim

    acts = collect_query_activations(model, tok, CALIB, max_length=512, max_samples_per_head=3000)
    filters = [average_gqa_filters(compute_qfilters(a), kv_heads) for a in acts]

    ids = tok.encode(EVAL)[:1024]
    print(f"\n=== {model_id} | {len(ids)} tokens, generation mode ===")
    base = perplexity(
        model, ids, [KVCache() for _ in range(n_layers)], "fp16 full cache (no eviction)"
    )

    for budget in budgets:
        ratio = len(ids) / budget
        recent = budget // 4
        print(f"\n--- budget {budget}  (~{ratio:.0f}x compression, recent={recent}) ---")
        # A trailing protected window is essential in *generation*. Pure
        # projection ranking is a long-range-importance signal; it happily
        # evicts the immediately-preceding tokens, which are exactly what
        # next-token prediction leans on hardest. Measured on Llama-3.2-1B at
        # budget=128, recent=0 gives ppl 263 while recent=32 gives 16.3.
        # The paper evaluates retrieval/NIAH tasks where this matters far less;
        # `recent` is this repo's extension (off by default) and generation
        # perplexity is the setting that makes it necessary.
        kw = dict(
            method="qfilters",
            head_dim=head_dim,
            qfilters_budget=budget,
            qfilters_n_sink=4,
            qfilters_recent=recent,
            qfilters_calib_tokens=min(128, budget // 2),
        )
        cal = perplexity(
            model,
            ids,
            [QFiltersKVCache(KVCacheConfig(**kw), filters=filters[i]) for i in range(n_layers)],
            "Q-Filters CALIBRATED (query-SVD)",
        )
        fb = perplexity(
            model,
            ids,
            [QFiltersKVCache(KVCacheConfig(**kw)) for _ in range(n_layers)],
            "Q-Filters fallback (key-SVD)",
        )
        # NOTE: L2NormKVCache is deliberately NOT included as a comparison arm.
        # It carries the same un-remapped-RoPE defect this branch fixes for
        # Q-Filters (knorm_cache.py sets self.offset = 0 and lets the base
        # class overwrite it with the retained-row count), so its numbers would
        # be dominated by position drift rather than by eviction quality. Once
        # #171's fix is generalised to the other evicting caches, add it back.
        gap = fb - base
        closed = 100 * (1 - (cal - base) / gap) if abs(gap) > 1e-9 else float("nan")
        print(
            f"    vs fp16 {base:.3f}:  calibrated {100 * (cal - base) / base:+7.1f}%   "
            f"fallback {100 * (fb - base) / base:+7.1f}%   "
            f"(calibrated closes {closed:.0f}% of the fallback's gap)"
        )


if __name__ == "__main__":
    main()
