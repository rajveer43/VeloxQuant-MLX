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
from veloxquant_mlx.cache.knorm_cache import L2NormKVCache
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
    "Longitude at sea resisted solution for far longer than latitude, which any "
    "navigator could fix by measuring the sun at noon or the pole star at night. "
    "Longitude required knowing the time at a reference meridian at the same "
    "instant as local time, and no pendulum clock could keep that reference "
    "aboard a rolling ship through changes of temperature and humidity. The "
    "astronomical alternative, the method of lunar distances, demanded tables of "
    "the moon's position accurate to a degree no observatory had yet reached, and "
    "a set of calculations that took a trained officer four hours to complete. "
    "John Harrison spent decades building sea clocks against this problem, moving "
    "from large machines with interlocking grasshopper escapements to a watch "
    "barely five inches across, and the board charged with judging his work kept "
    "raising what would count as proof. The eventual solution in practice was "
    "neither one method nor the other but both together, chronometers checked "
    "against lunars when the sky allowed, until the price of a reliable timepiece "
    "fell far enough that every merchant vessel could carry one. "
    "Bird migration was for centuries explained by theories that now read as "
    "fantasy, including the belief that swallows spent the winter in the mud at "
    "the bottom of ponds, an idea repeated by serious naturalists into the "
    "eighteenth century. The evidence that settled the question arrived piecemeal: "
    "a stork shot in Germany carrying a central African spear through its neck, "
    "then systematic ringing programmes that recovered numbered bands from birds "
    "thousands of miles from where they were fitted. What the recoveries revealed "
    "was more surprising than mere distance. Individual birds return not just to "
    "the same country but often to the same hedgerow, navigating by a combination "
    "of the sun's arc, the pattern of polarised light at dusk, star rotation "
    "around the celestial pole, and a magnetic sense whose receptor is still "
    "argued over. Young birds of some species make the first journey alone, "
    "without any adult to follow, which means the route is inherited rather than "
    "taught, while in others the knowledge is cultural and a population that "
    "loses its experienced birds loses the route with them. "
    "The problem of measuring the speed of light was thought hopeless by "
    "observers who assumed propagation was instantaneous. Galileo proposed "
    "stationing two people on hilltops with shuttered lanterns and found, "
    "unsurprisingly, only the delay of human reaction. The first real value came "
    "not from a terrestrial experiment but from watching the moons of Jupiter, "
    "whose eclipses ran early when Earth approached and late when it receded, a "
    "discrepancy Ole Romer interpreted as the time light took to cross the "
    "diameter of Earth's orbit. His figure was low, because the size of that "
    "orbit was itself poorly known, but the method was sound and the principle "
    "established. Later terrestrial measurements using toothed wheels and "
    "rotating mirrors closed the gap, and the constant eventually became so well "
    "determined that the metre was redefined in terms of it, inverting the "
    "original relationship: the speed of light is now exact by definition and it "
    "is the length of the metre that follows from it. "
    "Cartographic projection forces an unavoidable compromise, since no flat "
    "sheet can represent a sphere without distorting area, angle, or distance "
    "somewhere. The Mercator projection preserves angles, which is exactly what a "
    "navigator wants because a straight line drawn on the chart is a course of "
    "constant compass bearing, but it inflates land near the poles without limit "
    "and has been criticised for the political impression that inflation leaves. "
    "Equal-area projections correct the sizes and pay for it by shearing shapes "
    "into forms that look wrong to eyes trained on the familiar arrangement. "
    "There is no projection that is best in general, only projections that are "
    "appropriate for a purpose, and the long history of the subject is largely a "
    "history of people rediscovering that fact and proposing a new compromise "
    "they believe resolves it. "
) * 3

# The corpus above yields ~3.5k tokens under the Llama/Qwen tokenizers, so this
# cap is a ceiling rather than a target; budgets 256/512/1024 then run at
# roughly 13.6x/6.8x/3.4x compression. Any budget >= the token count would
# never fill the cache, and every arm would return the fp16 baseline exactly.
EVAL_TOKENS = 4096


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
    # Qwen2's ModelArgs omits head_dim; Llama's carries it. Derive when absent.
    head_dim = (
        getattr(model.args, "head_dim", None)
        or model.args.hidden_size // model.args.num_attention_heads
    )

    acts = collect_query_activations(model, tok, CALIB, max_length=512, max_samples_per_head=3000)
    filters = [average_gqa_filters(compute_qfilters(a), kv_heads) for a in acts]

    # Must exceed the largest budget or the cache never fills and no eviction
    # runs: at budget == len(ids) every arm returns the fp16 baseline exactly.
    ids = tok.encode(EVAL)[:EVAL_TOKENS]
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
        # L2NormKVCache is now a fair comparison arm: #174 generalised the
        # #171 RoPE-offset fix to knorm_cache.py, so its offset tracks the
        # true absolute token position through eviction instead of stalling
        # at the retained-row count. Uses knorm's own trailing-window knob
        # (knorm_recent) for the same reason qfilters_recent is needed above.
        kn = perplexity(
            model,
            ids,
            [
                L2NormKVCache(
                    KVCacheConfig(
                        method="knorm",
                        head_dim=head_dim,
                        knorm_budget=budget,
                        knorm_n_sink=4,
                        knorm_recent=recent,
                    )
                )
                for _ in range(n_layers)
            ],
            "L2Norm (key-norm eviction)",
        )
        gap = fb - base
        closed = 100 * (1 - (cal - base) / gap) if abs(gap) > 1e-9 else float("nan")
        print(
            f"    vs fp16 {base:.3f}:  calibrated {100 * (cal - base) / base:+7.1f}%   "
            f"fallback {100 * (fb - base) / base:+7.1f}%   "
            f"L2Norm {100 * (kn - base) / base:+7.1f}%   "
            f"(calibrated closes {closed:.0f}% of the fallback's gap)"
        )


if __name__ == "__main__":
    main()
