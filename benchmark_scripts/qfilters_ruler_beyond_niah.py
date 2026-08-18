"""RULER task categories beyond NIAH, for Q-Filters vs SnapKV / StreamingLLM / L2Norm.

Closes the gap the NIAH harness explicitly left open (see the "WHAT THIS IS
NOT" block in ``qfilters_ttft_throughput_niah.py``, and "Still not measured"
in the Q-Filters docs): the retrieval arm there is the needle family only, so
nothing established whether Q-Filters generalises past single-span lookup.

WHAT IS COVERED
---------------
Four task categories, generated synthetically in-process following the
constructions in RULER (Hsieh et al., "RULER: What's the Real Context Size of
Your Long-Context Language Models?", arXiv:2404.06654):

* **VT** — variable tracking. A chain of assignments (``VAR X1 = 12345``,
  ``VAR X2 = X1`` ...) is scattered through filler; the model must name every
  variable bound to a given value. Tests multi-hop traversal, not lookup:
  the answer spans positions that are individually unremarkable.
* **CWE** — common-word extraction. A word list mixes a few words repeated
  many times with many words repeated twice; the model must report the
  common ones. Tests aggregation over the whole context.
* **FWE** — frequent-word extraction. Word frequencies follow a Zeta
  distribution as in the paper, so the top-3 are separated by frequency
  alone rather than by a hard threshold.
* **QA** — multi-hop question answering over distractor paragraphs.

WHAT THIS IS NOT
----------------
* **Not the full RULER suite, and not RULER's own harness.** RULER ships 13
  tasks; this covers the four categories outside the needle family, at short
  contexts, with generators written here rather than RULER's code. Numbers
  are not comparable to published RULER scores — treat them as a
  *relative* comparison between the cache methods on this machine.
* **Not RULER's QA task.** RULER's QA wraps SQuAD and HotpotQA. Pulling those
  in would add a dataset dependency to a repo that has none, so the QA arm
  here is synthetic multi-hop over generated paragraphs. It measures the same
  capability shape (locate two facts, join them) but is an easier task, and
  it is labelled ``qa_synthetic`` everywhere to keep that distinction visible.
* **Not Expected Attention.** Not implemented in this repo, so it cannot be
  a comparison arm (issue #177 says the same).
* **Not a speedup claim.** Same caveat as the NIAH harness: these caches run
  a per-(B, H) Python loop in ``update_and_fetch``, so wall-clock is an
  artifact of the implementation. No timings are reported here at all.

PREFILL PATH-DEPENDENCE APPLIES HERE TOO
----------------------------------------
``qfilters_update`` absorbs a block then evicts to budget in one shot, so a
single-shot prefill makes one large eviction decision against a filter frozen
on that same block. The NIAH harness measured this costing coherence outright
at budget 256. This harness prefills the same way — one ``model(prompt,
cache=...)`` call — so the Q-Filters rows are that method's worst case, and a
chunked-prefill deployment should do better than what is reported here.

Usage:
    python benchmark_scripts/qfilters_ruler_beyond_niah.py [MODEL ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import KVCache

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from veloxquant_mlx.cache.base import KVCacheConfig  # noqa: E402
from veloxquant_mlx.cache.knorm_cache import L2NormKVCache  # noqa: E402
from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache  # noqa: E402
from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache  # noqa: E402
from veloxquant_mlx.cache.streaming_llm_cache import StreamingLLMKVCache  # noqa: E402
from veloxquant_mlx.quantizers.qfilters_calibration import (  # noqa: E402
    average_gqa_filters,
    collect_query_activations,
    compute_qfilters,
)

DEFAULT_MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
]

# Matched to the NIAH harness so the two result sets sit on the same axis.
BUDGETS = [256, 512, 1024]
CTX_LENS = [1024, 2048]
METHODS = ["fp16", "qfilters", "snapkv", "streaming_llm", "knorm"]
TASKS = ["vt", "cwe", "fwe", "qa_synthetic"]

# Per-task seed offsets. CWE and FWE both derive their answer from a shuffle of
# WORD_POOL, so without this they would produce identical expected words for
# the same (ctx, repeat) and the two tasks would not be independent draws.
TASK_SEED_OFFSET = {"vt": 0, "cwe": 10_000, "fwe": 20_000, "qa_synthetic": 30_000}

# Minimum fp16 score for a (model, task) cell to be treated as a real
# comparison. Below this the model cannot do the task with a *complete* cache,
# so every eviction arm sits in floor noise and any ordering between them is
# sampling luck. Measured at fp16, ctx 1024, 3 seeds: Llama-3.2-1B scores
# 0.11 / 0.16 / 0.26 on vt / cwe / fwe and belongs below this line; Qwen2.5-7B
# scores 0.66 / 1.00 / 0.78 and belongs above it.
CEILING_MIN = 0.50

# Calibration corpus for the query-SVD filters. Identical to the NIAH
# harness's, and deliberately disjoint from every eval construction below —
# no variable names, no word lists, no QA entities.
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

# Filler prose. Bland and repetitive in topic but not literally looped, so the
# task-bearing spans are the only distinctive ones. Shared with the NIAH
# harness on purpose: a difference between the two result sets should come
# from the task, not from the haystack.
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

# Word pool for CWE / FWE. Common English nouns, no overlap with the filler
# sentences above — otherwise filler occurrences would corrupt the counts the
# task is asking the model to make.
WORD_POOL = [
    "lantern",
    "harbour",
    "meadow",
    "compass",
    "granite",
    "thicket",
    "cobalt",
    "pelican",
    "trellis",
    "quarry",
    "sable",
    "juniper",
    "marlin",
    "fennel",
    "obsidian",
    "cypress",
    "pewter",
    "walrus",
    "saffron",
    "basalt",
    "otter",
    "amber",
    "ridge",
    "cinder",
    "willow",
    "kestrel",
    "onyx",
    "bramble",
    "canyon",
    "lichen",
    "mackerel",
    "topaz",
    "birch",
    "heron",
    "flint",
    "sorrel",
    "puffin",
    "slate",
    "hazel",
    "curlew",
    "opal",
    "alder",
]

# QA entity pool, kept disjoint from WORD_POOL and from the calibration text.
QA_PEOPLE = [
    "Marisol Vance",
    "Teodor Halvorsen",
    "Priya Raghunathan",
    "Callum Ashworth",
    "Ingrid Sorensen",
    "Rafael Duarte",
]
QA_PLACES = ["Kestrel Bay", "Thornfield", "Vellamo", "Ardenmoor", "Cape Lisandro", "Norwick"]
QA_ITEMS = ["seismograph", "astrolabe", "chronometer", "barograph", "theodolite", "hygrometer"]


# ── cache construction ───────────────────────────────────────────────────────
def make_caches(method: str, n_layers: int, head_dim: int, budget: int, filters=None):
    """One cache instance per layer for the named method at a matched budget.

    Every eviction method is configured to retain the same total token count,
    so a row difference is a difference in *which* tokens were kept, not how
    many. ``fp16`` ignores the budget and is the quality ceiling.
    """
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

    if method == "knorm":
        # Included per #174, which generalised the `_true_offset` RoPE fix to
        # this cache. Before that it carried the same position drift the NIAH
        # harness documents, and could not be a fair arm.
        cfg = KVCacheConfig(
            method="knorm",
            head_dim=head_dim,
            knorm_budget=budget,
            knorm_n_sink=4,
            knorm_recent=budget // 4,
        )
        return [L2NormKVCache(cfg) for _ in range(n_layers)]

    raise ValueError(f"unknown method {method!r}")


# ── filler ───────────────────────────────────────────────────────────────────
def _grow_filler(tok, ctx_tokens: int, rng: random.Random) -> list[str]:
    """Sentences totalling at least ``ctx_tokens`` tokens.

    Grown geometrically then trimmed: encoding after every single append is
    O(n^2) in tokeniser calls and dominates runtime at 2k contexts.
    """
    filler = [rng.choice(HAYSTACK_SENTENCES) for _ in range(8)]
    while len(tok.encode(" ".join(filler))) < ctx_tokens:
        filler.extend(rng.choice(HAYSTACK_SENTENCES) for _ in range(len(filler)))
    while len(filler) > 1 and len(tok.encode(" ".join(filler[:-1]))) >= ctx_tokens:
        filler.pop()
    return filler


def _scatter(filler: list[str], spans: list[str], rng: random.Random) -> str:
    """Insert ``spans`` at evenly-spread positions through ``filler``.

    Even spread rather than random placement: with random positions the
    variance across seeds is dominated by whether the chain happened to land
    inside a recency window, which would measure luck rather than method.
    """
    out = list(filler)
    n = len(spans)
    for i, span in enumerate(spans):
        frac = (i + 1) / (n + 1)
        pos = min(int(len(out) * frac), len(out))
        out.insert(pos, span)
    return " ".join(out)


# ── task builders ────────────────────────────────────────────────────────────
def build_vt(tok, ctx_tokens: int, seed: int, n_chains: int = 2, hops: int = 3):
    """Variable tracking: follow assignment chains to their root value.

    ``n_chains`` independent chains of ``hops`` variables each are scattered
    through the filler. One chain's root value is queried; the answer is every
    variable name in that chain. The other chain is a distractor with the same
    surface form, so surface matching on ``VAR`` alone does not solve it.

    ``hops`` defaults to 3 because that is where the fp16 ceiling sits in a
    usable range on the models here. Measured on Qwen2.5-7B at ctx 1024, fp16,
    4 seeds: 4 hops scored 0.46, 3 hops 0.66, 2 hops 1.00. At 4 hops the model
    starts inventing variable names that never appear in the text, and at 2 it
    saturates -- neither leaves room to see a cache method degrade.

    Returns ``(prompt, expected_strings, pool)``. Every expected string must
    appear in the decoded answer; ``pool`` is the candidate set used to
    compute the precision cap in ``score_answer``.
    """
    rng = random.Random(seed)
    chains: list[tuple[int, list[str]]] = []
    per_chain_spans: list[list[str]] = []
    used: set[str] = set()
    for _c in range(n_chains):
        value = rng.randint(10000, 99999)
        names: list[str] = []
        while len(names) < hops:
            nm = f"X{rng.randint(1, 999)}"
            if nm not in used:
                used.add(nm)
                names.append(nm)
        chain_spans = [f"VAR {names[0]} = {value}."]
        chain_spans += [f"VAR {names[i]} = {names[i - 1]}." for i in range(1, hops)]
        per_chain_spans.append(chain_spans)
        chains.append((value, names))

    target_value, target_names = chains[0]

    # Interleave the chains but keep each chain's spans in assignment order.
    # A global shuffle would place `VAR B = A` before `VAR A = 12345`, forcing
    # backward resolution over a scrambled graph -- much harder than RULER's
    # VT, which scatters hops through the text without reordering them. On
    # Qwen2.5-7B the shuffled form scored 0.29 at fp16, too low a ceiling to
    # tell cache methods apart.
    spans = [s for group in zip(*per_chain_spans, strict=True) for s in group]

    filler = _grow_filler(tok, ctx_tokens, rng)
    body = _scatter(filler, spans, rng)
    prompt = (
        "Below is a text containing variable assignments. A variable may be "
        "assigned a number directly, or assigned the value of another "
        "variable, in which case it inherits that variable's value.\n\n"
        f"{body}\n\n"
        f"Question: Find all variables that are assigned the value "
        f"{target_value}, including variables that inherit it indirectly "
        f"through other variables. There are exactly {hops}.\n"
        "Answer: The variables are"
    )
    # Pool = every variable name in the text, so naming all of them scores
    # hops / (n_chains * hops) rather than full marks.
    all_names = [n for _v, ns in chains for n in ns]
    return prompt, list(target_names), all_names


def build_cwe(tok, ctx_tokens: int, seed: int, n_common: int = 3):
    """Common-word extraction: report the words that repeat many times.

    ``n_common`` words appear 10x; the rest of the list appears 2x each. The
    list is padded with filler to the context length. Aggregation over the
    whole context — no single position carries the answer, which is precisely
    the property a recency-window method cannot exploit.
    """
    rng = random.Random(seed)
    pool = list(WORD_POOL)
    rng.shuffle(pool)
    common = pool[:n_common]
    rare = pool[n_common:]

    words = []
    for w in common:
        words.extend([w] * 10)
    for w in rare:
        words.extend([w] * 2)
    rng.shuffle(words)

    filler = _grow_filler(tok, max(ctx_tokens - len(words), 64), rng)
    body = " ".join(filler)
    prompt = (
        f"{body}\n\n"
        "Below is a list of words. Some words appear many times, most appear "
        "only twice.\n\n"
        f"{', '.join(words)}\n\n"
        f"Question: Which {n_common} words appear most often in the list "
        "above?\nAnswer: The words are"
    )
    return prompt, common, list(set(words))


def build_fwe(tok, ctx_tokens: int, seed: int, alpha: float = 2.0, top_k: int = 3):
    """Frequent-word extraction with Zeta-distributed frequencies.

    RULER draws word frequencies from a Zeta distribution rather than using a
    hard common/rare split, so the top-k are separated by frequency alone.
    That makes this strictly harder than CWE: the boundary between "frequent"
    and "not" is a judgement about counts, not a visible gap.
    """
    rng = random.Random(seed)
    pool = list(WORD_POOL)
    rng.shuffle(pool)
    n = min(len(pool), 20)
    pool = pool[:n]

    # Zeta-like: count(rank r) proportional to r^-alpha, floored at 2 so every
    # word actually appears and the top-k are a genuine ranking.
    counts = [max(2, int(round(40 * (r + 1) ** (-alpha)))) for r in range(n)]
    words = []
    for w, c in zip(pool, counts, strict=True):
        words.extend([w] * c)
    rng.shuffle(words)

    expected = pool[:top_k]
    filler = _grow_filler(tok, max(ctx_tokens - len(words), 64), rng)
    body = " ".join(filler)
    prompt = (
        f"{body}\n\n"
        "Below is a list of words whose frequencies differ.\n\n"
        f"{', '.join(words)}\n\n"
        f"Question: Which {top_k} words appear most frequently in the list "
        "above?\nAnswer: The words are"
    )
    return prompt, expected, list(set(words))


def build_qa_synthetic(tok, ctx_tokens: int, seed: int, n_distractor: int = 3):
    """Two-hop QA over generated paragraphs.

    NOT RULER's QA task, which wraps SQuAD/HotpotQA — see the module
    docstring. Two supporting facts sit far apart in the context and must be
    joined: person -> place, place -> item. Distractor paragraphs use the same
    template with different entities, so retrieving one hop is not enough.
    """
    rng = random.Random(seed)
    people = rng.sample(QA_PEOPLE, n_distractor + 1)
    places = rng.sample(QA_PLACES, n_distractor + 1)
    items = rng.sample(QA_ITEMS, n_distractor + 1)

    hop1, hop2, spans = [], [], []
    for i in range(n_distractor + 1):
        hop1.append(f"{people[i]} spent the survey season at {places[i]}.")
        hop2.append(f"The instrument installed at {places[i]} was a {items[i]}.")
    # Interleave so the two hops for the target are never adjacent.
    spans = hop1 + hop2

    filler = _grow_filler(tok, ctx_tokens, rng)
    body = _scatter(filler, spans, rng)
    prompt = (
        "Read the following text carefully and then answer the question.\n\n"
        f"{body}\n\n"
        f"Question: Which instrument was installed at the location where "
        f"{people[0]} spent the survey season?\nAnswer:"
    )
    # No pool: the answer space is free-form prose, not a candidate list.
    return prompt, [items[0]], None


BUILDERS = {
    "vt": build_vt,
    "cwe": build_cwe,
    "fwe": build_fwe,
    "qa_synthetic": build_qa_synthetic,
}


# ── scoring ──────────────────────────────────────────────────────────────────
# Chat models keep talking after answering: they emit an EOS-ish special token
# and then start a fresh turn that re-states or revises the answer. Only the
# first segment is the answer to the question that was asked.
_STOP_MARKERS = ("<|eot_id|>", "<|im_end|>", "<|endoftext|>", "<|start_header_id|>")


def _first_segment(text: str) -> str:
    """The answer up to the first turn boundary.

    Without this the scorer sees the model's follow-on turns too, and a model
    that answers wrongly then rambles through a dozen more candidates gets
    credited for whichever one happens to be right.
    """
    cut = len(text)
    for marker in _STOP_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    seg = text[:cut]
    # Newline-newline is a turn boundary for models that do not emit a special
    # token here at all.
    j = seg.find("\n\n")
    return seg[:j] if j != -1 else seg


def _normalise(text: str) -> str:
    """Lowercase and collapse to bare alphanumeric tokens for matching."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def score_answer(text: str, expected: list[str], pool: list[str] | None = None) -> float:
    """Score one answer in [0, 1]: recall of the expected items, capped by precision.

    Partial credit, not exact match. VT and CWE/FWE expect several items, and
    an all-or-nothing score would collapse "found two of three" and "found
    none" into the same 0 — exactly the distinction this evaluation exists to
    draw. Matching is on normalised whitespace-delimited tokens, so a
    substring hit inside a longer word does not count.

    Recall alone is not enough, though. On CWE a model that simply lists
    fourteen words from the list scores 100% recall without having aggregated
    anything, and that was observed on Llama-3.2-1B during development. So
    when ``pool`` is supplied (the candidate set the answer is drawn from),
    the score is multiplied by precision over the pool words actually named.
    Naming every candidate then scores ``len(expected) / len(pool)``, not 1.0.

    ``pool`` is None for tasks with a free-form answer space (QA), where there
    is no candidate set to measure precision against.
    """
    if not expected:
        return 0.0
    hay = _normalise(text)
    hits = sum(1 for e in expected if f" {_normalise(e).strip()} " in hay)
    recall = hits / len(expected)
    if pool is None or recall == 0.0:
        return recall
    named = sum(1 for w in pool if f" {_normalise(w).strip()} " in hay)
    precision = hits / named if named else 0.0
    return recall * precision


def run_case(
    model,
    tok,
    caches,
    prompt: str,
    expected: list[str],
    pool: list[str] | None = None,
    max_new: int = 48,
) -> tuple[float, str]:
    """Greedy-decode an answer and score it. Returns ``(score, first_segment)``.

    Greedy rather than sampled: the comparison is between caches, and sampling
    noise would need many more repeats per cell to see through.
    """
    ids = tok.encode(prompt)
    logits = model(mx.array([ids]), cache=caches)
    cur = mx.argmax(logits[0, -1]).reshape(1, 1)
    out = [int(cur.item())]
    for _ in range(max_new - 1):
        logits = model(cur, cache=caches)
        cur = mx.argmax(logits[0, -1]).reshape(1, 1)
        out.append(int(cur.item()))
    text = _first_segment(tok.decode(out))
    return score_answer(text, expected, pool), text


# ── driver ───────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*", default=None)
    ap.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="repeats per (task, ctx) cell; each uses a distinct seed",
    )
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    models = args.models or DEFAULT_MODELS
    out_path = (
        Path(args.out)
        if args.out
        else (_repo_root / "figures" / "qfilters" / "ruler_beyond_niah.json")
    )

    results: dict = {
        "_comment": (
            "RULER task categories outside the NIAH family, for issue #177. "
            "Generators follow the constructions in Hsieh et al. "
            "(arXiv:2404.06654) but are written in this repo, not RULER's own "
            "harness, and run at short contexts. Scores are NOT comparable to "
            "published RULER numbers -- they are a relative comparison between "
            "cache methods on identical prompts. 'qa_synthetic' is not RULER's "
            "QA task (which wraps SQuAD/HotpotQA); it is a synthetic two-hop "
            "task and is easier. Expected Attention is not implemented in this "
            "repo and is therefore absent as an arm."
        ),
        "scoring": (
            "Fraction of expected strings present in the greedy-decoded answer "
            "(partial credit), matched on normalised whitespace-delimited "
            "tokens. Reported per cell as the mean over seeds."
        ),
        "config": {
            "budgets": BUDGETS,
            "context_lengths": CTX_LENS,
            "methods": METHODS,
            "tasks": TASKS,
            "seeds_per_cell": args.seeds,
            "prefill": "one-shot (worst case for Q-Filters; see module docstring)",
            "qfilters_recent": "budget // 4",
            "knorm_recent": "budget // 4",
        },
        "models": {},
    }

    for model_id in models:
        print(f"\n{'=' * 78}\n{model_id}\n{'=' * 78}", flush=True)
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

        mrec: dict = {}
        samples: dict = {}
        prompt_lens: dict = {}

        for task in TASKS:
            print(f"\n--- {task} ---", flush=True)
            print(
                f"{'method':<16}{'budget':>8}"
                + "".join(f"{f'ctx {c}':>10}" for c in CTX_LENS)
                + f"{'mean':>9}"
            )
            mrec[task] = {}
            fp16_mean = None
            for method in METHODS:
                # fp16 ignores the budget entirely -- measure it once and carry
                # that row as the ceiling for every budget.
                budgets = [BUDGETS[0]] if method == "fp16" else BUDGETS
                for nb in budgets:
                    per_ctx = []
                    for ctx in CTX_LENS:
                        cell = []
                        for s in range(args.seeds):
                            # Offset per task: CWE and FWE both shuffle
                            # WORD_POOL, so a shared seed would hand them the
                            # same answer set and stop them being independent
                            # samples of the same model.
                            seed = ctx * 100 + s + TASK_SEED_OFFSET[task]
                            prompt, expected, pool = BUILDERS[task](tok, ctx, seed)
                            # CTX_LENS sizes the *filler*; the instruction,
                            # task spans and question add ~100-250 tokens on
                            # top. Record what was actually prefilled so the
                            # compression ratio a budget implies is checkable
                            # rather than inferred from the label. This is why
                            # budget 1024 still evicts at ctx 1024.
                            prompt_lens.setdefault(task, {}).setdefault(
                                str(ctx), len(tok.encode(prompt))
                            )
                            caches = make_caches(method, n_layers, head_dim, nb, filters)
                            sc, text = run_case(model, tok, caches, prompt, expected, pool)
                            cell.append(sc)
                            if s == 0 and ctx == CTX_LENS[0]:
                                samples[f"{task}/{method}/{nb}"] = {
                                    "expected": expected,
                                    "decoded": text[:200],
                                    "score": sc,
                                    "pool_size": len(pool) if pool else None,
                                }
                            del caches
                            gc.collect()
                        per_ctx.append(sum(cell) / len(cell))
                    mean = sum(per_ctx) / len(per_ctx)
                    if method == "fp16":
                        fp16_mean = mean
                    label = "fp16 (any)" if method == "fp16" else method
                    mrec[task].setdefault(method, {})[str(nb)] = {
                        "per_ctx": {str(c): v for c, v in zip(CTX_LENS, per_ctx, strict=True)},
                        "mean": mean,
                    }
                    print(
                        f"{label:<16}{nb:>8}"
                        + "".join(f"{v:>9.0%} " for v in per_ctx)
                        + f"{mean:>8.0%}",
                        flush=True,
                    )

            # Record whether this (model, task) cell can discriminate at all.
            # Where fp16 itself scores near the floor, the eviction arms are
            # measuring the model's inability to do the task, not the cache's
            # effect on it, and the row must not be read as a method comparison.
            # Llama-3.2-1B lands here on vt/cwe/fwe; Qwen2.5-7B clears it on all four.
            mrec[task]["_fp16_ceiling"] = fp16_mean
            mrec[task]["_discriminative"] = bool(fp16_mean is not None and fp16_mean >= CEILING_MIN)
            if not mrec[task]["_discriminative"]:
                print(
                    f"  [!] fp16 ceiling {fp16_mean:.0%} < {CEILING_MIN:.0%} -- "
                    f"this model cannot do {task}; rows above are not a method comparison",
                    flush=True,
                )

        results["models"][model_id] = {
            "tasks": mrec,
            "prompt_tokens": prompt_lens,
            "samples": samples,
        }
        # Write after each model: a 7B run is long enough that losing it to a
        # crash on the next model would be a real cost.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n  [checkpoint] {out_path}", flush=True)

        del model, tok, filters
        gc.collect()

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
