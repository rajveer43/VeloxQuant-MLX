"""Real-model paired-NLL check of SKVQ's channel reorder and bit-width defaults (#906).

``benchmark_skvq.py`` validates SKVQ's machinery on synthetic channels and says
in so many words that the paper's premise cannot be validated there. This
script runs the premise on real models: WikiText-2 test, teacher-forced, paired
against an fp16 cache on the same tokens.

Per prompt length and key/value bit-width it compares three reorder policies:

  * ``old``  -- the pre-#906 behaviour: every head gets the range-sorted
                channel permutation computed from the first 128 rows,
                sink rows included.
  * ``off``  -- ``skvq_reorder=False`` (identity grouping).
  * ``new``  -- the shipped per-head chooser: a head keeps the sorted order
                only if it lowers that head's round-trip error on the chunk.

plus KIVI-2 / KIVI-4 as references. Reported per config: mean paired dNLL
against fp16 (prefill tokens), its standard error (naive, and over 128-token
blocks, which is the honest one because neighbouring tokens are correlated),
top-1 agreement with fp16, perplexity, and the accounted full-KV ratio. It also
reports the paired difference ``new - old`` and ``new - off`` on the same
tokens, and whether a repeated run is bit-identical.

The ratios count bytes; they are not measured memory.

Usage (needs network once, for the WikiText-2 rows)::

    python benchmark_scripts/benchmark_skvq_realmodel.py \\
        --model mlx-community/Qwen3-4B-4bit \\
        --out /tmp/skvq_qwen3.json

Committed results: ``figures/skvq/realmodel_wikitext_results.json``.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from veloxquant_mlx import KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.cache.skvq_cache import SKVQKVCache

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = {"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test"}
N_ROWS = 2400  # first 2,400 rows of the test split: ~170k Qwen3 tokens
BIT_WIDTHS = ((2, 2), (3, 2), (4, 2), (4, 4))  # (key bits, value bits)
BLOCK = 128  # SKVQ's flush window, also the block size for the standard error


def fetch_rows(cache_path: Path | None) -> list[str]:
    """First ``N_ROWS`` WikiText-2 test rows (cached locally when a path is given)."""
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text())["rows"]
    rows: list[str] = []
    for offset in range(0, N_ROWS, 100):
        query = urllib.parse.urlencode({**DATASET, "offset": offset, "length": 100})
        with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as resp:
            rows += [r["row"]["text"] for r in json.load(resp)["rows"]]
    if cache_path is not None:
        cache_path.write_text(json.dumps({"rows": rows}))
    return rows


def _target_logprob(logits: mx.array, targets: mx.array) -> tuple[mx.array, mx.array]:
    """Log-prob of each target and the argmax, both ``[S]``."""
    x = logits.astype(mx.float32)
    lp = mx.take_along_axis(x, targets[:, None], axis=-1)[:, 0] - mx.logsumexp(x, axis=-1)
    return lp, mx.argmax(x, axis=-1)


def teacher_forced(model, caches, ids: list[int], n_prompt: int, chunk: int, n_cont: int):
    """Prefill ``ids[:n_prompt]`` in ``chunk``-sized pieces, then decode ``n_cont`` tokens.

    Returns per-position target log-probs (float64) and argmax ids: positions
    ``[0, n_prompt - 1)`` come from the prefill path, the rest from decode.
    """
    prompt = mx.array(ids[:n_prompt])[None]
    targets = mx.array(ids[1 : n_prompt + n_cont])
    lps, ams = [], []

    def record(logits: mx.array, lo: int, hi: int) -> None:
        lp, am = _target_logprob(logits[0], targets[lo:hi])
        mx.eval(lp, am)
        lps.append(lp)
        ams.append(am)

    i = 0
    while i < n_prompt - 1:
        n = min(chunk, n_prompt - 1 - i)
        logits = model(prompt[:, i : i + n], cache=caches)
        mx.eval(logits)
        record(logits, i, i + n)
        i += n
    record(model(prompt[:, -1:], cache=caches), n_prompt - 1, n_prompt)
    for j in range(n_cont - 1):
        logits = model(mx.array([[ids[n_prompt + j]]]), cache=caches)
        mx.eval(logits)
        record(logits, n_prompt + j, n_prompt + j + 1)
    lp = np.array(mx.concatenate(lps).tolist(), dtype=np.float64)
    am = np.array(mx.concatenate(ams).tolist(), dtype=np.int64)
    return lp, am


def _old_unconditional_perms(self, chunk: mx.array, bits: int):
    """The pre-#906 ``_per_head_perms``: sort every head, sink rows included."""
    B, H, r, D = chunk.shape
    x = chunk.astype(mx.float32).transpose(1, 0, 2, 3).reshape(H, B * r, D)
    perm = mx.argsort(mx.max(x, axis=1) - mx.min(x, axis=1), axis=-1).astype(mx.int32)
    return perm, mx.argsort(perm, axis=-1).astype(mx.int32)


def build_caches(model, method: str, **cfg):
    if method == "fp16":
        return make_prompt_cache(model)
    return KVCacheBuilder.for_model(model, KVCacheConfig(method=method, seed=0, **cfg))


def accounted_ratios(caches) -> dict[str, float | None]:
    """Accounted ratios for SKVQ caches (None for methods without byte counters).

    ``quantized_region`` is fp16 bytes over the compressed region only;
    ``end_to_end`` also charges the fp16 window, sinks and permutations.
    """
    if not isinstance(caches[0], SKVQKVCache):
        return {"quantized_region": None, "end_to_end": None}
    fk = sum(c.fp16_key_bytes for c in caches)
    fv = sum(c.fp16_value_bytes for c in caches)
    ck = sum(c.compressed_key_bytes for c in caches)
    cv = sum(c.compressed_value_bytes for c in caches)
    extra = sum(c.residual_fp16_bytes + c.perm_bytes for c in caches)
    return {"quantized_region": (fk + fv) / (ck + cv), "end_to_end": (fk + fv) / (ck + cv + extra)}


def block_se(d: np.ndarray) -> float:
    """Standard error of the mean over ``BLOCK``-token blocks."""
    m = len(d) // BLOCK
    blocks = d[: m * BLOCK].reshape(m, BLOCK).mean(axis=1)
    return float(blocks.std(ddof=1) / np.sqrt(m))


def summarize(d: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(d.mean()),
        "se_naive": float(d.std(ddof=1) / np.sqrt(len(d))),
        "se_block128": block_se(d),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--out", required=True, help="results JSON to write")
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[512, 2047, 4095])
    ap.add_argument("--chunk", type=int, default=2048, help="model prefill chunk size")
    ap.add_argument("--n-cont", type=int, default=32, help="teacher-forced decode tokens")
    ap.add_argument("--rows-cache", type=Path, default=None, help="local cache of the rows")
    args = ap.parse_args()

    model, tok = load(args.model)
    ids = tok.encode("\n\n".join(fetch_rows(args.rows_cache)))
    need = max(args.prompt_lens) + args.n_cont
    if len(ids) < need:
        raise SystemExit(f"only {len(ids)} tokens available, need {need}")

    new_perms = SKVQKVCache.__dict__["_per_head_perms"]
    skvq_configs = []  # (label, config kwargs, uses old permutation policy)
    for bk, bv in BIT_WIDTHS:
        kw = {"skvq_bits_key": bk, "skvq_bits_value": bv}
        skvq_configs += [
            (f"K{bk}/V{bv} old(sort-all)", kw, True),
            (f"K{bk}/V{bv} reorder-off", {**kw, "skvq_reorder": False}, False),
            (f"K{bk}/V{bv} NEW", kw, False),
        ]

    def run(method: str, cfg: dict, n: int, old: bool = False):
        SKVQKVCache._per_head_perms = _old_unconditional_perms if old else new_perms
        caches = build_caches(model, method, **cfg)
        out = teacher_forced(model, caches, ids, n, args.chunk, args.n_cont)
        return (*out, caches)

    results: dict = {}
    try:
        for n in args.prompt_lens:
            n_pre = n - 1
            lp_ref, am_ref, _ = run("fp16", {}, n)
            entry = {"fp16_ppl_prefill": float(np.exp(-lp_ref[:n_pre].mean())), "configs": {}}
            nll: dict[str, np.ndarray] = {}
            runs = [(lab, "skvq", cfg, old) for lab, cfg, old in skvq_configs]
            runs += [(f"kivi-{b}", "kivi", {"bit_width_inlier": b}, False) for b in (2, 4)]
            for label, method, cfg, old in runs:
                lp, am, caches = run(method, cfg, n, old)
                d = (-lp) - (-lp_ref)
                nll[label] = -lp
                ratios = accounted_ratios(caches)
                entry["configs"][label] = {
                    "ppl_prefill": float(np.exp(-lp[:n_pre].mean())),
                    "ppl_decode": float(np.exp(-lp[n_pre:].mean())),
                    "dnll_vs_fp16": float(d[:n_pre].mean()),
                    "se_naive": float(d[:n_pre].std(ddof=1) / np.sqrt(n_pre)),
                    "se_block128": block_se(d[:n_pre]),
                    "dnll_decode": float(d[n_pre:].mean()),
                    "top1_agree_prefill": float((am[:n_pre] == am_ref[:n_pre]).mean()),
                    "ratio_full_quantized_region": ratios["quantized_region"],
                    "ratio_full_end_to_end": ratios["end_to_end"],
                    "finite": bool(np.isfinite(lp).all()),
                }
            entry["paired_new_vs_variants"] = {
                f"K{bk}/V{bv} NEW minus {other}": summarize(
                    nll[f"K{bk}/V{bv} NEW"][:n_pre] - nll[f"K{bk}/V{bv} {other}"][:n_pre]
                )
                for bk, bv in BIT_WIDTHS
                for other in ("old(sort-all)", "reorder-off")
            }
            results[str(n)] = entry
            print(f"n={n}: done", flush=True)

        kw = {"skvq_bits_key": 4, "skvq_bits_value": 4}
        first = run("skvq", kw, 2047)[0]
        results["deterministic_repeat_bitidentical"] = bool(
            np.array_equal(first, run("skvq", kw, 2047)[0])
        )
    finally:
        SKVQKVCache._per_head_perms = new_perms

    payload = {
        "model": args.model,
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test, first 2400 rows",
        "chunk": args.chunk,
        "n_cont": args.n_cont,
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
