"""End-to-end RAG benchmark: fp16 KV cache vs KIVI on Apple Silicon (MLX).

For each question in a small grounded QA eval set, retrieves the top-k
passages from a fixed local corpus (TF-IDF, no embedding model, no new
dependency — see ``veloxquant_mlx/rag/``), builds a RAG prompt from the
retrieved context + question, and generates an answer twice: once with a
standard fp16 ``mlx_lm`` KV cache, once with
:class:`~veloxquant_mlx.cache.kivi_cache.KIVIKVCache`. Records peak memory,
wall-clock latency, throughput, answer quality (keyword-overlap against
gold keywords), and KIVI's realized KV-cache compression, then writes
``rag_kivi_benchmark_results.json`` plus a stdout summary table with %
deltas between the two configs.

Mirrors the structure and honesty rules of ``benchmark_kivi.py``: the fp16
baseline is *always* timed for real (never hardcoded), and KIVI is not
expected to be faster on Metal — only smaller. This script does not touch
KIVI's cache/quantizer/kernel internals; it only consumes the existing
``KIVIKVCache`` through the same construction pattern as
``benchmark_kivi.py``.

Caveat for small models / short RAG contexts: KIVI's per-group scale and
zero-point overhead is fixed cost that only pays off once the quantized
region is large relative to that overhead, and 2-bit quantization is more
lossy on a small (~1B parameter) model than a large one. With a short
corpus (k passages of a couple hundred tokens total) and an aggressive
``--bits 2``, expect KIVI's memory savings to be small or even negative and
its answer quality to drop noticeably — this is a real property of the
method at this scale, not a bug in the harness. Use a longer-context corpus
and/or a larger model and/or ``--bits 4`` to see KIVI's intended memory
regime.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_rag_kivi.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _peak_mb() -> float:
    import mlx.core as mx

    try:
        get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        return float(get_peak()) / (1024**2)
    except Exception:
        return float("nan")


def _reset_peak() -> None:
    import mlx.core as mx

    try:
        reset_peak = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
        reset_peak()
    except Exception:
        pass


def _hardware() -> dict:
    """Best-effort hardware record (chip + RAM) for honest provenance."""
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        import subprocess

        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if chip:
            info["chip"] = chip
        if mem:
            info["ram_gb"] = round(int(mem) / (1024**3), 1)
    except Exception:
        pass
    return info


def _build_rag_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using only the information in the context "
        "passages below. Be concise.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def _build_kivi_caches(model, b: int, group_size: int, residual_length: int) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.kivi_cache import KIVIKVCache

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            caches.append(_FallbackCache())
            continue
        hd = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        if hd is None:
            caches.append(_FallbackCache())
            continue
        cfg = KVCacheConfig(
            method="kivi",
            head_dim=hd,
            bit_width_inlier=b,
            kivi_group_size=group_size,
            residual_length=residual_length,
            seed=42 + i,
        )
        caches.append(KIVIKVCache(cfg))
    return caches


def _build_fp16_caches(model) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    layers = getattr(model, "layers", None) or model.model.layers
    return [_FallbackCache() for _ in layers]


def _generate(model, tokenizer, prompt: str, max_tokens: int, caches: list) -> tuple:
    from mlx_lm import generate

    t0 = time.time()
    try:
        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=caches,
        )
    except TypeError:
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    elapsed = time.time() - t0
    n_tok = len(tokenizer.encode(out)) if out else 0
    return out, n_tok, elapsed


def _kv_compression_stats(caches: list) -> dict:
    key_compressed = key_fp16 = 0
    val_compressed = val_fp16 = residual_fp16 = 0
    avg_bits = 16.0
    for c in caches:
        if hasattr(c, "compressed_key_bytes"):
            key_compressed += c.compressed_key_bytes
            key_fp16 += c.fp16_key_bytes
            val_compressed += getattr(c, "compressed_value_bytes", 0)
            val_fp16 += getattr(c, "fp16_value_bytes", 0)
            residual_fp16 += getattr(c, "residual_fp16_bytes", 0)
    if caches and hasattr(caches[0], "assigned_avg_bits"):
        avg_bits = float(caches[0].assigned_avg_bits)

    key_ratio = (key_fp16 / key_compressed) if key_compressed else 1.0
    total_fp16 = key_fp16 + val_fp16
    total_comp = key_compressed + val_compressed + residual_fp16
    full_kv_ratio = (total_fp16 / total_comp) if total_comp else 1.0
    return {
        "avg_bits": avg_bits,
        "key_compression": key_ratio,
        "full_kv_compression": full_kv_ratio,
    }


def _run_eval_set(
    model,
    tokenizer,
    name: str,
    build_caches_fn,
    retriever,
    eval_set: list,
    k: int,
    max_tokens: int,
) -> dict:
    from veloxquant_mlx.rag.scoring import keyword_overlap_score

    print(f"\n--- {name} ---", flush=True)
    per_question = []
    for item in eval_set:
        question = item["question"]
        contexts = retriever.retrieve(question, k=k)
        prompt = _build_rag_prompt(question, contexts)

        _reset_peak()
        caches = build_caches_fn()
        answer, n_tok, elapsed = _generate(model, tokenizer, prompt, max_tokens, caches)
        peak_mb = _peak_mb()
        throughput = n_tok / max(elapsed, 1e-6)
        quality = keyword_overlap_score(answer, item["gold_keywords"])
        compression = _kv_compression_stats(caches)

        print(
            f"  [{quality:.2f} quality] {n_tok} tok in {elapsed:.2f}s "
            f"({throughput:.1f} tok/s) peak={peak_mb:.0f}MB :: {question[:60]}",
            flush=True,
        )

        per_question.append(
            {
                "question": question,
                "answer": answer,
                "gold_keywords": item["gold_keywords"],
                "quality_score": quality,
                "peak_mb": peak_mb,
                "elapsed_s": elapsed,
                "throughput_tok_s": throughput,
                "tokens_generated": n_tok,
                "num_retrieved": len(contexts),
                **compression,
            }
        )

    n = len(per_question)
    avg = lambda key: sum(q[key] for q in per_question) / n if n else float("nan")
    return {
        "name": name,
        "avg_quality_score": avg("quality_score"),
        "avg_peak_mb": avg("peak_mb"),
        "avg_elapsed_s": avg("elapsed_s"),
        "avg_throughput_tok_s": avg("throughput_tok_s"),
        "avg_key_compression": avg("key_compression"),
        "avg_full_kv_compression": avg("full_kv_compression"),
        "avg_bits": avg("avg_bits"),
        "per_question": per_question,
    }


def _print_summary_table(results: list) -> None:
    header = f"{'config':<16s} {'quality':>8s} {'peak MB':>10s} {'latency s':>10s} {'tok/s':>8s}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<16s} {r['avg_quality_score']:8.2f} {r['avg_peak_mb']:10.1f} "
            f"{r['avg_elapsed_s']:10.2f} {r['avg_throughput_tok_s']:8.1f}"
        )

    if len(results) >= 2:
        base, kivi = results[0], results[1]

        def pct(a: float, b: float) -> float:
            return ((b - a) / a * 100.0) if a else float("nan")

        print("\nKIVI vs fp16-baseline (% change; negative = smaller/slower is expected for memory/latency):")
        print(f"  quality:    {pct(base['avg_quality_score'], kivi['avg_quality_score']):+.1f}%")
        print(f"  peak_mb:    {pct(base['avg_peak_mb'], kivi['avg_peak_mb']):+.1f}%")
        print(f"  latency_s:  {pct(base['avg_elapsed_s'], kivi['avg_elapsed_s']):+.1f}%")
        print(f"  throughput: {pct(base['avg_throughput_tok_s'], kivi['avg_throughput_tok_s']):+.1f}%")


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="RAG benchmark: fp16 vs KIVI KV cache")
    parser.add_argument("--model", required=True, help="HF model id (mlx-community/...)")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved passages")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--bits", type=int, default=2, help="KIVI bit-width")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument("--output", default=None, help="Path to write results JSON")
    args = parser.parse_args()

    from mlx_lm import load

    from veloxquant_mlx.rag.corpus import CORPUS
    from veloxquant_mlx.rag.eval_set import EVAL_SET
    from veloxquant_mlx.rag.retriever import TfidfRetriever

    retriever = TfidfRetriever([p["text"] for p in CORPUS])

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer = load(args.model)

    hw = _hardware()
    print(f"  hardware={hw}")
    print(f"  corpus={len(CORPUS)} passages, eval_set={len(EVAL_SET)} questions, k={args.k}")

    results = [
        _run_eval_set(
            model,
            tokenizer,
            "fp16-baseline",
            lambda: _build_fp16_caches(model),
            retriever,
            EVAL_SET,
            args.k,
            args.max_tokens,
        ),
        _run_eval_set(
            model,
            tokenizer,
            f"KIVI-{args.bits}bit",
            lambda: _build_kivi_caches(model, args.bits, args.group_size, args.residual_length),
            retriever,
            EVAL_SET,
            args.k,
            args.max_tokens,
        ),
    ]

    _print_summary_table(results)

    payload = {
        "model": args.model,
        "k": args.k,
        "max_tokens": args.max_tokens,
        "bits": args.bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "corpus_size": len(CORPUS),
        "eval_set_size": len(EVAL_SET),
        "hardware": hw,
        "results": results,
    }
    out_path = Path(args.output) if args.output else Path("rag_kivi_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
