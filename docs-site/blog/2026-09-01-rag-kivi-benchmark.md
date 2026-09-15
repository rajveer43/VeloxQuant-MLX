---
title: "RAG vs KIVI: what a real retrieval workload shows about KV-cache compression on Apple Silicon"
tags: [kivi, rag, benchmark]
---

> **TL;DR** — We built a small, self-contained RAG benchmark (`benchmark_scripts/benchmark_rag_kivi.py`) that runs the same retrieval + generation workload twice — once with a standard fp16 `mlx_lm` KV cache, once with `KIVIKVCache` — and measures memory, latency, throughput, and answer quality side by side. On a 1B-parameter model with short (~250-token) retrieved contexts, KIVI's *KV-cache byte accounting* showed real compression (3.85× key, 2.33× full-KV at 4-bit), but **total process peak memory barely moved**, because the KV cache is a small fraction of total memory at this model size and context length. This is the expected, honest result at this scale — not a bug — and it's a useful caution against assuming KV-cache compression numbers translate directly into end-to-end memory wins for every model/workload.

## Why a RAG benchmark, specifically

Every other KIVI benchmark in this repo (`benchmark_kivi.py`, `benchmark_kivi_multi_model.py`) uses one long synthetic prompt built by repeating a passage. That's the right way to isolate KIVI's behavior on long contexts, but it doesn't reflect how most people actually hit long-ish contexts in practice: retrieval-augmented generation, where a retriever pulls back a handful of passages and concatenates them into the prompt ahead of the question.

We wanted to know: on a realistic (if small) RAG pipeline, what do you actually see — in memory, latency, throughput, *and* answer quality — from swapping a standard KV cache for KIVI?

## What we built

- `veloxquant_mlx/rag/` — a fixed 20-passage corpus spanning KV-cache, RAG, MLX, and quantization topics, and a dependency-free TF-IDF + cosine-similarity retriever (`TfidfRetriever`). No embedding model, no vector database, no new dependency — this is deliberately the simplest thing that produces a real "retrieve top-k, concatenate, generate" prompt.
- `veloxquant_mlx/rag/eval_set.py` — 10 grounded questions, each answerable from the corpus, with gold keywords for scoring.
- `veloxquant_mlx/rag/scoring.py` — a keyword-overlap quality score (fraction of gold keywords present in the generated answer). No LLM-judge, no new dependency (`rouge-score` isn't installed in this environment) — simple and auditable.
- `benchmark_scripts/benchmark_rag_kivi.py` — for each eval question: retrieve top-k passages, build a RAG prompt, generate once with an fp16 cache and once with `KIVIKVCache`, and record peak memory (`mx.get_peak_memory()`), wall-clock latency, throughput, quality score, and KIVI's realized compressed-byte accounting. Mirrors `benchmark_kivi.py`'s cache-construction pattern and its rule that the fp16 baseline is always timed for real, never assumed.

## What we measured

Run on an Apple M4 (24GB), `mlx-community/Llama-3.2-1B-Instruct-4bit`, `k=4` retrieved passages (~250 prompt tokens), `max_tokens=60`, KIVI at 4-bit / group size 32 / residual length 32, averaged over the 10-question eval set:

| config | quality | peak MB | latency (s) | tok/s | key compression | full-KV compression |
|---|---|---|---|---|---|---|
| fp16-baseline | 0.68 | 921.1 | 0.74 | 81.4 | 1.00× | 1.00× |
| KIVI-4bit | 0.70 | 940.5 | 0.64 | 75.6 | 3.85× | 2.33× |

(Full per-question results, including generated answers, are in the committed `rag_kivi_benchmark_results.json`.)

### The KV cache compresses — but total memory doesn't drop

KIVI's own byte accounting shows real, expected compression: keys are ~3.85× smaller and the full KV region (including the fp16 residual window) is ~2.33× smaller. But **peak process memory was slightly higher under KIVI, not lower**. At this model size (1B params, ~700MB of 4-bit weights) and this context length (~250-300 tokens total), the KV cache itself is a small slice of total memory — model weights and activation buffers dominate. KIVI's fixed per-group overhead (a scale and zero-point per group of 32 elements) is proportionally more expensive to a KV cache this small, and the savings on such a small quantized region don't move the needle against the rest of the process's memory footprint.

This matches expectations set by the KIVI cache implementation's own documentation: the paper's benchmarks (and this repo's other KIVI benchmarks) use much larger models and much longer prefill lengths specifically because that's the regime where the KV cache is a meaningful fraction of total memory. A short RAG context on a 1B model is close to a worst case for showing KIVI's memory benefit end-to-end, even though the underlying compression math is working correctly.

### No throughput win, as expected on Metal

KIVI showed a small latency reduction (0.74s → 0.64s) in this run, but that's within the noise of a ~60-token generation on a tiny model — not a claim that KIVI is reliably faster here. As documented elsewhere in this repo, KIVI's reference implementation gets its throughput advantage from a fused CUDA kernel with no direct Metal equivalent; on Apple Silicon we don't expect — and don't claim — a reliable speedup.

### Answer quality was roughly comparable at 4-bit

Quality (keyword-overlap against gold answers) was statistically indistinguishable between fp16 and KIVI-4bit on this small eval set (0.68 vs 0.70, 10 questions — well within noise for a 10-item set). We did *not* see a meaningful quality regression from 4-bit KIVI on this RAG workload. (We also ran 2-bit, which did show a visible quality drop on this small model — consistent with 2-bit being a more aggressive setting that a 1B model handles less gracefully than a larger one; see the script's `--bits` flag to reproduce.)

## Takeaways

- KIVI's compressed-byte accounting is trustworthy and matches its design (per-channel keys, per-token values, fp16 residual) — that part of the story holds up on a real RAG workload, not just a synthetic long prompt.
- **KV-cache compression ratio is not the same as end-to-end memory savings.** Whether KIVI reduces *total* process memory depends on how large the KV cache is relative to model weights and activations — small model + short context is the regime where you'll see the least benefit. Larger models and/or longer retrieved contexts (more passages, longer passages, larger `k`) are where KIVI's memory case gets stronger.
- As with every other KIVI benchmark in this repo: no Metal throughput win should be expected or claimed.
- This benchmark and its corpus/eval-set are intentionally small and reproducible on a laptop. Rerun with `--k`, `--bits`, `--group-size`, `--residual-length`, and a larger `--model` to explore where the memory story changes.

```bash
PYTHONPATH=. python benchmark_scripts/benchmark_rag_kivi.py \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --k 4 --max-tokens 60 --bits 4
```
