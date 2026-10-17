"""Small grounded QA eval set for the RAG-vs-KIVI benchmark.

Each question is answerable from one or more passages in
:data:`veloxquant_mlx.rag.corpus.CORPUS`. ``gold_keywords`` lists the
distinct facts/terms a correct, grounded answer should mention; scoring is a
simple case-insensitive keyword-overlap fraction (see
:mod:`veloxquant_mlx.rag.scoring`), not exact string match, since free-form
generations rarely match a reference string verbatim.
"""

from __future__ import annotations

EVAL_SET: list[dict] = [
    {
        "question": "What does a key-value cache store, and why does its size grow with sequence length?",
        "gold_keywords": ["keys", "values", "attention", "previously generated", "token"],
    },
    {
        "question": "Why is KV cache size a binding constraint for long-context inference on Apple Silicon?",
        "gold_keywords": ["unified memory", "shared", "operating system"],
    },
    {
        "question": "How does KIVI quantize keys versus values?",
        "gold_keywords": ["per channel", "per token", "residual"],
    },
    {
        "question": "Why doesn't KIVI's memory saving translate into a Metal throughput speedup?",
        "gold_keywords": ["cuda kernel", "metal", "no direct", "equivalent"],
    },
    {
        "question": "What are the two stages of a typical RAG pipeline?",
        "gold_keywords": ["retriever", "rank", "language model", "generate"],
    },
    {
        "question": "How does retrieving more passages in RAG affect the KV cache?",
        "gold_keywords": ["context length", "grows", "kv cache"],
    },
    {
        "question": "How does TF-IDF score how relevant a document is to a query?",
        "gold_keywords": ["term", "frequency", "rare", "distinctive"],
    },
    {
        "question": "What is the residual window in a quantized KV cache?",
        "gold_keywords": ["recently generated", "full precision", "not", "quantized"],
    },
    {
        "question": "What lets MLX share arrays between the CPU and GPU without copying?",
        "gold_keywords": ["unified memory", "lazy evaluation"],
    },
    {
        "question": "How can you measure and reset peak memory usage in an MLX program?",
        "gold_keywords": ["get_peak_memory", "reset_peak_memory"],
    },
]
