"""Minimal retrieval-augmented generation (RAG) support for benchmarking.

This subpackage is deliberately small: a fixed local corpus
(:mod:`veloxquant_mlx.rag.corpus`) and a dependency-free TF-IDF retriever
(:mod:`veloxquant_mlx.rag.retriever`). It exists to produce realistic
"retrieved context + question" prompts for
``benchmark_scripts/benchmark_rag_kivi.py``, which compares a standard fp16
KV cache against :class:`~veloxquant_mlx.cache.kivi_cache.KIVIKVCache` on a
RAG workload. It is not a production retrieval system.
"""

from veloxquant_mlx.rag.corpus import CORPUS
from veloxquant_mlx.rag.eval_set import EVAL_SET
from veloxquant_mlx.rag.retriever import TfidfRetriever
from veloxquant_mlx.rag.scoring import keyword_overlap_score

__all__ = ["CORPUS", "EVAL_SET", "TfidfRetriever", "keyword_overlap_score"]
