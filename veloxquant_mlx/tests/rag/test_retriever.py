"""Tests for TfidfRetriever — fast, no model loading, no network."""

from __future__ import annotations

import pytest

from veloxquant_mlx.rag.corpus import CORPUS
from veloxquant_mlx.rag.retriever import TfidfRetriever
from veloxquant_mlx.rag.scoring import keyword_overlap_score

_FIXTURE = [
    "The quick brown fox jumps over the lazy dog in the sunny meadow.",
    "Apple Silicon uses a unified memory architecture shared by CPU and GPU.",
    "A key-value cache stores attention keys and values for past tokens.",
    "Quantization reduces the number of bits used to represent a value.",
    "The lazy dog slept all afternoon near the quiet meadow fence.",
]


def test_retrieve_returns_k_results() -> None:
    r = TfidfRetriever(_FIXTURE)
    out = r.retrieve("unified memory CPU GPU", k=2)
    assert len(out) == 2


def test_retrieve_ranks_relevant_passage_first() -> None:
    r = TfidfRetriever(_FIXTURE)
    out = r.retrieve("what does a key-value cache store", k=1)
    assert "key-value cache" in out[0]


def test_retrieve_k_larger_than_corpus_returns_all() -> None:
    r = TfidfRetriever(_FIXTURE)
    out = r.retrieve("quantization bits value", k=100)
    assert len(out) == len(_FIXTURE)


def test_retrieve_k_zero_returns_empty() -> None:
    r = TfidfRetriever(_FIXTURE)
    assert r.retrieve("anything", k=0) == []


def test_retrieve_empty_query_is_deterministic_and_full_length() -> None:
    r = TfidfRetriever(_FIXTURE)
    out = r.retrieve("", k=3)
    assert len(out) == 3


def test_retrieve_out_of_vocabulary_query_falls_back_to_corpus_order() -> None:
    r = TfidfRetriever(_FIXTURE)
    out = r.retrieve("zzz qqq nonexistentword", k=3)
    assert out == _FIXTURE[:3]


def test_retrieve_with_scores_sorted_descending() -> None:
    r = TfidfRetriever(_FIXTURE)
    scored = r.retrieve_with_scores("quantization bits value", k=5)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_retriever_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError):
        TfidfRetriever([])


def test_real_corpus_retrieves_relevant_topic() -> None:
    passages = [p["text"] for p in CORPUS]
    r = TfidfRetriever(passages)
    out = r.retrieve("How does KIVI quantize keys and values in the KV cache?", k=3)
    assert any("KIVI" in p for p in out)


def test_keyword_overlap_score_full_match() -> None:
    answer = "KIVI quantizes keys per channel and values per token, keeping a residual window."
    assert keyword_overlap_score(answer, ["per channel", "per token", "residual"]) == 1.0


def test_keyword_overlap_score_partial_match() -> None:
    answer = "KIVI quantizes keys per channel."
    score = keyword_overlap_score(answer, ["per channel", "per token", "residual"])
    assert score == pytest.approx(1 / 3)


def test_keyword_overlap_score_no_match() -> None:
    assert keyword_overlap_score("completely unrelated text", ["per channel", "residual"]) == 0.0


def test_keyword_overlap_score_empty_keywords_is_perfect() -> None:
    assert keyword_overlap_score("anything", []) == 1.0
