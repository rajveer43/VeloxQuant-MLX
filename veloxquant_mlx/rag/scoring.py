"""Dependency-free answer-quality scoring for the RAG benchmark.

Deliberately not an LLM-judge and not a metric requiring a new dependency
(e.g. ``rouge-score``, which is not installed in this repo's environment) —
a keyword-overlap fraction is enough to compare fp16 vs KIVI answer quality
on a small, fixed eval set where every question has known gold keywords.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def keyword_overlap_score(answer: str, gold_keywords: list[str]) -> float:
    """Fraction of ``gold_keywords`` phrases found in ``answer`` (case-insensitive).

    A keyword counts as present if all of its whitespace-separated words
    appear, in order, as a substring of the lowercased/normalized answer
    (so "per channel" matches "quantizes keys per channel" but not
    "channel per quant").
    """
    if not gold_keywords:
        return 1.0
    normalized = " ".join(_TOKEN_RE.findall(answer.lower()))
    hits = 0
    for phrase in gold_keywords:
        phrase_norm = " ".join(_TOKEN_RE.findall(phrase.lower()))
        if phrase_norm and phrase_norm in normalized:
            hits += 1
    return hits / len(gold_keywords)
