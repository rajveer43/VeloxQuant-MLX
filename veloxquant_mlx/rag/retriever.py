"""A minimal, dependency-free TF-IDF retriever for the RAG benchmark.

This is intentionally simple: it exists to produce a realistic "retrieved
context + question" prompt for exercising the KV cache under a RAG-style
workload, not to be a production retrieval system. No embedding model,
vector database, or new third-party dependency is used — cosine similarity
over TF-IDF vectors built with plain Python/``numpy`` (already a core
dependency of this repo) is sufficient for a ~20-50 passage corpus.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class TfidfRetriever:
    """TF-IDF + cosine-similarity retriever over a fixed list of passages.

    Args:
        passages: The documents to index (plain strings).
    """

    passages: list[str]
    _vocab: dict[str, int] = field(init=False, default_factory=dict)
    _idf: np.ndarray = field(init=False, default=None)
    _doc_vectors: np.ndarray = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.passages:
            raise ValueError("TfidfRetriever requires at least one passage")

        doc_tokens = [_tokenize(p) for p in self.passages]

        vocab: dict[str, int] = {}
        for tokens in doc_tokens:
            for tok in set(tokens):
                vocab.setdefault(tok, len(vocab))
        self._vocab = vocab

        n_docs = len(self.passages)
        n_terms = len(vocab)
        doc_freq = np.zeros(n_terms, dtype=np.float64)
        tf = np.zeros((n_docs, n_terms), dtype=np.float64)

        for i, tokens in enumerate(doc_tokens):
            counts = Counter(tokens)
            total = max(len(tokens), 1)
            for tok, c in counts.items():
                j = vocab[tok]
                tf[i, j] = c / total
                doc_freq[j] += 1

        # Smoothed IDF (add-one in numerator and denominator) so terms that
        # appear in every document don't collapse to zero weight.
        self._idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0

        vectors = tf * self._idf[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._doc_vectors = vectors / norms

    def _vectorize_query(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        if not tokens:
            return vec
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        for tok, c in counts.items():
            j = self._vocab.get(tok)
            if j is not None:
                vec[j] = (c / total) * self._idf[j]
        norm = math.sqrt(float(np.dot(vec, vec)))
        if norm > 0.0:
            vec = vec / norm
        return vec

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Return the top-``k`` passages by cosine similarity to ``query``.

        Ties and an empty/out-of-vocabulary query fall back to corpus order
        so the result is always deterministic and always has length
        ``min(k, len(passages))``.
        """
        if k <= 0:
            return []
        q = self._vectorize_query(query)
        scores = self._doc_vectors @ q
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.passages))]
        return [self.passages[i] for i in order]

    def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Same as :meth:`retrieve` but also returns the cosine similarity."""
        if k <= 0:
            return []
        q = self._vectorize_query(query)
        scores = self._doc_vectors @ q
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.passages))]
        return [(self.passages[i], float(scores[i])) for i in order]
