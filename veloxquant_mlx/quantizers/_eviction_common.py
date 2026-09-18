"""Shared eviction-scoring/bookkeeping helpers used by multiple quantizers.

``_attention_scores`` / ``*_get_kv`` / ``*_fp16_bytes`` / ``full_*_fp16_bytes``
were independently redefined, byte-for-byte identically, in nine
eviction-based quantizer modules (h2o, tova, chunkkv, squeeze, cam, kvzip,
keyformer, pyramidkv, morphkv). This module centralizes the state-free parts
of that boilerplate; each module's stateful update loop (which legitimately
differs per method) is untouched. See issue #421.
"""

from __future__ import annotations

import math

import mlx.core as mx


def attention_scores(query_proxy: mx.array, keys: mx.array) -> mx.array:
    """Softmax attention weights of query_proxy against each key row.

    Args:
        query_proxy: [D] — used as a stand-in for the true query.
        keys:        [n, D] — existing key rows.

    Returns:
        [n] softmax weights summing to ~1.
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy) * scale  # [n]
    return mx.softmax(logits, axis=-1)


def get_kv(keys: mx.array | None, values: mx.array | None) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays, or ``([0, 1], [0, 1])`` placeholders.

    Args:
        keys: Stored key rows, or None before the first update.
        values: Stored value rows, or None before the first update.

    Returns:
        ``(keys, values)`` as given, or a pair of zero-row placeholders when
        ``keys`` is None.
    """
    if keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return keys, values


def fp16_kv_bytes(keys: mx.array | None) -> int:
    """Bytes currently stored for K + V in fp16, given the stored keys.

    Args:
        keys: Stored key rows ``[n, D]``, or None before the first update.

    Returns:
        ``n * D * 2 * 2`` (K and V, 2 bytes/element each), or 0 if empty.
    """
    if keys is None:
        return 0
    n, d = keys.shape
    return n * d * 2 * 2


def full_fp16_kv_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2


__all__ = [
    "attention_scores",
    "get_kv",
    "fp16_kv_bytes",
    "full_fp16_kv_bytes",
]
