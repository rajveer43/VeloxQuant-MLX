"""Order-0 static Huffman entropy coder — the entropy-coding stage of KVTC-adapted.

Inspired by the entropy-coding stage of "KV Cache Transform Coding for
Compact Storage in LLM Inference" (NVIDIA, ICLR 2026, arXiv:2511.01815).
Used by ``quantizers/kvtc.py``. Documented as part of "KVTC-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

WHAT THIS IS — AND IS NOT
--------------------------
This is a small, dependency-free, **order-0, static, per-call** Huffman
coder built on the Python stdlib's ``heapq`` — no external compression
library. "Order-0" means the code table is built once from a single global
frequency count over the input symbols (no context modeling, no adaptivity
across positions). "Static, per-call" means a fresh table is built and
shipped alongside the payload for every call to :func:`entropy_encode` — it
is not adaptive across calls.

The paper's entropy-coding stage may use a more sophisticated scheme
(e.g. adaptive/context-modeled arithmetic coding); we do not claim to
match it. What we DO claim, and measure, is a **real, lossless, order-0
Huffman round-trip** with its **realized** encoded byte count — including
the code table's own storage cost, which is folded into
``quantizers/kvtc.py``'s byte accounting (never hidden). We never report
the theoretical Shannon-entropy lower bound as if it were the achieved
size — only the actual encoded bitstream length (rounded up to a whole
number of bytes) plus the table.

Public API
----------
entropy_encode(codes)              -> (payload_bytes, table)
entropy_decode(payload, table, n)  -> codes  (exact inverse)
table_nbytes(table)                -> int, the code table's own storage cost
"""

from __future__ import annotations

import heapq
import math
from itertools import count as _count

import numpy as np

__all__ = ["entropy_encode", "entropy_decode", "table_nbytes"]


def _build_huffman_codes(freqs: dict[int, int]) -> dict[int, str]:
    """Build a canonical-ish Huffman code (bit strings) from a frequency table.

    Degenerate single-symbol input gets a 1-bit code ("0") so encoding is
    always well-defined even when there is nothing to compress.
    """
    symbols = list(freqs.keys())
    if len(symbols) == 1:
        return {symbols[0]: "0"}

    # heapq needs a tiebreaker for entries with equal frequency (dicts and
    # tuples of node-lists are not orderable); a monotonically increasing
    # counter breaks ties deterministically without comparing payloads.
    tiebreak = _count()
    heap = [(freq, next(tiebreak), [[sym, ""]]) for sym, freq in freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        f1, _, pairs1 = heapq.heappop(heap)
        f2, _, pairs2 = heapq.heappop(heap)
        for pair in pairs1:
            pair[1] = "0" + pair[1]
        for pair in pairs2:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, (f1 + f2, next(tiebreak), pairs1 + pairs2))

    _, _, pairs = heap[0]
    return {sym: code for sym, code in pairs}


def entropy_encode(codes: np.ndarray) -> tuple[bytes, dict]:
    """Order-0 Huffman-encode an integer array.

    Args:
        codes: 1-D (or any-shape, flattened internally) array of
            non-negative integer symbols.

    Returns:
        ``(payload, table)`` where ``payload`` is the packed bitstream as
        ``bytes`` (padded to a whole byte with trailing zero bits — the
        padding amount is implicit since the caller always knows ``n``, the
        original element count, and decodes exactly ``n`` symbols) and
        ``table`` is a plain ``dict[int, str]`` mapping symbol -> Huffman
        code (bit string). The table's own size is real storage cost — see
        :func:`table_nbytes` — and must be included by callers in any
        reported byte accounting.
    """
    flat = np.asarray(codes).reshape(-1)
    n = int(flat.shape[0])
    if n == 0:
        return b"", {}

    ints = [int(x) for x in flat.tolist()]
    freqs: dict[int, int] = {}
    for x in ints:
        freqs[x] = freqs.get(x, 0) + 1

    table = _build_huffman_codes(freqs)

    bitstring = "".join(table[x] for x in ints)
    # Pad to a whole byte with zero bits; decode reads exactly n symbols so
    # trailing pad bits are never misinterpreted as a spurious extra symbol.
    pad = (-len(bitstring)) % 8
    bitstring += "0" * pad

    payload = int(bitstring, 2).to_bytes(len(bitstring) // 8, byteorder="big") if bitstring else b""
    return payload, table


def entropy_decode(payload: bytes, table: dict, n: int) -> np.ndarray:
    """Exact inverse of :func:`entropy_encode`.

    Args:
        payload: Packed bitstream from :func:`entropy_encode`.
        table: The code table returned alongside ``payload``.
        n: Number of symbols to decode (the original array length) — needed
            because the packed bitstream is padded to a byte boundary and
            carries no explicit length marker.

    Returns:
        1-D ``np.ndarray[int64]`` of length ``n``, bit-for-bit identical to
        the original input to :func:`entropy_encode`.
    """
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    if len(table) == 1:
        (sym,) = table.keys()
        return np.full((n,), sym, dtype=np.int64)

    inv = {code: sym for sym, code in table.items()}
    n_bits = len(payload) * 8
    bitstring = bin(int.from_bytes(payload, byteorder="big"))[2:].zfill(n_bits) if payload else ""

    out = np.empty((n,), dtype=np.int64)
    cur = ""
    i = 0
    pos = 0
    while i < n and pos < len(bitstring):
        cur += bitstring[pos]
        pos += 1
        if cur in inv:
            out[i] = inv[cur]
            i += 1
            cur = ""
    if i != n:
        raise ValueError(
            f"entropy_decode: expected {n} symbols, decoded {i} — payload/table/n mismatch."
        )
    return out


def table_nbytes(table: dict) -> int:
    """Real storage cost of a code table: a symbol (int32) + code-length
    (uint8) pair per entry, a conservative but honest fixed-width accounting
    (we do not try to further compress the table itself).
    """
    return len(table) * (4 + 1)
