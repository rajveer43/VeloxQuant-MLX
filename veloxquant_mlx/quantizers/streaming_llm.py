"""StreamingLLM-adapted quantizer primitives — sink + recency-window token eviction.

Inspired by "Efficient Streaming Language Models with Attention Sinks"
(Xiao et al., ICLR 2024, arXiv:2309.17453). Documented as "StreamingLLM-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

What StreamingLLM adds: **structural positional eviction** — tokens are kept or
dropped purely by their position (first N sink positions + last W recent positions),
with no attention scoring, no calibration, and no model weights. This is orthogonal
to SnapKV-adapted (score-based eviction) and all quantization methods.

The paper's key insight: attention sinks (the first few tokens of any sequence)
consistently receive high attention weight regardless of content, and most other
attention is locally recency-biased. Keeping N_sink + W_recent tokens preserves
both effects while bounding the cache to a constant size during streaming decode.

Adaptation decisions (documented, never hidden):
  1. **Cache-level implementation.** The paper patches the model's forward pass
     (attention mask + KV cache together). We implement equivalent semantics inside
     ``update_and_fetch`` by maintaining a frozen sink buffer and a FIFO recent window,
     concatenating them per call. Functionally equivalent within the cache-wrapper level.
  2. **No attention mask adjustment.** The paper adjusts position masks so tokens beyond
     the window are invisible to the query. A cache wrapper cannot inject attention masks;
     all returned K/V positions will be attended to by the model. Documented as a known
     limitation — the memory budget is still bounded.
  3. **No position-ID remapping.** We drop tokens and preserve original positions inside
     the returned K/V rows. The paper remaps RoPE position IDs; that requires model-level
     patching. Documented plainly.
  4. **Fixed sink count.** ``stream_n_sink`` is a fixed hyperparameter, not adaptive.

The cache never exceeds ``stream_n_sink + stream_window_size`` token positions, making
it constant-memory for arbitrarily long generation once the window fills.

This module holds the pure, side-effect-free numerics: concatenation, window trimming,
and byte accounting.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx


class StreamingWindow(NamedTuple):
    """A single-head sink+recent window for StreamingLLM-adapted eviction.

    Attributes:
        sink_keys:    [n_sink, D] fp16 — frozen attention-sink key rows.
        sink_values:  [n_sink, D] fp16 — frozen attention-sink value rows.
        recent_keys:  [n_recent, D] fp16 — rolling recent-window key rows.
        recent_values:[n_recent, D] fp16 — rolling recent-window value rows.
        n_sink:       int — sink count (== sink_keys.shape[0]).
        n_recent:     int — current recent-window occupancy (<= window_size).
        tokens_seen:  int — total token positions seen since creation.
    """

    sink_keys: mx.array
    sink_values: mx.array
    recent_keys: mx.array
    recent_values: mx.array
    n_sink: int
    n_recent: int
    tokens_seen: int


def init_streaming_window(n_sink: int, D: int) -> StreamingWindow:
    """Create an empty streaming window with the given sink count and head dimension."""
    empty = mx.zeros((0, D), dtype=mx.float16)
    return StreamingWindow(
        sink_keys=empty,
        sink_values=empty,
        recent_keys=empty,
        recent_values=empty,
        n_sink=0,
        n_recent=0,
        tokens_seen=0,
    )


def stream_update(
    window: StreamingWindow,
    new_keys: mx.array,
    new_values: mx.array,
    n_sink: int,
    window_size: int,
) -> StreamingWindow:
    """Absorb ``[S, D]`` new K/V tokens into the streaming window.

    The first ``n_sink`` positions ever seen are unconditionally frozen as sinks.
    Once frozen, sinks are never evicted. All subsequent tokens occupy the recent
    window, a FIFO buffer of capacity ``window_size``. When the FIFO exceeds
    ``window_size``, the oldest tokens are dropped from the front.

    Args:
        window: Current :class:`StreamingWindow` state.
        new_keys: ``[S, D]`` fp16/fp32 key rows to absorb.
        new_values: ``[S, D]`` fp16/fp32 value rows to absorb.
        n_sink: Number of initial tokens to freeze as sinks.
        window_size: FIFO capacity for recent tokens.

    Returns:
        Updated :class:`StreamingWindow`.
    """
    S = int(new_keys.shape[0])
    D = int(new_keys.shape[1])
    nk = new_keys.astype(mx.float16)
    nv = new_values.astype(mx.float16)

    tokens_seen_before = window.tokens_seen
    sink_k = window.sink_keys
    sink_v = window.sink_values
    recent_k = window.recent_keys
    recent_v = window.recent_values
    current_sinks = window.n_sink

    new_sink_k_list = []
    new_sink_v_list = []
    new_recent_k_list = []
    new_recent_v_list = []

    for i in range(S):
        token_idx = tokens_seen_before + i
        k_tok = nk[i : i + 1]  # [1, D]
        v_tok = nv[i : i + 1]

        if current_sinks < n_sink:
            # Still absorbing into sink buffer
            new_sink_k_list.append(k_tok)
            new_sink_v_list.append(v_tok)
            current_sinks += 1
        else:
            # Goes into recent window
            new_recent_k_list.append(k_tok)
            new_recent_v_list.append(v_tok)

    # Build updated sink buffer
    if new_sink_k_list:
        new_sink_part_k = mx.concatenate(new_sink_k_list, axis=0)
        new_sink_part_v = mx.concatenate(new_sink_v_list, axis=0)
        if int(sink_k.shape[0]) > 0:
            sink_k = mx.concatenate([sink_k, new_sink_part_k], axis=0)
            sink_v = mx.concatenate([sink_v, new_sink_part_v], axis=0)
        else:
            sink_k = new_sink_part_k
            sink_v = new_sink_part_v

    # Build updated recent window (concat existing + new recent, then trim from left)
    if new_recent_k_list:
        new_recent_part_k = mx.concatenate(new_recent_k_list, axis=0)
        new_recent_part_v = mx.concatenate(new_recent_v_list, axis=0)
        if int(recent_k.shape[0]) > 0:
            recent_k = mx.concatenate([recent_k, new_recent_part_k], axis=0)
            recent_v = mx.concatenate([recent_v, new_recent_part_v], axis=0)
        else:
            recent_k = new_recent_part_k
            recent_v = new_recent_part_v

    # Trim recent window to last window_size tokens
    if window_size > 0 and int(recent_k.shape[0]) > window_size:
        recent_k = recent_k[-window_size:]
        recent_v = recent_v[-window_size:]

    n_recent = int(recent_k.shape[0])

    return StreamingWindow(
        sink_keys=sink_k,
        sink_values=sink_v,
        recent_keys=recent_k,
        recent_values=recent_v,
        n_sink=current_sinks,
        n_recent=n_recent,
        tokens_seen=tokens_seen_before + S,
    )


def stream_get_kv(window: StreamingWindow) -> tuple[mx.array, mx.array]:
    """Return the concatenated ``[n_sink + n_recent, D]`` fp16 K and V tensors.

    If only sinks exist (no recent tokens yet), returns the sink arrays.
    If both exist, concatenates sink rows first, then recent rows.
    """
    n_s = int(window.sink_keys.shape[0])
    n_r = int(window.recent_keys.shape[0])

    if n_s == 0 and n_r == 0:
        D = 0  # degenerate empty — should not happen in practice
        return mx.zeros((0, 1), dtype=mx.float16), mx.zeros((0, 1), dtype=mx.float16)

    if n_s == 0:
        return window.recent_keys, window.recent_values
    if n_r == 0:
        return window.sink_keys, window.sink_values
    k = mx.concatenate([window.sink_keys, window.recent_keys], axis=0)
    v = mx.concatenate([window.sink_values, window.recent_values], axis=0)
    return k, v


def stream_fp16_bytes(window: StreamingWindow) -> int:
    """Bytes stored in the streaming window (fp16 K + V, sink + recent rows)."""
    n = window.n_sink + window.n_recent
    if n == 0:
        return 0
    D = int(window.sink_keys.shape[1]) if window.n_sink > 0 else int(window.recent_keys.shape[1])
    return n * D * 2 * 2  # K + V, fp16 = 2 bytes/element


def full_stream_fp16_bytes(n_tokens: int, D: int) -> int:
    """Bytes for uncompressed fp16 K + V at ``n_tokens`` positions, dim ``D``."""
    return n_tokens * D * 2 * 2


__all__ = [
    "StreamingWindow",
    "init_streaming_window",
    "stream_update",
    "stream_get_kv",
    "stream_fp16_bytes",
    "full_stream_fp16_bytes",
]
