# Packed storage roadmap

## Problem

Several methods report large **key-byte accounting** ratios while the default
runtime path still materializes **fp16** tensors in the parent `mlx_lm`
`KVCache` (quantize → dequantize → store). Users then expect Activity Monitor
RSS to drop by 7.5× or 16× and do not see it at short context.

## Goal

Track, per method, whether compressed state is **actually retained** in
resident memory during decode, versus accounting-only counters.

## Status matrix (starting point)

| Method | Default stores packed? | Notes |
| --- | --- | --- |
| `turboquant_rvq` | **Yes** (since v0.44.0+, see below) | Keys stored as two bit-packed uint32 RVQ index streams; dequantized transiently on fetch. Measured (not accounting): -12.8% peak memory vs. fp16 baseline, -44.7% vs. mlx-lm's own native `QuantizedKVCache(bits=4)`, at a 4002-token prompt on Llama-3.2-1B-Instruct-4bit. Full methodology and every intermediate finding (including two false starts in the measurement itself) in `docs/RVQ_PACKED_STORAGE_FINDINGS.md`. `nbytes`/`compressed_key_bytes` now reflect true packed size. |
| `vecinfer` | No (default) | Optional `fused_sdpa` / index ring buffer exists — not yet converted to the packed-storage pattern used for `turboquant_rvq` |
| `kivi` | No | Named alongside `turboquant_rvq`/`vecinfer` as a #27 tier-1 target; not yet converted |
| `rabitq` | Partial / fused path | Fused encode/attend Metal path aims to avoid materializing K/V |
| Eviction methods | N/A | Reduce token count (`offset`), not bit-width |

Update this table as methods change. Prefer linking a `results.json` that
measures RSS or cache `nbytes` for packed paths.

## Engineering work items

1. Inventory each cache class: what `update_and_fetch` writes to parent state.
2. Add a shared reporting helper: `accounting_bytes` vs `resident_cache_bytes`.
3. Extend `scripts/validate_kv_memory.py` with optional OS RSS sampling.
4. Document fused/packed flags clearly in quickstart (when they help, when
   launch overhead hurts).
5. Treat "resident compression" as a first-class claim type in PR template
   checkboxes (already sketched in CONTRIBUTING honesty rule).

## Success metric

A user can run one script and see side-by-side:

- accounting ratio
- resident cache bytes (or RSS delta at long context)
- tok/s

without reading source to learn that dequantization hides the savings.
