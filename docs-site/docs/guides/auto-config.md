---
id: auto-config
title: Hardware-Aware Auto Configuration
sidebar_label: Auto Config
slug: /guides/auto-config
---

# Hardware-aware auto configuration

A single compression configuration is unlikely to be optimal for every model, head dimension, sequence length, and hardware setup. `select_kv_cache_config()` picks a method, bit-width, and group size for you — no manual tuning required.

```python
from veloxquant_mlx import select_kv_cache_config, WorkloadSpec

result = select_kv_cache_config(WorkloadSpec(head_dim=128, seq_len=32_000))
print(result.config)
print(result.reason)

cache = KVCacheFactory.create(result.config)
```

That's the whole workflow: describe your job with a `WorkloadSpec`, call `select_kv_cache_config()`, and pass the returned `config` straight into `KVCacheFactory.create()` or `KVCacheBuilder.for_model()`.

## What it picks from

The selector chooses from a small, well-understood pool of quantization methods rather than the full 40-method registry — every member is servable, reports full key+value telemetry, and exposes a plain int bit-width/group-size knob:

| Method | Bit-width | When |
|---|---|---|
| `turboquant_rvq` | 4-bit | Short context — accuracy cost of aggressive compression isn't worth it yet |
| `kivi` | 2-bit | Mid-length context — balanced default |
| `kvquant` | 3-bit + outlier isolation | Long context — the cache itself now dominates memory |
| `gear` | 2-bit | Memory pressure — overrides everything else once the workload would push device memory too high |

Eviction and hybrid methods (`h2o`, `snapkv`, `xquant`, …) are out of scope: they change *which* tokens are kept, not how each kept token is encoded, so they don't fit a bit-width/group-size selection axis. If you need one of those, pick it directly via `KVCacheConfig`.

## Selection rules

Rules apply in this priority order:

1. **Memory pressure.** If `hardware` reports the workload's estimated fp16 footprint would push total memory usage to 75% or more, `gear` (2-bit) is selected regardless of sequence length.
2. **Sequence length.** Short contexts (under 2,048 tokens) favor `turboquant_rvq` (4-bit); long contexts (2,048+ up to 16,384) use `kivi`; contexts of 16,384 tokens or more favor `kvquant` (3-bit with outlier isolation).
3. **Head dimension.** If `head_dim >= 256`, the group size doubles (32 → 64) so each quantization group still holds enough elements to amortize per-group scale/zero-point overhead.

## Describing your workload

```python
from veloxquant_mlx import WorkloadSpec

workload = WorkloadSpec(
    head_dim=128,  # must be a power of 2
    seq_len=64_000,  # expected or worst-case sequence length
    n_layers=32,  # attention layers sharing this config (memory estimate only)
    batch_size=4,  # concurrent sequences (memory estimate only)
)
```

`n_layers` and `batch_size` only affect the memory-pressure estimate — they don't change which method wins in the sequence-length rule on their own, but scaling either one up can be enough to push you over the pressure threshold and force `gear`.

## Hardware detection

By default, `select_kv_cache_config()` auto-detects the machine's unified memory via MLX:

```python
result = select_kv_cache_config(workload)  # hardware=None -> auto-detects
```

Pass an explicit `HardwareInfo` to override — useful in tests, CI, or when serving multiple requests where you want to account for memory already in use:

```python
from veloxquant_mlx import HardwareInfo

hardware = HardwareInfo(
    total_memory_bytes=32 * 1024**3,  # 32 GB unified memory
    active_memory_bytes=4 * 1024**3,  # model weights + other caches already resident
)
result = select_kv_cache_config(workload, hardware)
```

If MLX device introspection fails for any reason (no Metal device, older MLX version), `detect_hardware_info()` fails closed to an empty `HardwareInfo()` rather than raising — you just lose the memory-pressure rule and fall back to sequence-length-only selection.

## Reading the decision

Every result carries a human-readable `reason` alongside the `config`, useful for logging or a CLI `--explain` flag:

```python
result = select_kv_cache_config(WorkloadSpec(head_dim=128, seq_len=100))
print(result.reason)
# "seq_len=100 < 2048 (short context): selected turboquant_rvq (4-bit) for higher precision"
```

## CLI

`select_kv_cache_config()` is also exposed as `veloxquant auto-config`, for callers that want to shell out rather than import Python — e.g. a driver app that already knows the workload shape and wants a JSON config back.

```bash
veloxquant auto-config \
  --head-dim 128 \
  --seq-len 32000 \
  --n-layers 32 \
  --batch-size 4

# Machine-readable JSON
veloxquant auto-config --seq-len 32000 --json

# Skip hardware auto-detection and supply memory explicitly
# (e.g. a caller that already read the Mac's memory via sysctl)
veloxquant auto-config \
  --seq-len 32000 \
  --total-memory-bytes 34359738368 \
  --active-memory-bytes 4294967296
```

All four `WorkloadSpec` fields have defaults (`--head-dim 128 --seq-len 4096 --n-layers 1 --batch-size 1`), so a bare `veloxquant auto-config` runs too. `--json` output reports only the selected method's own knob fields — not the other pool methods' unrelated `KVCacheConfig` defaults — alongside the `workload`, `hardware`, and `reason` that produced the pick:

```json
{
  "workload": {"head_dim": 128, "seq_len": 32000, "n_layers": 32, "batch_size": 4},
  "hardware": {"total_memory_bytes": 34359738368, "active_memory_bytes": 4294967296},
  "config": {"method": "kvquant", "head_dim": 128, "kvquant_bits": 3, "kvquant_group_size": 32, "kvquant_outlier_fraction": 0.01},
  "reason": "seq_len=32000 >= 16384 (long context): selected kvquant (3-bit NUQ + outlier isolation) for aggressive compression"
}
```

`--total-memory-bytes` and `--active-memory-bytes` mirror the `HardwareInfo` override shown above — pass `--total-memory-bytes` to skip `detect_hardware_info()` entirely; `--active-memory-bytes` is ignored unless `--total-memory-bytes` is also set.

## Good to know

- **Deterministic.** The same `WorkloadSpec` + `HardwareInfo` always produce the same `config` — there's no randomness or hidden state between calls.
- **Validated inputs.** `WorkloadSpec` rejects a non-power-of-2 `head_dim` and any non-positive `seq_len`, `n_layers`, or `batch_size` at construction time (`QuantizerConfigError`), so mistakes surface immediately rather than deep inside cache construction.
- **Just a starting point.** The selector optimizes for a general-purpose default; if you know your workload's accuracy/memory tradeoff better than the heuristic does, construct a `KVCacheConfig` directly instead.

## See also

- [Cache API reference](../api/cache) — `KVCacheConfig`, `KVCacheFactory`, `KVCacheBuilder`
- [Mixed Precision guide](./mixed-precision) — manual per-layer bit-width control
- [Mac Method Recommender](./mac-recommender) — chip/RAM-driven method choice via the CLI, a complementary tool for a different question ("what fits on my Mac" vs. "what fits this workload")
