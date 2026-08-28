---
id: auto-config-api
title: Auto Config API
sidebar_label: Auto Config
slug: /api/auto-config-api
---

# Auto Config API

`veloxquant_mlx.config`

Hardware-aware automatic KV-cache configuration. `select_kv_cache_config()` picks a method, bit-width, and group size from a small pool of servable quantization methods, given a workload description and (optionally) a hardware description.

---

## WorkloadSpec

```python
from veloxquant_mlx import WorkloadSpec
```

```python
@dataclass(frozen=True)
class WorkloadSpec:
    head_dim: int = 128
    seq_len: int = 4_096
    n_layers: int = 1
    batch_size: int = 1
```

Describes the job the KV cache needs to serve.

| Field | Type | Default | Description |
|---|---|---|---|
| `head_dim` | `int` | `128` | Attention head dimension. Must be a positive power of 2. |
| `seq_len` | `int` | `4096` | Expected (or worst-case) sequence length in tokens. Drives the short/long-context precision tradeoff. Must be `>= 1`. |
| `n_layers` | `int` | `1` | Attention layers sharing this config. Used only to size the memory-pressure estimate. Must be `>= 1`. |
| `batch_size` | `int` | `1` | Concurrent sequences. Used only to size the memory-pressure estimate. Must be `>= 1`. |

Raises `QuantizerConfigError` at construction time if `head_dim` is not a positive power of 2, or if `seq_len`, `n_layers`, or `batch_size` is less than 1.

### Methods

**`fp16_kv_bytes() -> int`** — Estimated fp16 footprint of the full K+V cache for this workload: `2 (K and V) * batch_size * n_layers * seq_len * head_dim * 2 bytes`.

---

## HardwareInfo

```python
from veloxquant_mlx import HardwareInfo
```

```python
@dataclass(frozen=True)
class HardwareInfo:
    total_memory_bytes: int | None = None
    active_memory_bytes: int = 0
```

Describes the machine the KV cache will run on.

| Field | Type | Default | Description |
|---|---|---|---|
| `total_memory_bytes` | `int \| None` | `None` | Total device memory (unified memory on Apple Silicon). `None` disables memory-pressure-based selection. |
| `active_memory_bytes` | `int` | `0` | Memory already in use before this cache is allocated (model weights, other requests' caches). |

### Methods

**`pressure_fraction(additional_bytes: int) -> float | None`** — Fraction of total memory used once `additional_bytes` is added: `(active_memory_bytes + additional_bytes) / total_memory_bytes`. Returns `None` if `total_memory_bytes` is unset or `0`.

---

## detect_hardware_info

```python
from veloxquant_mlx import detect_hardware_info
```

```python
def detect_hardware_info() -> HardwareInfo
```

Auto-detects `HardwareInfo` from the running MLX device via `mx.device_info()` and `mx.get_active_memory()`. Falls back to an empty (unknown-memory) `HardwareInfo()` if MLX is unavailable or device introspection raises for any reason — callers then get sequence-length-only selection instead of an exception.

---

## AutoConfigResult

```python
from veloxquant_mlx import AutoConfigResult
```

```python
@dataclass(frozen=True)
class AutoConfigResult:
    config: KVCacheConfig
    reason: str
```

| Field | Type | Description |
|---|---|---|
| `config` | `KVCacheConfig` | The resulting configuration, ready for `KVCacheFactory.create()` or `KVCacheBuilder.for_model()`. |
| `reason` | `str` | Short human-readable explanation of why this method/bit-width was chosen — useful for logging or a CLI `--explain` flag. |

---

## select_kv_cache_config

```python
from veloxquant_mlx import select_kv_cache_config
```

```python
def select_kv_cache_config(
    workload: WorkloadSpec,
    hardware: HardwareInfo | None = None,
) -> AutoConfigResult
```

Picks a `KVCacheConfig` for the given workload and hardware.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `workload` | `WorkloadSpec` | required | Description of the job. |
| `hardware` | `HardwareInfo \| None` | `None` | Description of the target machine. If `None`, auto-detects via `detect_hardware_info()`. |

### Selection rules, in priority order

1. **Memory pressure** — if `hardware.pressure_fraction(workload.fp16_kv_bytes())` is `>= 0.75`, selects `gear` (`gear_bits=2`) regardless of sequence length.
2. **Sequence length**:
   - `seq_len < 2048` → `turboquant_rvq` (`bit_width_inlier=4`)
   - `2048 <= seq_len < 16384` → `kivi` (`bit_width_inlier=2`)
   - `seq_len >= 16384` → `kvquant` (`kvquant_bits=3`, `kvquant_outlier_fraction=0.01`)
3. **Head dimension** — if `head_dim >= 256`, the relevant group-size field (`kivi_group_size`, `kvquant_group_size`, or `gear_group_size`) is set to `64` instead of the default `32`. This rule does not apply to the short-context branch (`turboquant_rvq` has no group-size field).

### Example

```python
from veloxquant_mlx import select_kv_cache_config, WorkloadSpec, HardwareInfo, KVCacheFactory

result = select_kv_cache_config(
    WorkloadSpec(head_dim=128, seq_len=64_000, n_layers=32, batch_size=4),
    HardwareInfo(total_memory_bytes=32 * 1024**3, active_memory_bytes=4 * 1024**3),
)
print(result.config.method)  # "kvquant"
print(result.reason)

cache = KVCacheFactory.create(result.config)
```

---

## See also

- [Auto Config guide](../guides/auto-config)
- [Cache API](./cache) — `KVCacheConfig`, `KVCacheFactory`, `KVCacheBuilder`
