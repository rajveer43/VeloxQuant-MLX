---
id: observers-api
title: Observers API
sidebar_label: Observers
slug: /api/observers-api
---

# Observers API

`veloxquant_mlx.observers`

All observers implement the `QuantizationObserver` interface: `on_event(event: QuantizationEvent) -> None`, `report()`, `reset()`. There is no `attach(cache)` method — observers are driven by feeding them `QuantizationEvent` instances directly; nothing in the cache layer emits these automatically today.

---

## QuantizationEvent

```python
from veloxquant_mlx.observers.base import QuantizationEvent
```

```python
@dataclass
class QuantizationEvent:
    stage: str
    input_shape: tuple
    elapsed_ms: float = 0.0
    memory_delta_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

| Field | Type | Description |
|---|---|---|
| `stage` | `str` | Name of the pipeline stage that emitted this event |
| `input_shape` | `tuple` | Shape of the input tensor at this stage |
| `elapsed_ms` | `float` | Wall-clock time for this stage, in milliseconds |
| `memory_delta_bytes` | `int` | Change in process RSS during this stage |
| `metadata` | `dict` | Stage-specific extra data (observer-specific keys, see below) |

---

## DistortionObserver

```python
from veloxquant_mlx.observers.distortion import DistortionObserver
```

Computes running empirical MSE and inner-product distortion, compared against TurboQuant's theoretical bounds.

### Constructor

```python
DistortionObserver(b: int = 2, d: int = 128, query: Optional[np.ndarray] = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `b` | `int` | `2` | Bit-width used, for computing the theoretical bound |
| `d` | `int` | `128` | Vector dimension |
| `query` | `Optional[np.ndarray]` | `None` | Fixed query vector for inner-product distortion tracking |

### Methods

```python
def on_event(self, event: QuantizationEvent) -> None
def report(self) -> DistortionReport
def reset(self) -> None
def plot(self, save_path: str) -> None
```

**`on_event(event)`** — Reads `event.metadata["x_original"]` and `event.metadata["x_reconstructed"]` (numpy arrays, shape `(batch, d)`); ignored if either key is absent.

**`report()`** — Returns a `DistortionReport`.

**`plot(save_path)`** — Saves a matplotlib figure reproducing the TurboQuant paper's Figure 3 (MSE vs. bit-width), requires `matplotlib`.

Static helpers (usable without an instance): `DistortionObserver.theoretical_mse_upper(b)`, `.theoretical_mse_lower(b)`, `.theoretical_ip_upper(b, d, y_norm_sq)`, `.theoretical_ip_lower(b, d, y_norm_sq)`.

### DistortionReport

| Field | Type | Description |
|---|---|---|
| `empirical_mse` | `float` | Observed mean squared reconstruction error |
| `theoretical_mse_upper` | `float` | Upper bound: `√(3π)/2 · 4^(-b)` |
| `theoretical_mse_lower` | `float` | Lower bound: `4^(-b)` |
| `mse_ratio` | `float` | `empirical_mse / theoretical_mse_upper` |
| `empirical_ip_distortion` | `float` | Mean squared inner-product error (only if `query` was set) |
| `n_samples` | `int` | Number of vectors observed |

---

## LatencyObserver

```python
from veloxquant_mlx.observers.latency import LatencyObserver
```

Records per-stage timing samples.

### Constructor

```python
LatencyObserver()
```

Takes no arguments.

### Methods

```python
def on_event(self, event: QuantizationEvent) -> None
def report(self) -> Dict[str, Dict[str, float]]
def reset(self) -> None
```

**`on_event(event)`** — Appends `event.elapsed_ms` to the sample list for `event.stage`.

**`report()`** — Returns `{stage: {"mean_ms": ..., "min_ms": ..., "max_ms": ..., "count": ...}}` for every stage seen.

---

## MemoryObserver

```python
from veloxquant_mlx.observers.memory import MemoryObserver
```

Tracks per-stage memory deltas, using whatever `memory_delta_bytes` the caller populates on each event.

### Constructor

```python
MemoryObserver()
```

Takes no arguments.

### Methods

```python
def on_event(self, event: QuantizationEvent) -> None
def report(self) -> Dict[str, int]
def peak_delta_bytes(self) -> int
```

**`report()`** — Returns `{stage: total_memory_delta_bytes}`, summed per stage.

**`peak_delta_bytes()`** — Largest single delta observed across all stages.

---

## KeyNormObserver

```python
from veloxquant_mlx.observers.key_norm import KeyNormObserver
```

Accumulates per-token key L2 norm² statistics — intended to inform RateQuant-style per-layer bit allocation, not automatic outlier routing.

### Constructor

```python
KeyNormObserver()
```

Takes no arguments.

### Methods

```python
def on_event(self, event: QuantizationEvent) -> None
def report(self) -> KeyNormReport
def reset(self) -> None
```

**`on_event(event)`** — Reads `event.metadata["key_l2_norm_sq"]` (a scalar or an iterable of floats); ignored if absent.

### KeyNormReport

| Field | Type | Description |
|---|---|---|
| `n_tokens` | `int` | Number of key norm² values accumulated |
| `mean_norm_sq` | `float` | Mean of accumulated norm² values |
| `min_norm_sq` | `float` | Minimum norm² observed |
| `max_norm_sq` | `float` | Maximum norm² observed |
| `heterogeneity_ratio` | `float` (property) | `max_norm_sq / min_norm_sq` — per RateQuant Theorem 3, values well above 1 indicate mixed-precision allocation will help |

---

## Example — all observers together

```python
import numpy as np
from veloxquant_mlx.observers.base import QuantizationEvent
from veloxquant_mlx.observers.distortion import DistortionObserver
from veloxquant_mlx.observers.memory import MemoryObserver
from veloxquant_mlx.observers.latency import LatencyObserver
from veloxquant_mlx.observers.key_norm import KeyNormObserver

dist_obs = DistortionObserver(b=2, d=128)
mem_obs = MemoryObserver()
lat_obs = LatencyObserver()
norm_obs = KeyNormObserver()

# Feed events at whatever point in your own pipeline has the relevant data
event = QuantizationEvent(
    stage="key_quantize",
    input_shape=(512, 128),
    elapsed_ms=1.2,
    memory_delta_bytes=4096,
    metadata={
        "x_original": original_keys,
        "x_reconstructed": decoded_keys,
        "key_l2_norm_sq": per_token_norm_sq,
    },
)
for obs in (dist_obs, mem_obs, lat_obs, norm_obs):
    obs.on_event(event)

print(f"MSE ratio    : {dist_obs.report().mse_ratio:.3f}")
print(f"Memory delta : {mem_obs.report()}")
print(f"Latency      : {lat_obs.report()}")
print(f"Heterogeneity: {norm_obs.report().heterogeneity_ratio:.2f}")
```

---

## See also

- [Observers guide](../guides/observers)
- [RateQuant](../algorithms/ratequant)
