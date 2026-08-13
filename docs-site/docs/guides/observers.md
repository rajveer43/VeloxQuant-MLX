---
id: observers
title: Observers
sidebar_label: Observers
slug: /guides/observers
---

# Observers

VeloxQuant-MLX includes four observer classes that implement the `QuantizationObserver` interface: `on_event(event)`, `report()`, `reset()`. They collect runtime metrics — distortion, latency, memory, and key norms — by receiving `QuantizationEvent` objects from a pipeline.

:::note[No `attach()` method]
Observers do not have an `attach(cache)` method. They are driven by feeding them `QuantizationEvent` instances directly via `on_event()`; nothing in the cache layer currently emits these events automatically. To use an observer today, call `on_event()` yourself at the point in your code where you have the relevant data (e.g. original vs. reconstructed keys, or timing/memory deltas).
:::

## Overview

| Observer | Tracks | Report type |
|---|---|---|
| `DistortionObserver` | Empirical MSE and inner-product distortion vs. TurboQuant's theoretical bounds | `DistortionReport` |
| `LatencyObserver` | Per-stage timing samples | `dict[str, dict[str, float]]` |
| `MemoryObserver` | Per-stage memory deltas | `dict[str, int]` |
| `KeyNormObserver` | Per-token key L2 norm² statistics (for RateQuant sensitivity) | `KeyNormReport` |

## QuantizationEvent

All observers consume `QuantizationEvent` objects:

```python
from veloxquant_mlx.observers.base import QuantizationEvent

event = QuantizationEvent(
    stage="key_quantize",  # name of the pipeline stage
    input_shape=(1, 8, 512, 128),
    elapsed_ms=1.23,
    memory_delta_bytes=4096,
    metadata={
        "x_original": original_keys_np,  # for DistortionObserver
        "x_reconstructed": decoded_keys_np,  # for DistortionObserver
        "key_l2_norm_sq": key_norms_np,  # for KeyNormObserver
    },
)
```

Each observer reads only the `metadata` keys it needs and silently ignores events missing them.

## DistortionObserver

Measures empirical MSE and inner-product distortion against the TurboQuant paper's theoretical bounds:

```python
from veloxquant_mlx.observers.distortion import DistortionObserver
from veloxquant_mlx.observers.base import QuantizationEvent

observer = DistortionObserver(b=2, d=128)  # bit-width and dim, for the theoretical bound

# Feed it events as you quantize/dequantize keys yourself
observer.on_event(
    QuantizationEvent(
        stage="key_quantize",
        input_shape=original_keys.shape,
        metadata={"x_original": original_keys, "x_reconstructed": decoded_keys},
    )
)

report = observer.report()
print(f"Empirical MSE       : {report.empirical_mse:.6f}")
print(f"Theoretical upper    : {report.theoretical_mse_upper:.6f}")
print(f"Theoretical lower    : {report.theoretical_mse_lower:.6f}")
print(f"MSE ratio (emp/upper): {report.mse_ratio:.3f}")
print(f"IP distortion        : {report.empirical_ip_distortion:.6f}")
print(f"Samples observed     : {report.n_samples}")
```

`DistortionReport` fields: `empirical_mse`, `theoretical_mse_upper`, `theoretical_mse_lower`, `mse_ratio`, `empirical_ip_distortion`, `n_samples`.

`DistortionObserver` also exposes a `.plot(save_path)` method that reproduces the TurboQuant paper's Figure 3 (MSE vs. bit-width) using matplotlib.

## LatencyObserver

Records per-stage timing samples:

```python
from veloxquant_mlx.observers.latency import LatencyObserver
from veloxquant_mlx.observers.base import QuantizationEvent
import time

observer = LatencyObserver()

t0 = time.perf_counter()
# ... do work ...
elapsed = (time.perf_counter() - t0) * 1000

observer.on_event(QuantizationEvent(stage="encode", input_shape=(512, 128), elapsed_ms=elapsed))

report = observer.report()
# report: {"encode": {"mean_ms": ..., "min_ms": ..., "max_ms": ..., "count": ...}, ...}
for stage, stats in report.items():
    print(f"{stage}: mean={stats['mean_ms']:.2f}ms count={stats['count']}")
```

## MemoryObserver

Tracks per-stage memory deltas (you supply the measurement, e.g. via `psutil`):

```python
from veloxquant_mlx.observers.memory import MemoryObserver
from veloxquant_mlx.observers.base import QuantizationEvent

observer = MemoryObserver()
observer.on_event(
    QuantizationEvent(stage="encode", input_shape=(512, 128), memory_delta_bytes=4096)
)

report = observer.report()  # {"encode": 4096, ...} — sum of deltas per stage
peak = observer.peak_delta_bytes()  # largest single delta observed across all stages
```

## KeyNormObserver

Accumulates per-token key L2 norm² — used to decide whether RateQuant-style per-layer bit allocation is likely to help (see the [RateQuant page](../algorithms/ratequant)):

```python
from veloxquant_mlx.observers.key_norm import KeyNormObserver
from veloxquant_mlx.observers.base import QuantizationEvent

observer = KeyNormObserver()  # zero-arg constructor

observer.on_event(
    QuantizationEvent(
        stage="key_norm",
        input_shape=(512, 128),
        metadata={"key_l2_norm_sq": per_token_norm_sq},  # scalar or iterable of floats
    )
)

report = observer.report()
print(f"Tokens observed : {report.n_tokens}")
print(f"Mean norm²      : {report.mean_norm_sq:.4f}")
print(f"Min / max norm² : {report.min_norm_sq:.4f} / {report.max_norm_sq:.4f}")
print(f"Heterogeneity   : {report.heterogeneity_ratio:.2f}")  # max/min; >>1 favors mixed-precision
```

`KeyNormReport` fields: `n_tokens`, `mean_norm_sq`, `min_norm_sq`, `max_norm_sq`, plus the computed `heterogeneity_ratio` property.

## Using multiple observers

Since observers only react to `on_event()`, you can fan the same event out to several:

```python
observers = [DistortionObserver(), LatencyObserver(), MemoryObserver()]

for obs in observers:
    obs.on_event(event)

for obs in observers:
    print(obs.report())
```

## See also

- [RateQuant guide](../algorithms/ratequant)
- [Observers API](../api/observers-api)
