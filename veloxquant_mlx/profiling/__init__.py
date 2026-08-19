"""Energy and throughput profiling harness for VeloxQuant on Apple Silicon.

What this package measures
--------------------------
* Wall-clock throughput (tokens/s), split into prefill and decode phases.
* Peak GPU memory, via ``mx.get_peak_memory()``.
* Whole-package energy, by integrating sampled ``powermetrics`` power over
  wall time (requires root; see :mod:`~veloxquant_mlx.profiling.power_sampler`).

What this package CANNOT measure
--------------------------------
* **Memory bandwidth / bytes actually moved.** MLX exposes no traffic-side
  counter -- its entire ``mx.metal`` surface is allocation-side -- and
  ``powermetrics`` reports power and residency, not DRAM bytes/s. KV traffic is
  therefore *derived analytically* from cache geometry
  (:func:`~veloxquant_mlx.profiling.energy.kv_bytes_per_token`) and is labelled
  ``DERIVED`` everywhere it is reported. It is a calculation, not a measurement.
* **Per-process energy attribution.** ``powermetrics`` reports package-level
  power, so anything else running on the machine is included in the total.

J/token is a *sampled estimate*, not a hardware energy counter. Its resolution
is bounded by the sampling interval. See the module docstring of
``power_sampler`` for the integration method and its error characteristics.
"""

from veloxquant_mlx.profiling.energy import (
    RunMetrics,
    kv_bytes_per_token,
    measure_generation,
)
from veloxquant_mlx.profiling.power_sampler import PowerSample, PowerSampler

__all__ = [
    "PowerSample",
    "PowerSampler",
    "RunMetrics",
    "kv_bytes_per_token",
    "measure_generation",
]
