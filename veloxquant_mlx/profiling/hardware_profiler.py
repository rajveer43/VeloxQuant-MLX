"""Apple-Silicon hardware detection for the auto-selection pipeline.

Extends the minimal :class:`veloxquant_mlx.config.auto_config.HardwareInfo`
(total/active memory) with everything the strategy planner needs from the
machine: chip identity and generation, MLX and macOS versions, Metal
availability, and (optionally) measured bandwidth/latency characteristics.

The detector never raises: if MLX is missing or device introspection fails it
returns a best-effort ``HardwareProfile`` built from stdlib calls alone (``platform``),
so callers degrade to unknown-hardware heuristics instead of crashing.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

#: Nominal peak memory bandwidth (GB/s) for the base Apple Silicon chips, used
#: only when no per-Mac measurement exists. Pro/Max/Ultra variants are faster
#: (approximately 2x/4x/8x the base); the planner treats this as a soft proxy
#: for "how fast can we stream KV bytes through the GPU", not a hard spec.
_BASE_BANDWIDTH_GBPS: dict[int, float] = {
    1: 68.3,  # M1
    2: 100.0,  # M2
    3: 100.0,  # M3
    4: 120.0,  # M4
}

#: Minimum base bandwidth assumed for an unknown chip generation.
_DEFAULT_BANDWIDTH_GBPS: float = 100.0


def chip_generation(chip: str) -> int:
    """Extract the numeric generation from a chip name.

    ``"M1"`` -> ``1``, ``"M4 Pro"`` -> ``4``, ``"M2 Ultra"`` -> ``2``.
    Returns ``0`` when no generation can be parsed (unknown or CPU-only host).
    """
    for i in range(1, 10):
        if f"M{i}" in chip.upper():
            return i
    return 0


def _stdlib_chip() -> str:
    """Best-effort chip name from stdlib hal only.

    ``platform.processor()`` returns ``"arm"`` on Apple Silicon; macOS stores
    the real brand ("Apple M4") under ``sysctl machdep.cpu.brand_string``, so
    probe that when available and fall back to the platform fields.
    """
    try:
        import subprocess

        brand = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode()
            .strip()
        )
        if "'" in brand:  # the tool may print "Apple M4"
            brand = brand.strip("'")
        return brand
    except Exception:
        return platform.processor() or platform.machine() or "unknown"


def _mlx_device_info() -> dict:
    """Call ``mx.device_info()`` entirely defensively."""
    try:
        import mlx.core as mx

        info = dict(mx.device_info())
        active = mx.get_active_memory()
        info["_active_memory"] = active
        return info
    except Exception:
        return {}


def _mlx_version() -> str | None:
    """MLX version from package metadata (``mlx.__version__`` is unreliable)."""
    try:
        from importlib.metadata import version

        return version("mlx")
    except Exception:
        return None


def _metal_available() -> bool:
    try:
        from veloxquant_mlx.metal import metal_available

        return bool(metal_available())
    except Exception:
        return True  # conservative: assume Metal until proven otherwise


def _macos_version() -> str | None:
    try:
        ver = platform.mac_ver()[0]
        return ver or None
    except Exception:
        return None


@dataclass(frozen=True)
class HardwareProfile:
    """Comprehensive description of the Apple-Silicon environment.

    Unlike the minimal ``HardwareInfo``, this is the object the auto-selection
    pipeline passes around: every field is a fact the planner can cite in an
    explanation.
    """

    chip: str = "unknown"
    chip_generation: int = 0
    total_memory_bytes: int | None = None
    available_memory_bytes: int = 0
    mlx_version: str | None = None
    macos_version: str | None = None
    metal_available: bool = True
    peak_memory_bandwidth_gbps: float | None = None
    avg_quantize_latency_ms_per_token: float | None = None

    @classmethod
    def detect(cls) -> HardwareProfile:
        return detect_hardware_profile()

    @property
    def bandwidth_gbps(self) -> float | None:
        """Likely peak bandwidth: measured, else nominal, else None."""
        if self.peak_memory_bandwidth_gbps is not None:
            return self.peak_memory_bandwidth_gbps
        if self.chip_generation:
            return _BASE_BANDWIDTH_GBPS.get(self.chip_generation, _DEFAULT_BANDWIDTH_GBPS)
        return None

    def to_dict(self) -> dict:
        return {
            "chip": self.chip,
            "chip_generation": self.chip_generation,
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "mlx_version": self.mlx_version,
            "macos_version": self.macos_version,
            "metal_available": self.metal_available,
            "peak_memory_bandwidth_gbps": self.peak_memory_bandwidth_gbps,
            "avg_quantize_latency_ms_per_token": self.avg_quantize_latency_ms_per_token,
        }


def detect_hardware_profile() -> HardwareProfile:
    """Detect a :class:`HardwareProfile` from MLX + stdlib introspection.

    **Process**:
    1. Call :func:`_mlx_device_info()` for chip, memory, active allocation
    2. Call :func:`_stdlib_chip()` as fallback if MLX unavailable
    3. Extract chip generation from chip name (M1–M4)
    4. Calculate available memory (total - active)
    5. Probe MLX version, macOS version, Metal availability
    6. Look up nominal peak bandwidth from chip generation table

    **Robustness**:
    - Never raises (all failures result in "unknown" fallback values)
    - MLX absence only affects chip/memory detection; OS fields come from stdlib
    - Metal unavailability doesn't fail; conservative assume True
    - Missing config fields default to None/0

    **Returns**:
        HardwareProfile with:
        - chip: Detected Apple Silicon model (e.g., "Apple M4")
        - chip_generation: Integer 1–4 (0 if unknown)
        - total_memory_bytes: GB from MLX (None if unavailable)
        - available_memory_bytes: total - active usage
        - mlx_version: MLX package version (None if not installed)
        - macos_version: macOS version string
        - metal_available: Whether Metal GPU support is available
        - peak_memory_bandwidth_gbps: Nominal peak (from table, generation-based)

    **Cost**: ~5–10ms (includes sysctl call for chip brand)

    **Example**:
        >>> hw = detect_hardware_profile()
        >>> print(f"{hw.chip} (gen {hw.chip_generation})")
        Apple M4 (gen 4)
        >>> print(f"Available: {hw.available_memory_bytes / 1e9:.1f} GiB")
        Available: 18.2 GiB
    """
    info = _mlx_device_info()
    chip = str(info.get("device_name") or _stdlib_chip())
    generation = chip_generation(chip)

    total = info.get("memory_size")
    active = info.get("_active_memory", 0)
    available = 0
    if isinstance(total, int) and total > 0:
        available = max(0, total - (int(active) if isinstance(active, int) else 0))

    has_metal = _metal_available()
    return HardwareProfile(
        chip=chip,
        chip_generation=generation,
        total_memory_bytes=total if isinstance(total, int) and total > 0 else None,
        available_memory_bytes=available,
        mlx_version=_mlx_version(),
        macos_version=_macos_version(),
        metal_available=has_metal,
        peak_memory_bandwidth_gbps=(
            _BASE_BANDWIDTH_GBPS.get(generation) if generation else None
        ),
    )


def measure_bandwidth_gbps(
    iterations: int = 32, bytes_per_transfer: int = 64 * 1024**2
) -> float | None:
    """Measure approximate peak GPU memory bandwidth via a copy benchmark.

    **Benchmark**: Allocates a tensor, performs repeated additions (forcing
    GPU memory copy), measures wall time, and computes GB/s throughput.

    **Method**: For each of N iterations:
    1. Allocate src tensor (fp16, ``bytes_per_transfer`` bytes)
    2. Perform add operation (forces GPU read + write)
    3. Evaluate result (force completion)

    **Calculation**: Bytes moved = (bytes_per_transfer × 2) × iterations
                     (factor 2 for read + write)
    Time = wall clock
    Bandwidth = bytes_moved / time / 1e9 (convert to GB/s)

    **Arguments**:
        iterations: Number of copy operations (default 32, ~2 seconds on M4)
        bytes_per_transfer: Size per copy (default 64 MiB)

    **Returns**:
        Measured bandwidth in GB/s, or None if:
        - MLX not installed
        - Metal GPU unavailable
        - Operation times out
        - Any exception occurs

    **Cost**: ~1–2 seconds (default parameters)

    **Accuracy**: Rough proxy; affected by GPU clock throttling, thermal state,
    competing workloads. Used only to override nominal peak bandwidth table
    when user explicitly requests measurement.

    **Example**:
        >>> bw = measure_bandwidth_gbps()
        >>> if bw:
        ...     print(f"Measured: {bw:.1f} GB/s")
        ... else:
        ...     print("Fallback to nominal peak")
        Measured: 118.5 GB/s
    """
    try:
        import time

        import mlx.core as mx

        src = mx.random.normal((bytes_per_transfer // 2,), dtype=mx.float16)
        mx.eval(src)

        start = time.perf_counter()
        for _ in range(iterations):
            dst = mx.add(src, 1)
            mx.eval(dst)
        elapsed = time.perf_counter() - start
        bytes_moved = iterations * bytes_per_transfer * 2  # read + write
        return (bytes_moved / elapsed) / 1e9
    except Exception:
        return None
