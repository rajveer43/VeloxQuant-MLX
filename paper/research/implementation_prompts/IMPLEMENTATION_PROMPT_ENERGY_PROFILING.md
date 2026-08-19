# Full Autonomous Implementation Prompt — Energy Profiling Harness (PoC)

Companion to #228 (energy-aware inference mode). This prompt covers **only the
measurement harness** — Step 1 (experiments A and B) of the investigation plan.
It deliberately stops before any Metal kernel work.

The deliverable is a harness that produces **real, committed J/token numbers**
for FP16 vs. VeloxQuant-compressed KV on Apple Silicon, plus the honest
statement of what those numbers do and do not establish.

---

## 0. Ground rules (apply to every phase)

- **Never invent an energy number.** This is the single most important rule in
  this prompt. The whole point of the harness is that no J/token figure exists
  yet for this project. A fabricated number is worse than no harness at all,
  because #228's product framing would be built on top of it. If a run cannot
  be completed, commit the harness and write "NOT YET RUN on hardware" in the
  CHANGELOG — this repo has done exactly that before (StreamingLLM `8f1259c`).
- **`powermetrics` requires root and this repo's test suite does not.** Every
  energy-measuring code path must degrade to `None`, never crash, never
  silently report zero, when run without privileges. CI runs unprivileged and
  must stay green.
- Work in small, buildable commits, one per phase, each passing tests before
  the next — matching the repo's existing one-commit-per-phase history.
- No new third-party dependencies. `powermetrics` is an OS binary parsed via
  `subprocess`; `mx.get_peak_memory()` / `mx.metal.device_info()` already
  exist in the pinned MLX (verified on MLX 0.32.0).
- Every new Python file gets a module docstring stating what it measures, what
  it cannot measure, and that J/token is sampled — not exact.
- Run `pytest veloxquant_mlx/tests/ -x -q` after each phase.
- Naming: `veloxquant_mlx/profiling/energy.py`, `.../power_sampler.py`,
  `benchmark_scripts/benchmark_energy.py`, `veloxquant_mlx/tests/profiling/
  test_energy.py`, `docs-site/docs/guides/energy-profiling.md`.

---

## 0.1 Environment facts already verified (do not re-derive)

Established on the target machine before this prompt was written:

| Fact | Value | How it constrains the design |
|---|---|---|
| Chip | Apple M4, arm64 | Real Apple Silicon — Step 1 **is** runnable here |
| `powermetrics` | present at `/usr/bin/powermetrics` | usable, but… |
| Privilege | **requires root; passwordless sudo is NOT configured** | harness must prompt-or-degrade, and cannot self-elevate. Root is checked *before* arg parsing, so no flag can be validated unprivileged |
| MLX | 0.32.0 | `mx.get_peak_memory`, `get_active_memory`, `get_cache_memory`, `reset_peak_memory` all present |
| `mx.metal.device_info()` | works (deprecated alias of `mx.device_info()`) | use `mx.device_info()`; the `metal.` form warns |
| GPU memory ceiling | `max_recommended_working_set_size` = 19.07 GB of 25.77 GB | model-size guard for the harness |
| Existing peak-memory idiom | `mx.metal.get_peak_memory()` in ~12 benchmark scripts | **deprecated form** — new code uses `mx.get_peak_memory()` (see `benchmark_vecinfer_comparison.py:78`, which already does the right thing with a fallback) |

**MLX exposes no bytes-moved / memory-bandwidth counter.** This is the key
design constraint and it is not what the issue assumed. See Phase 2.

---

## 0.2 One correction to the issue's plan, to apply throughout

The issue asks the harness to record "memory bandwidth" alongside GPU
utilization, and treats that as available. It is not directly available:

- MLX has **no** bandwidth or bytes-moved counter (the full public surface of
  `mx.metal` is `clear_cache`, `device_info`, `get_active_memory`,
  `get_cache_memory`, `get_peak_memory`, `is_available`, `reset_peak_memory`,
  `set_cache_limit`, `set_memory_limit`, `set_wired_limit`, `start_capture`,
  `stop_capture` — all allocation-side, none traffic-side).
- `powermetrics` reports GPU **power** and residency, not DRAM bytes/s, on
  Apple Silicon.

So bandwidth must be **derived analytically**, not sampled. Do not report a
measured bandwidth field; report a computed `kv_bytes_read_per_token` from
cache shape/dtype/bit-width, and label it as derived. Conflating a derived
figure with a measured one is precisely the credibility failure the issue's
own "non-goals" section warns against.

`mx.metal.start_capture()` / `stop_capture()` exist and can produce a GPU
trace for Xcode Instruments, which *can* show bandwidth — but that is an
interactive, non-scriptable path. Mention it in the docs as the escalation
route for Step 2; do not build the harness on it.

---

## Phase 1 — `power_sampler.py`: the privileged sampler, isolated

Create `veloxquant_mlx/profiling/power_sampler.py`. This is the **only** file
that knows about `powermetrics`. Keeping it isolated is what lets the rest of
the harness be unit-tested without root.

Implement:

```
@dataclass(frozen=True)
class PowerSample:
    cpu_mw: float | None
    gpu_mw: float | None
    ane_mw: float | None
    package_mw: float | None
    elapsed_s: float

class PowerSampler:
    """Background powermetrics sampler. Yields None-filled samples when
    unprivileged rather than raising."""
    def __init__(self, interval_ms: int = 100, samplers=("cpu_power","gpu_power")): ...
    def available(self) -> bool:      # False when not root / binary missing
    def __enter__(self) -> "PowerSampler": ...
    def __exit__(self, *exc) -> None: ...
    def energy_joules(self) -> float | None
    def mean_power_mw(self) -> dict[str, float | None]
```

Requirements:

- Spawn `powermetrics -i <interval> --samplers cpu_power,gpu_power
  -f plist` and read incrementally on a daemon thread. plist is far more
  stable to parse than the text output; parse with the stdlib `plistlib`.
  **Verify `-f plist` is accepted as the first step of this phase** — it could
  not be confirmed pre-flight, because `powermetrics` enforces its root check
  *before* validating arguments, so an unprivileged probe returns the same
  superuser error for every format value including bogus ones. If plist is
  rejected under sudo, fall back to parsing the default text output and say so
  in the module docstring; do not assume either format works.
- **Privilege detection must not be a `try: subprocess` that swallows
  everything.** Check `os.geteuid() == 0` first, then binary presence. The
  observed failure mode is the string `"powermetrics must be invoked as the
  superuser"` on stdout with a *zero* exit code in some invocations — so an
  exit-code-only check is unreliable. Treat any run that yields zero parsed
  samples as unavailable.
- `energy_joules()` integrates mean package power over wall time:
  `J = mean_package_W × elapsed_s`. Document in the docstring that this is a
  **sampled trapezoidal estimate**, not a hardware energy counter, and that
  the sampling interval bounds its resolution.
- Never raise from `__exit__`. A profiling failure must not fail the run being
  profiled.

Tests (`test_power_sampler.py`) — these run unprivileged in CI:

- `test_sampler_reports_unavailable_without_root` — when `os.geteuid()` is
  monkeypatched to non-zero, `available()` is False, the context manager still
  enters and exits cleanly, and `energy_joules()` returns `None` (**not** 0.0 —
  assert `is None`, because a silent 0.0 would propagate as a fake measurement).
- `test_plist_parsing_extracts_power_fields` — feed a **captured** plist
  fixture through the parser. Generate the fixture once on real hardware and
  commit it; do not hand-write a plausible-looking one.
- `test_sampler_never_raises_on_malformed_output` — truncated/garbage bytes
  yield no samples and no exception.

---

## Phase 2 — `energy.py`: metrics that work without root

Create `veloxquant_mlx/profiling/energy.py`. Everything here is unprivileged
and always available, so the harness stays useful even when `powermetrics`
isn't.

```
@dataclass(frozen=True)
class RunMetrics:
    tokens_generated: int
    wall_s: float
    tokens_per_s: float
    prefill_s: float
    decode_s: float
    peak_memory_mb: float
    kv_bytes_per_token: int        # DERIVED, not measured
    energy_j: float | None         # None when unprivileged
    j_per_token: float | None      # None when unprivileged
    mean_gpu_mw: float | None
    mean_cpu_mw: float | None
```

- `kv_bytes_per_token(config, n_layers, n_kv_heads, head_dim) -> int`:
  compute analytically from dtype/bit-width/budget. For eviction methods the
  per-token read is bounded by the **budget**, not the sequence length — that
  asymmetry (compression scales bytes/token by bit ratio; eviction *caps* it)
  is the single most useful thing this harness will show, and it is a
  calculation, not a measurement. Docstring must say so.
- Use `mx.get_peak_memory()` (not the deprecated `mx.metal.` form), with
  `mx.reset_peak_memory()` before each arm.
- **Separate prefill from decode timing.** The issue's energy model is about
  per-token decode cost; folding a one-shot prefill into a J/token average
  makes short runs look artificially expensive and is a real confound.
- Call `mx.eval()` / synchronize before stopping any timer. MLX is lazy; an
  unsynchronized timer measures graph construction, not execution. This is the
  most likely way to get a wrong-but-plausible number here.

Tests (`test_energy.py`):

- `test_kv_bytes_per_token_scales_with_bit_width` — 4-bit is ~1/4 of fp16.
- `test_kv_bytes_per_token_is_capped_by_budget_for_eviction` — at seq lengths
  well past budget, bytes/token stops growing.
- `test_metrics_degrade_to_none_without_power` — `j_per_token is None`, while
  `tokens_per_s` and `peak_memory_mb` are still populated.
- `test_j_per_token_is_none_not_zero_when_energy_missing` — guards the exact
  silent-zero bug that would fabricate a measurement.

---

## Phase 3 — `benchmark_energy.py`: experiments A and B only

Create `benchmark_scripts/benchmark_energy.py`. Arms, per the issue:

- **A — baseline**: stock `mlx_lm` cache, FP16 weights, FP16 KV.
- **B — VeloxQuant KV**: existing methods only, via `KVCacheConfig` +
  `KVCacheFactory.for_model(model, config)` (the established wiring; see
  `benchmark_scripts/qfilters_real_model_perplexity.py`). Cover at minimum one
  quantization arm (4-bit) and one eviction arm (Q-Filters at a fixed budget),
  because they reduce bytes/token by *different mechanisms* and the harness
  should be able to tell them apart.
- **C — Metal kernel**: **not implemented.** Leave a comment naming the
  precondition (a profiled bottleneck from A/B), not a stub.

Requirements:

- Identical prompt, identical decode length, identical seed across arms.
  Reuse the existing `PROMPT` / `MAX_TOKENS` convention from
  `benchmark_scripts/benchmark_core.py`.
- **Discard a warm-up run before every measured arm.** First-run Metal
  compilation and page-in are large and would otherwise be attributed to
  whichever arm ran first — a first-arm-looks-worse artifact.
- **Interleave and repeat arms** (A,B,A,B,…, ≥3 reps), report median plus
  spread. On a fan-cooled M4 under sustained load, thermal drift is a
  first-order confound: sequential arm-blocks would confound "ran later" with
  "used more energy". This directly serves the issue's thermal concern.
- Emit `energy_benchmark_results.json` and a table. Any `None` energy field
  prints as `n/a (requires sudo)` — **never** as a number or a dash that could
  be read as zero.
- Print the invocation needed for privileged mode, since the harness cannot
  elevate itself: `sudo python benchmark_scripts/benchmark_energy.py`.
- Guard model size against `max_recommended_working_set_size` (19.07 GB here)
  and fail with a clear message rather than thrashing swap — swap thrash would
  quietly corrupt the very energy numbers being collected.

---

## Phase 4 — Docs

`docs-site/docs/guides/energy-profiling.md`:

- How to run, both unprivileged (throughput + memory + derived bytes/token)
  and privileged (adds J/token).
- **What the numbers mean and what they do not.** State plainly: sampled
  power integration, not a hardware energy counter; whole-package attribution,
  so other processes on the machine contribute; bandwidth is derived from cache
  geometry, not measured.
- The Instruments/`start_capture` escalation path for anyone who genuinely
  needs measured bandwidth in Step 2.
- A results table that is **empty or marked NOT YET RUN** until a real run
  exists. Do not pre-fill it with the issue's illustrative 0.50/0.32/0.29
  figures — those are explicitly labelled illustrative targets in the issue,
  and copying them into docs is exactly how invented numbers enter a project.

Link from `docs-site/docs/algorithms/overview.md` and README where the other
guides are listed.

---

## Phase 5 — Run it, and report honestly

On the M4:

1. Run unprivileged; commit throughput/memory/bytes-per-token. These are real
   and need no sudo.
2. Run under `sudo` for J/token. If sudo is unavailable at that moment, say so
   and commit Phase-5 results as partial.
3. Fill the docs table with **whatever actually came out**, including a null
   or negative result.

**A null result is a real result and must be reported as one.** If B shows no
meaningful J/token reduction over A, that is a finding that directly answers
#228's open question 3 — and it must not be buried, softened, or retried until
it looks better. Equally, if compression *increases* energy because
dequantization cost exceeds the traffic saved, report that: the issue names
this as a hypothesis to test, and it is the outcome that would most change the
plan.

---

## Phase 6 — CHANGELOG + version bump

- CHANGELOG under a new minor version: harness added, what it measures, the
  root requirement, and either the real numbers or "NOT YET RUN on hardware".
- If Phase 5 produced numbers, state the hardware (Apple M4), MLX version
  (0.32.0), model, and rep count next to them. An energy figure without its
  hardware context is not reproducible.

---

## Phase 7 — Final verification

- `pytest veloxquant_mlx/tests/ -x -q` green, **run as a normal user** — this
  is the check that the unprivileged degradation path actually works.
- `ruff check` + `ruff format --check` clean.
- Confirm no fabricated number reached any committed file: grep the diff for
  the issue's illustrative `0.50`, `0.32`, `0.29`, and `36%`.

---

## Explicit non-goals

- No Metal kernel. Phase 3 arm C stays unimplemented by design.
- No `EnergyGuard`, no thermal policy, no energy "modes" — that is #228, and it
  should not be built before this harness reports.
- No mixed-precision-for-energy exploration (issue Step 3) until Step 1 numbers
  exist to justify it.
- No changes to any existing cache or quantizer. This PoC is purely additive;
  if it needs to modify an eviction cache to take a measurement, that is a
  signal the measurement is wrong, not that the cache is.

---

## Appendix — new files

```
veloxquant_mlx/profiling/__init__.py
veloxquant_mlx/profiling/power_sampler.py
veloxquant_mlx/profiling/energy.py
veloxquant_mlx/tests/profiling/__init__.py
veloxquant_mlx/tests/profiling/test_power_sampler.py
veloxquant_mlx/tests/profiling/test_energy.py
veloxquant_mlx/tests/profiling/fixtures/powermetrics_sample.plist   # captured, not hand-written
benchmark_scripts/benchmark_energy.py
docs-site/docs/guides/energy-profiling.md
```

## Appendix — files modified

```
CHANGELOG.md
README.md                                  (guide link)
docs-site/docs/algorithms/overview.md      (guide link)
pyproject.toml                             (version bump)
```
