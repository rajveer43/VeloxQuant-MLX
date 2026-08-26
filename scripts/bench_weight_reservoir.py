#!/usr/bin/env python3
"""Weight Reservoir PoC benchmark: quantize_model() vs save/load_reservoir().

Measures what docs/WEIGHT_RESERVOIR_IDEATION.md's revised claim actually
promises -- load-time and peak-RSS-during-load wins from skipping the
dequantize -> rotate -> Lloyd-Max-argmin pass, NOT cross-process RAM
sharing (Findings 1-2 in that doc show MLX has no zero-copy mmap path, so
sharing across processes does not happen for free).

Two measurements:
  A. Single process: wall-clock + peak RSS for quantize_model() (source
     weights -> TurboQuant compression) vs. save once + graft_reservoir()
     (deserialize only, no re-quantization).
  B. N concurrent processes loading the same already-saved reservoir vs.
     N concurrent processes each running quantize_model() from scratch --
     reports each process's own peak RSS (expect no sharing; the point is
     to quantify how much smaller each process's peak is with reservoir).

Usage
-----
::

    source .venv/bin/activate
    PYTHONPATH=. python scripts/bench_weight_reservoir.py \\
        --model mlx-community/Qwen3-4B-4bit --bits 4 --n-concurrent 4
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"


def _peak_rss_mb() -> float:
    # macOS ru_maxrss is bytes; Linux is KB. This repo targets macOS (see
    # docs/MEMORY_CONSTRAINT_FINDINGS.md), so assume bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _quantize_from_source(model_id: str, bits: int):
    import mlx_lm

    from veloxquant_mlx.weight.model_quantizer import quantize_model

    model, tokenizer = mlx_lm.load(model_id)
    model = quantize_model(model, bits=bits, use_hadamard=True)
    return model, tokenizer


def _run_subprocess(script_args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, __file__] + script_args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess failed: {proc.stderr[-4000:]}")
    # Last line is the JSON result
    last_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def _worker_quantize_from_source(model_id: str, bits: int) -> None:
    t0 = time.perf_counter()
    model, _ = _quantize_from_source(model_id, bits)
    import mlx.core as mx

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - t0
    print(json.dumps({"mode": "quantize_from_source", "elapsed_s": elapsed, "peak_rss_mb": _peak_rss_mb()}))


def _worker_graft_reservoir(model_id: str, bits: int, reservoir_path: str) -> None:
    import mlx.core as mx
    import mlx_lm

    from veloxquant_mlx.weight.reservoir import graft_reservoir

    t0 = time.perf_counter()
    # graft_reservoir() only needs a module tree with matching dotted names
    # to attach QuantizedLinear layers onto -- it does not need those names
    # to already be QuantizedLinear, and does not call quantize_weights()
    # at all. The raw mlx_lm-loaded model is enough; this is what makes the
    # "skip re-quantization" claim real rather than nominal.
    model, _ = mlx_lm.load(model_id)
    model = graft_reservoir(model, Path(reservoir_path))
    mx.eval(model.parameters())
    elapsed = time.perf_counter() - t0
    print(json.dumps({"mode": "graft_reservoir", "elapsed_s": elapsed, "peak_rss_mb": _peak_rss_mb()}))


def _worker_save_reservoir(model_id: str, bits: int, reservoir_path: str) -> None:
    import mlx.core as mx

    from veloxquant_mlx.weight.reservoir import save_reservoir

    model, _ = _quantize_from_source(model_id, bits)
    mx.eval(model.parameters())
    save_reservoir(model, Path(reservoir_path))
    print(json.dumps({"mode": "save_reservoir", "ok": True}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument("--output", default="figures/validation/weight_reservoir_results.json")
    # Internal worker dispatch (used when this script re-execs itself as a subprocess)
    parser.add_argument("--_worker", choices=["quantize", "graft", "save"], default=None)
    parser.add_argument("--_reservoir-path", default=None)
    args = parser.parse_args()

    if args._worker == "quantize":
        _worker_quantize_from_source(args.model, args.bits)
        return
    if args._worker == "graft":
        _worker_graft_reservoir(args.model, args.bits, args._reservoir_path)
        return
    if args._worker == "save":
        _worker_save_reservoir(args.model, args.bits, args._reservoir_path)
        return

    reservoir_path = REPO_ROOT / ".bench_tmp" / f"reservoir_{args.model.replace('/', '_')}_{args.bits}bit.vqrs"
    reservoir_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Saving reservoir for {args.model} ({args.bits}-bit)...")
    _run_subprocess(["--_worker", "save", "--model", args.model, "--bits", str(args.bits), "--_reservoir-path", str(reservoir_path)])
    reservoir_size_mb = reservoir_path.stat().st_size / 1024**2
    print(f"  reservoir file: {reservoir_size_mb:.1f} MB")

    print("[2/3] Single-process comparison: quantize_from_source vs graft_reservoir...")
    single_quantize = _run_subprocess(["--_worker", "quantize", "--model", args.model, "--bits", str(args.bits)])
    single_graft = _run_subprocess(["--_worker", "graft", "--model", args.model, "--bits", str(args.bits), "--_reservoir-path", str(reservoir_path)])
    print(f"  quantize_from_source: {single_quantize['elapsed_s']:.2f}s, peak RSS {single_quantize['peak_rss_mb']:.1f} MB")
    print(f"  graft_reservoir:      {single_graft['elapsed_s']:.2f}s, peak RSS {single_graft['peak_rss_mb']:.1f} MB")

    print(f"[3/3] Concurrent ({args.n_concurrent} processes) peak RSS: quantize_from_source vs graft_reservoir...")
    procs_q = [
        subprocess.Popen(
            [sys.executable, __file__, "--_worker", "quantize", "--model", args.model, "--bits", str(args.bits)],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(args.n_concurrent)
    ]
    concurrent_quantize = []
    for p in procs_q:
        out, err = p.communicate(timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"concurrent quantize worker failed: {err[-2000:]}")
        concurrent_quantize.append(json.loads(out.strip().splitlines()[-1]))

    procs_g = [
        subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--_worker",
                "graft",
                "--model",
                args.model,
                "--bits",
                str(args.bits),
                "--_reservoir-path",
                str(reservoir_path),
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(args.n_concurrent)
    ]
    concurrent_graft = []
    for p in procs_g:
        out, err = p.communicate(timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"concurrent graft worker failed: {err[-2000:]}")
        concurrent_graft.append(json.loads(out.strip().splitlines()[-1]))

    results = {
        "model": args.model,
        "bits": args.bits,
        "reservoir_file_mb": reservoir_size_mb,
        "single_process": {
            "quantize_from_source": single_quantize,
            "graft_reservoir": single_graft,
        },
        "concurrent": {
            "n": args.n_concurrent,
            "quantize_from_source": concurrent_quantize,
            "graft_reservoir": concurrent_graft,
        },
    }

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    print("\n=== Summary ===")
    print(f"Load time:  quantize_from_source={single_quantize['elapsed_s']:.2f}s  "
          f"graft_reservoir={single_graft['elapsed_s']:.2f}s  "
          f"speedup={single_quantize['elapsed_s']/single_graft['elapsed_s']:.2f}x")
    avg_q_rss = sum(r["peak_rss_mb"] for r in concurrent_quantize) / len(concurrent_quantize)
    avg_g_rss = sum(r["peak_rss_mb"] for r in concurrent_graft) / len(concurrent_graft)
    print(f"Concurrent per-process peak RSS (N={args.n_concurrent}): "
          f"quantize_from_source avg={avg_q_rss:.1f} MB  graft_reservoir avg={avg_g_rss:.1f} MB")
    print("NOTE: per Findings 1-2 in docs/WEIGHT_RESERVOIR_IDEATION.md, these are NOT "
          "shared physical pages -- each process pays its own RSS independently in both modes.")


if __name__ == "__main__":
    main()
