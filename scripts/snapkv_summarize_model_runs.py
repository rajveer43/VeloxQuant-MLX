"""Summarize repeated model checks and assert accelerated backend parity."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lines = [
        "# SnapKV 8–9B model validation",
        "",
        "Each model: two warmups and five measured trials per backend/scenario, alternating backend order. Plain cache plus SnapKV reference, MLX and Metal. Fresh caches; 16 greedy decode steps. Manual loop includes scalar token reads. This is backend parity and timing validation on one repetitive prompt, not a quality evaluation.",
        "",
        "| Model / budget / chunk | Backend | Prefill median [min–max] ms | Decode median ms/step |",
        "|---|---|---:|---:|",
    ]
    for file_index, path in enumerate(args.inputs):
        if file_index:
            lines.extend(
                [
                    "| Model / budget / chunk | Backend | Prefill median [min–max] ms | Decode median ms/step |",
                    "|---|---|---:|---:|",
                ]
            )
        data = json.loads(path.read_text())
        rows = data["results"]
        assert all(x["status"] == "passed" for x in rows), path
        label = Path(data["model"]).parts[-3].replace("models--mlx-community--", "")
        summary = []
        for budget, chunk in sorted({(x["budget"], x["chunk"]) for x in rows}, reverse=True):
            ref = next(
                x
                for x in rows
                if x["budget"] == budget and x["chunk"] == chunk and x["backend"] == "reference"
            )
            for backend in ("plain", "reference", "mlx", "metal"):
                group = [
                    x
                    for x in rows
                    if x["budget"] == budget
                    and x["chunk"] == chunk
                    and x["backend"] == backend
                    and not x["warmup"]
                ]
                if backend in ("mlx", "metal"):
                    assert all(
                        x["reference_max_logit_error"] == 0 and x["tokens"] == ref["tokens"]
                        for x in group
                    ), (path, budget, chunk, backend)
                prefill = [x["prefill_ms"] for x in group]
                decode = [x["decode_ms_per_step"] for x in group]
                result = dict(
                    budget=budget,
                    chunk=chunk,
                    backend=backend,
                    trials=len(group),
                    prefill_median_ms=statistics.median(prefill),
                    prefill_min_ms=min(prefill),
                    prefill_max_ms=max(prefill),
                    decode_median_ms=statistics.median(decode),
                )
                summary.append(result)
                lines.append(
                    f"| {label} / {budget} / {chunk} | {backend} | {statistics.median(prefill):.1f} [{min(prefill):.1f}–{max(prefill):.1f}] | {statistics.median(decode):.2f} |"
                )
        data["summary"] = summary
        data["measured_backend_parity"] = (
            "Zero reference logit error and identical generated tokens for all measured MLX/Metal runs."
        )
        path.write_text(json.dumps(data, indent=2))
        lines.extend(
            [
                "",
                f"{label}: {len(rows)} successful runs; prompt {data['prompt_tokens']} tokens. Revision `{Path(data['model']).name}`. Raw data: [{path.name}](benchmarks/{path.name}).",
                "",
            ]
        )
    lines += [
        "## Interpretation",
        "",
        "Both models pass the tested backend parity checks. Metal timings are mixed and do not establish a consistent model-level speedup. Auto remains MLX. System load, thermal state and power mode were not controlled; the Qwen download overlapped early GLM inference. Five measured samples per group are insufficient for a useful tail-latency estimate. No peak-memory or broad task-quality claims follow from these tests.",
        "",
    ]
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
