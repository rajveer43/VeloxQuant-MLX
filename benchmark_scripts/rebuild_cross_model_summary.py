"""Rebuild cross-model comparison from all existing per-model results.json files.

Scans figures/vecinfer/<model>/ directories, loads results.json files,
filters to the 8 standard configs, and regenerates:
  - figures/vecinfer/_summary/cross_model_comparison.png
  - figures/vecinfer/_summary/results_all.json
  - figures/vecinfer/_summary/SUMMARY.md (full markdown report)

Run from repo root:
    PYTHONPATH=. python benchmark_scripts/rebuild_cross_model_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PALETTE = {
    "fp16-baseline": "#4C72B0",
    "TQ-2bit": "#C44E52",
    "TQ-3bit": "#DD8452",
    "TQ-4bit": "#55A868",
    "RVQ-2bit": "#8172B2",
    "RVQ-1bit": "#CCB974",
    "VecInfer-2bit": "#00d4ff",
    "VecInfer-1bit": "#7c3aed",
}

_STD_CONFIGS = list(_PALETTE.keys())

# Models to exclude from rollup (MLA / broken architectures)
_EXCLUDE = {
    "DeepSeek-V2-Lite-Chat-4bit-mlx",  # MLA, all non-fp16 broken
    "Qwen3-4B-4bit",  # head_dim=80 (not power-of-2), all quant configs broken
}


def _ok(r: dict) -> bool:
    return not r.get("error") and r.get("tokens_generated", 0) > 0


def load_all_results(figures_dir: Path) -> dict:
    per_model = {}
    meta = {}
    for folder in sorted(figures_dir.iterdir()):
        if folder.name.startswith("_") or not folder.is_dir():
            continue
        if folder.name in _EXCLUDE:
            print(f"  SKIP (excluded): {folder.name}")
            continue
        rf = folder / "results.json"
        if not rf.exists():
            continue
        with open(rf) as f:
            data = json.load(f)

        model_id = data.get("model", f"mlx-community/{folder.name}")
        results = data.get("results", [])

        std_results = [r for r in results if r.get("name") in _STD_CONFIGS]

        # Need fp16-baseline + at least one quant config working
        fp16 = next((r for r in std_results if r["name"] == "fp16-baseline"), None)
        if fp16 is None or not _ok(fp16):
            print(f"  SKIP (no fp16 baseline): {folder.name}")
            continue

        working = [r for r in std_results if _ok(r)]
        if len(working) < 2:
            print(f"  SKIP (too few working configs): {folder.name}")
            continue

        print(f"  OK: {folder.name} — {len(std_results)} configs, {len(working)} working")
        per_model[model_id] = std_results
        meta[model_id] = {
            "head_dim": data.get("head_dim"),
            "n_kv_heads": data.get("n_kv_heads"),
            "n_q_heads": data.get("n_q_heads"),
            "n_layers": data.get("n_layers"),
        }
    return per_model, meta


def plot_cross_model(per_model: dict, out_path: Path) -> None:
    if not per_model:
        return

    # Only include configs that have at least one model with a working result
    config_names = [
        c
        for c in _STD_CONFIGS
        if any(_ok(r) for results in per_model.values() for r in results if r["name"] == c)
    ]
    models = list(per_model.keys())

    fig, axes = plt.subplots(2, 1, figsize=(max(16, len(models) * 2.6), 12))
    fig.suptitle(
        "Cross-model comparison · key compression and throughput\n"
        f"({len(models)} models · Apple Silicon MLX · 8 KV-cache configurations)",
        fontsize=14,
        fontweight="bold",
    )

    n_cfg = len(config_names)
    width = 0.85 / n_cfg
    x = np.arange(len(models))

    for ax_idx, (metric, ylabel, title) in enumerate(
        [
            ("key_compression", "Key compression (×)", "Key Compression Ratio"),
            ("throughput_tok_s", "Throughput (tok/s)", "Generation Throughput"),
        ]
    ):
        ax = axes[ax_idx]
        for i, cfg in enumerate(config_names):
            vals = []
            for m in models:
                hit = next((r for r in per_model[m] if r["name"] == cfg), None)
                val = hit[metric] if (hit and _ok(hit)) else 0.0
                vals.append(val)
            ax.bar(
                x + (i - n_cfg / 2 + 0.5) * width,
                vals,
                width,
                color=_PALETTE.get(cfg, "#999"),
                label=cfg,
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_xticks(x)
        short_names = [
            m.split("/")[-1]
            .replace("-Instruct", "")
            .replace("-4bit", "")
            .replace("-Chat", "")
            .replace("-it", "")
            for m in models
        ]
        ax.set_xticklabels(short_names, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=9, ncol=min(n_cfg, 4), loc="upper left")
            ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def build_summary_md(per_model: dict, meta: dict, out_path: Path) -> None:
    lines = [
        "# VecInfer comparative study",
        "",
        "Cross-model benchmark of fp16 baseline vs. four KV-cache compression",
        "methods at 8 configurations, run on Apple Silicon via MLX. Single prompt,",
        "max 120 generated tokens, `mlx_lm.generate(prompt_cache=...)`.",
        "",
        "See [`cross_model_comparison.png`](cross_model_comparison.png) for the",
        "combined bar chart, and `figures/vecinfer/<model>/comparison_summary.png`",
        "for per-model 4-panel summaries.",
        "",
        "## Models tested",
        "",
        "| Model | head_dim | n_kv_heads | n_q_heads | n_layers | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]

    def _note(model_id: str, m: dict, per_model: dict) -> str:
        results = per_model.get(model_id, [])
        non_fp16 = [r for r in results if r["name"] != "fp16-baseline"]
        working = sum(1 for r in non_fp16 if _ok(r))
        total = len(non_fp16)
        if working == total and total > 0:
            return f"full {working}/{total}"
        failed_names = [r["name"] for r in non_fp16 if r.get("error") or not _ok(r)]
        suffix = "; ".join(failed_names[:3]) + " OOM/failed" if failed_names else ""
        return f"{working}/{total} working; {suffix}" if suffix else f"{working}/{total} working"

    for model_id, m in meta.items():
        stem = model_id.split("/")[-1]
        hd = m.get("head_dim", "—")
        nkv = m.get("n_kv_heads", "—")
        nq = m.get("n_q_heads", "—")
        nl = m.get("n_layers", "—")
        note = _note(model_id, m, per_model)
        lines.append(f"| {stem} | {hd} | {nkv} | {nq} | {nl} | {note} |")

    # Excluded models note
    lines += [
        "",
        "**Excluded:**",
        "- **DeepSeek-V2-Lite-Chat-4bit-mlx** — MLA stores compressed KV at 192-dim",
        "  (non-standard shape); breaks all per-cache wrappers.",
        "- **Qwen3-4B-4bit** — head_dim=80 (not a power of 2); Walsh-Hadamard and",
        "  rotation-based methods require power-of-2 head_dim.",
        "",
    ]

    # Compression table
    lines += [
        "## Key compression ratio (higher is better)",
        "",
        "| Model | TQ-2bit | TQ-3bit | TQ-4bit | RVQ-2bit | RVQ-1bit | VecInfer-2bit | VecInfer-1bit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cfg_order = [
        "TQ-2bit",
        "TQ-3bit",
        "TQ-4bit",
        "RVQ-2bit",
        "RVQ-1bit",
        "VecInfer-2bit",
        "VecInfer-1bit",
    ]
    for model_id, results in per_model.items():
        stem = model_id.split("/")[-1]
        row = [stem]
        vals = {}
        for r in results:
            if r["name"] in cfg_order and _ok(r):
                vals[r["name"]] = r["key_compression"]
        best_compress = max(vals.values()) if vals else 0
        for cfg in cfg_order:
            v = vals.get(cfg)
            if v is None:
                row.append("—")
            elif v == best_compress and v > 1:
                row.append(f"**{v:.2f}×**")
            else:
                row.append(f"{v:.2f}×")
        lines.append("| " + " | ".join(row) + " |")

    # Throughput table
    lines += [
        "",
        "## Throughput (tok/s, higher is better)",
        "",
        "| Model | fp16 | TQ-2bit | TQ-3bit | TQ-4bit | RVQ-2bit | RVQ-1bit | VecInfer-2bit | VecInfer-1bit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_cfgs = ["fp16-baseline"] + cfg_order
    for model_id, results in per_model.items():
        stem = model_id.split("/")[-1]
        row = [stem]
        vals = {}
        for r in results:
            if r["name"] in all_cfgs and _ok(r):
                vals[r["name"]] = r["throughput_tok_s"]
        best_tput = max(vals.values()) if vals else 0
        for cfg in all_cfgs:
            v = vals.get(cfg)
            if v is None:
                row.append("—")
            elif v == best_tput and v > 0:
                row.append(f"**{v:.1f}**")
            else:
                row.append(f"{v:.1f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Key findings",
        "",
        "**VecInfer wins on raw compression**: 16× key compression at 1 bit/elem",
        "beats every other method on every model. RVQ-1bit (~7×) and TQ-2bit (~9×)",
        "are the next best.",
        "",
        "**TurboQuant / RVQ closely track fp16 throughput** on most models (within 5–10%).",
        "**VecInfer trades throughput for compression** — the nearest-centroid lookup",
        "runs in pure MLX without a fused Metal kernel. The paper's CUDA kernel fusion",
        "(Section 3.3, arxiv:2510.06175) is not portable to Apple Silicon.",
        "",
        "## When to pick which method",
        "",
        "| Goal | Best choice |",
        "|---|---|",
        "| Match fp16 throughput, modest compression | **RVQ-1bit** (~7×, ~100% fp16 throughput) |",
        "| Max compression, throughput tolerance | **VecInfer-1bit** (16×, ~50–90% fp16 throughput) |",
        "| Best key/throughput tradeoff at 2-bit | **TQ-2bit** (~9×) on dense models |",
        "| Long context where memory blows up | **VecInfer-1bit** — 16× cuts 4 GB cache to 256 MB |",
        "",
        "## Known gaps for VecInfer on MLX",
        "",
        "1. **head_dim must be power of 2** — Walsh-Hadamard requires 2^n head_dim.",
        "   Models like Qwen3-4B (head_dim=80) are incompatible.",
        "2. **head_dim=256 + small sub_dim → OOM** — chunked argmin allocates a large",
        "   `[chunk, n_centroids, sub_dim]` diff tensor. Use `key_sub_dim=8+` on large",
        "   head_dim models.",
        "3. **No fused dequant kernel** — dequantize on every `update_and_fetch` call,",
        "   eliminating the paper's main speedup over fp16.",
        "4. **MLA models (DeepSeek-V2)** — compressed latent KV at non-standard shape",
        "   breaks all per-cache wrappers.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


def main() -> None:
    figures_dir = Path("figures/vecinfer")
    if not figures_dir.exists():
        print("figures/vecinfer/ not found; run from repo root.")
        sys.exit(1)

    print("Loading per-model results...")
    per_model, meta = load_all_results(figures_dir)
    print(f"\n{len(per_model)} models included in rollup.\n")

    out_dir = figures_dir / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cross_model(per_model, out_dir / "cross_model_comparison.png")

    with open(out_dir / "results_all.json", "w") as f:
        json.dump({"per_model": per_model, "meta": meta}, f, indent=2)
    print(f"Saved: {out_dir / 'results_all.json'}")

    build_summary_md(per_model, meta, out_dir / "SUMMARY.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
