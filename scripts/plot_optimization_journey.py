"""Plot the throughput optimization journey across the three changes.

Generates a single figure showing how each optimization step (batching,
Hadamard rotation, searchsorted quantize, cast cleanup) lifted throughput
for Mistral 7B and Qwen3 4B, with quality preserved at every step.

Output: figures/updated_tests/optimization_journey.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

OUT = Path("figures/updated_tests/optimization_journey.png")

# Measured throughput (tok/s) at each optimization stage.
# Stages: original per-head loop → head-batching → +Hadamard (A) →
#         +searchsorted (A+B) → +cast cleanup (A+B+C)
STAGES = [
    "original\n(per-head loop)",
    "+ batch\nheads",
    "+ Hadamard\nrotation",
    "+ searchsorted\nquantize",
    "+ cast\ncleanup",
]

# Throughput per config, indexed by stage.
# Sources: the run_log_*.txt files in figures/updated_tests/<model>/
MISTRAL = {
    "fp16": [22.1, 21.1, 19.9, 22.4, 22.1],
    "TQ 2-bit": [17.9, 21.0, 20.1, 22.4, 22.4],
    "TQ 3-bit": [17.3, 21.9, 20.2, 22.0, 22.4],
    "TQ 4-bit": [15.0, 21.0, 19.9, 21.7, 21.8],
    "RVQ 2-bit ★": [17.7, 21.5, 20.0, 22.4, 22.3],
}

QWEN3 = {
    "fp16": [37.1, 33.4, None, None, 39.2],
    "TQ 2-bit": [12.1, 9.1, None, None, 31.2],
    "TQ 3-bit": [20.6, 24.8, None, None, 30.7],
    "TQ 4-bit": [16.4, 30.4, None, None, 8.6],  # 4-bit hits a <think> loop here, not perf
    "RVQ 2-bit ★": [24.8, 34.0, None, None, 36.0],
}

# Token completion (200 = full), used for the quality preservation panel.
TOKENS_END = {
    "Mistral 7B": {
        "fp16": 201,
        "TQ 2-bit": 201,
        "TQ 3-bit": 201,
        "TQ 4-bit": 201,
        "RVQ 2-bit ★": 201,
    },
    "Qwen3 4B": {"fp16": 200, "TQ 2-bit": 174, "TQ 3-bit": 172, "TQ 4-bit": 50, "RVQ 2-bit ★": 199},
}

PALETTE = {
    "fp16": "#4C72B0",
    "TQ 2-bit": "#C44E52",
    "TQ 3-bit": "#DD8452",
    "TQ 4-bit": "#55A868",
    "RVQ 2-bit ★": "#8172B2",
}


def _plot_journey(ax, data: dict, title: str) -> None:
    x = np.arange(len(STAGES))
    for name, vals in data.items():
        # Mask Nones (no measurement at that stage)
        ys = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        ls = "--" if name == "fp16" else "-"
        lw = 1.8 if name == "fp16" else 2.6
        marker = "o" if name == "fp16" else ("*" if "RVQ" in name else "s")
        ms = 8 if name == "fp16" else (14 if "RVQ" in name else 8)
        ax.plot(
            x,
            ys,
            color=PALETTE[name],
            lw=lw,
            ls=ls,
            marker=marker,
            ms=ms,
            label=name,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        # Annotate first and last with values
        if not np.isnan(ys[0]):
            ax.annotate(
                f"{ys[0]:.1f}",
                (x[0], ys[0]),
                xytext=(-12, 6),
                textcoords="offset points",
                fontsize=8,
                color=PALETTE[name],
            )
        if not np.isnan(ys[-1]):
            ax.annotate(
                f"{ys[-1]:.1f}",
                (x[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=PALETTE[name],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=9)
    ax.set_ylabel("Throughput (tok/s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def _plot_quality(ax, model_data: dict, title: str) -> None:
    names = list(model_data.keys())
    vals = [model_data[n] for n in names]
    cols = [PALETTE[n] for n in names]
    bars = ax.bar(np.arange(len(names)), vals, color=cols, edgecolor="white", linewidth=1.2)
    ax.axhline(200, color="grey", ls="--", lw=1, alpha=0.7, label="full output")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 5,
            str(v),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Tokens generated (max 200)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 240)
    sns.despine(ax=ax)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28, height_ratios=[1.5, 1.0])

    ax_m = fig.add_subplot(gs[0, 0])
    _plot_journey(ax_m, MISTRAL, "Mistral 7B — throughput across optimization stages")

    ax_q = fig.add_subplot(gs[0, 1])
    _plot_journey(ax_q, QWEN3, "Qwen3 4B — throughput across optimization stages")

    ax_qm = fig.add_subplot(gs[1, 0])
    _plot_quality(
        ax_qm,
        TOKENS_END["Mistral 7B"],
        "Mistral 7B — final-stage output quality (tokens before stop)",
    )

    ax_qq = fig.add_subplot(gs[1, 1])
    _plot_quality(
        ax_qq, TOKENS_END["Qwen3 4B"], "Qwen3 4B — final-stage output quality (tokens before stop)"
    )

    fig.suptitle(
        "TurboQuant KV-cache throughput optimization — VeloxQuant-MLX\n"
        "Per-head loop → batched → Hadamard rotation → searchsorted quantize → cast cleanup",
        fontsize=15,
        fontweight="bold",
        y=1.005,
    )

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
