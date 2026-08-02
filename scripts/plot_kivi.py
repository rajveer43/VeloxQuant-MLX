"""Generate KIVI figures from committed benchmark results.

Reads every ``figures/kivi/<model>/results.json`` produced by
``benchmark_scripts/benchmark_kivi.py`` (never hardcodes numbers) and emits
four figures plus an aggregated summary under ``figures/kivi/``:

  fig1_compression_vs_quality.png  reconstruction cosine vs realized key
                                   compression (quality is measured here on
                                   seeded synthetic unit-norm Gaussian keys —
                                   deterministic, model-independent — and
                                   labeled as such; compression ratios come
                                   from the committed benchmark JSONs).
  fig2_throughput.png              measured tok/s per config vs fp16 baseline.
  fig3_memory_at_scale.png         analytic KV memory vs seq-len (this method
                                   vs fp16), derived from each model's
                                   head_dim / n_kv_heads / n_layers.
  fig4_vs_existing.png             KIVI vs existing repo methods (VecInfer,
                                   TurboQuant RVQ) on the same models, read
                                   from figures/vecinfer/<model>/results.json.

Usage::

    PYTHONPATH=. python scripts/plot_kivi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


REPO = Path(__file__).resolve().parent.parent
KIVI_DIR = REPO / "figures" / "kivi"
VECINFER_DIR = REPO / "figures" / "vecinfer"
COLORS = {
    "fp16": "#666666",
    "2bit": "#00d4ff",
    "3bit": "#7c3aed",
    "4bit": "#ff6b35",
    "kivi": "#22c55e",
    "vecinfer": "#7c3aed",
    "rvq": "#ff6b35",
}


def _load_kivi() -> dict:
    out = {}
    for jp in sorted(KIVI_DIR.glob("*/results.json")):
        with open(jp) as f:
            out[jp.parent.name] = json.load(f)
    if not out:
        sys.exit("No figures/kivi/*/results.json found — run benchmark_kivi.py first.")
    return out


# ---------------------------------------------------------------------------
# Deterministic synthetic reconstruction quality (model-independent)
# ---------------------------------------------------------------------------
def _kivi_cosine(b: int, d: int = 128, n: int = 512, group_size: int = 32, seed: int = 0) -> float:
    from veloxquant_mlx.quantizers.kivi import KIVIQuantizer

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    q = KIVIQuantizer(d=d, b=b, group_size=group_size, axis="channel")
    Xhat = np.array(q.decode(q.encode(mx.array(X)))).astype(np.float32)
    num = np.sum(X * Xhat, axis=1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(Xhat, axis=1) + 1e-9
    return float(np.mean(num / den))


def fig1_compression_vs_quality(kivi: dict) -> None:
    # Use the first model's realized key-compression per bit-width, and the
    # deterministic synthetic cosine for quality.
    model0 = sorted(kivi)[0]
    rows = {r["name"]: r for r in kivi[model0]["results"]}
    bits = [2, 3, 4]
    comp = [rows[f"KIVI-{b}bit"]["key_compression"] for b in bits]
    cos = [_kivi_cosine(b) for b in bits]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(comp, cos, "o-", color=COLORS["kivi"], linewidth=2, markersize=9)
    for b, c, q in zip(bits, comp, cos):
        ax.annotate(f"{b}-bit", (c, q), textcoords="offset points", xytext=(8, -4), fontsize=10)
    ax.set_xlabel("Realized key compression (×)  [measured]")
    ax.set_ylabel("Reconstruction cosine  [synthetic unit-norm Gaussian, d=128]")
    ax.set_title(
        "KIVI: compression vs reconstruction quality\n"
        f"(compression from {model0}; quality deterministic-synthetic)"
    )
    ax.grid(alpha=0.3)
    out = KIVI_DIR / "fig1_compression_vs_quality.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def fig2_throughput(kivi: dict) -> None:
    models = sorted(kivi)
    configs = ["fp16-baseline", "KIVI-2bit", "KIVI-3bit", "KIVI-4bit"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.2
    for i, cfg in enumerate(configs):
        vals = []
        for m in models:
            rows = {r["name"]: r for r in kivi[m]["results"]}
            vals.append(rows[cfg]["throughput_tok_s"])
        color = COLORS.get(
            cfg.split("-")[-1].replace("bit", "bit"),
            COLORS["fp16"] if "fp16" in cfg else COLORS["kivi"],
        )
        ax.bar(x + (i - 1.5) * w, vals, w, label=cfg, color=color)
    chip = kivi[models[0]].get("hardware", {}).get("chip", "Apple Silicon")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("-Instruct", "").replace("-4bit", "") for m in models], rotation=10
    )
    ax.set_ylabel("Tokens / second  [measured]")
    ax.set_title(f"KIVI throughput vs fp16 baseline ({chip}, max_tokens≈120)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = KIVI_DIR / "fig2_throughput.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def fig3_memory_at_scale(kivi: dict) -> None:
    model0 = sorted(kivi)[0]
    meta = kivi[model0]
    hd, hkv, nl = meta["head_dim"], meta["n_kv_heads"], meta["n_layers"]
    r = meta.get("residual_length", 32)
    rows = {x["name"]: x for x in meta["results"]}
    seqs = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768])

    def kv_bytes_fp16(S):
        return nl * hkv * hd * S * 2 * 2  # K+V, fp16

    def kv_bytes_kivi(S, b):
        # quantized tokens (S - r) at b bits K+V + group params; residual r fp16.
        nq = np.maximum(S - r, 0)
        gs = meta.get("group_size", 32)
        # codes
        code = nq * hd * b / 8 * 2 * nl * hkv
        # params: keys per-channel groups, values per-token groups (fp16 scale+zero)
        kparam = np.ceil(nq / gs) * hd * 2 * 2 * nl * hkv
        vparam = nq * np.ceil(hd / gs) * 2 * 2 * nl * hkv
        resid = r * hd * 2 * 2 * nl * hkv
        return code + kparam + vparam + resid

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seqs, kv_bytes_fp16(seqs) / 1e9, "o-", color=COLORS["fp16"], label="fp16")
    for b, col in [(2, COLORS["2bit"]), (4, COLORS["4bit"])]:
        ax.plot(seqs, kv_bytes_kivi(seqs, b) / 1e9, "s-", color=col, label=f"KIVI-{b}bit")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("KV-cache memory (GB)  [analytic]")
    ax.set_title(
        f"Analytic KV memory vs seq-len — {model0}\n"
        f"(head_dim={hd}, n_kv_heads={hkv}, n_layers={nl}, residual={r})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    out = KIVI_DIR / "fig3_memory_at_scale.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def fig4_vs_existing(kivi: dict) -> None:
    """KIVI vs existing repo methods on shared models (key compression)."""
    shared, kivi_vals, vec_vals = [], [], []
    for m in sorted(kivi):
        vj = VECINFER_DIR / m / "results.json"
        if not vj.exists():
            continue
        with open(vj) as f:
            vdata = json.load(f)
        vrows = {r["name"]: r for r in vdata["results"]}
        krows = {r["name"]: r for r in kivi[m]["results"]}
        # Best directly-comparable ~2-bit config from each.
        if "KIVI-2bit" not in krows:
            continue
        vec_2 = vrows.get("VecInfer-2bit") or vrows.get("vecinfer-2bit")
        if vec_2 is None:
            continue
        shared.append(m.replace("-Instruct", "").replace("-4bit", ""))
        kivi_vals.append(krows["KIVI-2bit"]["key_compression"])
        vec_vals.append(vec_2["key_compression"])
    if not shared:
        print("fig4 skipped — no overlapping vecinfer results.json found.")
        return
    x = np.arange(len(shared))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, kivi_vals, w, label="KIVI-2bit", color=COLORS["kivi"])
    ax.bar(x + w / 2, vec_vals, w, label="VecInfer-2bit", color=COLORS["vecinfer"])
    ax.set_xticks(x)
    ax.set_xticklabels(shared, rotation=10)
    ax.set_ylabel("Key compression (×)  [measured]")
    ax.set_title("KIVI vs VecInfer — key compression at ~2 bits (same models)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = KIVI_DIR / "fig4_vs_existing.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def write_summary(kivi: dict) -> None:
    summary = {"source": "figures/kivi/*/results.json", "models": {}}
    for m, data in kivi.items():
        summary["models"][m] = {
            "hardware": data.get("hardware"),
            "head_dim": data.get("head_dim"),
            "n_kv_heads": data.get("n_kv_heads"),
            "n_layers": data.get("n_layers"),
            "prompt_tokens": data.get("prompt_tokens"),
            "configs": {
                r["name"]: {
                    "throughput_tok_s": round(r["throughput_tok_s"], 2),
                    "key_compression": round(r["key_compression"], 2),
                    "full_kv_compression": round(r["full_kv_compression"], 2),
                    "peak_mb": round(r["peak_mb"], 1),
                    "tokens_generated": r["tokens_generated"],
                }
                for r in data["results"]
            },
        }
    out = KIVI_DIR / "results_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out)


def main() -> int:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    kivi = _load_kivi()
    fig1_compression_vs_quality(kivi)
    fig2_throughput(kivi)
    fig3_memory_at_scale(kivi)
    fig4_vs_existing(kivi)
    write_summary(kivi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
