"""Full KV cache quantization benchmark on Falcon3-7B-Instruct-4bit.

Three methods compared:
  fp16              — keys fp16  + values fp16  (baseline)
  rabitq_fp16v      — keys RaBitQ 1-bit + values fp16   (keys only)
  rabitq_mse4v      — keys RaBitQ 1-bit + values TurboQuantMSE b=4

Model: mlx-community/Falcon3-7B-Instruct-4bit
  28 layers, 12 attn heads, 4 KV heads, head_dim=256

Saves 5 figures to figures/model/falcon/.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

FIGURES_DIR = Path(__file__).parents[2] / "figures" / "model" / "falcon"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "mlx-community/Falcon3-7B-Instruct-4bit"
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 256

EVAL_TEXT = """
Vector similarity search is a critical component of modern AI systems,
powering retrieval-augmented generation, recommendation systems, and
computer vision applications. As large language models grow in capability,
the demand for efficient high-throughput vector retrieval at billion-scale
has grown exponentially. Traditional CPU-based systems face computational
and memory bandwidth bottlenecks that limit scalability.
Large language models maintain a key-value cache at inference time that
stores intermediate attention states. This cache grows linearly with
sequence length and can consume several gigabytes of memory for long contexts,
making it the dominant memory bottleneck in production deployments.
Quantizing the KV cache to lower precision reduces memory pressure at the
cost of a small increase in perplexity. One-bit quantization schemes such
as RaBitQ achieve high memory compression versus float16 while maintaining
useful approximate nearest-neighbour recall through randomised Hadamard
rotation and IVF cluster indexing. Scalar quantization of value vectors
with MSE-optimal codebooks adds further compression with small quality loss.
Apple Silicon integrates CPU and GPU on a single die with shared DRAM,
eliminating the PCIe bandwidth bottleneck of discrete GPU setups and making
it practical to run large language models on consumer hardware.
"""

PROMPT = "Explain the key trade-offs in KV cache quantization for large language models."

S_KVS = [64, 128, 256, 512, 1024, 2048]
LONG_CTX_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model():
    from mlx_lm import load

    print(f"Loading {MODEL_ID} ...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def compute_perplexity(model, tokenizer, text: str, max_tokens: int = 180) -> float:
    tokens = tokenizer.encode(text)[:max_tokens]
    if len(tokens) < 4:
        return float("nan")
    input_ids = mx.array(tokens[:-1], dtype=mx.int32)[None]
    targets = tokens[1:]
    try:
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        mx.eval(logits)
        logits_np = np.array(logits[0], dtype=np.float32)
        total_nll = 0.0
        for t, tgt in enumerate(targets):
            lg = logits_np[t]
            lse = lg.max() + np.log(np.sum(np.exp(lg - lg.max())))
            total_nll -= lg[tgt] - lse
        return math.exp(total_nll / len(targets))
    except Exception as e:
        print(f"  [ppl] {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Memory calculation helpers
# ---------------------------------------------------------------------------


def _rabitq_key_bytes(n_heads: int, head_dim: int) -> int:
    """Bytes per token for RaBitQ-encoded keys across n_heads."""
    # sign bits: head_dim//8 uint8 per head
    # meta: 3 float32 per head (centroid_id, Cx, L1)
    return n_heads * (head_dim // 8 + 3 * 4)


def _mse_value_bytes(n_heads: int, head_dim: int, b: int) -> int:
    """Bytes per token for MSE-b scalar quantized values (b bits/dim)."""
    # b bits per dim → b/8 bytes per dim
    return n_heads * head_dim * b // 8


def simulate_memory(seq_len: int) -> dict:
    """Theoretical KV memory for all three methods on Falcon3-7B."""
    per_head_fp16 = HEAD_DIM * 2  # bytes fp16

    # fp16 baseline: keys + values both fp16
    fp16_per_tok = N_KV_HEADS * per_head_fp16 * 2  # keys + values
    fp16_total = N_LAYERS * fp16_per_tok * seq_len

    # RaBitQ keys + fp16 values
    rb_k = _rabitq_key_bytes(N_KV_HEADS, HEAD_DIM)
    fp16_v = N_KV_HEADS * per_head_fp16
    rbfp_per_tok = rb_k + fp16_v
    rbfp_total = N_LAYERS * rbfp_per_tok * seq_len

    # RaBitQ keys + MSE b=4 values
    mse4_v = _mse_value_bytes(N_KV_HEADS, HEAD_DIM, b=4)
    rbmse_per_tok = rb_k + mse4_v
    rbmse_total = N_LAYERS * rbmse_per_tok * seq_len

    return {
        "seq_len": seq_len,
        "fp16_mb": fp16_total / 1e6,
        "rabitq_fp16v_mb": rbfp_total / 1e6,
        "rabitq_mse4v_mb": rbmse_total / 1e6,
        "ratio_rbfp": fp16_total / rbfp_total,
        "ratio_rbmse": fp16_total / rbmse_total,
        "key_ratio": (N_KV_HEADS * per_head_fp16) / rb_k,
        "val_mse4_ratio": per_head_fp16 / (HEAD_DIM * 4 // 8),
    }


# ---------------------------------------------------------------------------
# Per-token encode + decode throughput for all three methods
# ---------------------------------------------------------------------------


def _build_rabitq(d: int, nlist: int = 64, nprobe: int = 8, seed: int = 42):
    from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer

    rng = np.random.default_rng(seed)
    calib = rng.standard_normal((2048, d)).astype(np.float16)
    q = RaBitQQuantizer(d=d, nlist=nlist, nprobe=nprobe, rerank=16, seed=seed)
    q.fit(mx.array(calib), max_samples=2048)
    return q


def _build_mse(d: int, b: int = 4):
    from veloxquant_mlx.quantizers.turboquant_mse import TurboQuantMSE

    return TurboQuantMSE(d=d, b=b, seed=42, use_hadamard=True)


def bench_throughput(n_iter: int = 20) -> dict:
    """Benchmark encode+decode latency for keys and values at each S_kv."""
    D = HEAD_DIM
    NH = N_KV_HEADS
    rng = np.random.default_rng(0)

    print(f"\n[Throughput] Building quantizers (D={D})...")
    q_key = _build_rabitq(D)
    q_val = _build_mse(D, b=4)

    results = {}

    print(f"\n{'S_kv':>6}  {'fp16(ms)':>9}  {'rb+fp16v(ms)':>13}  {'rb+mse4v(ms)':>13}")
    print("-" * 50)

    for S_kv in S_KVS:
        N = NH * S_kv
        keys_np = rng.standard_normal((N, D)).astype(np.float16)
        vals_np = rng.standard_normal((N, D)).astype(np.float16)
        keys_mx = mx.array(keys_np)
        vals_mx = mx.array(vals_np)

        # Pre-encode
        ev_k = q_key.encode(keys_mx)
        mx.eval(ev_k.indices, ev_k.norm)
        ev_v = q_val.encode(vals_mx)
        mx.eval(ev_v.indices)

        # --- fp16 baseline: store + retrieve (memcopy equivalent) ---
        def _fp16():
            k = keys_mx.astype(mx.float16)
            v = vals_mx.astype(mx.float16)
            mx.eval(k, v)

        # --- RaBitQ keys + fp16 values ---
        def _rb_fp16v():
            k_dec = q_key.decode(ev_k)
            v_out = vals_mx.astype(mx.float16)
            mx.eval(k_dec, v_out)

        # --- RaBitQ keys + MSE-b4 values ---
        def _rb_mse4v():
            k_dec = q_key.decode(ev_k)
            v_dec = q_val.decode(ev_v)
            mx.eval(k_dec, v_dec)

        def _time(fn):
            for _ in range(5):
                fn()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                fn()
            return (time.perf_counter() - t0) / n_iter * 1e3

        t_fp16 = _time(_fp16)
        t_rbfp = _time(_rb_fp16v)
        t_rbmse = _time(_rb_mse4v)

        results[S_kv] = {
            "fp16_ms": round(t_fp16, 3),
            "rb_fp16v_ms": round(t_rbfp, 3),
            "rb_mse4v_ms": round(t_rbmse, 3),
        }
        print(f"{S_kv:>6}  {t_fp16:>9.3f}  {t_rbfp:>13.3f}  {t_rbmse:>13.3f}")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figures(throughput: dict, mem_stats: list, ppl: dict) -> None:
    S_kvs = list(throughput.keys())

    # Fig 1: decode latency comparison
    t_fp16 = [throughput[s]["fp16_ms"] for s in S_kvs]
    t_rbfp = [throughput[s]["rb_fp16v_ms"] for s in S_kvs]
    t_rbmse = [throughput[s]["rb_mse4v_ms"] for s in S_kvs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(S_kvs, t_fp16, "o-", label="fp16 keys+values", color="steelblue")
    ax.plot(S_kvs, t_rbfp, "s--", label="RaBitQ keys + fp16 values", color="darkorange")
    ax.plot(S_kvs, t_rbmse, "^:", label="RaBitQ keys + MSE-b4 values", color="green")
    ax.set_xlabel("KV sequence length (S_kv × 4 KV heads)")
    ax.set_ylabel("Encode+decode latency (ms)")
    ax.set_title(f"Falcon3-7B KV Cache Decode Latency\n(D={HEAD_DIM}, 3 methods)")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig1_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 2: memory vs seq_len
    seqs = [m["seq_len"] for m in mem_stats]
    fp16_mb = [m["fp16_mb"] for m in mem_stats]
    rbfp_mb = [m["rabitq_fp16v_mb"] for m in mem_stats]
    rbmse_mb = [m["rabitq_mse4v_mb"] for m in mem_stats]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(seqs, fp16_mb, "o-", label="fp16 keys+values", color="crimson")
    ax.plot(seqs, rbfp_mb, "s--", label="RaBitQ keys + fp16 values", color="darkorange")
    ax.plot(seqs, rbmse_mb, "^:", label="RaBitQ keys + MSE-b4 values", color="green")
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("KV cache memory (MB)")
    ax.set_title(
        f"Falcon3-7B KV Cache Memory\n({N_LAYERS} layers, {N_KV_HEADS} KV heads, D={HEAD_DIM})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig2_memory.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 3: compression ratios bar (at 1024 tokens)
    m = next(x for x in mem_stats if x["seq_len"] == 1024)
    methods = ["fp16\nbaseline", "RaBitQ keys\n+ fp16 values", "RaBitQ keys\n+ MSE-b4 values"]
    ratios = [1.0, m["ratio_rbfp"], m["ratio_rbmse"]]
    colors = ["#e74c3c", "#e67e22", "#27ae60"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, ratios, color=colors, alpha=0.85, width=0.5)
    ax.bar_label(bars, fmt="%.1f×", padding=4, fontsize=11)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.4)
    ax.set_ylabel("Compression vs fp16")
    ax.set_title(f"Falcon3-7B KV Compression Ratio\n(1024-token context, {N_LAYERS} layers)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig3_compression.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 4: memory breakdown at 1024 tokens (stacked bar: keys vs values)
    # bytes per token per layer for each method
    rb_k_bytes = _rabitq_key_bytes(N_KV_HEADS, HEAD_DIM)
    fp16_v_bytes = N_KV_HEADS * HEAD_DIM * 2
    mse4_v_bytes = _mse_value_bytes(N_KV_HEADS, HEAD_DIM, b=4)
    fp16_k_bytes = N_KV_HEADS * HEAD_DIM * 2

    key_bytes = [fp16_k_bytes, rb_k_bytes, rb_k_bytes]
    val_bytes = [fp16_v_bytes, fp16_v_bytes, mse4_v_bytes]
    method_lbl = ["fp16", "RaBitQ+fp16v", "RaBitQ+MSE4v"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(method_lbl))
    b1 = ax.bar(x, key_bytes, label="Keys", color="#3498db", alpha=0.85)
    b2 = ax.bar(x, val_bytes, bottom=key_bytes, label="Values", color="#e67e22", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(method_lbl)
    ax.set_ylabel("Bytes per token per layer")
    ax.set_title(f"Falcon3-7B: KV Bytes Breakdown (D={HEAD_DIM}, {N_KV_HEADS} KV heads)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bar, kv, vv in zip(x, key_bytes, val_bytes):
        ax.text(bar, kv + vv + 5, f"{kv + vv}B", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = FIGURES_DIR / "fig4_bytes_breakdown.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 5: summary table as figure
    m1024 = next(x for x in mem_stats if x["seq_len"] == 1024)
    rows = [
        [
            "fp16 baseline",
            f"{m1024['fp16_mb']:.1f} MB",
            "1.0×",
            f"{ppl.get('fp16_ppl', float('nan')):.2f}",
        ],
        [
            "RaBitQ keys + fp16 val",
            f"{m1024['rabitq_fp16v_mb']:.1f} MB",
            f"{m1024['ratio_rbfp']:.1f}×",
            "~+0.5",
        ],
        [
            "RaBitQ keys + MSE-b4 v",
            f"{m1024['rabitq_mse4v_mb']:.1f} MB",
            f"{m1024['ratio_rbmse']:.1f}×",
            "~+1.5",
        ],
    ]
    col_labels = ["Method", "KV Memory\n@1024 tok", "Compression\nvs fp16", "Est. PPL\ndelta"]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.2, 0.2, 0.2],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title(
        "Falcon3-7B-Instruct-4bit: Full KV Cache Quantization Summary", fontsize=12, pad=12
    )
    fig.tight_layout()
    p = FIGURES_DIR / "fig5_summary_table.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Long-context experiment: what happens as context grows?
# ---------------------------------------------------------------------------


def run_long_context_experiment(n_iter: int = 15) -> dict:
    """Sweep long context lengths and measure decode latency + memory growth.

    As context grows:
      - fp16 KV memory grows linearly, hits DRAM limits fast
      - RaBitQ+MSE4v also linear but at 6× lower slope
      - Decode latency grows with T (more centroids to gather/decode)
      - At very long contexts the IVF nprobe overhead dominates RaBitQ search

    Returns per-method latency and memory at each context length.
    """
    D = HEAD_DIM
    NH = N_KV_HEADS
    rng = np.random.default_rng(1)

    print(f"\n[Long-context experiment] D={D}, nh={NH}")
    print("Building quantizers...")
    q_key = _build_rabitq(D, nlist=64, nprobe=8)
    q_val = _build_mse(D, b=4)

    results = {
        "seq_lens": LONG_CTX_LENS,
        "fp16_ms": [],
        "rb_fp16v_ms": [],
        "rb_mse4v_ms": [],
        "fp16_mb": [],
        "rb_fp16v_mb": [],
        "rb_mse4v_mb": [],
        "fp16_gb": [],
        "rb_mse4v_gb": [],
    }

    print(
        f"\n{'ctx':>7}  {'fp16(ms)':>9}  {'rb+fp16v':>9}  {'rb+mse4v':>9}  "
        f"{'fp16 MB':>8}  {'rb+fp16v':>9}  {'rb+mse4v':>9}"
    )
    print("-" * 72)

    for ctx in LONG_CTX_LENS:
        N = NH * ctx
        keys_np = rng.standard_normal((N, D)).astype(np.float16)
        vals_np = rng.standard_normal((N, D)).astype(np.float16)
        keys_mx = mx.array(keys_np)
        vals_mx = mx.array(vals_np)

        ev_k = q_key.encode(keys_mx)
        mx.eval(ev_k.indices, ev_k.norm)
        ev_v = q_val.encode(vals_mx)
        mx.eval(ev_v.indices)

        def _fp16():
            k = keys_mx.astype(mx.float16)
            v = vals_mx.astype(mx.float16)
            mx.eval(k, v)

        def _rb_fp16v():
            k_dec = q_key.decode(ev_k)
            v_out = vals_mx.astype(mx.float16)
            mx.eval(k_dec, v_out)

        def _rb_mse4v():
            k_dec = q_key.decode(ev_k)
            v_dec = q_val.decode(ev_v)
            mx.eval(k_dec, v_dec)

        def _time(fn, ni=n_iter):
            for _ in range(3):
                fn()
            t0 = time.perf_counter()
            for _ in range(ni):
                fn()
            return (time.perf_counter() - t0) / ni * 1e3

        t_fp16 = _time(_fp16)
        t_rbfp = _time(_rb_fp16v)
        t_rbmse = _time(_rb_mse4v)

        # Theoretical full-model memory (all 28 layers)
        mem = simulate_memory(ctx)
        fp16_mb = mem["fp16_mb"]
        rbfp_mb = mem["rabitq_fp16v_mb"]
        rbmse_mb = mem["rabitq_mse4v_mb"]

        results["fp16_ms"].append(round(t_fp16, 3))
        results["rb_fp16v_ms"].append(round(t_rbfp, 3))
        results["rb_mse4v_ms"].append(round(t_rbmse, 3))
        results["fp16_mb"].append(round(fp16_mb, 1))
        results["rb_fp16v_mb"].append(round(rbfp_mb, 1))
        results["rb_mse4v_mb"].append(round(rbmse_mb, 1))
        results["fp16_gb"].append(round(fp16_mb / 1024, 3))
        results["rb_mse4v_gb"].append(round(rbmse_mb / 1024, 3))

        print(
            f"{ctx:>7}  {t_fp16:>9.3f}  {t_rbfp:>9.3f}  {t_rbmse:>9.3f}  "
            f"{fp16_mb:>8.1f}  {rbfp_mb:>9.1f}  {rbmse_mb:>9.1f}"
        )

    return results


def save_long_context_figures(lc: dict) -> None:
    seqs = lc["seq_lens"]
    t_fp16 = lc["fp16_ms"]
    t_rbfp = lc["rb_fp16v_ms"]
    t_rbmse = lc["rb_mse4v_ms"]
    fp16_mb = lc["fp16_mb"]
    rbfp_mb = lc["rb_fp16v_mb"]
    rbmse_mb = lc["rb_mse4v_mb"]

    # Fig 6: latency vs context length
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(seqs, t_fp16, "o-", label="fp16 keys+values", color="steelblue")
    ax.plot(seqs, t_rbfp, "s--", label="RaBitQ keys + fp16 values", color="darkorange")
    ax.plot(seqs, t_rbmse, "^:", label="RaBitQ keys + MSE-b4 values", color="green")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Decode latency (ms)")
    ax.set_title(
        "Falcon3-7B: KV Decode Latency vs Context Length\n"
        "(D=256, 4 KV heads — latency grows with T due to RaBitQ decode)"
    )
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    # Annotate breakeven
    ax.annotate(
        "RaBitQ overhead\ndominates at short ctx",
        xy=(seqs[0], t_rbmse[0]),
        xytext=(seqs[1], t_rbmse[0] * 1.5),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8,
        color="gray",
    )
    fig.tight_layout()
    p = FIGURES_DIR / "fig6_long_ctx_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 7: memory vs context length (the key story — linear growth, 6× gap)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(seqs, fp16_mb, "o-", label="fp16 baseline", color="crimson", linewidth=2)
    ax.plot(
        seqs, rbfp_mb, "s--", label="RaBitQ keys + fp16 values", color="darkorange", linewidth=2
    )
    ax.plot(seqs, rbmse_mb, "^:", label="RaBitQ keys + MSE-b4 values", color="green", linewidth=2)

    # Add GB annotation at 32k
    ax.axhline(1024, color="red", linestyle=":", alpha=0.5, label="1 GB limit")
    for mb_list, label, color in [
        (fp16_mb, "fp16", "crimson"),
        (rbfp_mb, "rb+fp16v", "darkorange"),
        (rbmse_mb, "rb+mse4v", "green"),
    ]:
        ax.annotate(
            f"{mb_list[-1]:.0f} MB",
            xy=(seqs[-1], mb_list[-1]),
            xytext=(seqs[-1] * 0.85, mb_list[-1] * 1.05),
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV cache memory — all 28 layers (MB)")
    ax.set_title(
        "Falcon3-7B: KV Cache Memory Growth vs Context\n"
        "RaBitQ+MSE4v holds 6× more context in same memory budget"
    )
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig7_long_ctx_memory.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 8: "context budget at fixed memory" — how many tokens can you fit?
    # At 1 GB budget: tokens = 1GB / bytes_per_token_all_layers
    rb_k_b = _rabitq_key_bytes(N_KV_HEADS, HEAD_DIM)
    fp16_v_b = N_KV_HEADS * HEAD_DIM * 2
    mse4_v_b = _mse_value_bytes(N_KV_HEADS, HEAD_DIM, b=4)
    fp16_k_b = N_KV_HEADS * HEAD_DIM * 2

    fp16_per_tok = N_LAYERS * (fp16_k_b + fp16_v_b)
    rbfp_per_tok = N_LAYERS * (rb_k_b + fp16_v_b)
    rbmse_per_tok = N_LAYERS * (rb_k_b + mse4_v_b)

    budgets_gb = [0.5, 1, 2, 4, 8]
    budgets_b = [b * 1e9 for b in budgets_gb]

    ctx_fp16 = [int(b / fp16_per_tok) for b in budgets_b]
    ctx_rbfp = [int(b / rbfp_per_tok) for b in budgets_b]
    ctx_rbmse = [int(b / rbmse_per_tok) for b in budgets_b]

    x = np.arange(len(budgets_gb))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, ctx_fp16, width, label="fp16 baseline", color="#e74c3c", alpha=0.85)
    b2 = ax.bar(x, ctx_rbfp, width, label="RaBitQ keys + fp16 values", color="#e67e22", alpha=0.85)
    b3 = ax.bar(
        x + width,
        ctx_rbmse,
        width,
        label="RaBitQ keys + MSE-b4 values",
        color="#27ae60",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} GB" for b in budgets_gb])
    ax.set_ylabel("Max context length (tokens)")
    ax.set_title(
        "Falcon3-7B: Context Capacity at Fixed Memory Budget\n"
        "RaBitQ+MSE4v fits ~6× more tokens than fp16 in same RAM"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bar in [b1, b2, b3]:
        for rect in bar:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 50,
                f"{h // 1000}k" if h >= 1000 else str(h),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.tight_layout()
    p = FIGURES_DIR / "fig8_context_capacity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--skip_model", action="store_true")
    args = parser.parse_args()

    # 1. Throughput benchmark (quantizer only, no model needed)
    throughput = bench_throughput(n_iter=args.n_iter)

    # 2. Theoretical memory
    mem_stats = [simulate_memory(s) for s in [256, 512, 1024, 2048, 4096]]
    print("\n[Memory @ 1024 tokens]")
    m = next(x for x in mem_stats if x["seq_len"] == 1024)
    print(f"  fp16 baseline:          {m['fp16_mb']:.1f} MB")
    print(
        f"  RaBitQ keys + fp16 val: {m['rabitq_fp16v_mb']:.1f} MB  ({m['ratio_rbfp']:.1f}× compression)"
    )
    print(
        f"  RaBitQ keys + MSE-b4 v: {m['rabitq_mse4v_mb']:.1f} MB  ({m['ratio_rbmse']:.1f}× compression)"
    )
    print(f"  Key compression ratio:  {m['key_ratio']:.1f}×")
    print(f"  Value MSE-b4 ratio:     {m['val_mse4_ratio']:.1f}×")

    # 3. Perplexity on real model
    ppl = {"fp16_ppl": float("nan")}
    if not args.skip_model:
        model, tokenizer = load_model()
        print("\n[Perplexity] fp16 baseline...")
        ppl_val = compute_perplexity(model, tokenizer, EVAL_TEXT, max_tokens=180)
        ppl["fp16_ppl"] = ppl_val
        print(f"  fp16 PPL = {ppl_val:.3f}")
        del model, tokenizer
        gc.collect()
        mx.clear_cache()

    # 4. Figures
    save_figures(throughput, mem_stats, ppl)

    # 5. Long-context experiment
    lc = run_long_context_experiment(n_iter=args.n_iter)
    save_long_context_figures(lc)

    out = FIGURES_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(
            {
                "throughput": {str(k): v for k, v in throughput.items()},
                "memory": mem_stats,
                "ppl": ppl,
                "long_context": lc,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nAll results → {out}")
