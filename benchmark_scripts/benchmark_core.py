"""
Shared benchmark core for dense-attention models.
Used by benchmark_mistral7b.py, benchmark_qwen3_4b.py, benchmark_qwen3_8b.py.
"""

import math
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlx.core as mx
import mlx_lm
import numpy as np
import seaborn as sns
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ

PALETTE = {
    "fp16": "#4C72B0",
    "2bit": "#C44E52",
    "3bit": "#DD8452",
    "4bit": "#55A868",
    "accent": "#8172B2",
}
PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)
MAX_TOKENS = 200


# ── TurboQuant KV cache wrapper ────────────────────────────────────────────────
class TurboQuantMLXKVCache(_MLXKVCache):
    """Compresses keys via TurboQuantProd with per-vector normalization."""

    def __init__(self, n_kv_heads: int, head_dim: int, bits: int = 4, seed: int = 42) -> None:
        super().__init__()
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._bits = bits
        m = min(head_dim, 64)
        # Single shared quantizer across all heads. Folding the head axis
        # into the batch axis turns 256 small kernel launches into 1 big one.
        # use_hadamard=True swaps O(d²) QR matmul for O(d log d) Metal-native FFT.
        self._quantizer = TurboQuantProd(d=head_dim, b=bits, m=m, seed=seed, use_hadamard=True)
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0

    def update_and_fetch(self, keys, values):
        B, H, S, D = keys.shape
        # Match the model's native dtype (Qwen2-VL-bf16 uses bfloat16; text
        # models commonly fp16). Casting through fp16 on a bf16 model both
        # discards the wider exponent range — image-patch keys can have norms
        # well above the fp16 sweet spot — and forces extra Metal casts.
        kdtype = keys.dtype
        k_flat = keys.reshape(-1, D)
        norms = mx.linalg.norm(k_flat.astype(mx.float32), axis=-1, keepdims=True).astype(kdtype)
        safe = mx.maximum(norms, mx.array(1e-4, dtype=kdtype))
        k_unit = (k_flat / safe).astype(mx.float16)

        ev = self._quantizer.encode(k_unit)
        k_hat_u = self._quantizer.decode(ev)
        k_dequant = (k_hat_u.astype(kdtype) * safe).reshape(B, H, S, D)

        b_mse = max(self._bits - 1, 1)
        m_eff = self._quantizer._m_eff
        per_tok = (math.ceil(self._head_dim * b_mse / 8) + math.ceil(m_eff / 8) + 2 + 2) * H * B
        self._key_bytes_compressed += per_tok * S
        self._key_bytes_fp16 += H * B * S * self._head_dim * 2
        return super().update_and_fetch(k_dequant, values)

    @property
    def compressed_key_bytes(self) -> int:
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        return self._key_bytes_fp16


# ── TurboQuantRVQ KV cache wrapper ─────────────────────────────────────────────
class TurboQuantRVQMLXKVCache(_MLXKVCache):
    """Compresses keys via TurboQuantRVQ (two-pass residual quantization)."""

    def __init__(self, n_kv_heads: int, head_dim: int, bits: int = 2, seed: int = 42) -> None:
        super().__init__()
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._bits = bits
        self._quantizer = TurboQuantRVQ(d=head_dim, b=bits, seed=seed, use_hadamard=True)
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0

    def update_and_fetch(self, keys, values):
        B, H, S, D = keys.shape
        # Match the model's native dtype (Qwen2-VL-bf16 uses bfloat16; text
        # models commonly fp16). See TurboQuantMLXKVCache for rationale.
        kdtype = keys.dtype
        k_flat = keys.reshape(-1, D)
        norms = mx.linalg.norm(k_flat.astype(mx.float32), axis=-1, keepdims=True).astype(kdtype)
        safe = mx.maximum(norms, mx.array(1e-4, dtype=kdtype))
        k_unit = (k_flat / safe).astype(mx.float16)

        ev = self._quantizer.encode(k_unit)
        k_hat_u = self._quantizer.decode(ev)
        k_dequant = (k_hat_u.astype(kdtype) * safe).reshape(B, H, S, D)

        # RVQ stores 2*b bits/dim (two index sets), plus per-vector norm fp16
        per_tok = (math.ceil(self._head_dim * 2 * self._bits / 8) + 2) * H * B
        self._key_bytes_compressed += per_tok * S
        self._key_bytes_fp16 += H * B * S * self._head_dim * 2
        return super().update_and_fetch(k_dequant, values)

    @property
    def compressed_key_bytes(self) -> int:
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        return self._key_bytes_fp16


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_caches(orig_make_cache, n_kv_heads, head_dim, bits, n_layers):
    _ = orig_make_cache()  # discard; used only to mirror structure
    return [
        TurboQuantMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=bits, seed=i)
        for i in range(n_layers)
    ]


def _resolve_model_args(model):
    """Return the inner language-model args, handling VLM wrappers.

    Qwen2-VL's top-level model.args only has text_config; the real
    attention config lives in model.language_model.args.
    """
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        # VLM wrapper — try to reach inner language model
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)
    return args


def build_vlm_caches(model, bits: int, use_rvq: bool = False, seed: int = 42):
    """Build a quantized cache list for VLM or text-only models.

    Inspects each layer individually so models with heterogeneous head_dim
    (e.g. Qwen2-VL vision vs language layers) are handled correctly.
    Layers without attention fall back to a standard fp16 KVCache.
    """
    from mlx_lm.models.cache import KVCache as _FallbackCache

    layers = getattr(model, "layers", None) or model.model.layers
    args = _resolve_model_args(model)
    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            caches.append(_FallbackCache())
            continue
        n_kv = getattr(attn, "n_kv_heads", None)
        hd = getattr(attn, "head_dim", None)
        if n_kv is None or hd is None:
            if args is not None:
                n_kv = n_kv or getattr(
                    args, "num_key_value_heads", getattr(args, "num_attention_heads", 1)
                )
                hd = (
                    hd
                    or getattr(args, "head_dim", None)
                    or (args.hidden_size // args.num_attention_heads)
                )
        if hd is None:
            caches.append(_FallbackCache())
            continue
        cls = TurboQuantRVQMLXKVCache if use_rvq else TurboQuantMLXKVCache
        caches.append(cls(n_kv_heads=n_kv, head_dim=hd, bits=bits, seed=seed + i))
    return caches


def run(model, tokenizer, orig_make_cache, cache_factory, label):
    messages = [{"role": "user", "content": PROMPT}]
    prompt_txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    injected = []
    original = model.make_cache
    if cache_factory is not None:

        def _patch(*_, **__):
            c = cache_factory()
            injected.extend(c)
            return c

        model.make_cache = _patch

    mx.clear_cache()
    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt_txt, max_tokens=MAX_TOKENS, verbose=False
    )
    elapsed = time.perf_counter() - t0
    model.make_cache = original

    tq = [c for c in injected if isinstance(c, (TurboQuantMLXKVCache, TurboQuantRVQMLXKVCache))]
    kf = sum(c.fp16_key_bytes for c in tq)
    kc = sum(c.compressed_key_bytes for c in tq)
    ratio = f"{kf / kc:.2f}×" if kc > 0 else "—"
    toks = len(tokenizer.encode(response))
    print(f"\n{'=' * 64}")
    print(f"[{label}]  key compression: {ratio}")
    print(f"  {response[:480]}{'...' if len(response) > 480 else ''}")
    print(f"  {toks} tokens  {elapsed:.1f}s  ({toks / elapsed:.1f} tok/s)")
    return response, elapsed, injected


# ── SNR / cosine quality curves ────────────────────────────────────────────────
def compute_quality_curves(head_dim, bit_range=(2, 3, 4, 5, 6)):
    np.random.seed(42)
    x_raw = np.random.randn(64, head_dim)
    x_unit = (x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).astype(np.float16)
    x_mx = mx.array(x_unit)
    snrs, coss = [], []
    for b in bit_range:
        q = TurboQuantProd(d=head_dim, b=b, m=min(head_dim, 64), seed=0)
        ev = q.encode(x_mx)
        x_hat = q.decode(ev)
        mse = float(mx.mean((x_mx - x_hat) ** 2))
        var = float(mx.mean(x_mx**2))
        snr = 10 * np.log10(max(var / mse, 1e-10))
        cos = float(
            mx.mean(
                mx.sum(x_mx * x_hat, axis=1)
                / (mx.linalg.norm(x_mx, axis=1) * mx.linalg.norm(x_hat, axis=1))
            )
        )
        snrs.append(snr)
        coss.append(cos)
    return snrs, coss


# ── Figure generation ──────────────────────────────────────────────────────────
def generate_figures(
    out_dir,
    model_label,
    configs,
    colors,
    compress,
    tput,
    tokens_out,
    key_kb,
    snrs,
    coss,
    bit_range,
    head_dim,
    n_kv_heads,
    n_layers,
    token_counts,
    fp16_full,
    sm_fp16,
    sm_2b,
    sm_3b,
    sm_4b,
    responses,
):
    sns.set_theme(style="whitegrid", font_scale=1.15)
    x = np.arange(len(configs))
    bar_w = 0.55

    def _bar(ax, vals, ylabel, title, cols=None, hline=None, fmt=".1f"):
        c = cols or colors
        bars = ax.bar(x, vals, width=bar_w, color=c, edgecolor="white", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if hline is not None:
            ax.axhline(hline, color="grey", ls="--", lw=1, alpha=0.7)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.02,
                f"{v:{fmt}}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_ylim(0, max(vals) * 1.28)
        sns.despine(ax=ax)

    to_mb = lambda b: b / 1024**2
    br = list(bit_range)

    # ── Fig 1: Benchmark summary ───────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(
        f"TurboQuant KV Cache — {model_label}\nApple M4 · Dense Architecture · head_dim={head_dim}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    _bar(
        axes[0, 0],
        compress,
        "Key Compression Ratio (×)",
        "Key Compression Ratio",
        hline=1.0,
        fmt=".2f",
    )
    _bar(axes[0, 1], tput, "Tokens / second", "Generation Throughput (tok/s)", hline=tput[0])
    _bar(
        axes[1, 0], tokens_out, "Tokens Generated", "Tokens Generated (max 200)", fmt="d", hline=200
    )
    _bar(axes[1, 1], key_kb, "Key Cache Size (KB)", "Compressed Key Cache Size", fmt=".0f")
    fig1.tight_layout()
    fig1.savefig(f"{out_dir}/fig1_benchmark_summary.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig1_benchmark_summary.png")

    # ── Fig 2: Quality vs bits ─────────────────────────────────────────────────
    fig2, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(
        f"Quality vs Bits — {model_label} (head_dim={head_dim})", fontsize=14, fontweight="bold"
    )
    ax_s.plot(
        br, snrs, color=PALETTE["4bit"], marker="s", lw=2.2, ms=7, label=f"head_dim={head_dim}"
    )
    ax_c.plot(
        br, coss, color=PALETTE["4bit"], marker="s", lw=2.2, ms=7, label=f"head_dim={head_dim}"
    )
    ax_s.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7, label="0 dB")
    ax_s.axhline(10, color="green", ls="--", lw=1.5, alpha=0.7, label="10 dB (near-lossless)")
    ax_c.axhline(0.90, color="green", ls="--", lw=1.5, alpha=0.7, label="0.90 (near-lossless)")
    ax_c.axhline(0.80, color="orange", ls="--", lw=1.5, alpha=0.7, label="0.80 (degraded)")
    for b_idx, b in enumerate([3, 4]):
        ax_c.annotate(
            f"{b}b→{coss[b_idx + 1]:.2f}",
            xy=(b, coss[b_idx + 1]),
            xytext=(b + 0.15, coss[b_idx + 1] + (0.03 if b == 3 else -0.04)),
            fontsize=9,
            color=PALETTE["3bit"] if b == 3 else PALETTE["4bit"],
            arrowprops=dict(
                arrowstyle="->", lw=0.8, color=PALETTE["3bit"] if b == 3 else PALETTE["4bit"]
            ),
        )
    for ax, lbl in [(ax_s, "SNR (dB)"), (ax_c, "Cosine Similarity")]:
        ax.set_xlabel("Bit-width")
        ax.set_ylabel(lbl)
        ax.set_xticks(br)
        ax.legend(fontsize=9)
        sns.despine(ax=ax)
    ax_s.set_title("Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
    ax_c.set_title("Cosine Similarity (original vs TQ)", fontsize=12, fontweight="bold")
    ax_c.set_ylim(0.4, 1.05)
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/fig2_quality_vs_bits.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig2_quality_vs_bits.png")

    # ── Fig 3: Memory at scale ─────────────────────────────────────────────────
    def tq_bytes(tokens, bits):
        b_mse = max(bits - 1, 1)
        m = min(head_dim, 64)
        per = (math.ceil(head_dim * b_mse / 8) + math.ceil(m / 8) + 2 + 2) * n_kv_heads * n_layers
        return tokens * per

    tq2 = np.array([tq_bytes(t, 2) for t in token_counts])
    tq3 = np.array([tq_bytes(t, 3) for t in token_counts])
    tq4 = np.array([tq_bytes(t, 4) for t in token_counts])
    val = token_counts * n_layers * n_kv_heads * head_dim * 2

    fig3, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        f"KV Cache Memory at Scale — {model_label}\n"
        f"({n_layers} layers, head_dim={head_dim}, kv_heads={n_kv_heads})",
        fontsize=13,
        fontweight="bold",
    )
    ax_a.plot(
        token_counts,
        to_mb(fp16_full),
        color=PALETTE["fp16"],
        lw=2.5,
        marker="o",
        ms=5,
        label="fp16 K+V",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq2 + val),
        color=PALETTE["2bit"],
        lw=2.5,
        marker="D",
        ms=5,
        label="TQ 2-bit keys + fp16 values",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq3 + val),
        color=PALETTE["3bit"],
        lw=2.5,
        marker="s",
        ms=5,
        label="TQ 3-bit keys + fp16 values",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq4 + val),
        color=PALETTE["4bit"],
        lw=2.5,
        marker="^",
        ms=5,
        label="TQ 4-bit keys + fp16 values",
    )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(token_counts)
    ax_a.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=9
    )
    ax_a.set_xlabel("Context length")
    ax_a.set_ylabel("Memory (MB)")
    ax_a.set_title("Absolute Memory", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=9)
    sns.despine(ax=ax_a)

    r2 = fp16_full / (tq2 + val)
    r3 = fp16_full / (tq3 + val)
    r4 = fp16_full / (tq4 + val)
    ax_r.plot(token_counts, r2, color=PALETTE["2bit"], lw=2.5, marker="D", ms=5, label="TQ 2-bit")
    ax_r.plot(token_counts, r3, color=PALETTE["3bit"], lw=2.5, marker="s", ms=5, label="TQ 3-bit")
    ax_r.plot(token_counts, r4, color=PALETTE["4bit"], lw=2.5, marker="^", ms=5, label="TQ 4-bit")
    ax_r.fill_between(token_counts, r4, 1.0, alpha=0.12, color=PALETTE["4bit"])
    ax_r.fill_between(token_counts, r3, r4, alpha=0.10, color=PALETTE["3bit"])
    ax_r.fill_between(token_counts, r2, r3, alpha=0.08, color=PALETTE["2bit"])
    ax_r.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.6)
    ax_r.set_xscale("log", base=2)
    ax_r.set_xticks(token_counts)
    ax_r.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=9
    )
    ax_r.set_xlabel("Context length")
    ax_r.set_ylabel("Compression vs fp16")
    ax_r.set_title("Compression Ratio", fontsize=12, fontweight="bold")
    ax_r.legend(fontsize=9)
    sns.despine(ax=ax_r)
    fig3.tight_layout()
    fig3.savefig(f"{out_dir}/fig3_memory_at_scale.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig3_memory_at_scale.png")

    # ── Fig 4: Attention distortion ────────────────────────────────────────────
    N_k = len(sm_fp16)
    fig4, axes4 = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig4.suptitle(
        f"Attention Score Distortion — {model_label} (head_dim={head_dim})\n"
        f"{N_k} key vectors, query dot-product, softmax",
        fontsize=13,
        fontweight="bold",
    )
    for ax, (sm, label, col) in zip(
        axes4,
        [
            (sm_fp16, "fp16 Baseline (reference)", PALETTE["fp16"]),
            (sm_2b, f"TurboQuant 2-bit  cosine≈{coss[0]:.2f}", PALETTE["2bit"]),
            (sm_3b, f"TurboQuant 3-bit  cosine≈{coss[1]:.2f}", PALETTE["3bit"]),
            (sm_4b, f"TurboQuant 4-bit  cosine≈{coss[2]:.2f}", PALETTE["4bit"]),
        ],
    ):
        ax.bar(np.arange(N_k), sm, color=col, alpha=0.78, edgecolor="white", linewidth=0.5)
        ax.plot(
            np.arange(N_k),
            sm_fp16,
            color=PALETTE["fp16"],
            lw=1.5,
            ls="--",
            alpha=0.5,
            label="fp16 ref",
        )
        mse_a = np.mean((sm - sm_fp16) ** 2)
        cos_a = np.dot(sm, sm_fp16) / (np.linalg.norm(sm) * np.linalg.norm(sm_fp16))
        ax.set_ylabel("Attention weight")
        ax.set_title(
            f"{label}   |   MSE={mse_a:.2e}   cosine={cos_a:.4f}", fontsize=11, fontweight="bold"
        )
        ax.set_ylim(0, max(sm_fp16) * 1.45)
        sns.despine(ax=ax)
    axes4[-1].set_xlabel("Key Token Index")
    fig4.tight_layout()
    fig4.savefig(f"{out_dir}/fig4_attention_distortion.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig4_attention_distortion.png")

    # ── Fig 5: Output quality text comparison ─────────────────────────────────
    fig5, axes5 = plt.subplots(4, 1, figsize=(16, 16))
    fig5.suptitle(f"Generated Output Comparison — {model_label}", fontsize=14, fontweight="bold")
    labels = ["fp16 Baseline", "TurboQuant 2-bit", "TurboQuant 3-bit", "TurboQuant 4-bit"]
    for ax, resp, lbl, col in zip(axes5, responses, labels, colors):
        ax.set_facecolor(col + "18")
        ax.text(
            0.01,
            0.97,
            f"[{lbl}]",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=col,
            va="top",
        )
        wrapped = resp[:600] + ("..." if len(resp) > 600 else "")
        ax.text(
            0.01,
            0.82,
            wrapped,
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig5.tight_layout()
    fig5.savefig(f"{out_dir}/fig5_output_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig5_output_comparison.png")

    # ── Fig 6: Combined report ─────────────────────────────────────────────────
    fig6 = plt.figure(figsize=(20, 22))
    fig6.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(3, 2, figure=fig6, hspace=0.44, wspace=0.35)

    ax_a = fig6.add_subplot(gs[0, 0])
    bars = ax_a.bar(configs, compress, color=colors, edgecolor="white", lw=1.2)
    ax_a.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, compress):
        ax_a.text(
            b.get_x() + b.get_width() / 2,
            v + 0.06,
            f"{v:.2f}×",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax_a.set_title("A  Key Compression Ratio", fontsize=12, fontweight="bold", loc="left")
    ax_a.set_ylabel("Ratio vs fp16")
    sns.despine(ax=ax_a)
    ax_a.set_ylim(0, max(compress) * 1.28)

    ax_b = fig6.add_subplot(gs[0, 1])
    bars = ax_b.bar(configs, tput, color=colors, edgecolor="white", lw=1.2)
    ax_b.axhline(tput[0], color="grey", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, tput):
        ax_b.text(
            b.get_x() + b.get_width() / 2,
            v + 0.5,
            f"{v:.1f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax_b.set_title("B  Generation Throughput (tok/s)", fontsize=12, fontweight="bold", loc="left")
    ax_b.set_ylabel("tok/s")
    sns.despine(ax=ax_b)
    ax_b.set_ylim(0, max(tput) * 1.28)

    ax_c = fig6.add_subplot(gs[1, 0])
    ax_c.plot(br, coss, color=PALETTE["4bit"], marker="s", lw=2.2, ms=7)
    ax_c.axhline(0.90, color="green", ls="--", lw=1.5, alpha=0.7, label="0.90 near-lossless")
    ax_c.axhline(0.80, color="orange", ls="--", lw=1.5, alpha=0.7, label="0.80 degraded")
    ax_c.set_xlabel("Bit-width")
    ax_c.set_ylabel("Cosine Similarity")
    ax_c.set_xticks(br)
    ax_c.set_ylim(0.4, 1.05)
    ax_c.set_title("C  Quality vs Bits", fontsize=12, fontweight="bold", loc="left")
    ax_c.legend(fontsize=9)
    sns.despine(ax=ax_c)

    ax_d = fig6.add_subplot(gs[1, 1])
    ax_d.plot(
        token_counts,
        to_mb(fp16_full),
        color=PALETTE["fp16"],
        lw=2.5,
        marker="o",
        ms=5,
        label="fp16",
    )
    ax_d.plot(
        token_counts,
        to_mb(tq2 + val),
        color=PALETTE["2bit"],
        lw=2.5,
        marker="D",
        ms=5,
        label="TQ 2-bit",
    )
    ax_d.plot(
        token_counts,
        to_mb(tq3 + val),
        color=PALETTE["3bit"],
        lw=2.5,
        marker="s",
        ms=5,
        label="TQ 3-bit",
    )
    ax_d.plot(
        token_counts,
        to_mb(tq4 + val),
        color=PALETTE["4bit"],
        lw=2.5,
        marker="^",
        ms=5,
        label="TQ 4-bit",
    )
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks(token_counts)
    ax_d.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8
    )
    ax_d.set_xlabel("Context length")
    ax_d.set_ylabel("Memory (MB)")
    ax_d.set_title("D  KV Cache Memory at Scale", fontsize=12, fontweight="bold", loc="left")
    ax_d.legend(fontsize=9)
    sns.despine(ax=ax_d)

    ax_e = fig6.add_subplot(gs[2, :])
    w = 0.20
    ax_e.bar(
        np.arange(N_k) - 1.5 * w, sm_fp16, width=w, color=PALETTE["fp16"], alpha=0.85, label="fp16"
    )
    ax_e.bar(
        np.arange(N_k) - 0.5 * w,
        sm_2b,
        width=w,
        color=PALETTE["2bit"],
        alpha=0.85,
        label="TQ 2-bit",
    )
    ax_e.bar(
        np.arange(N_k) + 0.5 * w,
        sm_3b,
        width=w,
        color=PALETTE["3bit"],
        alpha=0.85,
        label="TQ 3-bit",
    )
    ax_e.bar(
        np.arange(N_k) + 1.5 * w,
        sm_4b,
        width=w,
        color=PALETTE["4bit"],
        alpha=0.85,
        label="TQ 4-bit",
    )
    ax_e.set_xlabel("Key Token Index")
    ax_e.set_ylabel("Attention weight")
    ax_e.set_title(
        f"E  Attention Distortion (head_dim={head_dim})", fontsize=12, fontweight="bold", loc="left"
    )
    ax_e.legend(fontsize=9)
    sns.despine(ax=ax_e)

    fig6.suptitle(
        f"TurboQuant KV Cache — {model_label} Full Benchmark Report\nApple M4 · veloxquant_mlx",
        fontsize=16,
        fontweight="bold",
        y=1.005,
    )
    fig6.savefig(f"{out_dir}/fig6_full_report.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig6_full_report.png")

    plt.close("all")


# ── Main entry point ───────────────────────────────────────────────────────────
def run_benchmark(model_id, out_dir, model_label):
    import os

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"Loading {model_id}...")
    model, tokenizer = mlx_lm.load(model_id)
    cfg = model.args
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    n_kv = cfg.num_key_value_heads
    n_layers = cfg.num_hidden_layers
    print(f"  {model_label}: {n_layers} layers, head_dim={hd}, kv_heads={n_kv}\n")

    if not hasattr(model, "make_cache"):

        def _default_make_cache():
            return [_MLXKVCache() for _ in range(n_layers)]

        model.make_cache = _default_make_cache

    orig_make_cache = model.make_cache

    def _make_tq(bits):
        return [
            TurboQuantMLXKVCache(n_kv_heads=n_kv, head_dim=hd, bits=bits, seed=i)
            for i in range(n_layers)
        ]

    resp_fp16, t_fp16, c_fp16 = run(model, tokenizer, orig_make_cache, None, "fp16 baseline")
    resp_2b, t_2b, c_2b = run(
        model, tokenizer, orig_make_cache, lambda: _make_tq(2), "TurboQuant 2-bit"
    )
    resp_3b, t_3b, c_3b = run(
        model, tokenizer, orig_make_cache, lambda: _make_tq(3), "TurboQuant 3-bit"
    )
    resp_4b, t_4b, c_4b = run(
        model, tokenizer, orig_make_cache, lambda: _make_tq(4), "TurboQuant 4-bit"
    )

    def _stats(caches):
        tq = [c for c in caches if isinstance(c, TurboQuantMLXKVCache)]
        return (
            sum(c.fp16_key_bytes for c in tq),
            sum(c.compressed_key_bytes for c in tq),
            max((c.offset for c in caches if hasattr(c, "offset")), default=0),
        )

    kf_2b, kc_2b, n_tok = _stats(c_2b)
    kf_3b, kc_3b, _ = _stats(c_3b)
    kf_4b, kc_4b, _ = _stats(c_4b)

    enc = tokenizer.encode
    configs = ["fp16\nbaseline", "TurboQuant\n2-bit", "TurboQuant\n3-bit", "TurboQuant\n4-bit"]
    colors = [PALETTE["fp16"], PALETTE["2bit"], PALETTE["3bit"], PALETTE["4bit"]]
    compress = [
        1.00,
        round(kf_2b / kc_2b, 2) if kc_2b > 0 else 0,
        round(kf_3b / kc_3b, 2) if kc_3b > 0 else 0,
        round(kf_4b / kc_4b, 2) if kc_4b > 0 else 0,
    ]
    tput = [
        len(enc(r)) / t
        for r, t in [(resp_fp16, t_fp16), (resp_2b, t_2b), (resp_3b, t_3b), (resp_4b, t_4b)]
    ]
    tok_out = [len(enc(r)) for r in [resp_fp16, resp_2b, resp_3b, resp_4b]]
    key_kb = [kf_4b / 1024, kc_2b / 1024, kc_3b / 1024, kc_4b / 1024]

    print(f"\n{'=' * 64}")
    print(f"SUMMARY — {model_label}")
    print(f"  Tokens cached : {n_tok}")
    print(f"  2-bit         : {compress[1]}× compression  |  {tput[1]:.1f} tok/s")
    print(f"  3-bit         : {compress[2]}× compression  |  {tput[2]:.1f} tok/s")
    print(f"  4-bit         : {compress[3]}× compression  |  {tput[3]:.1f} tok/s")
    print(f"  fp16 baseline :                    |  {tput[0]:.1f} tok/s")

    token_counts = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    fp16_full = token_counts * n_layers * n_kv * hd * 2 * 2

    print(f"\nComputing quality curves...")
    snrs, coss = compute_quality_curves(hd)

    np.random.seed(7)
    N_k = 32
    q_np = np.random.randn(hd).astype(np.float32)
    q_np /= np.linalg.norm(q_np)
    k_np = np.random.randn(N_k, hd).astype(np.float32)
    k_unit = k_np / np.linalg.norm(k_np, axis=1, keepdims=True)
    q_mx = mx.array(q_np.astype(np.float16)).reshape(1, -1)
    k_mx = mx.array(k_unit.astype(np.float16))

    def _attn(bits):
        qt = TurboQuantProd(d=hd, b=bits, m=min(hd, 64), seed=0)
        ev = qt.encode(k_mx)
        k_hat = qt.decode(ev)
        sc = np.array(k_hat @ q_mx.T).flatten()
        sm = np.exp(sc) / np.exp(sc).sum()
        return sm

    sc_fp16 = np.array(k_mx @ q_mx.T).flatten()
    sm_fp16 = np.exp(sc_fp16) / np.exp(sc_fp16).sum()
    sm_2b = _attn(2)
    sm_3b = _attn(3)
    sm_4b = _attn(4)

    print(f"Generating figures → {out_dir}/")
    generate_figures(
        out_dir=out_dir,
        model_label=model_label,
        configs=configs,
        colors=colors,
        compress=compress,
        tput=tput,
        tokens_out=tok_out,
        key_kb=key_kb,
        snrs=snrs,
        coss=coss,
        bit_range=range(2, 7),
        head_dim=hd,
        n_kv_heads=n_kv,
        n_layers=n_layers,
        token_counts=token_counts,
        fp16_full=fp16_full,
        sm_fp16=sm_fp16,
        sm_2b=sm_2b,
        sm_3b=sm_3b,
        sm_4b=sm_4b,
        responses=[resp_fp16, resp_2b, resp_3b, resp_4b],
    )
    print(f"Done — {model_label}\n")
    return compress, tput, tok_out


# ── v2: with TurboQuant RVQ 2-bit ─────────────────────────────────────────────
PALETTE_V2 = {
    "fp16": "#4C72B0",
    "2bit": "#C44E52",
    "3bit": "#DD8452",
    "4bit": "#55A868",
    "rvq2": "#8172B2",
}


def _generate_figures_v2(
    out_dir,
    model_label,
    configs,
    colors,
    compress,
    tput,
    tokens_out,
    key_kb,
    snrs,
    coss,
    bit_range,
    head_dim,
    n_kv_heads,
    n_layers,
    token_counts,
    fp16_full,
    sm_fp16,
    sm_2b,
    sm_3b,
    sm_4b,
    sm_rvq2,
    cos_rvq2,
    snr_rvq2,
    responses,
):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    x = np.arange(len(configs))
    bar_w = 0.6

    def _bar(ax, vals, ylabel, title, cols=None, hline=None, fmt=".1f"):
        c = cols or colors
        bars = ax.bar(x, vals, width=bar_w, color=c, edgecolor="white", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if hline is not None:
            ax.axhline(hline, color="grey", ls="--", lw=1, alpha=0.7)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.02,
                f"{v:{fmt}}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_ylim(0, max(vals) * 1.28)
        sns.despine(ax=ax)

    to_mb = lambda b: b / 1024**2

    # Fig 1: 5-config benchmark summary
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle(
        f"TurboQuant KV Cache (v2 with RVQ) — {model_label}\nApple M4 · head_dim={head_dim}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    _bar(
        axes[0, 0],
        compress,
        "Key Compression Ratio (×)",
        "Key Compression Ratio",
        hline=1.0,
        fmt=".2f",
    )
    _bar(axes[0, 1], tput, "Tokens / second", "Generation Throughput (tok/s)", hline=tput[0])
    _bar(
        axes[1, 0], tokens_out, "Tokens Generated", "Tokens Generated (max 200)", fmt="d", hline=200
    )
    _bar(axes[1, 1], key_kb, "Key Cache Size (KB)", "Compressed Key Cache Size", fmt=".0f")
    fig1.tight_layout()
    fig1.savefig(f"{out_dir}/fig1_benchmark_summary.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig1_benchmark_summary.png")

    # Fig 2: Quality vs bits, RVQ 2-bit highlighted
    fig2, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(
        f"Quality vs Bits — {model_label} (head_dim={head_dim})", fontsize=14, fontweight="bold"
    )
    br = list(bit_range)
    ax_s.plot(
        br, snrs, color=PALETTE_V2["4bit"], marker="s", lw=2.2, ms=7, label="single-pass TurboQuant"
    )
    ax_c.plot(
        br, coss, color=PALETTE_V2["4bit"], marker="s", lw=2.2, ms=7, label="single-pass TurboQuant"
    )
    ax_s.scatter(
        [2],
        [snr_rvq2],
        color=PALETTE_V2["rvq2"],
        marker="*",
        s=260,
        zorder=5,
        label=f"RVQ 2-bit (this work)",
    )
    ax_c.scatter(
        [2],
        [cos_rvq2],
        color=PALETTE_V2["rvq2"],
        marker="*",
        s=260,
        zorder=5,
        label=f"RVQ 2-bit (this work)",
    )
    ax_s.axhline(10, color="green", ls="--", lw=1.5, alpha=0.7, label="10 dB (near-lossless)")
    ax_c.axhline(0.90, color="green", ls="--", lw=1.5, alpha=0.7, label="0.90 (near-lossless)")
    ax_c.axhline(0.80, color="orange", ls="--", lw=1.5, alpha=0.7, label="0.80 (degraded)")
    for ax, lbl in [(ax_s, "SNR (dB)"), (ax_c, "Cosine Similarity")]:
        ax.set_xlabel("Bit-width")
        ax.set_ylabel(lbl)
        ax.set_xticks(br)
        ax.legend(fontsize=9)
        sns.despine(ax=ax)
    ax_s.set_title("Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
    ax_c.set_title("Cosine Similarity", fontsize=12, fontweight="bold")
    ax_c.set_ylim(0.4, 1.05)
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/fig2_quality_vs_bits.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig2_quality_vs_bits.png")

    # Fig 3: Memory at scale (now includes RVQ 2-bit line)
    def tq_bytes(tokens, bits):
        b_mse = max(bits - 1, 1)
        m = min(head_dim, 64)
        per = (math.ceil(head_dim * b_mse / 8) + math.ceil(m / 8) + 2 + 2) * n_kv_heads * n_layers
        return tokens * per

    def rvq_bytes(tokens, bits):
        per = (math.ceil(head_dim * 2 * bits / 8) + 2) * n_kv_heads * n_layers
        return tokens * per

    tq2 = np.array([tq_bytes(t, 2) for t in token_counts])
    tq3 = np.array([tq_bytes(t, 3) for t in token_counts])
    tq4 = np.array([tq_bytes(t, 4) for t in token_counts])
    rvq2 = np.array([rvq_bytes(t, 2) for t in token_counts])
    val = token_counts * n_layers * n_kv_heads * head_dim * 2

    fig3, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        f"KV Cache Memory at Scale — {model_label}\n"
        f"({n_layers} layers, head_dim={head_dim}, kv_heads={n_kv_heads})",
        fontsize=13,
        fontweight="bold",
    )
    ax_a.plot(
        token_counts,
        to_mb(fp16_full),
        color=PALETTE_V2["fp16"],
        lw=2.5,
        marker="o",
        ms=5,
        label="fp16 K+V",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq2 + val),
        color=PALETTE_V2["2bit"],
        lw=2.5,
        marker="D",
        ms=5,
        label="TQ 2-bit (single-pass)",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq3 + val),
        color=PALETTE_V2["3bit"],
        lw=2.5,
        marker="s",
        ms=5,
        label="TQ 3-bit",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq4 + val),
        color=PALETTE_V2["4bit"],
        lw=2.5,
        marker="^",
        ms=5,
        label="TQ 4-bit",
    )
    ax_a.plot(
        token_counts,
        to_mb(rvq2 + val),
        color=PALETTE_V2["rvq2"],
        lw=2.5,
        marker="*",
        ms=8,
        label="TQ RVQ 2-bit ★",
    )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(token_counts)
    ax_a.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8
    )
    ax_a.set_xlabel("Context length")
    ax_a.set_ylabel("Memory (MB)")
    ax_a.set_title("Absolute Memory", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=9)
    sns.despine(ax=ax_a)

    r2 = fp16_full / (tq2 + val)
    r3 = fp16_full / (tq3 + val)
    r4 = fp16_full / (tq4 + val)
    rr = fp16_full / (rvq2 + val)
    ax_r.plot(
        token_counts, r2, color=PALETTE_V2["2bit"], lw=2.5, marker="D", ms=5, label="TQ 2-bit"
    )
    ax_r.plot(
        token_counts, r3, color=PALETTE_V2["3bit"], lw=2.5, marker="s", ms=5, label="TQ 3-bit"
    )
    ax_r.plot(
        token_counts, r4, color=PALETTE_V2["4bit"], lw=2.5, marker="^", ms=5, label="TQ 4-bit"
    )
    ax_r.plot(
        token_counts, rr, color=PALETTE_V2["rvq2"], lw=2.5, marker="*", ms=8, label="TQ RVQ 2-bit ★"
    )
    ax_r.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.6)
    ax_r.set_xscale("log", base=2)
    ax_r.set_xticks(token_counts)
    ax_r.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8
    )
    ax_r.set_xlabel("Context length")
    ax_r.set_ylabel("Compression vs fp16")
    ax_r.set_title("Compression Ratio", fontsize=12, fontweight="bold")
    ax_r.legend(fontsize=9)
    sns.despine(ax=ax_r)
    fig3.tight_layout()
    fig3.savefig(f"{out_dir}/fig3_memory_at_scale.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig3_memory_at_scale.png")

    # Fig 4: Attention distortion 5 panels (added RVQ 2-bit)
    N_k = len(sm_fp16)
    fig4, axes4 = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig4.suptitle(
        f"Attention Score Distortion — {model_label} (head_dim={head_dim})\n"
        f"{N_k} key vectors, query dot-product, softmax",
        fontsize=13,
        fontweight="bold",
    )
    for ax, (sm, label, col) in zip(
        axes4,
        [
            (sm_fp16, "fp16 Baseline (reference)", PALETTE_V2["fp16"]),
            (sm_2b, f"TurboQuant 2-bit (single-pass)  cosine≈{coss[0]:.2f}", PALETTE_V2["2bit"]),
            (sm_3b, f"TurboQuant 3-bit  cosine≈{coss[1]:.2f}", PALETTE_V2["3bit"]),
            (sm_4b, f"TurboQuant 4-bit  cosine≈{coss[2]:.2f}", PALETTE_V2["4bit"]),
            (sm_rvq2, f"TurboQuant RVQ 2-bit ★  cosine≈{cos_rvq2:.2f}", PALETTE_V2["rvq2"]),
        ],
    ):
        ax.bar(np.arange(N_k), sm, color=col, alpha=0.78, edgecolor="white", linewidth=0.5)
        ax.plot(
            np.arange(N_k),
            sm_fp16,
            color=PALETTE_V2["fp16"],
            lw=1.5,
            ls="--",
            alpha=0.5,
            label="fp16 ref",
        )
        mse_a = np.mean((sm - sm_fp16) ** 2)
        cos_a = np.dot(sm, sm_fp16) / (np.linalg.norm(sm) * np.linalg.norm(sm_fp16))
        ax.set_ylabel("Attention weight")
        ax.set_title(
            f"{label}   |   MSE={mse_a:.2e}   cosine={cos_a:.4f}", fontsize=10, fontweight="bold"
        )
        ax.set_ylim(0, max(sm_fp16) * 1.45)
        sns.despine(ax=ax)
    axes4[-1].set_xlabel("Key Token Index")
    fig4.tight_layout()
    fig4.savefig(f"{out_dir}/fig4_attention_distortion.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig4_attention_distortion.png")

    # Fig 5: Output text comparison (5 panels)
    fig5, axes5 = plt.subplots(5, 1, figsize=(16, 18))
    fig5.suptitle(f"Generated Output Comparison — {model_label}", fontsize=14, fontweight="bold")
    labels = [
        "fp16 Baseline",
        "TurboQuant 2-bit",
        "TurboQuant 3-bit",
        "TurboQuant 4-bit",
        "TurboQuant RVQ 2-bit ★",
    ]
    for ax, resp, lbl, col in zip(axes5, responses, labels, colors):
        ax.set_facecolor(col + "18")
        ax.text(
            0.01,
            0.97,
            f"[{lbl}]",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=col,
            va="top",
        )
        wrapped = resp[:600] + ("..." if len(resp) > 600 else "")
        ax.text(
            0.01,
            0.82,
            wrapped,
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig5.tight_layout()
    fig5.savefig(f"{out_dir}/fig5_output_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig5_output_comparison.png")

    # Fig 6: Combined report
    fig6 = plt.figure(figsize=(20, 22))
    fig6.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(3, 2, figure=fig6, hspace=0.44, wspace=0.35)

    ax_a = fig6.add_subplot(gs[0, 0])
    bars = ax_a.bar(configs, compress, color=colors, edgecolor="white", lw=1.2)
    ax_a.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, compress):
        ax_a.text(
            b.get_x() + b.get_width() / 2,
            v + 0.06,
            f"{v:.2f}×",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax_a.set_title("A  Key Compression Ratio", fontsize=12, fontweight="bold", loc="left")
    ax_a.set_ylabel("Ratio vs fp16")
    sns.despine(ax=ax_a)
    ax_a.set_ylim(0, max(compress) * 1.28)

    ax_b = fig6.add_subplot(gs[0, 1])
    bars = ax_b.bar(configs, tput, color=colors, edgecolor="white", lw=1.2)
    ax_b.axhline(tput[0], color="grey", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, tput):
        ax_b.text(
            b.get_x() + b.get_width() / 2,
            v + 0.5,
            f"{v:.1f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax_b.set_title("B  Throughput (tok/s)", fontsize=12, fontweight="bold", loc="left")
    ax_b.set_ylabel("tok/s")
    sns.despine(ax=ax_b)
    ax_b.set_ylim(0, max(tput) * 1.28)

    ax_c = fig6.add_subplot(gs[1, 0])
    ax_c.plot(
        br, coss, color=PALETTE_V2["4bit"], marker="s", lw=2.2, ms=7, label="single-pass TurboQuant"
    )
    ax_c.scatter(
        [2], [cos_rvq2], color=PALETTE_V2["rvq2"], marker="*", s=300, zorder=5, label=f"RVQ 2-bit ★"
    )
    ax_c.axhline(0.90, color="green", ls="--", lw=1.5, alpha=0.7, label="0.90")
    ax_c.axhline(0.80, color="orange", ls="--", lw=1.5, alpha=0.7, label="0.80")
    ax_c.set_xlabel("Bit-width")
    ax_c.set_ylabel("Cosine Similarity")
    ax_c.set_xticks(br)
    ax_c.set_ylim(0.4, 1.05)
    ax_c.set_title(
        "C  Quality vs Bits (RVQ-2 highlighted)", fontsize=12, fontweight="bold", loc="left"
    )
    ax_c.legend(fontsize=9)
    sns.despine(ax=ax_c)

    ax_d = fig6.add_subplot(gs[1, 1])
    ax_d.plot(
        token_counts,
        to_mb(fp16_full),
        color=PALETTE_V2["fp16"],
        lw=2.5,
        marker="o",
        ms=5,
        label="fp16",
    )
    ax_d.plot(
        token_counts,
        to_mb(tq2 + val),
        color=PALETTE_V2["2bit"],
        lw=2.5,
        marker="D",
        ms=5,
        label="TQ 2-bit",
    )
    ax_d.plot(
        token_counts,
        to_mb(tq4 + val),
        color=PALETTE_V2["4bit"],
        lw=2.5,
        marker="^",
        ms=5,
        label="TQ 4-bit",
    )
    ax_d.plot(
        token_counts,
        to_mb(rvq2 + val),
        color=PALETTE_V2["rvq2"],
        lw=2.5,
        marker="*",
        ms=8,
        label="RVQ 2-bit ★",
    )
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks(token_counts)
    ax_d.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8
    )
    ax_d.set_xlabel("Context length")
    ax_d.set_ylabel("Memory (MB)")
    ax_d.set_title("D  KV Cache Memory at Scale", fontsize=12, fontweight="bold", loc="left")
    ax_d.legend(fontsize=9)
    sns.despine(ax=ax_d)

    ax_e = fig6.add_subplot(gs[2, :])
    w = 0.16
    ax_e.bar(
        np.arange(N_k) - 2 * w, sm_fp16, width=w, color=PALETTE_V2["fp16"], alpha=0.85, label="fp16"
    )
    ax_e.bar(
        np.arange(N_k) - w, sm_2b, width=w, color=PALETTE_V2["2bit"], alpha=0.85, label="TQ 2-bit"
    )
    ax_e.bar(np.arange(N_k), sm_3b, width=w, color=PALETTE_V2["3bit"], alpha=0.85, label="TQ 3-bit")
    ax_e.bar(
        np.arange(N_k) + w, sm_4b, width=w, color=PALETTE_V2["4bit"], alpha=0.85, label="TQ 4-bit"
    )
    ax_e.bar(
        np.arange(N_k) + 2 * w,
        sm_rvq2,
        width=w,
        color=PALETTE_V2["rvq2"],
        alpha=0.85,
        label="RVQ 2-bit ★",
    )
    ax_e.set_xlabel("Key Token Index")
    ax_e.set_ylabel("Attention weight")
    ax_e.set_title(
        f"E  Attention Distortion (head_dim={head_dim})", fontsize=12, fontweight="bold", loc="left"
    )
    ax_e.legend(fontsize=9)
    sns.despine(ax=ax_e)

    fig6.suptitle(
        f"TurboQuant KV Cache (v2 with RVQ) — {model_label}\nApple M4 · veloxquant_mlx",
        fontsize=16,
        fontweight="bold",
        y=1.005,
    )
    fig6.savefig(f"{out_dir}/fig6_full_report.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig6_full_report.png")
    plt.close("all")


def run_benchmark_v2(model_id, out_dir, model_label):
    """Benchmark with 5 configs: fp16, TQ 2-bit, TQ 3-bit, TQ 4-bit, TQ RVQ 2-bit."""
    import os

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"[v2] Loading {model_id}...")
    model, tokenizer = mlx_lm.load(model_id)
    cfg = model.args
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    n_kv = cfg.num_key_value_heads
    n_layers = cfg.num_hidden_layers
    print(f"  {model_label}: {n_layers} layers, head_dim={hd}, kv_heads={n_kv}\n")

    if not hasattr(model, "make_cache"):

        def _default_make_cache():
            return [_MLXKVCache() for _ in range(n_layers)]

        model.make_cache = _default_make_cache

    orig_make_cache = model.make_cache

    def _make_tq(bits):
        return [
            TurboQuantMLXKVCache(n_kv_heads=n_kv, head_dim=hd, bits=bits, seed=i)
            for i in range(n_layers)
        ]

    def _make_rvq(bits):
        return [
            TurboQuantRVQMLXKVCache(n_kv_heads=n_kv, head_dim=hd, bits=bits, seed=i)
            for i in range(n_layers)
        ]

    resp_fp16, t_fp16, c_fp16 = run(model, tokenizer, orig_make_cache, None, "fp16 baseline")
    resp_2b, t_2b, c_2b = run(
        model, tokenizer, orig_make_cache, lambda: _make_tq(2), "TurboQuant 2-bit"
    )
    resp_3b, t_3b, c_3b = run(
        model, tokenizer, orig_make_cache, lambda: _make_tq(3), "TurboQuant 3-bit"
    )
    resp_4b, t_4b, c_4b = run(
        model, tokenizer, orig_make_cache, lambda: _make_tq(4), "TurboQuant 4-bit"
    )
    resp_rvq2, t_rvq2, c_rvq2 = run(
        model, tokenizer, orig_make_cache, lambda: _make_rvq(2), "TurboQuant RVQ 2-bit"
    )

    def _stats(caches, cls):
        tq = [c for c in caches if isinstance(c, cls)]
        return (
            sum(c.fp16_key_bytes for c in tq),
            sum(c.compressed_key_bytes for c in tq),
            max((c.offset for c in caches if hasattr(c, "offset")), default=0),
        )

    kf_2b, kc_2b, n_tok = _stats(c_2b, TurboQuantMLXKVCache)
    kf_3b, kc_3b, _ = _stats(c_3b, TurboQuantMLXKVCache)
    kf_4b, kc_4b, _ = _stats(c_4b, TurboQuantMLXKVCache)
    kf_rvq, kc_rvq, _ = _stats(c_rvq2, TurboQuantRVQMLXKVCache)

    enc = tokenizer.encode
    configs = ["fp16\nbaseline", "TQ\n2-bit", "TQ\n3-bit", "TQ\n4-bit", "TQ RVQ\n2-bit ★"]
    colors = [
        PALETTE_V2["fp16"],
        PALETTE_V2["2bit"],
        PALETTE_V2["3bit"],
        PALETTE_V2["4bit"],
        PALETTE_V2["rvq2"],
    ]
    compress = [
        1.00,
        round(kf_2b / kc_2b, 2) if kc_2b > 0 else 0,
        round(kf_3b / kc_3b, 2) if kc_3b > 0 else 0,
        round(kf_4b / kc_4b, 2) if kc_4b > 0 else 0,
        round(kf_rvq / kc_rvq, 2) if kc_rvq > 0 else 0,
    ]
    tput = [
        len(enc(r)) / t
        for r, t in [
            (resp_fp16, t_fp16),
            (resp_2b, t_2b),
            (resp_3b, t_3b),
            (resp_4b, t_4b),
            (resp_rvq2, t_rvq2),
        ]
    ]
    tok_out = [len(enc(r)) for r in [resp_fp16, resp_2b, resp_3b, resp_4b, resp_rvq2]]
    key_kb = [kf_4b / 1024, kc_2b / 1024, kc_3b / 1024, kc_4b / 1024, kc_rvq / 1024]

    print(f"\n{'=' * 64}")
    print(f"SUMMARY (v2) — {model_label}")
    print(f"  Tokens cached       : {n_tok}")
    print(f"  TQ 2-bit (single)   : {compress[1]}× | {tput[1]:.1f} tok/s | {tok_out[1]} tokens")
    print(f"  TQ 3-bit            : {compress[2]}× | {tput[2]:.1f} tok/s | {tok_out[2]} tokens")
    print(f"  TQ 4-bit            : {compress[3]}× | {tput[3]:.1f} tok/s | {tok_out[3]} tokens")
    print(f"  TQ RVQ 2-bit ★      : {compress[4]}× | {tput[4]:.1f} tok/s | {tok_out[4]} tokens")
    print(f"  fp16 baseline       :       | {tput[0]:.1f} tok/s | {tok_out[0]} tokens")

    token_counts = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    fp16_full = token_counts * n_layers * n_kv * hd * 2 * 2

    print(f"\nComputing quality curves...")
    snrs, coss = compute_quality_curves(hd)

    # Synthetic RVQ 2-bit quality
    np.random.seed(42)
    x_raw = np.random.randn(64, hd)
    x_unit = (x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).astype(np.float16)
    x_mx = mx.array(x_unit)
    qrvq = TurboQuantRVQ(d=hd, b=2, seed=0)
    ev = qrvq.encode(x_mx)
    x_hat = qrvq.decode(ev)
    mse = float(mx.mean((x_mx - x_hat) ** 2))
    var = float(mx.mean(x_mx**2))
    snr_rvq2 = 10 * np.log10(max(var / mse, 1e-10))
    cos_rvq2 = float(
        mx.mean(
            mx.sum(x_mx * x_hat, axis=1)
            / (mx.linalg.norm(x_mx, axis=1) * mx.linalg.norm(x_hat, axis=1))
        )
    )
    print(f"  RVQ 2-bit synthetic: cosine={cos_rvq2:.4f}, SNR={snr_rvq2:.2f} dB")

    np.random.seed(7)
    N_k = 32
    q_np = np.random.randn(hd).astype(np.float32)
    q_np /= np.linalg.norm(q_np)
    k_np = np.random.randn(N_k, hd).astype(np.float32)
    k_unit = k_np / np.linalg.norm(k_np, axis=1, keepdims=True)
    q_mx = mx.array(q_np.astype(np.float16)).reshape(1, -1)
    k_mx = mx.array(k_unit.astype(np.float16))

    def _attn_tq(bits):
        qt = TurboQuantProd(d=hd, b=bits, m=min(hd, 64), seed=0)
        ev = qt.encode(k_mx)
        k_hat = qt.decode(ev)
        sc = np.array(k_hat @ q_mx.T).flatten()
        return np.exp(sc) / np.exp(sc).sum()

    def _attn_rvq(bits):
        qt = TurboQuantRVQ(d=hd, b=bits, seed=0)
        ev = qt.encode(k_mx)
        k_hat = qt.decode(ev)
        sc = np.array(k_hat @ q_mx.T).flatten()
        return np.exp(sc) / np.exp(sc).sum()

    sc_fp16 = np.array(k_mx @ q_mx.T).flatten()
    sm_fp16 = np.exp(sc_fp16) / np.exp(sc_fp16).sum()
    sm_2b = _attn_tq(2)
    sm_3b = _attn_tq(3)
    sm_4b = _attn_tq(4)
    sm_rvq2 = _attn_rvq(2)

    print(f"Generating figures → {out_dir}/")
    _generate_figures_v2(
        out_dir=out_dir,
        model_label=model_label,
        configs=configs,
        colors=colors,
        compress=compress,
        tput=tput,
        tokens_out=tok_out,
        key_kb=key_kb,
        snrs=snrs,
        coss=coss,
        bit_range=range(2, 7),
        head_dim=hd,
        n_kv_heads=n_kv,
        n_layers=n_layers,
        token_counts=token_counts,
        fp16_full=fp16_full,
        sm_fp16=sm_fp16,
        sm_2b=sm_2b,
        sm_3b=sm_3b,
        sm_4b=sm_4b,
        sm_rvq2=sm_rvq2,
        cos_rvq2=cos_rvq2,
        snr_rvq2=snr_rvq2,
        responses=[resp_fp16, resp_2b, resp_3b, resp_4b, resp_rvq2],
    )
    print(f"Done v2 — {model_label}\n")
    return compress, tput, tok_out


# ── v3: 6 configs (adds RVQ 1-bit) ────────────────────────────────────────────

PALETTE_V3 = {
    "fp16": "#4C72B0",
    "2bit": "#C44E52",
    "3bit": "#DD8452",
    "4bit": "#55A868",
    "rvq2": "#8172B2",
    "rvq1": "#E377C2",  # hot-pink — the new 1-bit star
}


def _read_model_cfg(model):
    """Return (head_dim, n_kv_heads, n_layers) robustly for any mlx_lm model.

    Handles:
    - Standard text models (Mistral, Qwen3, Llama, Phi) where model.args has
      hidden_size / num_key_value_heads / num_hidden_layers directly.
    - VLM-style wrappers (Qwen2-VL, Gemma3) where model.args.text_config is a
      plain dict containing the real attention config.
    - Falls back to layer inspection when args are missing/incomplete.
    """
    cfg = getattr(model, "args", None)
    # Unwrap a VLM args whose text_config is a dict (Gemma3, Qwen2-VL pattern)
    if cfg is not None and not hasattr(cfg, "hidden_size"):
        tc = getattr(cfg, "text_config", None)
        if isinstance(tc, dict):

            class _DictCfg:
                pass

            inner = _DictCfg()
            for k, v in tc.items():
                setattr(inner, k, v)
            cfg = inner
        else:
            lm = getattr(model, "language_model", None)
            if lm is not None:
                cfg = getattr(lm, "args", cfg)

    # Read n_kv and n_layers from cfg; defer head_dim to layer inspection
    # because the derived formula (hidden_size // n_heads) is wrong for GQA
    # models like Gemma3 where the actual per-head dim differs.
    hd = n_kv = n_layers = None
    if cfg is not None:
        n_kv = getattr(cfg, "num_key_value_heads", None)
        n_layers = getattr(cfg, "num_hidden_layers", None)
        # Only use the explicit head_dim attr — never derive from hidden_size
        hd = getattr(cfg, "head_dim", None)

    # Layer inspection — canonical source for head_dim, fills in any None
    layers = (
        getattr(model, "layers", None)
        or getattr(getattr(model, "model", None), "layers", None)
        or getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None)
        or []
    )
    if n_layers is None:
        n_layers = len(layers)
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            continue
        if hd is None:
            hd = getattr(attn, "head_dim", None)
        if n_kv is None:
            n_kv = getattr(attn, "n_kv_heads", None)
        if hd is not None and n_kv is not None:
            break

    return int(hd or 128), int(n_kv or 8), int(n_layers or 32)


def _rvq_quality_synthetic(hd, bits, seed=0, n=64):
    """Compute cosine and SNR for RVQ at given bits on synthetic Gaussian vectors."""
    np.random.seed(seed)
    x_raw = np.random.randn(n, hd)
    x_unit = (x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).astype(np.float16)
    x_mx = mx.array(x_unit)
    q = TurboQuantRVQ(d=hd, b=bits, seed=0, use_hadamard=True)
    ev = q.encode(x_mx)
    x_hat = q.decode(ev)
    mse = float(mx.mean((x_mx - x_hat) ** 2))
    var = float(mx.mean(x_mx**2))
    snr = 10 * np.log10(max(var / mse, 1e-10))
    cos = float(
        mx.mean(
            mx.sum(x_mx * x_hat, axis=1)
            / (mx.linalg.norm(x_mx, axis=1) * mx.linalg.norm(x_hat, axis=1))
        )
    )
    return cos, snr


def _generate_figures_v3(
    out_dir,
    model_label,
    configs,
    colors,
    compress,
    tput,
    tokens_out,
    key_kb,
    snrs,
    coss,
    bit_range,
    head_dim,
    n_kv_heads,
    n_layers,
    token_counts,
    fp16_full,
    sm_fp16,
    sm_2b,
    sm_3b,
    sm_4b,
    sm_rvq2,
    sm_rvq1,
    cos_rvq2,
    snr_rvq2,
    cos_rvq1,
    snr_rvq1,
    responses,
):
    """Generate 6-config (fp16/TQ2/TQ3/TQ4/RVQ2★/RVQ1★) report figures."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    x = np.arange(len(configs))
    bar_w = 0.72

    def _bar(ax, vals, ylabel, title, cols=None, hline=None, fmt=".1f"):
        c = cols or colors
        bars = ax.bar(x, vals, width=bar_w, color=c, edgecolor="white", linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if hline is not None:
            ax.axhline(hline, color="grey", ls="--", lw=1, alpha=0.7)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.02,
                f"{v:{fmt}}",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        ax.set_ylim(0, max(vals) * 1.30)
        sns.despine(ax=ax)

    to_mb = lambda b: b / 1024**2
    br = list(bit_range)
    N_k = len(sm_fp16)

    # ── Fig 1: 6-config summary ───────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle(
        f"TurboQuant KV Cache (v3 · RVQ 1-bit) — {model_label}\n"
        f"Apple M4 · head_dim={head_dim} · 6 configs",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    _bar(
        axes[0, 0],
        compress,
        "Key Compression Ratio (×)",
        "Key Compression Ratio",
        hline=1.0,
        fmt=".2f",
    )
    _bar(axes[0, 1], tput, "Tokens / second", "Generation Throughput (tok/s)", hline=tput[0])
    _bar(
        axes[1, 0], tokens_out, "Tokens Generated", "Tokens Generated (max 200)", fmt="d", hline=200
    )
    _bar(axes[1, 1], key_kb, "Key Cache Size (KB)", "Compressed Key Cache Size", fmt=".0f")
    fig1.tight_layout()
    fig1.savefig(f"{out_dir}/fig1_benchmark_summary.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig1_benchmark_summary.png")

    # ── Fig 2: Quality vs bits (both RVQ stars) ───────────────────────────────
    fig2, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(
        f"Quality vs Bits — {model_label} (head_dim={head_dim})", fontsize=13, fontweight="bold"
    )
    ax_s.plot(
        br, snrs, color=PALETTE_V3["4bit"], marker="s", lw=2.2, ms=7, label="single-pass TurboQuant"
    )
    ax_c.plot(
        br, coss, color=PALETTE_V3["4bit"], marker="s", lw=2.2, ms=7, label="single-pass TurboQuant"
    )
    ax_s.scatter(
        [2], [snr_rvq2], color=PALETTE_V3["rvq2"], marker="*", s=280, zorder=5, label="RVQ 2-bit ★"
    )
    ax_c.scatter(
        [2], [cos_rvq2], color=PALETTE_V3["rvq2"], marker="*", s=280, zorder=5, label="RVQ 2-bit ★"
    )
    ax_s.scatter(
        [1], [snr_rvq1], color=PALETTE_V3["rvq1"], marker="*", s=280, zorder=5, label="RVQ 1-bit ★"
    )
    ax_c.scatter(
        [1], [cos_rvq1], color=PALETTE_V3["rvq1"], marker="*", s=280, zorder=5, label="RVQ 1-bit ★"
    )
    ax_s.axhline(10, color="green", ls="--", lw=1.5, alpha=0.7, label="10 dB (near-lossless)")
    ax_c.axhline(0.90, color="green", ls="--", lw=1.5, alpha=0.7, label="0.90 near-lossless")
    ax_c.axhline(0.80, color="orange", ls="--", lw=1.5, alpha=0.7, label="0.80 degraded")
    for ax, lbl in [(ax_s, "SNR (dB)"), (ax_c, "Cosine Similarity")]:
        ax.set_xlabel("Bit-width")
        ax.set_ylabel(lbl)
        ax.set_xticks(br)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
    ax_s.set_title("Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
    ax_c.set_title("Cosine Similarity", fontsize=12, fontweight="bold")
    ax_c.set_ylim(0.4, 1.05)
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/fig2_quality_vs_bits.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig2_quality_vs_bits.png")

    # ── Fig 3: Memory at scale (two RVQ lines) ────────────────────────────────
    def tq_bytes(tokens, bits):
        b_mse = max(bits - 1, 1)
        m = min(head_dim, 64)
        per = (math.ceil(head_dim * b_mse / 8) + math.ceil(m / 8) + 2 + 2) * n_kv_heads * n_layers
        return tokens * per

    def rvq_bytes(tokens, bits):
        per = (math.ceil(head_dim * 2 * bits / 8) + 2) * n_kv_heads * n_layers
        return tokens * per

    tq2_ = np.array([tq_bytes(t, 2) for t in token_counts])
    tq3_ = np.array([tq_bytes(t, 3) for t in token_counts])
    tq4_ = np.array([tq_bytes(t, 4) for t in token_counts])
    rvq2_ = np.array([rvq_bytes(t, 2) for t in token_counts])
    rvq1_ = np.array([rvq_bytes(t, 1) for t in token_counts])
    val = token_counts * n_layers * n_kv_heads * head_dim * 2

    fig3, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        f"KV Cache Memory at Scale — {model_label}\n"
        f"({n_layers} layers, head_dim={head_dim}, kv_heads={n_kv_heads})",
        fontsize=12,
        fontweight="bold",
    )
    ax_a.plot(
        token_counts,
        to_mb(fp16_full),
        color=PALETTE_V3["fp16"],
        lw=2.5,
        marker="o",
        ms=5,
        label="fp16 K+V",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq2_ + val),
        color=PALETTE_V3["2bit"],
        lw=2.0,
        marker="D",
        ms=4,
        label="TQ 2-bit",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq3_ + val),
        color=PALETTE_V3["3bit"],
        lw=2.0,
        marker="s",
        ms=4,
        label="TQ 3-bit",
    )
    ax_a.plot(
        token_counts,
        to_mb(tq4_ + val),
        color=PALETTE_V3["4bit"],
        lw=2.0,
        marker="^",
        ms=4,
        label="TQ 4-bit",
    )
    ax_a.plot(
        token_counts,
        to_mb(rvq2_ + val),
        color=PALETTE_V3["rvq2"],
        lw=2.5,
        marker="*",
        ms=8,
        label="RVQ 2-bit ★",
    )
    ax_a.plot(
        token_counts,
        to_mb(rvq1_ + val),
        color=PALETTE_V3["rvq1"],
        lw=2.5,
        marker="P",
        ms=7,
        label="RVQ 1-bit ★",
    )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(token_counts)
    ax_a.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8
    )
    ax_a.set_xlabel("Context length")
    ax_a.set_ylabel("Memory (MB)")
    ax_a.set_title("Absolute Memory", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=8)
    sns.despine(ax=ax_a)

    r2 = fp16_full / (tq2_ + val)
    r3 = fp16_full / (tq3_ + val)
    r4 = fp16_full / (tq4_ + val)
    rr = fp16_full / (rvq2_ + val)
    r1 = fp16_full / (rvq1_ + val)
    ax_r.plot(
        token_counts, r2, color=PALETTE_V3["2bit"], lw=2.0, marker="D", ms=4, label="TQ 2-bit"
    )
    ax_r.plot(
        token_counts, r3, color=PALETTE_V3["3bit"], lw=2.0, marker="s", ms=4, label="TQ 3-bit"
    )
    ax_r.plot(
        token_counts, r4, color=PALETTE_V3["4bit"], lw=2.0, marker="^", ms=4, label="TQ 4-bit"
    )
    ax_r.plot(
        token_counts, rr, color=PALETTE_V3["rvq2"], lw=2.5, marker="*", ms=8, label="RVQ 2-bit ★"
    )
    ax_r.plot(
        token_counts, r1, color=PALETTE_V3["rvq1"], lw=2.5, marker="P", ms=7, label="RVQ 1-bit ★"
    )
    ax_r.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.6)
    ax_r.set_xscale("log", base=2)
    ax_r.set_xticks(token_counts)
    ax_r.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8
    )
    ax_r.set_xlabel("Context length")
    ax_r.set_ylabel("Compression vs fp16")
    ax_r.set_title("Compression Ratio", fontsize=12, fontweight="bold")
    ax_r.legend(fontsize=8)
    sns.despine(ax=ax_r)
    fig3.tight_layout()
    fig3.savefig(f"{out_dir}/fig3_memory_at_scale.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig3_memory_at_scale.png")

    # ── Fig 4: Attention distortion (6 panels) ────────────────────────────────
    fig4, axes4 = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig4.suptitle(
        f"Attention Score Distortion — {model_label} (head_dim={head_dim})\n"
        f"{N_k} key vectors, query dot-product, softmax",
        fontsize=12,
        fontweight="bold",
    )
    panels = [
        (sm_fp16, "fp16 Baseline (reference)", PALETTE_V3["fp16"]),
        (sm_2b, f"TurboQuant 2-bit (single-pass)  cos≈{coss[0]:.2f}", PALETTE_V3["2bit"]),
        (sm_3b, f"TurboQuant 3-bit  cos≈{coss[1]:.2f}", PALETTE_V3["3bit"]),
        (sm_4b, f"TurboQuant 4-bit  cos≈{coss[2]:.2f}", PALETTE_V3["4bit"]),
        (sm_rvq2, f"TurboQuant RVQ 2-bit ★  cos≈{cos_rvq2:.2f}", PALETTE_V3["rvq2"]),
        (sm_rvq1, f"TurboQuant RVQ 1-bit ★  cos≈{cos_rvq1:.2f}", PALETTE_V3["rvq1"]),
    ]
    for ax, (sm, label, col) in zip(axes4, panels):
        ax.bar(np.arange(N_k), sm, color=col, alpha=0.78, edgecolor="white", lw=0.5)
        ax.plot(
            np.arange(N_k),
            sm_fp16,
            color=PALETTE_V3["fp16"],
            lw=1.4,
            ls="--",
            alpha=0.5,
            label="fp16 ref",
        )
        mse_a = np.mean((sm - sm_fp16) ** 2)
        cos_a = np.dot(sm, sm_fp16) / (np.linalg.norm(sm) * np.linalg.norm(sm_fp16) + 1e-12)
        ax.set_ylabel("Attn weight")
        ax.set_title(
            f"{label}   |   MSE={mse_a:.2e}   cosine={cos_a:.4f}", fontsize=10, fontweight="bold"
        )
        ax.set_ylim(0, max(sm_fp16) * 1.45)
        sns.despine(ax=ax)
    axes4[-1].set_xlabel("Key Token Index")
    fig4.tight_layout()
    fig4.savefig(f"{out_dir}/fig4_attention_distortion.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig4_attention_distortion.png")

    # ── Fig 5: Output text comparison (6 panels) ─────────────────────────────
    resp_labels = [
        "fp16 Baseline",
        "TurboQuant 2-bit",
        "TurboQuant 3-bit",
        "TurboQuant 4-bit",
        "TurboQuant RVQ 2-bit ★",
        "TurboQuant RVQ 1-bit ★",
    ]
    fig5, axes5 = plt.subplots(6, 1, figsize=(16, 20))
    fig5.suptitle(f"Generated Output Comparison — {model_label}", fontsize=13, fontweight="bold")
    for ax, resp, lbl, col in zip(axes5, responses, resp_labels, colors):
        ax.set_facecolor(col + "18")
        ax.text(
            0.01,
            0.97,
            f"[{lbl}]",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=col,
            va="top",
        )
        wrapped = resp[:500] + ("..." if len(resp) > 500 else "")
        ax.text(
            0.01,
            0.80,
            wrapped,
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig5.tight_layout()
    fig5.savefig(f"{out_dir}/fig5_output_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig5_output_comparison.png")

    # ── Fig 6: Combined full report ───────────────────────────────────────────
    fig6 = plt.figure(figsize=(22, 24))
    fig6.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(3, 2, figure=fig6, hspace=0.46, wspace=0.36)

    ax_A = fig6.add_subplot(gs[0, 0])
    bars = ax_A.bar(configs, compress, color=colors, edgecolor="white", lw=1.1)
    ax_A.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, compress):
        ax_A.text(
            b.get_x() + b.get_width() / 2,
            v + 0.05,
            f"{v:.2f}×",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_A.set_title("A  Key Compression Ratio", fontsize=11, fontweight="bold", loc="left")
    ax_A.set_ylabel("Ratio vs fp16")
    sns.despine(ax=ax_A)
    ax_A.set_ylim(0, max(compress) * 1.30)
    ax_A.tick_params(axis="x", labelsize=7)

    ax_B = fig6.add_subplot(gs[0, 1])
    bars = ax_B.bar(configs, tput, color=colors, edgecolor="white", lw=1.1)
    ax_B.axhline(tput[0], color="grey", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, tput):
        ax_B.text(
            b.get_x() + b.get_width() / 2,
            v + 0.4,
            f"{v:.1f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_B.set_title("B  Throughput (tok/s)", fontsize=11, fontweight="bold", loc="left")
    ax_B.set_ylabel("tok/s")
    sns.despine(ax=ax_B)
    ax_B.set_ylim(0, max(tput) * 1.30)
    ax_B.tick_params(axis="x", labelsize=7)

    ax_C = fig6.add_subplot(gs[1, 0])
    ax_C.plot(br, coss, color=PALETTE_V3["4bit"], marker="s", lw=2.2, ms=7, label="single-pass TQ")
    ax_C.scatter(
        [2], [cos_rvq2], color=PALETTE_V3["rvq2"], marker="*", s=280, zorder=5, label="RVQ 2-bit ★"
    )
    ax_C.scatter(
        [1], [cos_rvq1], color=PALETTE_V3["rvq1"], marker="*", s=280, zorder=5, label="RVQ 1-bit ★"
    )
    ax_C.axhline(0.90, color="green", ls="--", lw=1.4, alpha=0.7, label="0.90")
    ax_C.axhline(0.80, color="orange", ls="--", lw=1.4, alpha=0.7, label="0.80")
    ax_C.set_xlabel("Bit-width")
    ax_C.set_ylabel("Cosine Similarity")
    ax_C.set_xticks(br)
    ax_C.set_ylim(0.4, 1.05)
    ax_C.set_title("C  Quality vs Bits", fontsize=11, fontweight="bold", loc="left")
    ax_C.legend(fontsize=8)
    sns.despine(ax=ax_C)

    ax_D = fig6.add_subplot(gs[1, 1])
    ax_D.plot(
        token_counts,
        to_mb(fp16_full),
        color=PALETTE_V3["fp16"],
        lw=2.5,
        marker="o",
        ms=5,
        label="fp16",
    )
    ax_D.plot(
        token_counts,
        to_mb(tq2_ + val),
        color=PALETTE_V3["2bit"],
        lw=2.0,
        marker="D",
        ms=4,
        label="TQ 2-bit",
    )
    ax_D.plot(
        token_counts,
        to_mb(tq4_ + val),
        color=PALETTE_V3["4bit"],
        lw=2.0,
        marker="^",
        ms=4,
        label="TQ 4-bit",
    )
    ax_D.plot(
        token_counts,
        to_mb(rvq2_ + val),
        color=PALETTE_V3["rvq2"],
        lw=2.5,
        marker="*",
        ms=8,
        label="RVQ 2-bit ★",
    )
    ax_D.plot(
        token_counts,
        to_mb(rvq1_ + val),
        color=PALETTE_V3["rvq1"],
        lw=2.5,
        marker="P",
        ms=7,
        label="RVQ 1-bit ★",
    )
    ax_D.set_xscale("log", base=2)
    ax_D.set_xticks(token_counts)
    ax_D.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=7
    )
    ax_D.set_xlabel("Context length")
    ax_D.set_ylabel("Memory (MB)")
    ax_D.set_title("D  KV Cache Memory at Scale", fontsize=11, fontweight="bold", loc="left")
    ax_D.legend(fontsize=8)
    sns.despine(ax=ax_D)

    ax_E = fig6.add_subplot(gs[2, :])
    w = 0.13
    offsets = np.linspace(-2.5 * w, 2.5 * w, 6)
    for (sm, col, lbl), off in zip(
        [
            (sm_fp16, PALETTE_V3["fp16"], "fp16"),
            (sm_2b, PALETTE_V3["2bit"], "TQ 2-bit"),
            (sm_3b, PALETTE_V3["3bit"], "TQ 3-bit"),
            (sm_4b, PALETTE_V3["4bit"], "TQ 4-bit"),
            (sm_rvq2, PALETTE_V3["rvq2"], "RVQ 2-bit ★"),
            (sm_rvq1, PALETTE_V3["rvq1"], "RVQ 1-bit ★"),
        ],
        offsets,
    ):
        ax_E.bar(np.arange(N_k) + off, sm, width=w, color=col, alpha=0.85, label=lbl)
    ax_E.set_xlabel("Key Token Index")
    ax_E.set_ylabel("Attention weight")
    ax_E.set_title(
        f"E  Attention Distortion (head_dim={head_dim})", fontsize=11, fontweight="bold", loc="left"
    )
    ax_E.legend(fontsize=8, ncol=3)
    sns.despine(ax=ax_E)

    fig6.suptitle(
        f"TurboQuant KV Cache (v3 · 6 configs) — {model_label}\n"
        f"Apple M4 · veloxquant_mlx · RVQ 1-bit & 2-bit ★",
        fontsize=15,
        fontweight="bold",
        y=1.005,
    )
    fig6.savefig(f"{out_dir}/fig6_full_report.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig6_full_report.png")
    plt.close("all")


def run_benchmark_v3_from_results(
    results_by_config, out_dir, model_label, head_dim, n_kv_heads, n_layers
):
    """Build figures from pre-collected per-config result dicts.

    Each value in results_by_config must have keys:
        tps, toks, elapsed, ratio_num, response
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    R = results_by_config

    def _tps(k):
        return R[k]["tps"] if k in R else 0.0

    def _toks(k):
        return R[k]["toks"] if k in R else 0

    def _rat(k):
        return R[k]["ratio_num"] if k in R else 1.0

    def _resp(k):
        return R[k]["response"] if k in R else ""

    hd, n_kv, nl = head_dim, n_kv_heads, n_layers

    # Compression ratios
    kf_ref = _rat("fp16") or 1.0  # always 1.0 for fp16
    compress = [
        1.00,
        _rat("tq2"),
        _rat("tq3"),
        _rat("tq4"),
        _rat("rvq2"),
        _rat("rvq1"),
    ]
    tput = [_tps(k) for k in ["fp16", "tq2", "tq3", "tq4", "rvq2", "rvq1"]]
    tok_out = [_toks(k) for k in ["fp16", "tq2", "tq3", "tq4", "rvq2", "rvq1"]]
    responses = [_resp(k) for k in ["fp16", "tq2", "tq3", "tq4", "rvq2", "rvq1"]]
    key_kb = [
        R.get("fp16", {}).get("fp16_key_bytes", 0) / 1024,
        R.get("tq2", {}).get("compressed_key_bytes", 0) / 1024,
        R.get("tq3", {}).get("compressed_key_bytes", 0) / 1024,
        R.get("tq4", {}).get("compressed_key_bytes", 0) / 1024,
        R.get("rvq2", {}).get("compressed_key_bytes", 0) / 1024,
        R.get("rvq1", {}).get("compressed_key_bytes", 0) / 1024,
    ]

    configs = [
        "fp16\nbaseline",
        "TQ\n2-bit",
        "TQ\n3-bit",
        "TQ\n4-bit",
        "RVQ\n2-bit ★",
        "RVQ\n1-bit ★",
    ]
    colors = [
        PALETTE_V3["fp16"],
        PALETTE_V3["2bit"],
        PALETTE_V3["3bit"],
        PALETTE_V3["4bit"],
        PALETTE_V3["rvq2"],
        PALETTE_V3["rvq1"],
    ]

    token_counts = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    fp16_full = token_counts * nl * n_kv * hd * 2 * 2

    print(f"\nComputing quality curves for head_dim={hd}...")
    snrs, coss = compute_quality_curves(hd)
    cos_rvq2, snr_rvq2 = _rvq_quality_synthetic(hd, 2)
    cos_rvq1, snr_rvq1 = _rvq_quality_synthetic(hd, 1)
    print(f"  RVQ 2-bit: cos={cos_rvq2:.4f}  SNR={snr_rvq2:.2f} dB")
    print(f"  RVQ 1-bit: cos={cos_rvq1:.4f}  SNR={snr_rvq1:.2f} dB")

    # Attention distortion on synthetic keys
    np.random.seed(7)
    N_k = 32
    q_np = np.random.randn(hd).astype(np.float32)
    q_np /= np.linalg.norm(q_np)
    k_np = np.random.randn(N_k, hd).astype(np.float32)
    k_unit = k_np / np.linalg.norm(k_np, axis=1, keepdims=True)
    q_mx = mx.array(q_np.astype(np.float16)).reshape(1, -1)
    k_mx = mx.array(k_unit.astype(np.float16))

    def _attn_tq(bits):
        qt = TurboQuantProd(d=hd, b=bits, m=min(hd, 64), seed=0, use_hadamard=True)
        ev = qt.encode(k_mx)
        k_hat = qt.decode(ev)
        sc = np.array(k_hat @ q_mx.T).flatten()
        return np.exp(sc) / np.exp(sc).sum()

    def _attn_rvq(bits):
        qt = TurboQuantRVQ(d=hd, b=bits, seed=0, use_hadamard=True)
        ev = qt.encode(k_mx)
        k_hat = qt.decode(ev)
        sc = np.array(k_hat @ q_mx.T).flatten()
        return np.exp(sc) / np.exp(sc).sum()

    sc_fp16 = np.array(k_mx @ q_mx.T).flatten()
    sm_fp16 = np.exp(sc_fp16) / np.exp(sc_fp16).sum()
    sm_2b = _attn_tq(2)
    sm_3b = _attn_tq(3)
    sm_4b = _attn_tq(4)
    sm_rvq2 = _attn_rvq(2)
    sm_rvq1 = _attn_rvq(1)

    print(f"Generating figures → {out_dir}/")
    _generate_figures_v3(
        out_dir=out_dir,
        model_label=model_label,
        configs=configs,
        colors=colors,
        compress=compress,
        tput=tput,
        tokens_out=tok_out,
        key_kb=key_kb,
        snrs=snrs,
        coss=coss,
        bit_range=range(2, 7),
        head_dim=hd,
        n_kv_heads=n_kv,
        n_layers=nl,
        token_counts=token_counts,
        fp16_full=fp16_full,
        sm_fp16=sm_fp16,
        sm_2b=sm_2b,
        sm_3b=sm_3b,
        sm_4b=sm_4b,
        sm_rvq2=sm_rvq2,
        sm_rvq1=sm_rvq1,
        cos_rvq2=cos_rvq2,
        snr_rvq2=snr_rvq2,
        cos_rvq1=cos_rvq1,
        snr_rvq1=snr_rvq1,
        responses=responses,
    )

    print(f"\n{'=' * 64}")
    print(f"SUMMARY (v3) — {model_label}")
    print(f"{'Config':<22} {'tok/s':>8} {'tokens':>8} {'compression':>13}")
    print(f"{'-' * 64}")
    labels = ["fp16 baseline", "TQ 2-bit", "TQ 3-bit", "TQ 4-bit", "RVQ 2-bit ★", "RVQ 1-bit ★"]
    for lbl, tps, toks, comp in zip(labels, tput, tok_out, compress):
        c = f"{comp:.2f}×" if comp != 1.0 else "—"
        print(f"  {lbl:<20} {tps:>8.1f} {toks:>8} {c:>13}")
    print(f"Done v3 — {model_label}\n")
