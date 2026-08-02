"""
TurboQuant KV cache benchmark on Gemma 4 4B (e4b-it-4bit).

Architecture note:
  Gemma 4 is hybrid: 35 sliding-attention + 7 full-attention layers.
  TurboQuant only targets the 4 full-attention KVCache objects (unbounded growth).
  The 20 RotatingKVCache (sliding-window=512) are left untouched.
  Full-attention head_dim = 512, kv_heads = 2.

Figures saved to: figures/gemma4/

Usage:
    python benchmark_gemma4.py
"""

import math
import time
from collections import Counter
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlx.core as mx
import mlx_lm.utils as _mlx_utils
import numpy as np
import seaborn as sns
from mlx_lm.models.cache import KVCache as _MLXKVCache
from mlx_lm.models.cache import RotatingKVCache

from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
OUT_DIR = "figures/gemma4"
PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)
MAX_TOKENS = 200
PALETTE = {
    "fp16": "#4C72B0",
    "3bit": "#DD8452",
    "4bit": "#55A868",
    "4bit_out": "#C44E52",
    "sliding": "#8172B2",
}


# ── Patched loader (strict=False for Gemma 4 multimodal weight keys) ──────────
_orig_load_model = _mlx_utils.load_model


def _load_model_non_strict(path, lazy=False, strict=True, model_config=None, **kw):
    return _orig_load_model(path, lazy=lazy, strict=False, model_config=model_config or {}, **kw)


_mlx_utils.load_model = _load_model_non_strict


# ── TurboQuant KV cache wrapper ────────────────────────────────────────────────
class TurboQuantMLXKVCache(_MLXKVCache):
    """Compresses keys via TurboQuantProd with per-vector normalization."""

    def __init__(self, n_kv_heads: int, head_dim: int, bits: int = 4, seed: int = 42) -> None:
        super().__init__()
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._bits = bits
        m = min(head_dim, 64)
        self._quantizers = [
            TurboQuantProd(d=head_dim, b=bits, m=m, seed=seed + i) for i in range(n_kv_heads)
        ]
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0

    def update_and_fetch(self, keys, values):
        B, H, S, _ = keys.shape
        head_results = []
        for h in range(H):
            batch_results = []
            for b in range(B):
                kv_f32 = keys[b, h, :, :].astype(mx.float32)
                norms = mx.linalg.norm(kv_f32, axis=-1, keepdims=True)
                safe_norms = mx.where(norms < 1e-8, mx.ones_like(norms), norms)
                kv_unit = (kv_f32 / safe_norms).astype(mx.float16)
                ev = self._quantizers[h].encode(kv_unit)
                k_unit_hat = self._quantizers[h].decode(ev)
                k_hat = (k_unit_hat.astype(mx.float32) * safe_norms).astype(keys.dtype)
                batch_results.append(k_hat)
            head_results.append(mx.stack(batch_results, axis=0))
        k_dequant = mx.stack(head_results, axis=1)

        b_mse = max(self._bits - 1, 1)
        m_eff = self._quantizers[0]._m_eff
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_tq_caches(base_caches, full_head_dim: int, full_kv_heads: int, bits: int):
    """
    Replace KVCache (full-attention) entries with TurboQuant wrappers.
    RotatingKVCache (sliding-window) entries are kept as-is.
    """
    result, idx = [], 0
    for c in base_caches:
        if isinstance(c, RotatingKVCache):
            result.append(c)
        else:
            result.append(
                TurboQuantMLXKVCache(
                    n_kv_heads=full_kv_heads,
                    head_dim=full_head_dim,
                    bits=bits,
                    seed=idx,
                )
            )
            idx += 1
    return result


def run(model, tokenizer, cache_factory, label: str):
    messages = [{"role": "user", "content": PROMPT}]
    prompt_txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    import mlx_lm

    original_make_cache = model.make_cache
    injected = []
    if cache_factory is not None:

        def _patch(*_, **__):
            c = cache_factory()
            injected.extend(c)
            return c

        model.make_cache = _patch

    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt_txt,
        max_tokens=MAX_TOKENS,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    model.make_cache = original_make_cache

    tq_caches = [c for c in injected if isinstance(c, TurboQuantMLXKVCache)]
    k_fp16 = sum(c.fp16_key_bytes for c in tq_caches)
    k_cmp = sum(c.compressed_key_bytes for c in tq_caches)
    ratio = f"{k_fp16 / k_cmp:.2f}×" if k_cmp > 0 else "—"
    toks = len(tokenizer.encode(response))

    print(f"\n{'=' * 64}")
    print(f"[{label}]  key compression: {ratio}")
    print(f"  {response[:500]}{'...' if len(response) > 500 else ''}")
    print(f"  {toks} tokens  {elapsed:.1f}s  ({toks / elapsed:.1f} tok/s)")
    return response, elapsed, injected


# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID}  (strict=False patch active)...")
from mlx_lm.utils import _download, load_tokenizer

model_path = _download(MODEL_ID)
model, _ = _mlx_utils.load_model(model_path, lazy=False, strict=False)
tokenizer = load_tokenizer(model_path)

tc = model.args.text_config
full_head_dim = tc["global_head_dim"]  # 512
local_hd = tc["head_dim"]  # 256
full_kv = tc["num_key_value_heads"]  # 2
n_layers = tc["num_hidden_layers"]  # 42
layer_types = tc["layer_types"]
n_full = Counter(layer_types)["full_attention"]
n_slid = Counter(layer_types)["sliding_attention"]

base_caches = model.make_cache()
n_kv_caches = sum(
    1 for c in base_caches if isinstance(c, _MLXKVCache) and not isinstance(c, RotatingKVCache)
)

print(f"\nModel architecture:")
print(f"  {n_layers} layers  ({n_slid} sliding + {n_full} full-attention)")
print(f"  full-attention: head_dim={full_head_dim}, kv_heads={full_kv}")
print(f"  sliding-window: head_dim={local_hd}, window=512")
print(f"  KVCache objects to quantize: {n_kv_caches}\n")


# ── Benchmark runs ─────────────────────────────────────────────────────────────
resp_fp16, t_fp16, c_fp16 = run(model, tokenizer, None, "fp16 baseline")

_orig_cache = model.make_cache  # capture before any patching

resp_3b, t_3b, c_3b = run(
    model,
    tokenizer,
    lambda: build_tq_caches(_orig_cache(), full_head_dim, full_kv, bits=3),
    "TurboQuant 3-bit",
)

resp_4b, t_4b, c_4b = run(
    model,
    tokenizer,
    lambda: build_tq_caches(_orig_cache(), full_head_dim, full_kv, bits=4),
    "TurboQuant 4-bit",
)


# ── Collect stats ──────────────────────────────────────────────────────────────
def stats(caches):
    tq = [c for c in caches if isinstance(c, TurboQuantMLXKVCache)]
    kf = sum(c.fp16_key_bytes for c in tq)
    kc = sum(c.compressed_key_bytes for c in tq)
    n_tok = max((c.offset for c in caches if hasattr(c, "offset")), default=0)
    return kf, kc, n_tok


kf_3b, kc_3b, n_tok_3b = stats(c_3b)
kf_4b, kc_4b, n_tok_4b = stats(c_4b)
n_tok = max(n_tok_3b, n_tok_4b)

print(f"\n{'=' * 64}")
print("SUMMARY")
print(f"  Architecture : Gemma 4 4B, {n_full} full-attention layers quantized")
print(f"  Tokens cached: {n_tok}")
print(f"  fp16 full-attn key size  : {kf_4b / 1024:.1f} KB")
print(f"  TQ 3-bit key size        : {kc_3b / 1024:.1f} KB  ({kf_3b / kc_3b:.2f}×)")
print(f"  TQ 4-bit key size        : {kc_4b / 1024:.1f} KB  ({kf_4b / kc_4b:.2f}×)")
print(
    f"\n  Throughput — fp16:{len(tokenizer.encode(resp_fp16)) / t_fp16:.1f} | "
    f"3-bit:{len(tokenizer.encode(resp_3b)) / t_3b:.1f} | "
    f"4-bit:{len(tokenizer.encode(resp_4b)) / t_4b:.1f} tok/s"
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES  (saved to figures/gemma4/)
# ═══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", font_scale=1.15)

CONFIGS = ["fp16\nbaseline", "TurboQuant\n3-bit", "TurboQuant\n4-bit"]
COLORS = [PALETTE["fp16"], PALETTE["3bit"], PALETTE["4bit"]]
COMPRESS = [
    1.00,
    round(kf_3b / kc_3b, 2) if kc_3b > 0 else 0,
    round(kf_4b / kc_4b, 2) if kc_4b > 0 else 0,
]
TPUT = [
    len(tokenizer.encode(resp_fp16)) / t_fp16,
    len(tokenizer.encode(resp_3b)) / t_3b,
    len(tokenizer.encode(resp_4b)) / t_4b,
]
TOKENS_OUT = [
    len(tokenizer.encode(resp_fp16)),
    len(tokenizer.encode(resp_3b)),
    len(tokenizer.encode(resp_4b)),
]
KEY_KB = [kf_4b / 1024, kc_3b / 1024, kc_4b / 1024]

x = np.arange(len(CONFIGS))
bar_w = 0.55


def _bar(ax, vals, ylabel, title, colors=COLORS, hline=None, fmt=".1f"):
    bars = ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if hline:
        ax.axhline(hline, color="grey", linestyle="--", lw=1, alpha=0.7)
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


# ── Fig 1: Benchmark summary ───────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    "TurboQuant KV Cache — Gemma 4 4B (hybrid: 35 sliding + 7 full-attention)\n"
    "Apple M4 · head_dim=512 (full-attn) · kv_heads=2",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
_bar(
    axes[0, 0], COMPRESS, "Key Compression Ratio (×)", "Key Compression Ratio", hline=1.0, fmt=".2f"
)
_bar(axes[0, 1], TPUT, "Tokens / second", "Generation Throughput (tok/s)", hline=TPUT[0])
_bar(axes[1, 0], TOKENS_OUT, "Tokens Generated", "Tokens Generated (max 200)", fmt="d", hline=200)
_bar(axes[1, 1], KEY_KB, "Full-Attention Key Cache (KB)", "Full-Attention Key Cache Size")
fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}/fig1_benchmark_summary.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/fig1_benchmark_summary.png")


# ── Fig 2: Hybrid architecture diagram ────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(16, 4))
fig2.suptitle(
    "Gemma 4 4B Hybrid Architecture — Which Layers TurboQuant Targets",
    fontsize=14,
    fontweight="bold",
)
for i, lt in enumerate(layer_types):
    col = PALETTE["sliding"] if lt == "sliding_attention" else PALETTE["4bit"]
    ax2.barh(0, 1, left=i, height=0.6, color=col, edgecolor="white", linewidth=0.8)
    if lt == "full_attention":
        ax2.text(
            i + 0.5, 0, "TQ", ha="center", va="center", fontsize=7, color="white", fontweight="bold"
        )

ax2.set_xlim(0, n_layers)
ax2.set_yticks([])
ax2.set_xlabel("Layer index (0 → 41)", fontsize=12)

patches = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=PALETTE["sliding"],
        label="Sliding-attention  (RotatingKVCache, window=512 — NOT quantized)",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=PALETTE["4bit"],
        label="Full-attention     (KVCache, unbounded — TurboQuant applied)",
    ),
]
ax2.legend(handles=patches, loc="upper right", fontsize=10)
ax2.set_title(
    f"42 total layers  ·  {n_slid} sliding (purple)  +  {n_full} full (green, labelled TQ)",
    fontsize=11,
)
sns.despine(ax=ax2, left=True)
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/fig2_hybrid_architecture.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/fig2_hybrid_architecture.png")


# ── Fig 3: Quality vs bits at head_dim=512 ────────────────────────────────────
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd as TQP

bit_range = [2, 3, 4, 5, 6]
dims = [256, 512]
dim_cols = [PALETTE["3bit"], PALETTE["4bit"]]
snr_r, cos_r = {}, {}

print("\nComputing SNR curves for fig3...")
np.random.seed(42)
for d in dims:
    snrs, coss = [], []
    x_raw = np.random.randn(64, d)
    x_unit = (x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).astype(np.float16)
    x_mx = mx.array(x_unit)
    for b in bit_range:
        q = TQP(d=d, b=b, m=min(d, 64), seed=0)
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
    snr_r[d] = snrs
    cos_r[d] = coss

fig3, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Quality vs Bits — Gemma 4 head dimensions", fontsize=14, fontweight="bold")
for d, col in zip(dims, dim_cols):
    ax_s.plot(
        bit_range, snr_r[d], color=col, marker="s", lw=2.2, markersize=7, label=f"head_dim={d}"
    )
    ax_c.plot(
        bit_range, cos_r[d], color=col, marker="s", lw=2.2, markersize=7, label=f"head_dim={d}"
    )

ax_s.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7, label="0 dB (noise=signal)")
ax_s.axhline(10, color="green", ls="--", lw=1.5, alpha=0.7, label="10 dB (near-lossless)")
ax_c.axhline(0.90, color="green", ls="--", lw=1.5, alpha=0.7, label="0.90 (near-lossless)")
ax_c.axhline(0.80, color="orange", ls="--", lw=1.5, alpha=0.7, label="0.80 (degraded)")

# Annotate our two runs
for ax, data, yoff in [(ax_c, cos_r, 0)]:
    for d, col, yo in zip(dims, dim_cols, [0.03, -0.05]):
        for b_idx, b in enumerate([3, 4]):
            v = data[d][b_idx + 1]
            ax.annotate(
                f"{b}b→{v:.2f}",
                xy=(b, v),
                xytext=(b + 0.1, v + yo),
                fontsize=8,
                color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
            )

for ax, lbl in [(ax_s, "SNR (dB)"), (ax_c, "Cosine Similarity")]:
    ax.set_xlabel("Bit-width")
    ax.set_ylabel(lbl)
    ax.set_xticks(bit_range)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
ax_s.set_title("Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
ax_c.set_title("Cosine Similarity (original vs TQ)", fontsize=12, fontweight="bold")
ax_c.set_ylim(0.4, 1.05)
fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}/fig3_quality_vs_bits.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/fig3_quality_vs_bits.png")


# ── Fig 4: Memory at scale — only full-attention layers ───────────────────────
token_counts = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])


def tq_bytes(tokens, bits, hd=512, kv=2, n_fa=n_kv_caches):
    b_mse = max(bits - 1, 1)
    m = min(hd, 64)
    per = (math.ceil(hd * b_mse / 8) + math.ceil(m / 8) + 2 + 2) * kv * n_fa
    return tokens * per


fp16_fa = token_counts * n_kv_caches * full_kv * full_head_dim * 2 * 2  # K+V
fp16_full_kv = token_counts * n_layers * full_kv * full_head_dim * 2 * 2  # hypothetical
tq3_b = np.array([tq_bytes(t, 3) for t in token_counts])
tq4_b = np.array([tq_bytes(t, 4) for t in token_counts])
val_fa = token_counts * n_kv_caches * full_kv * full_head_dim * 2  # values fp16

fig4, (ax_abs, ax_rat) = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(
    "Full-Attention KV Cache Memory at Scale — Gemma 4 4B\n"
    f"({n_kv_caches} full-attention KVCache objects, head_dim={full_head_dim}, kv_heads={full_kv})",
    fontsize=13,
    fontweight="bold",
)
to_mb = lambda b: b / 1024**2
ax_abs.plot(
    token_counts,
    to_mb(fp16_fa),
    color=PALETTE["fp16"],
    marker="o",
    lw=2.5,
    markersize=5,
    label="fp16 K+V (full-attn layers)",
)
ax_abs.plot(
    token_counts,
    to_mb(tq3_b + val_fa),
    color=PALETTE["3bit"],
    marker="s",
    lw=2.5,
    markersize=5,
    label="TQ 3-bit keys + fp16 values",
)
ax_abs.plot(
    token_counts,
    to_mb(tq4_b + val_fa),
    color=PALETTE["4bit"],
    marker="^",
    lw=2.5,
    markersize=5,
    label="TQ 4-bit keys + fp16 values",
)
ax_abs.set_xscale("log", base=2)
ax_abs.set_xticks(token_counts)
ax_abs.set_xticklabels([f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=9)
ax_abs.set_xlabel("Context Length (tokens)")
ax_abs.set_ylabel("Memory (MB)")
ax_abs.set_title("Absolute Memory", fontsize=12, fontweight="bold")
ax_abs.legend(fontsize=9)
sns.despine(ax=ax_abs)

r3 = fp16_fa / (tq3_b + val_fa)
r4 = fp16_fa / (tq4_b + val_fa)
ax_rat.plot(
    token_counts, r3, color=PALETTE["3bit"], marker="s", lw=2.5, markersize=5, label="TQ 3-bit"
)
ax_rat.plot(
    token_counts, r4, color=PALETTE["4bit"], marker="^", lw=2.5, markersize=5, label="TQ 4-bit"
)
ax_rat.fill_between(token_counts, r4, 1.0, alpha=0.12, color=PALETTE["4bit"])
ax_rat.fill_between(token_counts, r3, r4, alpha=0.10, color=PALETTE["3bit"])
ax_rat.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.6)
ax_rat.set_xscale("log", base=2)
ax_rat.set_xticks(token_counts)
ax_rat.set_xticklabels([f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=9)
ax_rat.set_xlabel("Context Length (tokens)")
ax_rat.set_ylabel("Compression vs fp16")
ax_rat.set_title("Compression Ratio vs fp16", fontsize=12, fontweight="bold")
ax_rat.legend(fontsize=9)
sns.despine(ax=ax_rat)
fig4.tight_layout()
fig4.savefig(f"{OUT_DIR}/fig4_memory_at_scale.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/fig4_memory_at_scale.png")


# ── Fig 5: Attention distortion at head_dim=512 ───────────────────────────────
np.random.seed(7)
N_k, d = 32, 512
q_np = np.random.randn(d).astype(np.float32)
q_np /= np.linalg.norm(q_np)
k_np = np.random.randn(N_k, d).astype(np.float32)
k_unit = k_np / np.linalg.norm(k_np, axis=1, keepdims=True)
q_mx = mx.array(q_np.astype(np.float16)).reshape(1, -1)
k_mx = mx.array(k_unit.astype(np.float16))

scores_true = np.array(k_mx @ q_mx.T).flatten()
sm_true = np.exp(scores_true) / np.exp(scores_true).sum()


def attn_tq(bits):
    qt = TQP(d=d, b=bits, m=min(d, 64), seed=0)
    ev = qt.encode(k_mx)
    k_hat = qt.decode(ev)
    sc = np.array(k_hat @ q_mx.T).flatten()
    sm = np.exp(sc) / np.exp(sc).sum()
    return sm


sm_3b = attn_tq(3)
sm_4b = attn_tq(4)

fig5, axes5 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig5.suptitle(
    "Attention Score Distortion — Gemma 4 Full-Attention Layers (head_dim=512)\n"
    "32 key vectors, query dot-product, softmax",
    fontsize=13,
    fontweight="bold",
)
for ax, (sm, label, col) in zip(
    axes5,
    [
        (sm_true, "fp16 Baseline (reference)", PALETTE["fp16"]),
        (sm_3b, "TurboQuant 3-bit  (cosine ~0.85+)", PALETTE["3bit"]),
        (sm_4b, "TurboQuant 4-bit  (cosine ~0.95+)", PALETTE["4bit"]),
    ],
):
    ax.bar(np.arange(N_k), sm, color=col, alpha=0.78, edgecolor="white", linewidth=0.5)
    ax.plot(
        np.arange(N_k),
        sm_true,
        color=PALETTE["fp16"],
        lw=1.5,
        ls="--",
        alpha=0.5,
        label="fp16 reference",
    )
    mse_a = np.mean((sm - sm_true) ** 2)
    cos_a = np.dot(sm, sm_true) / (np.linalg.norm(sm) * np.linalg.norm(sm_true))
    ax.set_ylabel("Attention weight")
    ax.set_title(
        f"{label}   |   MSE={mse_a:.2e}   cosine={cos_a:.4f}", fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, max(sm_true) * 1.45)
    sns.despine(ax=ax)
axes5[-1].set_xlabel("Key Token Index")
fig5.tight_layout()
fig5.savefig(f"{OUT_DIR}/fig5_attention_distortion.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/fig5_attention_distortion.png")


# ── Fig 6: Combined report ─────────────────────────────────────────────────────
fig6 = plt.figure(figsize=(20, 22))
fig6.patch.set_facecolor("#FAFAFA")
gs = gridspec.GridSpec(3, 2, figure=fig6, hspace=0.44, wspace=0.35)

# A — Compression
ax_a = fig6.add_subplot(gs[0, 0])
bars = ax_a.bar(CONFIGS, COMPRESS, color=COLORS, edgecolor="white", lw=1.2)
ax_a.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
for b, v in zip(bars, COMPRESS):
    ax_a.text(
        b.get_x() + b.get_width() / 2,
        v + 0.06,
        f"{v:.2f}×",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax_a.set_title(
    "A  Key Compression Ratio (full-attn layers only)", fontsize=12, fontweight="bold", loc="left"
)
ax_a.set_ylabel("Ratio vs fp16")
sns.despine(ax=ax_a)
ax_a.set_ylim(0, max(COMPRESS) * 1.28)

# B — Throughput
ax_b = fig6.add_subplot(gs[0, 1])
bars = ax_b.bar(CONFIGS, TPUT, color=COLORS, edgecolor="white", lw=1.2)
ax_b.axhline(TPUT[0], color="grey", ls="--", lw=1, alpha=0.7)
for b, v in zip(bars, TPUT):
    ax_b.text(
        b.get_x() + b.get_width() / 2,
        v + 0.5,
        f"{v:.1f}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax_b.set_title("B  Generation Throughput (tok/s)", fontsize=12, fontweight="bold", loc="left")
ax_b.set_ylabel("Tokens / second")
sns.despine(ax=ax_b)
ax_b.set_ylim(0, max(TPUT) * 1.28)

# C — Architecture strip
ax_c = fig6.add_subplot(gs[1, :])
for i, lt in enumerate(layer_types):
    col = PALETTE["sliding"] if lt == "sliding_attention" else PALETTE["4bit"]
    ax_c.barh(0, 1, left=i, height=0.5, color=col, edgecolor="white", linewidth=0.8)
    if lt == "full_attention":
        ax_c.text(
            i + 0.5, 0, "TQ", ha="center", va="center", fontsize=7, color="white", fontweight="bold"
        )
ax_c.set_xlim(0, n_layers)
ax_c.set_yticks([])
ax_c.set_xlabel("Layer index")
patches = [
    plt.Rectangle(
        (0, 0), 1, 1, color=PALETTE["sliding"], label="Sliding-attention (not quantized)"
    ),
    plt.Rectangle((0, 0), 1, 1, color=PALETTE["4bit"], label="Full-attention — TurboQuant"),
]
ax_c.legend(handles=patches, loc="upper right", fontsize=10)
ax_c.set_title(
    "C  Layer-by-Layer Architecture  (42 layers, 7 full-attention)",
    fontsize=12,
    fontweight="bold",
    loc="left",
)
sns.despine(ax=ax_c, left=True)

# D — Memory at scale
ax_d = fig6.add_subplot(gs[2, 0])
ax_d.plot(
    token_counts,
    to_mb(fp16_fa),
    color=PALETTE["fp16"],
    lw=2.5,
    marker="o",
    ms=5,
    label="fp16 (K+V)",
)
ax_d.plot(
    token_counts,
    to_mb(tq3_b + val_fa),
    color=PALETTE["3bit"],
    lw=2.5,
    marker="s",
    ms=5,
    label="TQ 3-bit",
)
ax_d.plot(
    token_counts,
    to_mb(tq4_b + val_fa),
    color=PALETTE["4bit"],
    lw=2.5,
    marker="^",
    ms=5,
    label="TQ 4-bit",
)
ax_d.set_xscale("log", base=2)
ax_d.set_xticks(token_counts)
ax_d.set_xticklabels([f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8)
ax_d.set_xlabel("Context length")
ax_d.set_ylabel("Memory (MB)")
ax_d.set_title("D  Full-Attention KV Cache at Scale", fontsize=12, fontweight="bold", loc="left")
ax_d.legend(fontsize=9)
sns.despine(ax=ax_d)

# E — Attention distortion
ax_e = fig6.add_subplot(gs[2, 1])
w = 0.28
ax_e.bar(np.arange(N_k) - w, sm_true, width=w, color=PALETTE["fp16"], alpha=0.85, label="fp16")
ax_e.bar(np.arange(N_k), sm_3b, width=w, color=PALETTE["3bit"], alpha=0.85, label="TQ 3-bit")
ax_e.bar(np.arange(N_k) + w, sm_4b, width=w, color=PALETTE["4bit"], alpha=0.85, label="TQ 4-bit")
ax_e.set_xlabel("Key Token Index")
ax_e.set_ylabel("Attention weight")
ax_e.set_title("E  Attention Distortion (head_dim=512)", fontsize=12, fontweight="bold", loc="left")
ax_e.legend(fontsize=9)
sns.despine(ax=ax_e)

fig6.suptitle(
    "TurboQuant KV Cache — Gemma 4 4B Full Benchmark Report\n"
    "Apple M4 · Hybrid Architecture · veloxquant_mlx",
    fontsize=16,
    fontweight="bold",
    y=1.005,
)
fig6.savefig(f"{OUT_DIR}/fig6_full_report.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/fig6_full_report.png")

print(f"\nAll figures saved to ./{OUT_DIR}/")
