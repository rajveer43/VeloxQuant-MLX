"""Does the calibrated Q-Filter predict REAL attention? (paper Figure 4)

Spearman correlation between each scorer and the observed mean attention
S^h_t = (1/(L-t+1)) sum_{i>=t} A^h_it, measured on true attention maps from a
real model. Compares query-SVD (calibrated), key-SVD (fallback), and K-norm.
"""

import sys

import mlx.core as mx
import mlx_lm
import numpy as np
from scipy.stats import spearmanr

from veloxquant_mlx.quantizers.qfilters import estimate_filter_dir
from veloxquant_mlx.quantizers.qfilters_calibration import (
    average_gqa_filters,
    collect_query_activations,
    compute_qfilters,
)

mid = sys.argv[1]
m, tok = mlx_lm.load(mid)
inner = getattr(m, "model", m)
args = m.args
H, KV, D = args.num_attention_heads, args.num_key_value_heads, args.head_dim

calib = [
    "The capital of France is Paris, a city of art and history. " * 40,
    "Machine learning models require large amounts of data to train. " * 40,
    "In 1969, humans first walked on the surface of the moon. " * 40,
    "Photosynthesis converts light energy into chemical energy. " * 40,
]
acts = collect_query_activations(m, tok, calib, max_length=512, max_samples_per_head=3000)
qfilt = [np.array(average_gqa_filters(compute_qfilters(a), KV), dtype=np.float64) for a in acts]

# --- capture K and true attention on a HELD-OUT text ---
held = (
    "Artificial intelligence has transformed many industries over the past decade. "
    "Researchers continue to explore new architectures for language understanding. "
    "The transformer architecture introduced attention as its core mechanism. "
) * 12
ids = tok.encode(held)[:512]
caps = {}
orig = []
for li, layer in enumerate(inner.layers):
    a = layer.self_attn
    orig.append((a, a.k_proj, a.q_proj))

    def mk(inner_fn, store, key):
        def f(x):
            o = inner_fn(x)
            store[key] = np.array(o.astype(mx.float32), copy=True)
            return o

        return f

    a.k_proj = mk(a.k_proj, caps, ("k", li))
    a.q_proj = mk(a.q_proj, caps, ("q", li))
try:
    mx.eval(m(mx.array([ids])))
finally:
    for a, k, q in orig:
        a.k_proj, a.q_proj = k, q

L = len(ids)
rows = {"query-SVD (calibrated)": [], "key-SVD (fallback)": [], "K-norm": []}
for li in range(len(inner.layers)):
    K = caps[("k", li)].reshape(L, KV, D).transpose(1, 0, 2)  # [KV, L, D]
    Q = caps[("q", li)].reshape(L, H, D).transpose(1, 0, 2)  # [H,  L, D]
    for kh in range(KV):
        grp = Q[kh * (H // KV) : (kh + 1) * (H // KV)]  # queries sharing this KV head
        k = K[kh].astype(np.float64)
        # true causal attention, averaged over the group
        S = np.zeros(L)
        for qh in grp:
            logits = (qh.astype(np.float64) @ k.T) / np.sqrt(D)
            logits[np.triu_indices(L, 1)] = -np.inf
            A = np.exp(logits - logits.max(1, keepdims=True))
            A /= A.sum(1, keepdims=True)
            S += np.array([A[t:, t].mean() for t in range(L)])
        S /= len(grp)
        kd = np.array(estimate_filter_dir(mx.array(k.astype(np.float32))), dtype=np.float64)
        rows["query-SVD (calibrated)"].append(spearmanr(k @ qfilt[li][kh], S).statistic)
        rows["key-SVD (fallback)"].append(spearmanr(k @ kd, S).statistic)
        rows["K-norm"].append(spearmanr(-np.linalg.norm(k, axis=1), S).statistic)

print(
    f"\n=== {mid} — Spearman vs TRUE attention (paper Fig. 4), {len(rows['K-norm'])} KV heads ==="
)
for n, v in rows.items():
    v = np.array(v)
    print(
        f"{n:24s} median {np.median(v):+.3f}   mean {v.mean():+.3f}   frac>0 {100 * np.mean(v > 0):5.1f}%"
    )
