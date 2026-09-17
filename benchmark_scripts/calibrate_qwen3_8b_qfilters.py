"""Calibrate real Q-Filters for Qwen3-8B, following the paper's own recipe.

The honest QFilters benchmark (benchmark_qwen3_8b_qfilters_honest.py) used
the *fallback* filter path -- filters=None, direction estimated from the
SVD of the first few observed KEYS, which recovers the dominant axis but
not its sign (the cache never sees queries, and the sign is exactly what a
query disambiguates per Theorem 3.3). That produced a fully reproducible
generation collapse once the cache exceeded its eviction budget.

This script closes that gap using veloxquant_mlx.quantizers.qfilters_calibration,
which already implements the paper's real mechanism (arXiv:2503.02812 §3.2):
hook each layer's query projection, gather real query activations on
calibration text, take the SVD's top right-singular vector per head,
sign-fix it against the paper's Theorem 3.3, and average query-head filters
down to KV heads for GQA. This is a one-time, pre-deployment calibration
pass -- not something that runs in the decode hot path.

Usage::

    PYTHONPATH=. python benchmark_scripts/calibrate_qwen3_8b_qfilters.py \\
        --model mlx-community/Qwen3-8B-4bit \\
        --output figures/qwen3_8b_qfilters_calibrated/qfilters_qwen3_8b.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


# Calibration corpus: general-purpose English prose, long enough per sample
# (paper uses 2048-token documents) to give the SVD a real query-drift signal.
# Not the Pile (what the paper used) -- a repo-local, license-clean substitute
# assembled from public-domain-style explanatory text repeated/varied to reach
# length, which is what's practical to embed in a benchmark script without
# a network dependency on a specific corpus. Flagged here, not concealed.
_CALIB_PASSAGES = [
    (
        "The transformer architecture processes sequences of tokens through "
        "stacked layers of self-attention and feed-forward networks. Each "
        "attention layer computes queries, keys, and values from the input "
        "representations, then uses the queries to attend over the keys and "
        "aggregate the values. This mechanism lets every position in a "
        "sequence gather information from every other position, which is "
        "what gives transformers their long-range modeling capacity. "
    ) * 20,
    (
        "Memory bandwidth, not raw compute, is often the binding constraint "
        "during autoregressive decoding on modern accelerators. Every "
        "generated token requires reading the entire set of model weights "
        "and the full key-value cache from memory, even though the actual "
        "arithmetic per token is comparatively small. As context length "
        "grows, the key-value cache itself becomes a larger and larger "
        "fraction of the total memory traffic, which is why cache "
        "compression techniques target it directly rather than the weights. "
    ) * 20,
    (
        "Apple Silicon unifies CPU and GPU memory into a single address "
        "space, which removes the need to copy data across a PCIe bus but "
        "also means the model weights, the KV cache, and the operating "
        "system all compete for the same finite pool. On a 24 gigabyte "
        "machine, an 8 billion parameter model at 4-bit precision already "
        "consumes a substantial fraction of that budget before a single "
        "token of context has been cached. "
    ) * 20,
    (
        "Vector quantization compresses a set of vectors by replacing each "
        "one with the index of its nearest neighbor in a small, shared "
        "codebook. The codebook is typically learned via k-means clustering "
        "over a representative sample of the vectors it will be asked to "
        "approximate. A codebook trained on data that doesn't resemble the "
        "true distribution of activations will reconstruct those "
        "activations poorly, regardless of how many centroids it has. "
    ) * 20,
    (
        "Eviction-based cache compression discards the least useful cached "
        "tokens once a fixed budget is exceeded, rather than approximating "
        "every token as quantization does. The quality of an eviction "
        "policy depends entirely on how well its scoring function predicts "
        "which tokens future queries will actually need, and a scoring "
        "function estimated from the wrong signal can evict tokens that "
        "turn out to matter, degrading generation quality in ways that are "
        "difficult to detect from throughput numbers alone. "
    ) * 20,
]


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="Calibrate real Q-Filters for Qwen3-8B")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--output", default="figures/qwen3_8b_qfilters_calibrated/qfilters_qwen3_8b.npz")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-samples-per-head", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from mlx_lm import load

    from veloxquant_mlx.quantizers.qfilters_calibration import (
        QFiltersCalibration,
        average_gqa_filters,
        collect_query_activations,
        compute_qfilters,
        save_qfilters,
    )

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer = load(args.model)

    margs = getattr(model, "args", None) or model.model.args
    n_q_heads = margs.num_attention_heads
    n_kv_heads = getattr(margs, "num_key_value_heads", None) or n_q_heads
    print(f"  num_attention_heads={n_q_heads} num_key_value_heads={n_kv_heads}")

    print(f"Collecting query activations on {len(_CALIB_PASSAGES)} calibration passages "
          f"(max_length={args.max_length})...", flush=True)
    per_layer_queries = collect_query_activations(
        model, tokenizer, _CALIB_PASSAGES,
        max_length=args.max_length,
        max_samples_per_head=args.max_samples_per_head,
        seed=args.seed,
    )
    print(f"  captured {len(per_layer_queries)} layers, "
          f"shape[0]={tuple(per_layer_queries[0].shape)}")

    print("Computing per-head Q-Filters (SVD, sign-fixed per Theorem 3.3)...", flush=True)
    per_layer_filters = []
    for li, q in enumerate(per_layer_queries):
        f = compute_qfilters(q, max_svd_samples=args.max_samples_per_head)
        if n_kv_heads != n_q_heads:
            f = average_gqa_filters(f, n_kv_heads)
        per_layer_filters.append(f)
        if li == 0:
            print(f"  layer 0 filters shape: {tuple(f.shape)}")

    calibration = QFiltersCalibration(
        filters=per_layer_filters,
        model_id=args.model,
        n_samples=args.max_samples_per_head,
        dataset="repo-local synthetic-prose calibration passages (see script docstring)",
    )

    out_path = save_qfilters(calibration, args.output)
    print(f"\nSaved calibration artifact: {out_path}")
    print(f"  n_layers={calibration.n_layers} model_id={calibration.model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
