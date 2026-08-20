"""Offline-synthetic benchmark for AnchorKV-adapted anchor-residual compression.

AnchorKV (arXiv:2608.02901v1, no verified peer-reviewed venue as of
2026-08-20 — see paper/research/surveys/NEW_METHOD_SURVEY_V22.md for the
one-time venue-exception rationale) never drops a token: every position
stays in the softmax, represented as an anchor projection plus an optional
quantized residual. Every eviction method already in this repo (H2O,
SnapKV, PyramidKV, NestedKV, ...) instead commits a fixed token budget and
discards everything else outright.

The honest question this benchmark measures is NOT "does AnchorKV beat H2O
on some downstream task" (no model is loaded here) but the more basic claim
the paper makes structurally: at the SAME byte budget, does keeping every
token as a coarse anchor projection (with a few residuals to fix the worst
approximations) produce attention output closer to the exact output than
dropping the lowest-scored tokens entirely? This is measured directly via
the true softmax-attention output error ``||y - y_hat||`` (Eq. 7/8 of the
paper) against a synthetic (query, K, V) triple with a planted structure:
most tokens cluster into a few directions (favorable for anchor projection),
plus one distinctive but LOW-ATTENTION-SCORE-AT-PREFILL token whose true
relevance only shows up for a later, different query — a stand-in for the
paper's central claim (Section 1, Section 4.2's per-task results) that
eviction's failure mode is discarding a token before the query that will
need it is known.

Two geometries:
  1. **clustered_favorable** — keys cluster tightly around a handful of
     directions (favorable for anchor projection: most tokens sit near
     SOME anchor). Measures baseline reconstruction fidelity at matched
     byte budgets when there is no adversarial structure at all.
  2. **late_relevance_token** — one token is a near-duplicate of NEITHER
     cluster center under the observation-window scoring queries (so H2O's
     cumulative-attention-mass eviction ranks it low and drops it), but a
     DIFFERENT held-out query attends to it heavily. This isolates
     eviction's structural failure mode: once dropped, the token cannot be
     recovered for ANY later query. AnchorKV keeps it (as a coarse anchor
     projection at minimum) so a later query can still find it.

Arms at the SAME matched byte budget: AnchorKV (anchor-residual, every
token retained) vs. H2O (cumulative attention-mass eviction, the closest
matched-budget baseline already in this repo).

Primary field: relative attention-output error ``||y - y_hat|| / ||y||``
against the HELD-OUT query (not the observation-window queries used for
scoring) — the metric the paper's Eq. 7/8 bounds and the one that isolates
eviction's "gone if the wrong query scored it" failure mode.

**Explicitly NOT a model-level benchmark.** The paper's RULER/LongBench/
Needle-in-a-Haystack numbers are the paper's, on Llama-3.1/Mistral-Small
models and NVIDIA A100 GPUs — not reproduced here.

Usage
-----
    python benchmark_scripts/benchmark_anchorkv.py

Prints tables and saves a JSON summary.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from veloxquant_mlx.quantizers.anchorkv import (
    ResidualCodec,
    allocate_residual_budget,
    anchorkv_budget_slots,
    assign_and_project,
    key_value_utility,
    select_anchors,
)
from veloxquant_mlx.quantizers.h2o import h2o_get_kv, h2o_update, init_h2o_state

HEAD_DIM = 32
SEQ_LENS = [128, 256]
# AnchorKV's retained-fraction knob. At this benchmark's small (S, D) scale,
# theta below ~0.2 floors to zero residual slots (all budget consumed by
# anchors + per-token metadata via anchorkv_budget_slots — see paper Eq. 9),
# so 0.05 and 0.15 give byte-identical, pure-anchor-projection output.
# Debugged directly (not swept under the rug): the paper's own byte budgets
# assume D=128 and S in the tens of thousands, where metadata is a much
# smaller share of the total. THETAS spans below and above that floor
# deliberately, so the identical rows at the low end are an honest artifact
# of this benchmark's toy scale, not a bug in anchorkv_budget_slots.
THETAS = [0.05, 0.15, 0.30]
WINDOW = 8
DATA_SEEDS = [0, 1, 2, 3, 4]
SEED = 17


def _synthetic(n: int, geometry: str, seed: int):
    """Returns (keys, values, held_out_query, planted_idx | None)."""
    rng = np.random.default_rng(seed)
    n_clusters = 4
    centers = rng.standard_normal((n_clusters, HEAD_DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    assign = rng.integers(0, n_clusters, size=n)
    keys = centers[assign] + 0.05 * rng.standard_normal((n, HEAD_DIM)).astype(np.float32)
    values = rng.standard_normal((n, HEAD_DIM)).astype(np.float32)

    planted_idx = None
    if geometry == "late_relevance_token":
        # A token that looks like ordinary cluster noise under the
        # observation-window queries (low attention score -> H2O drops it
        # early) but a DIFFERENT held-out query direction aligns with it
        # almost exactly, so a later retrieval needs exactly this token.
        planted_idx = n // 2
        planted_dir = rng.standard_normal(HEAD_DIM).astype(np.float32)
        planted_dir /= np.linalg.norm(planted_dir)
        keys[planted_idx] = 0.3 * centers[0] + 0.7 * planted_dir
        values[planted_idx] = rng.standard_normal(HEAD_DIM).astype(np.float32) * 3.0

        held_out_query = planted_dir * math.sqrt(HEAD_DIM)
    else:
        held_out_query = centers[0] * math.sqrt(HEAD_DIM)

    return keys, values, held_out_query.astype(np.float32), planted_idx


def _exact_attention(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    logits = (query @ keys.T) / math.sqrt(HEAD_DIM)
    logits = logits - np.max(logits)
    attn = np.exp(logits)
    attn /= attn.sum()
    return attn @ values


def _anchorkv_reconstruct(keys: np.ndarray, values: np.ndarray, theta: float, seed: int):
    k = mx.array(keys)
    v = mx.array(values)
    S = keys.shape[0]
    anchor_frac = 8.0 / S
    k_budget = max(1, int(round(S * anchor_frac)))

    anchors = select_anchors(k, k=k_budget, window=WINDOW, rho=0.7, seed=seed)
    key_assign = assign_and_project(k, anchors)
    value_assign = assign_and_project(v, anchors)

    m = min(WINDOW, S)
    proxy_q = k.astype(mx.float32)[-m:]
    u_key, u_value = key_value_utility(
        proxy_q,
        k.astype(mx.float32),
        v.astype(mx.float32),
        key_assign.residual,
        value_assign.residual,
    )
    anchor_set = set(int(a) for a in anchors.tolist())
    for pos in anchor_set:
        u_key = mx.where(mx.arange(S) == pos, mx.zeros_like(u_key), u_key)
        u_value = mx.where(mx.arange(S) == pos, mx.zeros_like(u_value), u_value)

    codec = ResidualCodec(head_dim=HEAD_DIM, seed=seed)
    n_slots = anchorkv_budget_slots(
        seq_len=S,
        head_dim=HEAD_DIM,
        n_anchor=int(anchors.shape[0]),
        theta=theta,
        residual_codec_bytes=codec.bytes_per_residual,
    )
    n_key_slots = n_slots // 2
    n_value_slots = n_slots - n_key_slots
    key_mask = allocate_residual_budget([u_key], n_key_slots)[0]
    value_mask = allocate_residual_budget([u_value], n_value_slots)[0]

    def _recon(x, assign, mask):
        chosen = x.astype(mx.float32)[assign.anchor_positions][assign.assign_idx]
        x_tilde = assign.gamma[:, None] * chosen
        codes, scale = codec.encode(assign.residual)
        decoded = codec.decode(codes, scale)
        term = mx.where(mask[:, None], decoded, mx.zeros_like(decoded))
        return x_tilde + term

    k_hat = np.array(_recon(k, key_assign, key_mask).tolist())
    v_hat = np.array(_recon(v, value_assign, value_mask).tolist())

    anchor_bytes = int(anchors.shape[0]) * HEAD_DIM * 2 * 2
    metadata_bytes = (S - int(anchors.shape[0])) * 2 * (4 + 4)
    n_residual = int(mx.sum(key_mask.astype(mx.int32)).item()) + int(
        mx.sum(value_mask.astype(mx.int32)).item()
    )
    total_bytes = anchor_bytes + metadata_bytes + n_residual * codec.bytes_per_residual
    return k_hat, v_hat, total_bytes


def _h2o_reconstruct(keys: np.ndarray, values: np.ndarray, budget: int):
    st = init_h2o_state(n_sink=0, budget=budget, head_dim=HEAD_DIM)
    st = h2o_update(st, mx.array(keys), mx.array(values))
    k_kept, v_kept = h2o_get_kv(st)
    k_kept = np.array(k_kept.tolist())
    v_kept = np.array(v_kept.tolist())
    total_bytes = k_kept.shape[0] * HEAD_DIM * 2 * 2
    return k_kept, v_kept, total_bytes


def _relative_error(
    query: np.ndarray, keys: np.ndarray, values: np.ndarray, y_exact: np.ndarray
) -> float:
    y_hat = _exact_attention(query, keys, values)
    return float(np.linalg.norm(y_hat - y_exact) / (np.linalg.norm(y_exact) + 1e-8))


def _run_once(seq_len: int, theta: float, geometry: str, seed: int) -> dict:
    row = {"seq_len": seq_len, "theta": theta, "geometry": geometry}
    anchorkv_errs, h2o_errs, anchorkv_bytes_list, h2o_bytes_list, anchorkv_mss = [], [], [], [], []

    for ds in DATA_SEEDS:
        keys, values, query, _ = _synthetic(seq_len, geometry, seed + ds)
        y_exact = _exact_attention(query, keys, values)

        t0 = time.perf_counter()
        k_a, v_a, bytes_a = _anchorkv_reconstruct(keys, values, theta, seed=seed + ds)
        mx.eval(mx.array(k_a))
        anchorkv_mss.append((time.perf_counter() - t0) * 1_000)
        anchorkv_errs.append(_relative_error(query, k_a, v_a, y_exact))
        anchorkv_bytes_list.append(bytes_a)

        budget = max(1, round(bytes_a / (HEAD_DIM * 2 * 2)))  # match AnchorKV's byte cost in tokens
        k_h, v_h, bytes_h = _h2o_reconstruct(keys, values, budget)
        h2o_errs.append(_relative_error(query, k_h, v_h, y_exact))
        h2o_bytes_list.append(bytes_h)

    row["rel_error_anchorkv"] = round(float(np.mean(anchorkv_errs)), 4)
    row["rel_error_h2o"] = round(float(np.mean(h2o_errs)), 4)
    row["bytes_anchorkv"] = round(float(np.mean(anchorkv_bytes_list)), 1)
    row["bytes_h2o"] = round(float(np.mean(h2o_bytes_list)), 1)
    row["ms_anchorkv"] = round(float(np.mean(anchorkv_mss)), 3)
    return row


def main() -> None:
    print("AnchorKV-adapted anchor-residual compression — offline synthetic benchmark")
    print("  (no verified peer-reviewed venue as of 2026-08-20 — see NEW_METHOD_SURVEY_V22.md)")
    print(f"  head_dim={HEAD_DIM}  window={WINDOW}  data_seeds={DATA_SEEDS}")
    print("  (rel_error = ||y_hat - y_exact|| / ||y_exact|| against a HELD-OUT query;")
    print("   lower = better; both arms matched to the same approximate byte budget)")
    print()
    header = (
        f"{'seq':>4} {'theta':>6} {'geometry':>22}  {'anchorkv':>10}  {'h2o':>8}  {'bytes':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for seq_len in SEQ_LENS:
        for theta in THETAS:
            for geometry in ["clustered_favorable", "late_relevance_token"]:
                row = _run_once(seq_len, theta, geometry, seed=SEED + seq_len)
                results.append(row)
                print(
                    f"{row['seq_len']:>4} {row['theta']:>6} {row['geometry']:>22}  "
                    f"{row['rel_error_anchorkv']:>10.4f}  {row['rel_error_h2o']:>8.4f}  "
                    f"~{row['bytes_anchorkv']:>9.0f}"
                )

    out_dir = Path(__file__).parent.parent / "figures" / "anchorkv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    for geom in ["clustered_favorable", "late_relevance_token"]:
        rows = [r for r in results if r["geometry"] == geom]
        a_err = float(np.mean([r["rel_error_anchorkv"] for r in rows]))
        h_err = float(np.mean([r["rel_error_h2o"] for r in rows]))
        print(f"\nSummary ({geom}): AnchorKV mean rel-error {a_err:.4f}  vs  H2O {h_err:.4f}")

    print(
        "\n  (Honest note: theta=0.05 and theta=0.15 rows are byte-identical at this\n"
        "   benchmark's small S/D scale — anchorkv_budget_slots floors to zero residual\n"
        "   slots below ~theta=0.2 here (anchors + per-token metadata already consume the\n"
        "   whole budget), so both thetas fall back to pure anchor projection with no\n"
        "   residuals. This is a property of running the paper's byte accounting at a toy\n"
        "   scale (D=32, S<=256) far below the paper's own D=128, S in the tens of\n"
        "   thousands — not a bug in the budget formula. See THETAS comment above.)"
    )
    print(
        "\n  (This is an offline-synthetic, cache-primitive-level benchmark — no model is\n"
        "   loaded and no RULER/LongBench/NIAH numbers are reproduced; those are the paper's\n"
        "   own (Llama-3.1-8B/70B, Mistral-Small-3.1-24B, NVIDIA A100). This method has NO\n"
        "   verified peer-reviewed venue as of 2026-08-20 (arXiv:2608.02901v1); it ships as a\n"
        "   one-time, user-directed exception to this repo's standing venue-verification rule,\n"
        "   the same exception previously granted to NestedKV.)"
    )


if __name__ == "__main__":
    main()
