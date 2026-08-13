"""Validate the three 2-bit accuracy improvements.

Compares cosine similarity, SNR, and per-vector memory across:
  - TurboQuantProd baseline (m=64)
  - TurboQuantProd with m=d (Change 1)
  - TurboQuantRVQ (Change 2)
  - TurboQuantProd + AdaptiveScalarCodebook (Change 3)
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ


D = 128
B = 2
N = 256
N_CALIB = 64
SEED = 42


def _gen_unit_norm(n: int, d: int, seed: int) -> mx.array:
    """Generate unit-norm vectors with a non-Gaussian channel mix.

    Real LM key distributions are heavy-tailed and not isotropic. We emulate
    this by mixing Student-t draws on half the coordinates with Gaussian on
    the other half — this gives a realistic kurtosis profile where an
    adaptive codebook should beat the default N(0,1) Lloyd-Max codebook.
    """
    rng = np.random.default_rng(seed)
    half = d // 2
    gauss = rng.standard_normal((n, d - half)).astype(np.float32)
    # Student-t with df=3 — heavy tails, no second-moment blow-up at this df.
    t = rng.standard_t(df=3, size=(n, half)).astype(np.float32)
    x = np.concatenate([gauss, t], axis=1)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return mx.array(x.astype(np.float16))


def _eval(name: str, x: mx.array, x_hat: mx.array, mem_bytes: int) -> dict:
    cos = float(
        mx.mean(
            mx.sum(x * x_hat, axis=1) / (mx.linalg.norm(x, axis=1) * mx.linalg.norm(x_hat, axis=1))
        )
    )
    err = (x - x_hat).astype(mx.float32)
    sig = float(mx.mean(x.astype(mx.float32) ** 2))
    noise = float(mx.mean(err**2))
    snr = 10.0 * math.log10(max(sig / max(noise, 1e-12), 1e-12))
    return {"name": name, "cosine": cos, "snr_db": snr, "mem_bytes": mem_bytes}


def _bytes_prod(d: int, b: int, m: int) -> int:
    """Per-vector storage for TurboQuantProd: indices (b-1 bits/d) + signs (1 bit/m) + r_norm (fp16)."""
    b_mse = max(b - 1, 1)
    return math.ceil(d * b_mse / 8) + math.ceil(m / 8) + 2


def _bytes_rvq(d: int, b: int) -> int:
    """Per-vector storage for TurboQuantRVQ: 2*b bits/d (two index sets)."""
    return math.ceil(d * 2 * b / 8)


def main() -> None:
    x = _gen_unit_norm(N, D, seed=SEED)

    rows: list[dict] = []

    # --- Baseline: TurboQuantProd m=64 ---
    qp_base = TurboQuantProd(d=D, b=B, m=min(D, 64), seed=SEED)
    ev = qp_base.encode(x)
    x_hat = qp_base.decode(ev)
    rows.append(_eval("Baseline TQ-Prod (m=64)", x, x_hat, _bytes_prod(D, B, min(D, 64))))

    # --- Change 1: TurboQuantProd m=d ---
    qp_m = TurboQuantProd(d=D, b=B, m=D, seed=SEED)
    ev = qp_m.encode(x)
    x_hat = qp_m.decode(ev)
    rows.append(_eval("Change 1 TQ-Prod (m=d)", x, x_hat, _bytes_prod(D, B, D)))

    # --- Change 2: TurboQuantRVQ ---
    qrvq = TurboQuantRVQ(d=D, b=B, seed=SEED)
    ev = qrvq.encode(x)
    x_hat = qrvq.decode(ev)
    rows.append(_eval("Change 2 TQ-RVQ (b=2 x2)", x, x_hat, _bytes_rvq(D, B)))

    # --- Bonus: TurboQuantRVQ b=1 (sign quant + Laplacian residual correction) ---
    qrvq1 = TurboQuantRVQ(d=D, b=1, seed=SEED)
    ev = qrvq1.encode(x)
    x_hat = qrvq1.decode(ev)
    rows.append(_eval("Extra   TQ-RVQ (b=1 x2)", x, x_hat, _bytes_rvq(D, 1)))

    # --- Change 3: TurboQuantProd + AdaptiveScalarCodebook ---
    qp_ad = TurboQuantProd(
        d=D,
        b=B,
        m=min(D, 64),
        seed=SEED,
        use_adaptive_codebook=True,
        n_calib=N_CALIB,
    )
    # Calibration pass: feed first N_CALIB vectors so codebook fits
    qp_ad.encode(x[:N_CALIB])
    # Evaluation pass on full set after calibration
    ev = qp_ad.encode(x)
    x_hat = qp_ad.decode(ev)
    rows.append(_eval("Change 3 TQ-Prod (adaptive)", x, x_hat, _bytes_prod(D, B, min(D, 64))))

    # --- Print table ---
    print(f"\n{'=' * 78}")
    print(f"2-bit TurboQuant accuracy comparison  (d={D}, b={B}, n={N})")
    print(f"{'=' * 78}")
    print(f"{'Method':<32}  {'cosine':>8}  {'SNR (dB)':>9}  {'bytes/vec':>10}")
    print(f"{'-' * 78}")
    for r in rows:
        print(f"{r['name']:<32}  {r['cosine']:>8.4f}  {r['snr_db']:>9.2f}  {r['mem_bytes']:>10}")
    print(f"{'=' * 78}\n")

    # --- Asserts ---
    by_name = {r["name"]: r for r in rows}
    c1 = by_name["Change 1 TQ-Prod (m=d)"]
    c2 = by_name["Change 2 TQ-RVQ (b=2 x2)"]
    c3 = by_name["Change 3 TQ-Prod (adaptive)"]

    assert c1["cosine"] > 0.75, f"Change 1 cosine {c1['cosine']:.4f} not > 0.75"
    assert c2["cosine"] > 0.82, f"Change 2 cosine {c2['cosine']:.4f} not > 0.82"
    rvq1 = by_name["Extra   TQ-RVQ (b=1 x2)"]
    assert rvq1["cosine"] > 0.80, f"RVQ b=1 cosine {rvq1['cosine']:.4f} not > 0.80"
    # Change 3 finding: on post-rotation unit-norm vectors the empirical
    # distribution is already near-Gaussian, so the fitted codebook is
    # essentially identical to the default N(0, 1/d) Lloyd-Max codebook.
    # The adaptive codebook only provides a measurable gain on real LM keys
    # whose post-rotation distribution has channel-specific kurtosis or skew
    # that the random rotation doesn't fully Gaussianize. We assert that the
    # adaptive path does not regress vs baseline rather than claiming a gain.
    base = by_name["Baseline TQ-Prod (m=64)"]
    assert c3["cosine"] >= base["cosine"] - 0.005, (
        f"Change 3 cosine {c3['cosine']:.4f} regressed vs baseline {base['cosine']:.4f}"
    )
    print("ALL ASSERTS PASSED")
    print(
        "\nNote: Change 3 (adaptive codebook) shows no synthetic-data gain — "
        "post-rotation Gaussianization already matches the default codebook. "
        "Real-model benefit must be measured on actual LM key tensors."
    )


def test_vlm_keys():
    """Verify RVQ 2-bit quality on image-like key tensors (large S, non-unit norms).

    VLM image patch tokens flood the KV cache with ~256-1024 vectors per layer
    during prefill. Their distribution after ViT projection is similar to text
    keys but with potentially larger norms. This test checks that our quantizer
    handles large-batch, non-unit-norm input correctly.
    """
    print("\n" + "=" * 64)
    print("VLM KEY TENSOR TEST  (d=128, S=512, simulated image prefill)")
    print("=" * 64)

    np.random.seed(7)
    D_vlm = 128
    S_vlm = 512  # typical image patch count
    H_vlm = 8  # kv heads

    # Simulate (B*H*S, D) batch as would arrive after reshape in update_and_fetch
    # Image keys tend to have larger norms than text keys (~5-20 range)
    raw = np.random.randn(S_vlm * H_vlm, D_vlm).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    scale = np.random.uniform(3.0, 15.0, size=(S_vlm * H_vlm, 1)).astype(np.float32)
    keys_np = (raw / norms * scale).astype(np.float16)
    keys_mx = mx.array(keys_np)

    # Normalize (mirrors update_and_fetch path)
    norms_mx = mx.linalg.norm(keys_mx.astype(mx.float32), axis=-1, keepdims=True).astype(mx.float16)
    safe = mx.maximum(norms_mx, mx.array(1e-4, dtype=mx.float16))
    k_unit = (keys_mx / safe).astype(mx.float16)

    results = {}
    for label, quantizer in [
        ("TQ-Prod 4-bit", TurboQuantProd(d=D_vlm, b=4, m=64, seed=0, use_hadamard=True)),
        ("TQ-RVQ 2-bit", TurboQuantRVQ(d=D_vlm, b=2, seed=0, use_hadamard=True)),
        ("TQ-Prod 2-bit", TurboQuantProd(d=D_vlm, b=2, m=D_vlm, seed=0, use_hadamard=True)),
    ]:
        ev = quantizer.encode(k_unit)
        k_hat = quantizer.decode(ev)
        cos = float(
            mx.mean(
                mx.sum(k_unit * k_hat, axis=-1)
                / (mx.linalg.norm(k_unit, axis=-1) * mx.linalg.norm(k_hat, axis=-1) + 1e-8)
            )
        )
        results[label] = cos
        print(f"  {label:<22}  cosine = {cos:.4f}")

    print()
    assert results["TQ-Prod 4-bit"] > 0.93, (
        f"TQ-Prod 4-bit VLM cosine {results['TQ-Prod 4-bit']:.4f} < 0.93"
    )
    assert results["TQ-RVQ 2-bit"] > 0.90, (
        f"TQ-RVQ 2-bit VLM cosine {results['TQ-RVQ 2-bit']:.4f} < 0.90"
    )
    assert results["TQ-RVQ 2-bit"] > results["TQ-Prod 2-bit"], (
        f"RVQ 2-bit ({results['TQ-RVQ 2-bit']:.4f}) should beat single-pass 2-bit ({results['TQ-Prod 2-bit']:.4f})"
    )

    print("VLM KEY TEST PASSED")
    print(f"  RVQ 2-bit cosine on image-like keys: {results['TQ-RVQ 2-bit']:.4f}")


def main_vlm():
    test_vlm_keys()


if __name__ == "__main__":
    import sys

    if "--vlm" in sys.argv:
        main_vlm()
    else:
        main()
