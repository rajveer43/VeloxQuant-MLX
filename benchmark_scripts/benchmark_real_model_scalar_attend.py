"""Real-model TTFT + decode tokens/sec benchmark for scalar_fused_decode_attend
(issue #307's underlying occupancy hypothesis), on Qwen3-4B-4bit.

Context — why this script exists and what it does NOT test
-------------------------------------------------------------
This session built `scalar_fused_decode_attend_batched`, which adds a new
outermost NL (num_layers) grid axis so ONE dispatch can cover every
transformer layer's decode-attend call at once. That technique is a real,
verified kernel-level win in isolation (see
docs/KV_KERNEL_ROOFLINE_FINDINGS.md's third addendum) — but it CANNOT speed
up real single-request decode on any standard transformer, because a
pre-norm decoder block computes layer L+1's attention input from layer L's
FULL output (attention -> residual -> MLP -> residual), not just its
attention output. There is no valid reordering of a standard model's
forward pass that lets multiple layers' attention be dispatched as one
batched call without changing what the model actually computes. This is a
structural fact about the residual stream, not an unimplemented patch —
confirmed by reading mlx_lm's own TransformerBlock/MLP source directly.

What IS real and testable on this architecture: `scalar_fused_decode_attend`
(the single-layer kernel, already shipped, NOT the new batched one) already
carries a `B` (batch/request) axis in its dispatch grid
(`n_tg = B * H_kv * S_q`). Multiple CONCURRENT requests have no cross-
layer sequential dependency — they are independent by construction — so
batching decode-attend across concurrent requests is a structurally valid
lever, and it's exactly the "or multiple requests" half of the roofline
doc's original Recommendation #2. This script tests THAT, on a real model,
with real prompts, real TTFT, and real decode tokens/sec.

No cache in this repo currently routes real mlx_lm generation through
`scalar_fused_decode_attend` — `KIVIKVCache` uses `kivi_group_quant_dequant`
(materializes fp16 K_hat/V_hat, then standard MLX SDPA) instead, and
`fused_sdpa.py`'s dispatcher for the sibling VecInfer kernel is a documented
no-op for the same reason. So this script wires up a minimal purpose-built
cache (`_ScalarAttendKIVICache`) that stores KIVI-quantized codes/scale/zero
and exposes them for the fused kernel, plus a monkeypatch of
`mlx_lm.models.base.scaled_dot_product_attention` (mirroring
`patch_mlx_lm_for_fused_sdpa`'s established pattern) that routes decode-shape
(`S_q == 1`) calls to `scalar_fused_decode_attend` and leaves prefill
(`S_q > 1`) on the standard path.

Usage: python benchmark_scripts/benchmark_real_model_scalar_attend.py
"""

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx
import mlx_lm.models.base as _mlx_base
from mlx_lm import load
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.metal._scalar_attend import scalar_fused_decode_attend

MODEL_ID = "mlx-community/Qwen3-4B-4bit"
GROUP_SIZE = 32
BIT_WIDTH = 2
LEVELS = (1 << BIT_WIDTH) - 1
EPS = 1e-8


class _ScalarAttendKIVICache(_MLXKVCache):
    """KIVI-quantized cache exposing raw codes for scalar_fused_decode_attend.

    Unlike this repo's `KIVIKVCache` (which dequantizes eagerly inside
    `update_and_fetch` and returns fp16 K_hat/V_hat for standard SDPA), this
    cache keeps the most recent tokens as KIVI-quantized uint8 codes +
    per-group (scale, zero) and exposes them via `quantized_state()` for the
    patched SDPA to feed directly into the fused kernel.

    IMPORTANT — incremental quantization, not full-history re-quantization:
    an earlier version of this cache re-quantized the ENTIRE growing K/V
    history from scratch on every decode step, which cost ~25ms/step across
    36 layers on Qwen3-4B (pure quantization arithmetic, nothing to do with
    the attend kernel) and completely swamped the ~0.3ms/layer the fused
    kernel itself takes — contaminating the benchmark with a test-harness
    artifact rather than measuring the kernel. This version quantizes ONLY
    the newly-appended group-size-aligned block each step and concatenates
    onto the existing quantized state, mirroring the incremental-flush
    discipline `KIVIKVCache._quantization_boundary()` already established
    in this repo for exactly this reason. Still simpler than production
    KIVI (no fp16 residual window — quantizes as soon as `group_size`
    tokens are available) since this cache exists only to drive this
    benchmark, not to ship.
    """

    def __init__(self) -> None:
        super().__init__()
        self._k_codes: Optional[mx.array] = None
        self._k_scale: Optional[mx.array] = None
        self._k_zero: Optional[mx.array] = None
        self._v_codes: Optional[mx.array] = None
        self._v_scale: Optional[mx.array] = None
        self._v_zero: Optional[mx.array] = None
        self._n_quantized: int = 0

    def _quant_keys(self, k: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        B, H, S, D = k.shape
        g = GROUP_SIZE
        GK = (S + g - 1) // g
        pad = GK * g - S
        x = k.astype(mx.float32)
        if pad:
            x = mx.concatenate([x, mx.broadcast_to(x[:, :, -1:, :], (B, H, pad, D))], axis=2)
        xg = x.reshape(B, H, GK, g, D)
        gmin = mx.min(xg, axis=3, keepdims=True)
        gmax = mx.max(xg, axis=3, keepdims=True)
        scale = mx.maximum((gmax - gmin) / LEVELS, EPS)
        codes = mx.clip(mx.round((xg - gmin) / scale), 0, LEVELS)
        codes = codes.reshape(B, H, GK * g, D)[:, :, :S, :].astype(mx.uint8)
        return codes, scale.reshape(B, H, GK, D).astype(mx.float32), gmin.reshape(
            B, H, GK, D
        ).astype(mx.float32)

    def _quant_values(self, v: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        B, H, S, D = v.shape
        g = GROUP_SIZE
        GV = (D + g - 1) // g
        pad = GV * g - D
        x = v.astype(mx.float32)
        if pad:
            x = mx.concatenate([x, mx.broadcast_to(x[:, :, :, -1:], (B, H, S, pad))], axis=3)
        xg = x.reshape(B, H, S, GV, g)
        gmin = mx.min(xg, axis=4, keepdims=True)
        gmax = mx.max(xg, axis=4, keepdims=True)
        scale = mx.maximum((gmax - gmin) / LEVELS, EPS)
        codes = mx.clip(mx.round((xg - gmin) / scale), 0, LEVELS)
        codes = codes.reshape(B, H, S, GV * g)[:, :, :, :D].astype(mx.uint8)
        return codes, scale.reshape(B, H, S, GV).astype(mx.float32), gmin.reshape(
            B, H, S, GV
        ).astype(mx.float32)

    def update_and_fetch(self, keys, values):
        k_all, v_all = super().update_and_fetch(keys, values)
        S = k_all.shape[2]

        # Quantize only in group_size-aligned blocks, and only the portion
        # that has not been quantized yet — group boundaries must land on
        # whole group_size chunks (see KIVIKVCache._quantization_boundary's
        # identical reasoning), so anything past the last full group waits.
        new_boundary = (S // GROUP_SIZE) * GROUP_SIZE
        if new_boundary > self._n_quantized:
            lo, hi = self._n_quantized, new_boundary
            k_block = k_all[:, :, lo:hi, :]
            v_block = v_all[:, :, lo:hi, :]
            kc, ks, kz = self._quant_keys(k_block)
            vc, vs, vz = self._quant_values(v_block)
            if self._k_codes is None:
                self._k_codes, self._k_scale, self._k_zero = kc, ks, kz
                self._v_codes, self._v_scale, self._v_zero = vc, vs, vz
            else:
                self._k_codes = mx.concatenate([self._k_codes, kc], axis=2)
                self._k_scale = mx.concatenate([self._k_scale, ks], axis=2)
                self._k_zero = mx.concatenate([self._k_zero, kz], axis=2)
                self._v_codes = mx.concatenate([self._v_codes, vc], axis=2)
                self._v_scale = mx.concatenate([self._v_scale, vs], axis=2)
                self._v_zero = mx.concatenate([self._v_zero, vz], axis=2)
            self._n_quantized = new_boundary

        return k_all, v_all

    def quantized_state(self):
        return (
            self._k_codes,
            self._k_scale,
            self._k_zero,
            self._v_codes,
            self._v_scale,
            self._v_zero,
        )

    def dequantized_kv(self, heads_per_kv: int) -> tuple[mx.array, mx.array]:
        """Reconstruct fp16 K_hat/V_hat from the quantized state, GQA-expanded
        to H_q heads — the baseline comparison path (dequant-then-SDPA),
        mirroring exactly what a KIVI-style cache would hand to standard
        MLX SDPA. Used by the UNPATCHED baseline so both arms of the
        comparison attend over the SAME quantized information; the only
        difference is dequant-then-SDPA vs. the fused on-the-fly kernel.
        """
        S_kv = self._k_codes.shape[2]
        D = self._k_codes.shape[3]
        kg = mx.arange(S_kv) // GROUP_SIZE
        k_hat = (
            self._k_codes.astype(mx.float32) * mx.take(self._k_scale, kg, axis=2)
            + mx.take(self._k_zero, kg, axis=2)
        ).astype(mx.float16)
        vgi = mx.arange(D) // GROUP_SIZE
        v_hat = (
            self._v_codes.astype(mx.float32) * mx.take(self._v_scale, vgi, axis=3)
            + mx.take(self._v_zero, vgi, axis=3)
        ).astype(mx.float16)
        if heads_per_kv > 1:
            k_hat = mx.repeat(k_hat, heads_per_kv, axis=1)
            v_hat = mx.repeat(v_hat, heads_per_kv, axis=1)
        return k_hat, v_hat


_original_sdpa = None
_patched = False
_patched_modules: list = []
_route_count = {"decode_fused": 0, "decode_dequant_baseline": 0, "fallback": 0}


def _patched_sdpa_factory(nsg: int, use_fused: bool):
    """Build the SDPA replacement for one arm of the comparison.

    Both arms attend over EXACTLY the same KIVI-quantized state
    (``cache.quantized_state()``) whenever it's available (``S_q == 1``
    decode steps past the first group_size tokens) — the only difference
    is ``use_fused=True`` routing to ``scalar_fused_decode_attend``
    on-the-fly vs. ``use_fused=False`` dequantizing to fp16 first via
    ``cache.dequantized_kv()`` and calling standard MLX SDPA, mirroring
    what a real KIVI-style cache does today. Before quantized state exists
    (prefill, and the first < group_size decode steps) and for any
    non-``_ScalarAttendKIVICache`` cache, both arms fall back identically
    to the original standard SDPA over the cache's own fp16 buffer — so
    the two arms are apples-to-apples everywhere except the one thing this
    benchmark is testing.
    """

    def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        S_q = queries.shape[2]
        if (
            S_q == 1
            and sinks is None
            and isinstance(cache, _ScalarAttendKIVICache)
            and cache._k_codes is not None
        ):
            if use_fused:
                k_codes, k_scale, k_zero, v_codes, v_scale, v_zero = cache.quantized_state()
                _route_count["decode_fused"] += 1
                return scalar_fused_decode_attend(
                    queries, k_codes, k_scale, k_zero, v_codes, v_scale, v_zero,
                    GROUP_SIZE, scale, nsg=nsg,
                )
            else:
                H_q = queries.shape[1]
                H_kv = cache._k_codes.shape[1]
                heads_per_kv = H_q // H_kv
                k_hat, v_hat = cache.dequantized_kv(heads_per_kv)
                _route_count["decode_dequant_baseline"] += 1
                return _original_sdpa(
                    queries, k_hat, v_hat, cache=None, scale=scale, mask=None, sinks=sinks
                )
        _route_count["fallback"] += 1
        return _original_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

    return _patched_sdpa


def _patch_sdpa_for_scalar_attend(model, nsg: int = 4, use_fused: bool = True) -> None:
    """Patch ``scaled_dot_product_attention`` everywhere it's bound.

    ``mlx_lm.models.base.scaled_dot_product_attention`` is the canonical
    definition, but every model architecture module does
    ``from .base import scaled_dot_product_attention`` at its own import
    time — a plain Python name binding, not a live reference back to
    ``base`` — so patching only ``mlx_lm.models.base`` has no effect once
    a model module has already been imported (as it has here, since
    ``load()`` runs before this is called). Patch both the canonical
    ``base`` module AND the specific model module the loaded model came
    from, so the model's own already-bound name is replaced too.
    """
    global _original_sdpa, _patched, _patched_modules
    if _patched:
        return
    _original_sdpa = _mlx_base.scaled_dot_product_attention
    patched_fn = _patched_sdpa_factory(nsg, use_fused)

    _mlx_base.scaled_dot_product_attention = patched_fn
    _patched_modules = [_mlx_base]

    model_module = type(model.model).__module__
    import importlib

    mod = importlib.import_module(model_module)
    if hasattr(mod, "scaled_dot_product_attention"):
        mod.scaled_dot_product_attention = patched_fn
        _patched_modules.append(mod)

    _patched = True


def _unpatch_sdpa() -> None:
    global _patched, _patched_modules
    if not _patched:
        return
    for mod in _patched_modules:
        mod.scaled_dot_product_attention = _original_sdpa
    _patched_modules = []
    _patched = False


def _make_caches(model) -> list:
    return [_ScalarAttendKIVICache() for _ in model.model.layers]


def _run_decode(model, tokenizer, batch_size: int, prompt_len: int, n_decode: int):
    """Prefill a batch of `batch_size` identical-length prompts, then decode
    `n_decode` tokens, measuring TTFT (prefill latency) and decode tokens/sec.

    Real prompts are built by repeating tokenizer output to a fixed length so
    every batch element has identical S_q at every step (mlx_lm's stock
    KVCache assumes uniform sequence length across the batch axis) — a
    standard simplification for a batched-latency microbenchmark, not
    padding/masking logic this script needs to get right for correctness of
    the *decode* measurement, since only per-step S_q=1 decode shape matters
    for the kernel being tested.
    """
    prompt_text = (
        "The history of artificial intelligence began in antiquity, with myths "
        "and legends of artificial beings endowed with intelligence or "
        "consciousness by master craftsmen. "
    ) * 8
    ids = tokenizer.encode(prompt_text)[:prompt_len]
    if len(ids) < prompt_len:
        ids = (ids * (prompt_len // max(1, len(ids)) + 1))[:prompt_len]
    prompt = mx.array([ids] * batch_size)  # [B, S]

    caches = _make_caches(model)

    mx.synchronize()
    t0 = time.perf_counter()
    logits = model(prompt, cache=caches)
    mx.eval(logits)
    mx.synchronize()
    ttft = time.perf_counter() - t0

    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # [B, 1]
    mx.eval(next_tok)

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        logits = model(next_tok, cache=caches)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_tok)
    mx.synchronize()
    decode_time = time.perf_counter() - t0

    decode_tps = (n_decode * batch_size) / decode_time
    return ttft, decode_tps


def main() -> None:
    print(f"[real-model] loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)
    print(
        f"[real-model] {len(model.model.layers)} layers, "
        f"H_q={model.model.layers[0].self_attn.n_heads}, "
        f"H_kv={model.model.layers[0].self_attn.n_kv_heads}\n"
    )

    prompt_len = 256
    n_decode = 30

    print("=" * 100)
    print("Apples-to-apples: BOTH arms attend over the SAME KIVI-quantized state.")
    print("baseline = dequantize-to-fp16 then standard MLX SDPA (what a real KIVI-style cache")
    print("does today) | fused = scalar_fused_decode_attend on-the-fly, no dequant materialized")
    print(f"Qwen3-4B-4bit, prompt_len={prompt_len}, n_decode={n_decode}")
    print("=" * 100)
    print(f"{'B':>3} {'TTFT base (s)':>14} {'decode tok/s base':>18} | "
          f"{'TTFT fused (s)':>15} {'decode tok/s fused':>19} {'speedup':>8}")

    for B in (1, 4, 16, 32):
        _route_count["decode_dequant_baseline"] = 0
        _route_count["fallback"] = 0
        _patch_sdpa_for_scalar_attend(model, nsg=4, use_fused=False)
        ttft_base, tps_base = _run_decode(model, tokenizer, B, prompt_len, n_decode)
        n_base = _route_count["decode_dequant_baseline"]
        _unpatch_sdpa()

        _route_count["decode_fused"] = 0
        _route_count["fallback"] = 0
        _patch_sdpa_for_scalar_attend(model, nsg=4, use_fused=True)
        ttft_fused, tps_fused = _run_decode(model, tokenizer, B, prompt_len, n_decode)
        n_fused, n_fallback = _route_count["decode_fused"], _route_count["fallback"]
        _unpatch_sdpa()

        print(
            f"{B:>3} {ttft_base:>14.3f} {tps_base:>18.1f} | "
            f"{ttft_fused:>15.3f} {tps_fused:>19.1f} {tps_fused / tps_base:>7.2f}x"
            f"   (routed: {n_base} dequant-baseline, {n_fused} fused-decode, {n_fallback} fallback)"
        )

    print(
        "\nReading this table: both arms use the SAME _ScalarAttendKIVICache "
        "(same quantized codes, same scale/zero, same group_size=32) for "
        "every decode step past the first group_size tokens — the ONLY "
        "difference is whether attention dequantizes to fp16 first (baseline, "
        "what a real KIVI-style cache does today) or runs "
        "scalar_fused_decode_attend directly against the quantized codes "
        "(fused). This isolates the SAME occupancy question the roofline "
        "doc's B-sweep asked synthetically "
        "(docs/KV_KERNEL_ROOFLINE_FINDINGS.md's occupancy sweep table), now "
        "measured through a real model's real forward pass, real prompts, and "
        "real greedy decoding — TTFT is real prefill latency (both arms use "
        "identical prefill; expect no difference), decode tok/s is real "
        "end-to-end throughput including all non-attention work (embeddings, "
        "MLPs, o_proj, sampling, per-step quantization), not an isolated "
        "kernel call."
    )


if __name__ == "__main__":
    main()
