# xKV — Cross-Layer Shared-Subspace Compression

**Method id:** `xkv` · **New in 0.27.0** · *Inspired by* [xKV (arXiv:2503.18893,
preprint)](https://arxiv.org/abs/2503.18893) — **xKV-adapted (VeloxQuant-MLX
implementation)**, faithful to the joint-SVD shared-subspace core, adapted at
the integration boundary via a shared fan-in/fan-out coordinator, and
simplified relative to the paper on grouping and decode-time reconstruction
(see [Adaptation notes](#adaptation-notes)).

xKV compresses the KV cache **across a group of nearby layers** by jointly
factorizing their key matrices into **one shared low-rank basis**, rather than
computing a separate basis per layer. The paper's empirical motivation: via
Centered Kernel Alignment (CKA), the *dominant singular subspaces* of
per-layer key caches are well aligned across groups of adjacent layers — far
more aligned than raw per-token cosine similarity suggests. A joint SVD over
the group's stacked keys captures that shared structure once; every member of
the group then stores only its own latent coordinates in that shared basis.

## How it differs from XQuant and MiniCache

The repo already ships two other cross-layer methods,
[XQuant](../algorithms/xquant) and [MiniCache](../algorithms/minicache). All
three exploit inter-layer redundancy, but via three structurally different
mechanisms:

| | XQuant | MiniCache | **xKV** |
|---|---|---|---|
| Mechanism | reuses an anchor's quantized **codes** | merges the **tensors** via SLERP | jointly factorizes a **group** into one shared SVD basis |
| Shared across layers | code assignment | direction vector | right singular vectors + mean (`V_g`, `K_mean_g`) |
| Group size | pairs (or N-way anchor+reusers) | pairs (or N-way primary+mergers) | fixed contiguous groups (`xkv_group_size`, any N) |
| Per-layer kept | own scale/zero (+ optional residual) | own magnitude scalars | own latent codes in the shared basis |
| Quantizes | yes (low-bit codes) | no — fp16 directions | yes (uniform-bit latent quantization) |
| Amortization axis | one anchor's codes reused by N reusers | one direction shared by 2 layers | one basis's storage cost shared by N layers |

XQuant shares *bin assignment*; MiniCache shares a *direction*; xKV shares an
entire *subspace* fit jointly across the group, which is the only one of the
three that requires seeing every group member's data *simultaneously* before
any of them can compress (a fan-in step), rather than one layer publishing
first and others reading its output.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="xkv",
    head_dim=128,
    xkv_group_size=2,            # layers per shared-subspace group (2 = pairs)
    xkv_rank=None,               # None -> energy-threshold rank selection
    xkv_energy_threshold=0.95,   # fraction of singular value energy to retain
    xkv_latent_bits=4,           # single-bit-width latent quantization
    xkv_group_quant_size=32,     # token group size for latent quantization
    xkv_max_ctx=8192,            # coordinator per-group token budget
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

:::note[Requires `for_model`]
Cross-layer subspace sharing needs the shared `XKVCoordinator` that
`KVCacheBuilder.for_model()` builds. Constructing a single cache via the
factory yields a degenerate (coordinator-less) **standalone member** —
equivalent to `xkv_group_size=1`, i.e. plain per-layer SVD compression with no
basis sharing. Useful for unit testing the projection/reconstruction path in
isolation.
:::

## How it works

**Grouping** (`pair_layers_grouped`, at build time): attention-bearing layers
are chunked into fixed-size contiguous groups of `xkv_group_size`. A trailing
partial group (fewer than `xkv_group_size` layers) is still a valid, smaller
group. Layer 0 of each group is the conventional "leader" (the only member
that reports the amortized basis storage cost — see
[Byte accounting](#byte-accounting)); the role is otherwise symmetric.

**Per forward pass**, for each group, at prefill:

1. Every member publishes its own raw keys for the current token range to the
   shared `XKVCoordinator` (a fan-in step).
2. Once **all** members of the group have published for that range, the
   coordinator computes a **single joint SVD** over the stacked (and
   group-mean-centered) key matrices, producing one shared basis
   `V_g` (right singular vectors) and `K_mean_g` (shared mean).
3. Every member — including whichever one triggered the computation —
   fetches and locally caches that identical basis (a fan-out step). A member
   that published *before* its peers finished uses a private, unshared,
   one-call fallback basis for that call's output only, and adopts the shared
   basis on its very next call.
4. Each member projects its own keys into `V_g`, quantizes the latent
   coordinates at a single bit-width (`xkv_latent_bits`), and reconstructs.

**Decode** (after the group's basis is frozen): each member projects new keys
directly into its already-cached `V_g` — no further coordinator interaction.
Unlike MiniCache (which coordinates every step), xKV's decode path needs no
cross-layer communication once prefill has settled.

## Byte accounting

- `compressed_key_bytes` — this layer's own latent codes only.
- `shared_basis_bytes` — the `V_g`/`K_mean_g` storage cost, reported as
  **nonzero only by the group's leader** (`member_idx == 0`). Followers
  report `0` here. This convention avoids double-counting the shared basis
  when a benchmark naively sums per-layer bytes across a model's layers — the
  correct group-level cost is the leader's `shared_basis_bytes` plus every
  member's own `compressed_key_bytes`.
- `fp16_key_bytes` / `value_fp16_bytes` — always the uncompressed cost.
  Values pass through unchanged (xKV compresses keys only, mirroring
  [SVDq](../algorithms/svdq)'s precedent in this repo).

## Adaptation notes

**Fidelity to the paper:** faithful to the core mechanism — CKA's empirical
motivation (dominant subspaces align across nearby layers) is implemented
exactly as a joint SVD over grouped, stacked key matrices, with a shared basis
amortized across the group. Adapted at the integration boundary: rather than
patching the attention forward pass, all per-layer caches share an
`XKVCoordinator` (the same pattern XQuant and MiniCache use), extended to a
fan-in-then-fan-out protocol since xKV — unlike those two — needs *every*
group member's data before any of them can compress.

**What we do NOT implement:**
- **CKA-based automatic layer grouping.** The paper picks which layers to
  group empirically per architecture; we use fixed-size contiguous groups
  (`xkv_group_size`) with no CKA validation step. A future version could add
  an optional CKA calibration pass.
- **"Selective Reconstruction."** The paper's decode-time latency
  optimization — exactly reconstructing a subset of group layers and deriving
  the rest — is a compute/latency trick orthogonal to the memory-compression
  mechanism. We fully reconstruct every layer on every fetch, like every
  other wrapper in this repo.
- **Values.** Keys only — the paper covers both tensors; we keep values fp16
  throughout, mirroring SVDq's existing precedent and keeping the byte
  accounting auditable.
- **Mixed-bit latent routing.** Unlike SVDq's importance-ranked hi/lo bit
  split, xKV's latent codes use a single uniform bit-width
  (`xkv_latent_bits`) — the shared basis is xKV's distinguishing feature, not
  a novel bit-allocation scheme. Callers who want mixed-bit latent coding on
  top of the shared basis can compose
  `veloxquant_mlx.quantizers.svdq.quantize_latents_mixed` directly.

**Known limitations:**
- A group member that publishes before all its peers have published for the
  same step produces that one call's output from a private, unshared basis
  (not stored) — a one-call transient during prefill, not a steady-state cost.
- No model-level (perplexity/throughput) benchmark has been run yet.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_xkv_cache.py` (14 tests) and
`veloxquant_mlx/tests/quantizers/test_xkv.py` (9 tests):

- Group-of-1 degeneracy: `joint_svd_compress` on a single matrix matches
  `svd_compress_keys` (SVDq's plain single-layer SVD) at the same rank
- Shared structure across synthetic layers reconstructs better than
  independent per-layer SVD on unrelated noise at the same rank (mechanism
  validation, not just plumbing)
- Round-trip projection/reconstruction recovers the input without
  quantization noise (float32 precision floor)
- All members of a group receive the identical shared basis after a settle
  round
- Only `member_idx == 0` reports nonzero `shared_basis_bytes`
- `compressed_key_bytes < fp16_key_bytes`
- Decode-time calls project into the frozen basis without re-triggering the
  joint SVD
- Coordinator `max_ctx` guard raises on prefill overflow
- `for_model` builds correct member/group assignment, including a trailing
  partial group
- Determinism

The offline harness in `benchmark_scripts/benchmark_xkv.py` (results in
`figures/xkv/results.json`) sweeps group size (2–4) and a
synthetic "shared fraction" knob against an independent-per-layer-SVD
baseline at matched rank:

- **Reconstruction quality (`mse_ratio`):** the shared basis lands within
  ~1% of independent per-layer SVD's MSE across every group size and shared
  fraction tested — essentially at parity, not a quality regression, even
  when synthetic structure is only weakly shared.
- **Byte savings (`byte_ratio`):** 0.80–0.92× the independent-SVD byte cost
  (8–20% fewer bytes), improving with larger group sizes — the amortization
  win the shared-basis mechanism is designed to deliver.
- **Output perturbation:** cosine-distance perturbation of a probe-query
  attention output is comparable between the shared and independent paths
  (within a few percent either direction across the sweep) — consistent with
  the near-parity MSE result above.

**No model-level benchmark has been run.** These are offline-synthetic,
reconstruction-quality and byte-accounting numbers — not perplexity or
throughput on a real model.

## When to use it

xKV targets models with groups of adjacent layers whose key caches share
structure (the paper's CKA finding suggests this is common in practice,
though we have not independently verified CKA alignment on any specific
model — see Adaptation notes). It is the natural complement to
[XQuant](../algorithms/xquant) and [MiniCache](../algorithms/minicache):
three different routes to the same inter-layer redundancy, distinguished by
*what* is shared (codes, a direction, or a subspace) and *how many* layers
can share it at once (pairs for XQuant/MiniCache's typical configuration; any
fixed group size for xKV).

| Method | Cross-layer mechanism | Quantizes | Amortized across |
|--------|----------------------|-----------|-------------------|
| XQuant | code reuse + residual | yes (low-bit) | one anchor -> N reusers |
| MiniCache | SLERP direction merge + retention | no (fp16 directions) | one pair |
| xKV | joint SVD -> shared subspace | yes (uniform-bit latents) | one basis -> N group members |
