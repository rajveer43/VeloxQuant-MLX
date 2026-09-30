# Deferred eviction attention contract

Issue #610 extends deferred eviction to `knorm`, `qfilters`, `keyformer`,
`morphkv`, `kvzip`, `curdkv`, `nestedkv`, and `rocketkv`.

`mlx_lm` constructs a mask before calling `update_and_fetch`. Each call now
returns the previously retained K/V rows followed by **all** new rows. Eviction
updates the stored state for the next call. Consequently, the current return
can exceed the configured storage budget; callers inspecting compression must
read retained state rather than infer it from the returned attention inputs.
NestedKV and RocketKV retain their existing policies of allowing decode storage
to grow after prefill compression.

Masks track chronological survivor positions using the same selection indices
as eviction. Keyformer keeps separate chronological positions because its
quantizer also shifts RoPE positions. Its existing RoPE offset policy is
unchanged. MorphKV, KVzip, and NestedKV now advance the model-facing offset by
the number of input tokens, keeping incoming RoPE positions on the same axis
as retained keys. RocketKV's `state` excludes capacity padding.

The regression suite in
`veloxquant_mlx/tests/cache/test_deferred_eviction_contract.py` covers initial
prefill, chunks smaller and larger than the budget, single-token-first input,
decode, sliding-window visibility, batched GQA, calibrated QFilters on MLX and
Metal, and full-prefill logits against a tiny uncompressed Llama model.

## Remaining limitations

PyramidKV and Squeeze are outside this change. Their different per-layer budgets
are incompatible with `mlx_lm` reusing one mask for every layer; this requires
the separate architecture change in issue #390. Issue #610 therefore remains
partially open until those two methods are addressed.

As with the earlier eviction fixes, explicit masks use head zero's survivor
positions and the model may reuse layer zero's mask. Plain causal masking is
correct for the returned chronological layout: every old row precedes every
new query. Sliding-window visibility is exact for the represented head/layer;
other heads or layers that retain different token positions may need different
window masks. This change does not add per-head or per-layer model integration.
