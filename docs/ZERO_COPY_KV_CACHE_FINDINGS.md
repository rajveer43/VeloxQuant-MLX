# Zero-Copy KV-Cache Access — Investigation (issue #255)

## TL;DR

The memory-marshaling cost issue #255 asks about is real, but it is
**concentrated in the standalone research caches** (`turboquant_prod`,
`turboquant_mse`, `polar`, `qjl`, `spectral` — `STANDALONE_METHODS` in
[`base.py`](../veloxquant_mlx/cache/base.py)), not in the servable
mlx_lm-protocol caches. Those store K/V as **NumPy arrays** and round-trip
through `mx.array()` / `np.array()` on every single `append_key` /
`append_value` / `attend` call. The servable caches (`kivi`, `turboquant_rvq`,
etc.) already stay MLX-native end to end — `kivi_cache.py` has zero
`np.array`/`mx.array` conversions.

Measured on `TurboQuantKVCache` (`turboquant_prod`, d=128, b=2, 500 cached
tokens, M-series Metal backend):

| Path | `attend()` wall time | Marshaling share |
|---|---|---|
| Default (`enable_vectorized_attend=False`) | 12.7 ms/call | **89%** (11.3 ms) is the scalar Python unpack loop + array conversions |
| `enable_vectorized_attend=True` | 0.48 ms/call (**26x faster**) | 7% (34 µs) is residual `np.array` slice → `mx.array()` marshaling |

The big win (26x) is an *existing, already-shipped* flag
(`enable_vectorized_attend`) that most callers don't turn on. The *further*
zero-copy win the issue is asking about — eliminating the NumPy round trip
entirely — is real but small in comparison: ~34 µs/call, ~7% of the
already-optimized path.

## Where the marshaling actually happens

`TurboQuantKVCache` ([turboquant_cache.py](../veloxquant_mlx/cache/turboquant_cache.py))
stores everything as `np.ndarray` ring buffers:

```python
self._k_indices_packed = np.zeros((capacity, self._idx_packed_len), dtype=np.uint8)
self._k_signs_packed   = np.zeros((capacity, self._sign_packed_len), dtype=np.uint8)
self._v_cache           = np.zeros((capacity, self._d), dtype=np.int8)
```

Every `append_key` converts the incoming MLX key to NumPy to write it in:

```python
k_np = np.array(k, dtype=np.float16).reshape(1, -1)   # mx -> np
...
idx_np = np.array(ev.indices[0], dtype=np.uint8)        # mx -> np (again, post-encode)
```

Every `attend` slices the NumPy ring buffer by physical index, unpacks bits
(in a per-row Python loop unless `enable_vectorized_attend=True`), and
converts back to MLX to run the actual math:

```python
phys = self._physical_indices(n)
k_indices_np = self._unpack_indices_block(self._k_indices_packed[phys])  # np slice + np loop
k_indices = mx.array(k_indices_np, dtype=mx.uint8)                        # np -> mx
...
v_int8 = mx.array(self._v_cache[phys], dtype=mx.int8)                     # np -> mx
```

This hits all four boundaries issue #255 called out:
- **compressed cache → attention**: `np.array(self._k_indices_packed[phys], ...)` and friends.
- **attention → compressed cache**: `np.array(v_int8, dtype=np.int8)` in `append_value`.
- **quantization → cache write**: `idx_np = np.array(ev.indices[0], ...)` in `append_key`.
- **dequantization → attention input**: the `k_indices`/`v_int8` reconstruction inside `attend()`.

`spectral_cache.py` (11 conversions) and `polar_cache.py`/`qjl_cache.py` (2
each) follow the same pattern at smaller scale. The mlx_lm-protocol caches
(`kivi_cache.py`, `turboquant_rvq_cache.py`, and the 35+ others reachable via
`KVCacheBuilder.for_model()`) already use MLX arrays as native storage — MLX
on Apple Silicon uses unified memory, so those never leave GPU-addressable
memory in the first place. This split exists because the standalone caches
predate the mlx_lm-protocol design and were built as a `KVCache` ABC
(`append_key`/`append_value`/`attend`) around plain NumPy ring buffers for
simplicity, not around MLX's own array/buffer lifecycle.

## Why `mx.array(numpy_array)` isn't "free" even on unified memory

Apple Silicon's unified memory means CPU and GPU share physical RAM, but
`mx.array(np_array)` still isn't a no-op: MLX has to wrap the NumPy buffer
in its own array/allocator bookkeeping and (depending on the array's origin
and stride) may need to copy to get contiguous data with MLX-owned lifetime.
Microbenchmark, isolated from any cache logic:

```
np.array((1000,128) fp16) -> mx.array(): ~27 µs/call
mx.array((1000,128) fp16) -> np.array(): ~4 µs/call
```

Per-call these look small, but `append_key`/`attend` do this *per token*,
and a naive unpack does it inside a **Python `for` loop over tokens** in
`_unpack_indices_block`/`_unpack_signs_block` when `enable_vectorized_attend`
is off — that loop, not the array-boundary crossing, is what actually
dominates (11.3 of 12.7 ms). The already-vectorized path removes the loop
and leaves only the structural marshaling (~34 µs), which is the true
"unnecessary memory marshaling between intermediate representations and the
actual KV-cache storage" the issue describes.

## Recommendation

1. **No large rewrite is justified by the data.** A full port of the 5
   standalone caches to MLX-native storage would chase a ~7%
   (`enable_vectorized_attend=True` baseline) tail, at real cost: NumPy
   ring buffers give O(1) mutable in-place slot writes
   (`self._v_cache[slot, :] = ...`) that MLX arrays don't offer as cheaply
   (MLX arrays are typically treated as immutable-per-step in these
   codepaths; in-place mutation would need `mx.array.at[...].set` gymnastics
   or reallocation-per-write, which could easily cost more than the
   marshaling it removes).
2. **`enable_vectorized_attend` should default to `True`** for the
   standalone methods, or at minimum be surfaced more prominently — it is
   the actual 26x win here and already shipped, just opt-in
   (`KVCacheConfig.enable_vectorized_attend: bool = False`,
   [base.py:108](../veloxquant_mlx/cache/base.py)). This is the
   highest-leverage, lowest-risk follow-up from this investigation.
3. **True zero-copy (MLX-native ring buffers) is worth prototyping only for
   the standalone caches under sustained decode workloads** (long-context
   generation where `attend()` is called thousands of times), where the
   residual ~34 µs/call compounds. It is not worth it for the servable
   caches, which already avoid the problem, and note the `fused_sdpa`
   Metal-kernel path (`metal/fused_sdpa.py`, VecInfer) already demonstrates
   the fully zero-copy end state — indices and codebooks live in MLX/Metal
   buffers throughout, with no NumPy intermediate at all. Any MLX-native
   rewrite of the standalone caches should follow that path's design rather
   than inventing a new one.
4. **quantization → cache write** and **compressed cache → attention** are
   the two boundaries worth targeting first if this is pursued, since they
   run once per token on the hot decode path; **attention → compressed
   cache** (`append_value`) and **dequantization → attention input** are
   comparatively cheap already (no unpack loop involved).

## Reproduction

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.profiling.kv_profiler import KVCacheProfiler, format_profile_table, profile_layers
import mlx.core as mx

cfg = KVCacheConfig(method="turboquant_prod", head_dim=128, bit_width_inlier=2, seed=42)
cache = KVCacheFactory.create(cfg)
prof = KVCacheProfiler(cache, head_dim=128, layer_id=0)

mx.random.seed(0)
keys = mx.random.normal((500, 128)).astype(mx.float16)
vals = mx.random.normal((500, 128)).astype(mx.float16)
q = mx.random.normal((128,)).astype(mx.float16)
mx.eval(keys, vals, q)

for i in range(500):
    prof.append_key(keys[i]); prof.append_value(vals[i]); prof.attend(q)

print(format_profile_table(profile_layers([prof])))
```

Re-run with `enable_vectorized_attend=True` in the config to reproduce the
26x improvement.
