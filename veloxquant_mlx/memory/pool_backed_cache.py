from __future__ import annotations

from typing import Any

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator

# Reserved for caches that have not been given an explicit owner id.
_DEFAULT_OWNER: int = -1


class PoolBackedKVCache:
    """Drop-in replacement for ``mlx_lm.models.cache.KVCache`` whose growth
    is tracked by a shared :class:`BlockPoolAllocator`.

    ``mlx_lm``'s stock ``KVCache`` grows its backing ``mx.array`` in fixed
    ``step=256``-token chunks, hardcoded per instance, with no visibility
    into how many times that growth happened across a server's caches or
    how much of the last chunk is wasted. This class keeps the same
    contiguous-buffer growth strategy — attention needs one contiguous
    ``(B, n_kv_heads, seq_len, head_dim)`` tensor every step, and RoPE /
    masking read ``cache.offset`` directly, so growth cannot be replaced
    with true block-paged storage without a per-step gather cost — but
    routes every growth step through ``pool.allocate()`` so it shows up in
    the pool's :class:`~veloxquant_mlx.memory.block_pool.AllocationStats`
    (allocation count, reuse, fragmentation) the same way any other pool
    consumer's growth does, and the growth chunk size becomes the pool's
    configured ``block_size`` instead of a hardcoded constant.

    This is the piece that puts :class:`BlockPoolAllocator` in a real
    model's decode hot path: unlike
    :class:`~veloxquant_mlx.memory.pooled_cache.PooledKVCache`, which wraps
    VeloxQuant's own ``append_key``/``append_value``/``attend`` interface
    (only implemented by the 5 "standalone" methods — see
    ``STANDALONE_METHODS`` in ``veloxquant_mlx/cache/base.py``), this class
    implements ``mlx_lm``'s ``update_and_fetch`` protocol directly, so it
    plugs into ``mlx_lm.generate()`` for any model exactly where a stock
    ``KVCache`` would.

    Args:
        pool: Shared BlockPoolAllocator to draw growth accounting from.
        owner: Opaque id identifying this cache's request/sequence. Two
            concurrent ``PoolBackedKVCache`` instances must use different
            owners so :meth:`release` only reclaims their own blocks.
        step: Token-chunk size to grow the backing buffer by. Defaults to
            ``pool.config.block_size`` so the pool's own block sizing is
            what determines growth granularity; pass an explicit value to
            decouple them (e.g. a larger step for a KV-heavy model while
            keeping a small pool block_size for finer-grained accounting).
    """

    def __init__(
        self,
        pool: BlockPoolAllocator,
        owner: int = _DEFAULT_OWNER,
        step: int | None = None,
    ) -> None:
        self.pool = pool
        self.owner = owner
        self.step = step if step is not None else pool.config.block_size
        self.keys: Any = None
        self.values: Any = None
        self.offset = 0

    def update_and_fetch(self, keys: Any, values: Any):
        """Append one decode step's keys/values and return the full history.

        Args:
            keys: New keys, shape (B, n_kv_heads, S, head_dim).
            values: New values, shape (B, n_kv_heads, S, head_dim).

        Returns:
            (keys, values) accumulated so far, each shape
            (B, n_kv_heads, offset, head_dim).
        """
        import mlx.core as mx

        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            n_new_tokens = n_steps * self.step
            k_shape = (B, n_kv_heads, n_new_tokens, k_head_dim)
            v_shape = (B, n_kv_heads, n_new_tokens, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            self.pool.allocate(stream="k", n_tokens=n_new_tokens, owner=self.owner)
            self.pool.allocate(stream="v", n_tokens=n_new_tokens, owner=self.owner)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def release(self) -> None:
        """Return every block this cache's growth checked out to the pool.

        Call this once the request finishes so its accounted blocks free
        up pool headroom for the next request. The backing mx.array itself
        is reclaimed by MLX's normal garbage collection when this object
        goes out of scope; this only clears the pool's bookkeeping.
        """
        self.pool.free_all(owner=self.owner)

    def size(self) -> int:
        return self.offset

    @property
    def state(self):
        if self.keys is None:
            return self.keys, self.values
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, v) -> None:
        self.keys, self.values = v
        self.offset = self.keys.shape[2] if self.keys is not None else 0

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v) -> None:
        if v is not None and v:
            raise ValueError("PoolBackedKVCache has no meta_state but a meta_state was set.")

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    def __repr__(self) -> str:
        return f"PoolBackedKVCache(owner={self.owner}, offset={self.offset}, step={self.step})"


def build_pooled_caches(
    model, pool: BlockPoolAllocator, owner: int, step: int | None = None
) -> list:
    """Build one PoolBackedKVCache per language-model layer.

    Drop-in replacement for ``mlx_lm.models.cache.make_prompt_cache`` /
    ``model.make_cache()`` for the "plain fp16, pool-tracked growth" case —
    every attention-bearing layer gets a PoolBackedKVCache sharing the same
    pool and owner, so a single ``pool.free_all(owner)`` (or calling
    :meth:`PoolBackedKVCache.release` on any one of them) releases every
    layer's growth accounting for this request at once.

    Args:
        model: Loaded mlx_lm model instance.
        pool: Shared BlockPoolAllocator every layer's cache draws from.
        owner: Opaque id (e.g. request id) shared by every layer's cache
            for this request.
        step: Growth chunk size in tokens; defaults to
            ``pool.config.block_size`` (see PoolBackedKVCache).

    Returns:
        List of PoolBackedKVCache instances, one per language-model layer.
    """
    layers = getattr(model, "layers", None) or model.model.layers
    return [PoolBackedKVCache(pool, owner=owner, step=step) for _ in layers]
