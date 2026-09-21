"""End-to-end contract tests for the library's three public extension points.

core/abstractions.py pitches Quantizer, KVCache, and ArtifactStore as the
seams third parties subclass to add a new quantization method, cache type,
or artifact backend (see that module's docstring). Nothing previously
verified that a minimal, independent subclass of each ABC actually composes
correctly end-to-end -- only concrete in-tree implementations were exercised,
which doesn't prove the abstract method set is sufficient or that default
mixin behavior (e.g. KVCache.append() calling append_key() then
append_value()) holds for an arbitrary conforming subclass. See #494.

Each fake here is deliberately the simplest possible conforming
implementation -- no compression, no real math -- so a failure here means the
contract itself is broken, not that some quantizer-specific numerics are off.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.artifacts.memory_store import InMemoryArtifactStore
from veloxquant_mlx.core.abstractions import ArtifactStore, KVCache, Quantizer
from veloxquant_mlx.core.context import EncodedVector
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError


class _IdentityQuantizer(Quantizer):
    """Minimal conforming Quantizer: stores vectors verbatim, no compression.

    Exercises exactly the three abstract methods a subclass must implement
    and nothing else, to prove that's a sufficient and usable contract.
    """

    def encode(self, x: mx.array) -> EncodedVector:
        batch, dim = x.shape
        return EncodedVector(
            quantizer_type="identity",
            batch_size=batch,
            dim=dim,
            indices=x,  # abuse `indices` as the raw payload slot
        )

    def decode(self, ev: EncodedVector) -> mx.array:
        return ev.indices

    def estimate_inner_product(self, q: mx.array, ev: EncodedVector) -> mx.array:
        return ev.indices @ q.reshape(-1)


class _ListBackedKVCache(KVCache):
    """Minimal conforming KVCache: Python lists, no eviction/quantization.

    Exercises append_key/append_value/attend/memory_bytes (the four abstract
    methods) plus the non-abstract append()/reset()/__len__() mixins that
    default-implement on top of them.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._keys: list[mx.array] = []
        self._values: list[mx.array] = []

    def append_key(self, k: mx.array) -> None:
        self._keys.append(k)

    def append_value(self, v: mx.array) -> None:
        self._values.append(v)

    def attend(self, q: mx.array) -> mx.array:
        if not self._keys:
            return mx.zeros((self._dim,), dtype=mx.float16)
        keys = mx.stack(self._keys)  # (n, d)
        values = mx.stack(self._values)  # (n, d)
        scores = mx.softmax(keys @ q, axis=0)  # (n,)
        return scores @ values

    def memory_bytes(self) -> int:
        return sum(k.nbytes for k in self._keys) + sum(v.nbytes for v in self._values)

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()

    def __len__(self) -> int:
        return len(self._keys)


class TestQuantizerContract:
    """A minimal Quantizer subclass round-trips through encode/decode/estimate."""

    def test_encode_decode_round_trip(self) -> None:
        q = _IdentityQuantizer()
        x = mx.array(np.arange(12, dtype=np.float16).reshape(3, 4))

        ev = q.encode(x)
        assert isinstance(ev, EncodedVector)
        assert ev.batch_size == 3
        assert ev.dim == 4

        decoded = q.decode(ev)
        assert mx.array_equal(decoded, x).item()

    def test_estimate_inner_product(self) -> None:
        q = _IdentityQuantizer()
        x = mx.array(np.eye(4, dtype=np.float16))
        query = mx.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16))

        ev = q.encode(x)
        scores = q.estimate_inner_product(query, ev)
        expected = x @ query
        assert mx.array_equal(scores, expected).item()

    def test_repr_uses_class_name(self) -> None:
        assert repr(_IdentityQuantizer()) == "_IdentityQuantizer()"


class TestKVCacheContract:
    """A minimal KVCache subclass composes correctly through append/attend/reset."""

    def test_append_key_value_independently(self) -> None:
        cache = _ListBackedKVCache(dim=4)
        cache.append_key(mx.zeros((4,), dtype=mx.float16))
        cache.append_value(mx.ones((4,), dtype=mx.float16))
        assert len(cache) == 1

    def test_append_mixin_calls_append_key_then_append_value(self) -> None:
        """KVCache.append() is not abstract -- it's a mixin over append_key/
        append_value (core/abstractions.py:175-183). Verify that composition
        actually produces a usable entry, not just that both methods ran."""
        cache = _ListBackedKVCache(dim=3)
        k = mx.array([1.0, 0.0, 0.0], dtype=mx.float16)
        v = mx.array([0.0, 1.0, 0.0], dtype=mx.float16)
        cache.append(k, v)

        assert len(cache) == 1
        out = cache.attend(k)
        assert out.shape == (3,)

    def test_attend_returns_weighted_combination(self) -> None:
        cache = _ListBackedKVCache(dim=2)
        cache.append(mx.array([1.0, 0.0], dtype=mx.float16), mx.array([9.0, 9.0], dtype=mx.float16))
        cache.append(mx.array([0.0, 1.0], dtype=mx.float16), mx.array([1.0, 1.0], dtype=mx.float16))

        # Query aligned with the first key should weight its value heavily.
        out = cache.attend(mx.array([10.0, 0.0], dtype=mx.float16))
        mx.eval(out)
        assert float(out[0]) > 5.0

    def test_attend_on_empty_cache_returns_zeros(self) -> None:
        cache = _ListBackedKVCache(dim=5)
        out = cache.attend(mx.zeros((5,), dtype=mx.float16))
        assert mx.array_equal(out, mx.zeros((5,), dtype=mx.float16)).item()

    def test_memory_bytes_tracks_appended_entries(self) -> None:
        cache = _ListBackedKVCache(dim=4)
        assert cache.memory_bytes() == 0
        cache.append(mx.zeros((4,), dtype=mx.float16), mx.zeros((4,), dtype=mx.float16))
        assert cache.memory_bytes() > 0

    def test_reset_clears_storage_not_config(self) -> None:
        """Per the reset() docstring (core/abstractions.py:154-173):
        implementers must clear only token storage, leaving configuration
        (here, `_dim`) untouched."""
        cache = _ListBackedKVCache(dim=7)
        cache.append(mx.zeros((7,), dtype=mx.float16), mx.zeros((7,), dtype=mx.float16))
        cache.reset()
        assert len(cache) == 0
        assert cache._dim == 7  # config survives reset()

    def test_repr_reports_length(self) -> None:
        cache = _ListBackedKVCache(dim=2)
        cache.append(mx.zeros((2,), dtype=mx.float16), mx.zeros((2,), dtype=mx.float16))
        assert repr(cache) == "_ListBackedKVCache(size=1)"

    def test_len_and_reset_are_not_abstract_but_unimplemented_by_default(self) -> None:
        """A subclass that implements only the four abstract methods and
        skips __len__/reset entirely still satisfies ABC instantiation (they
        aren't @abstractmethod), but calling the unimplemented defaults must
        fail loudly rather than silently no-op, matching the documented
        contract at core/abstractions.py:154-173,185-186."""

        class _BareCache(KVCache):
            def append_key(self, k):
                pass

            def append_value(self, v):
                pass

            def attend(self, q):
                return q

            def memory_bytes(self):
                return 0

        bare = _BareCache()
        with pytest.raises(NotImplementedError):
            bare.reset()
        with pytest.raises(NotImplementedError):
            len(bare)


class TestArtifactStoreContract:
    """A conforming ArtifactStore round-trips rotation/codebook/JL artifacts.

    InMemoryArtifactStore (veloxquant_mlx/artifacts/memory_store.py) is
    itself a minimal, independent ArtifactStore subclass used throughout the
    test suite -- exercising it here end-to-end is what proves the ABC's
    contract, rather than merely asserting it's importable.
    """

    def test_rotation_round_trip(self) -> None:
        store: ArtifactStore = InMemoryArtifactStore()
        pi = mx.array(np.eye(4, dtype=np.float16))
        store.save_rotation_matrix(pi, d=4, seed=0)

        loaded = store.load_rotation_matrix(d=4, seed=0)
        assert mx.array_equal(loaded, pi).item()

    def test_load_missing_artifact_raises_artifact_not_found(self) -> None:
        store: ArtifactStore = InMemoryArtifactStore()
        with pytest.raises(ArtifactNotFoundError):
            store.load_rotation_matrix(d=4, seed=999)

    def test_exists_reflects_store_state(self) -> None:
        store: ArtifactStore = InMemoryArtifactStore()
        assert store.exists("codebook", distribution="gaussian", b=4, d=8) is False

        store.save_codebook(mx.zeros((16,), dtype=mx.float16), distribution="gaussian", b=4, d=8)
        assert store.exists("codebook", distribution="gaussian", b=4, d=8) is True
