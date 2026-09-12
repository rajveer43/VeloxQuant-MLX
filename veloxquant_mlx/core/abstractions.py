"""Abstract base classes defining every extension point in the pipeline.

Declares the interfaces every concrete implementation across the codebase
satisfies: :class:`Quantizer` (encode/decode/estimate_inner_product),
:class:`Codebook` (quantize/dequantize), :class:`KVCache`
(append_key/append_value/attend/memory_bytes), :class:`ArtifactStore`
(load/save rotation matrices, codebooks, JL matrices), :class:`Preconditioner`
and :class:`Transform` (linear/invertible preconditioning), the Chain of
Responsibility :class:`QuantizationHandler` pipeline stage, and the
:class:`InnerProductStrategy`, :class:`CodebookStrategy`, and
:class:`QuantizationObserver` strategy/observer hooks. New quantization
methods, cache types, or artifact backends are added by subclassing one of
these rather than duck-typing against a concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from veloxquant_mlx.core.context import EncodedVector, QuantizationContext, TransformResult


class Quantizer(ABC):
    """Abstract base class for all vector quantizers.

    All public-facing quantizer objects must satisfy this interface.
    """

    @abstractmethod
    def encode(self, x: Any) -> EncodedVector:
        """Encode a batch of vectors into a compact representation.

        Args:
            x: Input array of shape (batch, d), fp16.

        Returns:
            EncodedVector containing the compressed representation.
        """

    @abstractmethod
    def decode(self, ev: EncodedVector) -> Any:
        """Reconstruct approximate vectors from an EncodedVector.

        Args:
            ev: Encoded representation produced by encode().

        Returns:
            Reconstructed array of shape (batch, d), fp16.
        """

    @abstractmethod
    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any:
        """Estimate inner products between a query and encoded keys.

        Args:
            q: Query vector, shape (d,) or (1, d), fp16.
            ev: Encoded key cache, batch_size == n_keys.

        Returns:
            Estimated inner products, shape (batch_size,), fp16.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Preconditioner(ABC):
    """Abstract base class for linear preconditioners (rotation/JL sketch)."""

    @abstractmethod
    def apply(self, x: Any) -> Any:
        """Apply the forward preconditioner transform.

        Args:
            x: Input array of shape (batch, d).

        Returns:
            Transformed array of shape (batch, out_dim).
        """

    @abstractmethod
    def apply_inverse(self, y: Any) -> Any:
        """Apply the inverse (transpose) preconditioner transform.

        Args:
            y: Transformed array of shape (batch, out_dim).

        Returns:
            Reconstructed array of shape (batch, d).
        """


class Codebook(ABC):
    """Abstract base class for scalar codebooks."""

    @abstractmethod
    def quantize(self, y: Any) -> Any:
        """Map coordinates to nearest-centroid indices.

        Args:
            y: Input array of shape (batch, d).

        Returns:
            Index array of shape (batch, d), dtype uint8.
        """

    @abstractmethod
    def dequantize(self, idx: Any) -> Any:
        """Retrieve centroid values for given indices.

        Args:
            idx: Index array of shape (batch, d), dtype uint8.

        Returns:
            Centroid array of shape (batch, d).
        """


class KVCache(ABC):
    """Abstract base class for KV cache implementations."""

    @abstractmethod
    def append_key(self, k: Any) -> None:
        """Append a new key vector to the cache.

        Args:
            k: Key vector, shape (d,), fp16.
        """

    @abstractmethod
    def append_value(self, v: Any) -> None:
        """Append a new value vector to the cache.

        Args:
            v: Value vector, shape (d,), fp16.
        """

    @abstractmethod
    def attend(self, q: Any) -> Any:
        """Compute attention-weighted value for a query.

        Args:
            q: Query vector, shape (d,), fp16.

        Returns:
            Attention output, shape (d,), fp16.
        """

    @abstractmethod
    def memory_bytes(self) -> int:
        """Return current memory footprint of the cache in bytes."""

    def append(self, k: Any, v: Any) -> None:
        """Append a key-value pair in one call.

        Args:
            k: Key vector, shape (d,).
            v: Value vector, shape (d,).
        """
        self.append_key(k)
        self.append_value(v)

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self)})"


class Transform(ABC):
    """Abstract base class for invertible vector transforms (used by PolarQuant)."""

    @abstractmethod
    def forward(self, x: Any) -> TransformResult:
        """Apply the forward transform.

        Args:
            x: Input array of shape (batch, d).

        Returns:
            TransformResult encapsulating all intermediate and final values.
        """

    @abstractmethod
    def inverse(self, result: TransformResult) -> Any:
        """Apply the inverse transform to reconstruct the original vector.

        Args:
            result: TransformResult from a prior forward() call.

        Returns:
            Reconstructed array of shape (batch, d).
        """


class QuantizationHandler(ABC):
    """Abstract base for Chain of Responsibility pipeline stages.

    Subclasses implement handle() to mutate a QuantizationContext and
    call _pass_to_next() to continue the chain.
    """

    _next: QuantizationHandler | None = None

    def set_next(self, handler: QuantizationHandler) -> QuantizationHandler:
        """Attach the next handler and return it to enable fluent chaining.

        Args:
            handler: The next handler in the chain.

        Returns:
            The handler argument, enabling a.set_next(b).set_next(c) idiom.
        """
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Process the context and optionally pass it downstream.

        Args:
            ctx: Mutable quantization context.

        Returns:
            Possibly mutated context (may be the same object).
        """

    def _pass_to_next(self, ctx: QuantizationContext) -> QuantizationContext:
        """Forward context to the next handler if one is attached.

        Args:
            ctx: Current context.

        Returns:
            Context after downstream processing (or the same ctx if end of chain).
        """
        if self._next is not None:
            return self._next.handle(ctx)
        return ctx

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Human-readable stage name used in DAG and Observer events."""


class InnerProductStrategy(ABC):
    """Strategy for estimating inner products between queries and encoded keys."""

    @abstractmethod
    def estimate(self, q: Any, encoded: EncodedVector) -> Any:
        """Estimate ⟨q, k⟩ for each encoded key.

        Args:
            q: Query vector, shape (d,) or (1, d).
            encoded: Compressed key representation.

        Returns:
            Estimated inner products, shape (batch_size,).
        """


class CodebookStrategy(ABC):
    """Strategy for computing optimal codebook centroids."""

    @abstractmethod
    def compute_centroids(self, b: int, d: int) -> Any:
        """Compute 2^b centroids for dimension d.

        Args:
            b: Bit-width (number of bits per code).
            d: Vector dimension (used to set distribution variance).

        Returns:
            Numpy array of shape (2^b,) containing sorted centroids.
        """


class ArtifactStore(ABC):
    """DAO interface for loading and saving precomputed quantization artifacts."""

    @abstractmethod
    def load_rotation_matrix(self, d: int, seed: int) -> Any:
        """Load a precomputed rotation matrix.

        Args:
            d: Matrix dimension.
            seed: Random seed used to generate the matrix.

        Returns:
            MLX array of shape (d, d), fp16.

        Raises:
            ArtifactNotFoundError: If the artifact does not exist.
        """

    @abstractmethod
    def save_rotation_matrix(self, Pi: Any, d: int, seed: int) -> None:
        """Persist a rotation matrix.

        Args:
            Pi: Array of shape (d, d).
            d: Dimension.
            seed: Seed used to generate Pi.
        """

    @abstractmethod
    def load_codebook(self, distribution: str, b: int, d: int) -> Any:
        """Load a precomputed codebook.

        Args:
            distribution: Distribution name (e.g. 'gaussian', 'beta').
            b: Bit-width.
            d: Vector dimension.

        Returns:
            MLX array of shape (2^b,), fp16.

        Raises:
            ArtifactNotFoundError: If the artifact does not exist.
        """

    @abstractmethod
    def save_codebook(self, cb: Any, distribution: str, b: int, d: int) -> None:
        """Persist a codebook.

        Args:
            cb: Codebook centroids array.
            distribution: Distribution name.
            b: Bit-width.
            d: Dimension.
        """

    @abstractmethod
    def load_jl_matrix(self, d: int, m: int, seed: int) -> Any:
        """Load a precomputed JL projection matrix.

        Args:
            d: Input dimension.
            m: Output (sketch) dimension.
            seed: Random seed.

        Returns:
            MLX array of shape (m, d), fp16.

        Raises:
            ArtifactNotFoundError: If the artifact does not exist.
        """

    @abstractmethod
    def save_jl_matrix(self, S: Any, d: int, m: int, seed: int) -> None:
        """Persist a JL projection matrix.

        Args:
            S: Array of shape (m, d).
            d: Input dimension.
            m: Output dimension.
            seed: Seed.
        """

    @abstractmethod
    def exists(self, artifact_type: str, **kwargs: Any) -> bool:
        """Check whether a specific artifact exists in the store.

        Args:
            artifact_type: One of 'rotation', 'codebook', 'jl'.
            **kwargs: Identifying parameters (d, seed, b, distribution, m).

        Returns:
            True if the artifact is present and loadable.
        """


class QuantizationObserver(ABC):
    """Observer for pipeline events (timing, memory, distortion)."""

    @abstractmethod
    def on_event(self, event: Any) -> None:
        """Handle a quantization pipeline event.

        Args:
            event: QuantizationEvent dataclass instance.
        """
