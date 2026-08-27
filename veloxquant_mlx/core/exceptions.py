from __future__ import annotations


class QuantizerConfigError(ValueError):
    """Raised when a quantizer or KV cache is misconfigured."""


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a required precomputed artifact is missing from the store."""


class CyclicPipelineError(RuntimeError):
    """Raised when a QuantizationGraph contains a cycle."""


class CodebookDimensionMismatch(ValueError):
    """Raised when a codebook's shape does not match the expected dimension."""


class BlockPoolExhaustedError(RuntimeError):
    """Raised when a BlockPoolAllocator has no free blocks left to allocate."""


class OwnerAlreadyActiveError(ValueError):
    """Raised when an owner id is reused while still checked out to another caller."""
