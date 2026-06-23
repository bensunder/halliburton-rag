"""
interfaces/embedder.py — Abstract embedder contract.

Any embedding provider must implement this interface.
Implementations: azure/embedder.py (Azure OpenAI)
                 local/embedder.py (sentence-transformers, offline)
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: list of strings to embed — must be non-empty,
                   each string must be non-empty

        Returns:
            list of float vectors, same length and order as input

        Raises:
            EmbeddingError: on provider failure, rate limit, or auth error
        """
        ...

    @abstractmethod
    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Convenience wrapper around embed()."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensions this embedder produces. e.g. 3072 for text-embedding-3-large."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier. e.g. 'text-embedding-3-large'."""
        ...


class EmbeddingError(Exception):
    """Raised when embedding fails. Caller routes document to DLQ."""
    def __init__(self, reason: str, retryable: bool = True, cause: Exception | None = None):
        self.reason = reason
        self.retryable = retryable
        self.cause = cause
        super().__init__(f"{reason}" + (f": {cause}" if cause else ""))
