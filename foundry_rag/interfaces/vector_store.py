"""
interfaces/vector_store.py — Abstract vector store contract.

Any vector store must implement this interface.
Implementations: azure/vector_store.py (Azure AI Search)
                 local/chroma_store.py (Chroma, for dev/testing)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """Single result returned from a vector search."""
    chunk_id: str
    text: str
    score: float                        # cosine similarity 0.0–1.0
    doc_type: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchRequest:
    """Query contract for vector search."""
    query_vector: list[float]
    top_k: int = 5
    min_score: float = 0.0
    # Metadata filters — applied before vector search (Azure AI Search $filter)
    filters: dict[str, Any] = field(default_factory=dict)
    # ACL enforcement — only return chunks the caller can access
    acl_groups: list[str] = field(default_factory=list)


class BaseVectorStore(ABC):

    @abstractmethod
    async def upsert(self, documents: list[dict[str, Any]]) -> int:
        """
        Upsert documents into the index.

        Args:
            documents: list of dicts from Chunk.to_search_document()
                       Must include embedding vector.

        Returns:
            count of successfully upserted documents

        Raises:
            VectorStoreError: on index failure
        """
        ...

    @abstractmethod
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """
        Vector similarity search with optional metadata filtering.

        Returns results sorted by score descending.
        Enforces acl_groups if provided — never returns chunks
        the caller is not authorized to see.
        """
        ...

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int:
        """
        Delete chunks by ID.
        Returns count of deleted documents.
        """
        ...

    @abstractmethod
    async def create_index(self) -> None:
        """
        Create the vector index with correct schema.
        Idempotent — safe to call if index already exists.
        """
        ...

    @abstractmethod
    async def index_exists(self) -> bool:
        """Return True if the index exists and is ready."""
        ...

    @abstractmethod
    async def document_count(self) -> int:
        """Return total number of documents in the index."""
        ...


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""
    def __init__(self, operation: str, reason: str, cause: Exception | None = None):
        self.operation = operation
        self.reason = reason
        self.cause = cause
        super().__init__(f"[{operation}] {reason}" + (f": {cause}" if cause else ""))
