"""
base.py — Abstract base class for all chunkers.

Contract: chunk(source) -> List[Chunk]

Every concrete chunker:
  1. Accepts a source-specific input type
  2. Returns List[Chunk] with chunk_total set correctly on every chunk
  3. Does NOT embed — that is the embed stage's job
  4. Does NOT write to Azure AI Search — that is the uploader's job
  5. Raises ChunkingError on unrecoverable parse failure (caller sends to DLQ)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .models import Chunk

S = TypeVar("S")  # Source type — different per chunker


class ChunkingError(Exception):
    """
    Raised when a chunker cannot process a source document.
    Caller should catch this and route the document to the dead-letter queue.
    """
    def __init__(self, source_id: str, reason: str, cause: Exception | None = None):
        self.source_id = source_id
        self.reason = reason
        self.cause = cause
        super().__init__(f"[{source_id}] {reason}" + (f": {cause}" if cause else ""))


class BaseChunker(ABC, Generic[S]):
    """
    Abstract base for all chunkers.

    Subclasses implement _split(source) and return raw chunks.
    Base class handles:
    - Setting chunk_total on every chunk after split
    - Filtering empty chunks before returning
    - Logging chunk count
    """

    @abstractmethod
    def _split(self, source: S) -> list[Chunk]:
        """
        Parse and split source into chunks.
        chunk_total will be set by the base class after this returns —
        implementations do not need to set it.
        """
        ...

    def chunk(self, source: S) -> list[Chunk]:
        """
        Public entry point. Call this, not _split directly.
        Raises ChunkingError on failure — never returns partial results.
        """
        chunks = self._split(source)
        chunks = [c for c in chunks if c.text.strip()]
        total = len(chunks)
        for c in chunks:
            c.chunk_total = total
        return chunks
