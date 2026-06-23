"""
models.py — Shared output contract for all chunkers.

Every chunker produces List[Chunk]. The Azure AI Search uploader
consumes List[ChunkSearchDocument]. Nothing else flows between layers.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DocType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    SLACK = "slack"
    TEAMS = "teams"
    CODE = "code"
    SAP = "sap"
    SQL = "sql"


class SensitivityTier(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class Chunk:
    """
    The canonical unit produced by every chunker.

    Rules:
    - text must be non-empty and >= 50 characters (quality gate minimum)
    - chunk_id is deterministic: SHA-256(source_id + chunk_index)
    - parent_doc_id links back to the originating document
    - acl_groups controls retrieval-time access filtering in Azure AI Search
    - embedding is None until the embed stage runs — chunkers do not embed
    """

    text: str
    doc_type: DocType
    source_id: str          # stable ID for the source document/thread/repo
    parent_doc_id: str      # same as source_id for top-level docs
    chunk_index: int        # 0-based position within the document
    chunk_total: int        # total chunks from this document (set post-split)

    # Provenance
    author: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    sensitivity_tier: SensitivityTier = SensitivityTier.INTERNAL

    # Access control — propagated from source document ACL
    acl_groups: list[str] = field(default_factory=list)

    # Source-type-specific metadata — kept flat for Azure AI Search filterability
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    # Set by embed stage, not by chunker
    embedding: list[float] | None = None

    @property
    def chunk_id(self) -> str:
        """Deterministic ID — same input always produces same ID."""
        raw = f"{self.source_id}::{self.chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def token_estimate(self) -> int:
        """Fast approximation: 1 token ≈ 4 characters (GPT tokenizer average)."""
        return len(self.text) // 4

    def to_search_document(self) -> dict[str, Any]:
        """
        Serialize to Azure AI Search index document schema.

        Field names match the index definition exactly — change here
        if the index schema changes, nowhere else.
        """
        if self.embedding is None:
            raise ValueError(
                f"Chunk {self.chunk_id} has no embedding. "
                "Run the embed stage before indexing."
            )

        return {
            "id": self.chunk_id,
            "text": self.text,
            "docType": self.doc_type.value,
            "sourceId": self.source_id,
            "parentDocId": self.parent_doc_id,
            "chunkIndex": self.chunk_index,
            "chunkTotal": self.chunk_total,
            "author": self.author,
            "createdAt": self.created_at.isoformat(),
            "language": self.language,
            "sensitivityTier": self.sensitivity_tier.value,
            "aclGroups": self.acl_groups,
            "tokenEstimate": self.token_estimate,
            "extraMetadata": self.extra_metadata,
            "contentVector": self.embedding,
        }

    def validate(self) -> list[str]:
        """
        Return list of validation errors. Empty list = valid.
        Called by the quality gate stage — not by chunkers themselves.
        """
        errors: list[str] = []
        if not self.text or not self.text.strip():
            errors.append("text is empty")
        if len(self.text.strip()) < 50:
            errors.append(f"text too short: {len(self.text.strip())} chars (min 50)")
        if self.token_estimate > 2048:
            errors.append(f"chunk too large: ~{self.token_estimate} tokens (max 2048)")
        if self.chunk_index < 0:
            errors.append(f"invalid chunk_index: {self.chunk_index}")
        return errors
