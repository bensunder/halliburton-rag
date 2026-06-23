"""
interfaces/pii_scrubber.py — Abstract PII scrubber contract.
interfaces/connector.py    — Abstract source connector contract.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator


# ===========================================================================
# PII Scrubber
# ===========================================================================

@dataclass
class ScrubResult:
    """Output of a PII scrub operation."""
    text: str                           # redacted text
    entities_found: list[str]           # entity types detected e.g. ["PERSON", "EMAIL"]
    was_modified: bool                  # True if any PII was redacted


class BasePIIScrubber(ABC):

    @abstractmethod
    async def scrub(self, text: str) -> ScrubResult:
        """
        Detect and redact PII from text.

        Replacements use typed placeholders:
          "John Smith" → "[PERSON]"
          "john@acme.com" → "[EMAIL]"
          "123-45-6789" → "[SSN]"
          "4111-1111-1111-1111" → "[CREDIT_CARD]"

        Args:
            text: raw text that may contain PII

        Returns:
            ScrubResult with redacted text and detected entity types

        Raises:
            PIIScrubError: on provider failure
        """
        ...

    @abstractmethod
    async def scrub_batch(self, texts: list[str]) -> list[ScrubResult]:
        """Scrub a batch of texts. More efficient than calling scrub() in a loop."""
        ...


class PIIScrubError(Exception):
    """Raised when PII scrubbing fails. Document must not proceed to embedding."""
    def __init__(self, reason: str, cause: Exception | None = None):
        self.reason = reason
        self.cause = cause
        super().__init__(f"{reason}" + (f": {cause}" if cause else ""))


# ===========================================================================
# Source Connector
# ===========================================================================

@dataclass
class RawDocument:
    """
    Output of a connector — raw content before parsing and chunking.

    source_id must be stable and unique across re-runs so the pipeline
    can detect and skip already-indexed documents.
    """
    source_id: str
    content: bytes | str            # raw bytes for binary (PDF), str for text
    content_type: str               # MIME type: "application/pdf", "text/plain", etc.
    filename: str
    author: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acl_groups: list[str] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """
    Abstract source connector.

    Connectors fetch raw documents from a source system.
    They do not parse, chunk, or embed — that is the pipeline's job.

    Implementations:
        connectors/blob_storage.py   — Azure Blob Storage
        connectors/confluence.py     — Confluence REST API v2
        connectors/slack.py          — Slack Events API
        connectors/sql.py            — SQLAlchemy
        connectors/git.py            — GitPython
        connectors/sap_csv.py        — pandas SAP CSV reader
    """

    @abstractmethod
    async def fetch_all(self) -> AsyncIterator[RawDocument]:
        """
        Yield all documents from the source.
        Used for initial full ingestion.

        Implementations must:
        - Handle pagination transparently
        - Yield one RawDocument at a time (do not buffer all in memory)
        - Set stable source_id on every document
        - Propagate acl_groups from source system permissions
        """
        ...

    @abstractmethod
    async def fetch_since(self, since: datetime) -> AsyncIterator[RawDocument]:
        """
        Yield documents modified since a given datetime.
        Used for incremental re-ingestion (scheduled runs).
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify connectivity to the source system.
        Called before ingestion starts — fails fast rather than
        discovering auth errors mid-run.
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name for logging. e.g. 'Confluence', 'SAP AP Export'."""
        ...


class ConnectorError(Exception):
    """Raised when a connector cannot reach or authenticate to its source."""
    def __init__(self, source: str, reason: str, cause: Exception | None = None):
        self.source = source
        self.reason = reason
        self.cause = cause
        super().__init__(f"[{source}] {reason}" + (f": {cause}" if cause else ""))
