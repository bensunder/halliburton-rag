"""
hierarchical.py — Hierarchical chunker for PDF and DOCX documents.

Strategy:
  1. Parse document into a heading/section tree using python-docx (DOCX)
     or pdfplumber + heuristic heading detection (PDF)
  2. Split at section boundaries first (respects semantic units)
  3. If a section exceeds max_tokens, split further at paragraph boundaries
  4. If a paragraph still exceeds max_tokens, apply sliding token window
     with overlap_ratio overlap

Why hierarchical over fixed-size?
  Fixed-size chunking cuts mid-sentence, mid-table, mid-argument.
  For compliance documents, procedures, and technical specs — the content
  Halliburton QA will query — section integrity is critical. A chunk
  containing half a procedure step and half the next one cannot be
  reliably retrieved or answered.

Dependencies:
  pip install python-docx pdfplumber tiktoken
"""

from __future__ import annotations

from ._tokenizer import token_count as _token_count

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ._tokenizer import token_count as _token_count_fn

from .base import BaseChunker, ChunkingError
from .models import Chunk, DocType, SensitivityTier

# GPT-4o tokenizer — same tokenizer used by text-embedding-3-large




def _token_chunks(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> Iterator[str]:
    """
    Sliding window split on token boundaries.
    Used only as a last resort when a paragraph exceeds max_tokens.
    """
    tokens = _ENC.encode(text)
    step = max(1, max_tokens - overlap_tokens)
    start = 0
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        yield _ENC.decode(window)
        start += step


@dataclass
class HierarchicalChunkerConfig:
    max_tokens: int = 512
    overlap_ratio: float = 0.10      # overlap = max_tokens * overlap_ratio
    min_section_chars: int = 80      # ignore headings with less content than this
    sensitivity_tier: SensitivityTier = SensitivityTier.INTERNAL
    acl_groups: list[str] = None     # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.acl_groups is None:
            self.acl_groups = []

    @property
    def overlap_tokens(self) -> int:
        return int(self.max_tokens * self.overlap_ratio)


@dataclass
class DocumentSource:
    """Input contract for HierarchicalChunker."""
    file_path: Path
    source_id: str
    author: str = ""
    created_at: datetime = None       # type: ignore[assignment]
    language: str = "en"
    extra_metadata: dict = None       # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.extra_metadata is None:
            self.extra_metadata = {}
        if not self.file_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.file_path}")


class HierarchicalChunker(BaseChunker[DocumentSource]):
    """
    Hierarchical PDF/DOCX chunker.

    Usage:
        chunker = HierarchicalChunker(HierarchicalChunkerConfig(
            max_tokens=512,
            overlap_ratio=0.10,
            acl_groups=["halliburton-qa-coe"],
        ))
        chunks = chunker.chunk(DocumentSource(
            file_path=Path("procedures/AP_audit_checklist.pdf"),
            source_id="doc-ap-audit-2024",
            author="Cheronda Bright",
        ))
    """

    def __init__(self, config: HierarchicalChunkerConfig | None = None) -> None:
        self.config = config or HierarchicalChunkerConfig()

    def _split(self, source: DocumentSource) -> list[Chunk]:
        suffix = source.file_path.suffix.lower()
        try:
            if suffix == ".docx":
                sections = self._parse_docx(source.file_path)
                doc_type = DocType.DOCX
            elif suffix == ".pdf":
                sections = self._parse_pdf(source.file_path)
                doc_type = DocType.PDF
            else:
                raise ChunkingError(
                    source.source_id,
                    f"Unsupported file type: {suffix}. Expected .docx or .pdf",
                )
        except ChunkingError:
            raise
        except Exception as e:
            raise ChunkingError(source.source_id, "Failed to parse document", e) from e

        chunks: list[Chunk] = []
        chunk_index = 0

        for section_title, section_text in sections:
            if len(section_text.strip()) < self.config.min_section_chars:
                continue

            for para_text in self._split_paragraphs(section_text):
                para_text = para_text.strip()
                if not para_text:
                    continue

                # Prepend section title to every chunk for retrieval context
                context_prefix = f"{section_title}\n\n" if section_title else ""
                full_text = context_prefix + para_text

                if _token_count(full_text) <= self.config.max_tokens:
                    chunks.append(self._make_chunk(
                        text=full_text,
                        doc_type=doc_type,
                        source=source,
                        chunk_index=chunk_index,
                        section_title=section_title,
                    ))
                    chunk_index += 1
                else:
                    # Paragraph too large — sliding window
                    for window_text in _token_chunks(
                        full_text,
                        self.config.max_tokens,
                        self.config.overlap_tokens,
                    ):
                        chunks.append(self._make_chunk(
                            text=window_text,
                            doc_type=doc_type,
                            source=source,
                            chunk_index=chunk_index,
                            section_title=section_title,
                        ))
                        chunk_index += 1

        return chunks

    def _make_chunk(
        self,
        text: str,
        doc_type: DocType,
        source: DocumentSource,
        chunk_index: int,
        section_title: str,
    ) -> Chunk:
        return Chunk(
            text=text,
            doc_type=doc_type,
            source_id=source.source_id,
            parent_doc_id=source.source_id,
            chunk_index=chunk_index,
            chunk_total=0,          # set by BaseChunker.chunk()
            author=source.author,
            created_at=source.created_at,
            language=source.language,
            sensitivity_tier=self.config.sensitivity_tier,
            acl_groups=list(self.config.acl_groups),
            extra_metadata={
                "file_name": source.file_path.name,
                "section_title": section_title,
                **source.extra_metadata,
            },
        )

    # ------------------------------------------------------------------
    # DOCX parsing
    # ------------------------------------------------------------------

    def _parse_docx(self, path: Path) -> list[tuple[str, str]]:
        """
        Returns list of (section_title, section_body) tuples.
        Heading 1/2 styles become section boundaries.
        """
        try:
            import docx  # python-docx
        except ImportError as e:
            raise ChunkingError(
                str(path), "python-docx not installed. Run: pip install python-docx", e
            ) from e

        doc = docx.Document(str(path))
        sections: list[tuple[str, str]] = []
        current_title = ""
        current_body: list[str] = []

        heading_styles = {"heading 1", "heading 2", "heading 3", "title"}

        for para in doc.paragraphs:
            style_name = para.style.name.lower()
            text = para.text.strip()
            if not text:
                continue

            if style_name in heading_styles:
                if current_body:
                    sections.append((current_title, "\n\n".join(current_body)))
                current_title = text
                current_body = []
            else:
                current_body.append(text)

        if current_body:
            sections.append((current_title, "\n\n".join(current_body)))

        return sections

    # ------------------------------------------------------------------
    # PDF parsing
    # ------------------------------------------------------------------

    def _parse_pdf(self, path: Path) -> list[tuple[str, str]]:
        """
        Returns list of (section_title, section_body) tuples.

        Uses pdfplumber for text extraction. Heading detection is
        heuristic: lines that are ALL CAPS, or match common heading
        patterns (numbered sections like "1.2 Scope"), or are short
        (<= 60 chars) followed by a blank line.
        """
        try:
            import pdfplumber
        except ImportError as e:
            raise ChunkingError(
                str(path), "pdfplumber not installed. Run: pip install pdfplumber", e
            ) from e

        sections: list[tuple[str, str]] = []
        current_title = ""
        current_body: list[str] = []

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = text.split("\n")

                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        i += 1
                        continue

                    if self._is_heading(line, lines, i):
                        if current_body:
                            sections.append((
                                current_title,
                                " ".join(current_body),
                            ))
                        current_title = line
                        current_body = []
                    else:
                        current_body.append(line)
                    i += 1

        if current_body:
            sections.append((current_title, " ".join(current_body)))

        return sections if sections else [("", " ".join(current_body))]

    def _is_heading(self, line: str, lines: list[str], idx: int) -> bool:
        """
        Heuristic heading detection for PDF text.
        Deliberately conservative — false negatives (missed headings) are
        better than false positives (splitting mid-paragraph).
        """
        # All caps and short
        if line.isupper() and len(line) <= 80:
            return True
        # Numbered section: "1.", "1.2", "1.2.3", "A.", "Section 3"
        if re.match(r"^(\d+\.)+\s+\w", line) or re.match(r"^[A-Z]\.\s+\w", line):
            return True
        # Short line followed by blank line (common PDF heading pattern)
        if (
            len(line) <= 60
            and idx + 1 < len(lines)
            and not lines[idx + 1].strip()
        ):
            return True
        return False

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on double newlines or single newlines after sentence end."""
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]
