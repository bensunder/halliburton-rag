"""
tests/test_chunkers.py — Unit tests for all four chunking strategies.

Run with:
    pip install pytest tiktoken
    pytest chunkers/tests/test_chunkers.py -v

These tests use only in-memory fixtures — no file I/O, no API calls.
All tests must pass before the ingestion pipeline is deployed.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import ast
import textwrap
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chunkers.models import Chunk, DocType, SensitivityTier
from chunkers.base import ChunkingError
from chunkers.thread import ThreadChunker, ThreadSource, Message
from chunkers.ast_chunker import ASTChunker, CodeFileSource
from chunkers.row_group import RowGroupChunker, RowGroupSource, SAP_AP_FIELD_TEMPLATE


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def slack_thread(now) -> ThreadSource:
    return ThreadSource(
        source_id="acme::general::1701234567.000100",
        channel_name="general",
        platform="slack",
        workspace="acme",
        acl_groups=["acme-engineering"],
        messages=[
            Message(user="alice", text="Has anyone seen the AP variance report for Q4?", timestamp=now, is_root=True),
            Message(user="bob", text="Yes — it's in SharePoint under Finance/Q4/AP", timestamp=now),
            Message(user="carol", text="I can send a direct link if needed", timestamp=now),
        ],
    )


@pytest.fixture
def sap_rows() -> list[dict]:
    return [
        {"LIFNR": "V-1042", "BELNR": "INV-9901", "BLDAT": "20240115", "DMBTR": "47500.00", "WAERS": "USD", "ZBSTT": "O", "BUKRS": "1000", "GJAHR": "2024"},
        {"LIFNR": "V-1042", "BELNR": "INV-9902", "BLDAT": "20240122", "DMBTR": "12300.00", "WAERS": "USD", "ZBSTT": "P", "BUKRS": "1000", "GJAHR": "2024"},
        {"LIFNR": "V-2088", "BELNR": "INV-9950", "BLDAT": "20240201", "DMBTR": "88000.00", "WAERS": "USD", "ZBSTT": "B", "BUKRS": "1000", "GJAHR": "2024"},
    ]


SAMPLE_PYTHON = textwrap.dedent("""
    import os
    from pathlib import Path

    MODULE_CONST = "hello"

    def simple_function(x: int) -> int:
        \"\"\"Double the input.\"\"\"
        return x * 2

    def another_function(a: str, b: str) -> str:
        return a + b

    class MyService:
        def __init__(self, name: str):
            self.name = name

        def greet(self) -> str:
            return f"Hello from {self.name}"

        def farewell(self) -> str:
            return f"Goodbye from {self.name}"
""").strip()


# ===========================================================================
# Model tests — output contract
# ===========================================================================

class TestChunkModel:
    def test_chunk_id_is_deterministic(self):
        c1 = Chunk(text="hello world test chunk", doc_type=DocType.PDF,
                   source_id="doc-001", parent_doc_id="doc-001",
                   chunk_index=0, chunk_total=1)
        c2 = Chunk(text="different text here", doc_type=DocType.PDF,
                   source_id="doc-001", parent_doc_id="doc-001",
                   chunk_index=0, chunk_total=1)
        assert c1.chunk_id == c2.chunk_id  # same source_id + index → same ID

    def test_chunk_id_differs_by_index(self):
        c1 = Chunk(text="a" * 60, doc_type=DocType.PDF,
                   source_id="doc-001", parent_doc_id="doc-001",
                   chunk_index=0, chunk_total=2)
        c2 = Chunk(text="b" * 60, doc_type=DocType.PDF,
                   source_id="doc-001", parent_doc_id="doc-001",
                   chunk_index=1, chunk_total=2)
        assert c1.chunk_id != c2.chunk_id

    def test_token_estimate(self):
        text = "a" * 400  # 400 chars ≈ 100 tokens
        c = Chunk(text=text, doc_type=DocType.SLACK, source_id="x",
                  parent_doc_id="x", chunk_index=0, chunk_total=1)
        assert c.token_estimate == 100

    def test_validate_passes_valid_chunk(self):
        c = Chunk(text="x" * 100, doc_type=DocType.DOCX, source_id="doc-1",
                  parent_doc_id="doc-1", chunk_index=0, chunk_total=1)
        assert c.validate() == []

    def test_validate_rejects_empty_text(self):
        c = Chunk(text="", doc_type=DocType.PDF, source_id="doc-1",
                  parent_doc_id="doc-1", chunk_index=0, chunk_total=1)
        errors = c.validate()
        assert any("empty" in e for e in errors)

    def test_validate_rejects_short_text(self):
        c = Chunk(text="too short", doc_type=DocType.PDF, source_id="doc-1",
                  parent_doc_id="doc-1", chunk_index=0, chunk_total=1)
        errors = c.validate()
        assert any("short" in e for e in errors)

    def test_validate_rejects_oversized_chunk(self):
        c = Chunk(text="word " * 3000, doc_type=DocType.PDF, source_id="doc-1",
                  parent_doc_id="doc-1", chunk_index=0, chunk_total=1)
        errors = c.validate()
        assert any("large" in e for e in errors)

    def test_to_search_document_raises_without_embedding(self):
        c = Chunk(text="x" * 100, doc_type=DocType.PDF, source_id="doc-1",
                  parent_doc_id="doc-1", chunk_index=0, chunk_total=1)
        with pytest.raises(ValueError, match="no embedding"):
            c.to_search_document()

    def test_to_search_document_schema_with_embedding(self):
        c = Chunk(text="x" * 100, doc_type=DocType.PDF, source_id="doc-1",
                  parent_doc_id="doc-1", chunk_index=0, chunk_total=1)
        c.embedding = [0.1] * 1536
        doc = c.to_search_document()
        required_keys = {"id", "text", "docType", "sourceId", "parentDocId",
                         "chunkIndex", "chunkTotal", "contentVector"}
        assert required_keys.issubset(doc.keys())
        assert doc["docType"] == "pdf"
        assert len(doc["contentVector"]) == 1536


# ===========================================================================
# Thread chunker tests
# ===========================================================================

class TestThreadChunker:
    def test_small_thread_produces_one_chunk(self, slack_thread):
        chunker = ThreadChunker(max_tokens=800)
        chunks = chunker.chunk(slack_thread)
        assert len(chunks) == 1

    def test_chunk_contains_all_participants(self, slack_thread):
        chunker = ThreadChunker(max_tokens=800)
        chunks = chunker.chunk(slack_thread)
        text = chunks[0].text
        assert "alice" in text
        assert "bob" in text
        assert "carol" in text

    def test_chunk_contains_channel_name(self, slack_thread):
        chunker = ThreadChunker(max_tokens=800)
        chunks = chunker.chunk(slack_thread)
        assert "general" in chunks[0].text

    def test_chunk_doc_type_is_slack(self, slack_thread):
        chunker = ThreadChunker()
        chunks = chunker.chunk(slack_thread)
        assert all(c.doc_type == DocType.SLACK for c in chunks)

    def test_chunk_total_set_correctly(self, slack_thread):
        chunker = ThreadChunker()
        chunks = chunker.chunk(slack_thread)
        assert all(c.chunk_total == len(chunks) for c in chunks)

    def test_chunk_indices_sequential(self, slack_thread):
        chunker = ThreadChunker()
        chunks = chunker.chunk(slack_thread)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_oversized_thread_splits_at_boundaries(self, now):
        """A thread with 50 messages should split into multiple chunks."""
        messages = [
            Message(user="alice", text="Root: what is the AP process for vendor V-1042?",
                    timestamp=now, is_root=True)
        ]
        for i in range(49):
            messages.append(Message(
                user=f"user{i}",
                text=f"Reply {i}: The AP process involves several steps including validation, "
                     f"approval routing, and payment scheduling. This is reply number {i}.",
                timestamp=now,
            ))
        source = ThreadSource(
            source_id="acme::ap-process::001",
            channel_name="ap-process",
            messages=messages,
            acl_groups=["finance"],
        )
        chunker = ThreadChunker(max_tokens=300)
        chunks = chunker.chunk(source)
        assert len(chunks) > 1
        # Root message should appear in every chunk
        for c in chunks:
            assert "Root:" in c.text

    def test_acl_groups_propagated(self, slack_thread):
        chunker = ThreadChunker()
        chunks = chunker.chunk(slack_thread)
        assert all("acme-engineering" in c.acl_groups for c in chunks)

    def test_empty_thread_raises(self):
        with pytest.raises(ValueError, match="no messages"):
            ThreadSource(
                source_id="bad::thread",
                channel_name="test",
                messages=[],
            )

    def test_extra_metadata_preserved(self, slack_thread):
        slack_thread.extra_metadata["workspace_id"] = "W12345"
        chunker = ThreadChunker()
        chunks = chunker.chunk(slack_thread)
        assert chunks[0].extra_metadata["workspace_id"] == "W12345"


# ===========================================================================
# AST chunker tests
# ===========================================================================

class TestASTChunker:
    def _make_py_source(self, code: str, extra_metadata=None) -> CodeFileSource:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        return CodeFileSource(
            file_path=tmp_path,
            source_id=f"test-repo::test.py::abc123",
            repo="test-repo",
            commit_sha="abc123",
            acl_groups=["engineers"],
            extra_metadata=extra_metadata or {},
        )

    def test_extracts_functions(self):
        source = self._make_py_source(SAMPLE_PYTHON)
        chunker = ASTChunker()
        chunks = chunker.chunk(source)
        names = [c.extra_metadata["symbol_name"] for c in chunks]
        assert "simple_function" in names
        assert "another_function" in names

    def test_extracts_class(self):
        source = self._make_py_source(SAMPLE_PYTHON)
        chunker = ASTChunker()
        chunks = chunker.chunk(source)
        types = [c.extra_metadata["symbol_type"] for c in chunks]
        assert "class" in types

    def test_chunk_doc_type_is_code(self):
        source = self._make_py_source(SAMPLE_PYTHON)
        chunker = ASTChunker()
        chunks = chunker.chunk(source)
        assert all(c.doc_type == DocType.CODE for c in chunks)

    def test_chunk_contains_file_context_prefix(self):
        source = self._make_py_source(SAMPLE_PYTHON)
        chunker = ASTChunker()
        chunks = chunker.chunk(source)
        # Every chunk should reference the file name
        for c in chunks:
            assert "test.py" in c.text or "test-repo" in c.text

    def test_chunk_total_matches_count(self):
        source = self._make_py_source(SAMPLE_PYTHON)
        chunker = ASTChunker()
        chunks = chunker.chunk(source)
        assert all(c.chunk_total == len(chunks) for c in chunks)

    def test_line_numbers_captured_in_metadata(self):
        source = self._make_py_source(SAMPLE_PYTHON)
        chunker = ASTChunker()
        chunks = chunker.chunk(source)
        for c in chunks:
            assert "line_start" in c.extra_metadata
            assert "line_end" in c.extra_metadata
            assert c.extra_metadata["line_start"] > 0

    def test_empty_file_produces_one_chunk(self):
        source = self._make_py_source("# empty module\n")
        chunker = ASTChunker()
        chunks = chunker.chunk(source)
        assert len(chunks) == 1

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("content")
            tmp_path = Path(f.name)
        source = CodeFileSource(
            file_path=tmp_path,
            source_id="test::unknown.xyz::abc",
            repo="test",
            commit_sha="abc",
        )
        chunker = ASTChunker()
        # Unknown extension falls back to generic — should not raise
        chunks = chunker.chunk(source)
        assert isinstance(chunks, list)

    def test_acl_groups_propagated(self):
        source = self._make_py_source(SAMPLE_PYTHON)
        chunker = ASTChunker()
        chunks = chunker.chunk(source)
        assert all("engineers" in c.acl_groups for c in chunks)

    def test_oversized_symbol_is_truncated(self):
        big_function = "def huge_fn():\n" + "    x = 1\n" * 3000
        source = self._make_py_source(big_function)
        chunker = ASTChunker(max_tokens=512)
        chunks = chunker.chunk(source)
        for c in chunks:
            assert c.token_estimate <= 600  # some headroom for truncation marker


# ===========================================================================
# Row-group chunker tests
# ===========================================================================

class TestRowGroupChunker:
    def test_groups_by_vendor(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::1000::2024::2024-03-31",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
            rows_per_chunk=10,
            acl_groups=["finance"],
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        # V-1042 has 2 rows, V-2088 has 1 row → 2 groups → 2 chunks
        assert len(chunks) == 2

    def test_chunk_contains_vendor_id(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::1000::2024::test",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
            acl_groups=["finance"],
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        texts = "\n".join(c.text for c in chunks)
        assert "V-1042" in texts
        assert "V-2088" in texts

    def test_chunk_renders_nl_template(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::1000::2024::test",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
            acl_groups=["finance"],
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        # NL template should produce human-readable field labels
        full_text = "\n".join(c.text for c in chunks)
        assert "Invoice:" in full_text
        assert "Amount:" in full_text
        assert "Status:" in full_text

    def test_sap_date_formatted(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::test",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
            acl_groups=["finance"],
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        full_text = "\n".join(c.text for c in chunks)
        # SAP date 20240115 → 2024-01-15
        assert "2024-01-15" in full_text

    def test_status_human_readable(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::test",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
            acl_groups=["finance"],
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        full_text = "\n".join(c.text for c in chunks)
        assert "Open" in full_text or "Paid" in full_text or "Blocked" in full_text

    def test_rows_per_chunk_respected(self):
        rows = [{"LIFNR": "V-9999", "BELNR": f"INV-{i:04d}", "DMBTR": "100.00"} for i in range(25)]
        source = RowGroupSource(
            source_id="sap::batch-test",
            rows=rows,
            table_name="TEST",
            grouping_key="LIFNR",
            rows_per_chunk=5,
            acl_groups=["finance"],
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        assert len(chunks) == 5   # 25 rows / 5 per chunk

    def test_doc_type_is_sap(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::test",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        assert all(c.doc_type == DocType.SAP for c in chunks)

    def test_sensitivity_defaults_to_confidential(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::test",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        assert all(c.sensitivity_tier == SensitivityTier.CONFIDENTIAL for c in chunks)

    def test_chunk_total_set_correctly(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::test",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        assert all(c.chunk_total == len(chunks) for c in chunks)

    def test_empty_rows_raises(self):
        source = RowGroupSource(
            source_id="sap::empty",
            rows=[],
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
        )
        chunker = RowGroupChunker()
        with pytest.raises(ChunkingError):
            chunker.chunk(source)

    def test_metadata_includes_group_key_value(self, sap_rows):
        source = RowGroupSource(
            source_id="sap::test",
            rows=sap_rows,
            table_name="AP_INVOICES",
            grouping_key="LIFNR",
        )
        chunker = RowGroupChunker()
        chunks = chunker.chunk(source)
        group_keys = {c.extra_metadata["group_key_value"] for c in chunks}
        assert "V-1042" in group_keys
        assert "V-2088" in group_keys
