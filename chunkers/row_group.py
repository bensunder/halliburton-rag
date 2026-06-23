"""
row_group.py — Row-group NL template chunker for SAP CSV exports and SQL tables.

Strategy:
  - Group rows by a grouping key (vendor_id for SAP, primary key prefix for SQL)
  - Render each group as structured natural language using a field template
  - One chunk = one logical group (e.g., all invoices for vendor V-1042 in Jan 2024)
  - Include table/schema context as chunk prefix

Why NL template over raw CSV?
  Embedding "V-1042,INV-9901,2024-01-15,47500.00,USD,OPEN" yields poor
  semantic retrieval. Embedding "Vendor V-1042 (Schlumberger): Invoice INV-9901
  dated 2024-01-15, amount 47,500.00 USD, status OPEN" retrieves on queries
  like 'show me open Schlumberger invoices over $40k' with dramatically higher
  recall. The NL template is the semantic bridge between structured data and
  the embedding space.

SAP AP fields mapped:
  LIFNR  → vendor_id
  BELNR  → invoice_number
  BLDAT  → invoice_date
  DMBTR  → amount
  WAERS  → currency
  ZTERM  → payment_terms
  ZBSTT  → status

Dependencies:
  pip install pandas tiktoken
"""

from __future__ import annotations

from ._tokenizer import token_count as _token_count

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ._tokenizer import token_count as _token_count_fn

from .base import BaseChunker, ChunkingError
from .models import Chunk, DocType, SensitivityTier

_MAX_TOKENS = 512




# ------------------------------------------------------------------
# Field renderers — convert raw field values to readable NL phrases
# ------------------------------------------------------------------

def _fmt_amount(val: Any, row: dict) -> str:
    currency = row.get("WAERS") or row.get("currency", "USD")
    try:
        return f"{float(val):,.2f} {currency}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_date(val: Any, _row: dict) -> str:
    if not val:
        return "unknown date"
    s = str(val)
    # SAP date format: YYYYMMDD
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _fmt_status(val: Any, _row: dict) -> str:
    status_map = {
        "O": "Open",
        "P": "Paid",
        "C": "Cancelled",
        "B": "Blocked",
        "OPEN": "Open",
        "PAID": "Paid",
    }
    return status_map.get(str(val).upper(), str(val))


# SAP AP field template: (output_label, source_field_names, formatter)
SAP_AP_FIELD_TEMPLATE: list[tuple[str, list[str], Callable]] = [
    ("Vendor",          ["LIFNR", "vendor_id", "vendor"],       lambda v, r: str(v)),
    ("Invoice",         ["BELNR", "invoice_number", "inv_num"], lambda v, r: str(v)),
    ("Date",            ["BLDAT", "invoice_date", "date"],       _fmt_date),
    ("Amount",          ["DMBTR", "amount", "total"],            _fmt_amount),
    ("Payment terms",   ["ZTERM", "payment_terms", "terms"],     lambda v, r: str(v)),
    ("Status",          ["ZBSTT", "status", "state"],            _fmt_status),
    ("Company code",    ["BUKRS", "company_code"],               lambda v, r: str(v)),
    ("Fiscal year",     ["GJAHR", "fiscal_year", "year"],        lambda v, r: str(v)),
    ("Posting date",    ["BUDAT", "posting_date"],               _fmt_date),
    ("Reference",       ["XBLNR", "reference", "ref"],          lambda v, r: str(v)),
]


def _render_row(row: dict, template: list[tuple[str, list[str], Callable]]) -> str:
    """Render a single row dict as a natural language line."""
    parts: list[str] = []
    for label, field_names, fmt in template:
        for fn in field_names:
            val = row.get(fn)
            if val is not None and str(val).strip() not in ("", "nan", "None"):
                parts.append(f"{label}: {fmt(val, row)}")
                break
    return " | ".join(parts) if parts else ""


@dataclass
class RowGroupSource:
    """
    Input contract for RowGroupChunker.

    rows: list of dicts — one per record. Can come from:
      - pandas DataFrame.to_dict("records")
      - csv.DictReader
      - SQLAlchemy query results as mappings
      - Direct SAP IDoc/BAPI response normalization

    source_id: stable identifier for the data batch
      SAP:  f"sap::{company_code}::{fiscal_year}::{export_date}"
      SQL:  f"sql::{db_name}::{table_name}::{query_hash}"
    """
    source_id: str
    rows: list[dict[str, Any]]
    table_name: str
    source_system: str = "SAP"          # "SAP" | "SQL" | other
    grouping_key: str = "LIFNR"         # field to group rows by
    rows_per_chunk: int = 10            # max rows per chunk (tune per token budget)
    field_template: list[tuple[str, list[str], Callable]] = field(
        default_factory=lambda: SAP_AP_FIELD_TEMPLATE
    )
    author: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    sensitivity_tier: SensitivityTier = SensitivityTier.CONFIDENTIAL  # financial = confidential
    acl_groups: list[str] = field(default_factory=list)
    extra_metadata: dict = field(default_factory=dict)


class RowGroupChunker(BaseChunker[RowGroupSource]):
    """
    Row-group NL template chunker for SAP and SQL.

    Usage — SAP AP invoices:
        import pandas as pd
        df = pd.read_csv("sap_ap_export_2024Q1.csv")
        chunker = RowGroupChunker()
        chunks = chunker.chunk(RowGroupSource(
            source_id="sap::1000::2024::2024-03-31",
            rows=df.to_dict("records"),
            table_name="AP_INVOICES_Q1_2024",
            grouping_key="LIFNR",
            rows_per_chunk=10,
            acl_groups=["halliburton-finance", "halliburton-ap-team"],
        ))

    Usage — SQL table:
        from sqlalchemy import create_engine, text
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(text("SELECT * FROM vendors LIMIT 5000"))]
        chunks = chunker.chunk(RowGroupSource(
            source_id="sql::erp::vendors::abc123",
            rows=rows,
            table_name="vendors",
            source_system="SQL",
            grouping_key="vendor_id",
        ))
    """

    def _split(self, source: RowGroupSource) -> list[Chunk]:
        if not source.rows:
            raise ChunkingError(source.source_id, "No rows to chunk")

        doc_type = DocType.SAP if source.source_system == "SAP" else DocType.SQL

        # Group rows by grouping_key
        groups: dict[str, list[dict]] = {}
        ungrouped: list[dict] = []

        for row in source.rows:
            key_val = row.get(source.grouping_key)
            if key_val is not None and str(key_val).strip() not in ("", "nan"):
                key_str = str(key_val).strip()
                groups.setdefault(key_str, []).append(row)
            else:
                ungrouped.append(row)

        chunks: list[Chunk] = []
        chunk_index = 0

        # Chunked groups
        for group_key, group_rows in groups.items():
            for batch_start in range(0, len(group_rows), source.rows_per_chunk):
                batch = group_rows[batch_start : batch_start + source.rows_per_chunk]
                text = self._render_group(
                    group_key=group_key,
                    rows=batch,
                    source=source,
                    batch_offset=batch_start,
                )
                # Safety: if rendered text is still too large, truncate rows
                while _token_count(text) > _MAX_TOKENS and len(batch) > 1:
                    batch = batch[:-1]
                    text = self._render_group(group_key, batch, source, batch_start)

                chunks.append(self._make_chunk(
                    text=text,
                    doc_type=doc_type,
                    source=source,
                    chunk_index=chunk_index,
                    group_key=group_key,
                    row_count=len(batch),
                ))
                chunk_index += 1

        # Ungrouped rows (no grouping key) — batch by rows_per_chunk
        for batch_start in range(0, len(ungrouped), source.rows_per_chunk):
            batch = ungrouped[batch_start : batch_start + source.rows_per_chunk]
            text = self._render_group(
                group_key="(ungrouped)",
                rows=batch,
                source=source,
                batch_offset=batch_start,
            )
            chunks.append(self._make_chunk(
                text=text,
                doc_type=doc_type,
                source=source,
                chunk_index=chunk_index,
                group_key="(ungrouped)",
                row_count=len(batch),
            ))
            chunk_index += 1

        return chunks

    def _render_group(
        self,
        group_key: str,
        rows: list[dict],
        source: RowGroupSource,
        batch_offset: int,
    ) -> str:
        """
        Render a group of rows as structured NL.

        Example output:
            Source: SAP | Table: AP_INVOICES_Q1_2024 | Group: Vendor V-1042
            Records 1–10 of 23

            Record 1: Vendor: V-1042 | Invoice: INV-9901 | Date: 2024-01-15 | Amount: 47,500.00 USD | Status: Open
            Record 2: Vendor: V-1042 | Invoice: INV-9902 | Date: 2024-01-22 | Amount: 12,300.00 USD | Status: Paid
            ...
        """
        grouping_label = source.grouping_key.upper()
        lines = [
            f"Source: {source.source_system} | "
            f"Table: {source.table_name} | "
            f"Group: {grouping_label} {group_key}",
            f"Records {batch_offset + 1}–{batch_offset + len(rows)}",
            "",
        ]
        for i, row in enumerate(rows, start=batch_offset + 1):
            rendered = _render_row(row, source.field_template)
            if rendered:
                lines.append(f"Record {i}: {rendered}")

        return "\n".join(lines)

    def _make_chunk(
        self,
        text: str,
        doc_type: DocType,
        source: RowGroupSource,
        chunk_index: int,
        group_key: str,
        row_count: int,
    ) -> Chunk:
        return Chunk(
            text=text,
            doc_type=doc_type,
            source_id=source.source_id,
            parent_doc_id=source.source_id,
            chunk_index=chunk_index,
            chunk_total=0,
            author=source.author,
            created_at=source.created_at,
            language=source.language,
            sensitivity_tier=source.sensitivity_tier,
            acl_groups=list(source.acl_groups),
            extra_metadata={
                "source_system": source.source_system,
                "table_name": source.table_name,
                "grouping_key": source.grouping_key,
                "group_key_value": group_key,
                "row_count": row_count,
                **source.extra_metadata,
            },
        )
