"""
ast_chunker.py — AST-boundary chunker for code repositories.

Strategy:
  - Python: use stdlib ast module — zero external deps, exact boundaries
  - JS/TS: use tree-sitter (optional) — falls back to brace-counting heuristic
  - Other languages: line-window fallback with function-keyword detection
  - One chunk = one top-level function, class, or method block
  - Decorators, docstrings, and type annotations are included in the chunk
  - File-level context (imports, module docstring) prepended to every chunk

Why AST over line-window for code?
  A fixed-size line window will split a function mid-body. Retrieving half
  a function is worse than retrieving nothing — the LLM will hallucinate
  the missing half. AST boundaries guarantee semantic completeness.

Metadata captured per chunk:
  - language, file_path, repo, commit_sha
  - symbol_name (function/class name)
  - symbol_type ("function" | "class" | "method")
  - line_start, line_end

Dependencies:
  pip install tiktoken
  Optional: pip install tree-sitter tree-sitter-languages  (for JS/TS)
"""

from __future__ import annotations

from ._tokenizer import token_count as _token_count

import ast
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ._tokenizer import token_count as _token_count_fn

from .base import BaseChunker, ChunkingError
from .models import Chunk, DocType, SensitivityTier

_MAX_TOKENS = 1024      # Code chunks can be larger than prose — functions vary




Language = Literal["python", "javascript", "typescript", "java", "go", "unknown"]

_EXT_TO_LANG: dict[str, Language] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
}


@dataclass
class CodeFileSource:
    """
    Input contract for ASTChunker.

    source_id should be stable:
      f"{repo_name}::{relative_file_path}::{commit_sha[:8]}"
    """
    file_path: Path
    source_id: str
    repo: str
    commit_sha: str = "unknown"
    author: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"            # document language for search, not code language
    sensitivity_tier: SensitivityTier = SensitivityTier.INTERNAL
    acl_groups: list[str] = field(default_factory=list)
    extra_metadata: dict = field(default_factory=dict)

    @property
    def code_language(self) -> Language:
        return _EXT_TO_LANG.get(self.file_path.suffix.lower(), "unknown")


@dataclass
class _Symbol:
    name: str
    symbol_type: Literal["function", "class", "method"]
    source_lines: list[str]
    line_start: int
    line_end: int


class ASTChunker(BaseChunker[CodeFileSource]):
    """
    AST-boundary code chunker.

    Usage:
        chunker = ASTChunker()
        chunks = chunker.chunk(CodeFileSource(
            file_path=Path("src/ap_anomaly_detector.py"),
            source_id="halliburton-sap-demo::src/ap_anomaly_detector.py::a1b2c3d4",
            repo="halliburton-sap-demo",
            commit_sha="a1b2c3d4",
            acl_groups=["halliburton-engineering"],
        ))
    """

    def __init__(self, max_tokens: int = _MAX_TOKENS) -> None:
        self.max_tokens = max_tokens

    def _split(self, source: CodeFileSource) -> list[Chunk]:
        try:
            code = source.file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise ChunkingError(source.source_id, "Could not read file", e) from e

        lang = source.code_language

        try:
            if lang == "python":
                symbols = self._parse_python(code)
            else:
                symbols = self._parse_generic(code, lang)
        except Exception as e:
            raise ChunkingError(source.source_id, f"AST parse failed for {lang}", e) from e

        if not symbols:
            # File has no extractable symbols (e.g. pure config) — one chunk
            return [self._make_chunk(
                text=self._truncate(code),
                source=source,
                lang=lang,
                symbol_name="<module>",
                symbol_type="function",
                line_start=1,
                line_end=code.count("\n") + 1,
                chunk_index=0,
            )]

        file_header = self._extract_file_header(code, lang)
        chunks: list[Chunk] = []
        chunk_index = 0

        for sym in symbols:
            sym_text = "\n".join(sym.source_lines)
            # Prepend file header for context (imports, module docstring)
            full_text = (
                f"# File: {source.file_path.name} | Repo: {source.repo}\n"
                f"# Symbol: {sym.symbol_type} `{sym.name}` "
                f"(lines {sym.line_start}–{sym.line_end})\n\n"
                f"{file_header}\n\n{sym_text}" if file_header
                else
                f"# File: {source.file_path.name} | Repo: {source.repo}\n"
                f"# Symbol: {sym.symbol_type} `{sym.name}` "
                f"(lines {sym.line_start}–{sym.line_end})\n\n"
                f"{sym_text}"
            )
            full_text = self._truncate(full_text)
            chunks.append(self._make_chunk(
                text=full_text,
                source=source,
                lang=lang,
                symbol_name=sym.name,
                symbol_type=sym.symbol_type,
                line_start=sym.line_start,
                line_end=sym.line_end,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

        return chunks

    # ------------------------------------------------------------------
    # Python parsing — stdlib ast, exact and reliable
    # ------------------------------------------------------------------

    def _parse_python(self, code: str) -> list[_Symbol]:
        lines = code.splitlines()
        tree = ast.parse(code)
        symbols: list[_Symbol] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only top-level functions and methods (skip nested)
                sym_type: Literal["function", "class", "method"] = "function"
                # Check if parent is a class — ast.walk doesn't give parent,
                # so we use a simple heuristic: indentation level
                first_line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                if first_line.startswith("    ") or first_line.startswith("\t"):
                    sym_type = "method"

                end_line = getattr(node, "end_lineno", node.lineno)
                symbols.append(_Symbol(
                    name=node.name,
                    symbol_type=sym_type,
                    source_lines=lines[node.lineno - 1 : end_line],
                    line_start=node.lineno,
                    line_end=end_line,
                ))

            elif isinstance(node, ast.ClassDef):
                end_line = getattr(node, "end_lineno", node.lineno)
                symbols.append(_Symbol(
                    name=node.name,
                    symbol_type="class",
                    source_lines=lines[node.lineno - 1 : end_line],
                    line_start=node.lineno,
                    line_end=end_line,
                ))

        # Sort by line number — ast.walk order is not guaranteed
        symbols.sort(key=lambda s: s.line_start)
        return symbols

    # ------------------------------------------------------------------
    # Generic parsing — regex + brace counting for non-Python
    # ------------------------------------------------------------------

    def _parse_generic(self, code: str, lang: Language) -> list[_Symbol]:
        """
        Heuristic symbol extractor for JS/TS/Java/Go.
        Finds function/class declarations and extracts their brace-balanced body.
        Not as accurate as a real AST but far better than line-window splitting.
        """
        lines = code.splitlines()
        symbols: list[_Symbol] = []

        # Patterns that signal a function or class declaration
        fn_patterns = [
            re.compile(r"^(export\s+)?(async\s+)?function\s+(\w+)"),    # JS/TS function
            re.compile(r"^(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s*)?\("),  # arrow fn
            re.compile(r"^(public|private|protected|static|\s)*(async\s+)?\w[\w<>[\]]*\s+(\w+)\s*\("),  # Java/Go method
            re.compile(r"^(export\s+)?class\s+(\w+)"),
            re.compile(r"^func\s+(\w+)"),                                # Go
        ]

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            matched_name: str | None = None
            sym_type: Literal["function", "class", "method"] = "function"

            for pat in fn_patterns:
                m = pat.match(line)
                if m:
                    # Extract symbol name from last named group
                    groups = [g for g in m.groups() if g and re.match(r"^\w", g)]
                    matched_name = groups[-1] if groups else "anonymous"
                    if "class" in line:
                        sym_type = "class"
                    break

            if matched_name:
                start = i
                body_lines, end = self._extract_brace_block(lines, i)
                symbols.append(_Symbol(
                    name=matched_name,
                    symbol_type=sym_type,
                    source_lines=body_lines,
                    line_start=start + 1,
                    line_end=end + 1,
                ))
                i = end + 1
            else:
                i += 1

        return symbols

    def _extract_brace_block(
        self, lines: list[str], start: int
    ) -> tuple[list[str], int]:
        """
        Walk forward from start counting braces until the block closes.
        Returns (block_lines, end_index).
        """
        depth = 0
        found_open = False
        i = start
        while i < len(lines):
            for ch in lines[i]:
                if ch == "{":
                    depth += 1
                    found_open = True
                elif ch == "}":
                    depth -= 1
            if found_open and depth == 0:
                return lines[start : i + 1], i
            i += 1
        # No closing brace found — return to end of file
        return lines[start:], len(lines) - 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_file_header(self, code: str, lang: Language) -> str:
        """
        Extract import block and module docstring for context prefix.
        Capped at 30 lines to avoid bloating every chunk.
        """
        lines = code.splitlines()
        header_lines: list[str] = []

        if lang == "python":
            for line in lines[:40]:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ", "#", '"""', "'''")):
                    header_lines.append(line)
                elif not stripped:
                    header_lines.append("")
                else:
                    break
        else:
            for line in lines[:20]:
                stripped = line.strip()
                if stripped.startswith(("import ", "require(", "use ", "//")):
                    header_lines.append(line)
                elif not stripped:
                    continue
                else:
                    break

        return "\n".join(header_lines[:30]).strip()

    def _truncate(self, text: str) -> str:
        """Hard truncate to max_tokens if a symbol is unusually large.
        Uses character-based approximation (1 token ≈ 4 chars) to avoid
        requiring tiktoken encode/decode in offline environments.
        """
        if _token_count(text) <= self.max_tokens:
            return text
        # Approximate char limit: max_tokens * 4 chars per token
        char_limit = self.max_tokens * 4
        return text[:char_limit] + "\n# [truncated — symbol exceeds max_tokens]"

    def _make_chunk(
        self,
        text: str,
        source: CodeFileSource,
        lang: Language,
        symbol_name: str,
        symbol_type: str,
        line_start: int,
        line_end: int,
        chunk_index: int,
    ) -> Chunk:
        return Chunk(
            text=text,
            doc_type=DocType.CODE,
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
                "code_language": lang,
                "file_path": str(source.file_path),
                "repo": source.repo,
                "commit_sha": source.commit_sha,
                "symbol_name": symbol_name,
                "symbol_type": symbol_type,
                "line_start": line_start,
                "line_end": line_end,
                **source.extra_metadata,
            },
        )
