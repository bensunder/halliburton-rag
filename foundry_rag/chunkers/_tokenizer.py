"""
_tokenizer.py — Tokenizer abstraction.

Production: uses tiktoken cl100k_base (same encoder as text-embedding-3-large).
Sandbox/offline: falls back to character-based estimation (1 token ≈ 4 chars).

Callers import token_count() from here — never import tiktoken directly.
This ensures the pipeline can run in air-gapped or restricted environments.
"""
from __future__ import annotations

def _make_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count(text: str) -> int:
            return len(enc.encode(text))
        return count
    except Exception:
        def count(text: str) -> int:
            return max(1, len(text) // 4)
        return count

token_count = _make_counter()
