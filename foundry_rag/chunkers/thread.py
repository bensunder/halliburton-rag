"""
thread.py — Thread-boundary chunker for Slack and Microsoft Teams.

Strategy:
  - One chunk = one thread (root message + all replies)
  - If thread exceeds max_tokens, split at reply boundaries (not mid-message)
  - Thread context (channel, participants, timestamp) injected as prefix
  - Never split a single message — a message is the atomic unit

Why thread boundary over fixed-size?
  Slack/Teams value lives in conversational context. A reply without its
  parent question is meaningless for retrieval. A question without the
  accepted answer is incomplete. Thread boundary preserves the Q&A pair
  that makes these messages worth indexing at all.

Input format:
  Accepts the Slack Conversations API export format and Teams Graph API
  format. Both are normalized to ThreadSource before chunking.

Dependencies:
  No external deps beyond stdlib — API clients handled by caller.
"""

from __future__ import annotations

from ._tokenizer import token_count as _token_count

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ._tokenizer import token_count as _token_count_fn

from .base import BaseChunker, ChunkingError
from .models import Chunk, DocType, SensitivityTier





@dataclass
class Message:
    """Single message within a thread."""
    user: str
    text: str
    timestamp: datetime
    is_root: bool = False           # True for the thread-starting message


@dataclass
class ThreadSource:
    """
    Input contract for ThreadChunker.

    Build this from Slack Conversations API or Teams Graph API responses.
    The chunker is API-agnostic — normalization is the caller's job.

    source_id should be stable and unique per thread:
      Slack:  f"{workspace_id}::{channel_id}::{thread_ts}"
      Teams:  f"{team_id}::{channel_id}::{message_id}"
    """
    source_id: str
    channel_name: str
    messages: list[Message]         # ordered oldest-first, root message at [0]
    platform: str = "slack"         # "slack" | "teams"
    workspace: str = ""
    language: str = "en"
    sensitivity_tier: SensitivityTier = SensitivityTier.INTERNAL
    acl_groups: list[str] = field(default_factory=list)
    extra_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError(f"ThreadSource {self.source_id} has no messages")


class ThreadChunker(BaseChunker[ThreadSource]):
    """
    Thread-boundary chunker for Slack and Teams.

    Usage:
        chunker = ThreadChunker(max_tokens=800)
        chunks = chunker.chunk(ThreadSource(
            source_id="acme::general::1701234567.123456",
            channel_name="general",
            platform="slack",
            messages=[
                Message(user="alice", text="Has anyone seen the AP variance report?",
                        timestamp=datetime(...), is_root=True),
                Message(user="bob", text="Yes, it's in SharePoint under Finance/Q4",
                        timestamp=datetime(...)),
            ],
            acl_groups=["halliburton-finance"],
        ))
    """

    def __init__(self, max_tokens: int = 800) -> None:
        self.max_tokens = max_tokens

    def _split(self, source: ThreadSource) -> list[Chunk]:
        if source.platform == "slack":
            doc_type = DocType.SLACK
        else:
            doc_type = DocType.TEAMS

        thread_text = self._render_thread(source)

        # Fast path: whole thread fits in one chunk
        if _token_count(thread_text) <= self.max_tokens:
            return [self._make_chunk(
                text=thread_text,
                doc_type=doc_type,
                source=source,
                chunk_index=0,
                participants=self._participants(source),
            )]

        # Slow path: split at reply boundaries
        return self._split_at_boundaries(source, doc_type)

    def _render_thread(self, source: ThreadSource) -> str:
        """
        Render full thread as a structured text block.

        Format:
            [Channel: #general | Platform: Slack | Participants: alice, bob]

            alice (2024-01-15 09:14): Has anyone seen the AP variance report?
            bob (2024-01-15 09:17): Yes, it's in SharePoint under Finance/Q4
        """
        participants = self._participants(source)
        thread_ts = source.messages[0].timestamp.strftime("%Y-%m-%d")

        header = (
            f"[Channel: #{source.channel_name} | "
            f"Platform: {source.platform.title()} | "
            f"Date: {thread_ts} | "
            f"Participants: {', '.join(participants)}]"
        )

        lines = [header, ""]
        for msg in source.messages:
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            role = "(root)" if msg.is_root else ""
            lines.append(f"{msg.user} {role}({ts}): {msg.text}".strip())

        return "\n".join(lines)

    def _render_message(self, msg: Message, source: ThreadSource) -> str:
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        role = "(root)" if msg.is_root else ""
        return f"{msg.user} {role}({ts}): {msg.text}".strip()

    def _split_at_boundaries(
        self, source: ThreadSource, doc_type: DocType
    ) -> list[Chunk]:
        """
        Split oversized thread at reply boundaries.
        Always include the root message in every chunk for context.
        """
        participants = self._participants(source)
        root_msg = source.messages[0]
        root_text = self._render_message(root_msg, source)

        header = (
            f"[Channel: #{source.channel_name} | "
            f"Platform: {source.platform.title()} | "
            f"Participants: {', '.join(participants)}]"
        )

        chunks: list[Chunk] = []
        chunk_index = 0
        current_lines: list[str] = [header, "", root_text]

        for msg in source.messages[1:]:    # skip root — already included
            msg_text = self._render_message(msg, source)
            candidate = "\n".join(current_lines + [msg_text])

            if _token_count(candidate) > self.max_tokens and len(current_lines) > 3:
                # Flush current window
                chunks.append(self._make_chunk(
                    text="\n".join(current_lines),
                    doc_type=doc_type,
                    source=source,
                    chunk_index=chunk_index,
                    participants=participants,
                ))
                chunk_index += 1
                # Start new window, always re-include root for context
                current_lines = [header, "", root_text, msg_text]
            else:
                current_lines.append(msg_text)

        # Flush remainder
        if len(current_lines) > 3:
            chunks.append(self._make_chunk(
                text="\n".join(current_lines),
                doc_type=doc_type,
                source=source,
                chunk_index=chunk_index,
                participants=participants,
            ))

        return chunks or [self._make_chunk(
            text=self._render_thread(source),
            doc_type=doc_type,
            source=source,
            chunk_index=0,
            participants=participants,
        )]

    def _make_chunk(
        self,
        text: str,
        doc_type: DocType,
        source: ThreadSource,
        chunk_index: int,
        participants: list[str],
    ) -> Chunk:
        root_ts = source.messages[0].timestamp
        return Chunk(
            text=text,
            doc_type=doc_type,
            source_id=source.source_id,
            parent_doc_id=source.source_id,
            chunk_index=chunk_index,
            chunk_total=0,
            author=source.messages[0].user,     # thread initiator
            created_at=root_ts,
            language=source.language,
            sensitivity_tier=source.sensitivity_tier,
            acl_groups=list(source.acl_groups),
            extra_metadata={
                "platform": source.platform,
                "channel_name": source.channel_name,
                "workspace": source.workspace,
                "participants": participants,
                "message_count": len(source.messages),
                **source.extra_metadata,
            },
        )

    @staticmethod
    def _participants(source: ThreadSource) -> list[str]:
        seen: dict[str, None] = {}
        for msg in source.messages:
            seen[msg.user] = None
        return list(seen.keys())
