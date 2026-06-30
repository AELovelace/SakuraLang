"""Shared helpers for bounded conversation compaction."""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

import security
from prompts import SUMMARIZE_PROMPT, WINDOW_SIZE

_LOG = logging.getLogger(__name__)

_COMPACT_BATCH_MESSAGES = 8
_COMPACT_BATCH_CHARS = 12_000
_COMPACT_MESSAGE_CHARS = 1_500


def _stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (dict, list, tuple)):
        try:
            return json.dumps(content, ensure_ascii=True, default=str)
        except Exception:
            return str(content)
    return str(content)


def _message_line(message) -> str:
    content = re.sub(r"\s+", " ", _stringify_content(getattr(message, "content", ""))).strip()
    if len(content) > _COMPACT_MESSAGE_CHARS:
        content = content[:_COMPACT_MESSAGE_CHARS] + "..."
    return f"{message.__class__.__name__}: {content}"


def _build_batches(messages: list) -> list[list[tuple[object, str]]]:
    batches: list[list[tuple[object, str]]] = []
    current: list[tuple[object, str]] = []
    current_chars = 0

    for message in messages:
        line = _message_line(message)
        projected = current_chars + len(line) + 1
        if current and (
            len(current) >= _COMPACT_BATCH_MESSAGES or projected > _COMPACT_BATCH_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append((message, line))
        current_chars += len(line) + 1

    if current:
        batches.append(current)

    return batches


def _token_estimate(messages: list) -> int:
    total_chars = sum(
        len(_stringify_content(getattr(message, "content", "")))
        for message in messages
    )
    return total_chars // 4


def compact_messages(
    messages: list,
    existing_summary: str,
    llm,
    *,
    keep_recent: int = WINDOW_SIZE,
    progress: Callable[[int, int, int], None] | None = None,
) -> tuple[str, list, int, int]:
    """Summarize old messages in bounded batches and keep a recent tail intact."""
    if len(messages) <= keep_recent:
        return existing_summary, [], 0, 0

    to_compact = list(messages[:-keep_recent])
    batches = _build_batches(to_compact)
    summary = existing_summary or ""
    total_batches = len(batches)

    for index, batch in enumerate(batches, start=1):
        if progress is not None:
            progress(index, total_batches, len(batch))
        prefix = (
            "Previous summary:\n"
            f"{summary}\n\nNew messages to incorporate:\n"
            if summary
            else "New messages to incorporate:\n"
        )
        history_text = "\n".join(line for _, line in batch)
        response = llm.invoke([
            SystemMessage(content=SUMMARIZE_PROMPT),
            HumanMessage(content=prefix + history_text),
        ])
        new_summary = response.content if isinstance(response.content, str) else ""
        if not new_summary.strip():
            raise RuntimeError(f"Summarizer returned an empty result for batch {index}/{total_batches}")
        for warning in security.check_output(new_summary):
            _LOG.warning("[security] %s in conversation summary", warning)
        summary = security.sanitize_external(new_summary)

    return summary, to_compact, _token_estimate(to_compact), total_batches
