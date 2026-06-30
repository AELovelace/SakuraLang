# -*- coding: utf-8 -*-
"""Helpers for separating explicit model thinking tags from visible answers."""

from __future__ import annotations

import re

_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")
_THINK_BLOCK_RE = re.compile(
    r"(?is)<think(?:ing)?>.*?</think(?:ing)?>|<think(?:ing)?>.*\Z"
)


def strip_thinking_text(text: str) -> str:
    """Remove explicit <think>...</think> blocks from persisted/displayed answers."""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = re.sub(r"(?is)</?think(?:ing)?>", "", cleaned)
    return cleaned.strip()


def _find_first_token(text: str, tokens: tuple[str, ...]) -> tuple[int, str] | None:
    lowered = text.lower()
    best: tuple[int, str] | None = None
    for token in tokens:
        idx = lowered.find(token)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, token)
    return best


def _held_suffix_len(text: str, tokens: tuple[str, ...]) -> int:
    lowered = text.lower()
    max_len = 0
    for token in tokens:
        for n in range(1, min(len(token), len(lowered)) + 1):
            if lowered.endswith(token[:n]):
                max_len = max(max_len, n)
    return max_len


class ThinkingStreamSplitter:
    """Incrementally split streamed chunks into answer and explicit thinking text.

    This only handles model-emitted tags such as <think>...</think>. It does not
    expose any hidden provider-side reasoning that is not present in the stream.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_thinking = False

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        if not chunk:
            return []
        self._buf += chunk
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        while self._buf:
            if self._in_thinking:
                found = _find_first_token(self._buf, _CLOSE_TAGS)
                if found is not None:
                    idx, token = found
                    if idx:
                        events.append(("thinking", self._buf[:idx]))
                    self._buf = self._buf[idx + len(token):]
                    self._in_thinking = False
                    events.append(("thinking_end", ""))
                    continue

                hold = 0 if final else _held_suffix_len(self._buf, _CLOSE_TAGS)
                emit = self._buf if hold == 0 else self._buf[:-hold]
                self._buf = "" if hold == 0 else self._buf[-hold:]
                if emit:
                    events.append(("thinking", emit))
                break

            found = _find_first_token(self._buf, _OPEN_TAGS)
            if found is not None:
                idx, token = found
                if idx:
                    events.append(("answer", self._buf[:idx]))
                self._buf = self._buf[idx + len(token):]
                self._in_thinking = True
                continue

            hold = 0 if final else _held_suffix_len(self._buf, _OPEN_TAGS)
            emit = self._buf if hold == 0 else self._buf[:-hold]
            self._buf = "" if hold == 0 else self._buf[-hold:]
            if emit:
                events.append(("answer", emit))
            break

        return events
