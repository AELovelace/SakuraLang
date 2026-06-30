# -*- coding: utf-8 -*-
"""Background workers for streaming LangGraph responses and context compaction.

Both workers emit events as Python tuples via a single pyqtSignal(object).
The tuple format mirrors the TUI's queue protocol so MainWindow can share
the same _handle_event() dispatch logic.

Event kinds emitted:
    ("plog", msg)                     pipeline log line
    ("classify", note)                classifier routing note
    ("rag", note, ctx)                RAG context retrieved
    ("tool", name, content)           tool output message
    ("thinking_note", text)           short routing/decision transparency note
    ("thinking_start", text)          first explicit <think> chunk
    ("thinking_append", text)         subsequent explicit <think> chunk
    ("thinking_end",)                 explicit thinking block closed
    ("ai_start", text)                first streamed AI chunk
    ("ai_append", text)               subsequent AI chunk
    ("ai_next",)                      reset AI segment index
    ("ai_strip_tool_calls",)          strip inline tool XML from AI bubble
    ("ai_strip_context_override",)    strip context-override AI bubble
    ("token_usage", dict)             {input_tokens, output_tokens, total_tokens}
    ("token_usage_fallback", tid)     ask main thread to estimate tokens
    ("compact_done", freed_int, summary_text)  context compacted
    ("done",)                         stream finished
    ("error", msg)                    stream failed
    ("tool_approval_needed", name, args_dict)  approval gate (from approval.py)
    ("title_done", thread_id, title)  chat title generated
"""
import threading

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from langchain_core.messages import (
    HumanMessage, AIMessageChunk, RemoveMessage, SystemMessage,
)

import approval
from compaction import compact_messages
from settings import SETTINGS
from tools import _clip_display_text
from llm import get_llm
from prompts import CHAT_TITLE_PROMPT, WINDOW_SIZE
from thinking_stream import ThinkingStreamSplitter, strip_thinking_text

TOOL_DISPLAY_MAX = 4000
RAG_DISPLAY_MAX = 6000


class _SignalQueue:
    """Thin wrapper so approval.py can call queue.put() while we emit a Qt signal."""
    def __init__(self, callback):
        self._cb = callback

    def put(self, item):
        self._cb(item)


def _trunc(text: str, limit: int = 120) -> str:
    return text[:limit] + "…" if len(text) > limit else text


class StreamWorker(QObject):
    """Runs graph.stream() on a QThread and emits each event tuple as a signal."""

    event_ready = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, graph, thread_id: str, text: str, cancel_event: threading.Event):
        super().__init__()
        self.graph = graph
        self.thread_id = thread_id
        self.text = text
        self._cancel = cancel_event

    def _emit(self, ev):
        self.event_ready.emit(ev)

    @pyqtSlot()
    def run(self):
        try:
            approval.set_notify_queue(_SignalQueue(self._emit))
            self._emit(("plog", "stream: started"))
            self._emit(("plog", f"request: {_trunc(self.text, 120)}"))
            pending_ai = False
            pending_thinking = False
            first_chunk = False
            splitter = ThinkingStreamSplitter()
            self._emit(("plog", "graph: opening stream"))
            for item in self.graph.stream(
                {"messages": [HumanMessage(content=self.text)]},
                config={"configurable": {"thread_id": self.thread_id}},
                stream_mode=["updates", "messages"],
            ):
                if self._cancel.is_set():
                    break
                mode, data = item

                if mode == "updates":
                    for node_name, update in data.items():
                        self._emit(("plog", f"node: {node_name}"))

                        if node_name == "classify" and update.get("routing_note"):
                            self._emit(("classify", update["routing_note"]))
                            self._emit(("plog", f"classify: {_trunc(update['routing_note'], 80)}"))
                            self._emit(("thinking_note", f"Routing decision: {update['routing_note']}"))

                        elif node_name == "web_research":
                            query = update.get("web_query", "")
                            result_count = len(update.get("web_results", []) or [])
                            note = update.get("routing_note", "")
                            if query:
                                self._emit(("plog", f"web: query={_trunc(query, 90)}"))
                                self._emit(("thinking_note", f"Using web research query: {query}"))
                            self._emit(("plog", f"web: results={result_count}"))
                            self._emit(("thinking_note", f"Web research returned {result_count} result(s)."))
                            if note:
                                idx = note.rfind("[Web")
                                self._emit(("plog", f"web: {note[idx:] if idx != -1 else note}"))

                        elif node_name == "rag":
                            note = update.get("routing_note", "")
                            ctx = update.get("rag_context", "")
                            idx = note.rfind("[RAG")
                            rag_note = note[idx:] if idx != -1 else "[RAG]"
                            clipped_ctx = _clip_display_text(ctx, RAG_DISPLAY_MAX, "RAG context") if ctx else ""
                            self._emit(("rag", rag_note, clipped_ctx))
                            self._emit(("plog", f"rag: context chars={len(ctx)}"))
                            self._emit(("thinking_note", f"Checked local documents: {rag_note}"))

                        elif node_name in ("respond", "clarify"):
                            self._emit(("plog", f"{node_name}: model returned update"))
                            for msg in update.get("messages", []):
                                content = getattr(msg, "content", "") or ""
                                tool_calls = getattr(msg, "tool_calls", None) or []
                                if tool_calls:
                                    names = ", ".join(tc.get("name", "tool") for tc in tool_calls)
                                    self._emit(("plog", f"{node_name}: tool_calls={names}"))
                                    self._emit(("thinking_note", f"Decided to call tool(s): {names}"))
                                elif content:
                                    self._emit(("plog", f"{node_name}: text chars={len(str(content))}"))
                                usage = getattr(msg, "usage_metadata", None)
                                if not usage:
                                    rm = getattr(msg, "response_metadata", {}) or {}
                                    tu = rm.get("token_usage", {})
                                    if tu:
                                        usage = {
                                            "input_tokens": tu.get("prompt_tokens", 0),
                                            "output_tokens": tu.get("completion_tokens", 0),
                                            "total_tokens": tu.get("total_tokens", 0),
                                        }
                                if usage:
                                    self._emit(("token_usage", dict(usage)))

                        elif node_name == "tools":
                            msgs = update.get("messages", [])
                            self._emit(("plog", f"tools: received {len(msgs)} message(s)"))
                            for msg in msgs:
                                name = getattr(msg, "name", "tool")
                                raw = getattr(msg, "content", "")
                                clipped = _clip_display_text(raw, TOOL_DISPLAY_MAX, f"tool output: {name}")
                                self._emit(("tool", name, clipped))
                                self._emit(("thinking_note", f"Tool finished: {name}"))
                            self._emit(("ai_next",))
                            pending_ai = False

                        elif node_name == "dispatch_context_override":
                            self._emit(("plog", "dispatch_context_override: rerouting request"))
                            mode_name = update.get("context_override_mode", "context")
                            reason = update.get("context_override_reason", "").strip()
                            detail = f" ({reason})" if reason else ""
                            self._emit(("thinking_note", f"Need {mode_name} context before answering{detail}."))
                            self._emit(("ai_strip_context_override",))
                            self._emit(("ai_next",))
                            pending_ai = False

                        elif node_name == "dispatch_text_tools":
                            msgs = update.get("messages", [])
                            self._emit(("plog", f"dispatch_text_tools: received {len(msgs)} message(s)"))
                            self._emit(("thinking_note", f"Parsed {len(msgs)} text tool result(s)."))
                            self._emit(("ai_strip_tool_calls",))
                            for msg in msgs:
                                name = getattr(msg, "name", "tool")
                                raw = getattr(msg, "content", "")
                                clipped = _clip_display_text(raw, TOOL_DISPLAY_MAX, f"tool output: {name}")
                                self._emit(("tool", name, clipped))
                                self._emit(("thinking_note", f"Tool finished: {name}"))
                            self._emit(("ai_next",))
                            pending_ai = False

                elif mode == "messages":
                    msg_chunk, metadata = data
                    if not isinstance(msg_chunk, AIMessageChunk):
                        continue
                    content = getattr(msg_chunk, "content", "") or ""
                    if metadata.get("langgraph_node", "") in ("respond", "clarify"):
                        chunk_usage = getattr(msg_chunk, "usage_metadata", None)
                        if chunk_usage and (
                            chunk_usage.get("input_tokens") or chunk_usage.get("total_tokens")
                        ):
                            self._emit(("token_usage", dict(chunk_usage)))
                        if content:
                            if not first_chunk:
                                first_chunk = True
                                node = metadata.get("langgraph_node", "unknown")
                                self._emit(("plog", f"ai: first token from {node}"))
                            for part_kind, part_text in splitter.feed(content):
                                if part_kind == "thinking":
                                    kind = "thinking_start" if not pending_thinking else "thinking_append"
                                    self._emit((kind, part_text))
                                    pending_thinking = True
                                elif part_kind == "thinking_end":
                                    self._emit(("thinking_end",))
                                    pending_thinking = False
                                elif part_kind == "answer":
                                    kind = "ai_start" if not pending_ai else "ai_append"
                                    self._emit((kind, part_text))
                                    pending_ai = True

            for part_kind, part_text in splitter.finish():
                if part_kind == "thinking":
                    kind = "thinking_start" if not pending_thinking else "thinking_append"
                    self._emit((kind, part_text))
                    pending_thinking = True
                elif part_kind == "thinking_end":
                    self._emit(("thinking_end",))
                    pending_thinking = False
                elif part_kind == "answer":
                    kind = "ai_start" if not pending_ai else "ai_append"
                    self._emit((kind, part_text))
                    pending_ai = True

        except Exception as exc:
            self._emit(("error", str(exc)))
        else:
            self._emit(("plog", "stream: graph finished"))
            self._emit(("token_usage_fallback", self.thread_id))
            self._emit(("done",))
        finally:
            self.finished.emit()


class CompactWorker(QObject):
    """Summarises and removes old messages from a LangGraph thread."""

    event_ready = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, graph, thread_id: str):
        super().__init__()
        self.graph = graph
        self.thread_id = thread_id

    def _emit(self, ev):
        self.event_ready.emit(ev)

    @pyqtSlot()
    def run(self):
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = self.graph.get_state(config=config)
            messages = list(state.values.get("messages", []))
            if len(messages) <= WINDOW_SIZE:
                self._emit(("compact_done", 0, ""))
                return

            existing = state.values.get("summary", "")
            cfg = SETTINGS["agent"]
            llm = get_llm(cfg["address"], streaming=False, timeout=None)
            self._emit(("plog", f"compact: keeping latest {WINDOW_SIZE} messages"))
            new_summary, removed_messages, freed, batch_count = compact_messages(
                messages,
                existing,
                llm,
                progress=lambda i, total, count: self._emit((
                    "plog",
                    f"compact: summarizing batch {i}/{total} ({count} messages)",
                )),
            )
            if not removed_messages:
                self._emit(("compact_done", 0, ""))
                return

            self.graph.update_state(
                config=config,
                values={
                    "summary": new_summary,
                    "messages": [RemoveMessage(id=m.id) for m in removed_messages],
                },
            )
            self._emit(("plog", f"compact: applied {batch_count} summary batch(es)"))
            self._emit(("compact_done", freed, new_summary))
        except Exception as exc:
            self._emit(("error", f"Compact failed: {exc}"))
        finally:
            self.finished.emit()


class TitleWorker(QObject):
    """Generates a chat title for a thread and emits ("title_done", thread_id, title)."""

    event_ready = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, graph, thread_id: str):
        super().__init__()
        self.graph = graph
        self.thread_id = thread_id

    @pyqtSlot()
    def run(self):
        import re
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = self.graph.get_state(config)
            messages = list(state.values.get("messages", []))
        except Exception:
            self.finished.emit()
            return

        try:
            from prompts import HEAT_TAG_RE, REWARD_TAG_RE
            lines = []
            for msg in messages[-3:]:
                content = getattr(msg, "content", "") or ""
                if not isinstance(content, str):
                    continue
                content = strip_thinking_text(REWARD_TAG_RE.sub("", HEAT_TAG_RE.sub("", content)))
                snippet = re.sub(r"\s+", " ", content).strip()[:280]
                if not snippet:
                    continue
                from langchain_core.messages import HumanMessage as HM, AIMessage as AM
                role = "User" if isinstance(msg, HM) else "Assistant" if isinstance(msg, AM) else msg.__class__.__name__
                lines.append(f"{role}: {snippet}")
            source = "\n".join(lines)
            if not source.strip():
                return

            tcfg = SETTINGS.get("titler", {})
            address = tcfg.get("address", "").strip()

            prompt = tcfg.get("system_prompt", "").strip() or CHAT_TITLE_PROMPT
            raw = ""
            if address:
                try:
                    response = get_llm(address, streaming=False, timeout=45).invoke([
                        SystemMessage(content=prompt),
                        HumanMessage(content=source),
                    ])
                    raw = response.content if isinstance(response.content, str) else ""
                except Exception:
                    pass

            def _clean(s: str) -> str:
                s = REWARD_TAG_RE.sub("", HEAT_TAG_RE.sub("", re.sub(r"\s+", " ", s or ""))).strip().strip("\"'`")
                s = s.splitlines()[0].strip() if s else ""
                s = re.sub(r"[.!?:;,]+$", "", s)
                if len(s) > 60:
                    clipped = s[:60].rsplit(" ", 1)[0].strip()
                    s = clipped or s[:60].strip()
                return s

            cleaned = _clean(raw)
            if not cleaned:
                human_lines = [l.split(": ", 1)[1] for l in source.splitlines() if l.startswith("User:")]
                if human_lines:
                    cleaned = _clean(human_lines[0])
            if not cleaned:
                return
            self.event_ready.emit(("title_done", self.thread_id, cleaned))
        finally:
            self.finished.emit()
