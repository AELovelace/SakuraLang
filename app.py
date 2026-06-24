# -*- coding: utf-8 -*-
import curses
import json
import queue
import re
import sqlite3
import subprocess
import threading
import time
import urllib.request
from collections import deque

from langchain_core.messages import (
    HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage, RemoveMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver

from settings import (
    SETTINGS, DEFAULT_SETTINGS, save_settings,
    CHAT_INPUT_MIN_H, CHAT_INPUT_MAX_H,
    GETCH_TIMEOUT_MS, ESCAPE_SEQUENCE_TIMEOUT_MS, SHIFT_ENTER_ESCAPE_SEQUENCES,
    STREAM_STALL_TIMEOUT_SEC,
    CHAT_RENDER_MAX_CHARS, TOOL_DISPLAY_MAX_CHARS, RAG_DISPLAY_MAX_CHARS,
)
from llm import get_llm, rebuild_llms
from tools import (
    TOOL_MAP,
    _contains_text_tool_call, _contains_context_override,
    _read_clipboard, _truncate_log_text, _clip_display_text,
)
from prompts import (
    SUMMARIZE_PROMPT, WINDOW_SIZE, CHAT_TITLE_PROMPT,
    MONITOR_URL, LOGS_URL, REMOTE_MONITOR_URL, REMOTE_LOGS_URL,
    MENU, VIEW_HOME, VIEW_CHAT, VIEW_SETTINGS, VIEW_HELP,
    ROLE_PAIR, ROLE_PREFIX,
    F_AGENT_ADDR, F_AGENT_PROMPT, F_AGENT_CWD,
    F_RESEARCHER_ADDR, F_RESEARCHER_PROMPT,
    F_BRAVE_API_KEY, F_BRAVE_BASE_URL, F_BRAVE_COUNT,
    F_BRAVE_COUNTRY, F_BRAVE_SEARCH_LANG, F_BRAVE_SAFESEARCH,
    F_CLASSIFIER_ADDR, F_CLASSIFIER_PROMPT,
    F_TITLER_ADDR, F_TITLER_PROMPT,
    F_LOVENSE_TOKEN, F_LOVENSE_UID, F_LOVENSE_PORT, F_LOVENSE_HOST,
    F_LOVENSE_CERT, F_LOVENSE_KEY,
    NUM_FIELDS,
)
import approval
from graph import builder


class App:
    def __init__(self, stdscr, graph):
        self.stdscr    = stdscr
        self.graph     = graph
        self.view      = VIEW_HOME
        self.prev_view = VIEW_HOME
        self.thread_id = self._new_thread_id()
        self.history   = []
        self.input_buf    = ""
        self.input_cursor = 0
        self._input_goal_col: int | None = None
        self.thinking     = False
        self.chat_scroll  = 0
        self._monitor_data: dict | list = {}
        self._gpu_history: deque = deque(maxlen=20)  # 20 × 0.5s = 10s rolling window
        self._remote_monitor_data: dict | list = {}
        self._remote_gpu_history: deque = deque(maxlen=20)
        self._remote_log_line: str = ""
        self._log_lines: list[str] = []
        self._pipeline_log: list[str] = []   # rolling timestamped pipeline event log
        self._stop_event    = threading.Event()
        self._stream_queue  = queue.Queue()
        self._cancel_event  = threading.Event()
        self._ai_idx: int | None = None
        self._active_request_thread_id: str | None = None
        self._last_queue_event: float = 0.0
        self._cancelling: bool = False
        self._ignore_stream: bool = False
        self._last_token_usage: dict = {}
        # Settings editor state
        self.settings_focus   = 0
        self.settings_bufs    = self._bufs_from_settings()
        self.settings_cursors = [len(b) for b in self.settings_bufs]
        # Chat list modal state
        self._chats_modal_open = False
        self._chats_list: list[dict] = []   # {"thread_id": str, "count": int, "title": str}
        self._chats_cursor = 0
        # Select mode — nano-style text selector for chat history
        self._select_mode:   bool      = False
        self._select_cursor: int       = 0
        self._select_anchor: int | None = None
        self._select_lines:  list      = []   # rendered (pair, text) lines cached by _draw_chat
        self._approval_pending: dict | None = None   # set when agent mode needs tool approval
        self._db_path = "chat_history.db"
        self._ensure_chat_titles_table()

    def _bufs_from_settings(self) -> list[str]:
        lv = SETTINGS.get("lovense", {})
        brave = SETTINGS.get("brave", {})
        titler = SETTINGS.get("titler", {})
        return [
            SETTINGS["agent"]["address"],
            SETTINGS["agent"]["system_prompt"],
            SETTINGS["agent"].get("cwd", ""),
            SETTINGS["researcher"]["address"],
            SETTINGS["researcher"]["system_prompt"],
            brave.get("api_key", ""),
            brave.get("base_url", DEFAULT_SETTINGS["brave"]["base_url"]),
            brave.get("count", "5"),
            brave.get("country", "us"),
            brave.get("search_lang", "en"),
            brave.get("safesearch", "moderate"),
            SETTINGS["classifier"]["address"],
            SETTINGS["classifier"]["system_prompt"],
            titler.get("address", ""),
            titler.get("system_prompt", ""),
            # Lovense fields
            lv.get("token", ""),
            lv.get("uid", ""),
            lv.get("callback_port", "34569"),
            lv.get("callback_host", ""),
            lv.get("cert_file", ""),
            lv.get("key_file",  ""),
        ]

    def _new_thread_id(self) -> str:
        return f"chat_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"

    def _activate_new_chat(self, *, clear_input: bool = True) -> None:
        self.thread_id   = self._new_thread_id()
        self.history     = []
        self.chat_scroll = 0
        self._ai_idx     = None
        if clear_input:
            self.input_buf    = ""
            self.input_cursor = 0
        self._input_goal_col = None

    def _insert_input_text(self, text: str) -> None:
        if not text:
            return
        pos = self.input_cursor
        self.input_buf = self.input_buf[:pos] + text + self.input_buf[pos:]
        self.input_cursor = pos + len(text)
        self._input_goal_col = None

    def _move_input_vertically(self, direction: int, field_w: int) -> None:
        vlines = self._compute_visual_lines(self.input_buf, field_w)
        row, col = self._cursor_to_visual(vlines, self.input_cursor)
        target_col = col if self._input_goal_col is None else self._input_goal_col
        next_row = row + direction
        if 0 <= next_row < len(vlines):
            start, line = vlines[next_row]
            self.input_cursor = start + min(target_col, len(line))
            self._input_goal_col = target_col

    def _read_escape_sequence(self) -> str:
        seq = []
        self.stdscr.timeout(ESCAPE_SEQUENCE_TIMEOUT_MS)
        try:
            while len(seq) < 12:
                next_key = self.stdscr.getch()
                if next_key == curses.ERR:
                    break
                if 0 <= next_key <= 255:
                    seq.append(chr(next_key))
                else:
                    break
        finally:
            self.stdscr.timeout(GETCH_TIMEOUT_MS)
        return "".join(seq)

    def _handle_chat_escape_sequence(self) -> bool:
        seq = self._read_escape_sequence()
        if seq in SHIFT_ENTER_ESCAPE_SEQUENCES:
            self._insert_input_text("\n")
            return True
        if seq in ("a", "A"):  # Alt+A → enter select mode
            self._enter_select_mode()
            return True
        return False

    def _chat_input_height(self, field_w: int, screen_h: int) -> int:
        visible_lines = len(self._compute_visual_lines(self.input_buf, field_w))
        desired = max(CHAT_INPUT_MIN_H, min(CHAT_INPUT_MAX_H, visible_lines))
        return max(CHAT_INPUT_MIN_H, min(desired, max(CHAT_INPUT_MIN_H, screen_h - 3)))

    def _ensure_chat_titles_table(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_titles (
                    thread_id   TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _get_chat_title(self, thread_id: str) -> str | None:
        try:
            conn = sqlite3.connect(self._db_path)
            cur = conn.cursor()
            cur.execute("SELECT title FROM chat_titles WHERE thread_id = ?", (thread_id,))
            row = cur.fetchone()
            conn.close()
            return str(row[0]).strip() if row and row[0] else None
        except Exception:
            return None

    def _save_chat_title(self, thread_id: str, title: str) -> None:
        cleaned = self._normalize_chat_title(title)
        if not cleaned:
            return
        try:
            conn = sqlite3.connect(self._db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO chat_titles (thread_id, title, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = CURRENT_TIMESTAMP
            """, (thread_id, cleaned))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _fallback_chat_title(self, thread_id: str, count: int = 0) -> str:
        return "New chat" if count == 0 else "Untitled chat"

    @staticmethod
    def _normalize_chat_title(title: str) -> str:
        cleaned = re.sub(r"\s+", " ", title or "").strip().strip("\"'`")
        cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
        cleaned = re.sub(r"[.!?:;,]+$", "", cleaned)
        if len(cleaned) > 60:
            clipped = cleaned[:60].rsplit(" ", 1)[0].strip()
            cleaned = clipped or cleaned[:60].strip()
        return cleaned

    def _build_chat_title_source(self, thread_id: str, limit: int = 3) -> str:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            messages = list(state.values.get("messages", []))
        except Exception:
            messages = []

        lines: list[str] = []
        for msg in messages[-limit:]:
            content = getattr(msg, "content", "") or ""
            if not isinstance(content, str):
                continue
            snippet = re.sub(r"\s+", " ", content).strip()
            if not snippet:
                continue
            snippet = snippet[:280]
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            elif isinstance(msg, ToolMessage):
                role = f"Tool {getattr(msg, 'name', 'tool')}"
            else:
                role = msg.__class__.__name__
            lines.append(f"{role}: {snippet}")
        return "\n".join(lines)

    def _title_worker(self, thread_id: str) -> None:
        source = self._build_chat_title_source(thread_id)
        if not source.strip():
            return
        tcfg = SETTINGS.get("titler", {})
        address = tcfg.get("address", DEFAULT_SETTINGS["titler"]["address"]).strip()
        if not address:
            return
        prompt = tcfg.get("system_prompt", "").strip() or CHAT_TITLE_PROMPT
        title = ""
        try:
            response = get_llm(address, streaming=False, timeout=45).invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=source),
            ])
            content = getattr(response, "content", "")
            title = content if isinstance(content, str) else ""
        except Exception:
            pass

        cleaned = self._normalize_chat_title(title)
        if not cleaned:
            human_lines = [line[6:] for line in source.splitlines() if line.startswith("User: ")]
            if human_lines:
                cleaned = self._normalize_chat_title(" ".join(human_lines[:1]))
        if not cleaned:
            return
        self._save_chat_title(thread_id, cleaned)
        self._stream_queue.put(("title_done", thread_id, cleaned))

    def _schedule_chat_title_refresh(self, thread_id: str) -> None:
        threading.Thread(target=self._title_worker, args=(thread_id,), daemon=True).start()

    # ------------------------------------------------------------------
    # Pipeline log
    # ------------------------------------------------------------------

    def _plog(self, msg: str) -> None:
        """Append a timestamped entry to the rolling pipeline event log.
        Only called from the main (UI) thread — all worker thread logs arrive
        via ("plog", msg) queue events which _drain_queue handles.
        """
        ts = time.strftime("%H:%M:%S")
        self._pipeline_log.append(f"{ts} {msg}")
        # Keep the log bounded so it never grows unbounded across a long session
        if len(self._pipeline_log) > 150:
            del self._pipeline_log[0]

    def run(self):
        curses.curs_set(1)
        self.stdscr.keypad(True)
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

        if curses.can_change_color() and curses.COLORS >= 16:
            curses.init_color(8,  1000, 714, 796)
            curses.init_color(9,   471, 235, 706)
            curses.init_color(10,  961, 663, 722)
            curses.init_color(11,  357, 808, 980)
            curses.init_color(12,  784, 667, 902)
            curses.init_color(13,  588, 902, 392)   # pastel lime-green
            curses.init_color(14, 1000, 949, 702)   # pastel yellow (RAG output)
            pink, purple = 8, 9
            trans_pink, trans_blue, pastel_purple, lime, pastel_yellow = 10, 11, 12, 13, 14
        else:
            pink, purple       = curses.COLOR_MAGENTA, curses.COLOR_BLUE
            trans_pink         = curses.COLOR_MAGENTA
            trans_blue         = curses.COLOR_CYAN
            pastel_purple      = curses.COLOR_MAGENTA
            lime               = curses.COLOR_GREEN
            pastel_yellow      = curses.COLOR_YELLOW

        curses.init_pair(1, purple,        pink)
        curses.init_pair(2, trans_pink,    curses.COLOR_BLACK)
        curses.init_pair(3, trans_blue,    curses.COLOR_BLACK)
        curses.init_pair(4, pastel_purple, curses.COLOR_BLACK)
        curses.init_pair(5, lime,          curses.COLOR_BLACK)
        curses.init_pair(6, pastel_yellow, curses.COLOR_BLACK)

        # 1-second getch timeout so the monitor panel refreshes without input
        self.stdscr.timeout(GETCH_TIMEOUT_MS)
        self._start_monitor_thread()

        # Start the Lovense callback server so we're ready to receive the toy
        # pairing POST as soon as the user scans their QR code.
        _lv_cfg  = SETTINGS.get("lovense", {})
        _lv_port = _lv_cfg.get("callback_port", "")
        if _lv_port:
            try:
                import lovense as _lv_mod
                # Supply the developer token so server-side commands work.
                _lv_mod.configure(_lv_cfg.get("token", ""))
                _lv_mod.start_callback_server(
                    int(_lv_port),
                    certfile=_lv_cfg.get("cert_file", ""),
                    keyfile =_lv_cfg.get("key_file",  ""),
                )
            except Exception:
                pass   # non-fatal — toy control just won't work until fixed

        while True:
            self.stdscr.erase()
            self._draw_menubar()
            if self.view == VIEW_CHAT:
                self._draw_chat()
            elif self.view == VIEW_SETTINGS:
                self._draw_settings()
            elif self.view == VIEW_HELP:
                self._draw_help()
            else:
                self._draw_home()
            # Draw chat-list modal on top of the active view when open
            if self._chats_modal_open:
                self._draw_chats_modal()
            # Draw tool-approval overlay on top of everything when in agent mode
            if self._approval_pending is not None:
                self._draw_approval_overlay()
            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key == curses.ERR:
                self._drain_queue()
                continue
            if not self._handle_key(key):
                break
            self._drain_queue()

        self._stop_event.set()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _handle_key(self, key) -> bool:
        # When the chats modal is open, it captures all keyboard input exclusively
        if self._chats_modal_open:
            return self._handle_chats_modal_key(key)
        # Approval overlay captures Y / N / ESC exclusively
        if self._approval_pending is not None:
            if key in (ord('y'), ord('Y')):
                approval.resolve(True)
                self._approval_pending = None
            elif key in (ord('n'), ord('N'), 27):
                approval.resolve(False)
                self._approval_pending = None
            return True
        if self.view == VIEW_CHAT and not self.thinking and not self._select_mode and key == 27:
            if self._handle_chat_escape_sequence():
                return True
        if key == curses.KEY_F1:
            self.view = VIEW_CHAT
        elif key == curses.KEY_F2:
            self._open_chats_modal()
        elif key == curses.KEY_F3:
            modes = ["auto", "agent", "plan", "chat"]
            cur = SETTINGS.get("mode", "auto")
            SETTINGS["mode"] = modes[(modes.index(cur) + 1) % len(modes)]
            save_settings(SETTINGS)
        elif key == curses.KEY_F5:
            if self.view == VIEW_CHAT and not self.thinking:
                self._compact_context()
        elif key == curses.KEY_F10:
            self.prev_view = self.view
            self.view      = VIEW_HELP
        elif key == curses.KEY_F12:
            self.prev_view        = self.view
            self.settings_bufs    = self._bufs_from_settings()
            self.settings_cursors = [len(b) for b in self.settings_bufs]
            self.settings_focus   = 0
            self.view             = VIEW_SETTINGS
        elif key == 27:  # ESC
            if self.thinking:
                self._cancel_event.set()   # ask stream worker to stop after current call
                self._ignore_stream = True # drop any events it puts in the queue
                self._flush_queue()        # discard events already queued
                self.thinking    = False   # immediately return UI to ready state
                self._cancelling = False
                self._ai_idx     = None
                self._active_request_thread_id = None
            elif self.view == VIEW_SETTINGS:
                self._commit_settings()
                self.view = self.prev_view
            elif self.view == VIEW_HELP:
                self.view = self.prev_view
            elif self.view == VIEW_HOME:
                return False
            else:
                self.view = VIEW_HOME
        elif key == curses.KEY_MOUSE:
            try:
                _, _, _, _, bstate = curses.getmouse()
                if self.view == VIEW_CHAT:
                    if bstate & curses.BUTTON4_PRESSED:
                        self.chat_scroll = max(0, self.chat_scroll + 3)
                    btn5 = getattr(curses, 'BUTTON5_PRESSED', 0)
                    if btn5 and bstate & btn5:
                        self.chat_scroll = max(0, self.chat_scroll - 3)
            except curses.error:
                pass
        elif self.view == VIEW_SETTINGS:
            self._handle_settings_key(key)
        elif self.view == VIEW_CHAT:
            if self._select_mode:
                return self._handle_select_key(key)
            h, w = self.stdscr.getmaxyx()
            if key == curses.KEY_PPAGE:
                self.chat_scroll = max(0, self.chat_scroll + h // 2)
            elif key == curses.KEY_NPAGE:
                self.chat_scroll = max(0, self.chat_scroll - h // 2)
            elif not self.thinking:
                field_w = w - 3
                pos     = self.input_cursor
                if key in (curses.KEY_ENTER, 10, 13):
                    # Peek ahead: if more input is queued, this newline came from paste — insert it.
                    # If nothing is waiting, the user pressed Enter — send.
                    self.stdscr.timeout(0)
                    lookahead = self.stdscr.getch()
                    self.stdscr.timeout(GETCH_TIMEOUT_MS)
                    if lookahead != curses.ERR:
                        curses.ungetch(lookahead)
                        self._insert_input_text("\n")
                    else:
                        self._input_goal_col = None
                        self._send()
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    if pos > 0:
                        self.input_buf    = self.input_buf[:pos - 1] + self.input_buf[pos:]
                        self.input_cursor = pos - 1
                        self._input_goal_col = None
                elif key == curses.KEY_DC:
                    if pos < len(self.input_buf):
                        self.input_buf = self.input_buf[:pos] + self.input_buf[pos + 1:]
                        self._input_goal_col = None
                elif key == curses.KEY_LEFT:
                    self.input_cursor = max(0, pos - 1)
                    self._input_goal_col = None
                elif key == curses.KEY_RIGHT:
                    self.input_cursor = min(len(self.input_buf), pos + 1)
                    self._input_goal_col = None
                elif key == curses.KEY_HOME:
                    self.input_cursor = 0
                    self._input_goal_col = None
                elif key == curses.KEY_END:
                    self.input_cursor = len(self.input_buf)
                    self._input_goal_col = None
                elif key == curses.KEY_UP:
                    self._move_input_vertically(-1, field_w)
                elif key == curses.KEY_DOWN:
                    self._move_input_vertically(1, field_w)
                elif key == 22:  # Ctrl+V — paste from Windows clipboard
                    clip = _read_clipboard()
                    if clip:
                        self._insert_input_text(clip)
                elif 32 <= key <= 126:
                    self._insert_input_text(chr(key))
        return True

    def _handle_settings_key(self, key) -> None:
        idx       = self.settings_focus
        buf       = self.settings_bufs[idx]
        pos       = self.settings_cursors[idx]
        is_prompt = idx in (F_AGENT_PROMPT, F_RESEARCHER_PROMPT, F_CLASSIFIER_PROMPT, F_TITLER_PROMPT)

        _, w    = self.stdscr.getmaxyx()
        field_w = max(10, w - 18 - 4)

        if key == 9:  # Tab → next field
            self.settings_focus = (idx + 1) % NUM_FIELDS
        elif key == curses.KEY_UP:
            if is_prompt:
                vlines = self._compute_visual_lines(buf, field_w)
                row, col = self._cursor_to_visual(vlines, pos)
                if row > 0:
                    start, line = vlines[row - 1]
                    self.settings_cursors[idx] = start + min(col, len(line))
                else:
                    self.settings_focus = (idx - 1) % NUM_FIELDS
            else:
                self.settings_focus = (idx - 1) % NUM_FIELDS
        elif key == curses.KEY_DOWN:
            if is_prompt:
                vlines = self._compute_visual_lines(buf, field_w)
                row, col = self._cursor_to_visual(vlines, pos)
                if row < len(vlines) - 1:
                    start, line = vlines[row + 1]
                    self.settings_cursors[idx] = start + min(col, len(line))
                else:
                    self.settings_focus = (idx + 1) % NUM_FIELDS
            else:
                self.settings_focus = (idx + 1) % NUM_FIELDS
        elif key == curses.KEY_LEFT:
            self.settings_cursors[idx] = max(0, pos - 1)
        elif key == curses.KEY_RIGHT:
            self.settings_cursors[idx] = min(len(buf), pos + 1)
        elif key == curses.KEY_HOME:
            self.settings_cursors[idx] = 0
        elif key == curses.KEY_END:
            self.settings_cursors[idx] = len(buf)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if pos > 0:
                self.settings_bufs[idx]    = buf[:pos - 1] + buf[pos:]
                self.settings_cursors[idx] = pos - 1
        elif key == curses.KEY_DC:
            if pos < len(buf):
                self.settings_bufs[idx] = buf[:pos] + buf[pos + 1:]
        elif key in (curses.KEY_ENTER, 10, 13):
            if is_prompt:
                self.settings_bufs[idx]    = buf[:pos] + "\n" + buf[pos:]
                self.settings_cursors[idx] = pos + 1
            else:
                self.settings_focus = (idx + 1) % NUM_FIELDS
        elif key == 22:  # Ctrl+V — paste from Windows clipboard
            clip = _read_clipboard()
            if clip:
                self.settings_bufs[idx]    = buf[:pos] + clip + buf[pos:]
                self.settings_cursors[idx] = pos + len(clip)
        elif 32 <= key <= 126:
            self.settings_bufs[idx]    = buf[:pos] + chr(key) + buf[pos:]
            self.settings_cursors[idx] = pos + 1

    def _commit_settings(self) -> None:
        SETTINGS["agent"]["address"]            = self.settings_bufs[F_AGENT_ADDR]
        SETTINGS["agent"]["system_prompt"]      = self.settings_bufs[F_AGENT_PROMPT]
        SETTINGS["agent"]["cwd"]               = self.settings_bufs[F_AGENT_CWD]
        SETTINGS["researcher"]["address"]       = self.settings_bufs[F_RESEARCHER_ADDR]
        SETTINGS["researcher"]["system_prompt"] = self.settings_bufs[F_RESEARCHER_PROMPT]
        SETTINGS.setdefault("brave", {})
        SETTINGS["brave"]["api_key"]            = self.settings_bufs[F_BRAVE_API_KEY]
        SETTINGS["brave"]["base_url"]           = self.settings_bufs[F_BRAVE_BASE_URL]
        SETTINGS["brave"]["count"]              = self.settings_bufs[F_BRAVE_COUNT]
        SETTINGS["brave"]["country"]            = self.settings_bufs[F_BRAVE_COUNTRY]
        SETTINGS["brave"]["search_lang"]        = self.settings_bufs[F_BRAVE_SEARCH_LANG]
        SETTINGS["brave"]["safesearch"]         = self.settings_bufs[F_BRAVE_SAFESEARCH]
        SETTINGS["classifier"]["address"]       = self.settings_bufs[F_CLASSIFIER_ADDR]
        SETTINGS["classifier"]["system_prompt"] = self.settings_bufs[F_CLASSIFIER_PROMPT]
        SETTINGS.setdefault("titler", {})
        SETTINGS["titler"]["address"]           = self.settings_bufs[F_TITLER_ADDR]
        SETTINGS["titler"]["system_prompt"]     = self.settings_bufs[F_TITLER_PROMPT]
        # Persist Lovense settings and restart the callback server on port change.
        old_port = SETTINGS.get("lovense", {}).get("callback_port", "")
        SETTINGS.setdefault("lovense", {})
        SETTINGS["lovense"]["token"]         = self.settings_bufs[F_LOVENSE_TOKEN]
        SETTINGS["lovense"]["uid"]           = self.settings_bufs[F_LOVENSE_UID]
        SETTINGS["lovense"]["callback_port"] = self.settings_bufs[F_LOVENSE_PORT]
        SETTINGS["lovense"]["callback_host"] = self.settings_bufs[F_LOVENSE_HOST]
        SETTINGS["lovense"]["cert_file"]     = self.settings_bufs[F_LOVENSE_CERT]
        SETTINGS["lovense"]["key_file"]      = self.settings_bufs[F_LOVENSE_KEY]
        save_settings(SETTINGS)
        rebuild_llms()
        # Also restart the callback server with updated cert/key if port changed.
        new_port = SETTINGS["lovense"]["callback_port"]
        if new_port != old_port and new_port:
            try:
                import lovense as _lv
                _lv.configure(SETTINGS["lovense"].get("token", ""))
                _lv.stop_callback_server()
                _lv.start_callback_server(
                    int(new_port),
                    certfile=SETTINGS["lovense"].get("cert_file", ""),
                    keyfile =SETTINGS["lovense"].get("key_file",  ""),
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Chat list modal
    # ------------------------------------------------------------------

    def _open_chats_modal(self) -> None:
        """Refresh the thread list from DB and open the modal overlay."""
        self._refresh_chats_list()
        # Pre-select the currently active thread so it's visible on open
        for i, entry in enumerate(self._chats_list):
            if entry["thread_id"] == self.thread_id:
                self._chats_cursor = i
                break
        else:
            self._chats_cursor = 0
        self._chats_modal_open = True

    def _refresh_chats_list(self) -> None:
        """Query chat_history.db for all thread IDs ordered by most recent checkpoint."""
        try:
            conn = sqlite3.connect(self._db_path)
            cur  = conn.cursor()
            # checkpoint_id is a UUIDv1 — lexicographic MAX gives the latest activity
            cur.execute("""
                SELECT c.thread_id, c.cnt, c.latest, COALESCE(t.title, '')
                FROM (
                    SELECT thread_id, COUNT(*) AS cnt, MAX(checkpoint_id) AS latest
                    FROM   checkpoints
                    GROUP  BY thread_id
                ) AS c
                LEFT JOIN chat_titles AS t
                       ON t.thread_id = c.thread_id
                ORDER BY c.latest DESC
            """)
            rows = cur.fetchall()
            conn.close()
            self._chats_list = [
                {
                    "thread_id": r[0],
                    "count": r[1],
                    "title": self._normalize_chat_title(r[3]) or self._fallback_chat_title(r[0], r[1]),
                }
                for r in rows
            ]
        except Exception:
            self._chats_list = []
        # Always include the active thread even if the DB is empty or unreachable
        existing_ids = {e["thread_id"] for e in self._chats_list}
        if self.thread_id not in existing_ids:
            self._chats_list.insert(0, {
                "thread_id": self.thread_id,
                "count": 0,
                "title": self._fallback_chat_title(self.thread_id, 0),
            })
        # Clamp cursor so it stays in bounds (e.g. after a deletion)
        self._chats_cursor = min(self._chats_cursor, max(0, len(self._chats_list) - 1))

    def _delete_chat(self, thread_id: str) -> None:
        """Permanently remove all checkpoint data for a thread from the SQLite DB."""
        try:
            conn = sqlite3.connect(self._db_path)
            cur  = conn.cursor()
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            try:
                # 'writes' table exists in most langgraph-checkpoint-sqlite versions
                cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                pass  # older schema versions may not have this table
            cur.execute("DELETE FROM chat_titles WHERE thread_id = ?", (thread_id,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _load_thread_history(self, thread_id: str) -> None:
        """Rebuild the in-memory display history from the LangGraph checkpoint state."""
        try:
            config   = {"configurable": {"thread_id": thread_id}}
            state    = self.graph.get_state(config)
            messages = list(state.values.get("messages", []))
            self.history = []
            for msg in messages:
                content = getattr(msg, "content", "") or ""
                if isinstance(msg, HumanMessage) and content:
                    self.history.append(("user", content))
                elif isinstance(msg, AIMessage):
                    # content can be a string; skip pure tool-call messages with no text
                    text = content if isinstance(content, str) else ""
                    if text.strip():
                        self.history.append(("ai", text))
                elif isinstance(msg, ToolMessage) and content:
                    name = getattr(msg, "name", "tool")
                    self.history.append(("tool", f"[Tool: {name}]\n{content}"))
        except Exception:
            self.history = []

    def _handle_chats_modal_key(self, key) -> bool:
        """Handle keyboard input while the chats modal is active."""
        if key in (27, curses.KEY_F2):              # ESC or F2 toggle → close
            self._chats_modal_open = False

        elif key == curses.KEY_UP:
            self._chats_cursor = max(0, self._chats_cursor - 1)

        elif key == curses.KEY_DOWN:
            if self._chats_list:
                self._chats_cursor = min(len(self._chats_list) - 1, self._chats_cursor + 1)

        elif key in (curses.KEY_ENTER, 10, 13):     # Enter → switch to selected thread
            if self._chats_list:
                selected         = self._chats_list[self._chats_cursor]
                self.thread_id   = selected["thread_id"]
                self.chat_scroll = 0
                self._load_thread_history(self.thread_id)
                self._chats_modal_open = False
                self.view = VIEW_CHAT

        elif key in (ord('d'), ord('D')):            # D → delete (non-active threads only)
            if self._chats_list:
                selected = self._chats_list[self._chats_cursor]
                tid      = selected["thread_id"]
                if tid != self.thread_id:            # guard: never delete the live thread
                    self._delete_chat(tid)
                    self._refresh_chats_list()
                    # Re-anchor cursor on the active thread after the list shrinks
                    for i, entry in enumerate(self._chats_list):
                        if entry["thread_id"] == self.thread_id:
                            self._chats_cursor = i
                            break

        elif key in (ord('n'), ord('N')):            # N → start a brand-new chat
            self._activate_new_chat()
            self._chats_modal_open = False
            self.view              = VIEW_CHAT

        elif key in (ord('r'), ord('R')):            # R → regenerate selected chat title
            if self._chats_list:
                selected = self._chats_list[self._chats_cursor]
                self._schedule_chat_title_refresh(selected["thread_id"])

        return True

    def _draw_approval_overlay(self) -> None:
        """Render a centered overlay asking the user to approve or reject a tool call."""
        if self._approval_pending is None:
            return
        h, w = self.stdscr.getmaxyx()
        bar  = curses.color_pair(1)
        name = self._approval_pending.get("name", "?")
        args = self._approval_pending.get("args", {})

        # Format args as indented key: value lines, capped to avoid overflow
        arg_lines = []
        for k, v in args.items():
            line = f"  {k}: {v!r}"
            if len(line) > w - 6:
                line = line[:w - 9] + "..."
            arg_lines.append(line)
        arg_lines = arg_lines[:12]   # never overflow the screen

        body = [f" Tool: {name} ", ""] + (arg_lines or ["  (no arguments)"]) + [""]
        modal_h = min(h - 4, len(body) + 4)
        modal_w = min(w - 4, max(40, max(len(ln) for ln in body) + 4))
        top     = max(1, (h - modal_h) // 2)
        left    = max(0, (w - modal_w) // 2)

        for row in range(modal_h):
            try:
                self.stdscr.addstr(top + row, left, " " * modal_w, bar)
            except curses.error:
                pass

        # Header
        try:
            hdr = f" Agent wants to call a tool "
            self.stdscr.addstr(top, left + 1, hdr[:modal_w - 2], bar)
        except curses.error:
            pass

        # Body lines
        for i, line in enumerate(body[:modal_h - 2]):
            try:
                self.stdscr.addstr(top + 1 + i, left + 1, line[:modal_w - 2])
            except curses.error:
                pass

        # Footer
        footer = " [Y] Approve   [N] Reject "
        try:
            self.stdscr.addstr(top + modal_h - 1, left + 1, footer[:modal_w - 2], bar)
        except curses.error:
            pass

    def _draw_chats_modal(self) -> None:
        """Render a centered overlay listing all chat threads."""
        h, w = self.stdscr.getmaxyx()

        # ── Dimensions ────────────────────────────────────────────────
        # header(1) + list entries + divider(1) + footer(1) = n + 3
        list_count = max(1, len(self._chats_list))
        modal_h    = min(h - 4, list_count + 3)
        modal_w    = min(w - 4, 72)
        y0         = max(0, (h - modal_h) // 2)
        x0         = max(0, (w - modal_w) // 2)
        inner_w    = modal_w - 2   # usable width inside 1-char side padding

        bar      = curses.color_pair(1)
        sel_attr = curses.A_REVERSE

        # ── Background fill ───────────────────────────────────────────
        for row in range(modal_h):
            try:
                self.stdscr.addstr(y0 + row, x0, " " * modal_w, bar)
            except curses.error:
                pass

        # ── Title row ─────────────────────────────────────────────────
        title = "── Chats ──"   # ── Chats ──
        try:
            self.stdscr.addstr(
                y0, x0 + max(0, (modal_w - len(title)) // 2),
                title, bar | curses.A_BOLD,
            )
        except curses.error:
            pass

        # ── Thread list ───────────────────────────────────────────────
        list_area_h = modal_h - 3   # subtract title, divider, footer rows
        # Scroll so the cursor row stays visible inside the list area
        scroll = max(0, self._chats_cursor - list_area_h + 1)

        for li, entry in enumerate(self._chats_list[scroll: scroll + list_area_h]):
            actual_i   = scroll + li
            is_current = (entry["thread_id"] == self.thread_id)
            is_sel     = (actual_i == self._chats_cursor)

            prefix  = "● "  if is_current else "  "   # ● marks the active thread
            title   = entry.get("title") or self._fallback_chat_title(entry["thread_id"], entry.get("count", 0))
            cnt_str = f"[{entry['count']}]"
            avail   = inner_w - len(cnt_str)
            label   = (prefix + title)[:avail].ljust(avail)
            line    = label + cnt_str

            try:
                self.stdscr.addstr(
                    y0 + 1 + li, x0 + 1, line[:inner_w],
                    bar | (sel_attr if is_sel else 0),
                )
            except curses.error:
                pass

        # ── Divider ───────────────────────────────────────────────────
        try:
            self.stdscr.addstr(y0 + modal_h - 2, x0 + 1, "─" * inner_w, bar)
        except curses.error:
            pass

        # ── Footer hint ───────────────────────────────────────────────
        # Remind user they can't delete the currently active thread
        active_selected = (
            bool(self._chats_list)
            and self._chats_list[self._chats_cursor]["thread_id"] == self.thread_id
        )
        footer = (
            " Enter:open  N:new  ESC:close  (D:delete — can't delete active) "
            if active_selected
            else " Enter:open  N:new  D:delete  ESC:close "
        )
        footer = footer.replace(" Enter:open  N:new ", " Enter:open  N:new  R:retitle  ")
        try:
            self.stdscr.addstr(y0 + modal_h - 1, x0 + 1, footer[:inner_w], bar)
        except curses.error:
            pass

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _enter_select_mode(self) -> None:
        if not self._select_lines:
            return
        total = len(self._select_lines)
        end   = total - self.chat_scroll
        # Place cursor at the last visible line so it feels natural
        self._select_cursor = max(0, min(end - 1, total - 1))
        self._select_anchor = self._select_cursor
        self._select_mode   = True

    def _handle_select_key(self, key) -> bool:
        total = len(self._select_lines)
        if total == 0:
            self._select_mode = False
            return True
        h, w    = self.stdscr.getmaxyx()
        INPUT_H = self._chat_input_height(w - 3, h)
        chat_h  = max(1, h - INPUT_H - 1 - 1)

        if key == 27:                          # ESC → exit without copying
            self._select_mode   = False
            self._select_anchor = None
        elif key == curses.KEY_UP:
            self._select_cursor = max(0, self._select_cursor - 1)
            self._select_scroll_to_cursor(total, chat_h)
        elif key == curses.KEY_DOWN:
            self._select_cursor = min(total - 1, self._select_cursor + 1)
            self._select_scroll_to_cursor(total, chat_h)
        elif key == curses.KEY_PPAGE:
            self._select_cursor = max(0, self._select_cursor - chat_h)
            self._select_scroll_to_cursor(total, chat_h)
        elif key == curses.KEY_NPAGE:
            self._select_cursor = min(total - 1, self._select_cursor + chat_h)
            self._select_scroll_to_cursor(total, chat_h)
        elif key == curses.KEY_HOME:
            self._select_cursor = 0
            self._select_scroll_to_cursor(total, chat_h)
        elif key == curses.KEY_END:
            self._select_cursor = total - 1
            self._select_scroll_to_cursor(total, chat_h)
        elif key == 11:                        # Ctrl+K → copy and exit
            self._copy_selection()
            self._select_mode   = False
            self._select_anchor = None
        return True

    def _select_scroll_to_cursor(self, total: int, chat_h: int) -> None:
        end   = total - self.chat_scroll
        start = max(0, end - chat_h)
        if self._select_cursor < start:
            self.chat_scroll = max(0, total - self._select_cursor - chat_h)
        elif self._select_cursor >= end:
            self.chat_scroll = max(0, total - self._select_cursor - 1)
        self.chat_scroll = max(0, min(self.chat_scroll, max(0, total - chat_h)))

    def _copy_selection(self) -> None:
        if not self._select_lines:
            return
        anchor = self._select_anchor if self._select_anchor is not None else self._select_cursor
        lo = min(anchor, self._select_cursor)
        hi = max(anchor, self._select_cursor)
        text = "\n".join(line for _, line in self._select_lines[lo:hi + 1])
        try:
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "$text = [Console]::In.ReadToEnd(); Set-Clipboard $text"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True,
            )
            proc.communicate(input=text, timeout=5)
        except Exception:
            pass

    def _redraw(self) -> None:
        self.chat_scroll = 0
        self.stdscr.erase()
        self._draw_menubar()
        self._draw_chat()
        self.stdscr.refresh()

    def _flush_queue(self) -> None:
        """Discard all pending events — called before starting a new stream."""
        while True:
            try:
                self._stream_queue.get_nowait()
            except queue.Empty:
                break

    def _send(self):
        text = self.input_buf
        if not text.strip():
            return
        self.input_buf    = ""
        self.input_cursor = 0
        self._input_goal_col = None
        self.chat_scroll  = 0

        # ── Slash commands ────────────────────────────────────────────────
        # /lovense commands are handled locally without hitting the LLM graph.
        if text.startswith("/lovense"):
            parts  = text.split(maxsplit=1)
            subcmd = parts[1].strip().lower() if len(parts) > 1 else "status"
            self.history.append(("user", text))
            if subcmd == "connect":
                self._lovense_connect()
            elif subcmd == "test":
                self._lovense_test()
            elif subcmd == "stop":
                self._lovense_stop()
            elif subcmd == "disconnect":
                self._lovense_disconnect()
            elif subcmd == "debug":
                self._lovense_debug()
            else:
                self._lovense_status()
            self._redraw()
            return

        self.history.append(("user", text))
        self._active_request_thread_id = self.thread_id
        self.thinking         = True
        self._cancelling      = False
        self._ignore_stream   = False
        self._log_lines       = []
        self._pipeline_log    = []                # fresh log for each new request
        self._ai_idx          = None
        self._last_token_usage = {}               # reset so bar shows "waiting..." until new data arrives
        self._last_queue_event = time.monotonic()
        self._cancel_event.clear()
        self._flush_queue()
        approval.set_notify_queue(self._stream_queue)
        self._start_log_thread()
        self._redraw()
        threading.Thread(target=self._stream_worker, args=(text,), daemon=True).start()
        threading.Thread(target=self._stream_watchdog, daemon=True).start()

    # ------------------------------------------------------------------
    # Lovense slash-command handlers
    # ------------------------------------------------------------------

    def _lovense_connect(self) -> None:
        """
        Request a pairing QR code from the Lovense API and show the URL in chat.

        The user must:
          1. Set their developer token and user ID in F12 > Settings.
          2. Type /lovense connect
          3. Open the QR URL in a browser and scan it with Lovense Remote.
          4. Once scanned, the callback server receives the toy info automatically.
        """
        lv_cfg = SETTINGS.get("lovense", {})
        token  = lv_cfg.get("token", "").strip()
        uid    = lv_cfg.get("uid", "").strip()
        port   = lv_cfg.get("callback_port", "34569").strip()
        host   = lv_cfg.get("callback_host", "").strip()

        # Validate we have the required credentials before hitting the API.
        if not token or not uid:
            self.history.append((
                "router",
                "[Lovense] No developer token or user ID set.\n"
                "Open F12 → Settings → Lovense and fill in Dev Token + User ID.",
            ))
            return

        # Auto-detect LAN IP if the user didn't supply a specific host.
        try:
            import lovense as _lv
            resolved_host = host or _lv.get_local_ip()
        except Exception as exc:
            self.history.append(("router", f"[Lovense] Failed to import lovense module: {exc}"))
            return

        callback_url = f"http://{resolved_host}:{port}/"
        # Upgrade to https:// if TLS certs are configured — Lovense Remote
        # requires HTTPS for real callbacks (plain HTTP will be refused).
        cert = lv_cfg.get("cert_file", "").strip()
        key  = lv_cfg.get("key_file",  "").strip()
        if cert and key:
            callback_url = f"https://{resolved_host}:{port}/"

        self.history.append(("router", f"[Lovense] Requesting QR code…  callback → {callback_url}"))
        self._redraw()

        # Run the HTTP call off the UI thread so curses doesn't freeze.
        def _worker():
            try:
                data = _lv.get_qr(token, uid, callback_url)
                qr_url    = data.get("qr", "")
                pair_code = data.get("code", "")
                msg = (
                    f"[Lovense] QR code ready!\n"
                    f"Open this URL and scan with Lovense Remote:\n"
                    f"  {qr_url}\n"
                    f"PC pairing code (if using Lovense Remote for PC): {pair_code}\n"
                    f"After scanning the callback server will receive your toy info."
                )
            except Exception as exc:
                msg = f"[Lovense] QR request failed: {exc}"
            self._stream_queue.put(("lovense_msg", msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _lovense_status(self) -> None:
        """Show current Lovense connection status in the chat."""
        try:
            import lovense as _lv
            if _lv.is_connected():
                names = _lv.get_toy_names()
                toy_str = ", ".join(names) if names else "unknown"
                msg = f"[Lovense] Connected — toys: {toy_str}"
            else:
                msg = (
                    "[Lovense] Not connected yet.\n"
                    "Use /lovense connect to get a pairing QR code, then scan it\n"
                    "with the Lovense Remote app to link your toy."
                )
        except Exception as exc:
            msg = f"[Lovense] Status check failed: {exc}"
        self.history.append(("router", msg))

    def _lovense_test(self) -> None:
        """Vibrate for 3 seconds to verify the toy command path is working."""
        try:
            import lovense as _lv
            if not _lv.is_connected():
                self.history.append(("router",
                    "[Lovense] Not connected — use /lovense connect first."))
                return
            _lv.activate(strength=10, duration_sec=3)
            self.history.append(("router",
                "[Lovense] Test vibration sent (3 s at strength 10).\n"
                "If you feel nothing, check that Lovense Remote is open and "
                "the toy is on, then try /lovense stop and /lovense test again."))
        except Exception as exc:
            self.history.append(("router", f"[Lovense] Test failed: {exc}"))

    def _lovense_stop(self) -> None:
        """Send a stop command immediately."""
        try:
            import lovense as _lv
            _lv.deactivate()
            self.history.append(("router", "[Lovense] Stop command sent."))
        except Exception as exc:
            self.history.append(("router", f"[Lovense] Stop failed: {exc}"))

    def _lovense_disconnect(self) -> None:
        """Clear toy state so the app treats the toy as unpaired."""
        try:
            import lovense as _lv
            _lv.disconnect()
            self.history.append(("router",
                "[Lovense] Disconnected — toy state cleared. "
                "Use /lovense connect to re-pair."))
        except Exception as exc:
            self.history.append(("router", f"[Lovense] Disconnect failed: {exc}"))

    def _lovense_debug(self) -> None:
        """Dump raw connection state so we can diagnose pairing problems."""
        try:
            import lovense as _lv
            info = _lv.get_debug_info()
            self.history.append(("router", f"[Lovense debug]\n{info}"))
        except Exception as exc:
            self.history.append(("router", f"[Lovense] Debug failed: {exc}"))

    def _stream_worker(self, text: str) -> None:
        q = self._stream_queue
        # Signal that the background thread has started; useful for spotting
        # hangs that occur before the first LangGraph node fires.
        q.put(("plog", "stream: started"))
        q.put(("plog", f"request: {_truncate_log_text(text, 120)}"))
        try:
            pending_ai  = False
            first_chunk = False   # gate for the 'ai: first token' log entry
            q.put(("plog", "graph: opening stream"))
            for item in self.graph.stream(
                {"messages": [HumanMessage(content=text)]},
                config={"configurable": {"thread_id": self.thread_id}},
                stream_mode=["updates", "messages"],
            ):
                if self._cancel_event.is_set():
                    break
                mode, data = item

                if mode == "updates":
                    for node_name, update in data.items():
                        # Log every node that fires so we can see exactly where
                        # the pipeline is (or was, when it hangs).
                        q.put(("plog", f"node: {node_name}"))
                        if node_name == "classify" and update.get("routing_note"):
                            q.put(("classify", update["routing_note"]))
                            q.put(("plog", f"classify: {update['routing_note']}"))
                        elif node_name == "web_research":
                            query = update.get("web_query", "")
                            result_count = len(update.get("web_results", []) or [])
                            note = update.get("routing_note", "")
                            if query:
                                q.put(("plog", f"web: query={_truncate_log_text(query, 90)}"))
                            q.put(("plog", f"web: results={result_count}"))
                            if note:
                                idx = note.rfind("[Web")
                                web_note = note[idx:] if idx != -1 else note
                                q.put(("plog", f"web: {web_note}"))
                        elif node_name == "rag":
                            # Surface what the RAG agent retrieved (shown in pastel
                            # yellow). Isolate just the [RAG: ...] segment so we don't
                            # repeat the classify note, and include the context text.
                            note = update.get("routing_note", "")
                            ctx  = update.get("rag_context", "")
                            idx  = note.rfind("[RAG")
                            rag_note = note[idx:] if idx != -1 else "[RAG]"
                            q.put(("rag", rag_note, ctx))
                            q.put(("plog", f"rag: context chars={len(ctx)}"))
                        elif node_name in ("respond", "clarify"):
                            q.put(("plog", f"{node_name}: model returned update"))
                            for msg in update.get("messages", []):
                                content = getattr(msg, "content", "") or ""
                                tool_calls = getattr(msg, "tool_calls", None) or []
                                if tool_calls:
                                    tool_names = ", ".join(tc.get("name", "tool") for tc in tool_calls)
                                    q.put(("plog", f"{node_name}: tool_calls={tool_names}"))
                                elif content:
                                    q.put(("plog", f"{node_name}: text chars={len(str(content))}"))
                                usage = getattr(msg, "usage_metadata", None)
                                if not usage:
                                    rm = getattr(msg, "response_metadata", {}) or {}
                                    tu = rm.get("token_usage", {})
                                    if tu:
                                        usage = {
                                            "input_tokens":  tu.get("prompt_tokens", 0),
                                            "output_tokens": tu.get("completion_tokens", 0),
                                            "total_tokens":  tu.get("total_tokens", 0),
                                        }
                                if usage:
                                    q.put(("token_usage", dict(usage)))
                        elif node_name == "tools":
                            q.put(("plog", f"tools: received {len(update.get('messages', []))} message(s)"))
                            for msg in update.get("messages", []):
                                q.put(("tool", getattr(msg, "name", "tool"), getattr(msg, "content", "")))
                            q.put(("ai_next",))
                            pending_ai = False
                        elif node_name == "dispatch_context_override":
                            q.put(("plog", "dispatch_context_override: rerouting request"))
                            q.put(("ai_strip_context_override",))
                            q.put(("ai_next",))
                            pending_ai = False
                        elif node_name == "dispatch_text_tools":
                            q.put(("plog", f"dispatch_text_tools: received {len(update.get('messages', []))} message(s)"))
                            q.put(("ai_strip_tool_calls",))
                            for msg in update.get("messages", []):
                                q.put(("tool", getattr(msg, "name", "tool"), getattr(msg, "content", "")))
                            q.put(("ai_next",))
                            pending_ai = False

                elif mode == "messages":
                    msg_chunk, metadata = data
                    if not isinstance(msg_chunk, AIMessageChunk):
                        continue
                    content = getattr(msg_chunk, "content", "") or ""
                    if metadata.get("langgraph_node", "") in ("respond", "clarify"):
                        # Capture usage from the final streaming chunk (usage_metadata is
                        # set on the last chunk when the model reports token counts inline).
                        chunk_usage = getattr(msg_chunk, "usage_metadata", None)
                        if chunk_usage and (chunk_usage.get("input_tokens") or chunk_usage.get("total_tokens")):
                            q.put(("token_usage", dict(chunk_usage)))
                        if content:
                            if not first_chunk:
                                first_chunk = True
                                q.put(("plog", f"ai: first token from {metadata.get('langgraph_node', 'unknown')}"))
                            q.put(("ai_start", content) if not pending_ai else ("ai_append", content))
                            pending_ai = True

        except Exception as exc:
            q.put(("error", str(exc)))
            return
        q.put(("plog", "stream: graph finished"))
        # Emit a sentinel so the UI thread can fill in a char-based estimate if
        # the model never reported real token usage (common with llama.cpp streaming).
        q.put(("token_usage_fallback", self.thread_id))
        q.put(("done",))

    def _stream_watchdog(self) -> None:
        """Force-complete if neither the pipeline queue nor the server log shows activity for 120s.

        Consumer hardware can take well over 60s to start its first token, but the
        llama.cpp server always emits log lines while it's working (prompt eval, token
        gen, etc.).  Watching those log lines gives us a real "sign of life" signal so
        we don't time out on a slow but healthy inference run.
        """
        last_log_line = ""
        last_log_ts   = time.monotonic()
        last_heartbeat_bucket = -1
        while self.thinking:
            time.sleep(2)
            if not self.thinking:
                return
            # Check whether the server produced any new log output since last poll.
            current_last = self._log_lines[-1] if self._log_lines else ""
            if current_last != last_log_line:
                last_log_line = current_last
                last_log_ts   = time.monotonic()
            # Activity = most recent of: a pipeline queue event OR a new server log line.
            last_activity = max(self._last_queue_event, last_log_ts)
            idle_for = time.monotonic() - last_activity
            heartbeat_bucket = int(idle_for // 15)
            if idle_for >= 15 and heartbeat_bucket != last_heartbeat_bucket:
                last_heartbeat_bucket = heartbeat_bucket
                queue_age = max(0, time.monotonic() - self._last_queue_event)
                log_age = max(0, time.monotonic() - last_log_ts)
                self._stream_queue.put((
                    "plog",
                    f"watchdog: waiting idle={idle_for:.0f}s queue_age={queue_age:.0f}s log_age={log_age:.0f}s",
                ))
            if (time.monotonic() - last_activity) > STREAM_STALL_TIMEOUT_SEC:
                self._stream_queue.put(("error", f"Stream stalled — no activity for {STREAM_STALL_TIMEOUT_SEC}s"))
                return

    def _drain_queue(self) -> None:
        if self._ignore_stream:
            self._flush_queue()
            return
        needs_redraw = False
        while True:
            try:
                event = self._stream_queue.get_nowait()
            except queue.Empty:
                break
            self._last_queue_event = time.monotonic()
            kind = event[0]
            if kind == "classify":
                self._plog(f"classify: {event[1][:45]}")
                self.history.append(("router", event[1]))
                needs_redraw = True
            elif kind == "rag":
                # event = ("rag", note, context). Show the RAG note plus the
                # retrieved context as one pastel-yellow block.
                note, ctx = event[1], event[2]
                clipped_ctx = _clip_display_text(ctx, RAG_DISPLAY_MAX_CHARS, "RAG context") if ctx else ""
                text = f"{note}\n{clipped_ctx}" if clipped_ctx else note
                self._plog("rag: context retrieved")
                self.history.append(("rag", text))
                self._ai_idx = None
                needs_redraw = True
            elif kind == "tool":
                self._plog(f"tool: {event[1]}")
                tool_content = _clip_display_text(event[2], TOOL_DISPLAY_MAX_CHARS, f"tool output: {event[1]}")
                self.history.append(("tool", f"[Tool: {event[1]}]\n{tool_content}"))
                self._ai_idx = None
                needs_redraw = True
            elif kind == "plog":
                # Timestamped log events emitted by the background stream worker.
                self._plog(event[1])
            elif kind == "ai_start":
                self._plog("ai: generating response")
                self.history.append(("ai", event[1]))
                self._ai_idx = len(self.history) - 1
                needs_redraw = True
                # Activate the Lovense toy when the AI starts its response.
                # We only activate on the very first ai_start (when _ai_idx was
                # None before) to avoid re-triggering on multi-segment responses.
                try:
                    import lovense as _lv
                    if _lv.is_connected():
                        _lv.activate()
                except Exception:
                    pass
            elif kind == "ai_append":
                if self._ai_idx is not None:
                    role, prev = self.history[self._ai_idx]
                    self.history[self._ai_idx] = (role, prev + event[1])
                    needs_redraw = True
            elif kind == "ai_next":
                self._ai_idx = None
            elif kind == "ai_strip_tool_calls":
                if self._ai_idx is not None:
                    role, prev = self.history[self._ai_idx]
                    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", prev, flags=re.DOTALL).strip()
                    tool_names = "|".join(sorted(re.escape(name) for name in TOOL_MAP))
                    cleaned = re.sub(rf"(?mis)^\s*(?:{tool_names})\s*\(.*\)\s*$", "", cleaned).strip()
                    if cleaned:
                        self.history[self._ai_idx] = (role, cleaned)
                    else:
                        self.history.pop(self._ai_idx)
                        self._ai_idx = None
                    needs_redraw = True
            elif kind == "ai_strip_context_override":
                if self._ai_idx is not None:
                    self.history.pop(self._ai_idx)
                    self._ai_idx = None
                    needs_redraw = True
            elif kind == "token_usage":
                self._last_token_usage = event[1]
                inp = event[1].get("input_tokens",  0)
                out = event[1].get("output_tokens", 0)
                self._plog(f"tokens in={inp:,} out={out:,}")
                needs_redraw = True
            elif kind == "token_usage_fallback":
                # Only estimate if the model never reported real usage this turn.
                if not self._last_token_usage.get("input_tokens"):
                    try:
                        tid = event[1]
                        config = {"configurable": {"thread_id": tid}}
                        state = self.graph.get_state(config)
                        messages = list(state.values.get("messages", []))
                        total_chars = sum(
                            len(str(getattr(m, "content", "") or ""))
                            for m in messages
                        )
                        if total_chars > 0:
                            est = total_chars // 4
                            self._last_token_usage = {
                                "input_tokens": est, "output_tokens": 0, "total_tokens": est,
                            }
                            self._plog(f"tokens est≈{est:,} (model did not report usage)")
                            needs_redraw = True
                    except Exception:
                        pass
            elif kind == "compact_done":
                self._plog(f"compact: {event[1]:,} tokens freed")
                self.thinking    = False
                self._cancelling = False
                self._ai_idx     = None
                self.history.append(("router", f"[Context compacted — {event[1]:,} tokens freed]"))
                needs_redraw = True
            elif kind == "done":
                self._plog("done ✓")
                self.thinking    = False
                self._cancelling = False
                self._ai_idx     = None
                needs_redraw     = True
                # Stop the toy when the AI finishes its full response.
                try:
                    import lovense as _lv
                    _lv.deactivate()
                except Exception:
                    pass
                if self._active_request_thread_id:
                    self._schedule_chat_title_refresh(self._active_request_thread_id)
                    self._active_request_thread_id = None
            elif kind == "error":
                self._plog(f"ERR: {event[1][:55]}")
                self.history.append(("router", f"[Error: {event[1]}]"))
                self.thinking    = False
                self._cancelling = False
                self._ai_idx     = None
                self._active_request_thread_id = None
                needs_redraw     = True
                # Also stop the toy on errors so it doesn't run indefinitely.
                try:
                    import lovense as _lv
                    _lv.deactivate()
                except Exception:
                    pass
            elif kind == "tool_approval_needed":
                _, tool_name, tool_args = event
                self._approval_pending = {"name": tool_name, "args": tool_args}
                needs_redraw = True
            elif kind == "lovense_msg":
                # Result from _lovense_connect() running off the UI thread.
                self.history.append(("router", event[1]))
                needs_redraw = True
            elif kind == "title_done":
                thread_id, title = event[1], event[2]
                for entry in self._chats_list:
                    if entry["thread_id"] == thread_id:
                        entry["title"] = title
                        break
                needs_redraw = needs_redraw or self._chats_modal_open
        if needs_redraw:
            self.stdscr.erase()
            self._draw_menubar()
            self._draw_chat()
            if self._chats_modal_open:
                self._draw_chats_modal()
            self.stdscr.refresh()

    # ------------------------------------------------------------------
    # Monitor / log background threads
    # ------------------------------------------------------------------

    def _compact_context(self) -> None:
        self.thinking       = True
        self._cancelling    = False
        self._ignore_stream = False
        self._log_lines     = ["Compacting context..."]
        self._last_queue_event = time.monotonic()
        self._cancel_event.clear()
        self._flush_queue()
        self._redraw()
        threading.Thread(target=self._compact_worker, daemon=True).start()
        threading.Thread(target=self._stream_watchdog, daemon=True).start()

    def _compact_worker(self) -> None:
        q = self._stream_queue
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state  = self.graph.get_state(config=config)
            messages = list(state.values.get("messages", []))
            if not messages:
                q.put(("compact_done", 0))
                return

            existing = state.values.get("summary", "")
            prefix   = f"Previous summary:\n{existing}\n\nNew messages:\n" if existing else ""
            history_text = "\n".join(
                f"{m.__class__.__name__}: {getattr(m, 'content', '')}"
                for m in messages
            )
            cfg      = SETTINGS["agent"]
            response = get_llm(cfg["address"], streaming=False).invoke([
                SystemMessage(content=SUMMARIZE_PROMPT),
                HumanMessage(content=prefix + history_text),
            ])
            new_summary = response.content if isinstance(response.content, str) else ""

            # Estimate tokens freed (rough character-based count)
            freed = sum(len(getattr(m, "content", "") or "") for m in messages) // 4

            self.graph.update_state(
                config=config,
                values={
                    "summary":  new_summary,
                    "messages": [RemoveMessage(id=m.id) for m in messages],
                },
            )
            q.put(("compact_done", freed))
        except Exception as exc:
            q.put(("error", f"Compact failed: {exc}"))

    def _start_monitor_thread(self):
        def _poll_system():
            while not self._stop_event.is_set():
                try:
                    with urllib.request.urlopen(MONITOR_URL, timeout=1) as resp:
                        raw = resp.read().decode()
                        try:
                            self._monitor_data = json.loads(raw)
                        except json.JSONDecodeError:
                            self._monitor_data = {"raw": raw[:200]}
                except Exception as exc:
                    self._monitor_data = {"error": str(exc)[:80]}
                self._stop_event.wait(1)

        def _poll_gpu():
            while not self._stop_event.is_set():
                try:
                    with urllib.request.urlopen(MONITOR_URL, timeout=1) as resp:
                        data = json.loads(resp.read().decode())
                        gpus = data.get("gpus", [])
                        if gpus:
                            self._gpu_history.append(gpus)
                except Exception:
                    pass
                self._stop_event.wait(0.5)

        def _poll_remote_system():
            while not self._stop_event.is_set():
                try:
                    with urllib.request.urlopen(REMOTE_MONITOR_URL, timeout=1) as resp:
                        raw = resp.read().decode()
                        try:
                            self._remote_monitor_data = json.loads(raw)
                        except json.JSONDecodeError:
                            self._remote_monitor_data = {"raw": raw[:200]}
                except Exception as exc:
                    self._remote_monitor_data = {"error": str(exc)[:80]}
                self._stop_event.wait(1)

        def _poll_remote_gpu():
            while not self._stop_event.is_set():
                try:
                    with urllib.request.urlopen(REMOTE_MONITOR_URL, timeout=1) as resp:
                        data = json.loads(resp.read().decode())
                        gpus = data.get("gpus", [])
                        if gpus:
                            self._remote_gpu_history.append(gpus)
                except Exception:
                    pass
                self._stop_event.wait(0.5)

        def _poll_remote_logs():
            while not self._stop_event.is_set():
                try:
                    with urllib.request.urlopen(REMOTE_LOGS_URL, timeout=2) as resp:
                        data = json.loads(resp.read().decode())
                        entries = data.get("lines", []) if isinstance(data, dict) else []
                        if entries:
                            last = entries[-1]
                            if isinstance(last, dict):
                                ts   = last.get("ts", "")
                                line = last.get("line", "")
                                t    = ts.split("T")[-1].split(".")[0] if "T" in ts else ts
                                self._remote_log_line = f"{t}  {line}"
                            else:
                                self._remote_log_line = str(last)[:200]
                except Exception:
                    pass
                self._stop_event.wait(2)

        threading.Thread(target=_poll_system,      daemon=True).start()
        threading.Thread(target=_poll_gpu,         daemon=True).start()
        threading.Thread(target=_poll_remote_system, daemon=True).start()
        threading.Thread(target=_poll_remote_gpu,    daemon=True).start()
        threading.Thread(target=_poll_remote_logs,   daemon=True).start()

    def _start_log_thread(self):
        def _poll():
            while self.thinking:
                try:
                    with urllib.request.urlopen(LOGS_URL, timeout=2) as resp:
                        data = json.loads(resp.read().decode())
                        entries = data.get("lines", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                        formatted = []
                        for entry in entries:
                            if isinstance(entry, dict):
                                ts   = entry.get("ts", "")
                                line = entry.get("line", "")
                                t    = ts.split("T")[-1].split(".")[0] if "T" in ts else ts
                                formatted.append(f"{t}  {line}")
                            else:
                                formatted.append(str(entry)[:200])
                        if formatted:
                            self._log_lines = formatted
                except Exception:
                    pass
                time.sleep(1)
        threading.Thread(target=_poll, daemon=True).start()

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _draw_menubar(self):
        _, w = self.stdscr.getmaxyx()
        bar = curses.color_pair(1)
        self.stdscr.attron(bar)
        try:
            self.stdscr.addstr(0, 0, " " * (w - 1))
        except curses.error:
            pass
        x = 1
        for key_name, label in MENU:
            item = f" {key_name} {label} "
            if x + len(item) < w:
                self.stdscr.addstr(0, x, item)
                x += len(item) + 1
        # Right-aligned indicators: mode badge then Lovense status
        mode_labels = {
            "auto":  " AUTO ",
            "agent": " AGNT ",
            "plan":  " PLAN ",
            "chat":  " CHAT ",
        }
        lv_label = " Lvns:[--] "
        try:
            import lovense as _lv
            lv_label = " Lvns:[ON] " if _lv.is_connected() else " Lvns:[--] "
        except Exception:
            pass
        cur_mode  = SETTINGS.get("mode", "auto")
        mode_text = mode_labels.get(cur_mode, f" {cur_mode.upper()[:4]} ")
        right_str = mode_text + "|" + lv_label
        right_col = max(0, w - len(right_str) - 1)
        self.stdscr.attroff(bar)
        mode_colors = {"auto": 0, "agent": curses.color_pair(5), "plan": curses.color_pair(6), "chat": curses.color_pair(3)}
        mode_attr = mode_colors.get(cur_mode, 0)
        try:
            self.stdscr.addstr(0, right_col, mode_text, bar | mode_attr)
            self.stdscr.addstr(0, right_col + len(mode_text), "|" + lv_label, bar)
        except curses.error:
            pass
        self.stdscr.attroff(bar)

    def _draw_home(self):
        h, w = self.stdscr.getmaxyx()
        lines = ["SakuraLang AI Terminal", "", "F1   Chat",
                 "F10  Help", "F12  Settings", "", "ESC  Quit"]
        start_y = h // 2 - len(lines) // 2
        for i, line in enumerate(lines):
            x = max(0, (w - len(line)) // 2)
            try:
                self.stdscr.addstr(start_y + i, x, line)
            except curses.error:
                pass

    def _draw_help(self):
        h, w  = self.stdscr.getmaxyx()
        bar   = curses.color_pair(1)
        pink  = curses.color_pair(2)

        def _hdr(y: int, title: str) -> None:
            self.stdscr.attron(bar)
            try:
                hdr = f" {title} "
                self.stdscr.addstr(y, 1, hdr)
                self.stdscr.addstr(y, 1 + len(hdr), " " * (w - 2 - len(hdr)))
            except curses.error:
                pass
            self.stdscr.attroff(bar)

        def _row(y: int, cmd: str, desc: str) -> None:
            col_w = max(26, w // 3)
            try:
                self.stdscr.addstr(y, 2, cmd[:col_w - 1].ljust(col_w), pink)
                self.stdscr.addstr(y, 2 + col_w, desc[:w - col_w - 4])
            except curses.error:
                pass

        HELP = [
            ("Navigation", [
                ("F1",                  "Open chat view"),
                ("F2",                  "Open / switch chat threads"),
                ("F3",                  "Cycle mode: Auto → Agent → Plan → Chat"),
                ("R (in Chats)",        "Regenerate the selected chat title"),
                ("F5",                  "Compact context (summarise history)"),
                ("F10",                 "This help screen"),
                ("F12",                 "Settings (agent, researcher, Brave, titler, Lovense)"),
                ("ESC",                 "Cancel stream / close view / quit"),
                ("PgUp / PgDn",         "Scroll chat history"),
                ("Mouse wheel",         "Scroll chat history"),
            ]),
            ("Select mode", [
                ("Alt+A",               "Enter select mode (anchor set at cursor)"),
                ("↑ ↓ PgUp PgDn",       "Move selection cursor"),
                ("Home / End",          "Jump to first / last line"),
                ("Ctrl+K",              "Copy selection to clipboard and exit"),
                ("ESC",                 "Exit without copying"),
            ]),
            ("Chat input", [
                ("Enter",               "Send message"),
                ("Shift+Enter / paste", "Insert newline (multi-line input)"),
                ("← → Home End",       "Move cursor in input field"),
                ("↑ ↓",                 "Move cursor between visual lines"),
                ("Auto-new chat",       "Each sent message starts a fresh thread"),
                ("Backspace / Del",     "Delete character"),
            ]),
            ("Slash commands", [
                ("/lovense status",     "Show Lovense toy connection status"),
                ("/lovense connect",    "Get a QR pairing URL (scan with Lovense Remote)"),
                ("/lovense test",       "Vibrate for 3 s to verify the connection"),
                ("/lovense stop",       "Stop toy immediately"),
                ("/lovense disconnect",  "Clear toy pairing state (re-pair with connect)"),
                ("/lovense debug",      "Dump raw Lovense connection state"),
            ]),
            ("AI pipeline", [
                ("Classifier",          "Routes every message: chat / RAG / clarify"),
                ("Agent (main AI)",     "Answers general queries; has tool access"),
                ("Researcher (Qwen)",   "Answers RAG queries grounded in context only"),
                ("Chat Titler",         "Generates short titles for completed chats"),
                ("RAG (Hybrid)",        "Dense + BM25 fusion, rerank, small-to-big"),
            ]),
        ]

        y = 1
        for section_title, rows in HELP:
            if y >= h - 2:
                break
            _hdr(y, section_title)
            y += 1
            for cmd, desc in rows:
                if y >= h - 2:
                    break
                _row(y, cmd, desc)
                y += 1
            y += 1   # blank line between sections

        try:
            self.stdscr.addstr(h - 1, 0, "  ESC  close help"[:w - 1])
        except curses.error:
            pass

    def _compute_visual_lines(self, text: str, field_w: int) -> list[tuple[int, str]]:
        """Return (char_start, display_text) for each word-wrapped visual line."""
        result = []
        paragraphs = text.split("\n")
        char_pos = 0
        for pi, para in enumerate(paragraphs):
            if not para:
                result.append((char_pos, ""))
            else:
                offset = 0
                remaining = para
                while len(remaining) > field_w:
                    split = remaining.rfind(" ", 0, field_w)
                    if split <= 0:
                        split = field_w
                    result.append((char_pos + offset, remaining[:split]))
                    skip = split + (1 if split < len(remaining) and remaining[split] == " " else 0)
                    offset += skip
                    remaining = remaining[skip:]
                result.append((char_pos + offset, remaining))
            char_pos += len(para)
            if pi < len(paragraphs) - 1:
                char_pos += 1
        return result

    def _cursor_to_visual(self, visual_lines: list, pos: int) -> tuple[int, int]:
        """Map a char-position to (visual_row, col) using precomputed visual lines."""
        for row in range(len(visual_lines) - 1):
            start_i    = visual_lines[row][0]
            start_next = visual_lines[row + 1][0]
            if start_i <= pos < start_next:
                return row, pos - start_i
        last = len(visual_lines) - 1
        return last, max(0, pos - visual_lines[last][0])

    def _draw_settings(self):
        h, w = self.stdscr.getmaxyx()
        bar  = curses.color_pair(1)

        LABEL_W = 18
        field_w = max(10, w - LABEL_W - 4)

        def _header(y: int, title: str) -> None:
            self.stdscr.attron(bar)
            try:
                hdr = f" {title} "
                self.stdscr.addstr(y, 1, hdr)
                self.stdscr.addstr(y, 1 + len(hdr), " " * (w - 2 - len(hdr)))
            except curses.error:
                pass
            self.stdscr.attroff(bar)

        y = 1
        _header(y, "Main Agent")
        y += 2
        self._draw_field(y, "  Address:       ", F_AGENT_ADDR,   LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  Working Dir:   ", F_AGENT_CWD,    LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  System Prompt: ", F_AGENT_PROMPT, LABEL_W, field_w, multiline=True)
        y += 4

        _header(y, "Researcher (Qwen / RAG)")
        y += 2
        self._draw_field(y, "  Address:       ", F_RESEARCHER_ADDR,   LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  System Prompt: ", F_RESEARCHER_PROMPT, LABEL_W, field_w, multiline=True)
        y += 4

        _header(y, "Brave Search")
        y += 2
        self._draw_field(y, "  API Key:       ", F_BRAVE_API_KEY,     LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  Base URL:      ", F_BRAVE_BASE_URL,    LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  Result Count:  ", F_BRAVE_COUNT,       LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  Country:       ", F_BRAVE_COUNTRY,     LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  Search Lang:   ", F_BRAVE_SEARCH_LANG, LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  SafeSearch:    ", F_BRAVE_SAFESEARCH,  LABEL_W, field_w, multiline=False)
        y += 4

        _header(y, "Classifier")
        y += 2
        self._draw_field(y, "  Address:       ", F_CLASSIFIER_ADDR,   LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  System Prompt: ", F_CLASSIFIER_PROMPT, LABEL_W, field_w, multiline=True)
        y += 4

        _header(y, "Chat Titler")
        y += 2
        self._draw_field(y, "  Address:       ", F_TITLER_ADDR,   LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  System Prompt: ", F_TITLER_PROMPT, LABEL_W, field_w, multiline=True)
        y += 4

        # Enter your developer token from https://www.lovense.com/user/developer/info
        # and a user ID string (any stable identifier you choose).
        # The callback port is where this app listens for the pairing POST from
        # Lovense Remote after the user scans the QR code.
        # Callback Host is your LAN IP — leave blank to auto-detect.
        _header(y, "Lovense (toy control)")
        y += 2
        self._draw_field(y, "  Dev Token:     ", F_LOVENSE_TOKEN, LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  User ID:       ", F_LOVENSE_UID,   LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  Callback Port: ", F_LOVENSE_PORT,  LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  Callback Host: ", F_LOVENSE_HOST,  LABEL_W, field_w, multiline=False)
        y += 2
        # TLS cert files from certbot (certonly --standalone -d bzz.sadgirlsclub.wtf).
        # Default certbot paths: C:\Certbot\live\<domain>\fullchain.pem + privkey.pem
        self._draw_field(y, "  TLS Cert File: ", F_LOVENSE_CERT,  LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  TLS Key File:  ", F_LOVENSE_KEY,   LABEL_W, field_w, multiline=False)
        y += 2

        footer = "  Tab: field   ↑↓ (prompt): line / (addr): field   ←→/Home/End: cursor   Enter: newline   ESC: save"
        try:
            self.stdscr.addstr(h - 1, 0, footer[:w - 1])
        except curses.error:
            pass

    def _draw_field(self, y: int, label: str, field_idx: int,
                    label_w: int, field_w: int, multiline: bool) -> None:
        active = (self.settings_focus == field_idx)
        val    = self.settings_bufs[field_idx]
        pos    = self.settings_cursors[field_idx] if active else 0

        try:
            self.stdscr.addstr(y, 1, label[:label_w])
        except curses.error:
            pass

        if not multiline:
            scroll  = max(0, pos - field_w + 1)
            view    = val[scroll:scroll + field_w]
            cur_col = pos - scroll
            self._render_line(y, 1 + label_w, view, cur_col if active else -1,
                              field_w, active)
        else:
            vlines = self._compute_visual_lines(val, field_w)
            if active:
                cur_row, cur_col_abs = self._cursor_to_visual(vlines, pos)
                v_start = max(0, cur_row - 2)
            else:
                cur_row = cur_col_abs = 0
                v_start = 0
            pad     = (0, "")
            visible = (vlines + [pad, pad])[v_start:v_start + 3]
            cont_label = " " * label_w
            for li, (_, line) in enumerate(visible):
                lbl      = label[:label_w] if li == 0 else cont_label
                abs_line = v_start + li
                in_cursor_line = active and (abs_line == cur_row)
                col = cur_col_abs if in_cursor_line else -1
                try:
                    self.stdscr.addstr(y + li, 1, lbl)
                except curses.error:
                    pass
                self._render_line(y + li, 1 + label_w, line, col,
                                  field_w, active)

    def _render_line(self, y: int, x: int, text: str, cur_col: int,
                     field_w: int, active: bool) -> None:
        """Draw one field line, painting the cursor character with A_REVERSE."""
        bg = curses.A_DIM if active else 0
        padded = text[:field_w].ljust(field_w)
        try:
            self.stdscr.addstr(y, x, padded, bg)
        except curses.error:
            pass
        if active and 0 <= cur_col <= len(text):
            col  = min(cur_col, field_w - 1)
            try:
                self.stdscr.addstr(y, x + col, "🌸")
            except curses.error:
                pass

    def _draw_chat(self):
        h, w      = self.stdscr.getmaxyx()
        field_w   = w - 3  # input always spans full width
        INPUT_H   = CHAT_INPUT_MIN_H if self.thinking else self._chat_input_height(field_w, h)
        sep_row   = h - INPUT_H - 1
        input_top = h - INPUT_H
        chat_top  = 1
        chat_h    = sep_row - chat_top

        # Right quarter = monitor panel; left = chat
        show_panel = (w >= 60)
        if show_panel:
            panel_w  = max(20, w // 4)
            vsep_col = w - panel_w - 1
            chat_x   = 0
            chat_w   = max(1, vsep_col - 1)
        else:
            vsep_col = 0
            chat_x   = 0
            chat_w   = w - 2

        if show_panel:
            self._draw_monitor_panel(chat_top, sep_row, panel_w, panel_x=vsep_col + 1)
            for row in range(chat_top, sep_row):
                try:
                    self.stdscr.addch(row, vsep_col, ord('|'), curses.color_pair(4))
                except curses.error:
                    pass

        # Chat history
        lines = []
        for role, content in self.history:
            pair   = curses.color_pair(ROLE_PAIR.get(role, 0))
            prefix = ROLE_PREFIX.get(role, "")
            for wrapped in self._wrap(prefix + content, chat_w):
                lines.append((pair, wrapped))

        self._select_lines = lines   # cache for select-mode key handler

        total      = len(lines)
        max_scroll = max(0, total - chat_h)
        self.chat_scroll = min(self.chat_scroll, max_scroll)
        end     = total - self.chat_scroll
        start   = max(0, end - chat_h)
        visible = lines[start:end]

        # Horizontal separator (full width)
        self.stdscr.attron(curses.color_pair(1))
        try:
            if self._select_mode:
                anchor = self._select_anchor if self._select_anchor is not None else self._select_cursor
                n_sel  = abs(self._select_cursor - anchor) + 1
                note   = f" SELECT {n_sel}ln | ^K:copy ESC:exit "
            elif self.chat_scroll > 0:
                note = f" ↑ {self.chat_scroll} "
            else:
                note = ""
            sep = ("-" * max(0, w - 1 - len(note))) + note if note else "-" * (w - 1)
            self.stdscr.addstr(sep_row, 0, sep[:w - 1])
        except curses.error:
            pass
        self.stdscr.attroff(curses.color_pair(1))

        for i, (pair, line) in enumerate(visible):
            y = chat_top + i
            if y < sep_row:
                try:
                    self.stdscr.addstr(y, chat_x, line, pair)
                except curses.error:
                    pass

        # Selection highlight overlay
        if self._select_mode and lines:
            anchor = self._select_anchor if self._select_anchor is not None else self._select_cursor
            sel_lo = min(anchor, self._select_cursor)
            sel_hi = max(anchor, self._select_cursor)
            for i, (pair, line) in enumerate(visible):
                abs_idx = start + i
                if sel_lo <= abs_idx <= sel_hi:
                    y = chat_top + i
                    if y < sep_row:
                        attr = curses.A_REVERSE | (curses.A_BOLD if abs_idx == self._select_cursor else 0)
                        try:
                            self.stdscr.addstr(y, chat_x, line[:chat_w].ljust(chat_w), pair | attr)
                        except curses.error:
                            pass

        if self.thinking:
            local_slots = INPUT_H - 2  # reserve 1 row for remote log, 1 for ESC hint
            log_display = self._log_lines[-local_slots:] if self._log_lines else ["thinking..."]
            for li, log_text in enumerate(log_display):
                try:
                    self.stdscr.addstr(input_top + li, 0, f"  {log_text.strip()}"[:w - 1])
                except curses.error:
                    pass
            if self._remote_log_line:
                try:
                    self.stdscr.addstr(input_top + INPUT_H - 2, 0,
                                       f"  {self._remote_log_line.strip()}"[:w - 1],
                                       curses.color_pair(4))
                except curses.error:
                    pass
            hint = "  ESC: stop"
            try:
                self.stdscr.addstr(input_top + INPUT_H - 1, 0, hint[:w - 1], curses.color_pair(4))
            except curses.error:
                pass
            return

        vlines           = self._compute_visual_lines(self.input_buf, field_w)
        cur_row, cur_col = self._cursor_to_visual(vlines, self.input_cursor)
        v_start          = max(0, cur_row - (INPUT_H - 1))
        pad              = (0, "")
        visible_in       = (vlines + [pad, pad])[v_start:v_start + INPUT_H]

        for li, (_, line) in enumerate(visible_in):
            row_y  = input_top + li
            prefix = "> " if li == 0 else "  "
            try:
                self.stdscr.addstr(row_y, 0, prefix)
            except curses.error:
                pass
            abs_vline = v_start + li
            col = cur_col if (abs_vline == cur_row) else -1
            self._render_line(row_y, 2, line, col, field_w, True)

        vis_li = cur_row - v_start
        if 0 <= vis_li < INPUT_H:
            self.stdscr.move(input_top + vis_li, min(2 + cur_col, w - 1))

    def _draw_monitor_panel(self, top: int, bottom: int, panel_w: int, panel_x: int = 1) -> None:
        pair    = curses.color_pair(4)
        avail_w = panel_w - 2
        # Split the panel vertically: top 2/3 = system stats, bottom 1/3 = pipeline log
        mid     = top + (bottom - top) * 2 // 3
        quarter = top + (mid - top) // 2   # divides stats area between local and remote

        # ── Top quarter: local system stats ───────────────────────────────
        title = "[ Monitor ]"
        try:
            self.stdscr.addstr(top, panel_x, title[:avail_w], pair | curses.A_BOLD)
        except curses.error:
            pass

        data = self._monitor_data
        row  = top + 1

        if not data:
            try:
                self.stdscr.addstr(row, panel_x, "connecting..."[:avail_w], pair)
            except curses.error:
                pass
        else:
            for line in self._format_monitor_lines(data, avail_w):
                if row >= quarter:
                    break
                try:
                    self.stdscr.addstr(row, panel_x, line, pair)
                except curses.error:
                    pass
                row += 1

        # ── Mid-stats divider + remote stats ──────────────────────────────
        try:
            self.stdscr.addstr(quarter, panel_x, ("─" * avail_w)[:avail_w], pair)
        except curses.error:
            pass

        rdata = self._remote_monitor_data
        rrow  = quarter + 1
        if not rdata:
            try:
                self.stdscr.addstr(rrow, panel_x, "remote: ..."[:avail_w], pair)
            except curses.error:
                pass
        else:
            for line in self._format_monitor_lines(rdata, avail_w, gpu_history=self._remote_gpu_history):
                if rrow >= mid:
                    break
                try:
                    self.stdscr.addstr(rrow, panel_x, line, pair)
                except curses.error:
                    pass
                rrow += 1

        # ── Divider ────────────────────────────────────────────────────────
        try:
            self.stdscr.addstr(mid, panel_x, ("─" * avail_w)[:avail_w], pair)
        except curses.error:
            pass

        # ── Bottom half: pipeline log ──────────────────────────────────────
        log_title = "[ Pipeline ]"
        log_row   = mid + 1
        try:
            self.stdscr.addstr(log_row, panel_x, log_title[:avail_w], pair | curses.A_BOLD)
        except curses.error:
            pass
        log_row += 1

        # Reserve the very last row for the context-window bar; shrink log area
        ctx_row   = bottom - 1
        log_avail = ctx_row - log_row   # rows available for timestamped events

        def _wrap_log_entry(entry: str, width: int) -> list[str]:
            """Word-wrap a pipeline log entry.  Continuation lines are indented
            9 chars (= len('HH:MM:SS ')) so the message text stays aligned."""
            INDENT = " " * 9
            out    = []
            while len(entry) > width:
                split = entry.rfind(" ", 0, width)
                if split <= 0:
                    split = width
                out.append(entry[:split])
                entry = INDENT + entry[split:].lstrip()
            out.append(entry)
            return out

        # Flatten all entries into wrapped display lines, then show the tail
        all_lines: list[str] = []
        for entry in self._pipeline_log:
            all_lines.extend(_wrap_log_entry(entry, avail_w))

        visible = all_lines[-log_avail:] if log_avail > 0 else []
        for line in visible:
            if log_row >= ctx_row:
                break
            try:
                self.stdscr.addstr(log_row, panel_x, line, pair)
            except curses.error:
                pass
            log_row += 1

        # ── Context-window bar ─────────────────────────────────────────────
        # Represents Mochi's 100 k-token context window as a row of block chars.
        # █ = used  ░ = free   Color: lime < 50 % | yellow 50-80 % | pink ≥ 80 %
        CTX_MAX  = 100_000
        inp_tok  = (self._last_token_usage or {}).get("input_tokens", 0)
        out_tok  = (self._last_token_usage or {}).get("output_tokens", 0)
        fill_pct = min(1.0, inp_tok / CTX_MAX)
        bar_w    = avail_w
        filled   = int(fill_pct * bar_w)
        empty    = bar_w - filled
        bar_chars = list("█" * filled + "░" * empty)

        # Embed a centered label so the count is always readable.
        if inp_tok > 0:
            pct_int = int(fill_pct * 100)
            label = f" {inp_tok:,}/{CTX_MAX // 1000}k ({pct_int}%) +{out_tok:,}out "
        else:
            label = " ctx: waiting... "
        if len(label) < bar_w:
            lstart = (bar_w - len(label)) // 2
            for i, ch in enumerate(label):
                if lstart + i < bar_w:
                    bar_chars[lstart + i] = ch
        bar_str = "".join(bar_chars)

        if   fill_pct >= 0.80:
            bar_pair = curses.color_pair(2)   # pink  — getting full
        elif fill_pct >= 0.50:
            bar_pair = curses.color_pair(6)   # yellow — halfway
        else:
            bar_pair = curses.color_pair(5)   # lime  — plenty of room

        try:
            self.stdscr.addstr(ctx_row, panel_x, bar_str[:avail_w], bar_pair)
        except curses.error:
            pass

    def _avg_gpu(self, gpu_idx: int, key: str, history=None) -> float | None:
        hist = history if history is not None else self._gpu_history
        samples = [
            snap[gpu_idx].get(key)
            for snap in hist
            if gpu_idx < len(snap) and snap[gpu_idx].get(key) is not None
        ]
        return sum(samples) / len(samples) if samples else None

    def _format_monitor_lines(self, data: dict | list | str, width: int, gpu_history=None) -> list[str]:
        if not isinstance(data, dict):
            if isinstance(data, list):
                return [str(x)[:width] for x in data]
            return [p[:width] for p in str(data).split("\n")]

        lines = []

        # Timestamp
        ts = data.get("updated_at", "")
        if ts:
            time_part = str(ts).split("T")[-1] if "T" in str(ts) else str(ts)
            lines.append(f"@ {time_part}"[:width])

        # System
        sys = data.get("system", {})
        if sys:
            lines.append("- system"[:width])
            cpu = sys.get("cpu_percent")
            if cpu is not None:
                lines.append(f"  CPU   {cpu:.1f}%"[:width])
            ram_u = sys.get("ram_used_gib")
            ram_t = sys.get("ram_total_gib")
            ram_p = sys.get("ram_percent")
            if ram_u is not None and ram_t is not None:
                lines.append(f"  RAM   {ram_u:.1f}/{ram_t:.1f} GiB"[:width])
            if ram_p is not None:
                lines.append(f"        {ram_p:.1f}%"[:width])

        # GPUs — VRAM from latest snapshot, loads averaged over last 10s
        gpus = data.get("gpus", [])
        if not gpus and self._gpu_history:
            gpus = self._gpu_history[-1]
        if gpus:
            lines.append("- gpu"[:width])
            for i, gpu in enumerate(gpus):
                name = gpu.get("name", f"GPU {i}")
                short = (name
                         .replace("AMD Radeon ", "")
                         .replace("NVIDIA GeForce ", "")
                         .replace("NVIDIA ", ""))
                lines.append(f"  [{i}] {short}"[:width])

                vram_u = gpu.get("vram_used_mib")
                vram_t = gpu.get("vram_total_mib")
                vram_p = gpu.get("vram_percent")
                if vram_u is not None and vram_t is not None:
                    lines.append(f"  VRAM  {vram_u}/{vram_t} MiB"[:width])
                if vram_p is not None:
                    lines.append(f"        {vram_p:.1f}%"[:width])

                util  = self._avg_gpu(i, "util_percent",   history=gpu_history)
                power = self._avg_gpu(i, "power_watts",    history=gpu_history)
                temp  = self._avg_gpu(i, "temperature_c",  history=gpu_history)
                clock = self._avg_gpu(i, "core_clock_mhz", history=gpu_history)
                parts = []
                if util  is not None: parts.append(f"{util:.0f}%")
                if power is not None: parts.append(f"{power:.0f}W")
                if temp  is not None: parts.append(f"{temp:.0f}°C")
                if parts:
                    lines.append(f"  {' '.join(parts)}"[:width])
                if clock is not None:
                    lines.append(f"  {clock:.0f} MHz"[:width])

        # Context token usage from last LLM call
        usage = self._last_token_usage
        if usage:
            inp  = usage.get("input_tokens",  0)
            out  = usage.get("output_tokens", 0)
            lines.append("- context"[:width])
            lines.append(f"  in:  {inp:,}"[:width])
            lines.append(f"  out: {out:,}"[:width])

        return lines

    @staticmethod
    def _wrap(text, width):
        text = _clip_display_text(text, CHAT_RENDER_MAX_CHARS, "chat entry")
        result = []
        for paragraph in text.split("\n"):
            if not paragraph:
                result.append("")
                continue
            start = 0
            plen = len(paragraph)
            while (plen - start) > width:
                split = paragraph.rfind(" ", start, start + width)
                if split <= 0:
                    split = start + width
                result.append(paragraph[start:split])
                start = split
                while start < plen and paragraph[start].isspace():
                    start += 1
            result.append(paragraph[start:])
        return result


def main():
    with SqliteSaver.from_conn_string("chat_history.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        def _run(stdscr):
            App(stdscr, graph).run()

        curses.wrapper(_run)
