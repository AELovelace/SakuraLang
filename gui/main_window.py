# -*- coding: utf-8 -*-
"""Main application window for SakuraLang GUI."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import urllib.request

from PyQt6.QtCore import QEvent, QObject, QThread, QTimer, Qt
from PyQt6.QtGui import QAction, QCloseEvent, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import approval
from prompts import LOGS_URL, REMOTE_LOGS_URL
from settings import SETTINGS, save_settings

from .chat_worker import StreamWorker, CompactWorker, TitleWorker
from .tools_dialog import ToolApprovalDialog
from .settings_dialog import SettingsDialog
from .chat_list_dialog import ChatListDialog
from .monitor_widget import MonitorWidget

# Role → (display label, accent colour, bubble colour, border colour)
_ROLE_STYLES: dict[str, tuple[str, str, str, str]] = {
    "user":    ("You",       "#a83255", "#fce4ec", "#f48fb1"),
    "ai":      ("Sakura",   "#2a5f96", "#e3f2fd", "#90caf9"),
    "tool":    ("Tool",      "#b45309", "#fef3c7", "#fcd34d"),
    "rag":     ("Documents", "#2d7045", "#e8f5e9", "#a5d6a7"),
    "router":  ("System",    "#6a4a72", "#ede6ee", "#c8b0d0"),
}

_MODES = ["auto", "agent", "plan", "chat"]
_MODE_LABELS = {"auto": "AUTO", "agent": "AGENT", "plan": "PLAN", "chat": "CHAT"}


class _ComposerKeyFilter(QObject):
    """Intercepts key events on the composer so Enter sends and Shift+Enter inserts a newline."""

    def __init__(self, send_callback, parent=None):
        super().__init__(parent)
        self._send = send_callback

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if isinstance(event, QKeyEvent) and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False  # Shift+Enter: let composer insert newline
                self._send()
                return True  # consume bare Enter
        return False


class MainWindow(QMainWindow):
    def __init__(self, graph, db_path: str = "chat_history.db") -> None:
        super().__init__()
        self.graph = graph
        self.db_path = db_path

        # Chat state
        self.thread_id: str = self._new_thread_id()
        self.history: list[tuple[str, str]] = []
        self._ai_idx: int | None = None
        self.thinking = False
        self._last_token_usage: dict = {}

        # Worker handles
        self._stream_thread: QThread | None = None
        self._stream_worker: StreamWorker | None = None
        self._cancel_event = threading.Event()

        # Server inference log lines (populated by background polling threads)
        self._agent_log_lines:      list[str] = []
        self._researcher_log_lines: list[str] = []
        self._log_stop = threading.Event()

        self.setWindowTitle("SakuraLang")
        self.resize(1280, 800)
        self._build_ui()
        self._apply_theme()
        self._load_initial_state()
        self._start_log_polling()

    # ── Initial state ─────────────────────────────────────────────────────

    def _load_initial_state(self):
        self._ensure_chat_titles_table()
        threads = self._list_threads()
        if threads:
            self.thread_id = threads[0]["thread_id"]
            self._load_thread_history(self.thread_id)
        else:
            self.thread_id = self._new_thread_id()
        self._refresh_thread_list()
        self._refresh_status()

    def _new_thread_id(self) -> str:
        return f"chat_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QToolBar("Actions", self)
        toolbar.setMovable(False)
        toolbar.setObjectName("mainToolbar")
        self.addToolBar(toolbar)

        new_action = QAction("New Chat", self)
        new_action.triggered.connect(self.create_new_chat)
        toolbar.addAction(new_action)

        chats_action = QAction("Chat List", self)
        chats_action.triggered.connect(self.open_chat_list)
        toolbar.addAction(chats_action)

        compact_action = QAction("Compact", self)
        compact_action.triggered.connect(self.compact_context)
        toolbar.addAction(compact_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        toolbar.addAction(settings_action)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Mode: "))
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("modeCombo")
        for m in _MODES:
            self.mode_combo.addItem(_MODE_LABELS[m], m)
        cur_mode = SETTINGS.get("mode", "auto")
        idx = _MODES.index(cur_mode) if cur_mode in _MODES else 0
        self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.mode_combo)

        # Cancel button shown only while streaming
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("cancelButton")
        self._cancel_btn.clicked.connect(self._cancel_stream)
        self._cancel_btn.setVisible(False)
        toolbar.addWidget(self._cancel_btn)

        # ── Main splitter: sidebar | chat+monitor ─────────────────────────
        outer_splitter = QSplitter(self)
        outer_splitter.setChildrenCollapsible(False)
        outer_splitter.setObjectName("mainSplitter")
        outer_splitter.setHandleWidth(8)

        # Left sidebar — thread list
        sidebar = QWidget(outer_splitter)
        sidebar.setObjectName("sidebarPanel")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(8)

        self._new_btn = QPushButton("New Chat")
        self._new_btn.setObjectName("newChatButton")
        self._new_btn.clicked.connect(self.create_new_chat)
        sidebar_layout.addWidget(self._new_btn)

        self._thread_list = QListWidget()
        self._thread_list.setObjectName("conversationList")
        self._thread_list.itemClicked.connect(self._on_thread_clicked)
        sidebar_layout.addWidget(self._thread_list, 1)

        # Centre — chat area + composer
        chat_panel = QWidget()
        chat_panel.setObjectName("chatAreaFrame")
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(16, 16, 16, 16)
        chat_layout.setSpacing(10)

        self._mode_label = QLabel()
        self._mode_label.setObjectName("statusSummaryCard")
        chat_layout.addWidget(self._mode_label)

        self._token_bar = QLabel("Tokens: —")
        self._token_bar.setObjectName("tokenBar")
        self._token_bar.setAlignment(Qt.AlignmentFlag.AlignRight)
        chat_layout.addWidget(self._token_bar)

        self._transcript = QTextBrowser()
        self._transcript.setObjectName("chatTranscript")
        self._transcript.setOpenExternalLinks(False)
        self._transcript.setFrameShape(QFrame.Shape.NoFrame)
        self._transcript.document().setDocumentMargin(8)
        chat_layout.addWidget(self._transcript, 1)

        composer_row = QHBoxLayout()
        composer_row.setSpacing(10)
        self._composer = QPlainTextEdit()
        self._composer.setObjectName("composerInput")
        self._composer.setPlaceholderText("Type a message… (Enter to send, Shift+Enter for newline)")
        self._composer.setFixedHeight(120)
        composer_row.addWidget(self._composer, 1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendButton")
        self._send_btn.clicked.connect(self._send)
        btn_col.addWidget(self._send_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.clicked.connect(self._cancel_stream)
        self._stop_btn.setVisible(False)
        btn_col.addWidget(self._stop_btn)

        composer_row.addLayout(btn_col)
        chat_layout.addLayout(composer_row)

        # ── Inference log panels (always visible, below composer) ─────────
        log_row = QHBoxLayout()
        log_row.setSpacing(8)

        agent_col = QVBoxLayout()
        agent_col.setSpacing(2)
        agent_col.addWidget(self._make_log_header("Agent log  (100.66.64.45)"))
        self._agent_log_browser = QTextBrowser()
        self._agent_log_browser.setObjectName("inferenceLog")
        self._agent_log_browser.setOpenExternalLinks(False)
        self._agent_log_browser.setFixedHeight(80)
        agent_col.addWidget(self._agent_log_browser)

        researcher_col = QVBoxLayout()
        researcher_col.setSpacing(2)
        researcher_col.addWidget(self._make_log_header("Researcher log  (100.83.3.32)"))
        self._researcher_log_browser = QTextBrowser()
        self._researcher_log_browser.setObjectName("inferenceLog")
        self._researcher_log_browser.setOpenExternalLinks(False)
        self._researcher_log_browser.setFixedHeight(80)
        researcher_col.addWidget(self._researcher_log_browser)

        log_row.addLayout(agent_col, 1)
        log_row.addLayout(researcher_col, 1)
        chat_layout.addLayout(log_row)

        # Timer that reads cached log data and refreshes the two browsers
        self._log_ui_timer = QTimer(self)
        self._log_ui_timer.timeout.connect(self._refresh_log_ui)
        self._log_ui_timer.start(1000)

        # Right — monitor panel
        self._monitor = MonitorWidget()
        self._monitor.setMinimumWidth(220)
        self._monitor.setMaximumWidth(340)

        # Inner splitter: chat area | monitor
        inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        inner_splitter.setChildrenCollapsible(True)
        inner_splitter.setHandleWidth(6)
        inner_splitter.addWidget(chat_panel)
        inner_splitter.addWidget(self._monitor)
        inner_splitter.setSizes([900, 260])

        outer_splitter.addWidget(sidebar)
        outer_splitter.addWidget(inner_splitter)
        outer_splitter.setSizes([260, 1020])
        self.setCentralWidget(outer_splitter)

        status_bar = QStatusBar(self)
        status_bar.setObjectName("mainStatusBar")
        self.setStatusBar(status_bar)

        # Enter sends; Shift+Enter inserts a newline
        self._key_filter = _ComposerKeyFilter(self._send, self)
        self._composer.installEventFilter(self._key_filter)

    @staticmethod
    def _make_log_header(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("inferenceLogHeader")
        return lbl

    # ── Server log polling ────────────────────────────────────────────────

    def _start_log_polling(self):
        """Poll both /api/sakura/logs endpoints in background daemon threads."""
        def _poll(url: str, attr: str):
            while not self._log_stop.is_set():
                try:
                    with urllib.request.urlopen(url, timeout=2) as resp:
                        data = json.loads(resp.read().decode())
                    entries = (
                        data.get("lines", []) if isinstance(data, dict)
                        else (data if isinstance(data, list) else [])
                    )
                    lines = []
                    for entry in entries:
                        if isinstance(entry, dict):
                            ts   = entry.get("ts", "")
                            line = entry.get("line", "")
                            t    = ts.split("T")[-1].split(".")[0] if "T" in ts else ts
                            lines.append(f"{t}  {line}" if t else line)
                        else:
                            lines.append(str(entry)[:200])
                    setattr(self, attr, lines)
                except Exception:
                    pass
                self._log_stop.wait(1)

        for url, attr in (
            (LOGS_URL,        "_agent_log_lines"),
            (REMOTE_LOGS_URL, "_researcher_log_lines"),
        ):
            t = threading.Thread(target=_poll, args=(url, attr), daemon=True)
            t.start()

    def _refresh_log_ui(self):
        """Push the latest cached log lines into the two QTextBrowsers."""
        def _render(lines: list[str], browser: QTextBrowser):
            html = "".join(
                f'<div style="font-size:10px;color:#6a5060;'
                f'font-family:Consolas,monospace;white-space:pre;">'
                f'{line.replace("&","&amp;").replace("<","&lt;")}</div>'
                for line in lines[-20:]
            )
            browser.setHtml(html)
            sb = browser.verticalScrollBar()
            sb.setValue(sb.maximum())

        _render(self._agent_log_lines,      self._agent_log_browser)
        _render(self._researcher_log_lines, self._researcher_log_browser)

    # ── Theme ─────────────────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8e9ee;
                color: #5e3f49;
            }
            QToolBar#mainToolbar {
                spacing: 8px;
                padding: 8px 12px;
                background: rgba(255, 248, 250, 0.96);
                border: none;
                border-bottom: 1px solid rgba(193, 142, 156, 0.35);
            }
            QToolBar#mainToolbar QToolButton {
                color: #764d59;
                background: rgba(255, 233, 239, 0.94);
                border: 1px solid rgba(193, 142, 156, 0.55);
                border-radius: 14px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QToolBar#mainToolbar QToolButton:hover {
                background: rgba(255, 245, 248, 0.98);
            }
            QComboBox#modeCombo {
                background: rgba(255, 233, 239, 0.94);
                color: #764d59;
                border: 1px solid rgba(193, 142, 156, 0.55);
                border-radius: 10px;
                padding: 4px 8px;
                font-weight: 700;
                min-width: 80px;
            }
            QPushButton#cancelButton {
                background: rgba(220, 100, 100, 0.85);
                color: #fff;
                border-radius: 12px;
                padding: 5px 14px;
                font-weight: 700;
            }
            QPushButton#cancelButton:hover {
                background: rgba(200, 60, 60, 0.95);
            }
            QPushButton#stopButton {
                background: rgba(220, 100, 100, 0.85);
                color: #fff;
                border-radius: 16px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton#stopButton:hover {
                background: rgba(200, 60, 60, 0.95);
            }
            QSplitter#mainSplitter::handle {
                background: rgba(206, 166, 177, 0.24);
            }
            QWidget#sidebarPanel {
                background: rgba(255, 248, 250, 0.9);
                border-right: 1px solid rgba(193, 142, 156, 0.28);
            }
            QPushButton#newChatButton, QPushButton#sendButton {
                background-color: rgba(191, 118, 141, 0.92);
                color: #fff8fb;
                border: 1px solid rgba(129, 76, 92, 0.35);
                border-radius: 16px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton#newChatButton:hover, QPushButton#sendButton:hover {
                background-color: rgba(177, 104, 128, 0.98);
            }
            QPushButton#newChatButton:disabled, QPushButton#sendButton:disabled {
                background-color: rgba(191, 118, 141, 0.45);
                color: rgba(255, 248, 250, 0.75);
            }
            QListWidget#conversationList {
                background: rgba(255, 252, 253, 0.82);
                border: 1px solid rgba(193, 142, 156, 0.24);
                border-radius: 16px;
                padding: 6px;
                outline: none;
                color: #664852;
            }
            QListWidget#conversationList::item {
                padding: 4px 6px;
                margin: 2px 0;
                border-radius: 10px;
            }
            QListWidget#conversationList::item:selected {
                background: rgba(239, 200, 211, 0.95);
                color: #5a3944;
            }
            QWidget#chatAreaFrame {
                background-color: #f8e9ee;
            }
            QLabel#statusSummaryCard {
                background: rgba(255, 248, 250, 0.78);
                color: #603f49;
                border: 1px solid rgba(193, 142, 156, 0.25);
                border-radius: 16px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#tokenBar {
                color: #8a6070;
                font-size: 11px;
                font-weight: 600;
                padding: 2px 6px;
            }
            QTextBrowser#chatTranscript {
                background: transparent;
                color: #4c3139;
                border: none;
                selection-background-color: rgba(191, 118, 141, 0.35);
            }
            QPlainTextEdit#composerInput {
                background: rgba(255, 251, 252, 0.85);
                color: #4c3139;
                border: 1px solid rgba(193, 142, 156, 0.28);
                border-radius: 18px;
                padding: 10px 14px;
                selection-background-color: rgba(191, 118, 141, 0.35);
            }
            QScrollBar:vertical {
                background: rgba(255, 246, 248, 0.55);
                width: 10px;
                margin: 8px 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(191, 118, 141, 0.65);
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent; height: 0px;
            }
            QStatusBar#mainStatusBar {
                background: rgba(255, 248, 250, 0.96);
                color: #734a56;
                border-top: 1px solid rgba(193, 142, 156, 0.28);
            }
            QWidget#monitorPanel {
                background: rgba(252, 245, 248, 0.88);
                border-left: 1px solid rgba(193, 142, 156, 0.22);
            }
            QLabel#monitorHeader {
                color: #a87888;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                padding: 4px 0 2px 0;
            }
            QLabel#monitorStats {
                color: #6a5060;
                font-size: 11px;
                font-family: Consolas, monospace;
            }
            QProgressBar#contextBar {
                border: 1px solid rgba(193, 142, 156, 0.3);
                border-radius: 4px;
                background: rgba(240, 220, 228, 0.6);
            }
            QProgressBar#contextBar::chunk {
                background: rgba(191, 118, 141, 0.75);
                border-radius: 3px;
            }
            QTextBrowser#pipelineLog {
                background: transparent;
                border: none;
                color: #7a6070;
                font-size: 11px;
                font-family: Consolas, monospace;
            }
            QLabel#inferenceLogHeader {
                color: #a87888;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                padding: 2px 0 1px 0;
            }
            QTextBrowser#inferenceLog {
                background: rgba(255, 248, 250, 0.7);
                border: 1px solid rgba(193, 142, 156, 0.22);
                border-radius: 8px;
                color: #6a5060;
                font-size: 10px;
                font-family: Consolas, monospace;
                padding: 3px 6px;
            }
        """)

    # ── Mode ──────────────────────────────────────────────────────────────

    def _on_mode_changed(self, index: int):
        mode = self.mode_combo.itemData(index)
        SETTINGS["mode"] = mode
        save_settings(SETTINGS)
        self._refresh_status()

    def _refresh_status(self):
        mode = SETTINGS.get("mode", "auto")
        agent_addr = SETTINGS.get("agent", {}).get("address", "—")
        self._mode_label.setText(
            f"Mode: {_MODE_LABELS.get(mode, mode.upper())}  |  "
            f"Agent: {agent_addr}  |  Thread: {self.thread_id}"
        )
        self.statusBar().showMessage("Ready.")

    # ── Thread / conversation management ──────────────────────────────────

    def _ensure_chat_titles_table(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_titles (
                    thread_id  TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _list_threads(self) -> list[dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT thread_id, COUNT(*) FROM checkpoints "
                    "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
                )
                rows = list(cur.fetchall())
            except sqlite3.OperationalError:
                rows = []
            titles: dict[str, str] = {}
            try:
                cur.execute("SELECT thread_id, title FROM chat_titles")
                titles = dict(cur.fetchall())
            except sqlite3.OperationalError:
                pass
            conn.close()
            return [
                {"thread_id": tid, "count": cnt, "title": titles.get(tid, "")}
                for tid, cnt in rows
            ]
        except Exception:
            return []

    def _get_title(self, thread_id: str) -> str | None:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT title FROM chat_titles WHERE thread_id = ?", (thread_id,))
            row = cur.fetchone()
            conn.close()
            return str(row[0]).strip() if row else None
        except Exception:
            return None

    def _save_title(self, thread_id: str, title: str):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO chat_titles (thread_id, title, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(thread_id) DO UPDATE SET title=excluded.title, "
                "updated_at=CURRENT_TIMESTAMP",
                (thread_id, title),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _refresh_thread_list(self):
        threads = self._list_threads()
        self._thread_list.blockSignals(True)
        self._thread_list.clear()
        for entry in threads:
            tid = entry["thread_id"]
            title = entry.get("title") or (
                "New chat" if entry.get("count", 0) == 0 else "Untitled chat"
            )
            prefix = "● " if tid == self.thread_id else "  "
            item = QListWidgetItem(f"{prefix}{title}")
            item.setData(Qt.ItemDataRole.UserRole, tid)
            if tid == self.thread_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            item.setToolTip(tid)
            self._thread_list.addItem(item)
        self._thread_list.blockSignals(False)

    def _on_thread_clicked(self, item: QListWidgetItem):
        tid = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(tid, str) and tid != self.thread_id:
            self.load_thread(tid)

    def load_thread(self, thread_id: str):
        if self.thinking:
            QMessageBox.information(
                self, "Busy", "Wait for the current response to finish."
            )
            return
        self.thread_id = thread_id
        self._load_thread_history(thread_id)
        self._refresh_thread_list()
        self._refresh_status()
        self._render_history()

    def _load_thread_history(self, thread_id: str):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            messages = list(state.values.get("messages", []))
        except Exception:
            self.history = []
            return

        self.history = []
        for msg in messages:
            content = getattr(msg, "content", "") or ""
            if isinstance(msg, HumanMessage) and content:
                self.history.append(("user", content))
            elif isinstance(msg, AIMessage):
                text = content if isinstance(content, str) else ""
                if text.strip():
                    self.history.append(("ai", text))
            elif isinstance(msg, ToolMessage) and content:
                name = getattr(msg, "name", "tool")
                self.history.append(("tool", f"[Tool: {name}]\n{content}"))
        self._ai_idx = None

    def create_new_chat(self):
        if self.thinking:
            QMessageBox.information(self, "Busy", "Wait for the response to finish.")
            return
        self.thread_id = self._new_thread_id()
        self.history = []
        self._ai_idx = None
        self._last_token_usage = {}
        self._monitor.clear_for_new_request()
        self._refresh_thread_list()
        self._refresh_status()
        self._render_history()
        self._composer.setFocus()
        self.statusBar().showMessage("New chat started.")

    def open_chat_list(self):
        dialog = ChatListDialog(self.db_path, self.thread_id, self)
        if not dialog.exec():
            return
        if dialog.new_chat_requested:
            self.create_new_chat()
        elif dialog.retitle_thread_id:
            self._schedule_title_refresh(dialog.retitle_thread_id)
            self._refresh_thread_list()
        elif dialog.selected_thread_id:
            self.load_thread(dialog.selected_thread_id)

    # ── Settings ──────────────────────────────────────────────────────────

    def open_settings(self):
        if self.thinking:
            QMessageBox.information(self, "Busy", "Wait for the response to finish.")
            return
        SettingsDialog(self).exec()
        self._refresh_status()

    # ── Context compaction ────────────────────────────────────────────────

    def compact_context(self):
        if self.thinking:
            QMessageBox.information(self, "Busy", "Wait for the response to finish.")
            return

        self.thinking = True
        self._cancel_btn.setVisible(True)
        self._stop_btn.setVisible(True)
        self._composer.setEnabled(False)
        self._send_btn.setEnabled(False)
        self.statusBar().showMessage("Compacting context…")
        self._monitor.clear_for_new_request()
        self._monitor.add_log("compact: starting")

        worker = CompactWorker(self.graph, self.thread_id)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.event_ready.connect(self._handle_event)
        worker.event_ready.connect(lambda ev: thread.quit() if ev[0] in ("compact_done", "error") else None)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    # ── Sending messages ──────────────────────────────────────────────────

    def _send(self):
        text = self._composer.toPlainText().strip()
        if not text or self.thinking:
            return

        # Handle /lovense slash commands locally without hitting the LLM
        if text.startswith("/lovense"):
            self._handle_lovense_command(text)
            self._composer.clear()
            return

        self.history.append(("user", text))
        self._ai_idx = None
        self._last_token_usage = {}
        self._monitor.clear_for_new_request()
        self._render_history()
        self._composer.clear()

        self.thinking = True
        self._cancel_event.clear()
        self._cancel_btn.setVisible(True)
        self._stop_btn.setVisible(True)
        self._composer.setEnabled(False)
        self._send_btn.setEnabled(False)
        self.statusBar().showMessage("Generating response…")

        try:
            import lovense as _lv
            if _lv.is_connected():
                _lv.activate()
        except Exception:
            pass

        worker = StreamWorker(self.graph, self.thread_id, text, self._cancel_event)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.event_ready.connect(self._handle_event)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._stream_thread = thread
        self._stream_worker = worker
        thread.start()

    def _cancel_stream(self):
        self._cancel_event.set()
        self.thinking = False
        self._cancel_btn.setVisible(False)
        self._stop_btn.setVisible(False)
        self._composer.setEnabled(True)
        self._send_btn.setEnabled(True)
        self._composer.setFocus()
        self.statusBar().showMessage("Cancelled.")
        try:
            import lovense as _lv
            _lv.deactivate()
        except Exception:
            pass

    # ── Event dispatch ────────────────────────────────────────────────────

    def _handle_event(self, event):
        """Process one event tuple from a background worker."""
        kind = event[0]

        if kind == "classify":
            note = event[1]
            self._monitor.add_log(f"classify: {note[:60]}")
            self.history.append(("router", note))
            self._render_history()

        elif kind == "rag":
            note, ctx = event[1], event[2]
            text = f"{note}\n{ctx}" if ctx else note
            self._monitor.add_log("rag: context retrieved")
            self.history.append(("rag", text))
            self._ai_idx = None
            self._render_history()

        elif kind == "tool":
            name, content = event[1], event[2]
            self._monitor.add_log(f"tool: {name}")
            self.history.append(("tool", f"[Tool: {name}]\n{content}"))
            self._ai_idx = None
            self._render_history()

        elif kind == "plog":
            self._monitor.add_log(event[1])

        elif kind == "ai_start":
            self._monitor.add_log("ai: generating response")
            self.history.append(("ai", event[1]))
            self._ai_idx = len(self.history) - 1
            self._render_history()

        elif kind == "ai_append":
            if self._ai_idx is not None:
                role, prev = self.history[self._ai_idx]
                self.history[self._ai_idx] = (role, prev + event[1])
                self._render_history()

        elif kind == "ai_next":
            self._ai_idx = None

        elif kind == "ai_strip_tool_calls":
            if self._ai_idx is not None:
                role, prev = self.history[self._ai_idx]
                from tools import TOOL_MAP
                cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", prev, flags=re.DOTALL).strip()
                tool_names = "|".join(sorted(re.escape(n) for n in TOOL_MAP))
                cleaned = re.sub(rf"(?mis)^\s*(?:{tool_names})\s*\(.*\)\s*$", "", cleaned).strip()
                if cleaned:
                    self.history[self._ai_idx] = (role, cleaned)
                else:
                    self.history.pop(self._ai_idx)
                    self._ai_idx = None
                self._render_history()

        elif kind == "ai_strip_context_override":
            if self._ai_idx is not None:
                self.history.pop(self._ai_idx)
                self._ai_idx = None
                self._render_history()

        elif kind == "token_usage":
            usage = event[1]
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            total = usage.get("total_tokens", inp + out)
            self._last_token_usage = usage
            self._token_bar.setText(f"In: {inp:,} · Out: {out:,} · Total: {total:,}")
            self._monitor.update_tokens(inp, out, total)
            self._monitor.add_log(f"tokens in={inp:,} out={out:,}")

        elif kind == "token_usage_fallback":
            if not self._last_token_usage.get("input_tokens"):
                try:
                    tid = event[1]
                    config = {"configurable": {"thread_id": tid}}
                    state = self.graph.get_state(config)
                    messages = list(state.values.get("messages", []))
                    total_chars = sum(
                        len(str(getattr(m, "content", "") or "")) for m in messages
                    )
                    if total_chars > 0:
                        est = total_chars // 4
                        self._last_token_usage = {"input_tokens": est}
                        self._monitor.update_tokens(est, 0, est)
                        self._monitor.add_log(f"tokens est≈{est:,} (no usage reported)")
                except Exception:
                    pass

        elif kind == "compact_done":
            freed = event[1]
            self.thinking = False
            self._cancel_btn.setVisible(False)
            self._stop_btn.setVisible(False)
            self._composer.setEnabled(True)
            self._send_btn.setEnabled(True)
            self._composer.setFocus()
            self._ai_idx = None
            self.history.append(("router", f"[Context compacted — {freed:,} tokens freed]"))
            self._render_history()
            self.statusBar().showMessage(f"Context compacted ({freed:,} tokens freed).")

        elif kind == "done":
            self._monitor.add_log("done ✓")
            self.thinking = False
            self._cancel_btn.setVisible(False)
            self._stop_btn.setVisible(False)
            self._composer.setEnabled(True)
            self._send_btn.setEnabled(True)
            self._composer.setFocus()
            self._ai_idx = None
            self.statusBar().showMessage("Response complete.")
            try:
                import lovense as _lv
                _lv.deactivate()
            except Exception:
                pass
            self._schedule_title_refresh(self.thread_id)
            self._refresh_thread_list()

        elif kind == "error":
            msg = event[1]
            self._monitor.add_log(f"ERR: {msg[:80]}")
            self.history.append(("router", f"[Error: {msg}]"))
            self.thinking = False
            self._cancel_btn.setVisible(False)
            self._stop_btn.setVisible(False)
            self._composer.setEnabled(True)
            self._send_btn.setEnabled(True)
            self._composer.setFocus()
            self._ai_idx = None
            self._render_history()
            self.statusBar().showMessage(f"Error: {msg[:80]}")
            try:
                import lovense as _lv
                _lv.deactivate()
            except Exception:
                pass

        elif kind == "tool_approval_needed":
            _, tool_name, tool_args = event
            dialog = ToolApprovalDialog(tool_name, tool_args, self)
            dialog.exec()
            approval.resolve(dialog.result_approved())

        elif kind == "lovense_msg":
            self.history.append(("router", event[1]))
            self._render_history()

        elif kind == "title_done":
            _, tid, title = event
            self._save_title(tid, title)
            self._refresh_thread_list()

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render_history(self):
        blocks: list[str] = []
        for role, text in self.history:
            label, accent, bubble, border = _ROLE_STYLES.get(
                role,
                (role.title(), "#2a5f96", "#e3f2fd", "#90caf9"),
            )
            safe = self._render_markdown(text.strip() or "[Empty]")
            blocks.append(
                f'<div style="background:{bubble};border:1px solid {border};'
                f'border-radius:22px;padding:14px 18px;margin-bottom:10px;">'
                f'<span style="color:{accent};font-size:11px;font-weight:700;'
                f'letter-spacing:0.08em;text-transform:uppercase;">{label}</span><br>'
                f'<span style="color:#2e2028;font-size:14px;line-height:1.65;">{safe}</span>'
                f'</div>'
            )
        self._transcript.setHtml("".join(blocks))
        cursor = self._transcript.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._transcript.setTextCursor(cursor)

    def _render_markdown(self, text: str) -> str:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe)
        safe = re.sub(r'\*(.+?)\*', r'<i>\1</i>', safe)
        safe = re.sub(
            r'`(.+?)`',
            r'<code style="background:rgba(0,0,0,0.08);padding:1px 4px;border-radius:3px;">\1</code>',
            safe,
        )
        safe = safe.replace("\n", "<br>")
        return safe

    # ── Title generation ──────────────────────────────────────────────────

    def _schedule_title_refresh(self, thread_id: str):
        worker = TitleWorker(self.graph, thread_id)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.event_ready.connect(self._handle_event)
        worker.event_ready.connect(lambda _: thread.quit())
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    # ── Lovense slash commands ─────────────────────────────────────────────

    def _handle_lovense_command(self, text: str):
        parts = text.split(maxsplit=1)
        subcmd = parts[1].strip().lower() if len(parts) > 1 else "status"
        self.history.append(("user", text))

        try:
            import lovense as _lv
        except ImportError:
            self.history.append(("router", "[Lovense] lovense module not installed."))
            self._render_history()
            return

        if subcmd == "connect":
            lv_cfg = SETTINGS.get("lovense", {})
            token = lv_cfg.get("token", "").strip()
            uid = lv_cfg.get("uid", "").strip()
            port = lv_cfg.get("callback_port", "34569").strip()
            host = lv_cfg.get("callback_host", "").strip()
            if not token or not uid:
                self.history.append(("router",
                    "[Lovense] No dev token or UID set — open Settings → Lovense."))
                self._render_history()
                return
            import threading as _th
            def _worker():
                try:
                    resolved = host or _lv.get_local_ip()
                    cert = lv_cfg.get("cert_file", "").strip()
                    key = lv_cfg.get("key_file", "").strip()
                    proto = "https" if cert and key else "http"
                    cb = f"{proto}://{resolved}:{port}/"
                    data = _lv.get_qr(token, uid, cb)
                    qr = data.get("qr", "")
                    code = data.get("code", "")
                    msg = (
                        f"[Lovense] QR code ready!\n"
                        f"Open this URL and scan with Lovense Remote:\n  {qr}\n"
                        f"PC pairing code: {code}"
                    )
                except Exception as exc:
                    msg = f"[Lovense] QR request failed: {exc}"
                self._handle_event(("lovense_msg", msg))
            _th.Thread(target=_worker, daemon=True).start()
            self.history.append(("router", "[Lovense] Requesting QR code…"))
        elif subcmd == "test":
            if not _lv.is_connected():
                self.history.append(("router", "[Lovense] Not connected — use /lovense connect first."))
            else:
                _lv.activate(strength=10, duration_sec=3)
                self.history.append(("router", "[Lovense] Test vibration sent (3 s at strength 10)."))
        elif subcmd == "stop":
            _lv.deactivate()
            self.history.append(("router", "[Lovense] Stop command sent."))
        elif subcmd == "disconnect":
            _lv.disconnect()
            self.history.append(("router", "[Lovense] Disconnected."))
        elif subcmd == "debug":
            info = _lv.get_debug_info()
            self.history.append(("router", f"[Lovense debug]\n{info}"))
        else:
            if _lv.is_connected():
                names = ", ".join(_lv.get_toy_names()) or "unknown"
                self.history.append(("router", f"[Lovense] Connected — toys: {names}"))
            else:
                self.history.append(("router",
                    "[Lovense] Not connected. Use /lovense connect to get a pairing QR code."))

        self._render_history()

    # ── Close ─────────────────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent):  # noqa: N802
        self._cancel_event.set()
        self._log_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.quit()
            self._stream_thread.wait(500)
        super().closeEvent(event)
