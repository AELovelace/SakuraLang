# -*- coding: utf-8 -*-
"""Chat list dialog — replaces the TUI's F2 chats modal."""
import sqlite3

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QVBoxLayout,
)


class ChatListDialog(QDialog):
    """Shows all LangGraph threads with their titles and message counts.

    After exec() check:
        .selected_thread_id  — thread to switch to (None if new or cancelled)
        .new_chat_requested  — True when user clicked "New Chat"
        .retitle_thread_id   — thread_id to regenerate a title for (or None)
    """

    def __init__(self, db_path: str, current_thread_id: str, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.current_thread_id = current_thread_id
        self.selected_thread_id: str | None = None
        self.new_chat_requested = False
        self.retitle_thread_id: str | None = None
        self._threads: list[dict] = []

        self.setWindowTitle("Chat History")
        self.resize(580, 420)
        self._build_ui()
        self._refresh()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._open_selected)
        layout.addWidget(self.list_widget, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        def _btn(label, slot):
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
            return b

        _btn("Open",     self._open_selected)
        _btn("New Chat", self._new_chat)
        _btn("Delete",   self._delete_selected)
        _btn("Retitle",  self._retitle_selected)
        btn_row.addStretch()
        _btn("Close",    self.reject)

        layout.addLayout(btn_row)

    # ── Data ──────────────────────────────────────────────────────────────

    def _load_threads(self) -> list[dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()

            # LangGraph checkpoints table — group by thread and count rows.
            threads = []
            try:
                cur.execute(
                    "SELECT thread_id, COUNT(*) FROM checkpoints "
                    "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
                )
                threads = list(cur.fetchall())
            except sqlite3.OperationalError:
                pass

            titles: dict[str, str] = {}
            try:
                cur.execute("SELECT thread_id, title FROM chat_titles")
                titles = dict(cur.fetchall())
            except sqlite3.OperationalError:
                pass

            conn.close()
            return [
                {"thread_id": tid, "count": cnt, "title": titles.get(tid, "")}
                for tid, cnt in threads
            ]
        except Exception:
            return []

    def _refresh(self):
        self._threads = self._load_threads()
        self.list_widget.clear()
        for entry in self._threads:
            tid = entry["thread_id"]
            title = entry.get("title") or (
                "New chat" if entry.get("count", 0) == 0 else "Untitled chat"
            )
            prefix = "● " if tid == self.current_thread_id else "  "
            label = f"{prefix}{title}  [{entry['count']}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, tid)
            if tid == self.current_thread_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.list_widget.addItem(item)

    def _selected_tid(self) -> str | None:
        items = self.list_widget.selectedItems()
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    # ── Actions ───────────────────────────────────────────────────────────

    def _open_selected(self):
        tid = self._selected_tid()
        if tid:
            self.selected_thread_id = tid
            self.accept()

    def _new_chat(self):
        self.new_chat_requested = True
        self.accept()

    def _delete_selected(self):
        tid = self._selected_tid()
        if not tid:
            return
        if tid == self.current_thread_id:
            QMessageBox.warning(self, "Cannot delete", "Cannot delete the active chat.")
            return
        reply = QMessageBox.question(
            self, "Delete chat",
            "Permanently delete this chat and all its messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "writes"):
                try:
                    cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
                except sqlite3.OperationalError:
                    pass
            try:
                cur.execute("DELETE FROM chat_titles WHERE thread_id = ?", (tid,))
            except sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()
        except Exception:
            pass
        self._refresh()

    def _retitle_selected(self):
        tid = self._selected_tid()
        if tid:
            self.retitle_thread_id = tid
            self.accept()
