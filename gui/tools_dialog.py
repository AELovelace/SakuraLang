# -*- coding: utf-8 -*-
"""Tool approval dialog — shown when the agent wants to execute a tool in Agent mode."""
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)


class ToolApprovalDialog(QDialog):
    """Modal dialog that lets the user approve or reject a tool call.

    Shows the tool name and its arguments.  Auto-rejects after 60 s if
    the user doesn't respond (matching the TUI behaviour).
    """

    def __init__(self, tool_name: str, tool_args: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Tool Call")
        self.setModal(True)
        self.resize(500, 300)
        self._approved = False
        self._countdown = 60

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("The agent wants to call a tool:")
        header.setStyleSheet("font-weight: 700; font-size: 14px; color: #5e3f49;")
        layout.addWidget(header)

        name_label = QLabel(f"Tool:  {tool_name}")
        name_label.setStyleSheet(
            "font-weight: 600; font-size: 13px; color: #a83255; padding: 4px 0;"
        )
        layout.addWidget(name_label)

        args_lines = "\n".join(
            f"  {k}: {v!r}" for k, v in (tool_args or {}).items()
        ) or "  (no arguments)"
        args_label = QLabel(args_lines)
        args_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px;"
            "background: #f0e8ec; padding: 10px 12px;"
            "border-radius: 10px; color: #3d2530;"
        )
        args_label.setWordWrap(True)
        layout.addWidget(args_label, 1)

        self._timer_label = QLabel(f"Auto-reject in {self._countdown}s")
        self._timer_label.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(self._timer_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        approve_btn = QPushButton("✓  Approve")
        approve_btn.setDefault(True)
        approve_btn.setStyleSheet(
            "QPushButton { background: #bf768d; color: #fff; border-radius: 14px;"
            "padding: 8px 20px; font-weight: 700; }"
            "QPushButton:hover { background: #a85070; }"
        )
        approve_btn.clicked.connect(self._approve)

        reject_btn = QPushButton("✕  Reject")
        reject_btn.setStyleSheet(
            "QPushButton { background: #e8d8dd; color: #5e3f49; border-radius: 14px;"
            "padding: 8px 20px; font-weight: 600; }"
            "QPushButton:hover { background: #d4b8c2; }"
        )
        reject_btn.clicked.connect(self.reject)

        btn_row.addStretch()
        btn_row.addWidget(approve_btn)
        btn_row.addWidget(reject_btn)
        layout.addLayout(btn_row)

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(1000)

    def _approve(self):
        self._approved = True
        self._tick_timer.stop()
        self.accept()

    def _tick(self):
        self._countdown -= 1
        self._timer_label.setText(f"Auto-reject in {self._countdown}s")
        if self._countdown <= 0:
            self._tick_timer.stop()
            self.reject()

    def result_approved(self) -> bool:
        return self._approved
