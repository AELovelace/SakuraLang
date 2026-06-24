# -*- coding: utf-8 -*-
"""Monitor panel — remote server stats (agent + researcher) + pipeline event log.

Polls http://<host>:8086/api/sakura/monitor for each server every second in
background threads (matching the TUI's polling strategy).  The QTimer only
reads the already-fetched data and updates labels — it never blocks the GUI.
"""
import threading
import time
import urllib.request
import json
from collections import deque

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QLabel, QProgressBar, QTextBrowser, QVBoxLayout, QWidget,
)

from prompts import MONITOR_URL, REMOTE_MONITOR_URL

# How many 1-second GPU snapshots to keep for averaging (≈ 10s rolling window)
_GPU_HISTORY_LEN = 10
CONTEXT_WINDOW   = 100_000


class MonitorWidget(QWidget):
    """Right-side panel showing live remote-server stats and the pipeline event log."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("monitorPanel")

        # Latest polled data from both servers (replaced atomically by polling threads)
        self._agent_data:      dict = {}
        self._researcher_data: dict = {}
        self._agent_gpu_hist:      deque = deque(maxlen=_GPU_HISTORY_LEN)
        self._researcher_gpu_hist: deque = deque(maxlen=_GPU_HISTORY_LEN)

        self._log:  list[str] = []
        self._stop = threading.Event()

        self._build_ui()
        self._start_polling()

        # Refresh UI every second (reads cached poll data — never blocks)
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._refresh_stats_ui)
        self._ui_timer.start(1000)

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        def _hdr(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setObjectName("monitorHeader")
            return lbl

        def _stat() -> QLabel:
            lbl = QLabel("—")
            lbl.setObjectName("monitorStats")
            lbl.setWordWrap(True)
            return lbl

        # ── Agent server section ──────────────────────────────────────────
        layout.addWidget(_hdr("Agent server"))
        self._agent_sys   = _stat()
        self._agent_gpu   = _stat()
        layout.addWidget(self._agent_sys)
        layout.addWidget(self._agent_gpu)

        # ── Researcher server section ─────────────────────────────────────
        layout.addWidget(_hdr("Researcher server"))
        self._researcher_sys = _stat()
        self._researcher_gpu = _stat()
        layout.addWidget(self._researcher_sys)
        layout.addWidget(self._researcher_gpu)

        # ── Context window ────────────────────────────────────────────────
        layout.addWidget(_hdr("Context window"))
        self._token_label = _stat()
        self._token_label.setText("In: — · Out: — · Total: —")
        layout.addWidget(self._token_label)

        self._ctx_bar = QProgressBar()
        self._ctx_bar.setObjectName("contextBar")
        self._ctx_bar.setMaximum(100)
        self._ctx_bar.setValue(0)
        self._ctx_bar.setTextVisible(False)
        self._ctx_bar.setFixedHeight(8)
        layout.addWidget(self._ctx_bar)

        # ── Pipeline log ──────────────────────────────────────────────────
        layout.addWidget(_hdr("Pipeline log"))
        self._log_browser = QTextBrowser()
        self._log_browser.setObjectName("pipelineLog")
        self._log_browser.setOpenExternalLinks(False)
        layout.addWidget(self._log_browser, 1)

    # ── Background polling ────────────────────────────────────────────────

    def _start_polling(self):
        """Start one daemon thread per server.  Threads write instance attrs;
        the QTimer reads them on the UI tick — thread-safe via Python's GIL
        and atomic dict replacement."""
        def _poll(url: str, data_attr: str, hist_attr: str):
            while not self._stop.is_set():
                try:
                    with urllib.request.urlopen(url, timeout=1) as resp:
                        raw = resp.read().decode()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = {"error": raw[:120]}
                    setattr(self, data_attr, data)
                    gpus = data.get("gpus", []) if isinstance(data, dict) else []
                    if gpus:
                        getattr(self, hist_attr).append(gpus)
                except Exception as exc:
                    setattr(self, data_attr, {"error": str(exc)[:80]})
                self._stop.wait(1)

        for url, da, ha in (
            (MONITOR_URL,        "_agent_data",      "_agent_gpu_hist"),
            (REMOTE_MONITOR_URL, "_researcher_data", "_researcher_gpu_hist"),
        ):
            t = threading.Thread(target=_poll, args=(url, da, ha), daemon=True)
            t.start()

    # ── Public API ────────────────────────────────────────────────────────

    def add_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"{ts} {msg}")
        if len(self._log) > 200:
            del self._log[0]
        lines = self._log[-60:]
        html = "".join(
            f'<div style="font-size:11px;color:#7a6070;font-family:Consolas,monospace;">'
            f'{line}</div>'
            for line in lines
        )
        self._log_browser.setHtml(html)
        sb = self._log_browser.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_tokens(self, token_in: int, token_out: int, total: int):
        self._token_label.setText(
            f"In: {token_in:,} · Out: {token_out:,} · Total: {total:,}"
        )
        pct = min(100, int(total * 100 / CONTEXT_WINDOW))
        self._ctx_bar.setValue(pct)

    def clear_for_new_request(self):
        self._log = []
        self._log_browser.clear()

    def closeEvent(self, event):
        self._stop.set()
        super().closeEvent(event)

    # ── Internal ──────────────────────────────────────────────────────────

    def _refresh_stats_ui(self):
        self._agent_sys.setText(self._fmt_sys(self._agent_data))
        self._agent_gpu.setText(self._fmt_gpu(self._agent_data, self._agent_gpu_hist))
        self._researcher_sys.setText(self._fmt_sys(self._researcher_data))
        self._researcher_gpu.setText(self._fmt_gpu(self._researcher_data, self._researcher_gpu_hist))

    @staticmethod
    def _fmt_sys(data: dict) -> str:
        """Format CPU/RAM line from one server's monitor payload."""
        if not data:
            return "connecting…"
        if "error" in data:
            return f"offline: {data['error'][:40]}"
        sys = data.get("system", {})
        if not sys:
            return "—"
        cpu   = sys.get("cpu_percent")
        ram_u = sys.get("ram_used_gib")
        ram_t = sys.get("ram_total_gib")
        parts = []
        if cpu   is not None:               parts.append(f"CPU {cpu:.0f}%")
        if ram_u is not None and ram_t is not None: parts.append(f"RAM {ram_u:.1f}/{ram_t:.1f} GiB")
        return "  ".join(parts) if parts else "—"

    @staticmethod
    def _fmt_gpu(data: dict, hist: deque) -> str:
        """Format GPU stat line(s) from one server's monitor payload."""
        if not data or "error" in data:
            return ""
        gpus = data.get("gpus", [])
        if not gpus and hist:
            gpus = hist[-1]   # fall back to last known snapshot
        if not gpus:
            return "GPU: —"

        lines = []
        for i, gpu in enumerate(gpus):
            name = (
                gpu.get("name", f"GPU {i}")
                .replace("AMD Radeon ", "")
                .replace("NVIDIA GeForce ", "")
                .replace("NVIDIA ", "")
            )
            vram_u = gpu.get("vram_used_mib")
            vram_t = gpu.get("vram_total_mib")
            util   = gpu.get("util_percent")
            power  = gpu.get("power_watts")
            temp   = gpu.get("temperature_c")

            # Average util/power/temp over the rolling history window
            if hist:
                def _avg(key, idx=i):
                    vals = [
                        s[idx].get(key)
                        for s in hist
                        if idx < len(s) and s[idx].get(key) is not None
                    ]
                    return sum(vals) / len(vals) if vals else None
                util  = _avg("util_percent")  or util
                power = _avg("power_watts")   or power
                temp  = _avg("temperature_c") or temp

            vram_str = f"VRAM {vram_u}/{vram_t} MiB" if (vram_u is not None and vram_t is not None) else ""
            detail = []
            if util  is not None: detail.append(f"{util:.0f}%")
            if power is not None: detail.append(f"{power:.0f}W")
            if temp  is not None: detail.append(f"{temp:.0f}°C")
            detail_str = " ".join(detail)

            parts = [p for p in (name, vram_str, detail_str) if p]
            lines.append("  ".join(parts))

        return "\n".join(lines)
