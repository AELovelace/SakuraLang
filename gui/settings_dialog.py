# -*- coding: utf-8 -*-
"""Tabbed settings dialog mirroring the 23-field TUI settings form (F12)."""
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTabWidget,
    QVBoxLayout, QWidget,
)

from settings import SETTINGS, DEFAULT_SETTINGS, save_settings
from llm import rebuild_llms


def _line(text: str = "") -> QLineEdit:
    w = QLineEdit(text)
    return w


def _password(text: str = "") -> QLineEdit:
    w = QLineEdit(text)
    w.setEchoMode(QLineEdit.EchoMode.Password)
    return w


def _text(text: str = "", placeholder: str = "") -> QPlainTextEdit:
    w = QPlainTextEdit(text)
    w.setFixedHeight(80)
    if placeholder:
        w.setPlaceholderText(placeholder)
    return w


def _tab(rows: list) -> QWidget:
    """Build a QWidget with a QFormLayout from (label, widget) pairs."""
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(16, 16, 16, 16)
    form.setVerticalSpacing(10)
    for label, widget in rows:
        form.addRow(label, widget)
    return page


class SettingsDialog(QDialog):
    """Six-tab settings dialog for SakuraLang configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(620, 480)

        s = SETTINGS
        ag = s.get("agent", {})
        rs = s.get("researcher", {})
        cl = s.get("classifier", {})
        ti = s.get("titler", {})
        br = s.get("brave", {})
        lv = s.get("lovense", {})
        df = DEFAULT_SETTINGS

        # ── Agent ────────────────────────────────────────────────────────
        self.agent_addr   = _line(ag.get("address",       df["agent"]["address"]))
        self.agent_prompt = _text(ag.get("system_prompt", ""), "Optional system prompt for the main agent")
        self.agent_cwd    = _line(ag.get("cwd",           ""))

        # ── Researcher ───────────────────────────────────────────────────
        self.res_addr   = _line(rs.get("address",       df["researcher"]["address"]))
        self.res_prompt = _text(rs.get("system_prompt", ""), "Optional system prompt for the researcher")

        # ── Classifier ───────────────────────────────────────────────────
        self.cls_addr   = _line(cl.get("address",       df["classifier"]["address"]))
        self.cls_prompt = _text(cl.get("system_prompt", ""), "Optional override for the classifier prompt")

        # ── Titler ───────────────────────────────────────────────────────
        self.tit_addr   = _line(ti.get("address",       df["titler"]["address"]))
        self.tit_prompt = _text(ti.get("system_prompt", ""), "Optional override for the titler prompt")

        # ── Brave Search ─────────────────────────────────────────────────
        self.brave_key        = _password(br.get("api_key",     ""))
        self.brave_key.setPlaceholderText("Brave Search subscription key")
        self.brave_url        = _line(br.get("base_url",    df["brave"]["base_url"]))
        self.brave_count      = _line(br.get("count",       df["brave"]["count"]))
        self.brave_country    = _line(br.get("country",     df["brave"]["country"]))
        self.brave_lang       = _line(br.get("search_lang", df["brave"]["search_lang"]))
        self.brave_safesearch = _line(br.get("safesearch",  df["brave"]["safesearch"]))

        # ── Lovense ──────────────────────────────────────────────────────
        self.lv_token = _password(lv.get("token",         ""))
        self.lv_uid   = _line(lv.get("uid",               ""))
        self.lv_port  = _line(lv.get("callback_port",     "34569"))
        self.lv_host  = _line(lv.get("callback_host",     ""))
        self.lv_cert  = _line(lv.get("cert_file",         ""))
        self.lv_key   = _line(lv.get("key_file",          ""))

        # Toy pickers — combo boxes populated from connected toy list
        self.lv_heat_combo   = QComboBox()
        self.lv_reward_combo = QComboBox()
        self._saved_heat_toy   = lv.get("heat_toy",   "")
        self._saved_reward_toy = lv.get("reward_toy", "")
        self._populate_toy_combos()

        tabs = QTabWidget()
        tabs.addTab(_tab([
            ("Address",        self.agent_addr),
            ("System prompt",  self.agent_prompt),
            ("Working dir",    self.agent_cwd),
        ]), "Agent")
        tabs.addTab(_tab([
            ("Address",        self.res_addr),
            ("System prompt",  self.res_prompt),
        ]), "Researcher")
        tabs.addTab(_tab([
            ("Address",        self.cls_addr),
            ("System prompt",  self.cls_prompt),
        ]), "Classifier")
        tabs.addTab(_tab([
            ("Address",        self.tit_addr),
            ("System prompt",  self.tit_prompt),
        ]), "Titler")
        tabs.addTab(_tab([
            ("API key",        self.brave_key),
            ("Base URL",       self.brave_url),
            ("Result count",   self.brave_count),
            ("Country",        self.brave_country),
            ("Search language",self.brave_lang),
            ("Safe search",    self.brave_safesearch),
        ]), "Brave Search")
        # Toy picker row: combo + refresh button side by side
        heat_row = QWidget()
        heat_layout = QHBoxLayout(heat_row)
        heat_layout.setContentsMargins(0, 0, 0, 0)
        heat_layout.addWidget(self.lv_heat_combo, 1)

        reward_row = QWidget()
        reward_layout = QHBoxLayout(reward_row)
        reward_layout.setContentsMargins(0, 0, 0, 0)
        reward_layout.addWidget(self.lv_reward_combo, 1)

        refresh_btn = QPushButton("Refresh toy list")
        refresh_btn.clicked.connect(self._populate_toy_combos)

        tabs.addTab(_tab([
            ("Dev token",      self.lv_token),
            ("User ID",        self.lv_uid),
            ("Callback port",  self.lv_port),
            ("Callback host",  self.lv_host),
            ("TLS cert file",  self.lv_cert),
            ("TLS key file",   self.lv_key),
            ("Heat toy (toy 1)",   heat_row),
            ("Reward toy (toy 2)", reward_row),
            ("",               refresh_btn),
        ]), "Lovense")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Changes take effect immediately.  "
            "Restart the Lovense callback server manually if you change port/host/certs."
        ))
        layout.addWidget(tabs, 1)
        layout.addWidget(buttons)

    def _populate_toy_combos(self) -> None:
        """Rebuild the heat/reward combo boxes from currently connected toys."""
        try:
            import lovense as _lv
            toys = _lv.get_toys()  # [{"id": str, "name": str}]
        except Exception:
            toys = []

        for combo, none_label, saved_id in (
            (self.lv_heat_combo,   "All toys (default)", self._saved_heat_toy),
            (self.lv_reward_combo, "None (disabled)",    self._saved_reward_toy),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(none_label, "")  # empty string = not assigned
            for toy in toys:
                display = f"{toy['name']}  [{toy['id']}]"
                combo.addItem(display, toy["id"])
            # Re-select the previously saved toy if it's still in the list
            idx = combo.findData(saved_id)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def _save_and_accept(self):
        SETTINGS["agent"]["address"]       = self.agent_addr.text().strip()
        SETTINGS["agent"]["system_prompt"] = self.agent_prompt.toPlainText().strip()
        SETTINGS["agent"]["cwd"]           = self.agent_cwd.text().strip()

        SETTINGS["researcher"]["address"]       = self.res_addr.text().strip()
        SETTINGS["researcher"]["system_prompt"] = self.res_prompt.toPlainText().strip()

        SETTINGS["classifier"]["address"]       = self.cls_addr.text().strip()
        SETTINGS["classifier"]["system_prompt"] = self.cls_prompt.toPlainText().strip()

        SETTINGS["titler"]["address"]       = self.tit_addr.text().strip()
        SETTINGS["titler"]["system_prompt"] = self.tit_prompt.toPlainText().strip()

        SETTINGS["brave"]["api_key"]     = self.brave_key.text().strip()
        SETTINGS["brave"]["base_url"]    = self.brave_url.text().strip()
        SETTINGS["brave"]["count"]       = self.brave_count.text().strip()
        SETTINGS["brave"]["country"]     = self.brave_country.text().strip()
        SETTINGS["brave"]["search_lang"] = self.brave_lang.text().strip()
        SETTINGS["brave"]["safesearch"]  = self.brave_safesearch.text().strip()

        SETTINGS["lovense"]["token"]         = self.lv_token.text().strip()
        SETTINGS["lovense"]["uid"]           = self.lv_uid.text().strip()
        SETTINGS["lovense"]["callback_port"] = self.lv_port.text().strip()
        SETTINGS["lovense"]["callback_host"] = self.lv_host.text().strip()
        SETTINGS["lovense"]["cert_file"]     = self.lv_cert.text().strip()
        SETTINGS["lovense"]["key_file"]      = self.lv_key.text().strip()
        SETTINGS["lovense"]["heat_toy"]      = self.lv_heat_combo.currentData() or ""
        SETTINGS["lovense"]["reward_toy"]    = self.lv_reward_combo.currentData() or ""

        save_settings(SETTINGS)
        rebuild_llms()
        self.accept()
