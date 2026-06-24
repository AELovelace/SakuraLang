# -*- coding: utf-8 -*-
"""QApplication bootstrap for SakuraLang GUI."""
import sys

from PyQt6.QtWidgets import QApplication

from langgraph.checkpoint.sqlite import SqliteSaver

from graph import builder
from .main_window import MainWindow

DB_PATH = "chat_history.db"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SakuraLang")

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        # Start Lovense callback server if configured
        from settings import SETTINGS
        lv_cfg = SETTINGS.get("lovense", {})
        lv_port = lv_cfg.get("callback_port", "").strip()
        if lv_port:
            try:
                import lovense as _lv
                _lv.configure(lv_cfg.get("token", ""))
                _lv.start_callback_server(
                    int(lv_port),
                    certfile=lv_cfg.get("cert_file", ""),
                    keyfile=lv_cfg.get("key_file", ""),
                )
            except Exception:
                pass

        window = MainWindow(graph, DB_PATH)
        window.show()
        sys.exit(app.exec())
