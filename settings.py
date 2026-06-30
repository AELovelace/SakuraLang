# -*- coding: utf-8 -*-
import json
import os

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "mode": "auto",   # auto | agent | plan | chat
    "agent": {
        "address": "http://100.66.64.45:9090/v1",
        "system_prompt": "",
        "cwd": "",
    },
    # The "researcher" is our third AI: a llama.cpp endpoint running Qwen 3.6 35ba3b.
    # It's fast and great at research, so it's the brain behind the Hybrid RAG pipeline
    # (query rewriting + final answer synthesis from retrieved context).
    "researcher": {
        "address": "http://100.83.3.32:9090/v1",
        "system_prompt": "",
    },
    "brave": {
        "api_key": "",
        "base_url": "https://api.search.brave.com/res/v1/web/search",
        "count": "5",
        "country": "us",
        "search_lang": "en",
        "safesearch": "moderate",
    },
    "classifier": {
        "address": "http://100.66.64.45:9091/v1",
        "system_prompt": "",
    },
    "titler": {
        "address": "http://100.83.3.32:9091/v1",
        "system_prompt": "",
    },
    # Lovense Standard API integration — toy connection and vibration control.
    # token / uid come from https://www.lovense.com/user/developer/info.
    # callback_port: the port this app listens on for the toy-pairing POST.
    # callback_host: LAN IP shown in the pairing URL (blank = auto-detect).
    # heat_toy: toy ID for <heat> tags (blank = all connected toys).
    # reward_toy: toy ID for <reward> tags (blank = disabled).
    "lovense": {
        "token":         "",
        "uid":           "",
        "callback_port": "34569",
        "callback_host": "",
        "cert_file":     "",
        "key_file":      "",
        "heat_toy":      "",
        "reward_toy":    "",
    },
}

# ---------------------------------------------------------------------------
# Display / timing constants used by both the app and graph layers
# ---------------------------------------------------------------------------

CHAT_INPUT_MIN_H           = 3
CHAT_INPUT_MAX_H           = 6
GETCH_TIMEOUT_MS           = 1000
ESCAPE_SEQUENCE_TIMEOUT_MS = 25
SHIFT_ENTER_ESCAPE_SEQUENCES = {
    "[13;2u",
    "[13;2~",
    "[27;2;13~",
}
STREAM_STALL_TIMEOUT_SEC = 120
CHAT_RENDER_MAX_CHARS    = 12000
TOOL_DISPLAY_MAX_CHARS   = 4000
RAG_DISPLAY_MAX_CHARS    = 6000


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
        result = json.loads(json.dumps(DEFAULT_SETTINGS))
        # Merge every known section so older settings.json files (which may pre-date
        # the researcher endpoint) still load cleanly and just pick up the new defaults.
        for section in ("agent", "researcher", "brave", "classifier", "titler", "lovense"):
            if section in data:
                result[section].update(data[section])
        if "mode" in data and data["mode"] in ("auto", "agent", "plan", "chat"):
            result["mode"] = data["mode"]
        return result
    except (FileNotFoundError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_SETTINGS))


def save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


SETTINGS: dict = load_settings()
