# -*- coding: utf-8 -*-
import json
import os
import pathlib
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import warnings
import curses
from collections import deque
from typing import Annotated
from typing_extensions import TypedDict

warnings.filterwarnings("ignore", message="Core Pydantic V1", module="pydantic")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk, ToolMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
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
    "classifier": {
        "address": "http://100.66.64.45:9091/v1",
        "system_prompt": "",
    },
    # Lovense Standard API integration — toy connection and vibration control.
    # token / uid come from https://www.lovense.com/user/developer/info.
    # callback_port: the port this app listens on for the toy-pairing POST.
    # callback_host: LAN IP shown in the pairing URL (blank = auto-detect).
    "lovense": {
        "token":         "",
        "uid":           "",
        "callback_port": "34569",
        "callback_host": "",
        "cert_file":     "",
        "key_file":      "",
    },
}


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
        result = json.loads(json.dumps(DEFAULT_SETTINGS))
        # Merge every known section so older settings.json files (which may pre-date
        # the researcher endpoint) still load cleanly and just pick up the new defaults.
        for section in ("agent", "researcher", "classifier", "lovense"):
            if section in data:
                result[section].update(data[section])
        return result
    except (FileNotFoundError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_SETTINGS))


def save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


SETTINGS: dict = load_settings()

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

_llm_cache: dict[tuple, ChatOpenAI] = {}


def get_llm(address: str, streaming: bool = True, json_mode: bool = False,
            timeout: int = 120) -> ChatOpenAI:
    # `timeout` is the full request budget (connect + read). The researcher (Qwen on
    # llama.cpp) needs a generous window because the 35B model can still be warming up
    # on the very first call, so we give that endpoint a dedicated 120s wait.
    key = (address, streaming, json_mode, timeout)
    if key not in _llm_cache:
        kwargs: dict = {
            "base_url": address,
            "api_key": "not-needed",
            "model": "local-model",
            "streaming": streaming,
            "timeout": timeout,  # prevent infinite hang when SSE stream stalls mid-response
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        _llm_cache[key] = ChatOpenAI(**kwargs)
    return _llm_cache[key]


def rebuild_llms() -> None:
    _llm_cache.clear()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    messages:     Annotated[list, add_messages]
    intent:       str
    domain:       str
    confidence:   float
    rag_needed:   bool
    tools_needed: list[str]
    routing_note: str
    summary:      str
    rag_context:  str   # retrieved+compressed context from the Hybrid RAG pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM = """\
You are a JSON classifier. Your ONLY output is a single JSON object — nothing else.
Do NOT greet the user. Do NOT explain. Do NOT use markdown. Do NOT add any text before or after the JSON.

Classify the user message with these exact fields:

intent: one of: chat, question, task, research, code, troubleshoot, document
domain: one of: general, coding, network, windows, hotel_it, verifone, ai_runtime, finance, legal_hr
confidence: number from 0.0 to 1.0
rag_needed: true or false
tools_needed: array of strings (empty array if none)

Your entire response must be exactly one JSON object like this:
{"intent":"troubleshoot","domain":"windows","confidence":0.92,"rag_needed":false,"tools_needed":[]}

Start your response with { and end with }. No other characters allowed."""

DOMAIN_PROMPTS = {
    "hotel_it":   "You are an expert Hotel IT support engineer.",
    "coding":     "You are an expert software engineer.",
    "network":    "You are an expert network engineer.",
    "windows":    "You are an expert Windows systems administrator.",
    "verifone":   "You are an expert Verifone payment systems technician.",
    "ai_runtime": "You are an expert AI infrastructure and runtime engineer.",
    "finance":    "You are a knowledgeable finance assistant.",
    "legal_hr":   "You are a knowledgeable HR and legal information assistant.",
    "general":    "You are a helpful general-purpose assistant.",
}

# Default persona for the Qwen researcher (our third AI). It is the model the RAG
# pipeline talks to: it rewrites queries and synthesizes answers strictly from the
# context we hand it, so we tell it to stay grounded and cite what it used.
RESEARCHER_SYSTEM = """\
You are a fast, precise research assistant. You answer using ONLY the retrieved context provided to you.
If the context does not contain the answer, say so plainly instead of guessing.
Prefer exact identifiers (model names, error codes, IPs, terminal IDs, PLUs, config keys) verbatim from the context.
Be concise and factual."""
CONFIDENCE_THRESHOLD  = 0.65
WINDOW_SIZE           = 20
SUMMARIZE_THRESHOLD   = 40

SUMMARIZE_PROMPT = """\
Summarize the key facts and context from the following conversation history.
Focus on: names, IP addresses, hostnames, error messages, resolved issues, and ongoing tasks.
Be concise — this summary will be prepended to future responses to preserve context."""

# Used by the RAG node: turn a messy user request into a tight, keyword-rich search
# query BEFORE retrieving. Searching the raw prompt is weak; a rewritten query that
# preserves exact identifiers (model names, error codes, IPs, IDs, PLUs) retrieves
# far better from IT docs and logs.
QUERY_REWRITE_PROMPT = """\
Rewrite the user's request into a single concise search query for retrieving IT documentation, logs, and configs.
Keep exact identifiers verbatim: model names, error codes, terminal IDs, PLUs, IP addresses, hostnames, config keys.
Expand obvious abbreviations and add a few high-signal keywords.
Output ONLY the search query text — no quotes, no labels, no explanation."""

CLASSIFIER_RETRY_MSG = (
    "WRONG. That output could not be parsed as JSON. "
    "Your ENTIRE response must be ONE JSON object — nothing before it, nothing after it. "
    "No prose. No markdown. No explanation. START WITH { END WITH }. Try again:"
)

CLASSIFIER_CONFIDENCE_RETRY_MSG = (
    "WRONG. Your confidence field is 0.0 — this is never a valid value. "
    "Confidence must be between 0.1 and 1.0 and reflect how certain you are of the classification. "
    "If genuinely unsure, use 0.5. Output the corrected JSON object now:"
)

MONITOR_URL = "http://100.66.64.45:8086/api/sakura/monitor"
LOGS_URL    = "http://100.66.64.45:8086/api/sakura/logs"

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS         = 0x00000008


def _run_proc(cmd: list[str], cwd=None, timeout: int = 15) -> tuple[str, str]:
    """Run a subprocess with timeout, killing the full process tree if it stalls."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        creationflags=_CREATE_NEW_PROCESS_GROUP,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return out.strip(), err.strip()
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, timeout=5,
        )
        try:
            out, err = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        note = "[Killed after timeout — use launch_app for long-running processes]"
        return out.strip(), (err.strip() + "\n" + note).strip()


@tool
def run_powershell(command: str) -> str:
    """Execute a short-lived PowerShell command and return stdout/stderr (15s timeout).
    For network ping tests use ping.exe directly (e.g. 'ping.exe -n 4 192.168.1.1'), not Test-Connection.
    Do NOT use this to start GUI apps or servers — use launch_app instead."""
    cwd = SETTINGS.get("agent", {}).get("cwd", "").strip() or None
    out, err = _run_proc(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=cwd,
    )
    if err:
        return f"STDOUT:\n{out}\nSTDERR:\n{err}" if out else f"STDERR:\n{err}"
    return out or "(no output)"


@tool
def run_python(code: str) -> str:
    """Execute a short-lived Python snippet and return stdout/stderr (15s timeout).
    Do NOT use this to start GUI apps or long-running scripts — use launch_app instead."""
    cwd = SETTINGS.get("agent", {}).get("cwd", "").strip() or None
    out, err = _run_proc([sys.executable, "-c", code], cwd=cwd)
    if err:
        return f"STDOUT:\n{out}\nSTDERR:\n{err}" if out else f"STDERR:\n{err}"
    return out or "(no output)"


@tool
def launch_app(command: str) -> str:
    """Launch a long-running application (GUI app, server, background script) and return immediately.
    The process is fully detached — use this instead of run_powershell/run_python when you want
    to START something without waiting for it to finish (e.g. 'python main.py', 'npm start')."""
    cwd = SETTINGS.get("agent", {}).get("cwd", "").strip() or None
    try:
        flags = _CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS
        subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return f"Launched (detached): {command}"
    except Exception as exc:
        return f"[Error: {exc}]"


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file on disk.  Creates the file and any missing
    parent directories if needed; overwrites if the file already exists.
    Returns the resolved absolute path on success."""
    try:
        target = pathlib.Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written: {target.resolve()}"
    except Exception as exc:
        return f"[Error: {exc}]"


TOOLS    = [run_powershell, run_python, launch_app, write_file]
TOOL_MAP = {t.name: t for t in TOOLS}


def _parse_text_tool_calls(content: str) -> list[dict]:
    """Parse <tool_call> XML blocks emitted by models that don't use structured function calling."""
    calls = []
    for block in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        fn_match = re.search(r"<function=(\w+)>", block.group(1))
        if not fn_match:
            continue
        name   = fn_match.group(1)
        params = {}
        for p in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", block.group(1), re.DOTALL):
            params[p.group(1)] = p.group(2).strip()
        calls.append({"name": name, "args": params})
    return calls


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response that may include prose or code fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        raw = re.search(r"\{.*\}", text, re.DOTALL)
        if raw:
            text = raw.group(0)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def classify(state: AgentState) -> dict:
    cfg        = SETTINGS["classifier"]
    sys_prompt = cfg["system_prompt"] or CLASSIFIER_SYSTEM
    llm        = get_llm(cfg["address"], streaming=False, json_mode=True)

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        state["messages"][-1],
    )

    raw  = ""
    data = {}
    for attempt in range(2):
        if attempt == 0:
            messages = [SystemMessage(content=sys_prompt), last_human]
        else:
            messages = [
                SystemMessage(content=sys_prompt),
                last_human,
                AIMessage(content=raw),
                HumanMessage(content=CLASSIFIER_RETRY_MSG),
            ]

        response = llm.invoke(messages)
        raw  = response.content if isinstance(response.content, str) else ""
        data = _extract_json(raw)
        if data:
            break

    retried = attempt > 0
    if not data:
        snippet = raw[:120].replace("\n", " ") if raw else "<empty>"
        return {
            "intent": "chat", "domain": "general", "confidence": 0.0,
            "rag_needed": False, "tools_needed": [],
            "routing_note": f"[Router → parse failed (2 attempts) | raw: {snippet}]",
        }

    # Shock retry: if confidence came back 0.0, demand a real value
    shocked = False
    if float(data.get("confidence", 0.0)) <= 0.0:
        shock_resp = llm.invoke([
            SystemMessage(content=sys_prompt),
            last_human,
            AIMessage(content=raw),
            HumanMessage(content=CLASSIFIER_CONFIDENCE_RETRY_MSG),
        ])
        shock_raw  = shock_resp.content if isinstance(shock_resp.content, str) else ""
        shock_data = _extract_json(shock_raw)
        if shock_data and float(shock_data.get("confidence", 0.0)) > 0.0:
            data = shock_data
            raw  = shock_raw
        shocked = True

    intent       = str(data.get("intent", "chat"))
    domain       = str(data.get("domain", "general"))
    confidence   = float(data.get("confidence", 0.0))
    rag_needed   = bool(data.get("rag_needed", False))
    tools_needed = list(data.get("tools_needed", []))

    tools_str  = ", ".join(tools_needed) if tools_needed else "none"
    rag_str    = " | RAG" if rag_needed else ""
    retry_str  = " | retried" if retried else ""
    shock_str  = " ⚡ zero-conf" if (shocked and confidence <= 0.0) else ""
    routing_note = (
        f"[Router → {intent}/{domain}"
        f" (conf: {confidence:.2f}){rag_str}{shock_str}"
        f" | tools: {tools_str}{retry_str}]"
    )

    return {
        "intent": intent, "domain": domain, "confidence": confidence,
        "rag_needed": rag_needed, "tools_needed": tools_needed,
        "routing_note": routing_note,
    }


def route(state: AgentState) -> str:
    conf = state.get("confidence", 0.0)
    if conf <= 0.0:
        return "respond"  # classifier failed completely — just answer
    if conf < CONFIDENCE_THRESHOLD:
        return "clarify"
    if state.get("rag_needed", False):
        return "rag"
    return "respond"


def _build_sys_prompt(base: str, state: AgentState) -> str:
    parts = [base]
    cwd = SETTINGS.get("agent", {}).get("cwd", "").strip()
    if cwd:
        parts.append(f"Current working directory: {cwd}")
    summary = state.get("summary", "")
    if summary:
        parts.append(f"Earlier conversation summary:\n{summary}")
    return "\n\n".join(parts)


def clarify(state: AgentState) -> dict:
    cfg      = SETTINGS["agent"]
    base     = (
        "The user's request is unclear. "
        "Ask one short clarifying question to better understand what they need. "
        "Do not answer the question itself."
    )
    response = get_llm(cfg["address"]).invoke([
        SystemMessage(content=_build_sys_prompt(base, state)),
        *state["messages"][-WINDOW_SIZE:],
    ])
    conf = state.get("confidence", 0.0)
    return {
        "messages":     [AIMessage(content=response.content)],
        "routing_note": f"[Router → low confidence ({conf:.2f}) — asking for clarification]",
    }


def rag(state: AgentState) -> dict:
    """Hybrid RAG node: rewrite the query, retrieve+rerank+compress local docs,
    and stash the resulting context for the `respond` node (Qwen) to answer from.

    Heavy lifting lives in rag.py. We only reach this node when the classifier set
    rag_needed=True. If no working dir is configured (or indexing/retrieval fails),
    we simply add a note and let `respond` answer normally without context."""
    note = state.get("routing_note", "")
    cwd  = SETTINGS.get("agent", {}).get("cwd", "").strip()

    # No corpus configured → nothing to retrieve from. Fall through to a normal answer.
    if not cwd or not os.path.isdir(cwd):
        return {"routing_note": note + " [RAG: skipped — no working dir]"}

    try:
        import rag as rag_engine  # local module; imported lazily to keep startup fast
    except Exception as exc:
        return {"routing_note": note + f" [RAG: unavailable — {exc}]"}

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        state["messages"][-1],
    )
    user_q = last_human.content if isinstance(last_human.content, str) else ""

    # 1) QUERY REWRITE — ask Qwen to turn the request into a clean search query.
    rcfg = SETTINGS["researcher"]
    search_q = user_q
    try:
        # 120s wait — the Qwen researcher may still be warming up on its first call.
        rewrite = get_llm(rcfg["address"], streaming=False, timeout=120).invoke([
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=user_q),
        ])
        rewritten = rewrite.content if isinstance(rewrite.content, str) else ""
        if rewritten.strip():
            search_q = rewritten.strip()
    except Exception:
        pass  # rewriting is best-effort; fall back to the raw prompt

    # 2) HYBRID RETRIEVAL + RERANK + SMALL-TO-BIG + COMPRESS (all in rag.py).
    try:
        rag_engine.ensure_indexed(cwd)
        context, sources = rag_engine.retrieve(search_q, cwd)
    except Exception as exc:
        return {"routing_note": note + f" [RAG: error — {exc}]"}

    if not context:
        return {"routing_note": note + " [RAG: no relevant matches]"}

    src_str = ", ".join(sources[:3])
    return {
        "rag_context":  context,
        "routing_note": note + f" [RAG: {len(sources)} src → {src_str}]",
    }


def respond(state: AgentState) -> dict:
    # If the RAG node retrieved context, answer with the Qwen researcher grounded
    # strictly in that context (no tools — it's a pure synthesis-from-context step).
    rag_context = state.get("rag_context", "")
    if rag_context:
        cfg    = SETTINGS["researcher"]
        base   = cfg["system_prompt"] or RESEARCHER_SYSTEM
        base   = base + "\n\nRetrieved context (answer using ONLY this):\n" + rag_context
        # 120s wait — give the Qwen researcher room to spin up before answering.
        llm    = get_llm(cfg["address"], timeout=120)
        resp   = llm.invoke([
            SystemMessage(content=_build_sys_prompt(base, state)),
            *state["messages"][-WINDOW_SIZE:],
        ])
        # Clear the context so it isn't reused on the next turn.
        return {"messages": [resp], "rag_context": ""}

    cfg           = SETTINGS["agent"]
    domain        = state.get("domain", "general")
    domain_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    custom        = cfg["system_prompt"]
    base          = (custom + "\n\n" + domain_prompt) if custom else domain_prompt

    llm      = get_llm(cfg["address"]).bind_tools(TOOLS)
    response = llm.invoke([
        SystemMessage(content=_build_sys_prompt(base, state)),
        *state["messages"][-WINDOW_SIZE:],
    ])
    return {"messages": [response]}


def summarize(state: AgentState) -> dict:
    messages = state["messages"]
    if len(messages) <= SUMMARIZE_THRESHOLD:
        return {}

    to_compress = messages[:-WINDOW_SIZE]
    cfg         = SETTINGS["agent"]
    existing    = state.get("summary", "")
    prefix      = f"Previous summary:\n{existing}\n\nNew messages to incorporate:\n" if existing else ""
    history_text = "\n".join(
        f"{m.__class__.__name__}: {getattr(m, 'content', '')}"
        for m in to_compress
    )

    response = get_llm(cfg["address"], streaming=False).invoke([
        SystemMessage(content=SUMMARIZE_PROMPT),
        HumanMessage(content=prefix + history_text),
    ])
    new_summary = response.content if isinstance(response.content, str) else ""

    return {
        "summary":  new_summary,
        "messages": [RemoveMessage(id=m.id) for m in to_compress],
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def dispatch_text_tools(state: AgentState) -> dict:
    """Execute tool calls embedded as <tool_call> XML in the AI's text response."""
    last    = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    calls   = _parse_text_tool_calls(content)
    results = []
    for call in calls:
        fn = TOOL_MAP.get(call["name"])
        if fn:
            try:
                output = fn.invoke(call["args"])
            except Exception as exc:
                output = f"[Error: {exc}]"
            results.append(ToolMessage(
                content=str(output),
                name=call["name"],
                tool_call_id=call["name"],
            ))
        else:
            # Return an explicit error so the model knows the call failed
            # instead of silently dropping it (which causes confusing retry loops).
            available = ", ".join(TOOL_MAP)
            results.append(ToolMessage(
                content=f"[Error: unknown tool '{call['name']}'. Available tools: {available}]",
                name=call["name"],
                tool_call_id=call["name"],
            ))
    return {"messages": results}


def _after_respond(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    if "<tool_call>" in (getattr(last, "content", "") or ""):
        return "dispatch_text_tools"
    return "summarize"


builder = StateGraph(AgentState)
builder.add_node("classify", classify)
builder.add_node("clarify",  clarify)
builder.add_node("rag",      rag)
builder.add_node("respond",              respond)
builder.add_node("tools",               ToolNode(TOOLS))
builder.add_node("dispatch_text_tools", dispatch_text_tools)
builder.add_node("summarize",           summarize)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, {
    "clarify": "clarify",
    "rag":     "rag",
    "respond": "respond",
})
builder.add_edge("rag",     "respond")
builder.add_edge("clarify", END)
builder.add_conditional_edges("respond", _after_respond, {
    "tools":               "tools",
    "dispatch_text_tools": "dispatch_text_tools",
    "summarize":           "summarize",
})
builder.add_edge("tools",               "respond")
builder.add_edge("dispatch_text_tools", "respond")
builder.add_edge("summarize",           END)

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

MENU       = [("F1", "Chat"), ("F2", "Chats"), ("F5", "Compact"), ("F10", "Help"), ("F12", "Settings")]
VIEW_HOME  = "home"
VIEW_CHAT  = "chat"
VIEW_SETTINGS = "settings"
VIEW_HELP  = "help"

ROLE_PAIR   = {"user": 2, "ai": 3, "router": 4, "tool": 5, "rag": 6}
ROLE_PREFIX = {"user": "You: ", "ai": "AI:  ", "router": "", "tool": "", "rag": ""}

F_AGENT_ADDR        = 0
F_AGENT_PROMPT      = 1
F_AGENT_CWD         = 2
F_RESEARCHER_ADDR   = 3
F_RESEARCHER_PROMPT = 4
F_CLASSIFIER_ADDR   = 5
F_CLASSIFIER_PROMPT = 6
# Lovense settings fields — token/uid/port/host for toy pairing.
F_LOVENSE_TOKEN     = 7
F_LOVENSE_UID       = 8
F_LOVENSE_PORT      = 9
F_LOVENSE_HOST      = 10
# TLS cert paths for HTTPS callback (certbot fullchain.pem / privkey.pem).
F_LOVENSE_CERT      = 11
F_LOVENSE_KEY       = 12
NUM_FIELDS          = 13

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App:
    def __init__(self, stdscr, graph):
        self.stdscr    = stdscr
        self.graph     = graph
        self.view      = VIEW_HOME
        self.prev_view = VIEW_HOME
        self.thread_id = "default"
        self.history   = []
        self.input_buf    = ""
        self.input_cursor = 0
        self.thinking     = False
        self.chat_scroll  = 0
        self._monitor_data: dict | list = {}
        self._gpu_history: deque = deque(maxlen=20)  # 20 × 0.5s = 10s rolling window
        self._log_lines: list[str] = []
        self._pipeline_log: list[str] = []   # rolling timestamped pipeline event log
        self._stop_event    = threading.Event()
        self._stream_queue  = queue.Queue()
        self._cancel_event  = threading.Event()
        self._ai_idx: int | None = None
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
        self._chats_list: list[dict] = []   # {"thread_id": str, "count": int}
        self._chats_cursor = 0
        self._db_path = "chat_history.db"

    def _bufs_from_settings(self) -> list[str]:
        lv = SETTINGS.get("lovense", {})
        return [
            SETTINGS["agent"]["address"],
            SETTINGS["agent"]["system_prompt"],
            SETTINGS["agent"].get("cwd", ""),
            SETTINGS["researcher"]["address"],
            SETTINGS["researcher"]["system_prompt"],
            SETTINGS["classifier"]["address"],
            SETTINGS["classifier"]["system_prompt"],
            # Lovense fields (indices 7-10)
            lv.get("token", ""),
            lv.get("uid", ""),
            lv.get("callback_port", "34569"),
            lv.get("callback_host", ""),
            lv.get("cert_file", ""),
            lv.get("key_file",  ""),
        ]

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
        self.stdscr.timeout(1000)
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
        if key == curses.KEY_F1:
            self.view = VIEW_CHAT
        elif key == curses.KEY_F2:
            self._open_chats_modal()
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
                    self.stdscr.timeout(1000)
                    if lookahead != curses.ERR:
                        curses.ungetch(lookahead)
                        self.input_buf    = self.input_buf[:pos] + "\n" + self.input_buf[pos:]
                        self.input_cursor = pos + 1
                    else:
                        self._send()
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    if pos > 0:
                        self.input_buf    = self.input_buf[:pos - 1] + self.input_buf[pos:]
                        self.input_cursor = pos - 1
                elif key == curses.KEY_DC:
                    if pos < len(self.input_buf):
                        self.input_buf = self.input_buf[:pos] + self.input_buf[pos + 1:]
                elif key == curses.KEY_LEFT:
                    self.input_cursor = max(0, pos - 1)
                elif key == curses.KEY_RIGHT:
                    self.input_cursor = min(len(self.input_buf), pos + 1)
                elif key == curses.KEY_HOME:
                    self.input_cursor = 0
                elif key == curses.KEY_END:
                    self.input_cursor = len(self.input_buf)
                elif key == curses.KEY_UP:
                    vlines = self._compute_visual_lines(self.input_buf, field_w)
                    row, col = self._cursor_to_visual(vlines, pos)
                    if row > 0:
                        start, line = vlines[row - 1]
                        self.input_cursor = start + min(col, len(line))
                elif key == curses.KEY_DOWN:
                    vlines = self._compute_visual_lines(self.input_buf, field_w)
                    row, col = self._cursor_to_visual(vlines, pos)
                    if row < len(vlines) - 1:
                        start, line = vlines[row + 1]
                        self.input_cursor = start + min(col, len(line))
                elif 32 <= key <= 126:
                    self.input_buf    = self.input_buf[:pos] + chr(key) + self.input_buf[pos:]
                    self.input_cursor = pos + 1
        return True

    def _handle_settings_key(self, key) -> None:
        idx       = self.settings_focus
        buf       = self.settings_bufs[idx]
        pos       = self.settings_cursors[idx]
        is_prompt = idx in (F_AGENT_PROMPT, F_RESEARCHER_PROMPT, F_CLASSIFIER_PROMPT)

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
        elif 32 <= key <= 126:
            self.settings_bufs[idx]    = buf[:pos] + chr(key) + buf[pos:]
            self.settings_cursors[idx] = pos + 1

    def _commit_settings(self) -> None:
        SETTINGS["agent"]["address"]            = self.settings_bufs[F_AGENT_ADDR]
        SETTINGS["agent"]["system_prompt"]      = self.settings_bufs[F_AGENT_PROMPT]
        SETTINGS["agent"]["cwd"]               = self.settings_bufs[F_AGENT_CWD]
        SETTINGS["researcher"]["address"]       = self.settings_bufs[F_RESEARCHER_ADDR]
        SETTINGS["researcher"]["system_prompt"] = self.settings_bufs[F_RESEARCHER_PROMPT]
        SETTINGS["classifier"]["address"]       = self.settings_bufs[F_CLASSIFIER_ADDR]
        SETTINGS["classifier"]["system_prompt"] = self.settings_bufs[F_CLASSIFIER_PROMPT]
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
                SELECT thread_id, COUNT(*) AS cnt, MAX(checkpoint_id) AS latest
                FROM   checkpoints
                GROUP  BY thread_id
                ORDER  BY latest DESC
            """)
            rows = cur.fetchall()
            conn.close()
            self._chats_list = [{"thread_id": r[0], "count": r[1]} for r in rows]
        except Exception:
            self._chats_list = []
        # Always include the active thread even if the DB is empty or unreachable
        existing_ids = {e["thread_id"] for e in self._chats_list}
        if self.thread_id not in existing_ids:
            self._chats_list.insert(0, {"thread_id": self.thread_id, "count": 0})
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
            new_id                 = "chat_" + time.strftime("%Y%m%d_%H%M%S")
            self.thread_id         = new_id
            self.history           = []
            self.chat_scroll       = 0
            self._chats_modal_open = False
            self.view              = VIEW_CHAT

        return True

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
        title = "\u2500\u2500 Chats \u2500\u2500"   # ── Chats ──
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

            prefix  = "\u25cf "  if is_current else "  "   # ● marks the active thread
            tid     = entry["thread_id"]
            cnt_str = f"[{entry['count']}]"
            avail   = inner_w - len(cnt_str)
            label   = (prefix + tid)[:avail].ljust(avail)
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
            self.stdscr.addstr(y0 + modal_h - 2, x0 + 1, "\u2500" * inner_w, bar)
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
        try:
            self.stdscr.addstr(y0 + modal_h - 1, x0 + 1, footer[:inner_w], bar)
        except curses.error:
            pass

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

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
        text = self.input_buf.strip()
        if not text:
            return
        self.input_buf    = ""
        self.input_cursor = 0
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
        self.thinking      = True
        self._cancelling   = False
        self._ignore_stream = False
        self._log_lines    = []
        self._pipeline_log = []                   # fresh log for each new request
        self._ai_idx       = None
        self._last_queue_event = time.monotonic()
        self._cancel_event.clear()
        self._flush_queue()
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
        try:
            pending_ai  = False
            first_chunk = False   # gate for the 'ai: first token' log entry
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
                        elif node_name == "rag":
                            # Surface what the RAG agent retrieved (shown in pastel
                            # yellow). Isolate just the [RAG: ...] segment so we don't
                            # repeat the classify note, and include the context text.
                            note = update.get("routing_note", "")
                            ctx  = update.get("rag_context", "")
                            idx  = note.rfind("[RAG")
                            rag_note = note[idx:] if idx != -1 else "[RAG]"
                            q.put(("rag", rag_note, ctx))
                        elif node_name in ("respond", "clarify"):
                            for msg in update.get("messages", []):
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
                            for msg in update.get("messages", []):
                                q.put(("tool", getattr(msg, "name", "tool"), getattr(msg, "content", "")))
                            q.put(("ai_next",))
                            pending_ai = False
                        elif node_name == "dispatch_text_tools":
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
                    if metadata.get("langgraph_node", "") in ("respond", "clarify") and content:
                        if not first_chunk:
                            first_chunk = True
                            q.put(("plog", "ai: first token"))
                        q.put(("ai_start", content) if not pending_ai else ("ai_append", content))
                        pending_ai = True

        except Exception as exc:
            q.put(("error", str(exc)))
            return
        q.put(("done",))

    def _stream_watchdog(self) -> None:
        """If no queue activity for 60 s while thinking, the stream is stalled — force-complete."""
        while self.thinking:
            time.sleep(2)
            if self.thinking and (time.monotonic() - self._last_queue_event) > 60:
                self._stream_queue.put(("error", "Stream stalled — no activity for 60s"))
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
                text = f"{note}\n{ctx}" if ctx else note
                self._plog("rag: context retrieved")
                self.history.append(("rag", text))
                self._ai_idx = None
                needs_redraw = True
            elif kind == "tool":
                self._plog(f"tool: {event[1]}")
                self.history.append(("tool", f"[Tool: {event[1]}]\n{event[2]}"))
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
                    if cleaned:
                        self.history[self._ai_idx] = (role, cleaned)
                    else:
                        self.history.pop(self._ai_idx)
                        self._ai_idx = None
                    needs_redraw = True
            elif kind == "token_usage":
                self._last_token_usage = event[1]
                inp = event[1].get("input_tokens",  0)
                out = event[1].get("output_tokens", 0)
                self._plog(f"tokens in={inp:,} out={out:,}")
                needs_redraw = True
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
            elif kind == "error":
                self._plog(f"ERR: {event[1][:55]}")
                self.history.append(("router", f"[Error: {event[1]}]"))
                self.thinking    = False
                self._cancelling = False
                self._ai_idx     = None
                needs_redraw     = True
                # Also stop the toy on errors so it doesn't run indefinitely.
                try:
                    import lovense as _lv
                    _lv.deactivate()
                except Exception:
                    pass
            elif kind == "lovense_msg":
                # Result from _lovense_connect() running off the UI thread.
                self.history.append(("router", event[1]))
                needs_redraw = True
        if needs_redraw:
            self.stdscr.erase()
            self._draw_menubar()
            self._draw_chat()
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

        threading.Thread(target=_poll_system, daemon=True).start()
        threading.Thread(target=_poll_gpu,   daemon=True).start()

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
        # Right-aligned Lovense connection indicator.
        # Shows "Lvns:●" (connected) or "Lvns:○" (not connected) so the user
        # can see at a glance whether a toy is paired without opening settings.
        try:
            import lovense as _lv
            connected  = _lv.is_connected()
            lv_label   = " Lvns:[ON] " if connected else " Lvns:[--] "
            lv_col     = max(0, w - len(lv_label) - 1)
            self.stdscr.addstr(0, lv_col, lv_label)
        except Exception:
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
        blue  = curses.color_pair(3)

        # ── Section header helper ────────────────────────────────────────────
        def _hdr(y: int, title: str) -> None:
            self.stdscr.attron(bar)
            try:
                hdr = f" {title} "
                self.stdscr.addstr(y, 1, hdr)
                self.stdscr.addstr(y, 1 + len(hdr), " " * (w - 2 - len(hdr)))
            except curses.error:
                pass
            self.stdscr.attroff(bar)

        # ── Row helper ───────────────────────────────────────────────────────
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
                ("F5",                  "Compact context (summarise history)"),
                ("F10",                 "This help screen"),
                ("F12",                 "Settings (addresses, prompts, Lovense)"),
                ("ESC",                 "Cancel stream / close view / quit"),
                ("PgUp / PgDn",         "Scroll chat history"),
                ("Mouse wheel",         "Scroll chat history"),
            ]),
            ("Chat input", [
                ("Enter",               "Send message"),
                ("Shift+Enter / paste", "Insert newline (multi-line input)"),
                ("\u2190 \u2192 Home End",       "Move cursor in input field"),
                ("\u2191 \u2193",                 "Move cursor between visual lines"),
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

        _header(y, "Classifier")
        y += 2
        self._draw_field(y, "  Address:       ", F_CLASSIFIER_ADDR,   LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  System Prompt: ", F_CLASSIFIER_PROMPT, LABEL_W, field_w, multiline=True)
        y += 4

        # ── Lovense section ──────────────────────────────────────────────────
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
            ch   = text[cur_col] if cur_col < len(text) else " "
            col  = min(cur_col, field_w - 1)
            try:
                self.stdscr.addstr(y, x + col, ch, curses.A_REVERSE)
            except curses.error:
                pass

    def _draw_chat(self):
        h, w      = self.stdscr.getmaxyx()
        INPUT_H   = 3
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

        field_w = w - 3  # input always spans full width

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

        total      = len(lines)
        max_scroll = max(0, total - chat_h)
        self.chat_scroll = min(self.chat_scroll, max_scroll)
        end     = total - self.chat_scroll
        start   = max(0, end - chat_h)
        visible = lines[start:end]

        # Horizontal separator (full width)
        self.stdscr.attron(curses.color_pair(1))
        try:
            note = f" ↑ {self.chat_scroll} " if self.chat_scroll > 0 else ""
            sep  = ("-" * (w - 1 - len(note))) + note if note else "-" * (w - 1)
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

        if self.thinking:
            log_display = self._log_lines[-(INPUT_H - 1):] if self._log_lines else ["thinking..."]
            for li, log_text in enumerate(log_display):
                try:
                    self.stdscr.addstr(input_top + li, 0, f"  {log_text.strip()}"[:w - 1])
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
        # Split the panel vertically: top half = system stats, bottom = pipeline log
        mid     = top + (bottom - top) // 2

        # ── Top half: system stats ─────────────────────────────────────────
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
                if row >= mid:
                    break
                try:
                    self.stdscr.addstr(row, panel_x, line, pair)
                except curses.error:
                    pass
                row += 1

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
        fill_pct = min(1.0, inp_tok / CTX_MAX)
        bar_w    = avail_w
        filled   = int(fill_pct * bar_w)
        empty    = bar_w - filled
        bar_str  = "\u2588" * filled + "\u2591" * empty   # █ … ░

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

    def _avg_gpu(self, gpu_idx: int, key: str) -> float | None:
        samples = [
            snap[gpu_idx].get(key)
            for snap in self._gpu_history
            if gpu_idx < len(snap) and snap[gpu_idx].get(key) is not None
        ]
        return sum(samples) / len(samples) if samples else None

    def _format_monitor_lines(self, data: dict | list | str, width: int) -> list[str]:
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

                util  = self._avg_gpu(i, "util_percent")
                power = self._avg_gpu(i, "power_watts")
                temp  = self._avg_gpu(i, "temperature_c")
                clock = self._avg_gpu(i, "core_clock_mhz")
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
        result = []
        for paragraph in text.split("\n"):
            if not paragraph:
                result.append("")
                continue
            while len(paragraph) > width:
                split = paragraph.rfind(" ", 0, width)
                if split <= 0:
                    split = width
                result.append(paragraph[:split])
                paragraph = "  " + paragraph[split:].lstrip()
            result.append(paragraph)
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with SqliteSaver.from_conn_string("chat_history.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        def _run(stdscr):
            App(stdscr, graph).run()

        curses.wrapper(_run)


if __name__ == "__main__":
    main()
