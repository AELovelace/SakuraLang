# -*- coding: utf-8 -*-
import ast
import json
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from langchain_core.tools import tool

from settings import SETTINGS, DEFAULT_SETTINGS
from prompts import WEB_SEARCH_TOOL_NAMES, WEB_SEARCH_HINTS

_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS         = 0x00000008


def _run_proc(cmd: list[str], cwd=None, timeout: int = 15) -> tuple[str, str]:
    """Run a subprocess with timeout, killing the full process tree if it stalls."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
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


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(str(value).strip())))
    except (TypeError, ValueError):
        return default


def _needs_web_research(user_text: str, intent: str, tools_needed: list[str]) -> bool:
    normalized_tools = {str(name).strip().lower() for name in tools_needed if str(name).strip()}
    if normalized_tools & WEB_SEARCH_TOOL_NAMES:
        return True
    lowered = user_text.lower()
    if any(hint in lowered for hint in WEB_SEARCH_HINTS):
        return True
    return intent == "research"


def _get_brave_settings() -> dict:
    return SETTINGS.get("brave", {})


def _normalize_brave_results(payload: dict) -> list[dict]:
    web = payload.get("web", {})
    raw_results = web.get("results", []) if isinstance(web, dict) else payload.get("results", [])
    normalized: list[dict] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        meta_url = item.get("meta_url", {})
        source = ""
        if isinstance(meta_url, dict):
            source = str(meta_url.get("hostname") or meta_url.get("netloc") or "").strip()
        normalized.append({
            "title": str(item.get("title") or "").strip(),
            "url": str(item.get("url") or "").strip(),
            "snippet": str(item.get("description") or item.get("snippet") or "").strip(),
            "age": str(item.get("age") or item.get("page_age") or "").strip(),
            "source": source,
        })
    return [item for item in normalized if item["title"] and item["url"]]


def _fallback_research_brief(results: list[dict]) -> str:
    summary_lines = []
    source_lines = []
    for item in results[:4]:
        source_label = item.get("source") or item.get("url", "")
        snippet = item.get("snippet", "")
        if snippet:
            summary_lines.append(f"- {item['title']}: {snippet}")
        else:
            summary_lines.append(f"- {item['title']} ({source_label})")
        source_lines.append(f"- {item['title']} — {item['url']}")
    if not summary_lines:
        summary_lines.append("- No strong web results were available.")
    return "Summary:\n" + "\n".join(summary_lines) + "\n\nSources:\n" + "\n".join(source_lines)


def _search_brave(query: str, count: int | None = None) -> dict:
    cfg = _get_brave_settings()
    api_key = str(cfg.get("api_key", "")).strip()
    if not api_key:
        raise RuntimeError("Brave API key is not configured")

    base_url = str(cfg.get("base_url", "")).strip() or DEFAULT_SETTINGS["brave"]["base_url"]
    count_value = count if count is not None else cfg.get("count", "5")
    params = {
        "q": query,
        "count": str(_clamp_int(count_value, default=5, minimum=1, maximum=10)),
        "country": str(cfg.get("country", "us")).strip() or "us",
        "search_lang": str(cfg.get("search_lang", "en")).strip() or "en",
        "safesearch": str(cfg.get("safesearch", "moderate")).strip() or "moderate",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": "SakuraLang/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Brave search HTTP {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Brave search network error: {exc.reason}") from exc

    results = _normalize_brave_results(payload)
    return {"query": query, "results": results}


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
def rag_search(query: str) -> str:
    """Search local documents and the knowledge base for relevant context.
    Prefer this over web search — use it first whenever local docs might have the answer."""
    from settings import SETTINGS as _S
    cwd = _S.get("agent", {}).get("cwd", "").strip()
    if not cwd:
        return "[RAG: no working directory configured — set one in Settings]"
    try:
        import rag as rag_engine
        rag_engine.ensure_indexed(cwd)
        context, sources = rag_engine.retrieve(query, cwd)
    except Exception as exc:
        return f"[RAG error: {exc}]"
    if not context:
        return "[RAG: no relevant matches found]"
    src_str = ", ".join(sources[:5])
    return f"Sources: {src_str}\n\n{context}"


@tool
def brave_web_search(query: str, count: int = 5) -> str:
    """Search the public web with Brave Search and return concise JSON results.
    Best for fresh information, recent changes, release notes, documentation, and news.
    Prefer rag_search first — only use this when local docs don't have the answer."""
    return json.dumps(_search_brave(query, count), ensure_ascii=False, indent=2)


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


@tool
def code_editor(path: str, mode: str = "read", old_string: str = "", new_string: str = "") -> str:
    """Read or edit a source file.
    mode='read'    — return the full file contents.
    mode='replace' — replace the first occurrence of old_string with new_string.
    mode='write'   — overwrite the entire file with new_string (creates parent dirs).
    Always use mode='read' before editing so you know the exact text to replace."""
    try:
        target = pathlib.Path(path).expanduser()
        if mode == "read":
            if not target.exists():
                return f"[Error: file not found: {target.resolve()}]"
            return target.read_text(encoding="utf-8", errors="replace")
        elif mode == "replace":
            if not target.exists():
                return f"[Error: file not found: {target.resolve()}]"
            content = target.read_text(encoding="utf-8", errors="replace")
            if old_string not in content:
                return f"[Error: old_string not found in {path}]"
            target.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
            return f"Edited: {target.resolve()}"
        elif mode == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_string, encoding="utf-8")
            return f"Written: {target.resolve()}"
        return f"[Error: unknown mode '{mode}' — use read, replace, or write]"
    except Exception as exc:
        return f"[Error: {exc}]"


TOOLS    = [run_powershell, run_python, launch_app, rag_search, brave_web_search, write_file, code_editor]
TOOL_MAP = {t.name: t for t in TOOLS}

# Tools available in Plan mode — read-only, no execution or file writes.
# rag_search is listed first so the model sees it as the preferred lookup tool.
PLAN_TOOLS      = [rag_search, brave_web_search]
PLAN_TOOL_NAMES = {t.name for t in PLAN_TOOLS}


def _parse_tool_arg_string(arg_text: str) -> dict:
    params: dict = {}
    arg_text = arg_text.strip()
    if not arg_text:
        return params
    try:
        parsed = ast.parse(f"f({arg_text})", mode="eval")
        call = parsed.body
        if isinstance(call, ast.Call):
            for kw in call.keywords:
                if kw.arg:
                    try:
                        value = ast.literal_eval(kw.value)
                    except Exception:
                        value = ""
                    params[kw.arg] = value
    except SyntaxError:
        pass
    return params


def _parse_text_tool_calls(content: str) -> list[dict]:
    """Parse tool calls emitted as XML or plain-text pseudo-calls."""
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
    if calls:
        return calls

    tool_names = "|".join(sorted(re.escape(name) for name in TOOL_MAP))
    pattern = rf"(?mi)^\s*(?P<name>{tool_names})\s*\((?P<args>.*)\)\s*$"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        calls.append({
            "name": match.group("name"),
            "args": _parse_tool_arg_string(match.group("args")),
        })
    return calls


def _contains_text_tool_call(content: str) -> bool:
    if "<tool_call>" in content:
        return True
    tool_names = "|".join(sorted(re.escape(name) for name in TOOL_MAP))
    return bool(re.search(rf"(?mi)^\s*(?:{tool_names})\s*\(", content))


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


def _extract_context_override(text: str) -> dict:
    data = _extract_json(text)
    if not data or not bool(data.get("context_override", False)):
        return {}
    mode = str(data.get("mode", "")).strip().lower()
    if mode not in ("web", "rag"):
        return {}
    return {
        "mode": mode,
        "reason": str(data.get("reason", "")).strip(),
        "query": str(data.get("query", "")).strip(),
    }


def _contains_context_override(content: str) -> bool:
    return bool(_extract_context_override(content))


def _read_clipboard() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.rstrip("\n")
    except Exception:
        return ""


def _truncate_log_text(text: str, limit: int = 80) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit - 3] + "..."


def _clip_display_text(text: str, limit: int, label: str) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    remaining = len(text) - limit
    return text[:limit] + f"\n[{label} truncated, {remaining:,} more chars omitted]"
