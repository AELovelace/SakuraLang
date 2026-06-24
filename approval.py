# -*- coding: utf-8 -*-
import threading

_event         = threading.Event()
_result        = [False]
_pending: dict = {}
_notify_queue  = None


def set_notify_queue(q) -> None:
    global _notify_queue
    _notify_queue = q


def request_approval(name: str, args: dict) -> bool:
    """Called from graph worker thread. Blocks until main thread resolves (60s timeout = reject)."""
    _pending.clear()
    _pending.update({"name": name, "args": args})
    _result[0] = False
    _event.clear()
    if _notify_queue is not None:
        _notify_queue.put(("tool_approval_needed", name, dict(args)))
    _event.wait(timeout=60)
    _pending.clear()
    return _result[0]


def resolve(approved: bool) -> None:
    """Called from main (UI) thread to unblock the worker thread."""
    _result[0] = approved
    _event.set()


def get_pending() -> dict:
    return dict(_pending)
