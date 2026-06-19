# -*- coding: utf-8 -*-
"""
lovense.py — Lovense Standard API integration for SakuraLang.

Handles three responsibilities:
  1. Running a small HTTP callback server to receive the toy-connection
     payload that Lovense Remote sends after a QR scan (Standard API Step 2).
  2. Requesting a pairing QR code from the Lovense developer API so the
     user can link their toy through the Lovense Remote app.
  3. Sending vibration start/stop commands to the locally-connected
     Lovense Remote app (Standard API Step 3 — "By local application").

Typical flow
────────────
  • User opens Lovense Remote and turns on their toy.
  • App calls get_qr(token, uid, callback_url) → returns a QR image URL.
  • User scans the QR with Lovense Remote → Remote POSTs the toy info
    to our callback server (running on localhost:<callback_port>).
  • App now has domain + httpsPort → activate() / deactivate() talk
    directly to Lovense Remote without going through the cloud.

Thread safety
─────────────
All mutable shared state is protected by _lock.  activate() and
deactivate() are safe to call from any thread.
"""

import json
import logging
import os
import socket
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)

# ── Lovense API endpoints ───────────────────────────────────────────
# QR code requests go to the Lovense cloud; toy commands talk directly to the
# local Lovense Remote app via the domain/port delivered by the callback.
_QR_ENDPOINT          = "https://api.lovense-api.com/api/lan/getQrCode"
# Server-side command endpoint — used when the local app path fails (e.g. the
# phone's domain resolves to localhost instead of its real LAN IP).
_CMD_SERVER_ENDPOINT  = "https://api.lovense-api.com/api/lan/v2/command"
_APP_PLATFORM         = "SakuraLang"   # Shown on the Lovense Remote screen

# Developer token — set via configure() so activate/deactivate can fall back
# to the cloud API without needing to import SETTINGS.
_dev_token: str = ""


def configure(dev_token: str) -> None:
    """Supply the developer token so server-side commands are available."""
    global _dev_token
    _dev_token = dev_token.strip()


# ── Shared mutable state (always access inside _lock) ───────────────────────
_lock             = threading.Lock()
_toy_domain:  str = ""   # e.g. "192-168-1-44.lovense.club"
_toy_port:    str = ""   # httpsPort, e.g. "34568"
_toys:       dict = {}   # { toy_id: {"name": str, "nickName": str, "id": str, "status": int} }
_uid:         str = ""
_utoken:      str = ""

# Path where we persist the last-known toy connection so it survives restarts.
_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lovense_state.json")


def _save_state() -> None:
    """Write current connection info to disk (called inside _lock)."""
    try:
        with open(_STATE_FILE, "w") as fh:
            json.dump({
                "domain": _toy_domain,
                "port":   _toy_port,
                "toys":   _toys,
                "uid":    _uid,
                "utoken": _utoken,
            }, fh)
    except Exception as exc:
        logger.debug("Could not save Lovense state: %s", exc)


def _load_state() -> None:
    """Restore connection info from disk if available (called at module import)."""
    global _toy_domain, _toy_port, _toys, _uid, _utoken
    try:
        with open(_STATE_FILE) as fh:
            data = json.load(fh)
        with _lock:
            _toy_domain = data.get("domain", "")
            _toy_port   = data.get("port",   "")
            _toys       = data.get("toys",   {})
            _uid        = data.get("uid",    "")
            _utoken     = data.get("utoken", "")
        if _toy_domain and _toy_port:
            logger.info("Lovense state restored: domain=%s port=%s", _toy_domain, _toy_port)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # no saved state — user must pair again
    except Exception as exc:
        logger.debug("Could not load Lovense state: %s", exc)


# Restore persisted state immediately so is_connected() works after a restart.
_load_state()

# ── Callback HTTP server state ───────────────────────────────────────────────
_server:        HTTPServer | None            = None
_server_thread: threading.Thread | None      = None


# ────────────────────────────────────────────────────────────────────────────
# Callback server
# ────────────────────────────────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    """
    Handles the single POST that Lovense Remote sends after the user scans
    the QR code.  The payload contains the toy domain, port, and toy list.
    """

    def do_GET(self):  # noqa: N802
        """Status page — visit http://<host>:<port>/ to confirm the server is reachable."""
        with _lock:
            connected = bool(_toy_domain and _toy_port)
            toys_list = [v.get("name", k) for k, v in _toys.items()]
        status = "connected" if connected else "waiting for pairing"
        body = (
            f"SakuraLang Lovense callback server\n"
            f"status : {status}\n"
            f"domain : {_toy_domain or '(none)'}\n"
            f"port   : {_toy_port   or '(none)'}\n"
            f"toys   : {', '.join(toys_list) or '(none)'}\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802  (Lovense API uses POST)
        # We accept any path — Lovense Remote POSTs to whatever URL we gave it.
        global _toy_domain, _toy_port, _toys, _uid, _utoken

        # Read the full body robustly.  Lovense Remote may omit Content-Length
        # or send chunked encoding, so we fall back to reading until the socket
        # idles rather than trusting the header.
        cl = self.headers.get("Content-Length")
        if cl is not None:
            body = self.rfile.read(int(cl))
        else:
            # No Content-Length — read in chunks until the connection goes quiet.
            chunks = []
            try:
                self.rfile._sock.settimeout(1.5)
                while True:
                    chunk = self.rfile.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
            except Exception:
                pass
            body = b"".join(chunks)

        # Always write the raw body to a debug log so we can inspect it even
        # if parsing fails.
        _debug_log = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".lovense_callback.log"
        )
        try:
            with open(_debug_log, "a", encoding="utf-8") as fh:
                import datetime
                fh.write(f"\n--- {datetime.datetime.now().isoformat()} ---\n")
                fh.write(f"path: {self.path}\n")
                fh.write(f"headers: {dict(self.headers)}\n")
                fh.write(f"body: {body!r}\n")
        except Exception:
            pass

        # Lovense Remote sends JSON, but try form-encoded as a fallback.
        data: dict = {}
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            try:
                from urllib.parse import parse_qs
                qs = parse_qs(body.decode(errors="replace"), keep_blank_values=True)
                data = {k: v[0] for k, v in qs.items()}
            except Exception:
                pass

        if not data:
            logger.warning(
                "Lovense callback: could not parse body (%d bytes): %r",
                len(body), body[:200],
            )
            # Return 200 anyway so Lovense Remote doesn't retry endlessly.
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"code":0}')
            return

        with _lock:
            _toy_domain = data.get("domain", "")
            _toy_port   = str(data.get("httpsPort", data.get("wssPort", "")))
            raw_toys = data.get("toys", {})
            if isinstance(raw_toys, str):
                try:
                    raw_toys = json.loads(raw_toys)
                except Exception:
                    raw_toys = {}
            _toys   = raw_toys
            _uid    = data.get("uid", "")
            _utoken = data.get("utoken", "")
            _save_state()

        toy_names = [v.get("name", k) for k, v in (raw_toys if isinstance(raw_toys, dict) else {}).items()]
        logger.info(
            "Lovense callback received: domain=%s port=%s toys=%s",
            _toy_domain, _toy_port, toy_names,
        )

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"code":0}')

    def log_message(self, fmt, *args):  # suppress noisy access-log output
        pass


def start_callback_server(
    port: int,
    certfile: str = "",
    keyfile:  str = "",
) -> None:
    """
    Start the callback server on *port*.

    If *certfile* and *keyfile* are provided (paths to a fullchain.pem and
    privkey.pem from certbot), the server runs HTTPS.  Otherwise plain HTTP
    is used (fine for localhost tests, but Lovense Remote requires HTTPS for
    real callbacks).

    Safe to call multiple times — does nothing if already running.
    """
    global _server, _server_thread

    if _server is not None:
        return  # already running

    use_tls = bool(certfile and keyfile)

    try:
        _server = HTTPServer(("0.0.0.0", port), _CallbackHandler)

        if use_tls:
            # Wrap the raw socket with TLS using the certbot-issued cert.
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
            _server.socket = ctx.wrap_socket(_server.socket, server_side=True)
            logger.info(
                "Lovense callback server listening on port %d (HTTPS, cert=%s)",
                port, certfile,
            )
        else:
            logger.info("Lovense callback server listening on port %d (HTTP)", port)

        _server_thread = threading.Thread(
            target=_server.serve_forever,
            daemon=True,
            name="lovense-callback",
        )
        _server_thread.start()
    except OSError as exc:
        logger.warning(
            "Could not start Lovense callback server on port %d: %s", port, exc
        )
        _server = None


def stop_callback_server() -> None:
    """Gracefully shut down the callback server."""
    global _server, _server_thread

    if _server is not None:
        _server.shutdown()
        _server = None
        _server_thread = None


# ────────────────────────────────────────────────────────────────────────────
# Public helpers
# ────────────────────────────────────────────────────────────────────────────

def get_local_ip() -> str:
    """
    Return the best-guess LAN IP of this machine.

    Works by opening a UDP socket toward a public address (no data is actually
    sent) and reading which local interface the OS would use.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def get_qr(token: str, uid: str, callback_url: str) -> dict:
    """
    Request a pairing QR code from the Lovense developer API.

    Parameters
    ──────────
    token        — your Lovense developer token (from the dashboard)
    uid          — any stable identifier for the user on your platform
    callback_url — URL that Lovense Remote will POST toy info to after
                   the QR scan (must be reachable from the user's phone,
                   e.g. "http://192.168.1.10:34569/")

    Returns a dict with at least:
        {"qr": "<QR image URL>", "code": "<PC pairing code>"}

    Raises RuntimeError on API or network failure.
    """
    # The Lovense getQrCode endpoint expects application/x-www-form-urlencoded
    # (the official Java SDK example uses UrlEncodedFormEntity, not JSON).
    # Sending JSON causes a 403 Forbidden response.
    payload = urllib.parse.urlencode({
        "token":       token,
        "uid":         uid,
        "uname":       uid,          # display name shown in Lovense Remote
        "utoken":      uid,          # per-user security token (keep it stable)
        "v":           "2",
        "callbackUrl": callback_url,
    }).encode()

    req = urllib.request.Request(
        _QR_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            # Cloudflare (which sits in front of the Lovense API) blocks requests
            # with the default Python-urllib user-agent (error 1010 — bot signature
            # detected).  A standard browser UA passes the check.
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as exc:
        # Read the response body — Lovense usually includes a useful error message
        # even on 4xx responses.  Surface it so the caller can show it to the user.
        try:
            detail = exc.read().decode()
        except Exception:
            detail = "(no body)"
        raise RuntimeError(
            f"Lovense API HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Lovense API unreachable: {exc}") from exc

    result = json.loads(body)

    # The Lovense API returns code=0 for success.
    if result.get("code") != 0:
        msg = result.get("message", "unknown error")
        raise RuntimeError(f"Lovense API error {result.get('code')}: {msg}")

    # "data" contains {"qr": url, "code": pairing_code}.
    return result.get("data", result)


def is_connected() -> bool:
    """True once a callback has been received with valid domain + port."""
    with _lock:
        return bool(_toy_domain and _toy_port)


def disconnect() -> None:
    """Clear all in-memory toy state and delete the persisted state file.

    Does NOT close the callback server — it keeps listening so the toy can
    re-pair by scanning a new QR code.
    """
    global _toy_domain, _toy_port, _toys, _uid, _utoken

    # Call deactivate() BEFORE acquiring _lock — deactivate() takes _lock
    # internally, so calling it inside our own with-_lock block would deadlock.
    deactivate()

    with _lock:
        _toy_domain = ""
        _toy_port   = ""
        _toys       = {}
        _uid        = ""
        _utoken     = ""
        # Remove persisted state so it does not reload on next startup
        try:
            import os
            if os.path.exists(_STATE_FILE):
                os.remove(_STATE_FILE)
        except Exception:
            pass
    logger.info("Lovense disconnected — state cleared")


def get_debug_info() -> str:
    """Return a human-readable dump of the current connection state."""
    with _lock:
        return (
            f"domain   : {_toy_domain or '(empty)'}\n"
            f"port     : {_toy_port   or '(empty)'}\n"
            f"uid      : {_uid        or '(empty)'}\n"
            f"toys     : {list(_toys.keys()) or '(none)'}\n"
            f"state file: {_STATE_FILE}"
        )


def get_toy_names() -> list[str]:
    """Return a human-readable list of every connected toy's name."""
    with _lock:
        names = []
        for tid, info in _toys.items():
            nick = (info.get("nickName") or "").strip()
            name = info.get("name", tid)
            names.append(nick if nick else name)
        return names


def activate(strength: int = 10, duration_sec: float = 0) -> None:
    """
    Start vibrating all connected toys.

    Tries the cloud server API first (works regardless of network topology),
    then falls back to the local Lovense Remote app path.

    strength     — vibration level 0-20 (Lovense API range)
    duration_sec — how long to run; 0 = indefinite (call deactivate() to stop)
    """
    with _lock:
        domain = _toy_domain
        port   = _toy_port
        uid    = _uid
        token  = _dev_token

    if not (domain and port):
        return   # no toy connected — silently skip

    local_payload = json.dumps({
        "command": "Function",
        "action":  f"Vibrate:{strength}",
        "timeSec": duration_sec,
        "apiVer":  1,
    }).encode()

    server_payload = json.dumps({
        "token":   token,
        "uid":     uid,
        "command": "Function",
        "action":  f"Vibrate:{strength}",
        "timeSec": duration_sec,
        "apiVer":  1,
    }).encode()

    threading.Thread(
        target=_send_command,
        args=(domain, port, local_payload, server_payload, token, uid),
        daemon=True,
        name="lovense-activate",
    ).start()


def deactivate() -> None:
    """Stop all toys immediately."""
    with _lock:
        domain = _toy_domain
        port   = _toy_port
        uid    = _uid
        token  = _dev_token

    if not (domain and port):
        return

    local_payload = json.dumps({
        "command": "Function",
        "action":  "Stop",
        "timeSec": 0,
        "apiVer":  1,
    }).encode()

    server_payload = json.dumps({
        "token":   token,
        "uid":     uid,
        "command": "Function",
        "action":  "Stop",
        "timeSec": 0,
        "apiVer":  1,
    }).encode()

    threading.Thread(
        target=_send_command,
        args=(domain, port, local_payload, server_payload, token, uid),
        daemon=True,
        name="lovense-deactivate",
    ).start()


# ────────────────────────────────────────────────────────────────────────────
# Internal
# ────────────────────────────────────────────────────────────────────────────

def _send_command(
    domain: str, port: str,
    local_payload: bytes, server_payload: bytes,
    token: str, uid: str,
) -> None:
    """
    Send a toy command, preferring the cloud server API and falling back to
    the local Lovense Remote app path.

    The server path works even when the phone's reported domain resolves to
    localhost (a common issue when the phone is on a different subnet or the
    Lovense Remote app can't determine its own LAN IP).
    """
    # ── Try cloud server API first ────────────────────────────────────────
    if token and uid:
        req = urllib.request.Request(
            _CMD_SERVER_ENDPOINT,
            data=server_payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read().decode()
            result = json.loads(body)
            if result.get("code") in (200, 0):
                logger.debug("Lovense server command OK: %s", body)
                return   # success — no need for local fallback
            logger.warning("Lovense server command non-OK: %s", body)
        except Exception as exc:
            logger.warning("Lovense server command failed: %s", exc)

    # ── Fall back to local Lovense Remote app path ────────────────────────
    _send_local_command(domain, port, local_payload)

def _send_local_command(domain: str, port: str, payload: bytes) -> None:
    """
    POST a command payload to the local Lovense Remote app.

    URL: https://{domain}:{port}/command
    The domain is a *.lovense.club hostname Lovense controls that resolves
    to the phone's LAN IP.  We try with full cert verification first; if
    that fails (some environments have trouble with the Lovense cert chain)
    we retry with verification disabled — acceptable because this is a
    purely local LAN connection.
    """
    url = f"https://{domain}:{port}/command"

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-platform":   _APP_PLATFORM,   # shown on the Lovense Remote screen
        },
        method="POST",
    )

    # Try with full cert verification first.
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=5, context=ctx) as _:
            return   # success
    except ssl.SSLError:
        pass   # fall through to no-verify retry
    except Exception as exc:
        logger.warning("Lovense command to %s:%s failed: %s", domain, port, exc)
        return

    # Retry without cert verification (handles Lovense cert chain edge-cases
    # on some systems).  Still uses TLS — just skips the CA check.
    try:
        ctx_noverify = ssl.create_default_context()
        ctx_noverify.check_hostname = False
        ctx_noverify.verify_mode    = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=5, context=ctx_noverify) as _:
            pass
    except Exception as exc:
        logger.warning("Lovense command (no-verify) to %s:%s failed: %s", domain, port, exc)
