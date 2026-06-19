# -*- coding: utf-8 -*-
"""
Quick Lovense diagnostic — run this directly in PowerShell:
  python lovense_test.py

It reads .lovense_state.json and settings.json then tries every plausible
command path, printing exactly what succeeds or fails.
"""
import json, os, ssl, sys, urllib.request, urllib.error

STATE_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lovense_state.json")
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

# ── Load state ───────────────────────────────────────────────────────────────
print("=== Lovense diagnostic ===\n")
try:
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"State file   : {STATE_FILE}")
    print(f"domain       : {state.get('domain', '(empty)')}")
    print(f"port         : {state.get('port',   '(empty)')}")
    print(f"uid          : {state.get('uid',    '(empty)')}")
    print(f"toys         : {list(state.get('toys', {}).keys()) or '(none)'}")
except FileNotFoundError:
    print("No .lovense_state.json — run /lovense connect and scan the QR first.")
    sys.exit(1)

try:
    with open(SETTINGS_FILE) as f:
        settings = json.load(f)
    token = settings.get("lovense", {}).get("token", "")
    uid   = settings.get("lovense", {}).get("uid",   "")
    print(f"dev token    : {'(set)' if token else '(empty)'}")
except Exception as e:
    token = uid = ""
    print(f"Could not read settings: {e}")

domain = state.get("domain", "")
port   = state.get("port",   "")
uid    = uid or state.get("uid", "")

local_payload = json.dumps({"command":"Function","action":"Vibrate:10","timeSec":3,"apiVer":1}).encode()
local_headers = {"Content-Type":"application/json","X-platform":"SakuraLang"}

server_payload = json.dumps({"token":token,"uid":uid,"command":"Function","action":"Vibrate:10","timeSec":3,"apiVer":1}).encode()
server_headers = {"Content-Type":"application/json","User-Agent":"Mozilla/5.0"}

# ── Try cloud server API first (works regardless of network topology) ────────
print(f"\n--- Attempt 1: Lovense cloud server API ---")
if not token or not uid:
    print("  SKIP: no token/uid in settings.json")
else:
    try:
        req = urllib.request.Request("https://api.lovense-api.com/api/lan/v2/command",
                                     data=server_payload, headers=server_headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode()
        print(f"  Response: {body}")
        result = json.loads(body)
        if result.get("code") in (200, 0):
            print("\n  SUCCESS via cloud API! Toy should be vibrating for 3 seconds.")
            sys.exit(0)
        else:
            print(f"  Non-OK code — continuing to local attempts")
    except urllib.error.HTTPError as e:
        try:   detail = e.read().decode()
        except: detail = ""
        print(f"  HTTP {e.code}: {detail}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

# ── Try local Lovense Remote app path ────────────────────────────────────────
if not domain or not port:
    print("\nNo domain/port — cannot try local path.")
    sys.exit(1)

for scheme in ("https", "http"):
    url = f"{scheme}://{domain}:{port}/command"
    print(f"\n--- Attempt: local {url} ---")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        kw = {"timeout":6,"context":ctx} if scheme=="https" else {"timeout":6}
        req = urllib.request.Request(url, data=local_payload, headers=local_headers, method="POST")
        with urllib.request.urlopen(req, **kw) as r:
            body = r.read().decode()
        print(f"  SUCCESS  HTTP {r.status}: {body}")
        sys.exit(0)
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")

print("\n=== All attempts failed. ===")

import json, os, ssl, sys, urllib.request, urllib.error

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lovense_state.json")

# ── Load state ───────────────────────────────────────────────────────────────
print("=== Lovense diagnostic ===\n")
try:
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"State file   : {STATE_FILE}")
    print(f"domain       : {state.get('domain', '(empty)')}")
    print(f"port         : {state.get('port',   '(empty)')}")
    print(f"uid          : {state.get('uid',    '(empty)')}")
    print(f"toys         : {list(state.get('toys', {}).keys()) or '(none)'}")
except FileNotFoundError:
    print("No .lovense_state.json found — app hasn't received a callback yet.")
    print("Run /lovense connect inside the app and scan the QR code first.")
    sys.exit(1)
except Exception as e:
    print(f"Could not read state file: {e}")
    sys.exit(1)

domain = state.get("domain", "")
port   = state.get("port",   "")

if not domain or not port:
    print("\nERROR: domain or port is empty — pair the toy first (/lovense connect).")
    sys.exit(1)

# ── Build the command ────────────────────────────────────────────────────────
url     = f"https://{domain}:{port}/command"
payload = json.dumps({
    "command": "Function",
    "action":  "Vibrate:10",
    "timeSec": 3,
    "apiVer":  1,
}).encode()
headers = {
    "Content-Type": "application/json",
    "X-platform":   "SakuraLang",
}

print(f"\nSending test vibrate (3 s) to: {url}\n")

# ── Try 1: full TLS verification ─────────────────────────────────────────────
print("--- Attempt 1: TLS with cert verification ---")
try:
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=8) as r:
        body = r.read().decode()
    print(f"SUCCESS  HTTP {r.status}  body: {body}")
    print("\nToy should be vibrating for 3 seconds!")
    sys.exit(0)
except urllib.error.HTTPError as e:
    try:   detail = e.read().decode()
    except: detail = "(no body)"
    print(f"HTTP error {e.code}: {detail}")
except ssl.SSLError as e:
    print(f"SSL error: {e}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# ── Try 2: TLS, no cert verification ────────────────────────────────────────
print("\n--- Attempt 2: TLS, cert verification disabled ---")
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=8, context=ctx) as r:
        body = r.read().decode()
    print(f"SUCCESS  HTTP {r.status}  body: {body}")
    print("\nToy should be vibrating for 3 seconds!")
    sys.exit(0)
except urllib.error.HTTPError as e:
    try:   detail = e.read().decode()
    except: detail = "(no body)"
    print(f"HTTP error {e.code}: {detail}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# ── Try 3: plain HTTP (some setups use httpPort instead of httpsPort) ────────
http_url = f"http://{domain}:{port}/command"
print(f"\n--- Attempt 3: plain HTTP ({http_url}) ---")
try:
    req = urllib.request.Request(http_url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=8) as r:
        body = r.read().decode()
    print(f"SUCCESS  HTTP {r.status}  body: {body}")
    print("\nToy should be vibrating!  NOTE: use httpPort not httpsPort.")
    sys.exit(0)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# ── Try 4: httpsPort is actually httpPort (common mixup) ────────────────────
alt_port = state.get("toys", {})   # in case httpPort was stored differently
print(f"\n--- Attempt 4: common alt ports (34567 / 30010 / 20010) ---")
for alt in ["34567", "30010", "20010"]:
    if alt == port:
        continue
    for scheme in ("https", "http"):
        alt_url = f"{scheme}://{domain}:{alt}/command"
        print(f"  trying {alt_url} ...", end=" ", flush=True)
        try:
            ctx2 = ssl.create_default_context()
            ctx2.check_hostname = False
            ctx2.verify_mode    = ssl.CERT_NONE
            req = urllib.request.Request(alt_url, data=payload, headers=headers, method="POST")
            kw  = {"timeout": 4, "context": ctx2} if scheme == "https" else {"timeout": 4}
            with urllib.request.urlopen(req, **kw) as r:
                body = r.read().decode()
            print(f"SUCCESS  HTTP {r.status}  body: {body}")
            print(f"\nToy should be vibrating!  Use port {alt} with {scheme}.")
            sys.exit(0)
        except Exception as e:
            print(f"{type(e).__name__}: {e}")

print("\n=== All attempts failed. ===")
print("Make sure:")
print("  1. Lovense Remote app is open on your phone / PC")
print("  2. The toy is turned on and connected in the app")
print("  3. This machine and the phone are on the same Wi-Fi network")
print(f"  4. Nothing is blocking port {port} (check Windows Firewall)")
