#!/usr/bin/env python3
"""
api_test.py - Test-Script für das lmsmodel REST Gateway
Simuliert API-Anfragen und zeigt die Ergebnisse an.

Verwendung:
    python api_test.py                        # Status + Modell-Liste
    python api_test.py --set 14               # Modell 14 als aktiv setzen
    python api_test.py --infer 14             # Inferenz mit Model ID 14
    python api_test.py --infer-active         # Inferenz mit aktivem Modell
    python api_test.py --host 192.168.1.5     # Anderer Host
    python api_test.py --all 14               # Setzt Modell + Inferenz-Test
"""
import json
import sys
import argparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# --- Farben ---
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def banner(text):
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{RESET}\n")

def ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")

def fail(msg):
    print(f"  {RED}✗{RESET} {msg}")

def info(msg):
    print(f"  {YELLOW}ℹ{RESET} {msg}")


def api_get(base_url, path):
    url = f"{base_url}{path}"
    try:
        req = Request(url)
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except HTTPError as e:
        fail(f"HTTP {e.code}: {e.reason}")
        return None
    except URLError as e:
        fail(f"Connection failed: {e.reason}")
        return None


def api_post(base_url, path, payload, timeout=600):
    url = f"{base_url}{path}"
    data = json.dumps(payload).encode('utf-8')
    try:
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except HTTPError as e:
        fail(f"HTTP {e.code}: {e.reason}")
        return None
    except URLError as e:
        fail(f"Connection failed: {e.reason}")
        return None


def test_status(base_url):
    banner("Server Status")
    result = api_get(base_url, "/api/v1/status")
    if result and result.get("status") == "success":
        ok(f"Server läuft!")
        active = result.get("active_model_id")
        if active:
            ok(f"Aktives Modell: ID {active}")
        else:
            info("Kein aktives Modell gesetzt.")
        return True
    else:
        fail("Server antwortet nicht!")
        info("Starte ihn mit: python lmsmodel.py server start")
        return False


def test_models(base_url):
    banner("Modell-Liste")
    result = api_get(base_url, "/api/v1/models")
    if not result or result.get("status") != "success":
        fail("Konnte Modell-Liste nicht abrufen")
        return []

    models = result.get("models", [])
    ok(f"{len(models)} Modelle gefunden:\n")

    print(f"  {'ID':>4}  {'Status':>8}  {'Size':>6}  {'Stab.':>6}  Modell-Key")
    print(f"  {'─'*4}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*40}")
    
    for m in models:
        mid = m.get("id", "?")
        key = m.get("model_key", "?")
        status = m.get("status", "?")
        cfg = m.get("config", {})
        size = cfg.get("param_size_b", "?")
        stab = cfg.get("stability_timeout", "?")
        
        color = GREEN if status == "loaded" else RESET
        size_str = f"{size}B" if isinstance(size, (int, float)) else "?"
        stab_str = f"{stab}s" if isinstance(stab, (int, float)) else "?"
        
        print(f"  {color}{mid:>4}  {status:>8}  {size_str:>6}  {stab_str:>6}  {key}{RESET}")

    print()
    return models


def test_set_model(base_url, model_id):
    banner(f"Modell setzen (ID: {model_id})")
    info(f"Setze Modell {model_id} als aktiv und lade es in den VRAM...")

    result = api_post(base_url, "/api/v1/model/set", {"model_id": model_id}, timeout=120)
    if result and result.get("status") == "success":
        active = result.get("active_model", {})
        ok(f"Aktives Modell: {active.get('model_key', '?')}")
        cfg = active.get("config", {})
        if cfg:
            info(f"  Größe: ~{cfg.get('param_size_b', '?')}B")
            info(f"  Stability Timeout: {cfg.get('stability_timeout', '?')}s")
            info(f"  Temperature: {cfg.get('temperature', '?')}")
        return True
    else:
        fail("Modell konnte nicht gesetzt werden!")
        return False


def test_active_model(base_url):
    banner("Aktives Modell")
    result = api_get(base_url, "/api/v1/model/active")
    if result and result.get("status") == "success":
        active = result.get("active_model")
        if active:
            ok(f"Modell ID: {active.get('id')}")
            ok(f"Key: {active.get('model_key')}")
            cfg = active.get("config", {})
            info(f"  Stability: {cfg.get('stability_timeout', '?')}s | Temp: {cfg.get('temperature', '?')}")
        else:
            info("Kein aktives Modell gesetzt.")
            info("Setze eins mit: python api_test.py --set <MODEL_ID>")
        return True
    return False


def test_infer(base_url, model_id=None, prompt="Hallo, wie geht es dir?"):
    label = f"Model ID: {model_id}" if model_id else "Aktives Modell"
    banner(f"Inferenz ({label})")
    info(f"Prompt: \"{prompt}\"")
    info("Warte auf Antwort...")
    print()

    payload = {"prompt": prompt, "mode": "fafr"}
    if model_id:
        payload["model_id"] = model_id

    result = api_post(base_url, "/api/v1/infer", payload, timeout=600)
    if result and result.get("status") == "success":
        response = result.get("response", "").strip()
        used_model = result.get("model_id", "?")
        ok(f"Antwort von Modell {used_model} ({len(response)} Zeichen):\n")
        print(f"  {CYAN}┌─ Modell-Antwort ─────────────────────────┐{RESET}")
        for line in response.split('\n'):
            print(f"  {CYAN}│{RESET} {line}")
        print(f"  {CYAN}└──────────────────────────────────────────┘{RESET}")
        return True
    else:
        fail("Inferenz fehlgeschlagen!")
        if result:
            fail(f"Error: {result.get('error', 'Unknown')}")
        return False


def main():
    parser = argparse.ArgumentParser(description="lmsmodel REST API Test Tool")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5050, help="Server port")
    parser.add_argument("--set", type=int, metavar="ID", help="Modell als aktiv setzen")
    parser.add_argument("--infer", type=int, metavar="ID", help="Inferenz mit Model ID")
    parser.add_argument("--infer-active", action="store_true", help="Inferenz mit aktivem Modell")
    parser.add_argument("--prompt", default="Hallo, wie geht es dir?", help="Prompt")
    parser.add_argument("--all", type=int, metavar="ID", help="Modell setzen + Inferenz-Test")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"\n{BOLD}lmsmodel API Test Tool{RESET}")
    print(f"Target: {base_url}\n")

    # Status
    if not test_status(base_url):
        sys.exit(1)

    # Model list
    models = test_models(base_url)

    # Set model
    if args.set:
        test_set_model(base_url, args.set)
        test_active_model(base_url)

    # Full test (set + infer)
    if args.all:
        test_set_model(base_url, args.all)
        test_infer(base_url, prompt=args.prompt)

    # Infer with specific model
    elif args.infer:
        test_infer(base_url, model_id=args.infer, prompt=args.prompt)

    # Infer with active model
    elif args.infer_active:
        test_active_model(base_url)
        test_infer(base_url, prompt=args.prompt)

    # No action specified
    elif not args.set:
        test_active_model(base_url)
        print()
        info("Befehle:")
        info("  --set <ID>          Modell als aktiv setzen")
        info("  --infer <ID>        Inferenz mit bestimmtem Modell")
        info("  --infer-active      Inferenz mit aktivem Modell")
        info("  --all <ID>          Modell setzen + Inferenz")
        info("  --prompt '...'      Eigenen Prompt verwenden")

    print()


if __name__ == "__main__":
    main()
