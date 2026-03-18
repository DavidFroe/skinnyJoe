#!/usr/bin/env python3
"""
SkinnyJoe CLI v4.0 – Multi-Slot Interface

Befehle:
  models              Alle Modelle mit Profilen anzeigen
  slots               Slot-Status anzeigen
  gpus                NVIDIA GPUs anzeigen
  load <model> -s <slot>   Modell in Slot laden
  unload -s <slot>    Slot entladen
  ask -s <slot> "prompt"   Anfrage an Slot
  status              Gesamtstatus
  tui                 Interaktive TUI
  kill                Daemon beenden
  help                Vollständige Hilfe
"""
import argparse
import requests
import sys
import json
import os
import signal
import struct
import subprocess
import shutil
import time
import base64
from pathlib import Path

MGMT_URL = os.environ.get("SKINNYJOE_URL", "http://localhost:8000")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DAEMON_SCRIPT  = os.path.join(BASE_DIR, "skinnyJoe_daemon.py")
INSTALL_SCRIPT = os.path.join(BASE_DIR, "install.sh")
VENV_PYTHON  = os.path.join(BASE_DIR, "venv", "bin", "python3")
PID_FILE         = os.path.join(BASE_DIR, "skinnyjoe.pid")
LOG_FILE         = os.path.join(BASE_DIR, "skinnyjoe.log")
DEFAULT_SLOT_FILE = os.path.join(BASE_DIR, ".default_slot")

# ANSI
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_BG_GREEN = "\033[42m"
C_WHITE = "\033[97m"

TYPE_COLORS = {"text2text": C_BLUE, "image2text": C_MAGENTA, "text2image": C_CYAN,
               "speech2text": C_GREEN, "text2speech": C_YELLOW}
TYPE_LABELS = {"text2text": "Text-zu-Text (LLMs)", "image2text": "Bild-zu-Text (Vision)",
               "text2image": "Text-zu-Bild (Diffusion)", "speech2text": "Sprache-zu-Text (Whisper)",
               "text2speech": "Text-zu-Sprache (TTS)"}
TYPE_ORDER = ["text2text", "image2text", "text2image", "speech2text", "text2speech"]


def _get_default_slot() -> int:
    try:
        return int(open(DEFAULT_SLOT_FILE).read().strip())
    except Exception:
        return 1


def _set_default_slot(n: int):
    with open(DEFAULT_SLOT_FILE, "w") as f:
        f.write(str(n))


def _resolve_slot(slot_arg):
    """None → gespeicherter Standard-Slot (default: 1)."""
    return slot_arg if slot_arg is not None else _get_default_slot()


def api_get(url, path, timeout=5):
    try:
        r = requests.get(f"{url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print(f"{C_RED}Nicht erreichbar: {url}{C_RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{C_RED}Fehler: {e}{C_RESET}", file=sys.stderr)
        sys.exit(1)

def api_post(url, path, data=None, timeout=600):
    try:
        r = requests.post(f"{url}{path}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print(f"{C_RED}Nicht erreichbar: {url}{C_RESET}", file=sys.stderr)
        sys.exit(1)
    except requests.HTTPError as e:
        try: detail = e.response.json().get("detail", str(e))
        except Exception: detail = str(e)
        print(f"{C_RED}Fehler: {detail}{C_RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{C_RED}Fehler: {e}{C_RESET}", file=sys.stderr)
        sys.exit(1)

def _get_slot_port(slot_id):
    """Holt den Port eines Slots vom Management-Server."""
    data = api_get(MGMT_URL, "/v1/slots")
    for s in data.get("slots", []):
        if s["id"] == slot_id:
            return s["port"]
    print(f"{C_RED}Slot {slot_id} nicht gefunden.{C_RESET}", file=sys.stderr)
    sys.exit(1)

def _slot_url(slot_id):
    """Baut die URL für einen Slot-Port."""
    port = _get_slot_port(slot_id)
    # Gleicher Host wie Management, anderer Port
    from urllib.parse import urlparse
    parsed = urlparse(MGMT_URL)
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


# ============================================================
# Befehle
# ============================================================

def cmd_models():
    data = api_get(MGMT_URL, "/v1/models")
    models = data.get("data", [])
    if not models:
        print("Keine Modelle gefunden.")
        return

    by_type = {}
    for m in models:
        by_type.setdefault(m.get("model_type", "text2text"), []).append(m)

    print(f"\n  {C_BOLD}SkinnyJoe Modelle{C_RESET}  ({len(models)} insgesamt)")
    print(f"  {'─' * 76}")

    for mtype in TYPE_ORDER:
        group = by_type.get(mtype, [])
        if not group: continue
        color = TYPE_COLORS.get(mtype, C_WHITE)
        print(f"\n  {color}{C_BOLD}▸ {TYPE_LABELS.get(mtype, mtype)}{C_RESET}\n")
        for m in group:
            mid = m["id"]
            name = m.get("name", "?")
            size = m.get("size_gb", 0)
            quant = m.get("quant") or ""
            params = m.get("params") or ""
            profile = m.get("profile", {})
            gpu_ids = profile.get("gpu_ids", [])
            gpu_layers = profile.get("gpu_layers", 0)
            ctx = profile.get("ctx", "?")
            has_mmproj = m.get("has_mmproj", False)

            hw = f"GPU G{','.join(str(g) for g in gpu_ids)}" if gpu_ids else "CPU"
            if gpu_layers == -1: hw += " (all)"
            elif gpu_layers > 0: hw += f" ({gpu_layers}L)"

            tags = m.get("tags", [])
            tag_str = f"{C_DIM}[{','.join(t for t in tags if t not in ('vision','diffusion'))}]{C_RESET}" if tags else ""

            mmproj = ""
            if mtype == "image2text":
                mmproj = f" {C_GREEN}+mmproj{C_RESET}" if has_mmproj else f" {C_RED}!mmproj{C_RESET}"

            nr = f"{mid:<4}"
            print(f"    {color}{nr}{C_RESET} {name:<33s} {size:>6.1f}GB  {quant:<8s} {C_DIM}CTX:{ctx:<6}{C_RESET} {hw:<14s} {tag_str}{mmproj}")

    print(f"\n  {'─' * 76}")
    print(f"  {C_DIM}Laden: sj load <ID> --slot <S>  |  z.B. T6, D2, W1, B1, S1  |  Profile in config.json{C_RESET}\n")


def cmd_slots():
    data = api_get(MGMT_URL, "/v1/slots")
    slots = data.get("slots", [])

    print(f"\n  {C_BOLD}SkinnyJoe Slots{C_RESET}  ({len(slots)} konfiguriert)")
    print(f"  {'─' * 60}")

    for s in slots:
        sid = s["id"]
        port = s["port"]
        status = s["status"]
        loaded = s.get("loaded_model")

        if status == "loaded" and loaded:
            st_color = C_GREEN
            st_str = f"{loaded['name']} ({loaded['model_type']})"
        elif status == "generating":
            st_color = C_YELLOW
            st_str = f"{loaded['name']} [GENERIERT]" if loaded else "[GENERIERT]"
        else:
            st_color = C_DIM
            st_str = "leer"

        print(f"    Slot {C_CYAN}{sid}{C_RESET}  Port {port}  {st_color}■{C_RESET} {st_str}")

    print(f"\n  {'─' * 60}")
    hw = api_get(MGMT_URL, "/status").get("hardware_alloc", {})
    if hw:
        print(f"  {C_DIM}GPU-Belegung: {', '.join(f'{g}→{s}' for g, s in hw.items())}{C_RESET}")
    print(f"  {C_DIM}Laden: sj load <N> --slot <S>  |  Entladen: sj unload --slot <S>{C_RESET}\n")


def cmd_gpus():
    data = api_get(MGMT_URL, "/v1/gpus")
    gpus = data.get("gpus", [])
    print()
    if not gpus:
        print(f"  {C_YELLOW}Keine NVIDIA GPUs erkannt.{C_RESET}\n")
        return
    print(f"  {C_BOLD}NVIDIA GPUs{C_RESET}  ({len(gpus)} erkannt)")
    print(f"  {'─' * 60}\n")
    for g in gpus:
        pct = (g["vram_used_gb"] / g["vram_total_gb"] * 100) if g["vram_total_gb"] > 0 else 0
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        bc = C_RED if pct > 80 else C_YELLOW if pct > 50 else C_GREEN
        print(f"    {C_CYAN}G{g['id']}{C_RESET}  {g['name']}")
        print(f"        VRAM: {bc}{bar}{C_RESET} {g['vram_used_gb']:.1f}/{g['vram_total_gb']:.1f} GB ({pct:.0f}%)\n")
    print(f"  {'─' * 60}\n")


def _parse_ctx(s: str) -> int:
    """Parse context size string: '16K'→16384, '8192'→8192, '32k'→32768."""
    s = s.strip()
    if s.upper().endswith("K"):
        return int(float(s[:-1]) * 1024)
    return int(s)


def cmd_load(model_id, slot_id, ctx_override=None):
    print(f"Lade Modell {model_id.upper()} → Slot {slot_id}...")
    payload = {"slot_id": slot_id, "model_id": model_id}
    if ctx_override:
        payload["ctx"] = ctx_override
    result = api_post(MGMT_URL, "/v1/load", payload)
    model = result.get("model", {})
    profile = result.get("profile", {})
    if model:
        gpu_ids = profile.get("gpu_ids", [])
        hw = f"GPU G{','.join(str(g) for g in gpu_ids)}" if gpu_ids else "CPU"
        print(f"{C_GREEN}[OK]{C_RESET} {model.get('name','?')} ({model.get('model_type','?')}) → Slot {slot_id} | {hw} | CTX:{profile.get('ctx','?')}")
    else:
        print(f"{C_GREEN}[OK]{C_RESET} Geladen.")


def cmd_unload(slot_id):
    result = api_post(MGMT_URL, "/v1/unload", {"slot_id": slot_id})
    print(f"{C_GREEN}[OK]{C_RESET} Slot {slot_id} entladen.")


def cmd_slot(action=None, value=None):
    if action == "set":
        try:
            n = int(value)
            assert n >= 1
        except Exception:
            print(f"{C_RED}Ungültige Slot-Nummer: '{value}'{C_RESET}", file=sys.stderr)
            sys.exit(1)
        _set_default_slot(n)
        print(f"Standard-Slot gesetzt: {C_BOLD}Slot {n}{C_RESET}")
    else:
        cur = _get_default_slot()
        print(f"Standard-Slot: {C_BOLD}{cur}{C_RESET}  (ändern: sj slot set N)")


def cmd_ctx(slot_id, value=None):
    slots = api_get(MGMT_URL, "/v1/slots").get("slots", [])
    slot = next((s for s in slots if s.get("id") == slot_id), None)
    if not slot:
        print(f"{C_RED}Slot {slot_id} nicht gefunden.{C_RESET}", file=sys.stderr); sys.exit(1)
    loaded = slot.get("loaded_model")
    if not loaded:
        print(f"{C_RED}Slot {slot_id} hat kein Modell.{C_RESET}", file=sys.stderr); sys.exit(1)

    current_ctx = loaded.get("ctx", "?")
    if value is None:
        print(f"  Slot {slot_id} | {loaded['name']} | CTX = {current_ctx}")
        return

    try:
        new_ctx = _parse_ctx(value)
    except ValueError:
        print(f"{C_RED}Ungültige CTX-Größe: '{value}' – Beispiel: 8K, 16384{C_RESET}", file=sys.stderr)
        sys.exit(1)

    model_id = loaded["id"]
    print(f"  CTX: {current_ctx} → {new_ctx} | Entlade Slot {slot_id}...")
    api_post(MGMT_URL, "/v1/unload", {"slot_id": slot_id})
    print(f"  Lade {model_id} mit CTX={new_ctx}...")
    result = api_post(MGMT_URL, "/v1/load", {"slot_id": slot_id, "model_id": model_id, "ctx": new_ctx})
    profile = result.get("profile", {})
    gpu_ids = profile.get("gpu_ids", [])
    hw = f"GPU G{','.join(str(g) for g in gpu_ids)}" if gpu_ids else "CPU"
    layers = profile.get("gpu_layers")
    hw_str = f"{hw} ({layers}L)" if layers and layers != -1 else hw
    print(f"{C_GREEN}[OK]{C_RESET} {loaded['name']} → Slot {slot_id} | {hw_str} | CTX:{new_ctx}")


def cmd_status():
    s = api_get(MGMT_URL, "/status")
    print(f"\n  {C_BOLD}SkinnyJoe Status{C_RESET}")
    print(f"  {'─' * 55}")
    print(f"  Management : Port {s.get('management_port', '?')}")
    print(f"  Modelle    : {s.get('models_count', '?')}")

    gpus = s.get("gpus", [])
    print(f"  GPUs       : {len(gpus)} erkannt" if gpus else f"  GPUs       : {C_DIM}keine{C_RESET}")

    hw = s.get("hardware_alloc", {})
    if hw:
        print(f"  GPU-Belegung: {', '.join(f'{g}→{s}' for g, s in hw.items())}")

    slots = s.get("slots", [])
    print(f"\n  {C_BOLD}Slots:{C_RESET}")
    for sl in slots:
        loaded = sl.get("loaded_model")
        if loaded:
            st = f"{C_GREEN}■{C_RESET} {loaded['name']} ({loaded['model_type']})"
            if sl["status"] == "generating":
                st += f" {C_YELLOW}[BUSY]{C_RESET}"
        else:
            st = f"{C_DIM}■ leer{C_RESET}"
        print(f"    Slot {sl['id']} (Port {sl['port']}): {st}")

    print(f"  {'─' * 55}\n")


def cmd_ask(prompts, slot_id, max_tokens=1024, show_thought=False, image_path=None):
    # Prüfe Slot-Status über Management
    slots_data = api_get(MGMT_URL, "/v1/slots")
    slot_info = None
    for s in slots_data.get("slots", []):
        if s["id"] == slot_id:
            slot_info = s
            break
    if not slot_info:
        print(f"{C_RED}Slot {slot_id} nicht gefunden.{C_RESET}", file=sys.stderr)
        sys.exit(1)
    if not slot_info.get("loaded_model"):
        print(f"{C_RED}Slot {slot_id} hat kein Modell. Nutze 'sj load <N> --slot {slot_id}'{C_RESET}", file=sys.stderr)
        sys.exit(1)

    model_type = slot_info["loaded_model"].get("model_type", "text2text")
    slot_base = _slot_url(slot_id)

    # Prompt zusammenbauen
    final_parts = []
    for p in prompts:
        if os.path.isfile(p) and not p.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            with open(p, "r", encoding="utf-8") as f:
                final_parts.append(f.read().strip())
        else:
            final_parts.append(p)
    prompt = "\n\n".join(final_parts)

    if model_type == "text2image":
        _ask_image(prompt, slot_base)
    elif model_type == "image2text" and image_path:
        _ask_vision(prompt, image_path, slot_base, max_tokens, show_thought)
    else:
        _ask_text(prompt, slot_base, max_tokens, show_thought)


def _stream_response(payload, base_url, show_thought):
    SEP_KEYWORD = "FINAL_ANSWER_STARTS_HERE"
    THINK_END = "</think>"
    THINK_START = "<think>"
    try:
        t_start = time.time()
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=600)
        resp.raise_for_status()
        full_text, printed_len, token_count, t_first = "", 0, 0, None
        print("=" * 40)
        for line in resp.iter_lines():
            if not line: continue
            ls = line.decode("utf-8")
            if not ls.startswith("data: "): continue
            raw = ls[6:]
            if raw == "[DONE]": break
            try:
                chunk = json.loads(raw)
                if "error" in chunk:
                    sys.stdout.write("\r\033[K")
                    print(f"\n{C_RED}Server: {chunk['error'].get('message','')}{C_RESET}", file=sys.stderr)
                    return
                if "choices" in chunk and chunk["choices"]:
                    c = chunk["choices"][0].get("delta", {}).get("content", "")
                    if not c: continue
                    if t_first is None: t_first = time.time()
                    token_count += 1; full_text += c
                    el = time.time() - t_start
                    if show_thought:
                        new = full_text[printed_len:].replace(THINK_START,"").replace(THINK_END,"").replace(SEP_KEYWORD,"")
                        if new: sys.stdout.write(f"\033[90m{new}\033[0m"); sys.stdout.flush()
                        printed_len = len(full_text)
                    else:
                        sn = full_text.replace(THINK_START,"").replace(THINK_END,"").replace(SEP_KEYWORD,"").replace("\n"," ").strip()
                        if sn:
                            d = sn[-40:] if len(sn) > 40 else sn
                            sys.stdout.write(f"\r\033[K[{el:6.1f}s|{token_count}tok] ...{d}"); sys.stdout.flush()
            except json.JSONDecodeError: continue
        t_done = time.time()
        sys.stdout.write("\r\033[K"); sys.stdout.flush()
        # Antwort extrahieren
        seps = [SEP_KEYWORD, "FINAL_ANSWER_STARTS", THINK_END]
        li, fl = -1, 0
        for sep in seps:
            i = full_text.rfind(sep)
            if i > li: li, fl = i, len(sep)
        if li != -1:
            st = li + fl
            while st < len(full_text) and full_text[st] in "\n\r :->`": st += 1
            ans = full_text[st:].strip()
            if ans: print(f"{C_BOLD}{ans}{C_RESET}")
            elif not show_thought: print(full_text.strip()[-200:])
        elif not show_thought:
            print(full_text.strip())
        # Timing
        tpe = (t_first - t_start) if t_first else (t_done - t_start)
        tg = (t_done - t_first) if t_first else 0
        ts = token_count / tg if tg > 0 else 0
        print(f"  Prompt-Eval: {tpe:.2f}s | Gen: {tg:.2f}s ({token_count} tok, {ts:.1f} tok/s) | Total: {t_done-t_start:.2f}s")
        print("=" * 40)
    except Exception as e:
        print(f"\n{C_RED}Fehler: {e}{C_RESET}", file=sys.stderr)


def _ask_text(prompt, base_url, max_tokens, show_thought):
    SYS = ("(Rule 1) Think internally, step by step. When finished, write: FINAL_ANSWER_STARTS_HERE "
           "Then the answer. Everything before is private thought, everything after is visible.")
    _stream_response({"messages": [{"role":"system","content":SYS},{"role":"user","content":prompt}],
                       "max_tokens": max_tokens, "stream": True}, base_url, show_thought)


def _ask_vision(prompt, image_path, base_url, max_tokens, show_thought):
    ip = Path(image_path)
    if not ip.exists():
        print(f"{C_RED}Bild nicht gefunden: {image_path}{C_RESET}", file=sys.stderr); sys.exit(1)
    with open(ip, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    ext = ip.suffix.lower().lstrip('.')
    mime = {'jpg':'jpeg','jpeg':'jpeg','png':'png','gif':'gif','webp':'webp'}.get(ext, 'jpeg')
    print(f"{C_DIM}Bild: {ip.name} ({ip.stat().st_size/1024:.0f} KB){C_RESET}")
    content = [{"type":"image_url","image_url":{"url":f"data:image/{mime};base64,{b64}"}},
               {"type":"text","text":prompt}]
    _stream_response({"messages": [{"role":"user","content":content}],
                       "max_tokens": max_tokens, "stream": True}, base_url, show_thought)


def _ask_image(prompt, base_url):
    print(f"Generiere Bild: \"{prompt[:60]}\"...")
    try:
        result = api_post(base_url, "/v1/images/generations", {"prompt": prompt}, timeout=1800)
        dl = result.get("data", [])
        if dl and dl[0].get("b64_json"):
            out = os.path.join(os.getcwd(), "response.jpg")
            with open(out, "wb") as f: f.write(base64.b64decode(dl[0]["b64_json"]))
            print(f"{C_GREEN}[OK]{C_RESET} Bild: {out}")
        else:
            print(f"{C_RED}Keine Bilddaten.{C_RESET}", file=sys.stderr)
    except Exception as e:
        print(f"{C_RED}Fehler: {e}{C_RESET}", file=sys.stderr)


def cmd_transcribe(audio_path, slot_id, language=None):
    """Transkribiert eine Audiodatei via Whisper auf einem Slot."""
    ap = Path(audio_path)
    if not ap.exists():
        print(f"{C_RED}Datei nicht gefunden: {audio_path}{C_RESET}", file=sys.stderr)
        sys.exit(1)

    # Prüfe Slot
    slots_data = api_get(MGMT_URL, "/v1/slots")
    slot_info = None
    for s in slots_data.get("slots", []):
        if s["id"] == slot_id:
            slot_info = s
            break
    if not slot_info:
        print(f"{C_RED}Slot {slot_id} nicht gefunden.{C_RESET}", file=sys.stderr)
        sys.exit(1)
    if not slot_info.get("loaded_model"):
        print(f"{C_RED}Slot {slot_id} hat kein Modell.{C_RESET}", file=sys.stderr)
        sys.exit(1)
    if slot_info["loaded_model"].get("model_type") != "speech2text":
        print(f"{C_RED}Slot {slot_id} hat kein Whisper-Modell geladen ({slot_info['loaded_model'].get('model_type','?')}).{C_RESET}",
              file=sys.stderr)
        sys.exit(1)

    slot_base = _slot_url(slot_id)
    print(f"{C_DIM}Audio: {ap.name} ({ap.stat().st_size/1024:.0f} KB){C_RESET}")
    print(f"Transkribiere...")

    try:
        with open(ap, "rb") as f:
            t_start = time.time()
            r = requests.post(
                f"{slot_base}/v1/audio/transcriptions",
                files={"file": (ap.name, f)},
                data={"language": language} if language else {},
                timeout=600
            )
            r.raise_for_status()
            result = r.json()
            t_done = time.time()

        text = result.get("text", "")
        print(f"\n{C_BOLD}{text}{C_RESET}")
        segments = result.get("segments", [])
        if segments:
            print(f"\n{C_DIM}Segmente:{C_RESET}")
            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                txt = seg.get("text", "")
                print(f"  {C_CYAN}[{start:>7.2f}s – {end:>7.2f}s]{C_RESET} {txt}")
        lang = result.get("language", "")
        if lang:
            print(f"\n{C_DIM}Sprache: {lang}  |  Dauer: {t_done - t_start:.2f}s{C_RESET}")
    except Exception as e:
        print(f"{C_RED}Fehler: {e}{C_RESET}", file=sys.stderr)


# ============================================================
# sj import – Model-Import-Assistent
# ============================================================

_GGUF_QUANT = {
    0:"F32", 1:"F16", 2:"Q4_0", 3:"Q4_1", 5:"Q5_0", 6:"Q5_1",
    7:"Q8_0", 8:"Q8_1", 9:"Q2_K", 10:"Q3_K_S", 11:"Q3_K_M", 12:"Q3_K_L",
    13:"Q4_K_S", 14:"Q4_K_M", 15:"Q5_K_S", 16:"Q5_K_M", 17:"Q6_K",
    18:"Q8_K", 26:"BF16", 30:"IQ4_NL", 31:"IQ3_S", 32:"IQ3_M",
    34:"IQ2_M", 36:"IQ4_XS", 37:"IQ1_M",
}

def _read_gguf_meta(path: Path) -> dict:
    """Liest GGUF-Header-Metadaten effizient (ohne das Modell zu laden)."""
    meta = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count    = struct.unpack("<Q", f.read(8))[0]
            meta["_version"] = version
            meta["_tensors"] = tensor_count

            def rs():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", errors="replace")

            def rv(t):
                if t == 0:  return struct.unpack("<B", f.read(1))[0]
                if t == 1:  return struct.unpack("<b", f.read(1))[0]
                if t == 2:  return struct.unpack("<H", f.read(2))[0]
                if t == 3:  return struct.unpack("<h", f.read(2))[0]
                if t == 4:  return struct.unpack("<I", f.read(4))[0]
                if t == 5:  return struct.unpack("<i", f.read(4))[0]
                if t == 6:  return struct.unpack("<f", f.read(4))[0]
                if t == 7:  return bool(struct.unpack("<B", f.read(1))[0])
                if t == 8:  return rs()
                if t == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    al = struct.unpack("<Q", f.read(8))[0]
                    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                    if et in sizes:
                        f.seek(al * sizes[et], 1)
                        return f"[{al}]"
                    if et == 8:  # string array → too variable, stop
                        raise StopIteration
                    return f"[{al}]"
                if t == 10: return struct.unpack("<Q", f.read(8))[0]
                if t == 11: return struct.unpack("<q", f.read(8))[0]
                if t == 12: return struct.unpack("<d", f.read(8))[0]
                raise ValueError(f"Unbekannter GGUF-Typ {t}")

            try:
                for _ in range(min(kv_count, 400)):
                    key = rs()
                    vtype = struct.unpack("<I", f.read(4))[0]
                    val = rv(vtype)
                    keep = (key.startswith("general.") or
                            key.endswith(".context_length") or
                            key.endswith(".block_count") or
                            key.endswith(".embedding_length") or
                            key.endswith(".head_count"))
                    if keep:
                        meta[key] = val
                    if key.startswith("tokenizer."):
                        break  # Vocab-Daten beginnen – stopp
            except StopIteration:
                pass
    except Exception as e:
        meta["_error"] = str(e)
    return meta


_VISION_ARCHS  = {"llava", "moondream", "qwen2_vl", "yi_vl", "idefics",
                  "bakllava", "minicpmv", "internvl", "cogvlm"}
_SPEECH_ARCHS  = {"whisper"}
_IMAGE_ARCHS   = {"flux", "stable_diffusion", "sdxl"}

def _analyze_model(src: Path) -> dict:
    """Vollständige Analyse eines Modell-Pfades."""
    a = {
        "path": src,
        "name": src.name,
        "size_gb": 0.0,
        "format": src.suffix.lower().lstrip(".") if src.is_file() else "dir",
        "model_type": "text2text",
        "confidence": "likely",
        "arch": None,
        "quant": None,
        "params_b": None,
        "ctx": None,
        "block_count": None,
        "notes": [],
        "gguf_meta": {},
    }

    if src.is_file():
        a["size_gb"] = src.stat().st_size / (1024**3)
    elif src.is_dir():
        a["size_gb"] = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / (1024**3)

    nl = src.name.lower()

    # ── GGUF Header ──────────────────────────────────────────
    if src.suffix.lower() == ".gguf":
        meta = _read_gguf_meta(src)
        a["gguf_meta"] = meta
        arch = meta.get("general.architecture", "")
        a["arch"] = arch or None
        ft = meta.get("general.file_type", -1)
        a["quant"] = _GGUF_QUANT.get(ft)
        pc = meta.get("general.parameter_count", 0)
        if pc > 0:
            a["params_b"] = pc / 1e9
        if arch:
            a["ctx"]         = meta.get(f"{arch}.context_length")
            a["block_count"] = meta.get(f"{arch}.block_count")
        if arch in _VISION_ARCHS:
            a["model_type"] = "image2text"; a["confidence"] = "sure"
        elif arch in _SPEECH_ARCHS:
            a["model_type"] = "speech2text"; a["confidence"] = "sure"
        elif arch:
            a["model_type"] = "text2text"; a["confidence"] = "sure"

    # ── Dateiname-Heuristiken (können GGUF-Ergebnis überschreiben) ──
    if "whisper" in nl:
        a["model_type"] = "speech2text"; a["confidence"] = "sure"
    elif any(x in nl for x in ("flux", "stable-diffusion", "sdxl", "sd-v", "sd_v")):
        a["model_type"] = "text2image"; a["confidence"] = "sure"
    elif any(x in nl for x in ("kokoro", "bark", "xtts", "speecht5", "-tts")):
        a["model_type"] = "text2speech"; a["confidence"] = "sure"
    elif any(x in nl for x in ("llava", "moondream", "qwen-vl", "qwen2-vl", "yi-vl",
                                "llava-llm", "vision-encoder")):
        a["model_type"] = "image2text"; a["confidence"] = "sure"

    # mmproj-Hinweis
    if "mmproj" in nl:
        a["model_type"] = "image2text"; a["confidence"] = "sure"
        a["notes"].append("mmproj (Vision-Encoder – gehört zu einem LLaVA/Vision-Modell)")

    # Parameter aus Dateiname schätzen (falls unbekannt)
    if a["params_b"] is None:
        for tok in nl.replace("-", " ").replace("_", " ").split():
            if tok.endswith("b") and tok[:-1].replace(".", "", 1).isdigit():
                try:
                    a["params_b"] = float(tok[:-1]); break
                except ValueError:
                    pass

    # Quantisierung aus Dateiname schätzen (falls GGUF-Header kein Ergebnis)
    if a["quant"] is None:
        for q in ("Q2_K","Q3_K_S","Q3_K_M","Q3_K_L","Q4_0","Q4_K_S","Q4_K_M",
                  "Q5_0","Q5_K_S","Q5_K_M","Q6_K","Q8_0","F16","BF16","F32"):
            if q.lower() in nl:
                a["quant"] = q; break

    # Companion-Dateien
    if a["model_type"] == "image2text" and src.is_file() and "mmproj" not in nl:
        mmprojs = list(src.parent.glob("*mmproj*"))
        if mmprojs:
            a["notes"].append(f"mmproj im selben Ordner: {mmprojs[0].name}")
        else:
            a["notes"].append("Kein mmproj gefunden – Vision evtl. nicht nutzbar ohne Encoder")

    # .safetensors / Verzeichnis → Hinweise
    if a["format"] in ("safetensors", "bin", "pt") and a["confidence"] == "likely":
        a["notes"].append("Format ohne eindeutigen Header – Typ aus Dateiname geschätzt")
    if src.is_dir():
        if (src / "model_index.json").exists():
            a["model_type"] = "text2image"; a["confidence"] = "sure"
            a["notes"].append("model_index.json gefunden → Diffusion-Pipeline")
        elif (src / "config.json").exists():
            a["notes"].append("config.json gefunden → HuggingFace-Transformers-Modell")

    return a


def _gpu_compat(size_gb: float, gpus: list) -> list:
    """Gibt eine Liste von Kompatibilitäts-Zeilen zurück."""
    lines = []
    if not gpus:
        lines.append(f"  {C_YELLOW}Keine NVIDIA-GPUs erkannt → CPU-only{C_RESET}")
        return lines

    for g in gpus:
        vram = g.get("vram_total_gb", 0)
        name = g.get("name", f"G{g['id']}")
        gid  = g["id"]
        if size_gb <= vram * 0.95:
            lines.append(f"  G{gid} {C_GREEN}✓{C_RESET}  {name} ({vram:.0f} GB) – passt vollständig")
        else:
            pct = int(size_gb / vram * 100)
            lines.append(f"  G{gid} {C_YELLOW}~{C_RESET}  {name} ({vram:.0f} GB) – {pct}% nötig → gpu_layers nötig")

    # Multi-GPU check
    if len(gpus) >= 2:
        total = sum(g.get("vram_total_gb", 0) for g in gpus)
        if size_gb <= total * 0.95:
            ids = "+".join(f"G{g['id']}" for g in gpus)
            lines.append(f"  {ids} {C_GREEN}✓{C_RESET}  Dual-GPU ({total:.0f} GB gesamt) – passt vollständig")
    return lines


def _suggest_profile(a: dict, gpus: list) -> dict:
    """Erstellt einen Vorschlag für ein config.json-Profil."""
    size = a["size_gb"]
    profile = {"ctx": a["ctx"] or 4096, "n_batch": 512, "n_threads": 8}

    if not gpus:
        profile["gpu_layers"] = 0
        profile["gpu_ids"] = []
        profile["_info"] = f"{size:.1f}GB – CPU-only (keine GPU erkannt)"
        return profile

    single_vram = gpus[0].get("vram_total_gb", 0) if gpus else 0
    total_vram  = sum(g.get("vram_total_gb", 0) for g in gpus)
    gpu_ids_all = [g["id"] for g in gpus]

    if a["model_type"] == "speech2text":
        profile["gpu_ids"] = [gpus[0]["id"]] if gpus else []
        profile["_info"] = f"{size:.1f}GB Whisper – GPU {gpus[0]['id']}"
        return profile

    if size <= single_vram * 0.95:
        profile["gpu_layers"] = -1
        profile["gpu_ids"] = [gpus[0]["id"]]
        profile["n_batch"] = 2048
        profile["_info"] = f"{size:.1f}GB – passt auf G{gpus[0]['id']} ({single_vram:.0f}GB)"
    elif len(gpus) >= 2 and size <= total_vram * 0.95:
        # Schätze gpu_layers: wie viele Layer passen auf die GPUs?
        blocks = a["block_count"] or 32
        frac = min(total_vram * 0.9 / size, 1.0)
        layers = max(1, int(blocks * frac))
        profile["gpu_layers"] = layers
        profile["gpu_ids"] = gpu_ids_all
        profile["n_batch"] = 1024
        profile["_info"] = (f"{size:.1f}GB – Dual-GPU "
                            f"G{'+G'.join(str(g) for g in gpu_ids_all)} + CPU-Rest")
    else:
        # Partial GPU offload
        if gpus:
            blocks = a["block_count"] or 32
            frac = min(single_vram * 0.85 / size, 1.0)
            layers = max(1, int(blocks * frac))
            profile["gpu_layers"] = layers
            profile["gpu_ids"] = [gpus[0]["id"]]
            profile["_info"] = (f"{size:.1f}GB – zu groß für {single_vram:.0f}GB VRAM "
                                f"→ {layers}/{blocks} Layer auf G{gpus[0]['id']}")
        else:
            profile["gpu_layers"] = 0
            profile["gpu_ids"] = []
            profile["_info"] = f"{size:.1f}GB – CPU-only"

    return profile


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "[J/n]" if default else "[j/N]"
    try:
        ans = input(f"  {prompt} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not ans:
        return default
    return ans in ("j", "ja", "y", "yes")


def _ask_choice(prompt: str, options: list, default: int = 0) -> int:
    for i, o in enumerate(options):
        marker = f"{C_GREEN}▸{C_RESET}" if i == default else " "
        print(f"  {marker} [{i+1}] {o}")
    try:
        raw = input(f"  {prompt} [1-{len(options)}, Enter={default+1}]: ").strip()
        idx = int(raw) - 1 if raw else default
        return max(0, min(len(options) - 1, idx))
    except (ValueError, EOFError, KeyboardInterrupt):
        return default


def cmd_import(source: str, copy_mode: bool = False, skip_test: bool = False):
    """Importiert ein Modell in die SkinnyJoe-Bibliothek."""
    src = Path(source).resolve()
    if not src.exists():
        print(f"{C_RED}Nicht gefunden: {src}{C_RESET}", file=sys.stderr)
        sys.exit(1)

    # ── Konfiguration laden ───────────────────────────────────
    config_path = Path(BASE_DIR) / "config.json"
    try:
        cfg = json.loads(config_path.read_text())
    except Exception as e:
        print(f"{C_RED}config.json nicht lesbar: {e}{C_RESET}", file=sys.stderr)
        sys.exit(1)
    lib_root = Path(cfg.get("models_dir", "/home/models/skinnyJoe"))

    # ── Analyse ───────────────────────────────────────────────
    print(f"\n  {C_BOLD}SkinnyJoe Import-Assistent{C_RESET}")
    print(f"  {'─' * 62}")
    print(f"  Analysiere: {C_CYAN}{src.name}{C_RESET}")

    a = _analyze_model(src)

    # GPU-Info holen (Daemon oder nvidia-smi)
    gpus = []
    try:
        gpus = api_get(MGMT_URL, "/v1/gpus").get("gpus", [])
    except SystemExit:
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,name,memory.total",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL)
                for line in out.strip().splitlines():
                    p = [x.strip() for x in line.split(",")]
                    if len(p) >= 3:
                        gpus.append({"id": int(p[0]), "name": p[1],
                                     "vram_total_gb": int(p[2]) / 1024})
            except Exception:
                pass

    # ── Analyse anzeigen ──────────────────────────────────────
    conf_color = C_GREEN if a["confidence"] == "sure" else C_YELLOW
    type_label = TYPE_LABELS.get(a["model_type"], a["model_type"])
    type_color = TYPE_COLORS.get(a["model_type"], C_WHITE)

    print(f"\n  {C_BOLD}Ergebnis:{C_RESET}")
    print(f"    Format      : {a['format'].upper()}"
          + (f"  (Arch: {C_CYAN}{a['arch']}{C_RESET})" if a['arch'] else ""))
    print(f"    Modell-Typ  : {type_color}{type_label}{C_RESET}"
          f"  {conf_color}({'sicher' if a['confidence']=='sure' else 'wahrscheinlich'}){C_RESET}")
    if a["params_b"]:
        print(f"    Parameter   : {a['params_b']:.1f}B")
    if a["quant"]:
        print(f"    Quantisierung: {a['quant']}")
    if a["ctx"]:
        print(f"    Max. Kontext: {a['ctx']:,} Token")
    print(f"    Dateigröße  : {a['size_gb']:.2f} GB")
    print(f"    VRAM-Bedarf : ~{a['size_gb']*1.08:.1f} GB (inkl. Overhead)")

    if a["notes"]:
        print(f"\n  {C_BOLD}Hinweise:{C_RESET}")
        for n in a["notes"]:
            print(f"    {C_YELLOW}»{C_RESET} {n}")

    if gpus:
        print(f"\n  {C_BOLD}GPU-Kompatibilität:{C_RESET}")
        for line in _gpu_compat(a["size_gb"], gpus):
            print(f"  {line}")
    else:
        print(f"\n  {C_DIM}Keine GPUs erkannt (Daemon läuft evtl. nicht){C_RESET}")

    # ── Wizard ────────────────────────────────────────────────
    print(f"\n  {'─' * 62}")

    # Typ bestätigen / ändern
    if a["confidence"] != "sure":
        print(f"\n  {C_YELLOW}Modell-Typ unsicher – bitte wählen:{C_RESET}")
        type_opts = list(TYPE_ORDER)
        cur = type_opts.index(a["model_type"]) if a["model_type"] in type_opts else 0
        chosen = _ask_choice("Typ", [TYPE_LABELS.get(t, t) for t in type_opts], default=cur)
        a["model_type"] = type_opts[chosen]
    else:
        print(f"\n  Erkannter Typ: {type_color}{type_label}{C_RESET}")
        if not _ask_yn("Korrekt?", default=True):
            type_opts = list(TYPE_ORDER)
            cur = type_opts.index(a["model_type"]) if a["model_type"] in type_opts else 0
            chosen = _ask_choice("Typ wählen",
                                 [TYPE_LABELS.get(t, t) for t in type_opts], default=cur)
            a["model_type"] = type_opts[chosen]

    # Zielverzeichnis
    dest_dir = lib_root / a["model_type"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    print(f"\n  Ziel: {C_CYAN}{dest}{C_RESET}")
    if dest.exists():
        print(f"  {C_YELLOW}Datei existiert bereits!{C_RESET}")
        if not _ask_yn("Überschreiben?", default=False):
            print("  Abgebrochen.")
            return

    # Profil vorschlagen
    profile = _suggest_profile(a, gpus)
    profile_name = src.stem  # ohne Extension

    print(f"\n  {C_BOLD}Vorgeschlagenes Profil ({profile_name}):{C_RESET}")
    for k, v in profile.items():
        print(f"    {k:<18}: {v}")

    add_profile = _ask_yn("Profil in config.json eintragen?", default=True)

    # Aktion ausführen
    action_word = "Kopiere" if copy_mode else "Verschiebe"
    print(f"\n  {action_word}: {C_DIM}{src.name}{C_RESET} → {C_CYAN}{dest_dir.name}/{C_RESET}")

    try:
        if copy_mode:
            if src.is_dir():
                shutil.copytree(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))
        else:
            shutil.move(str(src), str(dest))
    except Exception as e:
        print(f"  {C_RED}Fehler beim {'Kopieren' if copy_mode else 'Verschieben'}: {e}{C_RESET}",
              file=sys.stderr)
        sys.exit(1)

    print(f"  {C_GREEN}[OK]{C_RESET} {dest}")

    # Profil eintragen
    if add_profile:
        try:
            cfg.setdefault("model_profiles", {})[profile_name] = profile
            config_path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False))
            print(f"  {C_GREEN}[OK]{C_RESET} Profil '{profile_name}' in config.json eingetragen.")
        except Exception as e:
            print(f"  {C_YELLOW}Profil konnte nicht gespeichert werden: {e}{C_RESET}")

    # Quick-Test (nur text2text, wenn Daemon läuft und nicht übersprungen)
    if not skip_test and a["model_type"] == "text2text":
        print(f"\n  {C_DIM}Möchtest du das Modell direkt testen?{C_RESET}")
        print(f"  {C_DIM}(erfordert freien Slot; das Modell wird geladen und kurz befragt){C_RESET}")
        if _ask_yn("Schnelltest durchführen?", default=False):
            _quick_test(dest, profile_name, profile)

    print(f"\n  {'─' * 62}")
    print(f"  Fertig. Neu scannen mit: {C_CYAN}sj server restart{C_RESET}")
    print(f"  Dann laden mit:         {C_CYAN}sj load <N> --slot 1{C_RESET}\n")


def _quick_test(model_path: Path, profile_name: str, profile: dict):
    """Lädt das Modell in einen freien Slot und macht einen Kurztest."""
    try:
        slots_data = api_get(MGMT_URL, "/v1/slots")
        free_slot = next(
            (s["id"] for s in slots_data.get("slots", [])
             if not s.get("loaded_model")), None)
    except SystemExit:
        print(f"  {C_YELLOW}Daemon nicht erreichbar – kein Test möglich.{C_RESET}")
        return

    if free_slot is None:
        print(f"  {C_YELLOW}Kein freier Slot verfügbar – Test übersprungen.{C_RESET}")
        return

    # Modell-ID suchen
    models = api_get(MGMT_URL, "/v1/models").get("data", [])
    mid = next((m["id"] for m in models
                if m.get("name") == profile_name or
                   m.get("full_name", "").startswith(profile_name)), None)
    if mid is None:
        print(f"  {C_YELLOW}Modell noch nicht in Bibliothek sichtbar – "
              f"erst 'sj server restart' ausführen.{C_RESET}")
        return

    print(f"  Lade {mid} → Slot {free_slot}...")
    try:
        api_post(MGMT_URL, "/v1/load", {"slot_id": free_slot, "model_id": mid})
    except SystemExit:
        print(f"  {C_RED}Laden fehlgeschlagen.{C_RESET}")
        return

    print(f"  Test-Prompt: 'Hallo, antworte auf Deutsch in einem Satz.'")
    from urllib.parse import urlparse
    slot_url = api_get(MGMT_URL, "/v1/slots")
    port = next((s["port"] for s in slot_url.get("slots", [])
                 if s["id"] == free_slot), 8001)
    parsed = urlparse(MGMT_URL)
    slot_base = f"{parsed.scheme}://{parsed.hostname}:{port}"

    _stream_response({
        "messages": [{"role": "user", "content": "Hallo, antworte auf Deutsch in einem Satz."}],
        "max_tokens": 128, "stream": True
    }, slot_base, show_thought=False)

    print(f"  Entlade Slot {free_slot}...")
    try:
        api_post(MGMT_URL, "/v1/unload", {"slot_id": free_slot})
    except SystemExit:
        pass


def cmd_resources():
    """Zeigt RAM, VRAM, CPU/GPU-Auslastung und welche Modelle wo liegen."""
    import subprocess, shutil

    print(f"\n  {C_BOLD}SkinnyJoe Ressourcen{C_RESET}")
    print(f"  {'─' * 60}")

    # --- RAM ---
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":")
                mem[k.strip()] = int(v.strip().split()[0])  # kB
        total = mem["MemTotal"] / 1024 / 1024
        avail = mem["MemAvailable"] / 1024 / 1024
        used  = total - avail
        pct   = used / total * 100
        bar   = _bar(pct, 24)
        bc    = C_RED if pct > 85 else C_YELLOW if pct > 60 else C_GREEN
        print(f"\n  {C_BOLD}RAM{C_RESET}")
        print(f"    Gesamt: {total:.1f} GB   Belegt: {used:.1f} GB   Frei: {avail:.1f} GB")
        print(f"    {bc}{bar}{C_RESET}  {pct:.0f}%")
    except Exception as e:
        print(f"  RAM: {C_RED}Fehler: {e}{C_RESET}")

    # --- CPU ---
    try:
        load1, load5, _ = open("/proc/loadavg").read().split()[:3]
        cpu_count = os.cpu_count() or 1
        pct_load  = float(load1) / cpu_count * 100
        bc = C_RED if pct_load > 85 else C_YELLOW if pct_load > 50 else C_GREEN
        print(f"\n  {C_BOLD}CPU{C_RESET}  ({cpu_count} Kerne)")
        print(f"    Load: {load1} / {load5} (1m / 5m)   {bc}{pct_load:.0f}% Auslastung{C_RESET}")
    except Exception as e:
        print(f"  CPU: {C_RED}Fehler: {e}{C_RESET}")

    # --- VRAM (nvidia-smi) ---
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL
            )
            print(f"\n  {C_BOLD}NVIDIA GPUs (VRAM){C_RESET}")
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5: continue
                idx, name, used_mb, total_mb, util = parts
                used_gb  = int(used_mb) / 1024
                total_gb = int(total_mb) / 1024
                pct_vram = int(used_mb) / int(total_mb) * 100
                pct_util = int(util)
                bv = _bar(pct_vram, 20)
                bu = _bar(pct_util, 10)
                cv = C_RED if pct_vram > 85 else C_YELLOW if pct_vram > 60 else C_GREEN
                cu = C_RED if pct_util > 85 else C_YELLOW if pct_util > 50 else C_GREEN
                print(f"    G{idx}  {name}")
                print(f"       VRAM: {cv}{bv}{C_RESET} {used_gb:.1f}/{total_gb:.1f} GB ({pct_vram:.0f}%)")
                print(f"       Kern: {cu}{bu}{C_RESET} {pct_util}%")
        except Exception as e:
            print(f"  GPU: {C_RED}nvidia-smi Fehler: {e}{C_RESET}")
    else:
        print(f"\n  GPUs: {C_DIM}nvidia-smi nicht verfügbar{C_RESET}")

    # --- Geladene Modelle ---
    try:
        slots_data = api_get(MGMT_URL, "/v1/slots")
        loaded = [(s["id"], s["port"], s["loaded_model"])
                  for s in slots_data.get("slots", []) if s.get("loaded_model")]
        if loaded:
            print(f"\n  {C_BOLD}Geladene Modelle{C_RESET}")
            for sid, port, m in loaded:
                mtype  = m.get("model_type", "?")
                size   = m.get("size_gb", 0)
                tcolor = TYPE_COLORS.get(mtype, C_WHITE)
                print(f"    Slot {C_CYAN}{sid}{C_RESET} (:{port}): {tcolor}{m['name']}{C_RESET}")
                print(f"             {mtype}  ·  {size:.1f} GB")
        else:
            print(f"\n  {C_DIM}Keine Modelle geladen.{C_RESET}")
    except Exception:
        pass

    print(f"\n  {'─' * 60}\n")


def _bar(pct, width=20):
    filled = int(width * min(pct, 100) / 100)
    return "█" * filled + "░" * (width - filled)


BENCH_DATA_DIR = os.path.join(BASE_DIR, "bench_data")

_VISION_SIZES = [
    ("320×200",   320,  200),
    ("640×480",   640,  480),
    ("1024×768", 1024,  768),
    ("1080p",    1920, 1080),
    ("1440p",    2560, 1440),
    ("4K",       3840, 2160),
    ("8K",       7680, 4320),
]
_GEN_SIZES = [
    ("256×256",   256,  256),
    ("512×512",   512,  512),
    ("768×768",   768,  768),
    ("1024×1024",1024, 1024),
]
_AUDIO_DURATIONS = [("10s", 10), ("1min", 60), ("5min", 300)]
_TEXT_SIZES_K = [1, 2, 4, 8, 16]


def _bench_make_images():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    for label, w, h in _VISION_SIZES:
        path = os.path.join(BENCH_DATA_DIR, f"img_{w}x{h}.jpg")
        if os.path.exists(path):
            continue
        img = Image.new("RGB", (w, h), (100, 140, 180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w-1, 22], fill=(60, 90, 120))
        draw.text((6, 4), f"SkinnyJoe Bench  {w}x{h}", fill=(255, 255, 255))
        draw.rectangle([w//4, h//4, w*3//4, h*3//4], fill=(200, 100, 50))
        draw.text((w//4+6, h//4+6), "TEST PATTERN", fill=(255, 255, 200))
        img.save(path, quality=85)


def _bench_make_doc():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    path = os.path.join(BENCH_DATA_DIR, "doc_a4.png")
    if os.path.exists(path):
        return
    w, h = 1240, 1754   # DIN A4 @ 150 DPI
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((80, 60), "SkinnyJoe OCR Test – DIN A4", fill=(0, 0, 0))
    draw.line([(80, 95), (w-80, 95)], fill=(0, 0, 0), width=2)
    lines = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris",
        "nisi ut aliquip ex ea commodo consequat.",
        "",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse",
        "cillum dolore eu fugiat nulla pariatur.",
        "",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui",
        "officia deserunt mollit anim id est laborum.",
    ]
    for i, line in enumerate(lines):
        draw.text((80, 120 + i * 28), line, fill=(0, 0, 0))
    img.save(path)


def _bench_make_audio():
    import wave, array, math
    sr = 16000
    n_period = sr // 440
    period = array.array('h', [int(16383 * math.sin(2 * math.pi * i / n_period))
                                for i in range(n_period)])
    for label, dur in _AUDIO_DURATIONS:
        path = os.path.join(BENCH_DATA_DIR, f"audio_{label}.wav")
        if os.path.exists(path):
            continue
        n_total = sr * dur
        full, rem = divmod(n_total, n_period)
        samples = period * full
        samples.extend(period[:rem])
        with wave.open(path, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(samples.tobytes())


def _ensure_bench_data():
    os.makedirs(BENCH_DATA_DIR, exist_ok=True)
    _bench_make_images()
    _bench_make_doc()
    _bench_make_audio()


def _print_table(rows, headers):
    if not rows:
        return
    cw = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
          for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in cw) + "-+"
    print(f"\n{sep}")
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, cw)) + " |")
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, cw)) + " |")
    print(sep)


def _bench_text(slot_base, loaded, text_sizes_k, max_gen):
    BASE = ("The quick brown fox jumps over the lazy dog. "
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ")
    ctx = loaded.get("ctx") or 4096
    # Warmup: first request compiles Vulkan shaders – don't measure it
    print("  [warm] ", end="", flush=True)
    tw = time.time()
    try:
        requests.post(f"{slot_base}/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 1, "stream": False}, timeout=120)
        print(f"OK ({time.time()-tw:.1f}s)")
    except Exception:
        print("skip")

    rows = []
    for sk in text_sizes_k:
        tt = sk * 1000
        # Conservative: 3 chars/token (actual ~3.3-3.5 for this text)
        # Subtract overhead: template tokens + max_gen + safety margin
        max_prompt_tokens = min(tt, ctx - max_gen - 128)
        if max_prompt_tokens < 64:
            print(f"  [{sk:>3}K] CTX={ctx} zu klein – übersprungen")
            rows.append([f"{sk}K", "-", "-", "-", "-", "ctx<"])
            continue
        capped = max_prompt_tokens < tt  # true if size was reduced to fit CTX
        actual_k = f"{max_prompt_tokens//1000}.{(max_prompt_tokens%1000)//100}K"
        label = f"{sk}K" if not capped else f"{sk}K*"
        if capped:
            print(f"  [{sk:>3}K] HINWEIS: Kontextfenster CTX={ctx} – Prompt auf ~{actual_k} gedeckelt!")

        cn = max_prompt_tokens * 3
        pt = (BASE * ((cn // len(BASE)) + 1))[:cn]
        # Unique prefix per size to prevent KV-cache reuse between tests
        prompt = f"[bench-{sk}K] Summarize the following text:\n{pt}"
        print(f"  [{label:>10}] ", end="", flush=True)
        payload = {"messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_gen, "stream": True}
        ts = time.time()
        try:
            r = requests.post(f"{slot_base}/v1/chat/completions", json=payload,
                              stream=True, timeout=3600)
            r.raise_for_status()
            tf, tc = None, 0
            last_err = None
            for line in r.iter_lines():
                if not line: continue
                ls = line.decode("utf-8")
                if ls.startswith("data: "):
                    raw = ls[6:]
                    if raw == "[DONE]": break
                    try:
                        ch = json.loads(raw)
                        if ch.get("error"):
                            last_err = ch["error"].get("message", str(ch["error"]))
                            break
                        if ch.get("choices"):
                            c = ch["choices"][0].get("delta", {}).get("content", "")
                            if c:
                                if tf is None: tf = time.time()
                                tc += 1
                                sys.stdout.write(f"\r  [{label:>10}] {time.time()-ts:5.1f}s | {tc}tok...")
                                sys.stdout.flush()
                    except: continue
            td = time.time()
            sys.stdout.write(f"\r  [{label:>10}] ")
            if last_err:
                print(f"FEHLER: {last_err}")
                rows.append([label, "-", "-", "-", "-", "err"])
            elif tc == 0:
                print(f"FEHLER: 0 Tokens (CTX-Überlauf?)")
                rows.append([label, "-", "-", "-", "-", "0tok"])
            else:
                tp = (tf - ts) if tf else (td - ts)
                tg = (td - tf) if tf and tc > 0 else 0
                pp = max_prompt_tokens / tp if tp > 0 else 0  # actual tokens, not requested
                gs = tc / tg if tg > 0 else 0
                print(f"PP {tp:.2f}s ({pp:.0f}t/s) | Gen {tg:.2f}s ({gs:.1f}t/s) [{tc}tok]")
                rows.append([label, f"{tp:.2f}s", f"{pp:.0f}", f"{tg:.2f}s", f"{gs:.1f}", f"{tc}"])
        except Exception as e:
            print(f"FEHLER: {e}")
            rows.append([f"{sk}K", "-", "-", "-", "-", "err"])
    return rows, ["Größe", "PP", "PP t/s", "Gen", "Gen t/s", "Tok"]


def _bench_vision(slot_base, loaded):
    is_ocr = "ocr" in loaded.get("name", "").lower()
    if is_ocr:
        test_files = [("A4-Dok", os.path.join(BENCH_DATA_DIR, "doc_a4.png"))]
        prompt = "Transkribiere den Text in diesem Bild."
    else:
        test_files = [(label, os.path.join(BENCH_DATA_DIR, f"img_{w}x{h}.jpg"))
                      for label, w, h in _VISION_SIZES]
        prompt = "Beschreibe kurz was du siehst."
    rows = []
    for label, img_path in test_files:
        if not os.path.exists(img_path):
            print(f"  [{label:>10}] FEHLER: Testbild nicht gefunden")
            rows.append([label, "-", "-", "-", "-", "no-data"]); continue
        sz_kb = os.path.getsize(img_path) // 1024
        print(f"  [{label:>10}] {sz_kb}KB ", end="", flush=True)
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(img_path)[1].lower().lstrip('.')
        mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
        content = [{"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                   {"type": "text", "text": prompt}]
        payload = {"messages": [{"role": "user", "content": content}],
                   "max_tokens": 200, "stream": True}
        ts = time.time()
        try:
            r = requests.post(f"{slot_base}/v1/chat/completions", json=payload,
                              stream=True, timeout=3600)
            r.raise_for_status()
            tf, tc = None, 0
            for line in r.iter_lines():
                if not line: continue
                ls = line.decode("utf-8")
                if ls.startswith("data: "):
                    raw = ls[6:]
                    if raw == "[DONE]": break
                    try:
                        ch = json.loads(raw)
                        if ch.get("choices"):
                            c = ch["choices"][0].get("delta", {}).get("content", "")
                            if c:
                                if tf is None: tf = time.time()
                                tc += 1
                    except: continue
            td = time.time()
            tp = (tf - ts) if tf else (td - ts)
            tg = (td - tf) if tf and tc > 0 else 0
            gs = tc / tg if tg > 0 else 0
            print(f"TTFT {tp:.2f}s | Gen {tg:.2f}s ({gs:.1f}t/s) [{tc}tok]")
            rows.append([label, f"{sz_kb}KB", f"{tp:.2f}s", f"{tg:.2f}s", f"{gs:.1f}", f"{tc}"])
        except Exception as e:
            print(f"FEHLER: {e}")
            rows.append([label, f"{sz_kb}KB", "-", "-", "-", "err"])
    return rows, ["Auflösung", "Größe", "TTFT", "Gen", "Gen t/s", "Tok"]


def _bench_speech(slot_base):
    rows = []
    for label, dur_s in _AUDIO_DURATIONS:
        audio_path = os.path.join(BENCH_DATA_DIR, f"audio_{label}.wav")
        if not os.path.exists(audio_path):
            print(f"  [{label:>6}] FEHLER: Audiodatei nicht gefunden")
            rows.append([label, f"{dur_s}s", "-", "-", "no-data"]); continue
        print(f"  [{label:>6}] ", end="", flush=True)
        ts = time.time()
        try:
            with open(audio_path, "rb") as f:
                r = requests.post(f"{slot_base}/v1/audio/transcriptions",
                                  files={"file": ("audio.wav", f, "audio/wav")},
                                  data={"language": "en"}, timeout=3600)
            r.raise_for_status()
            td = time.time()
            elapsed = td - ts
            result = r.json()
            words = len(result.get("text", "").split())
            rtf = elapsed / dur_s
            print(f"{elapsed:.2f}s | RTF {rtf:.3f} | {words} Wörter")
            rows.append([label, f"{dur_s}s", f"{elapsed:.2f}s", f"{rtf:.3f}", f"{words}"])
        except Exception as e:
            print(f"FEHLER: {e}")
            rows.append([label, f"{dur_s}s", "-", "-", "err"])
    return rows, ["Dauer", "Audio", "Zeit", "RTF", "Wörter"]


def _bench_image_gen(slot_base):
    PROMPT = "a colorful mountain landscape with rivers, photorealistic, high quality"
    rows = []
    for label, w, h in _GEN_SIZES:
        print(f"  [{label:>10}] ", end="", flush=True)
        ts = time.time()
        try:
            r = requests.post(f"{slot_base}/v1/images/generations",
                              json={"prompt": PROMPT, "width": w, "height": h},
                              timeout=3600)
            r.raise_for_status()
            elapsed = time.time() - ts
            print(f"{elapsed:.2f}s")
            rows.append([label, f"{elapsed:.2f}s"])
        except Exception as e:
            print(f"FEHLER: {e}")
            rows.append([label, "err"])
    return rows, ["Größe", "Zeit"]


def cmd_bench(slot_id, sizes_str, max_gen):
    all_slots = api_get(MGMT_URL, "/v1/slots").get("slots", [])
    if slot_id is not None:
        slots_to_test = [s for s in all_slots if s.get("id") == slot_id]
        if not slots_to_test:
            print(f"{C_RED}Slot {slot_id} nicht gefunden.{C_RESET}", file=sys.stderr); sys.exit(1)
    else:
        slots_to_test = [s for s in all_slots if s.get("loaded_model")]
    if not slots_to_test:
        print(f"{C_RED}Keine Slots mit geladenem Modell.{C_RESET}", file=sys.stderr); sys.exit(1)

    size_map = {"1k": 1, "2k": 2, "4k": 4, "8k": 8, "16k": 16}
    if sizes_str:
        text_sizes = []
        for s in sizes_str:
            key = s.lower() if s.lower().endswith("k") else s.lower() + "k"
            if key in size_map:
                text_sizes.append(size_map[key])
            else:
                try: text_sizes.append(int(s.lower().rstrip("k")))
                except ValueError: pass
        if not text_sizes:
            text_sizes = _TEXT_SIZES_K
    else:
        text_sizes = _TEXT_SIZES_K

    print("  Prüfe Testdaten...")
    _ensure_bench_data()

    for sdata in slots_to_test:
        sid = sdata.get("id")
        loaded = sdata.get("loaded_model")
        if not loaded:
            print(f"\n  Slot {sid}: kein Modell geladen, übersprungen."); continue
        mtype = loaded.get("model_type", "text2text")
        name = loaded.get("name", "?")
        print(f"\n{'='*56}")
        print(f"  Benchmark | Slot {sid} | {name} ({mtype})")
        print(f"{'='*56}")
        slot_base = _slot_url(sid)
        if mtype == "text2text":
            rows, headers = _bench_text(slot_base, loaded, text_sizes, max_gen)
        elif mtype == "image2text":
            rows, headers = _bench_vision(slot_base, loaded)
        elif mtype == "speech2text":
            rows, headers = _bench_speech(slot_base)
        elif mtype == "text2image":
            rows, headers = _bench_image_gen(slot_base)
        else:
            print(f"  Kein Benchmark für Typ '{mtype}'."); continue
        _print_table(rows, headers)


def _find_daemon_pid_by_port(port=8000):
    """Sucht PID eines Prozesses der Port 8000 belegt."""
    try:
        out = subprocess.check_output(["ss", "-tlnp", f"sport = :{port}"],
                                      text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if f":{port}" in line and "pid=" in line:
                pid_part = [x for x in line.split(",") if x.startswith("pid=")]
                if pid_part:
                    return int(pid_part[0].split("=")[1])
    except Exception:
        pass
    return None


def server_start():
    """Startet den SkinnyJoe Daemon als Hintergrundprozess."""
    # PID-Datei prüfen
    if os.path.exists(PID_FILE):
        pid_str = open(PID_FILE).read().strip()
        try:
            os.kill(int(pid_str), 0)
            print(f"  SkinnyJoe läuft bereits (PID {pid_str})")
            return
        except (ProcessLookupError, ValueError):
            os.remove(PID_FILE)

    # Port prüfen
    existing = _find_daemon_pid_by_port(8000)
    if existing:
        with open(PID_FILE, "w") as f: f.write(str(existing))
        print(f"  SkinnyJoe läuft bereits (PID {existing}, Port 8000)")
        return

    py = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
    print(f"  Starte SkinnyJoe Daemon...")
    with open(LOG_FILE, "a") as log:
        proc = subprocess.Popen(
            [py, DAEMON_SCRIPT],
            cwd=BASE_DIR,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with open(PID_FILE, "w") as f: f.write(str(proc.pid))
    time.sleep(1.5)
    try:
        os.kill(proc.pid, 0)
        print(f"  {C_GREEN}[OK]{C_RESET} Daemon gestartet (PID {proc.pid}, Ports 8000-8004)")
        print(f"       Log: {LOG_FILE}")
    except ProcessLookupError:
        print(f"  {C_RED}[FEHLER]{C_RESET} Daemon konnte nicht gestartet werden.")
        print(f"          tail -20 {LOG_FILE}")


def server_stop():
    """Stoppt den SkinnyJoe Daemon."""
    pid = None
    if os.path.exists(PID_FILE):
        pid_str = open(PID_FILE).read().strip()
        try:
            os.kill(int(pid_str), 0)
            pid = int(pid_str)
        except (ProcessLookupError, ValueError):
            pass
        os.remove(PID_FILE)

    if pid is None:
        pid = _find_daemon_pid_by_port(8000)
        if pid:
            print(f"  Verwaisten Daemon gefunden (PID {pid})")

    if pid is None:
        print("  Kein Daemon läuft.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  {C_GREEN}[OK]{C_RESET} Daemon gestoppt (PID {pid})")
    except ProcessLookupError:
        print("  Prozess war bereits beendet.")
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def server_restart():
    server_stop()
    time.sleep(1)
    server_start()


def server_status():
    """Zeigt Daemon-Status inkl. Slot-Info."""
    pid = None
    running = False
    if os.path.exists(PID_FILE):
        pid_str = open(PID_FILE).read().strip()
        try:
            os.kill(int(pid_str), 0)
            pid, running = int(pid_str), True
        except (ProcessLookupError, ValueError):
            pass

    if not running:
        port_pid = _find_daemon_pid_by_port(8000)
        if port_pid:
            pid, running = port_pid, True

    print(f"\n  {C_BOLD}SkinnyJoe Daemon{C_RESET}")
    print(f"  {'─' * 50}")
    if running:
        print(f"  Status : {C_GREEN}LÄUFT{C_RESET} (PID {pid})")
    else:
        print(f"  Status : {C_RED}GESTOPPT{C_RESET}")
        print(f"  Starten: sj server start")
        print()
        return

    # Management-API abfragen
    try:
        import requests as _req
        r = _req.get(f"{MGMT_URL}/status", timeout=3)
        s = r.json()
        print(f"  API    : {C_GREEN}erreichbar{C_RESET} (Port {s.get('management_port','?')})")
        print(f"  Modelle: {s.get('models_count','?')}")
        print(f"\n  {C_BOLD}Slots:{C_RESET}")
        for sl in s.get("slots", []):
            lm = sl.get("loaded_model")
            if lm:
                col = C_YELLOW if sl["status"] == "generating" else C_GREEN
                print(f"    Slot {sl['id']} (:{sl['port']}): {col}■{C_RESET} {lm['name']}")
            else:
                print(f"    Slot {sl['id']} (:{sl['port']}): {C_DIM}■ leer{C_RESET}")
    except Exception:
        print(f"  API    : {C_YELLOW}nicht erreichbar (startet noch?){C_RESET}")

    print(f"  Log    : {LOG_FILE}")
    print(f"  {'─' * 50}\n")


def server_log():
    """Zeigt das Daemon-Log live."""
    if not os.path.exists(LOG_FILE):
        print(f"  Kein Log vorhanden: {LOG_FILE}")
        return
    print(f"  Log: {LOG_FILE}  (Ctrl+C zum Beenden)\n{'─'*60}")
    try:
        subprocess.run(["tail", "-f", "-n", "50", LOG_FILE])
    except KeyboardInterrupt:
        pass


def server_autostart(mode):
    """Delegiert Autostart-Verwaltung an install.sh."""
    if not os.path.exists(INSTALL_SCRIPT):
        print(f"  {C_RED}install.sh nicht gefunden: {INSTALL_SCRIPT}{C_RESET}")
        sys.exit(1)
    flag = {"on": "--autostart-on", "off": "--autostart-off", "status": "--autostart-status"}.get(mode)
    if not flag:
        print(f"  {C_RED}Unbekannter Modus: {mode}{C_RESET}")
        print("  Nutze: sj server autostart on|off|status")
        sys.exit(1)
    subprocess.run(["bash", INSTALL_SCRIPT, flag])


def cmd_server(action, autostart_mode="status"):
    actions = {
        "start": server_start,
        "stop": server_stop,
        "restart": server_restart,
        "status": server_status,
        "log": server_log,
    }
    if action == "autostart":
        server_autostart(autostart_mode)
    elif action in actions:
        actions[action]()
    else:
        print(f"  {C_RED}Unbekannte Aktion: {action}{C_RESET}")
        print("  sj server start|stop|restart|status|log|autostart on|off|status")


def cmd_kill():
    server_stop()


def cmd_tui():
    try:
        tui_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, tui_dir)
        from skinnyJoe_tui import main as tui_main
        tui_main()
    except ImportError as e:
        print(f"{C_RED}TUI nicht verfügbar: {e}{C_RESET}", file=sys.stderr)
        sys.exit(1)


def cmd_help():
    print(f"""
  {C_BOLD}SkinnyJoe CLI v4.0{C_RESET} – Multi-Slot KI-Server

  {C_BOLD}Architektur:{C_RESET}
    Management-API:  Port 8000 (Modelle listen, Slots verwalten)
    Slot-Ports:      8001-8004 (OpenAI-kompatible Inference)
    Jeder Slot = ein unabhaengiger Model-Endpoint

  {C_BOLD}Modelle & Slots:{C_RESET}
    sj models                       Alle Modelle + Profile anzeigen
    sj slots                        Slot-Status anzeigen
    sj load N{C_DIM}5{C_RESET} --slot 1               Modell N5 in Slot 1 laden
    sj load N{C_DIM}3{C_RESET} --slot 2               Modell N3 in Slot 2 laden
    sj unload --slot 1              Slot 1 entladen

  {C_BOLD}Anfragen (immer mit --slot):{C_RESET}
    sj ask --slot 1 "Was ist Python?"
    sj ask --slot 1 datei.txt
    sj ask --slot 1 --show-thought "..."
    sj ask --slot 2 --image foto.jpg "Beschreibe"

  {C_BOLD}Modell importieren:{C_RESET}
    sj import /pfad/modell.gguf        Analysieren, einordnen, verschieben
    sj import /pfad/modell.gguf --copy Kopieren statt verschieben
    sj import /pfad/modell.gguf --no-test  Kein Schnelltest

  {C_BOLD}Transkription (Whisper):{C_RESET}
    sj transcribe audio.wav --slot 1
    sj transcribe audio.mp3 --slot 1 --language de

  {C_BOLD}System:{C_RESET}
    sj gpus                         NVIDIA GPUs anzeigen
    sj status                       Gesamtstatus
    sj resources                    RAM/VRAM/CPU/GPU-Auslastung + geladene Modelle
    sj bench                        Alle geladenen Slots benchmarken
    sj bench --slot 1               Nur Slot 1 benchmarken
    sj bench --slot 1 --size 4K     Nur 4K-Test (text2text)
    sj tui                          {C_CYAN}Interaktive TUI{C_RESET}

  {C_BOLD}Server-Verwaltung:{C_RESET}
    sj server start                 Daemon starten
    sj server stop                  Daemon stoppen
    sj server restart               Daemon neu starten
    sj server status                Daemon-Status + Slots
    sj server log                   Live-Log anzeigen
    sj server autostart on          Autostart beim Boot aktivieren
    sj server autostart off         Autostart deaktivieren
    sj server autostart status      Autostart-Status anzeigen

  {C_BOLD}Hardware-Konflikte:{C_RESET}
    Modelle die gleiche GPUs brauchen koennen nicht gleichzeitig
    auf verschiedenen Slots geladen werden. CPU-Modelle haben
    keine GPU-Konflikte.

  {C_BOLD}Konfiguration:{C_RESET}
    Alle Model-Profile (GPU, CTX, etc.) in config.json unter
    "model_profiles". Externe Steuerung bestimmt nur WELCHES
    Modell auf WELCHEN Slot geladen wird.

  {C_BOLD}Nummern:{C_RESET}
    Modelle: N1, N2, N3, ...   (automatisch nummeriert)
    Slots:   1, 2, 3, 4       (konfiguriert in config.json)
    GPUs:    G0, G1, G2, ...   (NVIDIA Index)
""")


def main():
    parser = argparse.ArgumentParser(description="SkinnyJoe CLI v4.0", add_help=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("help")
    sub.add_parser("models")
    sub.add_parser("slots")
    sub.add_parser("gpus")
    sub.add_parser("status")
    sub.add_parser("resources")
    sub.add_parser("tui")
    sub.add_parser("kill")

    p_server = sub.add_parser("server")
    p_server.add_argument("action", nargs="?", default="status",
                          choices=["start","stop","restart","status","log","autostart"])
    p_server.add_argument("autostart_mode", nargs="?", default="status",
                          help="on|off|status (nur für autostart)")

    p_load = sub.add_parser("load")
    p_load.add_argument("model_id", type=str, help="Modell-ID (z.B. T6, D2, W1, B1, S1)")
    p_load.add_argument("--slot", "-s", type=int, default=None, help="Slot-Nummer (default: Standard-Slot)")
    p_load.add_argument("--ctx", type=str, default=None, metavar="SIZE",
                        help="Kontextfenster überschreiben: 8K, 16384, 32K, ...")

    p_unload = sub.add_parser("unload")
    p_unload.add_argument("--slot", "-s", type=int, default=None, help="Slot-Nummer (default: Standard-Slot)")

    p_ctx = sub.add_parser("ctx")
    p_ctx.add_argument("--slot", "-s", type=int, default=None, help="Slot-Nummer (default: Standard-Slot)")
    p_ctx.add_argument("value", nargs="?", default=None,
                       help="Neue CTX-Größe: 8K, 16384, 32K – ohne Angabe: anzeigen")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("prompts", nargs="+")
    p_ask.add_argument("--slot", "-s", type=int, default=None, help="Slot-Nummer (default: Standard-Slot)")
    p_ask.add_argument("--max-tokens", type=int, default=1024)
    p_ask.add_argument("--show-thought", action="store_true")
    p_ask.add_argument("--image", type=str, default=None, metavar="PFAD")

    p_import = sub.add_parser("import")
    p_import.add_argument("source", type=str, help="Pfad zur Modell-Datei oder -Verzeichnis")
    p_import.add_argument("--copy", action="store_true", help="Kopieren statt Verschieben")
    p_import.add_argument("--no-test", action="store_true", help="Schnelltest überspringen")

    p_transcribe = sub.add_parser("transcribe")
    p_transcribe.add_argument("audio", type=str, help="Pfad zur Audiodatei")
    p_transcribe.add_argument("--slot", "-s", type=int, default=None, help="Slot-Nummer (default: Standard-Slot)")
    p_transcribe.add_argument("--language", "-l", type=str, default=None, help="Sprache (z.B. de, en)")

    p_slot = sub.add_parser("slot")
    p_slot.add_argument("action", nargs="?", default=None, help="set")
    p_slot.add_argument("value", nargs="?", default=None, help="Slot-Nummer (bei 'set')")

    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--slot", "-s", type=int, default=None,
                         help="Slot-Nummer (ohne Angabe: alle geladenen Slots)")
    p_bench.add_argument("--size", type=str, nargs="+", default=[],
                         dest="sizes", metavar="SIZE",
                         help="Token-Größen für text2text: 1K 2K 4K 8K 16K (Standard: alle)")
    p_bench.add_argument("--max-gen", type=int, default=64)

    args = parser.parse_args()
    if not args.command:
        cmd_help(); sys.exit(0)

    cmds = {"help": cmd_help, "models": cmd_models, "slots": cmd_slots, "gpus": cmd_gpus,
            "status": cmd_status, "resources": cmd_resources, "kill": cmd_kill, "tui": cmd_tui}
    if args.command in cmds:
        cmds[args.command]()
    elif args.command == "load":
        ctx_override = _parse_ctx(args.ctx) if args.ctx else None
        cmd_load(args.model_id.upper(), _resolve_slot(args.slot), ctx_override)
    elif args.command == "ctx":
        cmd_ctx(_resolve_slot(args.slot), args.value)
    elif args.command == "unload":
        cmd_unload(_resolve_slot(args.slot))
    elif args.command == "ask":
        cmd_ask(args.prompts, _resolve_slot(args.slot), args.max_tokens, args.show_thought, args.image)
    elif args.command == "import":
        cmd_import(args.source, copy_mode=args.copy, skip_test=args.no_test)
    elif args.command == "transcribe":
        cmd_transcribe(args.audio, _resolve_slot(args.slot), args.language)
    elif args.command == "bench":
        cmd_bench(args.slot, args.sizes, args.max_gen)
    elif args.command == "slot":
        cmd_slot(args.action, args.value)
    elif args.command == "server":
        cmd_server(args.action, args.autostart_mode)

if __name__ == "__main__":
    main()
