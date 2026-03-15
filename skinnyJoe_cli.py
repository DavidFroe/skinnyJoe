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
import subprocess
import time
import base64
from pathlib import Path

MGMT_URL = os.environ.get("SKINNYJOE_URL", "http://localhost:8000")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DAEMON_SCRIPT  = os.path.join(BASE_DIR, "skinnyJoe_daemon.py")
INSTALL_SCRIPT = os.path.join(BASE_DIR, "install.sh")
VENV_PYTHON  = os.path.join(BASE_DIR, "venv", "bin", "python3")
PID_FILE     = os.path.join(BASE_DIR, "skinnyjoe.pid")
LOG_FILE     = os.path.join(BASE_DIR, "skinnyjoe.log")

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

TYPE_COLORS = {"text2text": C_BLUE, "image2text": C_MAGENTA, "text2image": C_CYAN}
TYPE_LABELS = {"text2text": "Text-zu-Text (LLMs)", "image2text": "Bild-zu-Text (Vision)",
               "text2image": "Text-zu-Bild (Diffusion)"}
TYPE_ORDER = ["text2text", "image2text", "text2image"]


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

            nr = f"N{mid:<3d}"
            print(f"    {color}{nr}{C_RESET} {name:<33s} {size:>6.1f}GB  {quant:<8s} {C_DIM}CTX:{ctx:<6}{C_RESET} {hw:<14s} {tag_str}{mmproj}")

    print(f"\n  {'─' * 76}")
    print(f"  {C_DIM}Laden: sj load <N> --slot <S>  |  Profile in config.json{C_RESET}\n")


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


def cmd_load(model_id, slot_id):
    print(f"Lade Modell N{model_id} → Slot {slot_id}...")
    result = api_post(MGMT_URL, "/v1/load", {"slot_id": slot_id, "model_id": model_id})
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


def cmd_bench(sizes, max_gen, slot_id):
    slot_base = _slot_url(slot_id)
    s = api_get(slot_base, "/status")
    loaded = s.get("loaded_model")
    if not loaded:
        print(f"{C_RED}Slot {slot_id} hat kein Modell.{C_RESET}", file=sys.stderr); sys.exit(1)
    print(f"{'='*56}\n  Benchmark | Slot {slot_id} | {loaded['name']}\n{'='*56}")
    BASE = ("The quick brown fox jumps over the lazy dog. "
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ")
    rows = []
    for sk in sizes:
        tt = sk * 1000; cn = max(100, (tt - 80) * 4)
        pt = (BASE * ((cn // len(BASE)) + 1))[:cn]
        print(f"  [{sk:>3}K] ", end="", flush=True)
        payload = {"messages": [{"role":"user","content":f"Summarize:\n{pt}"}], "max_tokens": max_gen, "stream": True}
        ts = time.time()
        try:
            r = requests.post(f"{slot_base}/v1/chat/completions", json=payload, stream=True, timeout=3600)
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
                        if "choices" in ch and ch["choices"]:
                            c = ch["choices"][0].get("delta",{}).get("content","")
                            if c:
                                if tf is None: tf = time.time()
                                tc += 1
                                sys.stdout.write(f"\r  [{sk:>3}K] {time.time()-ts:6.1f}s|{tc}tok...")
                                sys.stdout.flush()
                    except: continue
            td = time.time()
            sys.stdout.write(f"\r  [{sk:>3}K] ")
            tp = (tf-ts) if tf else (td-ts); tg = (td-tf) if tf and tc>0 else 0
            pp = tt/tp if tp>0 else 0; gs = tc/tg if tg>0 else 0
            print(f"PP {tp:.2f}s ({pp:.0f}t/s) | Gen {tg:.2f}s ({gs:.1f}t/s) [{tc}tok]")
            rows.append([f"{sk}K",f"{tp:.2f}s",f"{pp:.0f}",f"{tg:.2f}s",f"{gs:.1f}",f"{tc}"])
        except Exception as e:
            print(f"FEHLER: {e}"); rows.append([f"{sk}K","-","-","-","-","err"])
    if rows:
        hd = ["Größe","PP","PP t/s","Gen","Gen t/s","Tok"]
        cw = [max(len(h),max((len(r[i]) for r in rows),default=0)) for i,h in enumerate(hd)]
        sep = "+-"+"-+-".join("-"*w for w in cw)+"-+"
        print(f"\n{sep}")
        print("| "+" | ".join(h.ljust(w) for h,w in zip(hd,cw))+" |")
        print(sep)
        for row in rows: print("| "+" | ".join(c.ljust(w) for c,w in zip(row,cw))+" |")
        print(sep)


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

  {C_BOLD}System:{C_RESET}
    sj gpus                         NVIDIA GPUs anzeigen
    sj status                       Gesamtstatus
    sj resources                    RAM/VRAM/CPU/GPU-Auslastung + geladene Modelle
    sj bench --slot 1               Benchmark auf Slot 1
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
    p_load.add_argument("model_id", type=str, help="Modell-Nummer (z.B. N15 oder 15)")
    p_load.add_argument("--slot", "-s", type=int, default=1, help="Slot-Nummer (default: 1)")

    p_unload = sub.add_parser("unload")
    p_unload.add_argument("--slot", "-s", type=int, default=1, help="Slot-Nummer (default: 1)")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("prompts", nargs="+")
    p_ask.add_argument("--slot", "-s", type=int, default=1, help="Slot-Nummer (default: 1)")
    p_ask.add_argument("--max-tokens", type=int, default=1024)
    p_ask.add_argument("--show-thought", action="store_true")
    p_ask.add_argument("--image", type=str, default=None, metavar="PFAD")

    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--slot", "-s", type=int, default=1)
    p_bench.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p_bench.add_argument("--max-gen", type=int, default=64)

    args = parser.parse_args()
    if not args.command:
        cmd_help(); sys.exit(0)

    cmds = {"help": cmd_help, "models": cmd_models, "slots": cmd_slots, "gpus": cmd_gpus,
            "status": cmd_status, "resources": cmd_resources, "kill": cmd_kill, "tui": cmd_tui}
    if args.command in cmds:
        cmds[args.command]()
    elif args.command == "load":
        mid_str = args.model_id.lstrip("Nn")
        try:
            cmd_load(int(mid_str), args.slot)
        except ValueError:
            print(f"{C_RED}Ungültige Modell-Nummer: '{args.model_id}' — Beispiel: sj load N15 --slot 1{C_RESET}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "unload":
        cmd_unload(args.slot)
    elif args.command == "ask":
        cmd_ask(args.prompts, args.slot, args.max_tokens, args.show_thought, args.image)
    elif args.command == "bench":
        cmd_bench(args.sizes, args.max_gen, args.slot)
    elif args.command == "server":
        cmd_server(args.action, args.autostart_mode)

if __name__ == "__main__":
    main()
