#!/usr/bin/env python3
"""
SkinnyJoe TUI v4.0 – Interaktive Terminal-Oberfläche (Multi-Slot)

Curses-basierte TUI:
  - Modelle nach Typ gruppiert durchsuchen
  - Slots verwalten (laden/entladen)
  - GPU-Status + Hardware-Konflikte anzeigen
  - Prompts an Slots senden

Starten: sj tui  oder  python3 skinnyJoe_tui.py
"""
import curses
import json
import sys
import os
import time
import base64
import requests
from pathlib import Path
from urllib.parse import urlparse

MGMT_URL = os.environ.get("SKINNYJOE_URL", "http://localhost:8000")

CP_NORMAL = 0
CP_HEADER = 1
CP_SELECTED = 2
CP_LOADED = 3
CP_CATEGORY = 4
CP_GPU = 6
CP_STATUS_OK = 7
CP_HELP = 9
CP_ERROR = 10
CP_VISION = 11
CP_DIFFUSION = 12


def api_get(path, timeout=5):
    try:
        r = requests.get(f"{MGMT_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_post(path, data=None, timeout=600):
    try:
        r = requests.post(f"{MGMT_URL}{path}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def slot_url(port):
    parsed = urlparse(MGMT_URL)
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


class SkinnyJoeTUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.running = True
        self.cursor = 0
        self.scroll_offset = 0
        self.models = []
        self.slots = []
        self.gpus = []
        self.hw_alloc = {}
        self.status_msg = ""
        self.status_time = 0
        self.display_list = []

        self._setup_curses()
        self._refresh_data()

    def _setup_curses(self):
        curses.curs_set(0)
        curses.use_default_colors()
        self.stdscr.timeout(100)
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(CP_HEADER, curses.COLOR_WHITE, curses.COLOR_BLUE)
            curses.init_pair(CP_SELECTED, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(CP_LOADED, curses.COLOR_BLACK, curses.COLOR_GREEN)
            curses.init_pair(CP_CATEGORY, curses.COLOR_YELLOW, -1)
            curses.init_pair(CP_GPU, curses.COLOR_GREEN, -1)
            curses.init_pair(CP_STATUS_OK, curses.COLOR_GREEN, -1)
            curses.init_pair(CP_HELP, curses.COLOR_WHITE, curses.COLOR_BLUE)
            curses.init_pair(CP_ERROR, curses.COLOR_RED, -1)
            curses.init_pair(CP_VISION, curses.COLOR_MAGENTA, -1)
            curses.init_pair(CP_DIFFUSION, curses.COLOR_CYAN, -1)

    def _refresh_data(self):
        status = api_get("/status")
        if not status:
            self.status_msg = "Daemon nicht erreichbar!"
            self.status_time = time.time()
            return
        self.slots = status.get("slots", [])
        self.gpus = status.get("gpus", [])
        self.hw_alloc = status.get("hardware_alloc", {})

        md = api_get("/v1/models")
        if md:
            self.models = md.get("data", [])
        self._build_display_list()

    def _build_display_list(self):
        self.display_list = []
        for mtype, label in [("text2text", "TEXT-ZU-TEXT (LLMs)"),
                              ("image2text", "BILD-ZU-TEXT (Vision)"),
                              ("text2image", "TEXT-ZU-BILD (Diffusion)")]:
            group = [m for m in self.models if m.get("model_type") == mtype]
            if not group: continue
            self.display_list.append(("header", label, mtype))
            for m in group:
                self.display_list.append(("model", m, mtype))
        if self.cursor >= len(self.display_list):
            self.cursor = max(0, len(self.display_list) - 1)
        self._skip_headers_forward()

    def _skip_headers_forward(self):
        while self.cursor < len(self.display_list) and self.display_list[self.cursor][0] == "header":
            self.cursor += 1
        if self.cursor >= len(self.display_list):
            self.cursor = max(0, len(self.display_list) - 1)

    def _skip_headers_backward(self):
        while self.cursor > 0 and self.display_list[self.cursor][0] == "header":
            self.cursor -= 1

    def _get_selected_model(self):
        if 0 <= self.cursor < len(self.display_list) and self.display_list[self.cursor][0] == "model":
            return self.display_list[self.cursor][1]
        return None

    def _set_status(self, msg):
        self.status_msg = msg
        self.status_time = time.time()

    def _is_model_loaded(self, model_id):
        """Prüft ob ein Modell auf irgendeinem Slot geladen ist."""
        for s in self.slots:
            lm = s.get("loaded_model")
            if lm and lm.get("id") == model_id:
                return s["id"]
        return None

    def dialog_input(self, prompt_text, max_len=60):
        height, width = self.stdscr.getmaxyx()
        dw = min(max_len + 4, width - 4)
        dh = 3
        dy = height // 2 - 1
        dx = (width - dw) // 2
        win = curses.newwin(dh, dw, dy, dx)
        win.box()
        win.addnstr(0, 2, f" {prompt_text} ", dw - 4)
        win.refresh()
        curses.echo()
        curses.curs_set(1)
        try:
            result = win.getstr(1, 2, min(max_len, dw - 4)).decode("utf-8").strip()
        except Exception:
            result = ""
        curses.noecho()
        curses.curs_set(0)
        return result

    def draw(self):
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        if height < 10 or width < 40:
            self.stdscr.addstr(0, 0, "Terminal zu klein!")
            self.stdscr.refresh()
            return

        # Header
        hdr = " SkinnyJoe TUI v4.0 (Multi-Slot) "
        slots_busy = sum(1 for s in self.slots if s.get("loaded_model"))
        right = f" {slots_busy}/{len(self.slots)} Slots belegt "
        hdr_line = hdr + " " * max(0, width - len(hdr) - len(right)) + right
        try:
            self.stdscr.addnstr(0, 0, hdr_line[:width], width, curses.color_pair(CP_HEADER) | curses.A_BOLD)
        except curses.error: pass

        # Slot-Leiste
        slot_y = 1
        slot_line = " Slots: "
        for s in self.slots:
            lm = s.get("loaded_model")
            if lm:
                slot_line += f"[{s['id']}:{lm['name'][:12]}] "
            else:
                slot_line += f"[{s['id']}:leer] "
        try:
            self.stdscr.addnstr(slot_y, 0, slot_line[:width-1].ljust(width-1), width-1, curses.A_BOLD)
        except curses.error: pass

        # Modell-Liste
        list_top = 2
        list_bottom = height - 3
        list_height = list_bottom - list_top

        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        if self.cursor >= self.scroll_offset + list_height:
            self.scroll_offset = self.cursor - list_height + 1

        type_colors = {"text2text": CP_CATEGORY, "image2text": CP_VISION, "text2image": CP_DIFFUSION}

        for i in range(list_height):
            idx = self.scroll_offset + i
            y = list_top + i
            if y >= height - 2 or idx >= len(self.display_list): break

            entry = self.display_list[idx]
            is_sel = (idx == self.cursor)

            if entry[0] == "header":
                cp = type_colors.get(entry[2], CP_CATEGORY)
                line = f"  {entry[1]} {'─' * max(0, width - len(entry[1]) - 4)}"
                try: self.stdscr.addnstr(y, 0, line[:width-1], width-1, curses.color_pair(cp) | curses.A_BOLD)
                except curses.error: pass
            else:
                m = entry[1]
                mid = m.get("id", 0)
                name = m.get("name", "?")
                size = m.get("size_gb", 0)
                quant = m.get("quant") or ""
                profile = m.get("profile", {})
                gpu_ids = profile.get("gpu_ids", [])
                hw = f"G{','.join(str(g) for g in gpu_ids)}" if gpu_ids else "CPU"
                ctx = profile.get("ctx", "?")

                loaded_slot = self._is_model_loaded(mid)
                extra = f" [S{loaded_slot}]" if loaded_slot else ""

                line = f"  N{mid:<3d} {name:<30s} {size:>5.1f}GB {quant:<8s} CTX:{str(ctx):<5s} {hw:<8s}{extra}"
                line = line[:width-1].ljust(width-1)

                if is_sel and loaded_slot:
                    attr = curses.color_pair(CP_LOADED) | curses.A_BOLD
                elif is_sel:
                    attr = curses.color_pair(CP_SELECTED) | curses.A_BOLD
                elif loaded_slot:
                    attr = curses.color_pair(CP_STATUS_OK) | curses.A_BOLD
                else:
                    attr = curses.A_NORMAL
                try: self.stdscr.addnstr(y, 0, line, width-1, attr)
                except curses.error: pass

        # GPU-Zeile
        gpu_y = height - 2
        if self.gpus:
            gp = " | ".join(f"G{g['id']}:{g['name'][:12]} {g['vram_free_gb']:.0f}/{g['vram_total_gb']:.0f}GB"
                            for g in self.gpus)
            gpu_line = f" GPU: {gp}"
        else:
            gpu_line = " GPU: keine erkannt (CPU-Modus)"
        if self.hw_alloc:
            gpu_line += f"  |  Belegt: {', '.join(f'{g}→{s}' for g, s in self.hw_alloc.items())}"

        if self.status_msg and (time.time() - self.status_time) < 5:
            gpu_line += f"  |  {self.status_msg}"

        try: self.stdscr.addnstr(gpu_y, 0, gpu_line[:width-1].ljust(width-1), width-1, curses.color_pair(CP_GPU))
        except curses.error: pass

        # Hilfe
        help_y = height - 1
        help_line = " [Enter]Laden [u]Entladen [p]Prompt [i]Info [s]Slots [r]Rescan [q]Ende"
        try: self.stdscr.addnstr(help_y, 0, help_line[:width-1].ljust(width-1), width-1, curses.color_pair(CP_HELP))
        except curses.error: pass

        self.stdscr.refresh()

    def action_load(self):
        model = self._get_selected_model()
        if not model: return

        slot_str = self.dialog_input(f"Slot fuer N{model['id']} {model['name'][:20]} (1-{len(self.slots)})")
        if not slot_str: return
        try:
            slot_id = int(slot_str)
        except ValueError:
            self._set_status("Ungueltige Slot-Nummer"); return

        self._set_status(f"Lade N{model['id']} → Slot {slot_id}...")
        self.draw()

        result = api_post("/v1/load", {"slot_id": slot_id, "model_id": model["id"]})
        if result and "error" not in result:
            self._set_status(f"N{model['id']} → Slot {slot_id} geladen!")
        else:
            err = result.get("error", "Fehler") if result else "Keine Antwort"
            self._set_status(f"Fehler: {str(err)[:40]}")
        self._refresh_data()

    def action_unload(self):
        slot_str = self.dialog_input(f"Welchen Slot entladen? (1-{len(self.slots)})")
        if not slot_str: return
        try:
            slot_id = int(slot_str)
        except ValueError:
            self._set_status("Ungueltige Slot-Nummer"); return

        result = api_post("/v1/unload", {"slot_id": slot_id})
        if result and "error" not in result:
            self._set_status(f"Slot {slot_id} entladen.")
        else:
            self._set_status("Fehler beim Entladen.")
        self._refresh_data()

    def action_prompt(self):
        # Slot auswählen
        slot_str = self.dialog_input(f"Prompt an welchen Slot? (1-{len(self.slots)})")
        if not slot_str: return
        try:
            slot_id = int(slot_str)
        except ValueError:
            self._set_status("Ungueltige Slot-Nummer"); return

        # Slot finden
        slot_info = next((s for s in self.slots if s["id"] == slot_id), None)
        if not slot_info or not slot_info.get("loaded_model"):
            self._set_status(f"Slot {slot_id} hat kein Modell."); return

        lm = slot_info["loaded_model"]
        port = slot_info["port"]
        base = slot_url(port)

        curses.def_prog_mode()
        curses.endwin()

        print(f"\n{'=' * 60}")
        print(f"  Slot {slot_id} (Port {port}) | {lm['name']} ({lm['model_type']})")
        print(f"{'=' * 60}")

        try:
            prompt = input("\n  Prompt> ").strip()
            if not prompt:
                input("\n  [Enter] Zurueck..."); curses.reset_prog_mode(); self.stdscr.refresh(); return

            max_tokens = 1024
            mt = input("  Max-Tokens (leer=1024)> ").strip()
            if mt:
                try: max_tokens = int(mt)
                except ValueError: pass

            if lm["model_type"] == "text2image":
                print(f"\n  Generiere Bild: \"{prompt[:50]}\"...")
                result = api_post(f"/v1/load", timeout=5)  # dummy
                # Use slot URL directly
                try:
                    r = requests.post(f"{base}/v1/images/generations", json={"prompt": prompt}, timeout=1800)
                    r.raise_for_status()
                    data = r.json().get("data", [])
                    if data and data[0].get("b64_json"):
                        out = os.path.join(os.getcwd(), "response.jpg")
                        with open(out, "wb") as f: f.write(base64.b64decode(data[0]["b64_json"]))
                        print(f"  [OK] Bild: {out}")
                except Exception as e:
                    print(f"  Fehler: {e}")
            else:
                messages = [{"role": "user", "content": prompt}]
                payload = {"messages": messages, "max_tokens": max_tokens, "stream": True}
                try:
                    resp = requests.post(f"{base}/v1/chat/completions", json=payload, stream=True, timeout=600)
                    resp.raise_for_status()
                    full_text, tc, ts = "", 0, time.time()
                    print(f"\n  {'─' * 40}")
                    for line in resp.iter_lines():
                        if not line: continue
                        ls = line.decode("utf-8")
                        if ls.startswith("data: "):
                            raw = ls[6:]
                            if raw == "[DONE]": break
                            try:
                                ch = json.loads(raw)
                                if "choices" in ch and ch["choices"]:
                                    c = ch["choices"][0].get("delta", {}).get("content", "")
                                    if c:
                                        full_text += c; tc += 1
                                        sys.stdout.write(c); sys.stdout.flush()
                            except: continue
                    tt = time.time() - ts
                    print(f"\n  {'─' * 40}")
                    print(f"  {tc} Tokens in {tt:.1f}s ({tc/tt:.1f} tok/s)" if tt > 0 else "")
                except Exception as e:
                    print(f"\n  Fehler: {e}")

        except (KeyboardInterrupt, EOFError):
            print("\n  (Abgebrochen)")

        input("\n  [Enter] Zurueck...")
        curses.reset_prog_mode()
        self.stdscr.refresh()
        self._refresh_data()

    def action_info(self):
        model = self._get_selected_model()
        if not model: return
        curses.def_prog_mode()
        curses.endwin()
        print(f"\n{'=' * 60}")
        print(f"  Modell-Details: N{model['id']}")
        print(f"{'=' * 60}")
        print(f"  Name      : {model.get('full_name', model.get('name', '?'))}")
        print(f"  Typ       : {model.get('model_type', '?')}")
        print(f"  Format    : {model.get('format', '?')}")
        print(f"  Groesse   : {model.get('size_gb', 0)} GB")
        print(f"  Parameter : {model.get('params', '-')}")
        print(f"  Quant     : {model.get('quant', '-')}")
        print(f"  Tags      : {', '.join(model.get('tags', []))}")
        print(f"  Beschreib : {model.get('description', '-')}")
        print(f"  mmproj    : {'ja' if model.get('has_mmproj') else 'nein'}")
        prof = model.get("profile", {})
        print(f"  GPU-Layer : {prof.get('gpu_layers', 0)}")
        print(f"  GPU-IDs   : {prof.get('gpu_ids', [])}")
        print(f"  CTX       : {prof.get('ctx', '?')}")
        print(f"  Batch     : {prof.get('n_batch', '?')}")
        loaded_slot = self._is_model_loaded(model["id"])
        print(f"  Geladen   : {'Slot ' + str(loaded_slot) if loaded_slot else 'nein'}")
        print(f"{'=' * 60}")
        input("\n  [Enter] Zurueck...")
        curses.reset_prog_mode()
        self.stdscr.refresh()

    def action_show_slots(self):
        curses.def_prog_mode()
        curses.endwin()
        print(f"\n{'=' * 60}")
        print(f"  Slot-Uebersicht")
        print(f"{'=' * 60}")
        for s in self.slots:
            lm = s.get("loaded_model")
            st = s.get("status", "idle")
            if lm:
                print(f"  Slot {s['id']} (Port {s['port']}): {lm['name']} ({lm['model_type']}) [{st}]")
            else:
                print(f"  Slot {s['id']} (Port {s['port']}): leer")
        if self.hw_alloc:
            print(f"\n  GPU-Belegung: {', '.join(f'{g} -> {s}' for g, s in self.hw_alloc.items())}")
        print(f"{'=' * 60}")
        input("\n  [Enter] Zurueck...")
        curses.reset_prog_mode()
        self.stdscr.refresh()

    def handle_key(self, key):
        if key in (ord('q'), ord('Q'), 27): self.running = False
        elif key in (curses.KEY_UP, ord('k')):
            if self.cursor > 0: self.cursor -= 1; self._skip_headers_backward()
        elif key in (curses.KEY_DOWN, ord('j')):
            if self.cursor < len(self.display_list) - 1: self.cursor += 1; self._skip_headers_forward()
        elif key == curses.KEY_PPAGE:
            self.cursor = max(0, self.cursor - 10); self._skip_headers_forward()
        elif key == curses.KEY_NPAGE:
            self.cursor = min(len(self.display_list) - 1, self.cursor + 10); self._skip_headers_backward()
        elif key in (10, curses.KEY_ENTER): self.action_load()
        elif key in (ord('u'), ord('U')): self.action_unload()
        elif key in (ord('p'), ord('P')): self.action_prompt()
        elif key in (ord('i'), ord('I')): self.action_info()
        elif key in (ord('s'), ord('S')): self.action_show_slots()
        elif key in (ord('r'), ord('R')):
            api_post("/v1/rescan"); self._refresh_data(); self._set_status("Rescan fertig.")
        elif key == curses.KEY_RESIZE: self.stdscr.clear()

    def run(self):
        last_refresh = time.time()
        while self.running:
            self.draw()
            key = self.stdscr.getch()
            if key != -1: self.handle_key(key)
            if time.time() - last_refresh > 10:
                self._refresh_data(); last_refresh = time.time()


def main():
    try:
        r = requests.get(f"{MGMT_URL}/status", timeout=3)
        r.raise_for_status()
    except Exception:
        print(f"SkinnyJoe Daemon nicht erreichbar ({MGMT_URL}).")
        print("Starte den Daemon zuerst: sj-daemon")
        sys.exit(1)
    try:
        curses.wrapper(lambda stdscr: SkinnyJoeTUI(stdscr).run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
