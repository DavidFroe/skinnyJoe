#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
import argparse
import signal
import re
import pty
import select
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ANSI_RE = re.compile(r'\x1b(?:\[[0-9;]*[a-zA-Z~]|\].*?\x07|\(B)|\r')

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)

LMSMODEL_DIR = Path.home() / ".lmsmodel"
IDS_FILE = LMSMODEL_DIR / "ids.json"
CONFIGS_DIR = LMSMODEL_DIR / "configs"

VERBOSE = False

def log(msg): print(f"[INFO] {msg}", file=sys.stderr)
def log_debug(msg):
    if VERBOSE: print(f"[DEBUG] {msg}", file=sys.stderr)
def log_warn(msg): print(f"[WARN] {msg}", file=sys.stderr)
def log_error(msg): print(f"[ERROR] {msg}", file=sys.stderr)

def init_fs():
    LMSMODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

class DaemonManager:
    @staticmethod
    def _get_display_env() -> dict:
        env = os.environ.copy()
        env.setdefault("DISPLAY", ":0")
        env.setdefault("WAYLAND_DISPLAY", "wayland-0")
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        return env

    @staticmethod
    def is_running() -> bool:
        try:
            res = subprocess.run(["lms", "ps"], capture_output=True, env=DaemonManager._get_display_env(), text=True)
            return res.returncode == 0
        except FileNotFoundError: return False

    @staticmethod
    def start():
        # Fallback falls lms nicht läuft. Normalerweise machst du das ja per GUI.
        pass

class ModelRegistry:
    @staticmethod
    def _load_ids() -> dict:
        if not IDS_FILE.exists(): return {}
        try:
            with open(IDS_FILE, "r", encoding="utf-8") as f:
                return {int(k): v for k, v in json.load(f).items()}
        except: return {}

    @staticmethod
    def _save_ids(data: dict):
        with open(IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def sync():
        if not DaemonManager.is_running(): return
        res = subprocess.run(["lms", "ls", "--json"], capture_output=True, text=True, env=DaemonManager._get_display_env())
        if res.returncode != 0: return
        try: ls_data = json.loads(res.stdout)
        except: return
        
        db = ModelRegistry._load_ids()
        for k in db: db[k]["status"] = "missing"
        
        max_id = max(db.keys()) if db else 0
        for model in ls_data:
            path, key = model.get("path"), model.get("modelKey")
            if not path or not key: continue
            found = False
            for k, v in db.items():
                if v.get("path") == path or v.get("model_key") == key:
                    db[k]["status"] = "available"
                    db[k]["model_key"] = key
                    found = True
                    break
            if not found:
                max_id += 1
                db[max_id] = {"path": path, "model_key": key, "status": "available"}
        ModelRegistry._save_ids(db)

    @staticmethod
    def get_model_path(model_id: int) -> str:
        db = ModelRegistry._load_ids()
        if model_id not in db: raise RuntimeError(f"Model ID {model_id} not found.")
        return db[model_id].get("model_key", db[model_id].get("path"))

class ConfigStore:
    @staticmethod
    def get_stability_timeout(model_id: int) -> float:
        p = CONFIGS_DIR / f"{model_id}.json"
        if not p.exists(): return 5.0
        try:
            with open(p, "r", encoding="utf-8") as f:
                return float(json.load(f).get("stability_timeout", 5.0))
        except: return 5.0

class InferenceEngine:
    PROMPT_MARKER = "›"

    @staticmethod
    def is_model_loaded(model_key: str) -> bool:
        res = subprocess.run(["lms", "ps", "--json"], capture_output=True, env=DaemonManager._get_display_env(), text=True)
        if res.returncode == 0:
            try:
                for item in json.loads(res.stdout):
                    if model_key in [item.get("path"), item.get("modelKey"), item.get("identifier")]:
                        return True
            except: pass
        return False

    @staticmethod
    def load_model(model_key: str):
        if InferenceEngine.is_model_loaded(model_key): return
        log(f"Lade Modell in LM Studio: {model_key}...")
        subprocess.run(["lms", "load", model_key, "--yes"], capture_output=True, env=DaemonManager._get_display_env())

    @staticmethod
    def infer(model_id: int, model_key: str, prompt: str, system_prompt: str = "", timeout: int = 900) -> str:
        full_prompt = f"[SYSTEM: {system_prompt}]\n\n{prompt}" if system_prompt else prompt
        
        log(f"PTY Wrapper startet 'lms chat {model_key}' (Payload: {len(full_prompt)} Bytes)")

        master_fd, slave_fd = pty.openpty()
        env = DaemonManager._get_display_env()
        env["COLUMNS"] = "500"

        proc = subprocess.Popen(
            ["lms", "chat", model_key],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=env, preexec_fn=os.setsid
        )
        os.close(slave_fd)

        try:
            start_wait = time.time()
            while time.time() - start_wait < 30:
                ready, _, _ = select.select([master_fd], [], [], 0.5)
                if ready:
                    try:
                        chunk = os.read(master_fd, 8192).decode("utf-8", errors="replace")
                        if InferenceEngine.PROMPT_MARKER in strip_ansi(chunk): break
                    except OSError: break

            log("Sende gigantischen Prompt via Bracketed Paste...")
            os.write(master_fd, b"\x1b[200~") 
            
            prompt_bytes = full_prompt.encode("utf-8")
            chunk_size = 2048
            for i in range(0, len(prompt_bytes), chunk_size):
                os.write(master_fd, prompt_bytes[i:i+chunk_size])
                time.sleep(0.01)
                while True:
                    r, _, _ = select.select([master_fd], [], [], 0)
                    if r:
                        try: os.read(master_fd, 8192)
                        except OSError: break
                    else: break
            
            os.write(master_fd, b"\x1b[201~")
            time.sleep(0.05)
            os.write(master_fd, b"\r")
            log("Prompt vollständig eingespritzt. Warte auf Modell...")

            stability_sec = ConfigStore.get_stability_timeout(model_id)
            buf = ""
            start_infer = time.time()
            last_data = time.time()

            while time.time() - start_infer < timeout:
                r, _, _ = select.select([master_fd], [], [], 0.3)
                if r:
                    try:
                        chunk = os.read(master_fd, 8192).decode("utf-8", errors="replace")
                        if not chunk: break
                        buf += chunk
                        last_data = time.time()
                    except OSError: break
                else:
                    if time.time() - last_data >= stability_sec and buf:
                        if InferenceEngine.PROMPT_MARKER in strip_ansi(buf.split('\n')[-1]):
                            break

            cleaned = strip_ansi(buf)
            lines = cleaned.split('\n')
            response_lines = []
            capturing = False
            
            for line in reversed(lines):
                s = line.strip()
                if not capturing:
                    if s.startswith(InferenceEngine.PROMPT_MARKER) or s == InferenceEngine.PROMPT_MARKER:
                        capturing = True
                    continue
                if capturing:
                    if s.startswith(InferenceEngine.PROMPT_MARKER) or s == InferenceEngine.PROMPT_MARKER: break
                    if "Type a message" in s or "/use commands" in s or (s and s[0] in "╭╰│"): continue
                    response_lines.append(line)
            
            response_lines.reverse()
            return "\n".join(response_lines).strip() or cleaned

        finally:
            try:
                os.write(master_fd, b"\x03") 
                time.sleep(0.1)
                os.write(master_fd, b"/exit\r")
            except OSError: pass
            
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except: proc.kill()
            
            try: proc.wait(timeout=2)
            except: pass
            try: os.close(master_fd)
            except: pass

class APIHandler(BaseHTTPRequestHandler):
    _active_model_id = None
    _active_mode = "fafr"

    def _send_json(self, code, data):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/v1/models':
            ModelRegistry.sync()
            db = ModelRegistry._load_ids()
            models = [{"id": int(mid), "model_key": info.get("model_key", ""), "status": info.get("status", "unknown")} for mid, info in db.items()]
            self._send_json(200, {"status": "success", "models": models})
        
        elif self.path == '/api/v1/model/active':
            if self._active_model_id is None:
                self._send_json(200, {"status": "success", "active_model": None})
            else:
                try: real_key = ModelRegistry.get_model_path(self._active_model_id)
                except: real_key = "unknown"
                self._send_json(200, {"status": "success", "active_model": {"id": self._active_model_id, "model_key": real_key, "mode": self._active_mode}})
        
        elif self.path == '/api/v1/status':
            self._send_json(200, {"status": "success"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length else b'{}'
        try: payload = json.loads(post_data.decode('utf-8'))
        except: return self._send_json(400, {"error": "Invalid JSON"})

        if self.path == '/api/v1/model/set':
            mid = payload.get("model_id")
            if not mid: return self._send_json(400, {"error": "Missing model_id"})
            try:
                model_key = ModelRegistry.get_model_path(int(mid))
                InferenceEngine.load_model(model_key)
                APIHandler._active_model_id = int(mid)
                self._send_json(200, {"status": "success", "active_model": {"id": int(mid), "model_key": model_key, "mode": self._active_mode}})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.path in ['/api/v1/ask', '/api/v1/infer']:
            mid = payload.get("model_id", self._active_model_id)
            if not mid: return self._send_json(400, {"error": "Kein Modell aktiv. Bitte vorher per /api/v1/model/set setzen."})
            
            prompt = payload.get("prompt", "")
            sys_prompt = payload.get("system_prompt", "")
            log(f"API empfängt PTY-Anfrage ({content_length} Bytes)...")
            
            try:
                model_key = ModelRegistry.get_model_path(int(mid))
                ans = InferenceEngine.infer(int(mid), model_key, prompt, sys_prompt)
                if self.path == '/api/v1/ask': self._send_json(200, {"response": ans})
                else: self._send_json(200, {"status": "success", "response": ans})
                log("Antwort erfolgreich an Agenten gesendet.")
            except Exception as e:
                log_error(f"Inference Fehler: {e}")
                self._send_json(500, {"error": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    init_fs()
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        ModelRegistry.sync()
        db = ModelRegistry._load_ids()
        log("Gefundene Modelle:")
        for k, v in db.items():
            print(f"[{k}] {v['status']} - {v['model_key']}")
    else:
        log("Starte lmsmodel2 (PTY Bracketed Paste) Server auf Port 5050...")
        ThreadingHTTPServer(('0.0.0.0', 5050), APIHandler).serve_forever()
