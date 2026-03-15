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
from http.server import HTTPServer, BaseHTTPRequestHandler

# ANSI escape code stripper
ANSI_RE = re.compile(r'\x1b(?:\[[0-9;]*[a-zA-Z]|\].*?\x07|\(B)|\r')

def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes and carriage returns from terminal output."""
    return ANSI_RE.sub('', text)

# --- Globals & Paths ---
LMSMODEL_DIR = Path.home() / ".lmsmodel"
IDS_FILE = LMSMODEL_DIR / "ids.json"
CONFIGS_DIR = LMSMODEL_DIR / "configs"
HISTORY_FILE = LMSMODEL_DIR / "history.json"
SERVER_PID_FILE = LMSMODEL_DIR / "server.pid"
SERVER_LOG_FILE = LMSMODEL_DIR / "server.log"

START_SCRIPT = Path.home() / "LLMStudio" / "start-lms.sh"
STOP_SCRIPT = Path.home() / "LLMStudio" / "stop-lms.sh"


# --- Logging (stderr only) ---
VERBOSE = False

def log(msg):
    print(f"[INFO] {msg}", file=sys.stderr)

def log_debug(msg):
    if VERBOSE:
        print(f"[DEBUG] {msg}", file=sys.stderr)

def log_warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr)

def log_error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)

# --- Initialization ---
def init_fs():
    LMSMODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Core Modules ---

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
            # lms ps returns 0 if daemon is reachable and responding
            result = subprocess.run(
                ["lms", "ps"],
                capture_output=True,
                env=DaemonManager._get_display_env(),
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            raise RuntimeError("'lms' command not found. Please install LM Studio CLI.")

    @staticmethod
    def start():
        if not DaemonManager.is_running():
            log("LM Studio Daemon is offline. Starting via start-lms.sh...")
            if not START_SCRIPT.exists():
                raise RuntimeError(f"Start script not found: {START_SCRIPT}")
            subprocess.run([str(START_SCRIPT)], env=DaemonManager._get_display_env(), check=True)
            log("Waiting 5 seconds for daemon to initialize...")
            time.sleep(5)
            # Give it more time if not yet ready
            retries = 5
            while not DaemonManager.is_running() and retries > 0:
                log("Daemon not yet ready, waiting...")
                time.sleep(2)
                retries -= 1
            if not DaemonManager.is_running():
                raise RuntimeError("Failed to start LM Studio Daemon.")
            log("Daemon is now online.")

    @staticmethod
    def stop():
        log("Stopping LM Studio Daemon to free VRAM...")
        if not STOP_SCRIPT.exists():
            raise RuntimeError(f"Stop script not found: {STOP_SCRIPT}")
        subprocess.run([str(STOP_SCRIPT)], env=DaemonManager._get_display_env(), check=True)
        log("Daemon stopped.")


class ModelRegistry:
    @staticmethod
    def _load_ids() -> dict:
        if not IDS_FILE.exists():
            return {}
        try:
            with open(IDS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert string keys to int where possible
                return {int(k): v for k, v in data.items()}
        except Exception as e:
            log_warn(f"Failed to read ids.json: {e}. Starting fresh.")
            return {}

    @staticmethod
    def _save_ids(data: dict):
        with open(IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def sync():
        """Runs 'lms ls', parses outputs, assigns integer IDs, missing old ones."""
        if not DaemonManager.is_running():
            log("LM Studio daemon is offline. Starting it to sync models...")
            try:
                DaemonManager.start()
            except Exception as e:
                log_warn(f"Could not start daemon: {e}. Using cached list.")
                return

        result = subprocess.run(["lms", "ls", "--json"], capture_output=True, text=True, env=DaemonManager._get_display_env())
        if result.returncode != 0:
            log_warn(f"Failed to list models: {result.stderr.strip()}. Using cached list.")
            return
        
        try:
            ls_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            log_warn("Failed to parse 'lms ls --json' output. Using cached list.")
            return

        current_models = []
        for item in ls_data:
            path = item.get("path")
            model_key = item.get("modelKey")
            if path and model_key:
                current_models.append({"path": path, "model_key": model_key})

        db = ModelRegistry._load_ids()
        
        # Mark all as missing initially
        for k in db.keys():
            db[k]["status"] = "missing"

        # Check existing and add new
        max_id = max(db.keys()) if db else 0
        
        for model in current_models:
            # Check if it exists (by path OR model_key for backward compatibility)
            found = False
            for k, v in db.items():
                if v.get("path") == model["path"] or v.get("model_key") == model["model_key"]:
                    db[k]["status"] = "available"
                    db[k]["model_key"] = model["model_key"] # Ensure model_key is populated
                    found = True
                    break
            
            if not found:
                max_id += 1
                db[max_id] = {
                    "path": model["path"],
                    "model_key": model["model_key"],
                    "status": "available"
                }

        ModelRegistry._save_ids(db)

    @staticmethod
    def get_list() -> dict:
        return ModelRegistry._load_ids()

    @staticmethod
    def get_model_path(model_id: int) -> str:
        db = ModelRegistry._load_ids()
        if model_id not in db:
            raise RuntimeError(f"Model ID {model_id} not found in registry.")
        
        model_info = db[model_id]
        if model_info["status"] != "available":
            raise RuntimeError(f"Model ID {model_id} is marked as missing. Please check.")
            
        # Return model_key if available, fallback to path for old ids.json entries
        return model_info.get("model_key", model_info.get("path"))


class ConfigStore:
    @staticmethod
    def _get_path(model_id: int) -> Path:
        return CONFIGS_DIR / f"{model_id}.json"

    @staticmethod
    def get_all(model_id: int) -> dict:
        p = ConfigStore._get_path(model_id)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Migration: if config exists but is missing the new auto fields, recreate them
        if "param_size_b" not in data and "model_key" in data:
            ConfigStore.ensure_defaults(model_id, data["model_key"], force=True)
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                
        return data

    @staticmethod
    def get_stability_timeout(model_id: int) -> float:
        """Get custom stability timeout for a model, default to 1.5s."""
        cfg = ConfigStore.get_all(model_id)
        return float(cfg.get("stability_timeout", 1.5))

    @staticmethod
    def set(model_id: int, key: str, value: str):
        # Value is passed as string from CLI, try to cast to int/float/bool if appropriate
        if value.lower() in ["true", "false"]:
            val = value.lower() == "true"
        else:
            try:
                if "." in value:
                    val = float(value)
                else:
                    val = int(value)
            except ValueError:
                val = value  # Keep as string
        
        data = ConfigStore.get_all(model_id)
        data[key] = val
        with open(ConfigStore._get_path(model_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        log(f"Set config '{key}' = {val} for Model ID {model_id}")

    @staticmethod
    def ensure_defaults(model_id: int, model_key: str, force: bool = False):
        """Auto-create a config file with sensible defaults if none exists."""
        p = ConfigStore._get_path(model_id)
        
        existing_data = {}
        if p.exists() and not force:
            return  # Already configured
            
        if p.exists() and force:
            with open(p, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

        # Parse model size from key name
        # Handles "-14b", "_14b", "-8x3b" (MoE) -> extracts the main param count roughly
        # For MoE like 8x3b we will just multiply to get a rough estimate for stability timeout
        param_billions = 7  # default assume 7B
        
        # Check for MoE first (e.g. 8x3b)
        moe_match = re.search(r'(?:[-_/])?(\d+)x(\d+)[bB](?:[-_/.v]|$)', model_key.lower())
        if moe_match:
            param_billions = int(moe_match.group(1)) * int(moe_match.group(2))
        else:
            # Check for standard sizing (e.g. 103b, 7b)
            size_match = re.search(r'(?:[-_/])?(?:qwen3-|llama-3[.-]?[0-9]*[-_])?(\d+)[bB](?:[-_/.v]|$)', model_key.lower())
            if not size_match:
                # Try generic regex as fallback if prefix matched nothing
                size_match = re.search(r'(?:^|[-_/])(\d+)[bB](?:[-_/.v]|$)', model_key.lower())
                
            if size_match:
                param_billions = int(size_match.group(1))

        # Set stability_timeout based on model size
        if param_billions <= 10:
            stability = 1.5
        elif param_billions <= 30:
            stability = 5.0
        elif param_billions <= 70:
            stability = 10.0
        else:
            stability = 15.0

        defaults = {
            "model_key": model_key,
            "param_size_b": param_billions,
            "stability_timeout": stability,
            "temperature": 0.8,
            "max_tokens": -1,  # -1 = unlimited (LM Studio default)
            "notes": "Auto-generated config. Edit with: lmsmodel config <ID> <key> <value>"
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(defaults, f, indent=4)
        log(f"Created default config for Model ID {model_id} ({model_key}, ~{param_billions}B, stability: {stability}s)")


class HistoryManager:
    @staticmethod
    def get_history() -> list:
        if not HISTORY_FILE.exists():
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []

    @staticmethod
    def append(role: str, content: str):
        hist = HistoryManager.get_history()
        hist.append({"role": role, "content": content})
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=4)

    @staticmethod
    def clear():
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        log("History cleared.")


class InferenceEngine:
    PROMPT_MARKER = "›"  # U+203A - lms chat uses this as prompt character

    @staticmethod
    def is_model_loaded(model_key: str) -> bool:
        result = subprocess.run(["lms", "ps", "--json"], capture_output=True, text=True, env=DaemonManager._get_display_env())
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                for item in data:
                    if model_key in [item.get("path"), item.get("modelKey"), item.get("identifier")]:
                        return True
            except:
                pass
        return False

    @staticmethod
    def load_model(model_key: str):
        if InferenceEngine.is_model_loaded(model_key):
            log(f"Model '{model_key}' is already loaded. Skipping load.")
            return
        log(f"Loading model: {model_key} via 'lms load'...")
        result = subprocess.run(
            ["lms", "load", model_key, "--yes"],
            capture_output=True, text=True,
            env=DaemonManager._get_display_env()
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to load model: {result.stderr.strip()}")
        log("Model loaded successfully.")

    @staticmethod
    def _pty_read_until(fd: int, marker: str, timeout: int) -> str:
        """Read from PTY fd until marker char appears on the last line, or timeout."""
        buf = ""
        start = time.time()
        while time.time() - start < timeout:
            ready, _, _ = select.select([fd], [], [], 0.5)
            if ready:
                try:
                    chunk = os.read(fd, 8192).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    buf += chunk
                    cleaned = strip_ansi(buf)
                    last_lines = [l for l in cleaned.split('\n') if l.strip()]
                    if last_lines and marker in last_lines[-1]:
                        log_debug(f"PTY: marker '{marker}' found after {time.time()-start:.1f}s")
                        return buf
                except OSError:
                    break
        log_debug(f"PTY: read_until timed out after {timeout}s (buf len: {len(buf)})")
        return buf

    @staticmethod
    def _pty_read_response(fd: int, marker: str, timeout: int, stability_seconds: float = 1.5) -> str:
        """Read model response using stability detection.
        
        Keeps reading data until no new data arrives for >= stability_seconds
        AND the output ends with the prompt marker. This naturally handles
        multiple rapid TUI redraws (which happen in <0.1s) because the
        stability window won't trigger during them.
        """
        buf = ""
        start = time.time()
        last_data_time = time.time()

        while time.time() - start < timeout:
            ready, _, _ = select.select([fd], [], [], 0.3)
            if ready:
                try:
                    chunk = os.read(fd, 8192).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    buf += chunk
                    last_data_time = time.time()
                except OSError:
                    break
            else:
                # No new data in this cycle
                elapsed_since_data = time.time() - last_data_time
                if elapsed_since_data >= stability_seconds and buf:
                    # Check if output has settled with a prompt marker at the end
                    cleaned = strip_ansi(buf)
                    last_lines = [l for l in cleaned.split('\n') if l.strip()]
                    if last_lines and marker in last_lines[-1]:
                        log_debug(f"PTY: stable response detected after {time.time()-start:.1f}s "
                                  f"(quiet for {elapsed_since_data:.1f}s, buf: {len(buf)} bytes)")
                        return buf

        log_debug(f"PTY: read_response timed out after {timeout}s (buf len: {len(buf)})")
        return buf

    @staticmethod
    def infer(model_id: int, model_key: str, prompt: str, system_prompt: str = "", timeout: int = 360) -> str:
        """Run inference by piping through 'lms chat' CLI via pseudo-terminal."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"[SYSTEM: {system_prompt}]  {prompt}"

        # Replace newlines with spaces to avoid lms chat interpreting them as separate inputs
        full_prompt_safe = full_prompt.replace("\n", " ")

        log_debug(f"Spawning: lms chat {model_key}")
        log_debug(f"Prompt: {full_prompt_safe[:200]}...")

        master_fd, slave_fd = pty.openpty()
        env = DaemonManager._get_display_env()
        env["COLUMNS"] = "500"

        proc = subprocess.Popen(
            ["lms", "chat", model_key],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=env
        )
        os.close(slave_fd)

        try:
            # Phase 1: Wait for initial prompt (›)
            log_debug("Phase 1: Waiting for lms chat to be ready...")
            init_output = InferenceEngine._pty_read_until(master_fd, InferenceEngine.PROMPT_MARKER, timeout=30)
            cleaned_init = strip_ansi(init_output)
            log_debug(f"Phase 1 done. ({len(cleaned_init)} chars)")

            # Phase 2: Send the prompt
            # TUI frameworks (like inquirer.js) often drop input if text + enter are sent instantly
            log_debug("Phase 2: Sending prompt (simulating human input)...")
            os.write(master_fd, full_prompt_safe.encode("utf-8"))
            time.sleep(0.1)  # Let the TUI process the text buffer
            os.write(master_fd, b"\r")  # Send carriage return to trigger submission
            time.sleep(0.1)  # Let the TUI start processing


            # Phase 3: Read ALL output using stability detection
            # This will absorb all TUI redraws (rapid, <0.1s gaps) AND the model response,
            # and only return once output has been quiet for `stability_seconds` with a › at the end.
            stability_sec = ConfigStore.get_stability_timeout(model_id)
            log_debug(f"Phase 3: Awaiting model response (stability-based, timeout: {timeout}s, stability_window: {stability_sec}s)...")
            raw_response = InferenceEngine._pty_read_response(
                master_fd, 
                InferenceEngine.PROMPT_MARKER, 
                timeout=timeout,
                stability_seconds=stability_sec
            )
            cleaned = strip_ansi(raw_response)
            log_debug(f"Phase 3 done. Raw cleaned output ({len(cleaned)} chars):\n{cleaned[:800]}")

            # Phase 4: Parse the response - find the LONGEST animation frame
            # lms chat redraws the terminal for every token (streaming), creating
            # hundreds of progressive "animation frames" in the PTY buffer.
            # The stability timeout may fire mid-frame, making the LAST frame
            # incomplete. We collect ALL frames and pick the LONGEST one.
            
            lines = cleaned.split('\n')
            prompt_start = full_prompt_safe[:15].strip()
            
            # Find all indices where our prompt was echoed
            prompt_indices = []
            for i, line in enumerate(lines):
                s = line.strip()
                if s.startswith(InferenceEngine.PROMPT_MARKER) and prompt_start in s:
                    prompt_indices.append(i)
            
            best_response = ""
            
            if prompt_indices:
                for idx in prompt_indices:
                    # Extract text after this prompt echo until next prompt or TUI element
                    frame_lines = []
                    for line in lines[idx + 1:]:
                        s = line.strip()
                        if s.startswith(InferenceEngine.PROMPT_MARKER):
                            break
                        if "Type a message" in s or "/use commands" in s:
                            break
                        if not s and not frame_lines:
                            continue
                        if s and s[0] in "╭╰│":
                            continue
                        if "lms chat" in s or "Chatting with" in s:
                            continue
                        frame_lines.append(line)
                    
                    candidate = "\n".join(frame_lines).strip()
                    if len(candidate) > len(best_response):
                        best_response = candidate
                
                log_debug(f"Phase 4: Found {len(prompt_indices)} animation frames, picked longest ({len(best_response)} chars)")
            else:
                log_warn("Phase 4: Could not find echoed prompt in output! Falling back to raw extraction.")
                best_response = cleaned

            result = best_response
            log_debug(f"Phase 4 done. Parsed response ({len(result)} chars): {result[:200]}...")

            if not result:
                log_warn("No response text was captured from the model.")
                log_debug(f"Full cleaned output was:\n{cleaned}")

            return result

        finally:
            # Phase 5: Exit cleanly
            log_debug("Phase 5: Cleaning up lms chat process...")
            try:
                os.write(master_fd, b"exit\r")
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            try:
                os.close(master_fd)
            except OSError:
                pass
            log_debug("Cleanup complete.")


# --- Operating Modes ---

class Modes:
    @staticmethod
    def faf(model_id: int, prompt: str, system_prompt: str):
        DaemonManager.start()
        model_key = ModelRegistry.get_model_path(model_id)
        ConfigStore.ensure_defaults(model_id, model_key)
        InferenceEngine.load_model(model_key)
        log("Sending inference request...")
        response = InferenceEngine.infer(model_id, model_key, prompt, system_prompt)
        sys.stdout.write(response + "\n")
        sys.stdout.flush()

        DaemonManager.stop()

    @staticmethod
    def fafr(model_id: int, prompt: str, system_prompt: str):
        DaemonManager.start()
        model_key = ModelRegistry.get_model_path(model_id)
        ConfigStore.ensure_defaults(model_id, model_key)
        InferenceEngine.load_model(model_key)
        log("Sending inference request...")
        response = InferenceEngine.infer(model_id, model_key, prompt, system_prompt)
        sys.stdout.write(response + "\n")
        sys.stdout.flush()

    @staticmethod
    def assistent(model_id: int, prompt: str, system_prompt: str):
        DaemonManager.start()
        model_key = ModelRegistry.get_model_path(model_id)
        ConfigStore.ensure_defaults(model_id, model_key)
        InferenceEngine.load_model(model_key)
        # Build context from history
        history = HistoryManager.get_history()
        if history:
            context_parts = []
            for msg in history:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                context_parts.append(f"[{role}]: {content}")
            context_str = " | ".join(context_parts)
            full_prompt = f"[Previous conversation: {context_str}] Now respond to: {prompt}"
        else:
            full_prompt = prompt

        log("Sending inference request...")
        response = InferenceEngine.infer(model_id, model_key, full_prompt, system_prompt)

        HistoryManager.append("user", prompt)
        HistoryManager.append("assistant", response)

        sys.stdout.write(response + "\n")
        sys.stdout.flush()

    @staticmethod
    def demon(model_id: int, prompt_file: str, system_prompt: str):
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise RuntimeError(f"File not found: {prompt_file}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            raise RuntimeError("Prompt file is empty.")

        DaemonManager.start()
        model_key = ModelRegistry.get_model_path(model_id)
        ConfigStore.ensure_defaults(model_id, model_key)
        InferenceEngine.load_model(model_key)

        results = []
        for i, prompt in enumerate(lines):
            log(f"Processing prompt {i+1}/{len(lines)}...")
            response = InferenceEngine.infer(model_id, model_key, prompt, system_prompt)
            results.append(f"Q: {prompt}\nA: {response}\n---")

        final_str = "\n".join(results)
        sys.stdout.write(final_str)
        sys.stdout.flush()

        # Notify
        subprocess.run(["notify-send", "LM Studio", "Alle Prompts abgearbeitet!"], env=DaemonManager._get_display_env())

    @staticmethod
    def dispatch(mode: str, model_id: int, prompt: str, system_prompt: str):
        if mode == "faf":
            Modes.faf(model_id, prompt, system_prompt)
        elif mode == "fafr":
            Modes.fafr(model_id, prompt, system_prompt)
        elif mode == "assistent":
            Modes.assistent(model_id, prompt, system_prompt)
        elif mode == "demon":
            Modes.demon(model_id, prompt, system_prompt)
        else:
            raise RuntimeError(f"Unknown mode: {mode}")



# --- REST Gateway Server ---

class APIHandler(BaseHTTPRequestHandler):
    _active_model_id = None  # Server-wide active model
    _active_mode = "fafr"    # Default inference mode

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format%args))

    def _send_json(self, code, data):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def _do_inference(self, model_id, prompt, mode=None, system_prompt=""):
        """Shared inference logic used by both /infer and /ask."""
        mode = mode or APIHandler._active_mode
        log(f"Inference (Mode: {mode}, Model: {model_id}, Prompt: {prompt[:50]}...)")
        
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            Modes.dispatch(mode, int(model_id), str(prompt), str(system_prompt))
            result = sys.stdout.getvalue()
            sys.stdout = old_stdout
            return result
        except BaseException as e:
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
            raise

    def do_GET(self):
        if self.path == '/api/v1/models':
            try:
                ModelRegistry.sync()
                db = ModelRegistry.get_list()
                models = []
                for mid, info in db.items():
                    model_key = info.get("model_key", info.get("path", ""))
                    ConfigStore.ensure_defaults(int(mid), model_key)
                    cfg = ConfigStore.get_all(int(mid))
                    models.append({
                        "id": int(mid),
                        "model_key": model_key,
                        "path": info.get("path", ""),
                        "status": info.get("status", "unknown"),
                        "config": cfg
                    })
                self._send_json(200, {"status": "success", "models": models})
            except BaseException as e:
                log_error(f"Model list error: {e}")
                self._send_json(500, {"status": "error", "error": str(e)})

        elif self.path == '/api/v1/model/active':
            mid = APIHandler._active_model_id
            if mid is None:
                self._send_json(200, {"status": "success", "active_model": None, 
                                       "message": "No model set. POST to /api/v1/model/set"})
            else:
                cfg = ConfigStore.get_all(mid)
                db = ModelRegistry.get_list()
                info = db.get(str(mid), {})
                self._send_json(200, {
                    "status": "success",
                    "active_model": {
                        "id": mid,
                        "model_key": info.get("model_key", cfg.get("model_key", "?")),
                        "mode": APIHandler._active_mode,
                        "config": cfg
                    }
                })

        elif self.path == '/api/v1/status':
            mid = APIHandler._active_model_id
            self._send_json(200, {
                "status": "success",
                "message": "lmsmodel REST Gateway is running",
                "active_model_id": mid,
                "active_mode": APIHandler._active_mode
            })
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length else b'{}'

        try:
            payload = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "error": "Invalid JSON"})
            return

        if self.path == '/api/v1/model/set':
            model_id = payload.get("model_id")
            if not model_id:
                self._send_json(400, {"status": "error", "error": "Missing model_id"})
                return

            model_id = int(model_id)
            # Optional: set the default inference mode too
            mode = payload.get("mode", APIHandler._active_mode)
            if mode in ("faf", "fafr", "assistent", "demon"):
                APIHandler._active_mode = mode

            try:
                model_key = ModelRegistry.get_model_path(model_id)
                ConfigStore.ensure_defaults(model_id, model_key)
                DaemonManager.start()
                InferenceEngine.load_model(model_key)
                APIHandler._active_model_id = model_id
                cfg = ConfigStore.get_all(model_id)
                log(f"Active model set to ID {model_id} ({model_key}), mode={APIHandler._active_mode}")
                self._send_json(200, {
                    "status": "success",
                    "message": f"Active model set to {model_key}",
                    "active_model": {"id": model_id, "model_key": model_key, 
                                     "mode": APIHandler._active_mode, "config": cfg}
                })
            except BaseException as e:
                log_error(f"Model set error: {e}")
                self._send_json(500, {"status": "error", "error": str(e)})

        elif self.path == '/api/v1/ask':
            # ── SIMPLEST ENDPOINT ──
            # Just send {"prompt": "..."} and get the answer back.
            prompt = payload.get("prompt")
            if not prompt:
                self._send_json(400, {"status": "error", "error": "Missing prompt"})
                return
            if not APIHandler._active_model_id:
                self._send_json(400, {"status": "error", 
                    "error": "No active model. POST to /api/v1/model/set first."})
                return
            
            try:
                result = self._do_inference(APIHandler._active_model_id, prompt)
                self._send_json(200, {"response": result.strip()})
            except BaseException as e:
                log_error(f"Ask error: {e}")
                self._send_json(500, {"status": "error", "error": str(e)})

        elif self.path == '/api/v1/infer':
            model_id = payload.get("model_id") or APIHandler._active_model_id
            prompt = payload.get("prompt")
            mode = payload.get("mode")
            system_prompt = payload.get("system_prompt", "")

            if not model_id:
                self._send_json(400, {"status": "error", 
                    "error": "No model_id given and no active model set."})
                return
            if not prompt:
                self._send_json(400, {"status": "error", "error": "Missing prompt"})
                return

            try:
                result = self._do_inference(model_id, prompt, mode, system_prompt)
                self._send_json(200, {"status": "success", "model_id": int(model_id), "response": result})
            except BaseException as e:
                log_error(f"Gateway error: {e}")
                error_msg = str(e) if not isinstance(e, SystemExit) else "SystemExit (check daemon/model status)"
                self._send_json(500, {"status": "error", "error": error_msg})
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int):
    """Run the REST gateway (called from the daemonized process or foreground)."""
    # Write PID
    SERVER_PID_FILE.write_text(str(os.getpid()))

    # Redirect output to log file
    log_fh = open(SERVER_LOG_FILE, "a", buffering=1)  # line-buffered
    sys.stdout = log_fh
    sys.stderr = log_fh

    server_address = ('', port)
    httpd = HTTPServer(server_address, APIHandler)
    log(f"REST Gateway started on port {port} (PID: {os.getpid()})")

    def _shutdown(signum, frame):
        log("Received shutdown signal. Stopping server...")
        httpd.server_close()
        SERVER_PID_FILE.unlink(missing_ok=True)
        log("Server stopped.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _shutdown(None, None)


class ServerManager:
    @staticmethod
    def _get_pid() -> int:
        if SERVER_PID_FILE.exists():
            try:
                pid = int(SERVER_PID_FILE.read_text().strip())
                os.kill(pid, 0)  # check if alive
                return pid
            except (ValueError, ProcessLookupError, PermissionError):
                SERVER_PID_FILE.unlink(missing_ok=True)
        return 0

    @staticmethod
    def status():
        pid = ServerManager._get_pid()
        if pid:
            log(f"Server is RUNNING (PID: {pid})")
            if SERVER_LOG_FILE.exists():
                # Show last 5 lines of log
                lines = SERVER_LOG_FILE.read_text().strip().split('\n')
                for line in lines[-5:]:
                    print(f"  {line}", file=sys.stderr)
        else:
            log("Server is NOT running.")

    @staticmethod
    def start(port: int):
        pid = ServerManager._get_pid()
        if pid:
            log(f"Server is already running (PID: {pid}). Use 'server restart' to restart.")
            return

        log(f"Starting REST Gateway on port {port} (daemonizing)...")
        child_pid = os.fork()
        if child_pid == 0:
            # Child process: detach from terminal
            os.setsid()
            # Fork again to fully daemonize
            grandchild = os.fork()
            if grandchild == 0:
                # Grandchild: this is the actual server
                # Close inherited fds
                sys.stdin.close()
                run_server(port)
            else:
                os._exit(0)  # First child exits
        else:
            # Parent: wait for first child to exit
            os.waitpid(child_pid, 0)
            time.sleep(0.5)  # Give server time to start
            new_pid = ServerManager._get_pid()
            if new_pid:
                log(f"Server started successfully (PID: {new_pid})")
                log(f"Log file: {SERVER_LOG_FILE}")
            else:
                log_error("Server failed to start. Check log: " + str(SERVER_LOG_FILE))

    @staticmethod
    def stop():
        pid = ServerManager._get_pid()
        if not pid:
            log("Server is not running.")
            return
        log(f"Stopping server (PID: {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit
            for _ in range(20):
                time.sleep(0.25)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            else:
                log_warn("Server did not exit gracefully, sending SIGKILL...")
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        SERVER_PID_FILE.unlink(missing_ok=True)
        log("Server stopped.")

    @staticmethod
    def restart(port: int):
        ServerManager.stop()
        time.sleep(0.5)
        ServerManager.start(port)

    @staticmethod
    def tail_log():
        if not SERVER_LOG_FILE.exists():
            log("No log file found. Start the server first.")
            return
        log(f"Tailing {SERVER_LOG_FILE} (Ctrl+C to stop)...")
        try:
            # Use tail -f for live following
            subprocess.run(["tail", "-f", "-n", "50", str(SERVER_LOG_FILE)])
        except KeyboardInterrupt:
            pass


# --- CLI Setup ---

def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="lmsmodel - VRAM and LLM Manager for LM Studio CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # list
    subparsers.add_parser("list", help="List models from registry (syncs with lms ls)")

    # config <ID> show | <ID> <key> <val>
    parser_config = subparsers.add_parser("config", help="Manage model configurations")
    parser_config.add_argument("id", type=int, help="Model ID")
    parser_config.add_argument("key", help="Key to set, or 'show' to view config")
    parser_config.add_argument("value", nargs="?", help="Value to set")

    # infer
    parser_infer = subparsers.add_parser("infer", help="Run inference")
    parser_infer.add_argument("id", type=int, help="Model ID")
    parser_infer.add_argument("prompt", help="Prompt string OR path to file (if mode is demon)")
    parser_infer.add_argument("--mode", choices=["faf", "fafr", "assistent", "demon"], required=True, help="Operating mode")
    parser_infer.add_argument("--system", help="Optional System Prompt", default="")

    # clear
    subparsers.add_parser("clear", help="Clear assistant history")

    # server
    parser_server = subparsers.add_parser("server", help="Manage REST Gateway")
    parser_server.add_argument("action", nargs="?", default="status",
                               choices=["start", "stop", "status", "restart", "log"],
                               help="Server action (default: status)")
    parser_server.add_argument("--port", type=int, default=5050, help="Port to listen on")

    args = parser.parse_args()
    
    if args.verbose:
        VERBOSE = True

    init_fs()

    try:
        if args.command == "list":
            DaemonManager.start()
            ModelRegistry.sync()
            db = ModelRegistry.get_list()
            log("Model Registry:")
            for k, v in db.items():
                print(f"[{k}] {v['status'].upper()} - {v['path']}", file=sys.stderr)
                
        elif args.command == "config":
            if args.key.lower() == "show":
                cfg = ConfigStore.get_all(args.id)
                print(json.dumps(cfg, indent=4), file=sys.stderr)
            else:
                if args.value is None:
                    log_error("You must provide a value to set.")
                    sys.exit(1)
                ConfigStore.set(args.id, args.key, args.value)
                
        elif args.command == "infer":
            Modes.dispatch(args.mode, args.id, args.prompt, args.system)
            
        elif args.command == "clear":
            HistoryManager.clear()
            
        elif args.command == "server":
            if args.action == "start":
                ServerManager.start(args.port)
            elif args.action == "stop":
                ServerManager.stop()
            elif args.action == "restart":
                ServerManager.restart(args.port)
            elif args.action == "log":
                ServerManager.tail_log()
            else:  # status (default)
                ServerManager.status()
            
        else:
            parser.print_help(sys.stderr)
    except RuntimeError as e:
        log_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
