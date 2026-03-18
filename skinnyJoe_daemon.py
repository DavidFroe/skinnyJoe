#!/usr/bin/env python3
"""
SkinnyJoe Daemon v4.0 – Multi-Slot Model Server

Architektur:
  - Management-API auf Port 8000: Modelle auflisten, Slots verwalten
  - N Slot-Ports (z.B. 8001-8004): Jeweils ein OpenAI-kompatibler Endpoint
  - Jeder Slot kann EIN Modell laden
  - Model-Konfiguration (GPU, CTX, etc.) fest in config.json
  - Externe Steuerung: Nur WELCHES Modell auf WELCHEN Slot
  - Hardware-Konflikte werden automatisch erkannt (GPU-Überschneidung)

Modell-Typen:
  - text2text    .gguf (LLM)         → Chat/Completion
  - image2text   .gguf (Vision/VL)   → Bildbeschreibung
  - text2image   .safetensors (Flux) → Bildgenerierung
"""

import os
import re
import json
import gc
import time
import logging
import multiprocessing
import subprocess
import threading
import base64
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s')
logger = logging.getLogger("SkinnyJoe")

# ============================================================
# Konstanten
# ============================================================

MODEL_TYPE_TEXT2TEXT = "text2text"
MODEL_TYPE_TEXT2IMAGE = "text2image"
MODEL_TYPE_IMAGE2TEXT = "image2text"
MODEL_TYPE_SPEECH2TEXT = "speech2text"
MODEL_TYPE_TEXT2SPEECH = "text2speech"

# Typ-Kürzel für Model-IDs: T=Text2Text, D=Bild2Text, B=Bild, W=Whisper, S=Sprache
TYPE_PREFIXES = {
    "text2text":   "T",
    "image2text":  "D",
    "text2image":  "B",
    "speech2text": "W",
    "text2speech": "S",
}

KNOWN_TYPE_DIRS = {
    "text2text": MODEL_TYPE_TEXT2TEXT,
    "image2text": MODEL_TYPE_IMAGE2TEXT,
    "text2image": MODEL_TYPE_TEXT2IMAGE,
    "speech2text": MODEL_TYPE_SPEECH2TEXT,
    "text2speech": MODEL_TYPE_TEXT2SPEECH,
}

VISION_PATTERNS = [
    re.compile(r'llava', re.IGNORECASE),
    re.compile(r'[-_]vl[-_.]', re.IGNORECASE),
    re.compile(r'moondream', re.IGNORECASE),
    re.compile(r'yi[-_]vl', re.IGNORECASE),
    re.compile(r'cogvlm', re.IGNORECASE),
    re.compile(r'internvl', re.IGNORECASE),
]
MMPROJ_PATTERN = re.compile(r'mmproj', re.IGNORECASE)
QUANT_PATTERN = re.compile(
    r'(Q[0-9]+_K_[SML]|Q[0-9]+_[0-9]+|Q[0-9]+|[Ff]16|[Ff]32|[Bb][Ff]16|fp16|fp32)',
    re.IGNORECASE,
)
PARAMS_PATTERN = re.compile(r'(\d+\.?\d*)\s*[Bb](?:\b|[-_])')

TAG_KEYWORDS = {
    'code': ['code', 'coder', 'coding'],
    'instruct': ['instruct', 'instruction'],
    'uncensored': ['uncen', 'uncensored'],
    'chat': ['chat', 'assistant'],
    'roleplay': ['rp', 'roleplay', 'maid', 'horror', 'stheno'],
    'fast': ['flash', 'lite', 'light', 'tiny'],
    'MoE': ['moe', 'mixture'],
    'function': ['raven', 'hermes', 'function'],
}


# ============================================================
# Datenmodelle
# ============================================================

class GpuInfo(BaseModel):
    id: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float

class ModelInfo(BaseModel):
    id: str
    name: str
    full_name: str
    path: str
    format: str
    model_type: str
    size_gb: float
    params: Optional[str] = None
    quant: Optional[str] = None
    tags: List[str] = []
    mmproj_path: Optional[str] = None
    description: str = ""

class ChatMessage(BaseModel):
    role: str
    content: Any

class GenerateRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False

class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    size: Optional[str] = "1024x1024"
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None

class ExtLoadRequest(BaseModel):
    slot_id: int
    model_id: str
    ctx: Optional[int] = None  # override context size from profile

class ExtUnloadRequest(BaseModel):
    slot_id: int

class LoadProfileRequest(BaseModel):
    profile: str


# ============================================================
# GPU-Erkennung
# ============================================================

def detect_gpus() -> List[GpuInfo]:
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(GpuInfo(
                        id=int(parts[0]), name=parts[1],
                        vram_total_gb=round(float(parts[2]) / 1024, 1),
                        vram_free_gb=round(float(parts[3]) / 1024, 1),
                        vram_used_gb=round(float(parts[4]) / 1024, 1),
                    ))
    except Exception as e:
        logger.debug(f"GPU-Erkennung: {e}")
    return gpus


# ============================================================
# Model-Scanner
# ============================================================

def _is_mmproj(fn): return bool(MMPROJ_PATTERN.search(fn))
def _is_vision(fn): return any(p.search(fn) for p in VISION_PATTERNS)
def _extract_quant(s):
    m = QUANT_PATTERN.search(s)
    return m.group(1).upper() if m else None
def _extract_params(s):
    m = PARAMS_PATTERN.search(s)
    return m.group(1) + "B" if m else None

def _extract_tags(s):
    lo = s.lower()
    return [t for t, kws in TAG_KEYWORDS.items() if any(k in lo for k in kws)]

def _clean_name(fn, mx=35):
    n = Path(fn).stem
    n = QUANT_PATTERN.sub('', n)
    n = re.sub(r'\(\d+\)', '', n)
    n = re.sub(r'[-_]{2,}', '-', n).strip('-_ ')
    return n[:mx-2] + ".." if len(n) > mx else n

def _make_desc(mt, tags, params):
    p = []
    if mt == MODEL_TYPE_TEXT2TEXT:
        p.append("Code-LLM" if 'code' in tags else "Roleplay-LLM" if 'roleplay' in tags
                 else "Function-Calling" if 'function' in tags else "Text-LLM")
    elif mt == MODEL_TYPE_IMAGE2TEXT: p.append("Vision (Bild→Text)")
    elif mt == MODEL_TYPE_TEXT2IMAGE: p.append("Diffusion (Text→Bild)")
    if params: p.append(params)
    for t in ['instruct', 'uncensored', 'MoE', 'fast']:
        if t in tags: p.append(t.capitalize() if t != 'MoE' else t)
    return " · ".join(p)

def _pair_mmproj(stem, mmproj_files):
    if not mmproj_files: return None
    mc = QUANT_PATTERN.sub('', stem.lower())
    mc = re.sub(r'[-_](text[-_]model|instruct|chat)[-_]?', '-', mc, flags=re.I).strip('-_ ')
    best, best_s = None, 0
    for mp in mmproj_files:
        c = mp.stem.lower()
        c = re.sub(r'[-_]?mmproj[-_]?(model[-_]?)?', '', c, flags=re.I)
        c = re.sub(r'^mmproj[-_]?', '', c, flags=re.I)
        c = QUANT_PATTERN.sub('', c).strip('-_ ')
        s = sum(1 for a, b in zip(mc, c) if a == b)
        if s > best_s: best, best_s = mp, s
    return str(best) if best and best_s >= 4 else None

def _collect_model_files(base_dir: Path):
    """Sammelt Modelldateien aus Top-Level und Typ-Unterverzeichnissen.
    Gibt Liste von (path, forced_type_or_None) zurück."""
    entries = []
    for e in sorted(base_dir.iterdir()):
        if e.is_file():
            entries.append((e, None))
        elif e.is_dir():
            if e.name in KNOWN_TYPE_DIRS:
                forced = KNOWN_TYPE_DIRS[e.name]
                for f in sorted(e.iterdir()):
                    if f.is_file():
                        entries.append((f, forced))
                    elif f.is_dir() and (
                        (f / "model_index.json").exists()   # diffusers
                        or (f / "config.json").exists()     # TTS / HF models
                        or forced == MODEL_TYPE_TEXT2SPEECH  # always include text2speech dirs
                    ):
                        entries.append((f, forced))
            elif (e / "model_index.json").exists():
                entries.append((e, None))
    return entries


def scan_models(models_dir, speech_backends=None):
    mp = Path(models_dir)
    if not mp.exists(): return []

    entries = _collect_model_files(mp)

    # Separate mmproj files from model files
    mmproj_files = []
    model_entries = []
    for e, forced_type in entries:
        if e.is_file() and e.suffix.lower() == '.gguf' and _is_mmproj(e.name):
            mmproj_files.append(e)
        else:
            model_entries.append((e, forced_type))

    models, idx = [], 1
    for e, forced_type in model_entries:
        if e.is_dir():
            sz = round(sum(f.stat().st_size for f in e.rglob("*") if f.is_file()) / (1024**3), 2)
            mt = forced_type or MODEL_TYPE_TEXT2IMAGE
            fmt = "tts" if mt == MODEL_TYPE_TEXT2SPEECH else "diffusers"
            models.append(ModelInfo(id=str(idx), name=e.name[:35], full_name=e.name, path=str(e),
                format=fmt, model_type=mt, size_gb=sz,
                description=_make_desc(mt, [], None)))
            idx += 1; continue

        ext = e.suffix.lower()
        if ext not in ('.gguf', '.safetensors', '.bin', '.pt'):
            continue

        sz = round(e.stat().st_size / (1024**3), 2)
        stem = e.stem
        tags = _extract_tags(stem)

        # Typ-Erkennung: Unterverzeichnis überschreibt Auto-Detection
        if forced_type:
            mt = forced_type
            fmt = "gguf" if ext == '.gguf' else ext.lstrip('.')
        elif ext == '.safetensors':
            mt, fmt = MODEL_TYPE_TEXT2IMAGE, "safetensors"
        elif _is_vision(stem):
            mt, fmt = MODEL_TYPE_IMAGE2TEXT, "gguf"
            if "vision" not in tags: tags.append("vision")
        else:
            mt, fmt = MODEL_TYPE_TEXT2TEXT, "gguf"

        mmp = _pair_mmproj(stem, mmproj_files) if mt == MODEL_TYPE_IMAGE2TEXT else None
        models.append(ModelInfo(id=str(idx), name=_clean_name(e.name), full_name=stem,
            path=str(e), format=fmt, model_type=mt, size_gb=sz,
            params=_extract_params(stem), quant=_extract_quant(stem),
            tags=tags, mmproj_path=mmp, description=_make_desc(mt, tags, _extract_params(stem))))
        idx += 1

    # Virtual entries from speech_backends config
    for backend_name, bconf in (speech_backends or {}).items():
        engine = bconf.get("engine", "openai-whisper")
        path = bconf.get("path", "")
        sz = round(Path(path).stat().st_size / (1024**3), 2) if path and Path(path).exists() else 0.0
        info_str = bconf.get("_info", f"{engine} STT backend")
        models.append(ModelInfo(id=str(idx), name=backend_name[:35], full_name=backend_name,
            path=path, format=engine, model_type=MODEL_TYPE_SPEECH2TEXT, size_gb=sz,
            tags=["stt", engine], description=info_str))
        idx += 1

    return models


# ============================================================
# Hardware-Manager
# ============================================================

class HardwareManager:
    def __init__(self):
        self.gpu_alloc: Dict[int, int] = {}   # gpu_id → slot_id

    def check_conflict(self, gpu_ids: List[int], exclude_slot: int = None) -> List[str]:
        conflicts = []
        for gid in gpu_ids:
            if gid in self.gpu_alloc:
                occ = self.gpu_alloc[gid]
                if exclude_slot is not None and occ == exclude_slot:
                    continue
                conflicts.append(f"GPU G{gid} belegt von Slot {occ}")
        return conflicts

    def allocate(self, slot_id: int, gpu_ids: List[int]):
        for gid in gpu_ids:
            self.gpu_alloc[gid] = slot_id

    def release(self, slot_id: int):
        self.gpu_alloc = {g: s for g, s in self.gpu_alloc.items() if s != slot_id}


# ============================================================
# Slot – ein Port mit einem Modell
# ============================================================

class Slot:
    def __init__(self, slot_id: int, port: int):
        self.id = slot_id
        self.port = port
        self.model: Any = None
        self.model_info: Optional[ModelInfo] = None
        self.profile: Dict[str, Any] = {}
        self.is_generating = False

    def load(self, info: ModelInfo, profile: dict):
        self.unload()
        if info.model_type == MODEL_TYPE_IMAGE2TEXT:
            self._load_vision(info, profile)
        elif info.model_type == MODEL_TYPE_SPEECH2TEXT:
            self._load_whisper(info, profile)
        elif info.model_type == MODEL_TYPE_TEXT2SPEECH:
            self._load_tts(info, profile)
        elif info.format == "gguf":
            self._load_gguf(info, profile)
        elif info.format in ("safetensors", "diffusers"):
            self._load_diffusers(info, profile)
        else:
            raise ValueError(f"Unbekanntes Format: {info.format}")
        self.model_info = info
        self.profile = profile
        logger.info(f"Slot {self.id}: '{info.name}' geladen "
                     f"(gpu_layers={profile.get('gpu_layers',0)}, gpu_ids={profile.get('gpu_ids',[])})")

    def unload(self):
        if self.model is not None:
            name = self.model_info.name if self.model_info else "?"
            logger.info(f"Slot {self.id}: '{name}' entladen")
            del self.model
            self.model = None
            self.model_info = None
            self.profile = {}
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except ImportError: pass

    def _load_gguf(self, info, profile):
        from llama_cpp import Llama
        cpu_count = multiprocessing.cpu_count()
        gpu_ids = profile.get("gpu_ids", [])
        kwargs = dict(
            model_path=info.path,
            n_ctx=profile.get("ctx", 4096),
            n_gpu_layers=profile.get("gpu_layers", 0),
            n_threads=profile.get("n_threads", min(cpu_count, 8)),
            n_threads_batch=profile.get("n_threads_batch", cpu_count),
            n_batch=profile.get("n_batch", 512),
            flash_attn=profile.get("flash_attn", False),
            main_gpu=gpu_ids[0] if gpu_ids else 0,
            verbose=True,
        )
        if gpu_ids and len(gpu_ids) > 1:
            mx = max(gpu_ids) + 1
            ts = [0.0] * mx
            p = 1.0 / len(gpu_ids)
            for g in gpu_ids: ts[g] = p
            kwargs["tensor_split"] = ts
        self.model = Llama(**kwargs)

    def _load_vision(self, info, profile):
        from llama_cpp import Llama
        if not info.mmproj_path:
            raise ValueError(f"Kein mmproj für Vision-Modell '{info.name}'")
        try:
            from llama_cpp.llama_chat_format import Llava16ChatHandler
            handler = Llava16ChatHandler(clip_model_path=info.mmproj_path, verbose=False)
        except Exception:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            handler = Llava15ChatHandler(clip_model_path=info.mmproj_path, verbose=False)
        gpu_ids = profile.get("gpu_ids", [])
        self.model = Llama(
            model_path=info.path, chat_handler=handler,
            n_ctx=profile.get("ctx", 4096),
            n_gpu_layers=profile.get("gpu_layers", 0),
            n_threads=profile.get("n_threads", min(multiprocessing.cpu_count(), 8)),
            n_batch=profile.get("n_batch", 512),
            main_gpu=gpu_ids[0] if gpu_ids else 0,
            verbose=True,
        )

    def _load_diffusers(self, info, profile):
        import torch
        from diffusers import FluxPipeline, StableDiffusionPipeline
        dtype = torch.bfloat16
        gpu_ids = profile.get("gpu_ids", [])
        device = f"cuda:{gpu_ids[0]}" if gpu_ids and torch.cuda.is_available() else "cpu"
        local_kw = {"local_files_only": True} if info.format == "diffusers" else {}
        try:
            loader = FluxPipeline.from_pretrained if info.format == "diffusers" else FluxPipeline.from_single_file
            pipe = loader(info.path, torch_dtype=dtype, **local_kw)
            gid = int(device.split(":")[1]) if ":" in device else 0
            pipe.enable_model_cpu_offload(gpu_id=gid if device.startswith("cuda") else 0)
            self.model = pipe
        except Exception:
            loader = StableDiffusionPipeline.from_pretrained if info.format == "diffusers" else StableDiffusionPipeline.from_single_file
            pipe = loader(info.path, torch_dtype=dtype, **local_kw)
            pipe.to(device)
            self.model = pipe

    def _load_whisper(self, info, profile):
        """Whisper speech-to-text Modell laden (openai-whisper)."""
        import whisper as oai_whisper
        gpu_ids = profile.get("gpu_ids", [])
        device_cfg = profile.get("device", "auto")
        if device_cfg == "auto":
            device = "cuda" if gpu_ids else "cpu"
        else:
            device = device_cfg
        self.model = oai_whisper.load_model(info.path, device=device)

    def _load_tts(self, info, profile):
        """Text-to-Speech Modell laden – Platzhalter, Inferenz noch nicht implementiert."""
        logger.info(f"TTS-Modell '{info.name}' erkannt ({info.full_name}). Inferenz-Backend ausstehend.")
        # Modell-Pfad merken; eigentliche Initialisierung folgt wenn Backend gewählt ist
        self.model = {"__tts_stub__": True, "name": info.full_name, "path": info.path}

    def generate_text(self, request: GenerateRequest, defaults: dict):
        if not self.model:
            raise HTTPException(400, f"Slot {self.id}: Kein Modell geladen.")
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        temp = request.temperature if request.temperature is not None else defaults.get("temperature", 0.8)
        max_t = request.max_tokens if request.max_tokens is not None else defaults.get("max_tokens", 2048)
        top_p = request.top_p if request.top_p is not None else defaults.get("top_p", 0.95)
        top_k = request.top_k if request.top_k is not None else defaults.get("top_k", 40)
        rep_p = request.repeat_penalty if request.repeat_penalty is not None else defaults.get("repeat_penalty", 1.1)
        self.is_generating = True
        try:
            if request.stream:
                it = self.model.create_chat_completion(
                    messages=msgs, temperature=temp, top_p=top_p, top_k=top_k,
                    repeat_penalty=rep_p, presence_penalty=request.presence_penalty,
                    max_tokens=max_t, stream=True)
                def gen():
                    try:
                        for chunk in it:
                            yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        self.is_generating = False
                return StreamingResponse(gen(), media_type="text/event-stream")
            else:
                try:
                    return self.model.create_chat_completion(
                        messages=msgs, temperature=temp, top_p=top_p, top_k=top_k,
                        repeat_penalty=rep_p, presence_penalty=request.presence_penalty,
                        max_tokens=max_t, stream=False)
                finally:
                    self.is_generating = False
        except Exception as e:
            self.is_generating = False
            raise HTTPException(500, str(e))

    def generate_image(self, request: ImageRequest, img_defaults: dict):
        if not self.model:
            raise HTTPException(400, f"Slot {self.id}: Kein Modell geladen.")
        w = request.width or img_defaults.get("width", 1024)
        h = request.height or img_defaults.get("height", 1024)
        steps = request.num_inference_steps or img_defaults.get("num_inference_steps", 20)
        guidance = request.guidance_scale or img_defaults.get("guidance_scale", 3.5)
        if request.size and not request.width:
            try:
                parts = request.size.split("x")
                w, h = int(parts[0]), int(parts[1])
            except (ValueError, IndexError): pass
        self.is_generating = True
        try:
            result = self.model(prompt=request.prompt, width=w, height=h,
                                num_inference_steps=steps, guidance_scale=guidance)
            from io import BytesIO
            buf = BytesIO()
            result.images[0].save(buf, "JPEG", quality=95)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"created": int(time.time()),
                    "data": [{"b64_json": b64, "revised_prompt": request.prompt}]}
        except Exception as e:
            raise HTTPException(500, str(e))
        finally:
            self.is_generating = False


def _stable_model_ids(models: list, ids_file: str) -> list:
    """Assign stable typed IDs (T1, D2, W3, B1, S1).
    Prefix encodes model type; number is per-type sequential.
    Once assigned, IDs never change. Stored in model_ids.json."""
    try:
        with open(ids_file) as f:
            mapping: Dict[str, str] = json.load(f)
    except Exception:
        mapping = {}

    # Höchste vergebene Nummer pro Prefix bestimmen
    next_n: Dict[str, int] = {}
    for val in mapping.values():
        if isinstance(val, str) and len(val) >= 2:
            prefix, num = val[0], val[1:]
            try:
                next_n[prefix] = max(next_n.get(prefix, 0), int(num))
            except ValueError:
                pass

    dirty = False
    result = []

    for m in models:
        key = f"{m.model_type}:{m.full_name}"
        if key not in mapping:
            prefix = TYPE_PREFIXES.get(m.model_type, "X")
            n = next_n.get(prefix, 0) + 1
            next_n[prefix] = n
            mapping[key] = f"{prefix}{n}"
            dirty = True
        new_m = ModelInfo(
            id=mapping[key], name=m.name, full_name=m.full_name,
            path=m.path, format=m.format, model_type=m.model_type, size_gb=m.size_gb,
            params=m.params, quant=m.quant, tags=list(m.tags),
            mmproj_path=m.mmproj_path, description=m.description,
        )
        result.append(new_m)

    if dirty:
        try:
            with open(ids_file, "w") as f:
                json.dump(mapping, f, indent=2)
        except Exception as e:
            logger.warning(f"model_ids.json konnte nicht geschrieben werden: {e}")

    # Sortieren: nach Präfix-Reihenfolge, dann Nummer
    prefix_order = {p: i for i, p in enumerate(["T", "D", "B", "W", "S"])}
    def _sort_key(m):
        p, n = m.id[0], m.id[1:]
        return (prefix_order.get(p, 99), int(n) if n.isdigit() else 0)
    return sorted(result, key=_sort_key)


# ============================================================
# Slot-Manager
# ============================================================

class SlotManager:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)

        self._ids_file = os.path.join(os.path.dirname(os.path.abspath(config_path)), "model_ids.json")

        self.management_port = self.config.get("management_port", 8000)
        self.defaults = {"temperature": 0.8, "max_tokens": 2048, "top_p": 0.95, "top_k": 40,
                         "ctx": 4096, "gpu_layers": 0, "n_batch": 512, "flash_attn": False}
        self.defaults.update(self.config.get("defaults", {}))
        self.image_defaults = self.config.get("image_defaults",
            {"width": 1024, "height": 1024, "num_inference_steps": 20, "guidance_scale": 3.5})
        self.profiles: Dict[str, dict] = self.config.get("model_profiles", {})
        self.named_profiles: Dict[str, dict] = self.config.get("profiles", {})

        self.slots: Dict[int, Slot] = {}
        for s in self.config.get("slots", [{"id": 1, "port": 8001}]):
            self.slots[s["id"]] = Slot(s["id"], s["port"])

        models_dir = self.config.get("models_dir", os.path.join(os.path.dirname(config_path), "models"))
        self.speech_backends = self.config.get("speech_backends", {})
        self.models = _stable_model_ids(scan_models(models_dir, self.speech_backends), self._ids_file)
        self.gpus = detect_gpus()
        self.hardware = HardwareManager()
        self._lock = threading.Lock()

        logger.info(f"SlotManager: {len(self.models)} Modelle, {len(self.slots)} Slots, {len(self.gpus)} GPUs")
        for m in self.models:
            logger.info(f"  {m.id} {m.name} ({m.model_type}, {m.size_gb}GB)")

    def get_profile(self, full_name: str) -> dict:
        base = dict(self.defaults)
        if full_name in self.profiles:
            base.update(self.profiles[full_name])
        return base

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        return next((m for m in self.models if m.id == model_id.upper()), None)

    def refresh_gpus(self):
        self.gpus = detect_gpus()

    def rescan(self):
        models_dir = self.config.get("models_dir", "models")
        self.models = _stable_model_ids(scan_models(models_dir, self.speech_backends), self._ids_file)

    def load_model(self, slot_id: int, model_id: str, ctx_override: Optional[int] = None):
        with self._lock:
            if slot_id not in self.slots:
                raise HTTPException(404, f"Slot {slot_id} existiert nicht. Verfügbar: {list(self.slots.keys())}")
            slot = self.slots[slot_id]
            if slot.is_generating:
                raise HTTPException(409, f"Slot {slot_id} generiert gerade.")

            model = self.get_model(model_id)
            if not model:
                raise HTTPException(404, f"Modell {model_id.upper()} nicht gefunden.")

            profile = self.get_profile(model.full_name)
            if ctx_override:
                profile["ctx"] = ctx_override
            gpu_ids = profile.get("gpu_ids", [])

            conflicts = self.hardware.check_conflict(gpu_ids, exclude_slot=slot_id)
            if conflicts:
                raise HTTPException(409, f"Hardware-Konflikt: {'; '.join(conflicts)}")

            self.hardware.release(slot_id)
            try:
                slot.load(model, profile)
            except Exception as e:
                logger.error(f"Slot {slot_id}: Laden fehlgeschlagen: {e}")
                raise HTTPException(500, f"Laden fehlgeschlagen: {e}")
            self.hardware.allocate(slot_id, gpu_ids)

            return {
                "status": "loaded", "slot_id": slot_id,
                "model": {"id": model.id, "name": model.name, "model_type": model.model_type,
                          "size_gb": model.size_gb},
                "profile": {k: v for k, v in profile.items() if k in
                            ("gpu_layers", "gpu_ids", "ctx", "n_batch")},
            }

    def unload_slot(self, slot_id: int):
        with self._lock:
            if slot_id not in self.slots:
                raise HTTPException(404, f"Slot {slot_id} existiert nicht.")
            slot = self.slots[slot_id]
            if slot.is_generating:
                raise HTTPException(409, f"Slot {slot_id} generiert gerade.")
            slot.unload()
            self.hardware.release(slot_id)
            return {"status": "unloaded", "slot_id": slot_id}

    def get_status(self):
        slots_info = []
        for sid, slot in sorted(self.slots.items()):
            loaded = None
            if slot.model_info:
                mi = slot.model_info
                loaded = {"id": mi.id, "name": mi.name, "model_type": mi.model_type,
                          "size_gb": mi.size_gb, "profile": slot.profile}
            slots_info.append({
                "id": slot.id, "port": slot.port,
                "status": "generating" if slot.is_generating else ("loaded" if slot.model else "idle"),
                "loaded_model": loaded,
            })
        return {
            "management_port": self.management_port,
            "slots": slots_info,
            "gpus": [g.model_dump() for g in self.gpus],
            "models_count": len(self.models),
            "hardware_alloc": {f"G{g}": f"Slot {s}" for g, s in self.hardware.gpu_alloc.items()},
        }


# ============================================================
# Management-API (Port 8000)
# ============================================================

def create_management_app(sm: SlotManager) -> FastAPI:
    app = FastAPI(title="SkinnyJoe Management v4.0")

    @app.get("/v1/models")
    def list_models():
        result = []
        for m in sm.models:
            profile = sm.get_profile(m.full_name)
            result.append({
                "id": m.id, "name": m.name, "full_name": m.full_name,
                "model_type": m.model_type, "format": m.format, "size_gb": m.size_gb,
                "params": m.params, "quant": m.quant, "tags": m.tags,
                "description": m.description, "has_mmproj": bool(m.mmproj_path),
                "profile": {k: v for k, v in profile.items() if k in
                            ("gpu_layers", "gpu_ids", "ctx", "n_batch", "n_threads", "flash_attn")},
                "object": "model", "owned_by": "SkinnyJoe",
            })
        return {"data": result}

    @app.get("/v1/slots")
    def list_slots():
        slots_info = []
        for sid, slot in sorted(sm.slots.items()):
            loaded = None
            if slot.model_info:
                mi = slot.model_info
                loaded = {"id": mi.id, "name": mi.name, "model_type": mi.model_type,
                          "size_gb": mi.size_gb,
                          "ctx": slot.profile.get("ctx", sm.defaults.get("ctx", 4096))}
            slots_info.append({
                "id": slot.id, "port": slot.port,
                "status": "generating" if slot.is_generating else ("loaded" if slot.model else "idle"),
                "loaded_model": loaded,
            })
        return {"slots": slots_info}

    @app.get("/v1/gpus")
    def list_gpus():
        sm.refresh_gpus()
        return {"gpus": [g.model_dump() for g in sm.gpus], "count": len(sm.gpus)}

    @app.post("/v1/load")
    def load_model(request: ExtLoadRequest):
        return sm.load_model(request.slot_id, request.model_id, ctx_override=request.ctx)

    @app.post("/v1/unload")
    def unload_model(request: ExtUnloadRequest):
        return sm.unload_slot(request.slot_id)

    @app.post("/v1/rescan")
    def rescan():
        sm.rescan()
        return {"status": "rescanned", "count": len(sm.models)}

    @app.get("/v1/profiles")
    def list_profiles():
        result = []
        for name, p in sm.named_profiles.items():
            # Resolve model_name → model_id if possible
            model_id = None
            model_found = None
            model_name = p.get("model_name", "")
            for m in sm.models:
                if m.full_name == model_name or m.name == model_name:
                    model_id = m.id
                    model_found = m.name
                    break
            result.append({
                "name": name,
                "slot_id": p.get("slot_id"),
                "model_name": model_name,
                "model_id": model_id,
                "model_found": model_found,
                "description": p.get("description", ""),
            })
        return {"profiles": result}

    @app.post("/v1/load-profile")
    def load_profile(body: LoadProfileRequest):
        profile_name = body.profile
        if not profile_name:
            raise HTTPException(400, "Pflichtfeld: 'profile'")
        p = sm.named_profiles.get(profile_name)
        if not p:
            raise HTTPException(404, f"Profil '{profile_name}' nicht gefunden. Verfügbar: {list(sm.named_profiles.keys())}")
        slot_id = p.get("slot_id")
        model_name = p.get("model_name", "")
        model_id = None
        for m in sm.models:
            if m.full_name == model_name or m.name == model_name:
                model_id = m.id
                break
        if model_id is None:
            raise HTTPException(404, f"Modell '{model_name}' aus Profil '{profile_name}' nicht im models_dir gefunden.")
        return sm.load_model(slot_id, model_id)

    @app.get("/status")
    def status():
        return sm.get_status()

    return app


# ============================================================
# Slot-API (Port 800X) – OpenAI-kompatibel
# ============================================================

def create_slot_app(slot: Slot, sm: SlotManager) -> FastAPI:
    app = FastAPI(title=f"SkinnyJoe Slot {slot.id} (Port {slot.port})")

    @app.post("/v1/chat/completions")
    def chat(request: GenerateRequest):
        if not slot.model or not slot.model_info:
            raise HTTPException(400, f"Slot {slot.id}: Kein Modell geladen.")
        if slot.model_info.model_type not in (MODEL_TYPE_TEXT2TEXT, MODEL_TYPE_IMAGE2TEXT):
            raise HTTPException(400, f"Slot {slot.id}: Modell ist {slot.model_info.model_type}, nicht text2text/image2text.")
        return slot.generate_text(request, sm.defaults)

    @app.post("/v1/images/generations")
    def images(request: ImageRequest):
        if not slot.model or not slot.model_info:
            raise HTTPException(400, f"Slot {slot.id}: Kein Modell geladen.")
        if slot.model_info.model_type != MODEL_TYPE_TEXT2IMAGE:
            raise HTTPException(400, f"Slot {slot.id}: Modell ist {slot.model_info.model_type}, nicht text2image.")
        return slot.generate_image(request, sm.image_defaults)

    @app.post("/v1/audio/transcriptions")
    async def transcribe(request: Request):
        """OpenAI-kompatible Whisper Transkription."""
        if not slot.model or not slot.model_info:
            raise HTTPException(400, f"Slot {slot.id}: Kein Modell geladen.")
        if slot.model_info.model_type != MODEL_TYPE_SPEECH2TEXT:
            raise HTTPException(400, f"Slot {slot.id}: Modell ist {slot.model_info.model_type}, nicht speech2text.")

        form = await request.form()
        audio_file = form.get("file")
        if not audio_file:
            raise HTTPException(400, "Kein 'file' im Request.")
        language = form.get("language")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name
        slot.is_generating = True
        try:
            result = slot.model.transcribe(tmp_path, language=language or None, fp16=False)
            text = result["text"].strip()
            seg_list = [{"start": round(s["start"], 2), "end": round(s["end"], 2),
                         "text": s["text"].strip()}
                        for s in result.get("segments", [])]
        finally:
            slot.is_generating = False
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return {"text": text, "language": result.get("language", ""), "segments": seg_list}

    @app.get("/v1/models")
    def models():
        if slot.model_info:
            mi = slot.model_info
            return {"data": [{"id": mi.id, "name": mi.name, "model_type": mi.model_type,
                              "loaded": True, "object": "model", "owned_by": "SkinnyJoe"}]}
        return {"data": []}

    @app.get("/status")
    def status():
        loaded = None
        if slot.model_info:
            mi = slot.model_info
            loaded = {"id": mi.id, "name": mi.name, "model_type": mi.model_type,
                      "size_gb": mi.size_gb}
        return {"slot_id": slot.id, "port": slot.port,
                "loaded_model": loaded, "is_generating": slot.is_generating}

    return app


# ============================================================
# Server-Start
# ============================================================

def run_server(app, port):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main():
    import uvicorn
    config_path = "/home/david/skinnyjoe/config.json"
    sm = SlotManager(config_path)

    # Slot-Server in Daemon-Threads starten
    for slot in sm.slots.values():
        app = create_slot_app(slot, sm)
        t = threading.Thread(target=run_server, args=(app, slot.port),
                             name=f"slot-{slot.id}", daemon=True)
        t.start()
        logger.info(f"Slot {slot.id} → Port {slot.port}")

    # Management-Server im Hauptthread (Signal-Handling)
    logger.info(f"Management-API → Port {sm.management_port}")
    logger.info(f"Bereit. {len(sm.slots)} Slots, {len(sm.models)} Modelle.")
    mgmt_app = create_management_app(sm)
    uvicorn.run(mgmt_app, host="0.0.0.0", port=sm.management_port)


if __name__ == "__main__":
    main()
