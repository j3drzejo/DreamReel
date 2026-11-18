"""Central configuration and small shared helpers for the sleep-documentary generator.

IMPORTANT: this module must be imported before torch is imported anywhere, because it
sets environment variables (MPS CPU-fallback etc.) that torch reads at import time.
"""
from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported MPS ops fall back to CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

import warnings  # noqa: E402

for _cat in (FutureWarning, UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=_cat)

# --------------------------------------------------------------------------- video
VIDEO_W, VIDEO_H, FPS = 1920, 1080, 24
X264_PRESET = "medium"
X264_CRF = 20
AUDIO_BITRATE = "192k"
AUDIO_SR = 44100
CROSSFADE_SECONDS = 1.0
VIGNETTE_STRENGTH = 0.55
SECONDS_PER_IMAGE = 24.0          # one new AI image about every 24 s (generation is the slow part)
SHOT_MIN_SECONDS = 4.0            # each image is cut into shots of 4-8 s with different framing/camera moves
SHOT_MAX_SECONDS = 8.0
END_FADE_SECONDS = 4.0            # fade to black at the very end
MIN_TOTAL_SECONDS = 605.0         # hard guarantee: output is always 10+ minutes

# --------------------------------------------------------------------------- narration
NUM_CHAPTERS = 10                 # 8-12 allowed
WORDS_PER_MIN_TARGET = 160        # conservative: Kokoro @0.92 speed really runs ~135-145 wpm
DEFAULT_DURATION_MIN = 12.0
KOKORO_VOICE = "am_michael"       # alternatives: am_fenrir, bm_george
KOKORO_SPEED = 0.92
KOKORO_SR = 24000
SENTENCE_GROUP_PAUSE = 0.35       # inside a paragraph, between synthesized chunks
PARAGRAPH_PAUSE = 0.8
CHAPTER_PAUSE = 2.0
LEAD_IN_SECONDS = 2.0
OUTRO_SECONDS = 5.0

# --------------------------------------------------------------------------- mastering
VOICE_LUFS = -16.0
VOICE_PEAK_DB = -3.0
FIRE_IDLE_DB = 20.0               # fireplace sits this many dB below the voice while the narrator is silent
FIRE_ACTIVE_EXTRA_DB = 4.0        # ...and this much further down while the narrator is speaking (=-28 dB)

# --------------------------------------------------------------------------- models
LLM_LARGE = "Qwen/Qwen2.5-7B-Instruct"
LLM_SMALL = "Qwen/Qwen3-4B-Instruct-2507"   # Apache-2.0, fits comfortably in 8-10 GB
IMAGE_TIERS = [
    "Tongyi-MAI/Z-Image-Turbo",
    "black-forest-labs/FLUX.2-klein-4B",
    "stabilityai/sdxl-turbo",
]

BUILD_CACHE = ".build_cache"


@dataclass
class Settings:
    topic: Optional[str] = None
    duration_min: float = DEFAULT_DURATION_MIN
    output_dir: Path = Path(".")
    llm_model: str = "auto"
    image_model: str = "auto"
    voice: str = KOKORO_VOICE
    speed: float = KOKORO_SPEED
    seed: int = 1234
    keep_cache: bool = False
    fresh: bool = False
    fire_db: float = FIRE_IDLE_DB
    image_interval: float = SECONDS_PER_IMAGE

    @property
    def target_words(self) -> int:
        return int(max(self.duration_min, 10.0) * WORDS_PER_MIN_TARGET)


# --------------------------------------------------------------------------- logging / io
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sanitize_folder_name(name: str, max_len: int = 110) -> str:
    name = name.replace("|", "-")
    name = re.sub(r'[<>:"/\\?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    if len(name) > max_len:
        name = name[:max_len].rsplit(" ", 1)[0].strip()
    return name or "Untitled Documentary"


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s" if h else f"{m}m {sec}s"


def fmt_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# --------------------------------------------------------------------------- hardware
def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def total_ram_gb() -> float:
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1e9
        except Exception:
            pass
    return 16.0


def usable_memory_gb(device: str) -> float:
    """Memory (GB) a model can realistically use on this device."""
    if device == "cuda":
        import torch

        return torch.cuda.get_device_properties(0).total_memory / 1e9
    if device == "mps":
        return total_ram_gb() * 0.62   # Metal's recommended working-set limit
    return total_ram_gb() * 0.5


def pick_llm(cfg: Settings, device: str) -> str:
    if cfg.llm_model != "auto":
        return cfg.llm_model
    return LLM_LARGE if usable_memory_gb(device) >= 16.5 else LLM_SMALL


def free_memory() -> None:
    """Release Python garbage and accelerator caches. Call after `del model`."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


# --------------------------------------------------------------------------- ffmpeg
def find_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_exe()
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg not found. Install it: macOS `brew install ffmpeg`, Ubuntu/Debian "
        "`sudo apt install ffmpeg`, Windows `winget install Gyan.FFmpeg` (then reopen the terminal)."
    )


def probe_duration(path: Path) -> float:
    r = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not m:
        raise RuntimeError(f"Could not read duration of {path}")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
