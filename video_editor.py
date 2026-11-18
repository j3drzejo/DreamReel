"""1080p / 24 fps / H.264 / AAC rendering with Ken Burns motion, crossfades, vignette and no on-screen text.

Every frame is computed with Pillow using *fractional* crop boxes (sub-pixel accurate bicubic resize),
so the slow zooms and pans are perfectly smooth - no integer-rounding jitter like FFmpeg's zoompan.
Frames are streamed straight into ffmpeg's stdin; no temporary frame files are written.
"""
from __future__ import annotations

import bisect
import math
import os
import random
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

import config
from config import (AUDIO_BITRATE, CROSSFADE_SECONDS, END_FADE_SECONDS, FPS, VIDEO_H, VIDEO_W,
                    VIGNETTE_STRENGTH, X264_CRF, X264_PRESET, log)
from image_gen import build_image_plan  # noqa: F401  (re-exported for callers)

W, H = VIDEO_W, VIDEO_H
X = CROSSFADE_SECONDS


def _smooth(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


# --------------------------------------------------------------------------- overlays
def build_vignette(strength: float = VIGNETTE_STRENGTH) -> Image.Image:
    """Warm, soft vignette as a multiply layer (white = unchanged)."""
    ys = (np.arange(H) - H / 2 + 0.5) / (H / 2)
    xs = (np.arange(W) - W / 2 + 0.5) / (W / 2)
    r = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / math.sqrt(2)
    v = np.clip((r - 0.30) / 0.70, 0, 1)
    v = v * v * (3 - 2 * v)
    bright = 1.0 - strength * v
    center, edge = np.array([1.0, 0.975, 0.92]), np.array([1.0, 0.90, 0.76])
    tint = center[None, None, :] * (1 - v[..., None]) + edge[None, None, :] * v[..., None]
    return Image.fromarray(np.clip(bright[..., None] * tint * 255, 0, 255).astype(np.uint8), "RGB")


def _font(size: int) -> ImageFont.FreeTypeFont:
    for c in ("/System/Library/Fonts/Supplemental/Georgia.ttf", "/Library/Fonts/Georgia.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
              "C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/times.ttf", "DejaVuSerif.ttf", "Georgia.ttf"):
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------- motion
def _motion(idx: int, seed: int) -> dict:
    rng = random.Random(seed * 7919 + idx)
    mode = ["zoom_in", "pan_right", "zoom_out", "pan_left", "zoom_in_drift", "pan_right_zoom"][(idx + rng.randrange(0, 2)) % 6]
    zmax = 1.0 + rng.uniform(0.09, 0.14)
    jitter = lambda: rng.uniform(-0.04, 0.04)
    if mode == "zoom_in":
        return dict(z0=1.0, z1=zmax, c0=(0.5, 0.5), c1=(0.5 + jitter(), 0.5 + jitter()))
    if mode == "zoom_out":
        return dict(z0=zmax, z1=1.0, c0=(0.5 + jitter(), 0.5 + jitter()), c1=(0.5, 0.5))
    if mode == "zoom_in_drift":
        return dict(z0=1.0, z1=zmax, c0=(0.5, 0.5), c1=(0.5 + rng.choice([-1, 1]) * 0.12, 0.5 + jitter()))
    z = 1.12
    if mode == "pan_right":
        return dict(z0=z, z1=z, c0=(0.0, 0.5 + jitter()), c1=(1.0, 0.5 + jitter()))
    if mode == "pan_left":
        return dict(z0=z, z1=z, c0=(1.0, 0.5 + jitter()), c1=(0.0, 0.5 + jitter()))
    return dict(z0=1.05, z1=1.13, c0=(0.0, 0.5), c1=(1.0, 0.5))  # pan_right_zoom


class Renderer:
    def __init__(self, plan: List[Dict], timeline: Dict, script: Dict, seed: int = 0):
        self.plan, self.total = plan, timeline["total_duration"]
        self.starts = [s["start"] for s in plan]
        self.n = len(plan)
        self.motion = [_motion(i, seed) for i in range(self.n)]
        self.vignette = build_vignette()
        self.black = Image.new("RGB", (W, H), (0, 0, 0))
        self._cache: Dict[int, Image.Image] = {}
        self._lock = threading.Lock()

    # -- source images (small LRU: only neighbouring segments are ever needed)
    def _src(self, j: int) -> Image.Image:
        with self._lock:
            img = self._cache.get(j)
            if img is None:
                img = Image.open(self.plan[j]["path"]).convert("RGB")
                if img.size != (W, H):
                    img = img.resize((W, H), Image.LANCZOS)
                img.load()
                self._cache[j] = img
            return img

    def trim_cache(self, t: float) -> None:
        i = max(0, bisect.bisect_right(self.starts, t) - 1)
        with self._lock:
            for k in [k for k in self._cache if k < i - 1 or k > i + 2]:
                del self._cache[k]

    def _view(self, j: int, t: float) -> Image.Image:
        life0 = 0.0 if j == 0 else self.starts[j] - X / 2
        life1 = self.total if j == self.n - 1 else self.starts[j + 1] + X / 2
        p = min(1.0, max(0.0, (t - life0) / max(1e-6, life1 - life0)))
        m = self.motion[j]
        z = m["z0"] + (m["z1"] - m["z0"]) * p
        half = 0.5 / z
        cx = min(1 - half, max(half, m["c0"][0] + (m["c1"][0] - m["c0"][0]) * p))
        cy = min(1 - half, max(half, m["c0"][1] + (m["c1"][1] - m["c0"][1]) * p))
        box = ((cx - half) * W, (cy - half) * H, (cx + half) * W, (cy + half) * H)
        return self._src(j).resize((W, H), Image.BICUBIC, box=box)

    def frame(self, k: int) -> bytes:
        t = k / FPS
        i = min(self.n - 1, max(0, bisect.bisect_right(self.starts, t) - 1))
        if i + 1 < self.n and t >= self.starts[i + 1] - X / 2:
            a = _smooth((t - (self.starts[i + 1] - X / 2)) / X)
            img = Image.blend(self._view(i, t), self._view(i + 1, t), a)
        elif i > 0 and t < self.starts[i] + X / 2:
            a = _smooth((t - (self.starts[i] - X / 2)) / X)
            img = Image.blend(self._view(i - 1, t), self._view(i, t), a)
        else:
            img = self._view(i, t)
        img = ImageChops.multiply(img, self.vignette)

        # sleep-friendly: gently darken the picture over the final 40% of the video
        dim = 1.0 - 0.40 * _smooth((t - 0.6 * self.total) / (0.4 * self.total))

        fade = dim
        if t < 1.5:
            fade = min(fade, _smooth(t / 1.5))
        if t > self.total - END_FADE_SECONDS:
            fade = min(fade, _smooth((self.total - t) / END_FADE_SECONDS))
        if fade < 1.0:
            img = Image.blend(self.black, img, fade)
        return img.tobytes()


# --------------------------------------------------------------------------- render
def render_video(plan: List[Dict], timeline: Dict, script: Dict, audio_path: Path, out_path: Path,
                 workdir: Path, seed: int = 0) -> Path:
    total = timeline["total_duration"]
    nframes = int(round(total * FPS))
    renderer = Renderer(plan, timeline, script, seed)
    tmp_out = workdir / "video_rendering.mp4"
    ffmpeg_log = workdir / "ffmpeg.log"

    cmd = [config.find_ffmpeg(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF), "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-r", str(FPS),
           "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "44100",
           "-movflags", "+faststart", "-t", f"{total:.3f}", str(tmp_out)]
    log(f"Rendering {nframes} frames ({total:.0f}s) at {W}x{H} {FPS}fps ...")
    with open(ffmpeg_log, "wb") as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
        workers = max(1, min(6, (os.cpu_count() or 2) - 1))
        chunk, t0, next_report = workers * 8, time.time(), 0.05
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0, nframes, chunk):
                    renderer.trim_cache(start / FPS)
                    for data in pool.map(renderer.frame, range(start, min(nframes, start + chunk))):
                        proc.stdin.write(data)
                    done = min(nframes, start + chunk) / nframes
                    if done >= next_report:
                        el = time.time() - t0
                        log(f"  rendered {done * 100:.0f}%  (ETA {el / done * (1 - done) / 60:.1f} min)")
                        next_report += 0.05
            proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed (exit {rc}): {ffmpeg_log.read_text(errors='ignore')[-1500:]}")
    shutil.move(str(tmp_out), str(out_path))
    return out_path


# --------------------------------------------------------------------------- thumbnail
def make_thumbnail(plan: List[Dict], title: str, out_path: Path) -> Path:
    """1280x720 JPG: first scene image, brightened, with a big readable title (thumbnails may carry text; the video never does)."""
    from PIL import ImageEnhance, ImageFilter
    img = Image.open(plan[0]["path"]).convert("RGB").resize((1280, 720), Image.LANCZOS)
    img = ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(1.25)).enhance(1.12)
    shade = Image.new("L", (1280, 720), 0)
    d = ImageDraw.Draw(shade)
    for x in range(1280):
        d.line([(x, 0), (x, 720)], fill=int(190 * max(0.0, 1 - x / 900)))
    img = Image.composite(Image.new("RGB", (1280, 720), (0, 0, 0)), img, shade.filter(ImageFilter.GaussianBlur(20)))
    words = title.upper().split()
    size = 96
    while True:
        font = _font(size)
        lines, cur = [], ""
        for w in words:
            t = f"{cur} {w}".strip()
            if d.textlength(t, font=font) > 700 and cur:
                lines.append(cur); cur = w
            else:
                cur = t
        lines.append(cur)
        if len(lines) * size * 1.12 <= 560 or size <= 48:
            break
        size -= 6
    draw = ImageDraw.Draw(img)
    y = (720 - len(lines) * size * 1.12) / 2
    for ln in lines:
        draw.text((60, y), ln, font=font, fill=(255, 236, 190), stroke_width=6, stroke_fill=(0, 0, 0))
        y += size * 1.12
    img.save(out_path, "JPEG", quality=92)
    return out_path
