"""1080p / 24 fps / H.264 / AAC rendering: dynamic multi-shot Ken Burns, crossfades, vignette. No on-screen text.

Every AI image is split into several 4-8 second "shots", each with its own framing (wide, close detail, pan,
tilt, push-in, pull-out) so the picture changes every few seconds without needing 150+ generated images.
Every frame is computed with Pillow using *fractional* crop boxes (sub-pixel accurate bicubic resize),
so zooms and pans are perfectly smooth. Frames are streamed straight into ffmpeg's stdin.
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
from typing import Dict, List

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

import config
from config import (AUDIO_BITRATE, CROSSFADE_SECONDS, END_FADE_SECONDS, FPS, SHOT_MAX_SECONDS, SHOT_MIN_SECONDS,
                    VIDEO_H, VIDEO_W, VIGNETTE_STRENGTH, X264_CRF, X264_PRESET, log)
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


# --------------------------------------------------------------------------- shots
def expand_shots(plan: List[Dict], seed: int = 0) -> List[Dict]:
    """Split every image segment into consecutive shots of roughly 4-8 s (never one static image for long)."""
    rng = random.Random(seed * 31 + 7)
    mid = (SHOT_MIN_SECONDS + SHOT_MAX_SECONDS) / 2
    out: List[Dict] = []
    for seg in plan:
        d = seg["end"] - seg["start"]
        n = max(1, int(round(d / mid)))
        while n > 1 and d / n < SHOT_MIN_SECONDS:
            n -= 1
        while d / n > SHOT_MAX_SECONDS:
            n += 1
        cuts = [seg["start"]] + [seg["start"] + d * k / n + rng.uniform(-0.1, 0.1) * d / n for k in range(1, n)] + [seg["end"]]
        for k in range(n):
            out.append({**seg, "start": cuts[k], "end": cuts[k + 1], "shot": k, "nshots": n})
    return out


FOCI = [(0.5, 0.5), (0.30, 0.50), (0.70, 0.50), (0.50, 0.32), (0.50, 0.68),
        (0.28, 0.34), (0.72, 0.66), (0.72, 0.34), (0.28, 0.66)]
MODES = ["zoom_in", "pan", "zoom_out", "tilt", "diag_zoom", "pan", "zoom_in", "tilt"]


def _motion(mode: str, f: tuple, rng: random.Random) -> dict:
    zs = rng.uniform(0.20, 0.32)
    sign = rng.choice([-1, 1])
    if mode == "zoom_in":                     # push in on a point of interest
        z0 = rng.uniform(1.0, 1.10)
        return dict(z0=z0, z1=z0 + zs, c0=(0.5, 0.5), c1=f)
    if mode == "zoom_out":                    # pull back from a detail to the wide view
        z1 = rng.uniform(1.0, 1.10)
        return dict(z0=z1 + zs, z1=z1, c0=f, c1=(0.5, 0.5))
    if mode == "pan":                         # full-width horizontal slide with a slight push
        z = rng.uniform(1.32, 1.42)
        return dict(z0=z, z1=z + 0.05, c0=(0.5 - sign * 0.5, f[1]), c1=(0.5 + sign * 0.5, f[1] + rng.uniform(-0.05, 0.05)))
    if mode == "tilt":                        # vertical slide
        z = rng.uniform(1.32, 1.42)
        return dict(z0=z, z1=z, c0=(f[0], 0.5 - sign * 0.5), c1=(f[0] + rng.uniform(-0.05, 0.05), 0.5 + sign * 0.5))
    z0 = rng.uniform(1.05, 1.15)              # diag_zoom: diagonal push-in
    return dict(z0=z0, z1=z0 + zs, c0=(1 - f[0], 1 - f[1]), c1=f)


class Renderer:
    def __init__(self, plan: List[Dict], timeline: Dict, script: Dict, seed: int = 0):
        self.plan = expand_shots(plan, seed)
        self.total = timeline["total_duration"]
        self.starts = [s["start"] for s in self.plan]
        self.n = len(self.plan)
        self.motion: List[dict] = []
        per_image: Dict[str, tuple] = {}
        prev_mode = None
        for seg in self.plan:
            rng = random.Random(f"{seed}-{seg['file']}-{seg['shot']}")
            if seg["file"] not in per_image:   # one shuffled focal order + mode offset per image
                r = random.Random(f"{seed}-{seg['file']}")
                foci = FOCI[:]
                r.shuffle(foci)
                per_image[seg["file"]] = (foci, r.randrange(len(MODES)))
            foci, off = per_image[seg["file"]]
            mode = MODES[(off + seg["shot"]) % len(MODES)]
            if mode == prev_mode:
                mode = MODES[(off + seg["shot"] + 1) % len(MODES)]
            prev_mode = mode
            self.motion.append(_motion(mode, foci[seg["shot"] % len(foci)], rng))
        self.vignette = build_vignette()
        self.black = Image.new("RGB", (W, H), (0, 0, 0))
        self._cache: Dict[str, Image.Image] = {}
        self._lock = threading.Lock()

    # -- source images (small cache: only neighbouring shots are ever needed)
    def _src(self, j: int) -> Image.Image:
        key = self.plan[j]["path"]
        with self._lock:
            img = self._cache.get(key)
            if img is None:
                img = Image.open(key).convert("RGB")
                if img.size != (W, H):
                    img = img.resize((W, H), Image.LANCZOS)
                img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=60, threshold=2))  # crops stay crisp
                img.load()
                self._cache[key] = img
            return img

    def trim_cache(self, t: float) -> None:
        i = max(0, bisect.bisect_right(self.starts, t) - 1)
        keep = {self.plan[k]["path"] for k in range(max(0, i - 1), min(self.n, i + 3))}
        with self._lock:
            for k in [k for k in self._cache if k not in keep]:
                del self._cache[k]

    def _view(self, j: int, t: float) -> Image.Image:
        life0 = 0.0 if j == 0 else self.starts[j] - X / 2
        life1 = self.total if j == self.n - 1 else self.starts[j + 1] + X / 2
        p = min(1.0, max(0.0, (t - life0) / max(1e-6, life1 - life0)))
        p = 0.5 * p + 0.5 * _smooth(p)          # gentle ease, still moving through the crossfades
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
        fade = 1.0 - 0.40 * _smooth((t - 0.6 * self.total) / (0.4 * self.total))
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
    log(f"{len(plan)} images -> {renderer.n} shots (avg {total / renderer.n:.1f}s each)")
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
                lines.append(cur)
                cur = w
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
