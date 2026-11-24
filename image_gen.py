"""16:9 scene images with a local diffusion model (Z-Image-Turbo / FLUX.2-klein / SDXL-Turbo fallback).

No torch import at module level: video_editor imports build_image_plan() from here.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import config
import shots
from config import CHAPTER_PAUSE, SECONDS_PER_IMAGE, VIDEO_H, VIDEO_W, free_memory, log

STYLE = ("cinematic oil painting style, historically accurate, warm atmospheric lighting, soft volumetric light, "
         "rich painterly detail, consistent muted warm color palette, 16:9 widescreen composition, no text, no writing, no letters, "
         "no watermark, no modern objects, figures seen from behind or at a distance, no faces looking at camera")
SHOT_VARIANTS = [
    "wide establishing shot", "intimate close-up of everyday objects", "medium shot of quiet daily activity",
    "atmospheric detail shot with drifting dust motes in the light", "view through a doorway or window",
    "calm empty interior at dusk", "low angle looking up at the architecture", "rooftops and horizon at twilight",
]

MODEL_SPECS: Dict[str, dict] = {
    # need_gb = approximate GPU memory to keep the whole pipeline resident (no CPU offload)
    "Tongyi-MAI/Z-Image-Turbo": dict(w=1280, h=720, steps=9, guidance=0.0, dtype="bf16", need_gb=22),
    "black-forest-labs/FLUX.2-klein-4B": dict(w=1280, h=720, steps=4, guidance=1.0, dtype="bf16", need_gb=16),
    "stabilityai/sdxl-turbo": dict(w=1024, h=576, steps=4, guidance=0.0, dtype="fp16", need_gb=8, clip=True),
}
DEFAULT_SPEC = dict(w=1280, h=720, steps=8, guidance=0.0, dtype="bf16", need_gb=20)


def parse_size(txt: Optional[str], spec: dict) -> Tuple[int, int]:
    """'1920x1080' -> (1920, 1088): multiples of 16 as diffusion models require (the extra rows are cropped later)."""
    if not txt:
        return spec["w"], spec["h"]
    m = re.match(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*$", txt)
    if not m:
        raise ValueError(f"--image-size must look like 1920x1080, got {txt!r}")
    w, h = (int(round(int(v) / 16.0)) * 16 for v in m.groups())
    return max(256, w), max(256, h)


def plan_runtime(spec: dict, device: str) -> dict:
    """Use the GPU as fully as its memory allows: keep the whole model resident and generate several images per call.
    Smaller GPUs fall back to CPU offload (slow but works); Apple/CPU run one image at a time."""
    if device != "cuda":
        return {"offload": False, "batch": 1}
    vram = config.usable_memory_gb(device)
    if vram < spec["need_gb"]:
        return {"offload": True, "batch": 1}
    return {"offload": False, "batch": int(max(1, min(6, (vram - spec["need_gb"]) // 3 + 1)))}


# --------------------------------------------------------------------------- plan (shared with video_editor)
def build_image_plan(timeline: Dict, seconds_per_image: float = SECONDS_PER_IMAGE) -> List[Dict]:
    """Split the video into consecutive image segments, ~one per 25 s, chapter by chapter."""
    chs, total = timeline["chapters"], timeline["total_duration"]
    bounds = [0.0] + [c["voice_start"] - CHAPTER_PAUSE / 2 for c in chs[1:]] + [total]
    plan: List[Dict] = []
    for i, ch in enumerate(chs):
        s, e = bounds[i], bounds[i + 1]
        n = max(2, round((e - s) / seconds_per_image))
        for j in range(n):
            plan.append({"chapter": i, "sub": j, "start": s + (e - s) * j / n, "end": s + (e - s) * (j + 1) / n,
                         "file": f"ch{i + 1:02d}_{j + 1:02d}.png"})
    return plan


STYLE_SHORT = "oil painting, muted warm palette, soft candlelight, distant figures"


def _finish(scene: str, clip: bool) -> str:
    if clip:  # CLIP text encoders read only 77 tokens: keep the scene first and the style short
        scene = " ".join(scene.replace("\n", " ").split()[:30]).rstrip(",.;: ")
        return f"{scene}. {STYLE_SHORT}"
    return f"{scene}. {STYLE}"


def _seg_prompt(seg: Dict, ch: Dict, clip: bool) -> str:
    return _finish(seg["prompt"], clip) if seg.get("prompt") else _prompt_for(ch, seg["sub"], clip)


def _prompt_for(ch: Dict, j: int, clip: bool = False) -> str:
    subs = ch.get("sub_scenes") or []
    if j == 0:
        scene = ch["image_prompt"]
    elif j <= len(subs):
        scene = subs[j - 1]
    else:
        scene = f"{ch['image_prompt']}, {SHOT_VARIANTS[j % len(SHOT_VARIANTS)]}"
    if clip:  # CLIP text encoders read only 77 tokens: keep the scene first and the style short
        scene = " ".join(scene.replace("\n", " ").split()[:30]).rstrip(",.;: ")
        return f"{scene}. {STYLE_SHORT}"
    return f"{scene}. {STYLE}"


# --------------------------------------------------------------------------- model handling
def _choose_chain(cfg: config.Settings, device: str) -> List[str]:
    tiers = config.IMAGE_TIERS
    if cfg.image_model != "auto":
        chain = [cfg.image_model]
        if cfg.image_model in tiers:
            chain += tiers[tiers.index(cfg.image_model) + 1:]
        elif tiers[-1] not in chain:
            chain.append(tiers[-1])
        return chain
    mem = config.usable_memory_gb(device)
    start = 0 if mem >= 22 else 1 if mem >= 15 else 2
    log(f"Image model auto-select: ~{mem:.0f} GB usable on {device} -> starting with {tiers[start]}")
    if device == "cpu":
        log("WARNING: no GPU detected. Image generation on CPU will be very slow (minutes per image).")
    return tiers[start:]


def _load_pipe(model_id: str, spec: dict, device: str, offload: bool):
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if spec["dtype"] == "bf16" else torch.float16)
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    log(f"Loading image model {model_id} ({str(dtype).split('.')[-1]}, "
        f"{'CPU offload: GPU too small to hold it' if offload else 'fully on ' + device}) ...")
    try:
        pipe = DiffusionPipeline.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if device == "cuda":
        pipe.enable_model_cpu_offload() if offload else pipe.to("cuda")
    elif device == "mps":
        pipe.to("mps")
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    else:
        pipe.to("cpu")
    for fn in ("enable_vae_tiling", "enable_vae_slicing"):
        try:
            getattr(pipe, fn)()
        except Exception:
            pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def _to_1080p(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    target = VIDEO_W / VIDEO_H
    if abs(w / h - target) > 0.01:  # center-crop to 16:9 so there are never black bars
        if w / h > target:
            nw = int(h * target)
            img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(w / target)
            img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return img.resize((VIDEO_W, VIDEO_H), Image.LANCZOS)


def _generate_batch(pipe, spec: dict, prompts: List[str], seeds: List[int], w: int, h: int) -> List[Image.Image]:
    import torch

    out: List[Optional[Image.Image]] = [None] * len(prompts)
    pending = list(range(len(prompts)))
    for attempt in range(3):
        gens = [torch.Generator(device="cpu").manual_seed(seeds[i] + attempt * 7919) for i in pending]
        one = len(pending) == 1
        kwargs = dict(prompt=prompts[pending[0]] if one else [prompts[i] for i in pending], width=w, height=h,
                      num_inference_steps=spec["steps"], generator=gens[0] if one else gens)
        try:
            imgs = pipe(guidance_scale=spec["guidance"], **kwargs).images
        except TypeError:
            imgs = pipe(**kwargs).images
        nxt = []
        for i, img in zip(pending, imgs):
            if np.asarray(img.convert("RGB")).mean() > 3:   # reject all-black (NaN) outputs
                out[i] = img
            else:
                nxt.append(i)
        pending = nxt
        if not pending:
            break
        log("  black image produced, retrying with another seed")
    if pending:
        raise RuntimeError("Model keeps producing black images (fp16 NaNs?)")
    return out  # type: ignore[return-value]


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


# --------------------------------------------------------------------------- public entry
def generate_images(cfg: config.Settings, script: Dict, timeline: Dict, cache: Path) -> List[Dict]:
    img_dir = cache / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if script["chapters"][0].get("units"):
        plan = shots.build_unit_plan(script, timeline)      # one image per narration passage, timed to the speech
    else:
        plan = build_image_plan(timeline, cfg.image_interval)
    for seg in plan:
        seg["path"] = str(img_dir / seg["file"])
    import hashlib
    sig = hashlib.md5("|".join(s.get("prompt", "") + s["file"] for s in plan).encode()).hexdigest()
    sigf = img_dir / ".sig"
    if sigf.exists() and sigf.read_text().strip() != sig:   # prompts changed: old pictures would not match the story
        for f in img_dir.glob("*.png"):
            f.unlink()
    sigf.write_text(sig)
    todo = [s for s in plan if not Path(s["path"]).exists()]
    log(f"Images: {len(plan)} total ({len(plan) - len(todo)} cached, {len(todo)} to generate)")
    if not todo:
        return plan

    device = config.get_device()
    if device != "cuda" and len(todo) > 40:
        log(f"WARNING: {len(todo)} images on {device} will take a long time (no CUDA GPU). Raise --image-interval "
            f"(e.g. 12) for fewer images, or run on a machine with an NVIDIA GPU.")
    chain = _choose_chain(cfg, device)
    marker = img_dir / ".model"
    if marker.exists():
        prev = marker.read_text().strip()
        if prev in chain:
            chain = [prev] + [m for m in chain if m != prev]

    done = False
    for model_id in chain:
        if marker.exists() and marker.read_text().strip() != model_id:
            for f in img_dir.glob("*.png"):   # never mix visual styles from two models
                f.unlink()
        todo = [s for s in plan if not Path(s["path"]).exists()]
        spec = MODEL_SPECS.get(model_id, DEFAULT_SPEC)
        rt = plan_runtime(spec, device)
        w, h = parse_size(cfg.image_size, spec)
        pipe = None
        try:
            pipe = _load_pipe(model_id, spec, device, rt["offload"])
            marker.write_text(model_id)
            batch = rt["batch"]
            log(f"Generating {len(todo)} images at {w}x{h}, {batch} per batch")
            i = 0
            while i < len(todo):
                grp = todo[i:i + batch]
                log(f"Images {i + 1}-{i + len(grp)}/{len(todo)}  (chapter {grp[0]['chapter'] + 1})")
                try:
                    imgs = _generate_batch(pipe, spec,
                                           [_seg_prompt(g, script["chapters"][g["chapter"]], spec.get("clip", False)) for g in grp],
                                           [cfg.seed + g["chapter"] * 1000 + g["sub"] for g in grp], w, h)
                except Exception as exc:
                    if _is_oom(exc) and batch > 1:
                        batch = max(1, batch // 2)
                        free_memory()
                        log(f"  GPU memory full: continuing with {batch} per batch")
                        continue
                    raise
                for g, img in zip(grp, imgs):
                    tmp = Path(g["path"] + ".tmp.png")
                    _to_1080p(img).save(tmp, "PNG")
                    tmp.replace(g["path"])
                i += len(grp)
            done = True
        except Exception as exc:
            log(f"Image model {model_id} failed: {type(exc).__name__}: {exc}")
        finally:
            if pipe is not None:
                del pipe
            free_memory()
        if done:
            break
        log("Falling back to the next (lighter) image model ...")
    if not done:
        raise RuntimeError("All image models failed. Try --image-model stabilityai/sdxl-turbo or upgrade diffusers.")
    return plan
