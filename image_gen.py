"""16:9 scene images with a local diffusion model (Z-Image-Turbo / FLUX.2-klein / SDXL-Turbo fallback).

No torch import at module level: video_editor imports build_image_plan() from here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

import config
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
    "Tongyi-MAI/Z-Image-Turbo": dict(w=1280, h=720, steps=9, guidance=0.0, dtype="bf16", offload=True),
    "black-forest-labs/FLUX.2-klein-4B": dict(w=1280, h=720, steps=4, guidance=1.0, dtype="bf16", offload=True),
    "stabilityai/sdxl-turbo": dict(w=1024, h=576, steps=4, guidance=0.0, dtype="fp16", offload=False),
}
DEFAULT_SPEC = dict(w=1280, h=720, steps=8, guidance=0.0, dtype="bf16", offload=True)


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


def _prompt_for(ch: Dict, j: int) -> str:
    subs = ch.get("sub_scenes") or []
    if j == 0:
        scene = ch["image_prompt"]
    elif j <= len(subs):
        scene = subs[j - 1]
    else:
        scene = f"{ch['image_prompt']}, {SHOT_VARIANTS[j % len(SHOT_VARIANTS)]}"
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
    start = 0 if mem >= 22 else 1 if mem >= 14 else 2
    log(f"Image model auto-select: ~{mem:.0f} GB usable on {device} -> starting with {tiers[start]}")
    if device == "cpu":
        log("WARNING: no GPU detected. Image generation on CPU will be very slow (minutes per image).")
    return tiers[start:]


def _load_pipe(model_id: str, spec: dict, device: str):
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if spec["dtype"] == "bf16" else torch.float16)
    log(f"Loading image model {model_id} ({str(dtype).split('.')[-1]}) ...")
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if device == "cuda":
        pipe.enable_model_cpu_offload() if spec["offload"] else pipe.to("cuda")
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


def _generate_one(pipe, spec: dict, prompt: str, seed: int) -> Image.Image:
    import torch

    for attempt in range(3):
        gen = torch.Generator(device="cpu").manual_seed(seed + attempt * 7919)
        kwargs = dict(prompt=prompt, width=spec["w"], height=spec["h"], num_inference_steps=spec["steps"], generator=gen)
        try:
            img = pipe(guidance_scale=spec["guidance"], **kwargs).images[0]
        except TypeError:
            img = pipe(**kwargs).images[0]
        if np.asarray(img.convert("RGB")).mean() > 3:   # reject all-black (NaN) outputs
            return img
        log("  black image produced, retrying with another seed")
    raise RuntimeError("Model keeps producing black images (fp16 NaNs?)")


# --------------------------------------------------------------------------- public entry
def generate_images(cfg: config.Settings, script: Dict, timeline: Dict, cache: Path) -> List[Dict]:
    img_dir = cache / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    plan = build_image_plan(timeline)
    for seg in plan:
        seg["path"] = str(img_dir / seg["file"])
    todo = [s for s in plan if not Path(s["path"]).exists()]
    log(f"Images: {len(plan)} total ({len(plan) - len(todo)} cached, {len(todo)} to generate)")
    if not todo:
        return plan

    device = config.get_device()
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
        pipe = None
        try:
            pipe = _load_pipe(model_id, spec, device)
            marker.write_text(model_id)
            for n, seg in enumerate(todo, 1):
                ch = script["chapters"][seg["chapter"]]
                seed = cfg.seed + seg["chapter"] * 100 + seg["sub"]
                log(f"Image {n}/{len(todo)}  (chapter {seg['chapter'] + 1}, scene {seg['sub'] + 1})")
                img = _to_1080p(_generate_one(pipe, spec, _prompt_for(ch, seg["sub"]), seed))
                tmp = Path(seg["path"] + ".tmp.png")
                img.save(tmp, "PNG")
                tmp.replace(seg["path"])
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
