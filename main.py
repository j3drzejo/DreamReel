"""Pipeline orchestration: scan -> script -> voice -> images -> audio master -> video -> info.txt.

Each ML stage loads its model, runs, then deletes it and clears the GPU cache before the next stage
starts, so only ONE model is ever resident in memory. Every stage caches its output in
<project>/.build_cache/ and a re-run resumes from the last completed step.
"""
from __future__ import annotations

import config  # noqa: F401  (must come first: sets env vars before torch is imported)

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from config import BUILD_CACHE, Settings, fmt_duration, fmt_timestamp, log, read_json, write_json


class Project:
    """Output folder named after the video title. Created as soon as a title exists."""

    def __init__(self, folder: Optional[Path] = None):
        self.folder = folder

    @property
    def cache(self) -> Optional[Path]:
        return None if self.folder is None else self.folder / BUILD_CACHE

    def create(self, base: Path, title: str, topic: str) -> Path:
        base.mkdir(parents=True, exist_ok=True)
        name = config.sanitize_folder_name(title)
        folder, k = base / name, 2
        while folder.exists():
            folder, k = base / f"{name} ({k})", k + 1
        (folder / BUILD_CACHE).mkdir(parents=True)
        self.folder = folder
        write_json(self.cache / "meta.json", {"title": title, "topic": topic, "created": time.time()})
        return folder


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a 10+ minute 1080p sleep/history documentary, fully offline.")
    p.add_argument("--topic", help='Subject, e.g. "Life aboard an 18th-century sailing ship". Omit to auto-pick a fresh one.')
    p.add_argument("--duration", type=float, default=config.DEFAULT_DURATION_MIN,
                   help="Target length in minutes (minimum 10, default 12)")
    p.add_argument("--output-dir", default=".", help="Where the project folder is created (default: current directory)")
    p.add_argument("--llm", default="auto", help="Hugging Face LLM id (default: auto-select for your memory)")
    p.add_argument("--image-model", default="auto", help="Hugging Face diffusion model id (default: auto-select)")
    p.add_argument("--tts", choices=["auto", "dia", "kokoro"], default="auto",
                   help="Voice engine. auto = Dia-1.6B on a CUDA GPU with >=10 GB, otherwise Kokoro. dia forces Dia.")
    p.add_argument("--dia-model", default=config.DIA_MODEL, help="Dia checkpoint (default nari-labs/Dia-1.6B-0626)")
    p.add_argument("--recast-voice", action="store_true",
                   help="Dia: pick a new narrator voice instead of reusing assets/voice/dia_voice.wav")
    p.add_argument("--voice", default=config.KOKORO_VOICE, help="Kokoro voice (fallback engine): am_michael, am_fenrir, bm_george ...")
    p.add_argument("--speed", type=float, default=config.KOKORO_SPEED, help="Kokoro speaking speed (default 0.92)")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                   help="Force the compute device for all models (default auto: cuda > mps > cpu)")
    p.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto",
                   help="Video encoder: auto uses NVENC on NVIDIA GPUs when ffmpeg supports it, else libx264 (CPU)")
    p.add_argument("--image-size", help="Generate images at this size, e.g. 1920x1080 (default: the model's native 1280x720). "
                                        "Bigger = sharper but slower")
    p.add_argument("--fire-db", type=float, default=config.FIRE_IDLE_DB,
                   help="Fireplace level in dB below the voice when it is silent (default 20; 24 while speaking)")
    p.add_argument("--image-interval", type=float, default=config.SECONDS_PER_IMAGE,
                   help="Seconds of narration per generated AI image (default 5.5): every image is generated from the passage being "
                        "spoken. Higher = fewer images, faster. 4-8 is the sweet spot for engagement.")
    p.add_argument("--notes", help="Text file with your own trusted sources/facts about the topic (highest priority for accuracy)")
    p.add_argument("--seed", type=int, default=1234, help="Base seed for image generation")
    p.add_argument("--keep-cache", action="store_true", help="Keep .build_cache/ after a successful run")
    p.add_argument("--fresh", action="store_true", help="Do not resume an unfinished project; start a new one")
    p.add_argument("--smoke-test", action="store_true",
                   help="Test the full audio mix + 1080p render with real speech (Kokoro if installed) and placeholder images, ~1 min")
    return p.parse_args(argv)


def build_info(script: dict, timeline: dict, duration_s: float) -> str:
    lines = [script["description"].strip(), "", "Chapters:"]
    for ch in timeline["chapters"]:
        lines.append(f"{fmt_timestamp(0 if ch['index'] == 0 else ch['voice_start'])} {ch['title']}")
    tags = ["sleepstory", "history", "relaxing", "documentary"] + [t.replace(" ", "") for t in script.get("tags", [])]
    seen, hashtags = set(), []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            hashtags.append("#" + t)
    if script.get("sources"):
        lines += ["", "Sources:"] + [f"{x['title']} - {x['url']}" for x in script["sources"]]
    lines += ["", " ".join(hashtags[:15]), "", "Narration and visuals in this video are AI-generated. Please enable the 'altered or synthetic content' disclosure when uploading."]
    desc = "\n".join(lines).replace('"', "'")
    title = script["title"].replace('"', "'")
    topic = script.get("topic", "").replace('"', "'")
    tags = ", ".join(script.get("tags", [])).replace('"', "'")
    return (f'title: "{title}"\ndescription: "{desc}"\nduration: "{fmt_duration(duration_s)}"\n'
            f'topic: "{topic}"\ntags: "{tags}"\n')


def write_research_notes(path: Path, script: dict) -> None:
    """Human-readable fact sheet + sources so you can fact-check the video before publishing."""
    sh = script.get("sheet") or {}
    out = [f"# {script['title']}", "", f"Grounded in sources: {'yes' if script.get('grounded') else 'NO - model knowledge only, verify before publishing'}", ""]
    out += [f"**Period:** {sh.get('period', '')}", f"**Region:** {sh.get('region', '')}", f"**Setting:** {sh.get('setting', '')}", ""]
    for key, label in (("daily_life", "What people did"), ("clothing", "Clothing"), ("architecture_objects", "Buildings, objects, tools"),
                       ("food_drink", "Food and drink"), ("uncertain", "Uncertain or debated"), ("anachronisms", "Excluded anachronisms")):
        out += [f"## {label}"] + [f"- {x}" for x in sh.get(key, [])] + [""]
    out += ["## Sources"] + [f"- {x['title']}: {x['url']}" for x in script.get("sources", [])]
    path.write_text("\n".join(out), encoding="utf-8")


def run(cfg: Settings) -> int:
    import audio_mixer
    import image_gen
    import scanner
    import script_gen
    import tts_gen
    import video_editor

    config.find_ffmpeg()
    cfg.output_dir = Path(cfg.output_dir)
    if cfg.duration_min < 10:
        log("Duration raised to the 10 minute minimum.")
        cfg.duration_min = 10.0
    roots = [Path.cwd(), Path.cwd() / "output", cfg.output_dir]
    t_start = time.time()
    log(f"Device: {config.get_device()} | {config.hardware_summary()}")

    project = Project()
    if not cfg.fresh:
        resumed = scanner.find_incomplete_project(roots, cfg.topic)
        if resumed:
            project = Project(resumed)
            log(f"Resuming unfinished project: {resumed.name}")
    existing = scanner.scan_existing(roots, exclude=project.folder)
    log(f"Found {len(existing)} existing project(s): " + (", ".join(e.title for e in existing) or "none"))
    seeds = [] if cfg.topic else scanner.pick_seed_candidates(existing)

    # ---- Stage 1: script (LLM loaded, used, deleted inside)
    script = script_gen.generate_script(cfg, existing, seeds, project)
    cache = project.cache

    # ---- Stage 2: voiceover (Kokoro loaded, used, deleted inside)
    timeline = tts_gen.generate_voiceover(cfg, script, cache)

    # ---- Stage 3: images (diffusion pipeline loaded, used, deleted inside)
    plan = image_gen.generate_images(cfg, script, timeline, cache)

    # ---- Stage 4: ambient mix + mastering (numpy/scipy only)
    master = cache / "master.wav"
    if not master.exists():
        audio_mixer.build_master(cfg, timeline, cache)
    else:
        log("Mastered audio found in cache - skipping")

    # ---- Stage 5: render
    video_tmp = cache / "video_final.mp4"
    if not video_tmp.exists():
        video_editor.render_video(plan, timeline, script, master, video_tmp, cache, seed=cfg.seed, encoder=cfg.encoder)
    video_editor.make_thumbnail(plan, script["title"], project.folder / "thumbnail.jpg")
    final = project.folder / "video.mp4"
    shutil.move(str(video_tmp), str(final))

    # ---- Finish
    duration = config.probe_duration(final)
    if duration < 600:
        log(f"WARNING: rendered duration is {duration:.0f}s (< 10 min)")
    (project.folder / "info.txt").write_text(build_info(script, timeline, duration), encoding="utf-8")
    write_research_notes(project.folder / "research.md", script)
    write_json(cache / "done.json", {"done": True})
    if not cfg.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
    log(f"DONE in {fmt_duration(time.time() - t_start)}: {final}  ({fmt_duration(duration)})")
    return 0


def _smoke_voice(texts, sr, cfg, work):
    """Real speech for the smoke test: Kokoro -> espeak-ng -> macOS say -> (last resort) noise."""
    import shutil
    import subprocess

    import numpy as np
    import soundfile as sf
    from scipy import signal

    try:
        import tts_gen

        backend = tts_gen.open_backend(cfg)
        try:
            out = [tts_gen._synth_chapter(backend, t)[0] for t in texts]
            sr_used = backend.sr
        finally:
            backend.close()
            config.free_memory()
        log(f"Smoke test voice: {backend.name}")
        return out, sr_used
    except Exception as exc:
        log(f"Speech engine unavailable for smoke test ({type(exc).__name__}: {exc})")

    def _load_resampled(path):
        x, fs = sf.read(str(path), dtype="float32")
        x = x if x.ndim == 1 else x.mean(axis=1)
        return signal.resample_poly(x, sr, fs).astype(np.float32) if fs != sr else x

    for tool in ("espeak-ng", "espeak", "say"):
        exe = shutil.which(tool)
        if not exe:
            continue
        try:
            out = []
            for i, t in enumerate(texts):
                p = work / f"fallback_{i}.{'aiff' if tool == 'say' else 'wav'}"
                cmd = [exe, "-o", str(p), t] if tool == "say" else [exe, "-s", "130", "-w", str(p), t.replace("\n", " ")]
                subprocess.run(cmd, check=True, capture_output=True)
                if tool == "say":
                    wav = p.with_suffix(".wav")
                    subprocess.run([config.find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(p), str(wav)], check=True)
                    p = wav
                out.append(_load_resampled(p))
            log(f"Smoke test voice: {tool} (robotic fallback - install Kokoro for the real narrator voice)")
            return out, sr
        except Exception:
            continue
    log("WARNING: no speech engine found - using synthetic noise as a stand-in voice. Mix levels are still tested, "
        "but this will NOT sound like a human voice.")
    rng = np.random.default_rng(1)
    out = []
    for i in range(len(texts)):
        n = int(9 * sr)
        t = np.arange(n) / sr
        env = np.clip(np.sin(2 * np.pi * 3.2 * t) * np.sin(2 * np.pi * 0.7 * t + i), 0, 1)
        out.append((rng.standard_normal(n) * env * 0.2).astype(np.float32))
    return out, sr


def smoke_test(cfg: Settings) -> int:
    """Exercise mixer + renderer end-to-end with synthetic data. No downloads, no GPU."""
    import numpy as np
    import soundfile as sf
    from PIL import Image, ImageDraw

    import audio_mixer
    import image_gen
    import tts_gen
    import video_editor

    config.find_ffmpeg()
    work = Path(tempfile.mkdtemp(prefix="sleepdoc_smoke_"))
    log(f"Smoke test in {work}")
    sr = config.KOKORO_SR
    titles = ["The Arrival", "Morning Light", "The Hearth"]
    texts = [
        "Tonight, you step through a heavy wooden door, and the cold air of the evening falls away behind you.\n\n"
        "Take a slow breath. Let your shoulders soften, and let the day grow quiet.",
        "In the morning, pale light slips across the stone floor. Somewhere nearby, a kettle begins to hum.\n\n"
        "You listen to the soft sounds of a world waking gently.",
        "By the hearth, the fire crackles and glows. Warm shadows move slowly along the walls.\n\n"
        "Rest now. Everything is calm, and everything is safe.",
    ]
    chapters_audio, sr = _smoke_voice(texts, sr, cfg, work)
    durs = [len(x) / sr for x in chapters_audio]
    pieces = [np.zeros(int(config.LEAD_IN_SECONDS * sr), np.float32)]
    for i, x in enumerate(chapters_audio):
        pieces.append(x)
        if i < len(chapters_audio) - 1:
            pieces.append(np.zeros(int(config.CHAPTER_PAUSE * sr), np.float32))
    sf.write(work / "voiceover.wav", np.concatenate(pieces), sr, subtype="FLOAT")
    timeline = tts_gen.build_timeline(titles, durs, [len(t.split()) for t in texts], min_total=0, sample_rate=sr)
    import shots
    script = {"chapters": []}
    for t, tx in zip(titles, texts):
        sents = shots.split_sentences(tx)
        units = shots.make_units(sents, shots.target_words_for(cfg.image_interval))
        script["chapters"].append({"title": t, "narration": tx, "sentences": sents, "units": units})

    plan = shots.build_unit_plan(script, timeline)
    (work / "images").mkdir()
    for k, seg in enumerate(plan):
        yy, xx = np.mgrid[0:1080, 0:1920]
        base = np.stack([(xx / 1920 * 200 + k * 30) % 255, (yy / 1080 * 160 + 40) % 255, np.full_like(xx, 90 + k * 12)], -1)
        img = Image.fromarray(base.astype(np.uint8), "RGB")
        d = ImageDraw.Draw(img)
        for gx in range(0, 1920, 160):
            d.line([(gx, 0), (gx, 1080)], fill=(255, 255, 255), width=2)
        for gy in range(0, 1080, 120):
            d.line([(0, gy), (1920, gy)], fill=(255, 255, 255), width=2)
        seg["path"] = str(work / "images" / seg["file"])
        img.save(seg["path"])

    master = audio_mixer.build_master(cfg, timeline, work)
    out = video_editor.render_video(plan, timeline, script, master, work / "smoke.mp4", work, seed=1, encoder=cfg.encoder)
    video_editor.make_thumbnail(plan, "A Night Inside a Medieval Monastery", work / "thumbnail.jpg")
    dur = config.probe_duration(out)
    log(f"SMOKE TEST OK: {out}  ({dur:.1f}s, expected {timeline['total_duration']:.1f}s)")
    return 0 if abs(dur - timeline["total_duration"]) < 0.5 else 1


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = Settings(topic=a.topic, duration_min=a.duration, output_dir=Path(a.output_dir), llm_model=a.llm,
                   image_model=a.image_model, voice=a.voice, speed=a.speed, seed=a.seed, keep_cache=a.keep_cache,
                   fresh=a.fresh, fire_db=a.fire_db, image_interval=a.image_interval, notes=a.notes,
                   tts=a.tts, dia_model=a.dia_model, recast_voice=a.recast_voice, device=a.device, encoder=a.encoder,
                   image_size=a.image_size)
    config.set_device_override(a.device)
    config.gpu_setup()
    try:
        return smoke_test(cfg) if a.smoke_test else run(cfg)
    except KeyboardInterrupt:
        log("Interrupted. Run the same command again to resume where it stopped.")
        return 130
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        log("Your progress is saved in .build_cache/ - fix the problem and re-run to resume.")
        raise


if __name__ == "__main__":
    sys.exit(main())
