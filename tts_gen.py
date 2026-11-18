"""Voiceover with Kokoro-82M (local, no API key). One WAV per chapter, cached for resume."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

import config
from config import (CHAPTER_PAUSE, KOKORO_SR, LEAD_IN_SECONDS, MIN_TOTAL_SECONDS, OUTRO_SECONDS,
                    PARAGRAPH_PAUSE, SENTENCE_GROUP_PAUSE, free_memory, log, write_json)


# --------------------------------------------------------------------------- text prep
def _tts_clean(text: str) -> str:
    text = text.replace("&", " and ").replace("%", " percent").replace("…", "...")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunks(paragraph: str, max_chars: int = 320) -> List[str]:
    """Group whole sentences into chunks so no word is ever cut off mid-sentence."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])["\')\]]*\s+(?=[A-Z"\'(])', paragraph) if s.strip()]
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * KOKORO_SR), dtype=np.float32)


def _fade_edges(x: np.ndarray, ms: float = 8.0) -> np.ndarray:
    n = min(int(KOKORO_SR * ms / 1000), len(x) // 2)
    if n > 0:
        x = x.copy()
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


# --------------------------------------------------------------------------- Kokoro wrapper
class _Kokoro:
    def __init__(self, voice: str):
        self.voice = voice
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 82M model is fast on CPU; avoids MPS quirks
        except Exception:
            self.device = "cpu"
        self.pipe = self._build(self.device)

    def _build(self, device: str):
        from kokoro import KPipeline

        lang = self.voice[0]  # a = American English, b = British English
        try:
            return KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)
        except TypeError:
            return KPipeline(lang_code=lang)

    def synth(self, text: str, speed: float) -> np.ndarray:
        for attempt in range(2):
            try:
                parts = []
                for item in self.pipe(text, voice=self.voice, speed=speed, split_pattern=None):
                    audio = item.audio if hasattr(item, "audio") else item[2]
                    if audio is None:
                        continue
                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()
                    parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
                if not parts:
                    raise RuntimeError("Kokoro returned no audio")
                return np.concatenate(parts)
            except Exception as exc:
                if attempt == 0 and self.device != "cpu":
                    log(f"Kokoro failed on {self.device} ({exc}); retrying on CPU")
                    self.device = "cpu"
                    self.pipe = self._build("cpu")
                    continue
                raise

    def close(self) -> None:
        del self.pipe


def _synth_chapter(k: _Kokoro, narration: str, speed: float) -> np.ndarray:
    paragraphs = [p for p in re.split(r"\n\s*\n", narration) if p.strip()]
    out: List[np.ndarray] = []
    for pi, para in enumerate(paragraphs):
        chunks = _chunks(_tts_clean(para))
        for ci, chunk in enumerate(chunks):
            out.append(_fade_edges(k.synth(chunk, speed)))
            if ci < len(chunks) - 1:
                out.append(_silence(SENTENCE_GROUP_PAUSE))
        if pi < len(paragraphs) - 1:
            out.append(_silence(PARAGRAPH_PAUSE))
    return np.concatenate(out) if out else _silence(1.0)


# --------------------------------------------------------------------------- timeline
def build_timeline(titles: List[str], durations: List[float], words: List[int],
                   lead_in: float = LEAD_IN_SECONDS, min_total: float = MIN_TOTAL_SECONDS) -> Dict:
    t = lead_in
    chapters = []
    for i, (title, d, w) in enumerate(zip(titles, durations, words)):
        chapters.append({"index": i, "title": title, "voice_start": round(t, 3), "voice_end": round(t + d, 3), "words": w})
        t += d + CHAPTER_PAUSE
    voice_end = chapters[-1]["voice_end"]
    total = voice_end + OUTRO_SECONDS
    if total < min_total:
        log(f"WARNING: narration is short; extending the ambient outro by {min_total - total:.0f}s to reach 10+ minutes.")
        total = min_total
    return {
        "sample_rate": KOKORO_SR,
        "lead_in": lead_in,
        "chapters": chapters,
        "voice_end": voice_end,
        "outro": round(total - voice_end, 3),
        "total_duration": round(total, 3),
    }


# --------------------------------------------------------------------------- public entry
def generate_voiceover(cfg: config.Settings, script: Dict, cache: Path) -> Dict:
    audio_dir = cache / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    chapters = script["chapters"]
    paths = [audio_dir / f"chapter_{i + 1:02d}.wav" for i in range(len(chapters))]
    missing = [i for i, p in enumerate(paths) if not p.exists()]

    if missing:
        log(f"Loading Kokoro voice '{cfg.voice}' ...")
        k = _Kokoro(cfg.voice)
        try:
            for i in missing:
                log(f"Synthesizing chapter {i + 1}/{len(chapters)}: {chapters[i]['title']}")
                audio = _synth_chapter(k, chapters[i]["narration"], cfg.speed)
                tmp = paths[i].with_suffix(".tmp.wav")
                sf.write(tmp, audio, KOKORO_SR, subtype="FLOAT")
                tmp.replace(paths[i])
                log(f"  -> {len(audio) / KOKORO_SR:.1f}s")
        finally:
            k.close()
            del k
            free_memory()
    else:
        log("All chapter audio found in cache - skipping TTS")

    durations = [sf.info(str(p)).frames / KOKORO_SR for p in paths]
    words = [len(c["narration"].split()) for c in chapters]
    timeline = build_timeline([c["title"] for c in chapters], durations, words)

    pieces: List[np.ndarray] = [_silence(LEAD_IN_SECONDS)]
    for i, p in enumerate(paths):
        data, sr = sf.read(str(p), dtype="float32")
        assert sr == KOKORO_SR
        pieces.append(data if data.ndim == 1 else data.mean(axis=1))
        if i < len(paths) - 1:
            pieces.append(_silence(CHAPTER_PAUSE))
    sf.write(cache / "voiceover.wav", np.concatenate(pieces), KOKORO_SR, subtype="FLOAT")
    write_json(cache / "timeline.json", timeline)
    log(f"Voiceover: {timeline['voice_end']:.0f}s of speech, total video length {timeline['total_duration']:.0f}s")
    return timeline
