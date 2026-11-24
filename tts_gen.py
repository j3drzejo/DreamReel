"""Voiceover: Dia-1.6B (GPU, one consistent calm male voice) or Kokoro-82M (CPU/GPU fallback).

Dia has no built-in voices: every generation would sound like a different person. So the app first CASTS a voice:
it generates several takes of a sample paragraph, measures pitch / pitch variation / pace, and keeps the take that is
lowest-pitched (male), steadiest and slowest-paced. That take is saved to assets/voice/dia_voice.wav (+ .txt) and is
used as the audio prompt for every chunk of every video, so the whole channel keeps the same narrator.
Drop your own dia_voice.wav + dia_voice.txt (exact transcript) there to use a different voice.

One WAV per chapter is cached for resume.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

import config
from config import (CHAPTER_PAUSE, KOKORO_SR, LEAD_IN_SECONDS, MIN_TOTAL_SECONDS, OUTRO_SECONDS,
                    PARAGRAPH_PAUSE, SENTENCE_GROUP_PAUSE, free_memory, log, read_json, write_json)

DIA_SR = 44100
VOICE_SAMPLE = ("Tonight, you step through a heavy wooden door, and the cold air of the evening falls away behind you. "
                "Take a slow breath, and let the day grow quiet. Everything here is calm, and everything is safe.")
CAST_CANDIDATES = 8
DIA_TEMPERATURE = 1.4      # Dia default is 1.8; lower = steadier, calmer delivery
DIA_CFG_SCALE = 3.0
DIA_TOP_P = 0.90
DIA_TOP_K = 45


# --------------------------------------------------------------------------- text prep
def _tts_clean(text: str) -> str:
    text = text.replace("&", " and ").replace("%", " percent").replace("…", "...")
    text = text.replace("[", "(").replace("]", ")")            # square brackets are Dia speaker/sound tags
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunks(paragraph: str, max_chars: int = 320, min_chars: int = 0) -> List[str]:
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
    if min_chars and len(chunks) > 1 and len(chunks[-1]) < min_chars and len(chunks[-2]) + 1 + len(chunks[-1]) <= max_chars * 1.3:
        chunks[-2] += " " + chunks.pop()
    return chunks


def _silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _fade_edges(x: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n > 0:
        x = x.copy()
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


# --------------------------------------------------------------------------- voice measurement
def trim_silence(x: np.ndarray, sr: int, thr_db: float = -42.0, margin: float = 0.06) -> np.ndarray:
    if len(x) == 0:
        return x
    peak = float(np.abs(x).max()) or 1.0
    win = max(1, int(0.01 * sr))
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, mode="same"))
    idx = np.where(env > peak * 10 ** (thr_db / 20))[0]
    if len(idx) == 0:
        return x
    a, b = max(0, idx[0] - int(margin * sr)), min(len(x), idx[-1] + int(margin * sr))
    return x[a:b]


def estimate_f0(x: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """(median f0 Hz, f0 spread in semitones, voiced fraction) by frame-wise normalized autocorrelation."""
    target = 16000
    g = math.gcd(sr, target)
    y = signal.resample_poly(x.astype(np.float64), target // g, sr // g) if sr != target else x.astype(np.float64)
    frame, hop = int(0.04 * target), int(0.01 * target)
    lo, hi = int(target / 400), int(target / 65)
    rms_all = np.sqrt(np.mean(y * y)) + 1e-9
    f0s, total = [], 0
    for s in range(0, len(y) - frame - hi, hop):
        seg = y[s:s + frame + hi]
        total += 1
        if np.sqrt(np.mean(seg[:frame] ** 2)) < 0.35 * rms_all:
            continue
        a = seg[:frame] - seg[:frame].mean()
        e0 = float(np.dot(a, a)) + 1e-9
        best_lag, best = 0, 0.0
        for lag in range(lo, hi):
            b = seg[lag:lag + frame] - seg[lag:lag + frame].mean()
            c = float(np.dot(a, b)) / math.sqrt(e0 * (float(np.dot(b, b)) + 1e-9))
            if c > best:
                best, best_lag = c, lag
        if best > 0.55 and best_lag:
            f0s.append(target / best_lag)
    if len(f0s) < 5:
        return 0.0, 99.0, 0.0
    f0s = np.array(f0s)
    med = float(np.median(f0s))
    keep = f0s[(f0s > med / 1.6) & (f0s < med * 1.6)]     # drop octave errors
    st = 12 * np.log2(keep / np.median(keep))
    return float(np.median(keep)), float(np.std(st)), len(f0s) / max(1, total)


def voice_score(x: np.ndarray, sr: int, text: str) -> Tuple[float, dict]:
    """Higher = more like a calm, low male narrator. -inf = reject."""
    dur = len(x) / sr
    f0, spread, voiced = estimate_f0(x, sr)
    cps = len(text) / max(dur, 1e-6)
    info = {"f0": f0, "spread": spread, "cps": cps, "dur": dur, "voiced": voiced}
    if not (6.0 <= dur <= 28.0) or voiced < 0.15 or not (65 <= f0 <= 175):
        return -math.inf, info
    score = -abs(f0 - 108) / 15 - 0.6 * max(0.0, spread - 2.5) - 0.4 * abs(cps - 12.5)
    return score, info


# --------------------------------------------------------------------------- Kokoro
class _Kokoro:
    name = "kokoro"
    sr = KOKORO_SR
    max_chars, min_chars, chunk_pause = 320, 0, SENTENCE_GROUP_PAUSE

    def __init__(self, voice: str, speed: float):
        self.voice, self.speed = voice, speed
        self.device = "cuda" if config.get_device() == "cuda" else "cpu"   # 82M model: CPU is fine, MPS is flaky
        self.pipe = self._build(self.device)

    def _build(self, device: str):
        from kokoro import KPipeline

        lang = self.voice[0]  # a = American English, b = British English
        try:
            return KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)
        except TypeError:
            return KPipeline(lang_code=lang)

    def speak(self, text: str) -> np.ndarray:
        for attempt in range(2):
            try:
                parts = []
                for item in self.pipe(text, voice=self.voice, speed=self.speed, split_pattern=None):
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
        self.pipe = None


# --------------------------------------------------------------------------- Dia
def voice_dir() -> Path:
    for base in (Path.cwd() / "assets" / "voice", Path(__file__).resolve().parent / "assets" / "voice"):
        if (base / "dia_voice.wav").is_file():
            return base
    return Path(__file__).resolve().parent / "assets" / "voice"


def _envelope(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(1)


def strip_audio_prompt(out: np.ndarray, anchor: np.ndarray, sr: int) -> np.ndarray:
    """Some Dia APIs return the audio prompt followed by the new speech. Remove the prompt if it is present."""
    if len(out) < len(anchor) + sr // 2:
        return out
    hop = int(0.05 * sr)
    a, b = _envelope(anchor, hop), _envelope(out[:len(anchor)], hop)
    n = min(len(a), len(b))
    if n > 4 and np.std(a[:n]) > 0 and np.std(b[:n]) > 0 and np.corrcoef(a[:n], b[:n])[0, 1] > 0.75:
        return out[len(anchor):]
    return out


class _Dia:
    name = "dia"
    sr = DIA_SR
    max_chars, min_chars, chunk_pause = 260, 110, 0.45

    def __init__(self, cfg: config.Settings):
        self.cfg = cfg
        self.device = config.get_device()
        self.api = ""
        self._n = 0
        self._load()
        self.sample = VOICE_SAMPLE
        self.anchor, self.anchor_path = self._get_voice()
        self.anchor_rms = float(np.sqrt(np.mean(self.anchor ** 2))) + 1e-9
        self.cps = len(self.sample) / (len(self.anchor) / self.sr)      # this narrator's characters per second

    # ---- model loading (Hugging Face transformers first, native `dia` package as alternative)
    def _load(self) -> None:
        import torch

        mid = self.cfg.dia_model
        cuda = self.device == "cuda"
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
        log(f"Loading Dia ({mid}) on {self.device} ...")
        errs = []
        try:
            from transformers import AutoProcessor, DiaForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(mid)
            try:
                self.model = DiaForConditionalGeneration.from_pretrained(mid, dtype=dtype)
            except TypeError:
                self.model = DiaForConditionalGeneration.from_pretrained(mid, torch_dtype=dtype)
            self.model = self.model.to(self.device).eval()
            self.api = "hf"
            return
        except Exception as exc:
            errs.append(f"transformers: {type(exc).__name__}: {exc}")
        try:
            from dia.model import Dia

            self.model = Dia.from_pretrained(mid, compute_dtype="bfloat16" if dtype == torch.bfloat16 else
                                             "float16" if cuda else "float32", device=torch.device(self.device))
            self.api = "native"
            return
        except Exception as exc:
            errs.append(f"dia package: {type(exc).__name__}: {exc}")
        raise RuntimeError("Dia could not be loaded (" + " | ".join(errs) + "). Update with `pip install -U transformers` "
                           "or `pip install git+https://github.com/nari-labs/dia.git`.")

    def _generate(self, text: str, audio: Optional[np.ndarray] = None, seed: Optional[int] = None) -> np.ndarray:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            if self.api == "hf":
                kw = {"text": [text], "padding": True, "return_tensors": "pt"}
                if audio is not None:
                    kw["audio"] = audio
                inputs = self.processor(**kw).to(self.device)
                plen = self.processor.get_audio_prompt_len(inputs["decoder_attention_mask"]) if audio is not None else None
                out = self.model.generate(**inputs, max_new_tokens=3072, guidance_scale=DIA_CFG_SCALE,
                                          temperature=DIA_TEMPERATURE, top_p=DIA_TOP_P, top_k=DIA_TOP_K)
                dec = self.processor.batch_decode(out, audio_prompt_len=plen) if plen is not None else self.processor.batch_decode(out)
                a = dec[0]
            else:
                a = self.model.generate(text, audio_prompt=(str(self.anchor_path) if audio is not None else None),
                                        use_torch_compile=False, verbose=False, cfg_scale=DIA_CFG_SCALE,
                                        temperature=DIA_TEMPERATURE, top_p=DIA_TOP_P, cfg_filter_top_k=DIA_TOP_K, max_tokens=3072)
        if hasattr(a, "detach"):
            a = a.detach().float().cpu().numpy()
        a = np.nan_to_num(np.asarray(a, dtype=np.float32))
        a = a if a.ndim == 1 else a.reshape(-1) if a.shape[0] == 1 else a.mean(axis=0 if a.shape[0] <= 2 else 1)
        if audio is not None and self.api == "native":
            a = strip_audio_prompt(a, audio, self.sr)
        return a

    # ---- voice casting
    def _get_voice(self) -> Tuple[np.ndarray, Path]:
        vdir = voice_dir()
        wav, txt = vdir / "dia_voice.wav", vdir / "dia_voice.txt"
        if wav.is_file() and not self.cfg.recast_voice:
            if not txt.is_file():
                raise RuntimeError(f"{wav} exists but {txt} is missing. Put the exact words spoken in the clip into that file.")
            x, fs = sf.read(str(wav), dtype="float32")
            x = x if x.ndim == 1 else x.mean(axis=1)
            if fs != self.sr:
                g = math.gcd(fs, self.sr)
                x = signal.resample_poly(x, self.sr // g, fs // g).astype(np.float32)
            self.sample = re.sub(r"^\s*\[S1\]\s*", "", txt.read_text(encoding="utf-8").strip())
            log(f"Using saved narrator voice {wav}")
            return x, wav
        return self._cast_voice(vdir)

    def _cast_voice(self, vdir: Path) -> Tuple[np.ndarray, Path]:
        log(f"Casting a calm male narrator: {CAST_CANDIDATES} takes, picking the lowest, steadiest one ...")
        (vdir / "candidates").mkdir(parents=True, exist_ok=True)
        best: Optional[Tuple[float, np.ndarray]] = None
        fallback: Optional[Tuple[float, np.ndarray]] = None
        for seed in range(CAST_CANDIDATES):
            try:
                a = trim_silence(self._generate("[S1] " + self.sample, None, seed=self.cfg.seed + seed), self.sr)
            except Exception as exc:
                log(f"  take {seed + 1}: failed ({type(exc).__name__}: {exc})")
                continue
            if len(a) < self.sr:
                continue
            a = a / (np.abs(a).max() + 1e-9) * 0.7
            sf.write(str(vdir / "candidates" / f"take_{seed + 1}.wav"), a, self.sr)
            score, info = voice_score(a, self.sr, self.sample)
            log(f"  take {seed + 1}: pitch {info['f0']:.0f} Hz, variation {info['spread']:.1f} st, "
                f"pace {info['cps']:.1f} ch/s -> {'rejected' if score == -math.inf else f'score {score:.2f}'}")
            if score > -math.inf and (best is None or score > best[0]):
                best = (score, a)
            if info["f0"] > 0 and (fallback is None or info["f0"] < fallback[0]):
                fallback = (info["f0"], a)
        if best is None and fallback is None:
            raise RuntimeError("Dia produced no usable voice sample")
        if best is None:
            log("  WARNING: no take sounded clearly like a low male voice; using the lowest-pitched one. "
                f"Listen to {vdir / 'candidates'} and copy your favourite to dia_voice.wav (+ dia_voice.txt).")
        voice = (best or (0, fallback[1]))[1]
        vdir.mkdir(parents=True, exist_ok=True)
        sf.write(str(vdir / "dia_voice.wav"), voice, self.sr)
        (vdir / "dia_voice.txt").write_text(self.sample, encoding="utf-8")
        log(f"  narrator voice saved to {vdir / 'dia_voice.wav'} (reused for every future video; --recast-voice to redo)")
        return voice, vdir / "dia_voice.wav"

    # ---- narration
    def speak(self, text: str) -> np.ndarray:
        full = f"[S1] {self.sample} [S1] {text}"
        expected = len(text) / self.cps
        best: Optional[Tuple[float, np.ndarray]] = None
        for attempt in range(4):
            self._n += 1
            a = trim_silence(self._generate(full, self.anchor, seed=self.cfg.seed + 7919 * self._n), self.sr)
            ratio = (len(a) / self.sr) / max(expected, 1e-6)
            off = abs(math.log(max(ratio, 1e-3)))
            if best is None or off < best[0]:
                best = (off, a)
            if 0.6 <= ratio <= 1.7:
                break
            log(f"    take {attempt + 1} sounds off ({len(a) / self.sr:.1f}s, expected about {expected:.1f}s); retrying")
        a = best[1]
        rms = float(np.sqrt(np.mean(a ** 2))) + 1e-9
        return (a * float(np.clip(self.anchor_rms / rms, 0.35, 2.8))).astype(np.float32)   # even level across chunks

    def close(self) -> None:
        self.model = None
        self.processor = None


# --------------------------------------------------------------------------- backend choice
def choose_backend_name(cfg: config.Settings) -> Tuple[str, str]:
    want = getattr(cfg, "tts", "auto")
    if want == "kokoro":
        return "kokoro", "requested"
    if want == "dia":
        return "dia", "requested"
    dev = config.get_device()
    if dev == "cuda" and config.usable_memory_gb(dev) >= 10:
        return "dia", f"CUDA GPU with {config.usable_memory_gb(dev):.0f} GB"
    return "kokoro", "no CUDA GPU with >= 10 GB (Dia needs one); use --tts dia to force it"


def backend_signature(cfg: config.Settings, name: str) -> str:
    return f"dia:{cfg.dia_model}" if name == "dia" else f"kokoro:{cfg.voice}:{cfg.speed}"


def open_backend(cfg: config.Settings):
    name, why = choose_backend_name(cfg)
    log(f"Voice engine: {name} ({why})")
    if name == "dia":
        try:
            return _Dia(cfg)
        except Exception as exc:
            free_memory()
            log(f"WARNING: Dia failed ({type(exc).__name__}: {exc}). Falling back to Kokoro for this run.")
    return _Kokoro(cfg.voice, cfg.speed)


# --------------------------------------------------------------------------- synthesis
def _synth_chapter(backend, narration: str):
    """Return (audio, paragraph_marks) where marks are [t0, t1] seconds of each paragraph inside the chapter audio."""
    sr = backend.sr
    paragraphs = [p for p in re.split(r"\n\s*\n", narration) if p.strip()]
    out: List[np.ndarray] = []
    marks: List[List[float]] = []
    pos = 0
    for pi, para in enumerate(paragraphs):
        t0 = pos
        chunks = _chunks(_tts_clean(para), backend.max_chars, backend.min_chars)
        for ci, chunk in enumerate(chunks):
            a = _fade_edges(backend.speak(chunk), sr)
            out.append(a)
            pos += len(a)
            if ci < len(chunks) - 1:
                out.append(_silence(backend.chunk_pause, sr))
                pos += len(out[-1])
        marks.append([round(t0 / sr, 3), round(pos / sr, 3)])
        if pi < len(paragraphs) - 1:
            out.append(_silence(PARAGRAPH_PAUSE, sr))
            pos += len(out[-1])
    return (np.concatenate(out) if out else _silence(1.0, sr)), marks


# --------------------------------------------------------------------------- timeline
def build_timeline(titles: List[str], durations: List[float], words: List[int],
                   lead_in: float = LEAD_IN_SECONDS, min_total: float = MIN_TOTAL_SECONDS, marks=None,
                   sample_rate: int = KOKORO_SR) -> Dict:
    t = lead_in
    chapters = []
    for i, (title, d, w) in enumerate(zip(titles, durations, words)):
        ent = {"index": i, "title": title, "voice_start": round(t, 3), "voice_end": round(t + d, 3), "words": w}
        if marks and marks[i]:
            ent["paragraphs"] = [[round(t + a, 3), round(t + b, 3)] for a, b in marks[i]]
        chapters.append(ent)
        t += d + CHAPTER_PAUSE
    voice_end = chapters[-1]["voice_end"]
    total = voice_end + OUTRO_SECONDS
    if total < min_total:
        log(f"WARNING: narration is short; extending the ambient outro by {min_total - total:.0f}s to reach 10+ minutes.")
        total = min_total
    return {
        "sample_rate": sample_rate,
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

    want_sig = backend_signature(cfg, choose_backend_name(cfg)[0])
    marker = audio_dir / ".engine"
    if marker.exists() and marker.read_text().strip() != want_sig:
        log("Voice engine changed since the last run: re-synthesizing the narration")
        for f in list(audio_dir.glob("chapter_*.wav")) + list(audio_dir.glob("chapter_*.json")):
            f.unlink()
    missing = [i for i, p in enumerate(paths) if not p.exists()]

    if missing:
        backend = open_backend(cfg)
        try:
            marker.write_text(backend_signature(cfg, backend.name))
            for i in missing:
                log(f"Synthesizing chapter {i + 1}/{len(chapters)}: {chapters[i]['title']}")
                audio, marks = _synth_chapter(backend, chapters[i]["narration"])
                write_json(paths[i].with_suffix(".json"), marks)
                tmp = paths[i].with_suffix(".tmp.wav")
                sf.write(tmp, audio, backend.sr, subtype="FLOAT")
                tmp.replace(paths[i])
                log(f"  -> {len(audio) / backend.sr:.1f}s")
        finally:
            backend.close()
            del backend
            free_memory()
    else:
        log("All chapter audio found in cache - skipping TTS")

    sr = sf.info(str(paths[0])).samplerate
    durations = [sf.info(str(p)).frames / sf.info(str(p)).samplerate for p in paths]
    words = [len(c["narration"].split()) for c in chapters]
    marks = []
    for p in paths:
        try:
            marks.append(read_json(p.with_suffix(".json")))
        except Exception:
            marks.append(None)
    timeline = build_timeline([c["title"] for c in chapters], durations, words, marks=marks, sample_rate=sr)

    pieces: List[np.ndarray] = [_silence(LEAD_IN_SECONDS, sr)]
    for i, p in enumerate(paths):
        data, fs = sf.read(str(p), dtype="float32")
        assert fs == sr
        pieces.append(data if data.ndim == 1 else data.mean(axis=1))
        if i < len(paths) - 1:
            pieces.append(_silence(CHAPTER_PAUSE, sr))
    sf.write(cache / "voiceover.wav", np.concatenate(pieces), sr, subtype="FLOAT")
    write_json(cache / "timeline.json", timeline)
    log(f"Voiceover: {timeline['voice_end']:.0f}s of speech, total video length {timeline['total_duration']:.0f}s")
    return timeline
