# Sleep Documentary Generator

Generates a 10+ minute 1080p "what was life really like…" sleep/history documentary completely offline:
local LLM script → Kokoro voiceover → local diffusion images → procedural fireplace → Ken Burns video.

## 1. Setup

Requirements: **Python 3.10 – 3.12**, ~25 GB free disk (model downloads), 16 GB+ RAM/VRAM.

### macOS (Apple Silicon, MPS)
```bash
brew install ffmpeg espeak-ng
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch                      # default macOS wheel includes MPS
pip install -r requirements.txt
```

### Linux (NVIDIA CUDA)
```bash
sudo apt update && sudo apt install -y ffmpeg espeak-ng libsndfile1
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick your CUDA version at pytorch.org
pip install -r requirements.txt
```

### Windows (NVIDIA CUDA, PowerShell)
```powershell
winget install Gyan.FFmpeg
winget install eSpeak-NG.eSpeak-NG
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
Reopen the terminal after installing ffmpeg/espeak-ng so they are on `PATH`.

### Verify the install (no models, ~1 minute)
```bash
python generate_video.py --smoke-test
```
This speaks a short narration with your Kokoro voice (falls back to espeak-ng or macOS `say`), mixes it with the synthesized fireplace, and renders a ~35 s 1080p clip. Open the printed `smoke.mp4` path and listen: you should hear a calm male voice over a soft fireplace.

## 2. Run
```bash
python generate_video.py                                   # auto-picks a topic you haven't done yet
python generate_video.py --topic "Life aboard an 18th-century sailing ship"
python generate_video.py --topic "A night in a Viking longhouse" --duration 15
```
First run downloads the models from Hugging Face (several GB, one time). Output:

```
./<Video Title>/video.mp4     1920x1080, 24 fps, H.264 + AAC 192k, 10+ min
./<Video Title>/thumbnail.jpg 1280x720 thumbnail (title text is on the thumbnail only)
./<Video Title>/info.txt      title, description (with chapter timestamps + tags), duration, topic
```

### Options
| Flag | Meaning |
|---|---|
| `--topic` | Subject. Omit to auto-pick one not already covered by existing folders |
| `--duration` | Target minutes (min 10, default 12) |
| `--output-dir` | Where project folders go (default: current directory) |
| `--llm` | Override LLM (default: Qwen2.5-7B-Instruct if ≥16.5 GB usable, else Qwen3-4B-Instruct-2507) |
| `--image-model` | Override image model (default auto: Z-Image-Turbo → FLUX.2-klein-4B → SDXL-Turbo by memory) |
| `--voice` / `--speed` | Kokoro voice (`am_michael`, `am_fenrir`, `bm_george`) and pace (default 0.92) |
| `--fire-db` | Fireplace level below the voice in silence (default 20; 24 while speaking) |
| `--image-interval` | Seconds of video per new AI image (default 24). Every image is cut into 4–8 s shots with different framing and camera moves, so the picture changes every few seconds. Lower = more unique images but slower |
| `--notes FILE` | Your own trusted facts/sources about the topic (highest priority). Best way to maximise accuracy |
| `--tts auto\|dia\|kokoro` | Voice engine. `auto` = **Dia-1.6B** on a CUDA GPU with ≥10 GB, otherwise Kokoro |
| `--recast-voice` | Dia: pick a new narrator instead of reusing `assets/voice/dia_voice.wav` |
| `--device auto\|cuda\|mps\|cpu` | Force the compute device for every model |
| `--encoder auto\|nvenc\|x264` | Video encoder (auto = NVENC on NVIDIA GPUs when ffmpeg supports it) |
| `--image-size 1920x1080` | Native image size (default 1280x720); sharper but slower |
| `--keep-cache` | Keep `.build_cache/` (script, audio chunks, PNGs) after success |
| `--fresh` | Don't resume an unfinished project |
| `--smoke-test` | Model-free end-to-end test of mixer + renderer |

## 3. How it works
- **Duplicate prevention**: scans the working directory and `./output/` for `info.txt` (and unfinished projects), passes covered titles to the LLM, and re-asks if the new title is too similar. 20 built-in fallback topics seed the choice.
- **One model at a time**: LLM → Kokoro → diffusion pipeline. Each is deleted, `gc.collect()`ed and the CUDA/MPS cache emptied before the next loads.
- **Resume**: everything is cached in `<project>/.build_cache/` (outline, per-chapter text, per-chapter WAV, each PNG). Re-run the same command after a crash/Ctrl-C and it continues. Use `--fresh` to start over.
- **Fireplace**: drop your own `assets/fireplace.wav|mp3` next to the scripts to use it; otherwise a seamless procedural hearth (brown-noise rumble + randomized pops/ticks) is synthesized.
- **Mix**: voice normalized to −16 LUFS (peak ceiling −3 dB); fireplace sits −20 dB under it, dipping to −24 dB while the narrator speaks; 5 s fade-out.
- **Video**: every image is cut into 4–8 s shots (wide, close detail, pan, tilt, push-in, pull-out), smooth fractional-crop camera motion (no jitter), 1 s crossfades, warm vignette, **no on-screen text** (voiceover + images only), gradual darkening over the last 40% for sleep, fade in/out. Length is guaranteed ≥ 10:05 (the ambient outro is extended if narration is unexpectedly short).

## 4. Troubleshooting
- **`ZImagePipeline`/`Flux2…` not found or image model fails** → `pip install -U diffusers` (or `pip install git+https://github.com/huggingface/diffusers`). The app automatically falls back to SDXL-Turbo meanwhile.
- **Out of memory** → `--llm Qwen/Qwen3-4B-Instruct-2507 --image-model stabilityai/sdxl-turbo`; close other apps.
- **Kokoro errors about espeak** → install `espeak-ng` (see above).
- **Hugging Face downloads slow/blocked** → `pip install -U huggingface_hub`, or pre-download with `huggingface-cli download <model>`.
- **Changed voice/model after a partial run** → add `--fresh`.
- Check each model's license before monetizing videos (SDXL-Turbo uses Stability's community license; Qwen2.5-7B, Qwen3-4B, Kokoro are Apache-2.0). YouTube requires disclosure of realistic synthetic content; `info.txt` includes an AI-generated note.

## 5. Faceless-channel practices built in
- No subtitles or on-screen text in the video; only the thumbnail carries a title.
- Opening hook within the first ten words, then a short welcome (dim the screen, say where you're listening from).
- New image every ~25 s with slow motion, one locked painterly palette, no faces toward camera (consistent look, fewer AI face artifacts).
- Keyword-led titles, SEO description with chapter timestamps, tags and hashtags in `info.txt`; AI-content disclosure reminder.
- Publish on a steady schedule (e.g. weekly); use the same voice, style and thumbnail look so the channel feels like one brand.

## 6. Historical accuracy pipeline
1. The LLM names 4 relevant Wikipedia articles; the app downloads their text (needs internet for this step only; offline it continues and warns).
2. From those excerpts (plus your optional `--notes` file) it builds a **fact sheet**: period, what people did by time of day, clothing, buildings/tools/light, food, sensory details, uncertain points, and a list of **anachronisms** that must not appear.
3. The chapter outline and every narration prompt are built from the fact sheet; the narrator must hedge uncertain points, invent no names/dates/statistics, and teach real facts.
4. Any sentence that mentions an anachronism (e.g. clocks, potatoes, "minutes") is automatically removed.
5. Image prompts are assembled in code: period anchor first, then a concrete scene (who, doing what, where, with which objects), then period clothing/objects, with banned words scrubbed. The anchor is placed first so even SDXL's 77-token limit keeps it.
6. Next to `video.mp4` you get `research.md` (the fact sheet and sources) so you can fact-check before publishing, and `info.txt` lists the sources in the description.
A small local model can still make mistakes: skim `research.md` and the narration before uploading.

## 7. Dia narrator and GPU use
**Voice.** Dia-1.6B has no built-in voices, so the app *casts* one the first time: it generates 8 takes of a sample paragraph, measures pitch, pitch variation and pace, and keeps the lowest, steadiest male take. It is saved to `assets/voice/dia_voice.wav` (+ `dia_voice.txt`) and used as the audio prompt for every chunk of every video, so the channel keeps one narrator. All takes are kept in `assets/voice/candidates/`: listen, and copy your favourite over `dia_voice.wav` (put the exact spoken words in `dia_voice.txt`). Every Dia chunk is checked against the expected duration and regenerated (up to 4 takes) if it sounds truncated or rushed, and levels are matched between chunks. If Dia cannot load, the app falls back to Kokoro and says so.

**Where the GPU is used (CUDA):** LLM fully on the GPU (14B model with ≥34 GB, 7B with ≥17 GB, else 4B) · Dia narration · image model resident on the GPU with several images per batch (CPU offload only if VRAM is too small; falls back to smaller batches on out-of-memory) · frame compositing (zoom, crossfade, vignette, fades) via torch · H.264 encoding via NVENC. Audio mastering and the fireplace synthesis stay on the CPU because they are faster there. Apple Silicon (MPS) accelerates the LLM and images; CPU is the last resort and warns about long runtimes.

**Images.** Every image is generated from the narration passage being spoken (about every 5.5 s, `--image-interval`), so there are roughly 10 images per minute of video. Each image gets one gentle camera move; there are no repeated images with random moves.
