# DreamReel

### Local AI video generation for faceless channels

Turn a topic into a calm, long-form history or sleep documentary: a local language model writes the script, Kokoro creates the narration, a local image model paints the scenes, and FFmpeg renders the finished video. No hosted AI API or API key is required.

> **Models run locally, but are downloaded first.** The first full run fetches several gigabytes of model weights from Hugging Face. Check each model's terms before publishing or monetizing generated videos.

**Suggested GitHub repository name:** `dreamreel-local-ai-video-generator`  
**Suggested description:** `Generate faceless sleep and history videos locally with AI — scripts, narration, images, thumbnails, and FFmpeg rendering.`  
**Suggested topics:** `ai-video`, `local-ai`, `text-to-video`, `faceless-youtube`, `python`, `kokoro-tts`, `diffusers`

## What it makes

- 10+ minute, 1080p (1920×1080), 24 fps H.264/AAC videos
- AI-written history scripts and chapter plans using Qwen through Transformers
- Narration using Kokoro TTS, with selectable voices and speaking speed
- Historically themed images from local Diffusers models, with lower-memory fallbacks
- A synthesized fireplace ambience, mixed and mastered with the narration
- A title thumbnail and an `info.txt` containing a description, chapters, tags, and an AI-content disclosure reminder
- Resumable generation: completed stages are cached so an interrupted run can continue

The default image models need substantial memory. CUDA and Apple Silicon (MPS) are supported; CPU generation is possible but image generation can be very slow. Plan for **at least 25 GB of free disk space** for model downloads and **16 GB or more of RAM** for a practical setup.

## Requirements

- Python **3.10–3.12**
- FFmpeg (the `imageio-ffmpeg` package provides a fallback binary)
- `espeak-ng` is recommended for Kokoro phonemization
- A supported local model runtime; a compatible GPU is strongly recommended

## Install

Create and activate a virtual environment from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your hardware **before** the remaining dependencies. Use the command from [the official PyTorch selector](https://pytorch.org/get-started/locally/), then install the project requirements:

```bash
python -m pip install torch
python -m pip install -r requirements.txt
```

On macOS, install the optional system tools with Homebrew:

```bash
brew install ffmpeg espeak-ng
```

On Debian or Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg espeak-ng libsndfile1
```

On Windows PowerShell, create and activate the environment with:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install FFmpeg and eSpeak NG and reopen the terminal so they are on `PATH`. If you need CUDA, select the matching PyTorch command on the official PyTorch page rather than using the generic command above.

## Quick start

Generate a video about a specific topic:

```bash
python generate_video.py --topic "Life aboard an 18th-century sailing ship" --output-dir output
```

Or let DreamReel choose a topic not already found in its scanned project folders:

```bash
python generate_video.py --output-dir output
```

The first full run downloads the selected language and image models. Results are saved in a title-named folder:

```text
output/
└── <generated title>/
    ├── video.mp4
    ├── thumbnail.jpg
    └── info.txt
```

Try a short renderer/audio smoke test before a full generation:

```bash
python generate_video.py --smoke-test
```

This test uses placeholder images and does not load the language or image-generation models. It tries Kokoro for speech, which may download its smaller voice model on first use; if unavailable, it tries a system speech engine. It writes its test video and thumbnail to a temporary directory.

## Options

| Option | Description |
|---|---|
| `--topic TEXT` | Topic to use. Omit to choose a fresh topic automatically. |
| `--duration MINUTES` | Target duration (default 12; values below 10 are raised to 10). |
| `--output-dir PATH` | Output folder (default: current directory). |
| `--llm MODEL_ID` | Hugging Face language-model ID (default: choose based on available memory). |
| `--image-model MODEL_ID` | Hugging Face image-model ID (default: choose based on available memory). |
| `--voice VOICE` | Kokoro voice (default: `am_michael`; examples: `am_fenrir`, `bm_george`). |
| `--speed VALUE` | Narration speed (default: `0.92`). |
| `--fire-db VALUE` | Fireplace level below the voice during silence (default: `20` dB). |
| `--seed INTEGER` | Base seed for image and video motion generation (default: `1234`). |
| `--keep-cache` | Keep intermediate build files after success. |
| `--fresh` | Start a new project instead of resuming an unfinished one. |
| `--smoke-test` | Render a short test video without loading the LLM or image model. |

For all flags:

```bash
python generate_video.py --help
```

## How it works

1. Scans the current directory and output folders for previous projects, then generates a topic and chapter outline.
2. Writes narration chapter by chapter with a local instruction-tuned language model.
3. Synthesizes speech with Kokoro.
4. Generates scene images with a local Diffusers model. The pipeline can fall back to smaller image models if loading or generation fails.
5. Mixes narration with a procedural fireplace ambience (or `assets/fireplace.wav`, `.mp3`, `.flac`, `.ogg`, or `.m4a` if supplied).
6. Renders a slow-pan, cross-faded video and creates a separate thumbnail.

Intermediate files live in `.build_cache/` inside the project folder. Re-run the same command after an interruption to resume; use `--fresh` to start another project. Remove the cache after a successful run with the default settings, or retain it with `--keep-cache`.

## Troubleshooting

- **Out of memory:** try `--llm Qwen/Qwen3-4B-Instruct-2507 --image-model stabilityai/sdxl-turbo`, and close other GPU-heavy applications.
- **Image pipeline class/model unavailable:** update Diffusers using the version or install instructions required by that model. The generator attempts lighter model fallbacks.
- **Kokoro or phonemizer errors:** install `espeak-ng` and verify it is available on `PATH`.
- **FFmpeg not found:** install FFmpeg, reopen the terminal, or check that `imageio-ffmpeg` installed correctly.
- **Resume with changed model or voice settings:** use `--fresh` to avoid reusing intermediate files created with prior settings.
- **Slow generation:** model downloads happen once per cache location; CPU-only image generation may take a long time.

## Publishing and licensing

Generated videos may contain AI-generated narration and visuals. Review YouTube's current disclosure requirements and the licenses and usage terms for every model and asset you use, especially before monetization. DreamReel currently does not include a project `LICENSE` file; choose and add one before publishing the source if you want to grant others reuse rights.
