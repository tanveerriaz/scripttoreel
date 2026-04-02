<div align="center">

<img src="assets/branding/app-icon.svg" alt="ScriptToReel" width="96" height="96" />

# ScriptToReel

**Turn a topic into a 1080p MP4** — local AI pipeline on **macOS**: stock APIs, LLM script, TTS, and **FFmpeg** (VideoToolbox when available).

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![macOS](https://img.shields.io/badge/macOS-Apple_Silicon-000000?style=for-the-badge&logo=apple&logoColor=white)](#prerequisites)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-render-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local_LLM-111111?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)

[Architecture](docs/ARCHITECTURE.md) · [Dev plan](PLAN.md) · [Dashboard](#optional-web-dashboard)

</div>

---

**New here?** Install prerequisites → copy `config/api_keys.env.example` → run the [First-time checklist](#first-time-checklist) below.

**Branding:** App icon is `assets/branding/app-icon.svg` (public domain film reel from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Movie-reel.svg); see `assets/branding/README.md`).

---

## Features

- **End-to-end pipeline** — Six modules from research → metadata → script & voice → orchestration → render → validation.
- **Stock media** — Pexels, Pixabay, Unsplash, Freesound (all optional; graceful degradation without keys).
- **Flexible LLM** — **Ollama** locally or **OpenRouter** in the cloud; script generation with your chosen model.
- **Voiceover** — Coqui TTS with **macOS `say`** fallback when needed.
- **Pro export** — **1920×1080**, 30 fps, H.264 (**VideoToolbox** / **libx264** fallback), AAC stereo, BT.709 MP4.
- **Human in the loop** — Edit `orchestration.json` before re-render to tune scenes and timing.
- **Optional dashboard** — `python server.py` → **http://localhost:8080** over the same CLI pipeline.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **macOS** | Developed on Apple Silicon (M4); Intel Mac may work with the same tools. |
| **Python** | **3.12+** recommended (CI/dev tested on **3.14**). |
| **FFmpeg + ffprobe** | `brew install ffmpeg` — needed for metadata, render, validation. |
| **Ollama** | Default LLM for script generation: [ollama.com](https://ollama.com) — or use **OpenRouter** instead (see [API keys](#api-keys-all-free-tier)). |

Optional: free API keys for stock media (Pexels, Pixabay, Unsplash, Freesound). The pipeline still runs without them (placeholder visuals + voiceover).

---

## Installation

```bash
git clone https://github.com/tanveerriaz/scripttoreel.git
cd scripttoreel

# Recommended: virtual environment (avoids --break-system-packages)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Verify tools:

```bash
python main.py --help
ffmpeg -version
ollama --version   # if using Ollama
```

---

## First-time checklist

1. **API template** — `cp config/api_keys.env.example config/api_keys.env` and add any keys you want (all optional for a minimal run).
2. **LLM** — Either:
   - Run `ollama serve`, then `ollama pull llama3.2` (or set `OLLAMA_MODEL` in `api_keys.env`), **or**
   - Set `USE_OPENROUTER=true` and `OPENROUTER_API_KEY` in `api_keys.env` (see example file).
3. **Create a project** — `python main.py --init --topic "Your topic" --duration 5`  
   Note the printed **project ID** (slug from your topic).
4. **Run the pipeline** — `python main.py --run --project <project_id>`
5. **Output** — `projects/<project_id>/output/final_video.mp4`

Edit `projects/<project_id>/orchestration.json` before a re-render if you want to tweak scenes or timing (Module 4 output).

---

## How it works

Six **modules** run in order; each reads/writes JSON (and media) under `projects/<id>/`. Contracts are **Pydantic** models in `src/utils/json_schemas.py`.

For a diagram and file-by-file map, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.  
For the original story-style dev plan, see **PLAN.md** (optional for end users).

---

## Quick Start (reference)

```bash
# 1. Dependencies (see Installation)
pip install -r requirements.txt

# 2. Keys (optional)
cp config/api_keys.env.example config/api_keys.env

# 3. Ollama (if not using OpenRouter)
ollama pull llama3.2
ollama serve   # separate terminal

# 4. New project
python main.py --init --topic "Haunted Places in Pakistan" --duration 5

# 5. Full pipeline
python main.py --run --project haunted_places_in_pakistan

# 6. Status / validate
python main.py --status --project haunted_places_in_pakistan
python main.py --validate --project haunted_places_in_pakistan
```

Video path: `projects/<project_id>/output/final_video.mp4`

---

## Optional: web dashboard

```bash
source .venv/bin/activate   # if you use a venv
python server.py
```

Open **http://localhost:8080** — browser UI over the same CLI pipeline (`Flask` is listed in `requirements.txt`).

---

## API Keys (all free tier)

Copy the template: `cp config/api_keys.env.example config/api_keys.env`, then edit **`config/api_keys.env`** (gitignored — never commit it).

| Key | Where to get it |
|-----|----------------|
| `PEXELS_API_KEY` | https://www.pexels.com/api/ |
| `PIXABAY_API_KEY` | https://pixabay.com/api/docs/ |
| `UNSPLASH_ACCESS_KEY` | https://unsplash.com/developers |
| `FREESOUND_API_KEY` | https://freesound.org/apiv2/apply/ |

**OpenRouter** (cloud LLM instead of Ollama): set `USE_OPENROUTER=true`, `OPENROUTER_API_KEY`, and optionally `OPENROUTER_MODEL` in `api_keys.env`.

**No stock API keys** — Module 1 may yield few or no assets; later modules still produce a video (e.g. placeholders + TTS).

---

## CLI Reference

```
python main.py --init --topic "TOPIC" --duration MINUTES
    Create a new project directory

python main.py --run --project PROJECT_ID
    Run all 6 modules end-to-end

python main.py --module N --project PROJECT_ID
    Run a single module (1-6)

python main.py --status --project PROJECT_ID
    Show pipeline progress table

python main.py --validate --project PROJECT_ID
    Run Module 6 quality validation only
```

---

## Pipeline Modules

| Module | What it does |
|--------|-------------|
| **1 — Research** | Searches Pexels, Pixabay, Unsplash, Freesound → downloads assets → `assets_raw.json` |
| **2 — Metadata** | ffprobe + Pillow + librosa + OpenCV → enriches assets → `assets.json` |
| **3 — Script+TTS** | LLM script → Coqui TTS or macOS `say` → `script.json`, `voiceover.wav` |
| **4 — Orchestration** | Asset matching, timeline, transitions → `orchestration.json` (**human edit point**) |
| **5 — Render** | FFmpeg scene build, concat, audio mix → `output/final_video.mp4` |
| **6 — Validation** | ffprobe checks + report → `validation_report.json` |

---

## Project Directory Structure

```
projects/my_project/
├── project.json          ← metadata + pipeline status
├── assets_raw.json       ← Module 1
├── assets.json           ← Module 2
├── script.json           ← Module 3
├── orchestration.json    ← Module 4 — edit before re-render
├── validation_report.json
├── assets/
│   ├── raw/              ← video / image / audio (gitignored patterns; see .gitignore)
│   └── processed/
└── output/
    └── final_video.mp4
```

---

## Output Specs

- Resolution: 1920×1080  
- Frame rate: 30fps  
- Video: H.264 (VideoToolbox on Apple Silicon, **libx264** fallback)  
- Audio: AAC 192kbps stereo  
- Color: BT.709  
- Container: MP4, `faststart` for streaming  

---

## Troubleshooting

**Ollama not running**

```
OllamaNotAvailableError: Cannot connect to Ollama at http://localhost:11434
```

→ `ollama serve`, then `ollama pull llama3.2` — or enable OpenRouter in `api_keys.env`.

**TTS**

- Primary: Coqui TTS (pulls **torch** + model on first run, ~hundreds of MB).  
- Fallback: macOS `say`.

**No assets from APIs**

Add at least one stock key, or accept placeholder-heavy output from Module 5.

**VideoToolbox missing**

Pipeline falls back to **libx264**. Check: `ffmpeg -encoders | grep videotoolbox`

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```
