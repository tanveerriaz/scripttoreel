# VideoForge

A free, local, real-time AI video generation pipeline for Mac M4 Pro.

Turns a topic string into a fully rendered 1080p MP4 — using local Ollama (LLM), macOS TTS, stock footage APIs, and FFmpeg VideoToolbox hardware encoding.

---

## Quick Start

```bash
# 1. Install Python dependencies
pip3 install -r requirements.txt --break-system-packages

# 2. Set your API keys
cp config/api_keys.env config/api_keys.env   # already exists — edit it
# Add your free keys for Pexels, Pixabay, Unsplash, Freesound

# 3. Make sure Ollama is running with a model
ollama pull llama3.2
ollama serve  # keep this running in a separate terminal

# 4. Create a new project
python main.py --init --topic "Haunted Places in Pakistan" --duration 5

# 5. Run the full pipeline
python main.py --run --project haunted_places_in_pakistan

# 6. Check status any time
python main.py --status --project haunted_places_in_pakistan

# 7. Validate the output
python main.py --validate --project haunted_places_in_pakistan
```

The final video will be at: `projects/haunted_places_in_pakistan/output/final_video.mp4`

---

## API Keys (all free tier)

Edit `config/api_keys.env`:

| Key | Where to get it |
|-----|----------------|
| `PEXELS_API_KEY` | https://www.pexels.com/api/ |
| `PIXABAY_API_KEY` | https://pixabay.com/api/docs/ |
| `UNSPLASH_ACCESS_KEY` | https://unsplash.com/developers |
| `FREESOUND_API_KEY` | https://freesound.org/apiv2/apply/ |

**No keys needed to test the pipeline** — modules gracefully skip sources with missing keys and will still generate a video with whatever assets are available (including the TTS voiceover and color-generated clips).

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
| **2 — Metadata** | ffprobe + Pillow + librosa + OpenCV → enriches assets with quality scores → `assets.json` |
| **3 — Script+TTS** | Ollama LLM generates structured script → Coqui TTS (or macOS `say`) → `voiceover.wav` |
| **4 — Orchestration** | Maps assets to segments by mood/tags/duration → timeline + transitions + audio levels → `orchestration.json` |
| **5 — Render** | Parses orchestration.json → per-scene clips → concat → audio mix → final encode (h264_videotoolbox) |
| **6 — Validation** | 10 ffprobe checks + metadata embedding + rich console report → `validation_report.json` |

---

## Project Directory Structure

```
projects/my_project/
├── project.json          ← metadata + pipeline status
├── assets_raw.json       ← raw API search results (Module 1 output)
├── assets.json           ← enriched with metadata + quality scores (Module 2 output)
├── script.json           ← LLM-generated script with voiceover paths (Module 3 output)
├── orchestration.json    ← full scene/audio plan (Module 4 output) — edit before render!
├── validation_report.json
├── assets/
│   ├── raw/
│   │   ├── video/
│   │   ├── image/
│   │   └── audio/
│   └── processed/
└── output/
    └── final_video.mp4
```

---

## Output Specs

- Resolution: 1920×1080
- Frame rate: 30fps
- Video codec: h264 (VideoToolbox hardware on M4 Pro, libx264 fallback)
- Audio codec: AAC 192kbps stereo
- Color space: BT.709 (YouTube standard)
- Container: MP4 with `-movflags +faststart`

---

## Troubleshooting

**Ollama not running:**
```
OllamaNotAvailableError: Cannot connect to Ollama at http://localhost:11434
```
→ Run `ollama serve` in a separate terminal, then `ollama pull llama3.2`

**TTS voiceover quality:**
- Primary: Coqui TTS (`tts_models/en/ljspeech/tacotron2-DDC`) — requires `torch` + model download (~200MB on first run)
- Fallback: macOS `say -v Samantha` — always available, lower quality

**No assets downloaded (all API keys missing):**
Module 1 will produce an empty `assets_raw.json`. Module 5 will still render using color placeholder clips. Add at least one API key for real stock footage.

**FFmpeg VideoToolbox not available:**
VideoToolbox is detected automatically. If not found, falls back to `libx264` (software). Check with: `ffmpeg -encoders | grep videotoolbox`

---

## Running Tests

```bash
python3 -m pytest tests/ -v
# 102 tests across 7 MVPs
```

---

## Architecture Notes

- `orchestration.json` is the **edit point** — you can manually edit scene assignments, timing, or transitions before running Module 5
- All modules are independently runnable via `--module N`
- Pydantic schemas enforce data contracts between every module
- Module 3 retries Ollama up to 3× on bad JSON; falls back to macOS `say` for TTS
