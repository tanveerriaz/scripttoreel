# ScriptToReel

Local AI video generation pipeline for Mac M4 Pro. Turns a topic string into a 1080p MP4 using Ollama (LLM), edge-tts/macOS TTS, stock footage APIs, and MoviePy v2.0 + FFmpeg VideoToolbox encoding.

## Quick Reference

```bash
# Run all tests
python3 -m pytest tests/ -v

# Init a project
python main.py --init --topic "Topic Here" --duration 5

# Run full pipeline
python main.py --run --project <project_id>

# Run single module (1-6)
python main.py --module N --project <project_id>

# Check status
python main.py --status --project <project_id>
```

## Architecture

6-module pipeline, each independently runnable. Data flows through JSON files in the project directory:

```
Module 1 (Research)       → assets_raw.json
Module 2 (Metadata)       → assets.json
Module 3 (Script+TTS)     → script.json + voiceover.wav
Module 4 (Orchestration)  → orchestration.json  ← human edit point
Module 5 (FFmpeg Render)  → output/final_video.mp4
Module 6 (Validation)     → validation_report.json
```

### Source Layout

```
main.py                          — Click CLI entry point
src/
  project_manager.py             — create/load/update project.json
  module_1_research.py           — API search + asset download (Pexels, Pixabay, Unsplash, Freesound)
  module_2_metadata.py           — ffprobe/Pillow/librosa/OpenCV metadata + quality scoring
  module_3_script_voiceover.py   — Ollama LLM script gen + edge-tts TTS (macOS say fallback)
                                   Includes _enhance_hook: prepends best hook from HookEngine to intro
  module_4_orchestration.py      — Asset-to-segment matching, timeline, transitions, audio mix plan
                                   Visual coherence scoring: dedup + color temperature smoothing
  module_5_ffmpeg_render.py      — MoviePy v2.0 scene render (Ken Burns, color grade, vignette,
                                   letterbox, CrossFadeIn); audio mix stays as ffmpeg subprocess
  module_6_validation.py         — 10 ffprobe checks, metadata embedding, validation report
  hook_engine.py                 — 12-pattern HookEngine (question/stat/controversy/story/…)
                                   LLM generation via OpenRouter/Ollama + template fallback
  ai_director.py                 — AI-driven scene timing and transition direction
  utils/
    json_schemas.py              — All Pydantic models (Asset, Script, Scene, Orchestration, etc.)
    api_handlers.py              — PexelsClient, PixabayClient, UnsplashClient, FreesoundClient
    config_loader.py             — YAML + dotenv loading
    ffmpeg_builder.py            — Fluent FFmpeg command builder (audio mixing only)
config/
  ffmpeg_presets.yaml            — Output specs, transitions, color grades, audio levels
  ollama_prompts.yaml            — LLM prompt templates for script gen + hook engine
  api_keys.env                   — API keys (gitignored)
```

### Key Patterns

- **Pydantic everywhere**: All JSON contracts between modules are Pydantic models in `src/utils/json_schemas.py`. Always validate data against these schemas.
- **Pipeline status**: Each module updates `project.json` pipeline status via `src/project_manager.py:update_pipeline_status()`.
- **Graceful degradation**: Missing API keys skip that source (no crash). Missing Coqui TTS falls back to macOS `say`. Missing VideoToolbox falls back to libx264.
- **Module classes**: Each module is a class (e.g., `ResearchModule`, `RenderModule`) with a `run()` method, instantiated with `project_dir: Path`.

## Config

- `config/ffmpeg_presets.yaml` — output codec/bitrate/resolution, transition filters, color grade EQ values, audio volume levels
- `config/ollama_prompts.yaml` — system/user prompt templates with `{topic}`, `{duration_sec}` placeholders
- `config/api_keys.env` — loaded via python-dotenv; missing keys return `None`

## Tests

191 tests across 13 test files:

```
tests/test_story_0_1.py        — Project init (slug, dirs, project.json)
tests/test_story_0_2.py        — Status command
tests/test_story_0_3.py        — Config loading
tests/test_story_1_1.py        — Pexels API client
tests/test_story_1_2_3.py      — Pixabay, Unsplash, Freesound clients
tests/test_story_1_4_5.py      — Asset download + assets_raw.json
tests/test_story_2.py          — Metadata extraction (ffprobe, Pillow, librosa, OpenCV)
tests/test_story_3.py          — Script generation + TTS voiceover
tests/test_story_4.py          — Scene orchestration
tests/test_story_5.py          — MoviePy rendering (scene clips, concat, full render)
tests/test_story_6.py          — Quality validation
tests/test_hook_engine.py      — HookEngine: all 12 patterns, LLM mock, fallback, hook_style
tests/test_visual_coherence.py — Color temperature scoring, dedup, coherence penalties
tests/test_production_plan.py  — ProductionPlanModule schema and parsing
```

Tests use mocks extensively for external services (API calls, Ollama, ffprobe, TTS). Fixtures create temp project directories.

## Dependencies

Python 3.9+, key packages: `click`, `rich`, `pydantic`, `requests`, `Pillow`, `opencv-python`, `librosa`, `pydub`, `soundfile`, `pyyaml`, `python-dotenv`, `moviepy>=2.0.0`, `numpy`, `edge-tts`.

External tools: `ffmpeg`/`ffprobe` (with VideoToolbox on macOS), `ollama` (local LLM server).

## Output Specs

1920x1080, 30fps, h264 (VideoToolbox HW / libx264 SW fallback), AAC 192kbps stereo, BT.709 colorspace, MP4 with faststart.
