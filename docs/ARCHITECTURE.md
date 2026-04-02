# Architecture

How ScriptToReel is wired: one CLI entry point, six pipeline stages, JSON files as the contract between steps.

## High-level flow

```
Topic + duration
       │
       ▼
┌──────────────┐    assets_raw.json
│ Module 1     │◄── Pexels, Pixabay, Unsplash, Freesound (optional keys)
│ Research     │
└──────┬───────┘
       ▼
┌──────────────┐    assets.json
│ Module 2     │◄── ffprobe, Pillow, librosa, OpenCV
│ Metadata     │
└──────┬───────┘
       ▼
┌──────────────┐    script.json, voiceover.wav
│ Module 3     │◄── Ollama or OpenRouter; Coqui TTS or macOS `say`
│ Script+TTS   │
└──────┬───────┘
       ▼
┌──────────────┐    orchestration.json  ← edit here before render
│ Module 4     │
│ Orchestration│
└──────┬───────┘
       ▼
┌──────────────┐    output/final_video.mp4
│ Module 5     │◄── FFmpeg (VideoToolbox or libx264)
│ Render       │
└──────┬───────┘
       ▼
┌──────────────┐    validation_report.json
│ Module 6     │
│ Validation   │
└──────────────┘
```

## Repository layout

| Path | Role |
|------|------|
| `main.py` | Click CLI: `--init`, `--run`, `--module`, `--status`, `--validate` |
| `server.py` | Optional Flask UI on port 8080 (`dashboard.html`) |
| `src/project_manager.py` | Creates/loads `project.json`, updates pipeline status |
| `src/module_1_research.py` … `module_6_validation.py` | One class per stage, each implements `run()` |
| `src/utils/json_schemas.py` | Pydantic models for every JSON artifact |
| `src/utils/api_handlers.py` | HTTP clients for stock media APIs |
| `src/utils/config_loader.py` | YAML prompts/presets + `api_keys.env` via dotenv |
| `src/utils/ffmpeg_builder.py` | Fluent helper for FFmpeg command lines |
| `config/ollama_prompts.yaml` | LLM system/user templates |
| `config/ffmpeg_presets.yaml` | Codecs, transitions, color grades, audio levels |
| `config/api_keys.env` | Local secrets (gitignored); use `api_keys.env.example` |
| `tests/` | Pytest; mocks for network, Ollama, ffprobe, TTS |

## Data contracts

Each module reads/writes files under `projects/<project_id>/`. Schemas live in `json_schemas.py`; invalid data fails fast with validation errors.

## Optional: web dashboard

`server.py` shells out to `main.py` for jobs. It does not replace the CLI; it is a convenience layer for the same pipeline.

## Further reading

- `README.md` — install, first run, troubleshooting  
- `PLAN.md` — original MVP stories and acceptance criteria (developer-oriented)
