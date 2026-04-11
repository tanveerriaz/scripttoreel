## Learned User Preferences

- When the user asks for a LinkedIn post, close with this tagline: Curious mind. Builder mode. 🇸🇬
- For this user’s workflow, script generation is typically via OpenRouter (not Ollama); keep explanations and copy aligned with that while noting Ollama remains supported in the project.
- For the Flask dashboard header, prefers a compact logo with the ScriptToReel wordmark beside it rather than a large icon alone; when they supply Gemini-generated brand specs, align colors and layout to those specs.
- TTS and voiceover: speaker voice and gender should match user intent; they have flagged an unwanted male-sounding default.

## Learned Workspace Facts

- Public GitHub remote for this repo: https://github.com/tanveerriaz/scripttoreel.git (remote name `origin`, default branch `main`; GitHub repo name is lowercase `scripttoreel`).
- Current pipeline stack includes local SDXL image generation (`stabilityai/stable-diffusion-xl-base-1.0`) and an updated local-first TTS chain prioritizing `kokoro-onnx`, then `edge-tts`, `piper`, and macOS `say` fallback.
- The Flask dashboard (`server.py`) is served at http://localhost:8080 by default (port 8080, binds `0.0.0.0`).
