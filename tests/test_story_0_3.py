"""
MVP 0, Story 0.3 — Configuration Loading tests.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_ollama_prompts, load_ffmpeg_presets, load_api_keys


def test_ollama_prompts_load():
    prompts = load_ollama_prompts()
    assert "script_generation" in prompts
    assert "system" in prompts["script_generation"]
    assert "user_template" in prompts["script_generation"]


def test_ffmpeg_presets_load():
    presets = load_ffmpeg_presets()
    assert "output" in presets
    assert "default" in presets["output"]
    assert presets["output"]["default"]["codec"] == "h264_videotoolbox"


def test_ffmpeg_presets_has_color_grades():
    presets = load_ffmpeg_presets()
    assert "color_grades" in presets
    assert "dark_mysterious" in presets["color_grades"]


def test_missing_api_key_returns_none(tmp_path, monkeypatch):
    """A missing key should return None, not raise KeyError."""
    # Point to a blank env file
    blank = tmp_path / "blank.env"
    blank.write_text("")
    # Clear any existing env vars that might be set
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    monkeypatch.delenv("FREESOUND_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("USE_OPENROUTER", raising=False)

    keys = load_api_keys(env_path=blank)
    assert keys["PEXELS_API_KEY"] is None
    assert keys["PIXABAY_API_KEY"] is None
    assert keys["OPENROUTER_API_KEY"] is None
    assert keys.get("USE_OPENROUTER") in (None, "")


def test_api_keys_has_ollama_defaults(tmp_path, monkeypatch):
    blank = tmp_path / "blank.env"
    blank.write_text("")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    keys = load_api_keys(env_path=blank)
    assert keys["OLLAMA_BASE_URL"] == "http://localhost:11434"
    assert keys["OLLAMA_MODEL"] == "llama3.2"
