"""Configuration loading utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent  # repo root (ScriptToReel)


def _config_dir() -> Path:
    return _ROOT / "config"


def load_ollama_prompts() -> dict[str, Any]:
    path = _config_dir() / "ollama_prompts.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_ffmpeg_presets() -> dict[str, Any]:
    path = _config_dir() / "ffmpeg_presets.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_api_keys(env_path: Optional[Path] = None) -> dict[str, Optional[str]]:
    """Load API keys from config/api_keys.env. Missing keys return None."""
    path = env_path or (_config_dir() / "api_keys.env")
    if path.exists():
        load_dotenv(path, override=False)

    return {
        "PEXELS_API_KEY": os.environ.get("PEXELS_API_KEY"),
        "PIXABAY_API_KEY": os.environ.get("PIXABAY_API_KEY"),
        "UNSPLASH_ACCESS_KEY": os.environ.get("UNSPLASH_ACCESS_KEY"),
        "FREESOUND_API_KEY": os.environ.get("FREESOUND_API_KEY"),
        "OLLAMA_BASE_URL": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "llama3.2"),
        "USE_OPENROUTER": os.environ.get("USE_OPENROUTER", ""),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
        "OPENROUTER_MODEL": os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat"),
    }
