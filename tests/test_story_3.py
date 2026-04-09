"""
MVP 3 — Module 3: Script Generation & Voiceover tests.

Covers Stories 3.1–3.4:
  3.1 Ollama connection + script generation
  3.2 Script segment validation
  3.3 TTS voiceover generation (macOS say fallback)
  3.4 Voiceover concatenation + script.json output
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module_3_script_voiceover import ScriptModule, OllamaNotAvailableError
from src.utils.json_schemas import Script, ScriptSegment, SegmentType, Mood


# API keys with OpenRouter configured for tests.
_API_KEYS_OLLAMA_ONLY = {
    "PEXELS_API_KEY": None,
    "PIXABAY_API_KEY": None,
    "UNSPLASH_ACCESS_KEY": None,
    "FREESOUND_API_KEY": None,
    "OPENROUTER_API_KEY": "fake-test-key",
    "OPENROUTER_MODEL": "anthropic/claude-sonnet-4-5",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SCRIPT_JSON = {
    "title": "Haunted Places in Pakistan",
    "topic": "Haunted Places in Pakistan",
    "duration_sec": 120,
    "mood": "mysterious",
    "visual_style": "documentary",
    "color_palette": ["#1a1a2e", "#16213e", "#0f3460"],
    "segments": [
        {
            "id": 1,
            "type": "intro",
            "text": "Pakistan holds many dark secrets, whispered tales of the supernatural.",
            "duration_sec": 15,
            "visual_cues": ["misty mountains", "old ruins"],
            "mood_tags": ["mysterious"],
            "b_roll_keywords": ["haunted", "ancient ruins", "pakistan"],
            "sfx_cues": ["wind"],
            "music_cues": ["ambient drone"],
            "transitions": {"in": "fade_in", "out": "dissolve"},
            "text_overlay": {"enabled": True, "text": "Haunted Places in Pakistan", "position": "bottom_third", "style": "lower_third"},
        },
        {
            "id": 2,
            "type": "narration",
            "text": "The ruins of Mohenjo-daro have long been shrouded in mystery.",
            "duration_sec": 30,
            "visual_cues": ["ancient ruins", "stone walls"],
            "mood_tags": ["dark"],
            "b_roll_keywords": ["mohenjo-daro", "ruins", "ancient"],
            "sfx_cues": ["distant whisper"],
            "music_cues": ["low strings"],
            "transitions": {"in": "dissolve", "out": "dissolve"},
            "text_overlay": {"enabled": False, "text": "", "position": "bottom_third", "style": "lower_third"},
        },
        {
            "id": 3,
            "type": "outro",
            "text": "These places remind us that history leaves echoes that never truly fade.",
            "duration_sec": 75,
            "visual_cues": ["sunset over ruins"],
            "mood_tags": ["mysterious"],
            "b_roll_keywords": ["ruins sunset", "ancient"],
            "sfx_cues": [],
            "music_cues": ["fade out"],
            "transitions": {"in": "dissolve", "out": "fade_out"},
            "text_overlay": {"enabled": False, "text": "", "position": "bottom_third", "style": "lower_third"},
        },
    ],
    "background_music_style": "ambient",
    "overall_pacing": "slow",
}


@pytest.fixture
def tmp_project(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "assets" / "audio").mkdir(parents=True)
    # Write minimal project.json
    from datetime import datetime, timezone
    from src.utils.json_schemas import ProjectMetadata, ProjectPipeline
    meta = ProjectMetadata(
        project_id="test",
        topic="Haunted Places in Pakistan",
        duration_min=2,
        duration_sec=120,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        project_dir=str(project_dir),
    )
    (project_dir / "project.json").write_text(
        json.dumps(meta.model_dump(), default=str)
    )
    return project_dir


def _make_fake_wav(path: Path, duration_sec: float = 2.0):
    """Write a minimal valid WAV file."""
    sr = 22050
    n = int(sr * duration_sec)
    samples = (np.sin(2 * np.pi * 220 * np.arange(n) / sr) * 16000).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# Story 3.1 — Ollama request format + error handling
# ---------------------------------------------------------------------------

def test_ollama_request_format(tmp_project):
    """The module must call the LLM and parse the resulting JSON into a Script."""
    module = ScriptModule(tmp_project, api_keys=_API_KEYS_OLLAMA_ONLY)

    with patch("src.module_3_script_voiceover.call_llm", return_value=json_to_str(SAMPLE_SCRIPT_JSON)):
        script = module.generate_script_ollama("Haunted Places in Pakistan", 2)

    assert script is not None
    assert script.title == "Haunted Places in Pakistan"


def test_ollama_unreachable_raises_clear_error(tmp_project):
    module = ScriptModule(tmp_project, api_keys=_API_KEYS_OLLAMA_ONLY)
    import requests as req
    with patch("src.module_3_script_voiceover.call_llm", side_effect=req.ConnectionError("refused")):
        with pytest.raises(Exception):
            module.generate_script_ollama("test topic", 1)


def test_script_json_parses_to_pydantic(tmp_project):
    module = ScriptModule(tmp_project)
    script = module.parse_script_json(json.dumps(SAMPLE_SCRIPT_JSON))
    assert isinstance(script, Script)
    assert script.title == "Haunted Places in Pakistan"
    assert len(script.segments) == 3


def test_retry_on_bad_json(tmp_project):
    """LLM returns invalid JSON twice; succeeds on 3rd attempt."""
    module = ScriptModule(tmp_project, api_keys=_API_KEYS_OLLAMA_ONLY)
    responses = [
        "not json at all {{",
        "{broken",
        json_to_str(SAMPLE_SCRIPT_JSON),
    ]
    with patch("src.module_3_script_voiceover.call_llm", side_effect=responses):
        script = module.generate_script_ollama("test", 2)
    assert script is not None


# ---------------------------------------------------------------------------
# Story 3.2 — Segment validation
# ---------------------------------------------------------------------------

def test_segment_durations_sum_to_target(tmp_project):
    module = ScriptModule(tmp_project)
    script = module.parse_script_json(json.dumps(SAMPLE_SCRIPT_JSON))
    total = sum(s.duration_sec for s in script.segments)
    assert abs(total - script.duration_sec) <= script.duration_sec * 0.15


def test_first_segment_is_intro(tmp_project):
    module = ScriptModule(tmp_project)
    script = module.parse_script_json(json.dumps(SAMPLE_SCRIPT_JSON))
    assert script.segments[0].type == SegmentType.INTRO


def test_last_segment_is_outro(tmp_project):
    module = ScriptModule(tmp_project)
    script = module.parse_script_json(json.dumps(SAMPLE_SCRIPT_JSON))
    assert script.segments[-1].type == SegmentType.OUTRO


def test_no_empty_segment_text(tmp_project):
    module = ScriptModule(tmp_project)
    script = module.parse_script_json(json.dumps(SAMPLE_SCRIPT_JSON))
    for seg in script.segments:
        assert seg.text.strip() != ""


# ---------------------------------------------------------------------------
# Story 3.3 — TTS via macOS `say` fallback
# ---------------------------------------------------------------------------

def test_macos_say_fallback_produces_wav(tmp_project):
    """The macOS `say` fallback should produce a valid WAV file."""
    module = ScriptModule(tmp_project)
    out = tmp_project / "assets" / "audio" / "test_seg.wav"
    module._macos_say_fallback("Hello world, this is a test.", out)
    assert out.exists()
    assert out.stat().st_size > 1000, "WAV file too small"


def test_wav_file_is_valid(tmp_project):
    module = ScriptModule(tmp_project)
    out = tmp_project / "assets" / "audio" / "valid_test.wav"
    module._macos_say_fallback("Testing audio output.", out)
    # soundfile or wave should be able to read it
    with wave.open(str(out)) as wf:
        assert wf.getnframes() > 0


# ---------------------------------------------------------------------------
# Story 3.4 — Concatenation + script.json
# ---------------------------------------------------------------------------

def test_concatenation_produces_longer_audio(tmp_project):
    """Concatenated WAV from 3 × 2s segments should be ≥ 5s total."""
    from pydub import AudioSegment as PydubSegment

    seg_paths = []
    for i in range(3):
        p = tmp_project / "assets" / "audio" / f"seg_{i}.wav"
        _make_fake_wav(p, duration_sec=2.0)
        seg_paths.append(p)

    module = ScriptModule(tmp_project)
    out = tmp_project / "assets" / "audio" / "concat.wav"
    module._concatenate_wavs(seg_paths, out, pause_ms=300)

    combined = PydubSegment.from_wav(str(out))
    assert combined.duration_seconds >= 5.5  # 3×2s + 2×0.3s pause


def test_script_json_written(tmp_project):
    module = ScriptModule(tmp_project)
    script = module.parse_script_json(json.dumps(SAMPLE_SCRIPT_JSON))

    # Attach fake voiceover paths
    for seg in script.segments:
        p = tmp_project / "assets" / "audio" / f"vo_{seg.id}.wav"
        _make_fake_wav(p, 2.0)
        seg.voiceover_path = str(p)
    script.total_voiceover_path = str(tmp_project / "assets" / "audio" / "voiceover.wav")

    module.save_script(script)

    out = tmp_project / "script.json"
    assert out.exists()
    raw = json.loads(out.read_text())
    loaded = Script(**raw)
    assert loaded.title == "Haunted Places in Pakistan"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def json_to_str(d: dict) -> str:
    return json.dumps(d)


def _mock_ollama_resp(text: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": text}
    return resp
