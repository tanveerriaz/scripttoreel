"""
Edge TTS voice selection tests — Story 3.3 extension.

Covers:
  - edge-tts is attempted first in generate_voiceover_segment
  - Per-segment voice is passed to edge-tts
  - Script narrator_voice propagates to segments without a voice set
  - Fallback chain: edge-tts fails → piper fails → macOS say
  - ScriptSegment.voice field accepted in schema
  - Script.narrator_voice and testimonial_voices fields accepted in schema
  - AVAILABLE_VOICES constant contains expected voices
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module_3_script_voiceover import (
    ScriptModule,
    AVAILABLE_VOICES,
    DEFAULT_NARRATOR_VOICE,
)
from src.utils.json_schemas import Script, ScriptSegment, SegmentType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "assets" / "audio").mkdir(parents=True)
    from datetime import datetime, timezone
    from src.utils.json_schemas import ProjectMetadata
    meta = ProjectMetadata(
        project_id="test-edge",
        topic="Test Topic",
        duration_min=1,
        duration_sec=60,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        project_dir=str(project_dir),
    )
    (project_dir / "project.json").write_text(
        json.dumps(meta.model_dump(), default=str)
    )
    return project_dir


def _make_fake_wav(path: Path, duration_sec: float = 1.0):
    sr = 22050
    n = int(sr * duration_sec)
    samples = (np.sin(2 * np.pi * 220 * np.arange(n) / sr) * 16000).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# AVAILABLE_VOICES constant
# ---------------------------------------------------------------------------

def test_available_voices_contains_required_voices():
    required = {
        "en-US-GuyNeural",
        "en-US-ChristopherNeural",
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-GB-RyanNeural",
        "en-GB-SoniaNeural",
    }
    assert required.issubset(set(AVAILABLE_VOICES))


def test_default_narrator_voice_is_aria_neural():
    assert DEFAULT_NARRATOR_VOICE == "en-US-AriaNeural"


# ---------------------------------------------------------------------------
# Schema: ScriptSegment.voice field
# ---------------------------------------------------------------------------

def test_script_segment_accepts_voice_field():
    seg = ScriptSegment(
        id=1,
        type=SegmentType.NARRATION,
        text="Hello world",
        duration_sec=5.0,
        voice="en-US-AriaNeural",
    )
    assert seg.voice == "en-US-AriaNeural"


def test_script_segment_voice_defaults_to_none():
    seg = ScriptSegment(id=1, type=SegmentType.NARRATION, text="Hello", duration_sec=5.0)
    assert seg.voice is None


# ---------------------------------------------------------------------------
# Schema: Script narrator_voice and testimonial_voices
# ---------------------------------------------------------------------------

def test_script_narrator_voice_default():
    script = Script(title="T", topic="T", duration_sec=60)
    assert script.narrator_voice == "en-US-AriaNeural"


def test_script_narrator_voice_customisable():
    script = Script(
        title="T", topic="T", duration_sec=60,
        narrator_voice="en-GB-RyanNeural",
    )
    assert script.narrator_voice == "en-GB-RyanNeural"


def test_script_testimonial_voices_default_empty():
    script = Script(title="T", topic="T", duration_sec=60)
    assert script.testimonial_voices == []


def test_script_testimonial_voices_stored():
    voices = ["en-US-JennyNeural", "en-US-AriaNeural"]
    script = Script(
        title="T", topic="T", duration_sec=60,
        testimonial_voices=voices,
    )
    assert script.testimonial_voices == voices


# ---------------------------------------------------------------------------
# edge-tts in the fallback chain (after Kokoro)
# ---------------------------------------------------------------------------

def test_edge_tts_called_first(tmp_project):
    """When Kokoro is unavailable, _edge_tts runs before piper or macOS say."""
    module = ScriptModule(tmp_project)
    call_order = []

    def fake_edge_tts(text, out_path, voice=DEFAULT_NARRATOR_VOICE):
        call_order.append("edge_tts")
        _make_fake_wav(out_path)

    def fake_piper(text, out_path):
        call_order.append("piper")
        _make_fake_wav(out_path)

    with (
        patch.object(module, "_kokoro_tts", side_effect=RuntimeError("skip kokoro")),
        patch.object(module, "_edge_tts", side_effect=fake_edge_tts),
        patch.object(module, "_piper_tts", side_effect=fake_piper),
    ):
        module.generate_voiceover_segment("Hello test.", 1)

    assert call_order == ["edge_tts"], "edge-tts should be the only engine called when it succeeds"


def test_edge_tts_receives_segment_voice(tmp_project):
    """The voice parameter must be forwarded to _edge_tts."""
    module = ScriptModule(tmp_project)
    received_voice = []

    def fake_edge_tts(text, out_path, voice=DEFAULT_NARRATOR_VOICE):
        received_voice.append(voice)
        _make_fake_wav(out_path)

    with (
        patch.object(module, "_kokoro_tts", side_effect=RuntimeError("skip kokoro")),
        patch.object(module, "_edge_tts", side_effect=fake_edge_tts),
    ):
        module.generate_voiceover_segment("Hello test.", 1, voice="en-GB-SoniaNeural")

    assert received_voice == ["en-GB-SoniaNeural"]


def test_default_voice_used_when_none_specified(tmp_project):
    """When no voice is passed, DEFAULT_NARRATOR_VOICE should be used."""
    module = ScriptModule(tmp_project)
    received_voice = []

    def fake_edge_tts(text, out_path, voice=DEFAULT_NARRATOR_VOICE):
        received_voice.append(voice)
        _make_fake_wav(out_path)

    with (
        patch.object(module, "_kokoro_tts", side_effect=RuntimeError("skip kokoro")),
        patch.object(module, "_edge_tts", side_effect=fake_edge_tts),
    ):
        module.generate_voiceover_segment("Hello test.", 1)

    assert received_voice == [DEFAULT_NARRATOR_VOICE]


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

def test_falls_back_to_piper_when_edge_tts_fails(tmp_project):
    """If edge-tts raises, piper must be attempted next."""
    module = ScriptModule(tmp_project)
    call_order = []

    def fake_edge_tts(text, out_path, voice=DEFAULT_NARRATOR_VOICE):
        call_order.append("edge_tts")
        raise RuntimeError("edge-tts unavailable")

    def fake_piper(text, out_path):
        call_order.append("piper")
        _make_fake_wav(out_path)

    with (
        patch.object(module, "_kokoro_tts", side_effect=RuntimeError("skip kokoro")),
        patch.object(module, "_edge_tts", side_effect=fake_edge_tts),
        patch.object(module, "_piper_tts", side_effect=fake_piper),
    ):
        out = module.generate_voiceover_segment("Hello test.", 2)

    assert call_order == ["edge_tts", "piper"]
    assert out.exists()


def test_falls_back_to_macos_say_when_piper_fails(tmp_project):
    """If both edge-tts and piper fail, macOS say must be attempted."""
    module = ScriptModule(tmp_project)
    call_order = []

    def fake_edge_tts(text, out_path, voice=DEFAULT_NARRATOR_VOICE):
        call_order.append("edge_tts")
        raise RuntimeError("edge-tts unavailable")

    def fake_piper(text, out_path):
        call_order.append("piper")
        raise FileNotFoundError("piper not found")

    def fake_say(text, out_path):
        call_order.append("macos_say")
        _make_fake_wav(out_path)

    with (
        patch.object(module, "_kokoro_tts", side_effect=RuntimeError("skip kokoro")),
        patch.object(module, "_edge_tts", side_effect=fake_edge_tts),
        patch.object(module, "_piper_tts", side_effect=fake_piper),
        patch.object(module, "_macos_say_fallback", side_effect=fake_say),
    ):
        out = module.generate_voiceover_segment("Hello test.", 3)

    assert call_order == ["edge_tts", "piper", "macos_say"]
    assert out.exists()


# ---------------------------------------------------------------------------
# generate_all_voiceovers — voice propagation
# ---------------------------------------------------------------------------

def test_narrator_voice_propagates_to_segments_without_voice(tmp_project):
    """Segments without a voice set should inherit script.narrator_voice."""
    module = ScriptModule(tmp_project)
    received_voices = []

    def fake_edge_tts(text, out_path, voice=DEFAULT_NARRATOR_VOICE):
        received_voices.append(voice)
        _make_fake_wav(out_path)

    script = Script(
        title="T", topic="T", duration_sec=30,
        narrator_voice="en-US-ChristopherNeural",
        segments=[
            ScriptSegment(id=1, type=SegmentType.NARRATION, text="Segment one", duration_sec=15),
            ScriptSegment(id=2, type=SegmentType.NARRATION, text="Segment two", duration_sec=15),
        ],
    )

    with (
        patch.object(module, "_kokoro_tts", side_effect=RuntimeError("skip kokoro")),
        patch.object(module, "_edge_tts", side_effect=fake_edge_tts),
    ):
        module.generate_all_voiceovers(script)

    assert received_voices == ["en-US-ChristopherNeural", "en-US-ChristopherNeural"]


def test_per_segment_voice_overrides_narrator_voice(tmp_project):
    """A segment with voice set should use that voice, not script.narrator_voice."""
    module = ScriptModule(tmp_project)
    received_voices = []

    def fake_edge_tts(text, out_path, voice=DEFAULT_NARRATOR_VOICE):
        received_voices.append(voice)
        _make_fake_wav(out_path)

    script = Script(
        title="T", topic="T", duration_sec=30,
        narrator_voice="en-US-GuyNeural",
        segments=[
            ScriptSegment(id=1, type=SegmentType.NARRATION, text="Narrator line", duration_sec=15),
            ScriptSegment(
                id=2, type=SegmentType.NARRATION, text="Testimonial line",
                duration_sec=15, voice="en-US-JennyNeural",
            ),
        ],
    )

    with (
        patch.object(module, "_kokoro_tts", side_effect=RuntimeError("skip kokoro")),
        patch.object(module, "_edge_tts", side_effect=fake_edge_tts),
    ):
        module.generate_all_voiceovers(script)

    assert received_voices[0] == "en-US-GuyNeural"    # narrator
    assert received_voices[1] == "en-US-JennyNeural"  # testimonial override


# ---------------------------------------------------------------------------
# edge-tts not installed — graceful error
# ---------------------------------------------------------------------------

def test_edge_tts_raises_when_not_installed(tmp_project):
    """_edge_tts should raise RuntimeError when edge_tts package is missing."""
    module = ScriptModule(tmp_project)
    out = tmp_project / "assets" / "audio" / "test.wav"

    with patch.dict("sys.modules", {"edge_tts": None}):
        with pytest.raises((RuntimeError, ImportError)):
            module._edge_tts("Hello", out)
