"""
MVP 6 — Module 6: Quality Validation tests.

Covers Stories 6.1–6.3:
  6.1 ffprobe-based checks (10 checks)
  6.2 Metadata embedding
  6.3 Validation report + exit code
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module_6_validation import ValidationModule
from src.utils.json_schemas import ValidationReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tiny_mp4(path: Path, duration: float = 3.0, w: int = 1920, h: int = 1080):
    """Create a real MP4 with correct specs for validation."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=white:size={w}x{h}:duration={duration}:rate=30",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:duration={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        str(path),
    ], capture_output=True, check=True)


def _make_bad_mp4(path: Path, duration: float = 3.0):
    """Create an MP4 with wrong resolution (non-1080p)."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=red:size=640x480:duration={duration}:rate=15",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ], capture_output=True, check=True)


@pytest.fixture
def tmp_project(tmp_path):
    pd = tmp_path / "project"
    (pd / "output").mkdir(parents=True)
    from datetime import datetime, timezone
    from src.utils.json_schemas import ProjectMetadata
    meta = ProjectMetadata(
        project_id="test",
        topic="Test Validation Topic",
        duration_min=0.05,
        duration_sec=3.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        project_dir=str(pd),
    )
    (pd / "project.json").write_text(json.dumps(meta.model_dump(), default=str))
    # Write minimal script.json
    (pd / "script.json").write_text(json.dumps({
        "title": "Test Video",
        "topic": "Test Validation Topic",
        "duration_sec": 3.0,
        "segments": [],
    }))
    return pd


# ---------------------------------------------------------------------------
# Story 6.1 — All 10 checks pass on a valid output
# ---------------------------------------------------------------------------

def test_all_checks_pass_on_valid_output(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4, duration=3.0)
    (tmp_project / "project.json")  # already exists

    module = ValidationModule(tmp_project)
    report = module.run()

    failed = [c for c in report.checks if not c.passed]
    assert report.passed, f"Checks failed: {[(c.name, c.message) for c in failed]}"


def test_codec_check_passes_h264(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    module = ValidationModule(tmp_project)
    probe = module.probe_output(str(mp4))
    check = module.check_codec(probe)
    assert check.passed


def test_resolution_check_passes_1080p(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    module = ValidationModule(tmp_project)
    probe = module.probe_output(str(mp4))
    check = module.check_resolution(probe)
    assert check.passed


def test_duration_check_passes_within_tolerance(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4, duration=3.0)
    module = ValidationModule(tmp_project)
    probe = module.probe_output(str(mp4))
    check = module.check_duration(probe, target_sec=3.0)
    assert check.passed


def test_audio_stream_check_passes(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    module = ValidationModule(tmp_project)
    probe = module.probe_output(str(mp4))
    check = module.check_audio_streams(probe)
    assert check.passed


def test_file_size_check_passes(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    module = ValidationModule(tmp_project)
    check = module.check_file_size(str(mp4))
    assert check.passed


def test_codec_check_fails_on_wrong_codec(tmp_project):
    """A file encoded with a different codec should fail the codec check."""
    mp4 = tmp_project / "output" / "bad_codec.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=green:size=320x240:duration=2:rate=30",
        "-c:v", "libx265", "-pix_fmt", "yuv420p",
        str(mp4),
    ], capture_output=True, check=True)
    module = ValidationModule(tmp_project)
    probe = module.probe_output(str(mp4))
    check = module.check_codec(probe)
    assert not check.passed


def test_resolution_check_fails_wrong_resolution(tmp_project):
    mp4 = tmp_project / "output" / "low_res.mp4"
    _make_bad_mp4(mp4)
    module = ValidationModule(tmp_project)
    probe = module.probe_output(str(mp4))
    check = module.check_resolution(probe)
    assert not check.passed


# ---------------------------------------------------------------------------
# Story 6.2 — Metadata embedding
# ---------------------------------------------------------------------------

def test_metadata_embedded_title(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    embedded = tmp_project / "output" / "with_meta.mp4"

    module = ValidationModule(tmp_project)
    module.embed_metadata(str(mp4), str(embedded), title="My Test Video", description="A test description")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(embedded)],
        capture_output=True, text=True
    )
    data = json.loads(probe.stdout)
    tags = data.get("format", {}).get("tags", {})
    assert tags.get("title") == "My Test Video"


def test_metadata_embedded_description(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    embedded = tmp_project / "output" / "with_meta2.mp4"

    module = ValidationModule(tmp_project)
    module.embed_metadata(str(mp4), str(embedded), title="T", description="Haunted places of Pakistan")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(embedded)],
        capture_output=True, text=True
    )
    data = json.loads(probe.stdout)
    tags = data.get("format", {}).get("tags", {})
    # description maps to 'comment' in MP4 container
    assert "haunted" in (tags.get("comment", "") + tags.get("description", "")).lower()


# ---------------------------------------------------------------------------
# Story 6.3 — Validation report
# ---------------------------------------------------------------------------

def test_report_json_valid_schema(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    module = ValidationModule(tmp_project)
    module.run()

    report_path = tmp_project / "validation_report.json"
    assert report_path.exists()
    raw = json.loads(report_path.read_text())
    loaded = ValidationReport(**raw)
    assert len(loaded.checks) >= 6


def test_report_all_pass_returns_passed_true(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    module = ValidationModule(tmp_project)
    report = module.run()
    assert report.passed is True


def test_pipeline_status_updated_to_complete(tmp_project):
    mp4 = tmp_project / "output" / "final_video.mp4"
    _make_tiny_mp4(mp4)
    module = ValidationModule(tmp_project)
    module.run()

    meta = json.loads((tmp_project / "project.json").read_text())
    assert meta["pipeline"]["module_6_validation"] == "complete"
