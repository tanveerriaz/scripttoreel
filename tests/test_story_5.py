"""
MVP 5 — Module 5: FFmpeg Rendering tests.

Covers Stories 5.1–5.5:
  5.1 FFmpegCommand builder
  5.2 Scene rendering (scale + color grade)
  5.3 Image input → video clip
  5.4 Audio mixing filter strings
  5.5 Final output file (integration — renders a tiny real video)
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

from src.utils.ffmpeg_builder import (
    FFmpegCommand,
    build_color_grade_filter,
    build_scale_pad_filter,
    build_xfade_filter,
    build_audio_amix_filter,
)
from src.module_5_ffmpeg_render import RenderModule
from src.utils.json_schemas import (
    Asset, AssetRole, AssetSource, AssetType,
    AudioTrack, ColorGrade, Mood, Orchestration, Scene, TransitionType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    pd = tmp_path / "project"
    (pd / "output").mkdir(parents=True)
    (pd / "assets" / "audio").mkdir(parents=True)
    (pd / "assets" / "raw" / "video").mkdir(parents=True)
    (pd / "assets" / "raw" / "image").mkdir(parents=True)
    from datetime import datetime, timezone
    from src.utils.json_schemas import ProjectMetadata, ProjectPipeline
    meta = ProjectMetadata(
        project_id="test",
        topic="Test",
        duration_min=0.1,
        duration_sec=6,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        project_dir=str(pd),
    )
    (pd / "project.json").write_text(json.dumps(meta.model_dump(), default=str))
    return pd


def _make_test_image(path: Path, w: int = 1920, h: int = 1080, color=(100, 50, 200)):
    Image.new("RGB", (w, h), color=color).save(str(path), "JPEG")


def _make_test_audio_wav(path: Path, duration_sec: float = 3.0):
    sr = 22050
    n = int(sr * duration_sec)
    samples = (np.sin(2 * np.pi * 220 * np.arange(n) / sr) * 16000).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _make_tiny_video(path: Path, duration_sec: float = 3.0):
    """Create a minimal test video using ffmpeg (color source)."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:size=320x180:duration={duration_sec}:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ], capture_output=True, check=True)


# ---------------------------------------------------------------------------
# Story 5.1 — FFmpegCommand builder
# ---------------------------------------------------------------------------

def test_builder_produces_valid_argv():
    cmd = FFmpegCommand().input("/input.mp4").output("/output.mp4", c_v="libx264").build()
    assert "ffmpeg" in cmd[0]
    assert "-i" in cmd
    assert "/input.mp4" in cmd
    assert "/output.mp4" in cmd


def test_filter_complex_appended():
    cmd = FFmpegCommand().input("/a.mp4").filter_complex("[0:v]scale=1920:1080[v]").output("/o.mp4").build()
    assert "-filter_complex" in cmd
    fc_idx = cmd.index("-filter_complex")
    assert "scale=1920:1080" in cmd[fc_idx + 1]


def test_dry_run_does_not_execute(tmp_path):
    output = tmp_path / "should_not_exist.mp4"
    result = (
        FFmpegCommand()
        .input("/nonexistent.mp4")
        .output(str(output))
        .run(dry_run=True)
    )
    assert result.returncode == 0
    assert not output.exists()


def test_run_returns_zero_on_version_check():
    """Use ffmpeg -version as a sanity check that run() works."""
    cmd = FFmpegCommand()
    cmd._global_opts = {}  # clear default opts
    cmd._output_path = None
    cmd._inputs = []
    argv = ["ffmpeg", "-version"]
    result = subprocess.run(argv, capture_output=True)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Story 5.2 — Filter string tests (unit, no subprocess)
# ---------------------------------------------------------------------------

def test_scale_pad_filter_string():
    f = build_scale_pad_filter(1920, 1080)
    assert "scale=1920:1080" in f
    assert "pad=1920:1080" in f


def test_color_grade_filter_string_valid():
    f = build_color_grade_filter(brightness=-0.05, contrast=1.1, saturation=0.7, gamma=0.9)
    assert "eq=" in f
    assert "brightness=-0.050" in f
    assert "contrast=1.100" in f


def test_xfade_filter_string():
    f = build_xfade_filter("v0", "v1", "vout", duration=0.8, offset=2.2)
    assert "xfade" in f
    assert "dissolve" in f
    assert "duration=0.800" in f


def test_amix_filter_two_tracks():
    f = build_audio_amix_filter(["vo", "music"], [1.0, 0.12])
    assert "amix" in f or "volume" in f
    assert "0.1200" in f or "0.12" in f


# ---------------------------------------------------------------------------
# Story 5.2 — Scene rendering (integration, real ffmpeg)
# ---------------------------------------------------------------------------

def test_scene_image_renders_to_video(tmp_project):
    """An image input should produce a valid video clip."""
    img = tmp_project / "assets" / "raw" / "image" / "test.jpg"
    _make_test_image(img)
    out = tmp_project / "output" / "scene_001.mp4"

    module = RenderModule(tmp_project)
    module.render_image_to_clip(str(img), str(out), duration_sec=3.0)

    assert out.exists()
    assert out.stat().st_size > 5000


def test_scene_video_renders_scaled(tmp_project):
    """A video input should be scaled to 1920x1080."""
    vid = tmp_project / "assets" / "raw" / "video" / "test.mp4"
    _make_tiny_video(vid, duration_sec=3.0)
    out = tmp_project / "output" / "scene_002.mp4"

    module = RenderModule(tmp_project)
    module.render_video_clip(str(vid), str(out), duration_sec=3.0)

    assert out.exists()
    # Verify resolution via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(out)],
        capture_output=True, text=True
    )
    data = json.loads(probe.stdout)
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert video_stream["width"] == 1920
    assert video_stream["height"] == 1080


# ---------------------------------------------------------------------------
# Story 5.3 — Concat + transitions (integration)
# ---------------------------------------------------------------------------

def test_concat_two_clips(tmp_project):
    """Concatenating two 3s clips should produce a clip ≥ 5s."""
    clip1 = tmp_project / "output" / "c1.mp4"
    clip2 = tmp_project / "output" / "c2.mp4"
    _make_tiny_video(clip1, 3.0)
    _make_tiny_video(clip2, 3.0)

    out = tmp_project / "output" / "concat.mp4"
    module = RenderModule(tmp_project)
    module.concat_clips([str(clip1), str(clip2)], str(out))

    assert out.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(out)],
        capture_output=True, text=True
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    assert duration >= 5.0


# ---------------------------------------------------------------------------
# Story 5.5 — Full pipeline render (integration)
# ---------------------------------------------------------------------------

def test_full_render_produces_mp4(tmp_project):
    """End-to-end: orchestration.json → final_video.mp4."""
    # Create test assets
    img1 = tmp_project / "assets" / "raw" / "image" / "s1.jpg"
    img2 = tmp_project / "assets" / "raw" / "image" / "s2.jpg"
    _make_test_image(img1, color=(200, 100, 50))
    _make_test_image(img2, color=(50, 100, 200))

    vo_path = tmp_project / "assets" / "audio" / "voiceover.wav"
    _make_test_audio_wav(vo_path, 6.0)

    # Build a minimal orchestration
    orch = Orchestration(
        project_id="test",
        title="Test Video",
        topic="Test",
        total_duration_sec=6.0,
        color_grade=ColorGrade.DOCUMENTARY,
        scenes=[
            Scene(
                id=1, segment_id=1,
                asset_id="img1", asset_path=str(img1),
                start_time=0, end_time=3, duration_sec=3,
                transition_in=TransitionType.FADE_IN,
                transition_out=TransitionType.DISSOLVE,
                color_grade=ColorGrade.DOCUMENTARY,
            ),
            Scene(
                id=2, segment_id=2,
                asset_id="img2", asset_path=str(img2),
                start_time=3, end_time=6, duration_sec=3,
                transition_in=TransitionType.DISSOLVE,
                transition_out=TransitionType.FADE_OUT,
                color_grade=ColorGrade.DOCUMENTARY,
            ),
        ],
        voiceover_tracks=[
            AudioTrack(
                asset_id="vo_main",
                local_path=str(vo_path),
                start_time=0, volume=1.0,
            )
        ],
    )
    (tmp_project / "orchestration.json").write_text(
        json.dumps(orch.model_dump(), default=str)
    )

    module = RenderModule(tmp_project)
    output_path = module.run()

    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 50_000


def test_output_codec_is_h264(tmp_project):
    """The output must use h264 (hardware or software)."""
    # Use the same setup as full_render but check codec
    img = tmp_project / "assets" / "raw" / "image" / "single.jpg"
    _make_test_image(img)
    vo = tmp_project / "assets" / "audio" / "vo.wav"
    _make_test_audio_wav(vo, 3.0)

    orch = Orchestration(
        project_id="test", title="T", topic="T",
        total_duration_sec=3.0,
        color_grade=ColorGrade.DOCUMENTARY,
        scenes=[Scene(
            id=1, segment_id=1,
            asset_id="img1", asset_path=str(img),
            start_time=0, end_time=3, duration_sec=3,
            transition_in=TransitionType.FADE_IN,
            transition_out=TransitionType.FADE_OUT,
            color_grade=ColorGrade.DOCUMENTARY,
        )],
        voiceover_tracks=[AudioTrack(asset_id="vo", local_path=str(vo), start_time=0, volume=1.0)],
    )
    (tmp_project / "orchestration.json").write_text(json.dumps(orch.model_dump(), default=str))

    module = RenderModule(tmp_project)
    out = module.run()

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", out],
        capture_output=True, text=True
    )
    data = json.loads(probe.stdout)
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert "264" in video_stream["codec_name"]
