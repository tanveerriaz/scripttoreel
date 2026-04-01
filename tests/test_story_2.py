"""
MVP 2 — Module 2: Metadata Extraction tests.

Covers Stories 2.1–2.5:
  2.1 Video metadata via ffprobe
  2.2 Image metadata via Pillow
  2.3 Audio metadata via librosa
  2.4 Dominant color extraction via OpenCV
  2.5 Quality score + assets.json output
"""
from __future__ import annotations

import json
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module_2_metadata import MetadataModule
from src.utils.json_schemas import Asset, AssetSource, AssetType, AssetRole


# ---------------------------------------------------------------------------
# Fixtures — create minimal test files on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def test_image(tmp_path) -> Path:
    """A 1920x1080 red JPEG."""
    img = Image.new("RGB", (1920, 1080), color=(200, 30, 30))
    p = tmp_path / "test.jpg"
    img.save(str(p), "JPEG")
    return p


@pytest.fixture
def test_image_square(tmp_path) -> Path:
    img = Image.new("RGB", (500, 500), color=(0, 0, 255))
    p = tmp_path / "square.jpg"
    img.save(str(p), "JPEG")
    return p


@pytest.fixture
def test_audio(tmp_path) -> Path:
    """A 3-second 44100Hz mono WAV."""
    p = tmp_path / "test.wav"
    with wave.open(str(p), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        # 3 seconds of 440Hz sine wave
        samples = (np.sin(2 * np.pi * 440 * np.arange(44100 * 3) / 44100) * 32767).astype(np.int16)
        wf.writeframes(samples.tobytes())
    return p


@pytest.fixture
def red_image(tmp_path) -> Path:
    """A pure red 100x100 image for color extraction test."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    p = tmp_path / "red.jpg"
    img.save(str(p), "JPEG")
    return p


def _make_asset(asset_id: str, local_path: str, type_: AssetType) -> Asset:
    return Asset(
        id=asset_id,
        type=type_,
        role=AssetRole.B_ROLL,
        source=AssetSource.LOCAL,
        local_path=local_path,
    )


# ---------------------------------------------------------------------------
# Story 2.2 — Image metadata via Pillow
# ---------------------------------------------------------------------------

def test_image_metadata_dimensions(test_image):
    m = MetadataModule.__new__(MetadataModule)
    meta = m.extract_image_metadata(test_image)
    assert meta.width == 1920
    assert meta.height == 1080


def test_image_metadata_aspect_ratio(test_image):
    m = MetadataModule.__new__(MetadataModule)
    ratio = m._compute_aspect_ratio(1920, 1080)
    assert ratio == "16:9"


def test_image_metadata_aspect_ratio_square(test_image_square):
    m = MetadataModule.__new__(MetadataModule)
    ratio = m._compute_aspect_ratio(500, 500)
    assert ratio == "1:1"


def test_image_metadata_format(test_image):
    m = MetadataModule.__new__(MetadataModule)
    meta = m.extract_image_metadata(test_image)
    assert meta.format.upper() in ("JPEG", "JPG")


def test_image_missing_file_raises():
    m = MetadataModule.__new__(MetadataModule)
    with pytest.raises(FileNotFoundError):
        m.extract_image_metadata(Path("/nonexistent/file.jpg"))


# ---------------------------------------------------------------------------
# Story 2.3 — Audio metadata via librosa
# ---------------------------------------------------------------------------

def test_audio_duration(test_audio):
    m = MetadataModule.__new__(MetadataModule)
    meta = m.extract_audio_metadata(test_audio)
    assert abs(meta.duration_sec - 3.0) < 0.2


def test_audio_sample_rate(test_audio):
    m = MetadataModule.__new__(MetadataModule)
    meta = m.extract_audio_metadata(test_audio)
    assert meta.sample_rate == 44100


def test_audio_missing_file_raises():
    m = MetadataModule.__new__(MetadataModule)
    with pytest.raises(FileNotFoundError):
        m.extract_audio_metadata(Path("/nonexistent.wav"))


# ---------------------------------------------------------------------------
# Story 2.4 — Dominant color extraction
# ---------------------------------------------------------------------------

def test_dominant_colors_returns_5_hex(test_image):
    m = MetadataModule.__new__(MetadataModule)
    colors = m.extract_dominant_colors(test_image, n=5)
    assert len(colors) == 5
    for c in colors:
        assert c.startswith("#"), f"Not a hex color: {c}"
        assert len(c) == 7


def test_dominant_colors_red_image(red_image):
    """For a pure red image the most dominant color should be reddish."""
    m = MetadataModule.__new__(MetadataModule)
    colors = m.extract_dominant_colors(red_image, n=3)
    # Parse first color's R channel
    r = int(colors[0][1:3], 16)
    assert r > 180, f"Expected dominant color to be red, got {colors[0]}"


# ---------------------------------------------------------------------------
# Story 2.5 — Quality score + assets.json
# ---------------------------------------------------------------------------

def test_quality_score_1080p_image(test_image):
    m = MetadataModule.__new__(MetadataModule)
    asset = _make_asset("img_1", str(test_image), AssetType.IMAGE)
    # Manually set metadata as would happen in run()
    asset = asset.model_copy(update={
        "resolution": "1920x1080",
        "aspect_ratio": "16:9",
    })
    score = m.compute_quality_score(asset)
    assert score >= 6.0, f"Expected high quality score for 1080p 16:9, got {score}"


def test_quality_score_low_res(tmp_path):
    m = MetadataModule.__new__(MetadataModule)
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))
    p = tmp_path / "low.jpg"
    img.save(str(p))
    asset = _make_asset("img_low", str(p), AssetType.IMAGE)
    asset = asset.model_copy(update={"resolution": "320x240", "aspect_ratio": "4:3"})
    score = m.compute_quality_score(asset)
    assert score < 6.0, f"Expected low quality score for 320x240, got {score}"


def test_assets_json_written_and_valid(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a mock assets_raw.json
    fake_img_path = tmp_path / "test.jpg"
    Image.new("RGB", (1920, 1080), (100, 50, 200)).save(str(fake_img_path))

    raw_asset = Asset(
        id="img_001",
        type=AssetType.IMAGE,
        role=AssetRole.B_ROLL,
        source=AssetSource.LOCAL,
        local_path=str(fake_img_path),
    )
    (project_dir / "assets_raw.json").write_text(
        json.dumps([raw_asset.model_dump()], default=str)
    )

    module = MetadataModule(project_dir)
    assets = module.run()

    out = project_dir / "assets.json"
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert len(loaded) >= 1
    for item in loaded:
        Asset(**item)  # validates schema


def test_ready_for_use_set_on_good_asset(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    img_path = tmp_path / "hd.jpg"
    Image.new("RGB", (1920, 1080), (100, 50, 200)).save(str(img_path))
    raw = Asset(id="img_hd", type=AssetType.IMAGE, role=AssetRole.B_ROLL,
                source=AssetSource.LOCAL, local_path=str(img_path))
    (project_dir / "assets_raw.json").write_text(
        json.dumps([raw.model_dump()], default=str)
    )

    module = MetadataModule(project_dir)
    assets = module.run()

    good = [a for a in assets if a.quality_score >= 5.0]
    assert len(good) >= 1
    for a in good:
        assert a.ready_for_use is True
