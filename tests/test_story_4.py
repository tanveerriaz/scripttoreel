"""
MVP 4 — Module 4: Scene Planning & Orchestration tests.

Covers Stories 4.1–4.4:
  4.1 Asset-to-segment matching (scoring algorithm)
  4.2 Timeline construction (no gaps, sequential IDs)
  4.3 Transitions & color grading
  4.4 Audio mix plan + orchestration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module_4_orchestration import OrchestrationModule
from src.utils.json_schemas import (
    Asset, AssetRole, AssetSource, AssetType, ColorGrade,
    Mood, Orchestration, Scene, Script, ScriptSegment,
    SegmentTransitions, TextOverlay, TransitionType,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_video_asset(id_: str, mood: Mood, tags: list[str], duration: float = 10.0,
                      quality: float = 7.0, local_path: str = "/fake/video.mp4") -> Asset:
    return Asset(
        id=id_,
        type=AssetType.VIDEO,
        role=AssetRole.B_ROLL,
        source=AssetSource.LOCAL,
        local_path=local_path,
        duration_sec=duration,
        resolution="1920x1080",
        aspect_ratio="16:9",
        dominant_mood=mood,
        visual_tags=tags,
        quality_score=quality,
        ready_for_use=True,
    )


def _make_image_asset(id_: str, mood: Mood = Mood.NEUTRAL, local_path: str = "/fake/img.jpg") -> Asset:
    return Asset(
        id=id_,
        type=AssetType.IMAGE,
        role=AssetRole.B_ROLL,
        source=AssetSource.LOCAL,
        local_path=local_path,
        resolution="1920x1080",
        aspect_ratio="16:9",
        dominant_mood=mood,
        quality_score=6.0,
        ready_for_use=True,
    )


def _make_segment(id_: int, type_: str = "narration", duration: float = 10.0,
                  mood: str = "neutral", keywords: list[str] | None = None) -> ScriptSegment:
    from src.utils.json_schemas import SegmentType
    return ScriptSegment(
        id=id_,
        type=SegmentType(type_),
        text=f"Segment {id_} narration text.",
        duration_sec=duration,
        mood_tags=[mood],
        b_roll_keywords=keywords or [],
        transitions=SegmentTransitions(),
        text_overlay=TextOverlay(),
    )


def _make_script(segments: list[ScriptSegment], mood: Mood = Mood.NEUTRAL) -> Script:
    return Script(
        title="Test Video",
        topic="Test Topic",
        duration_sec=sum(s.duration_sec for s in segments),
        mood=mood,
        segments=segments,
    )


@pytest.fixture
def tmp_project(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "assets" / "audio").mkdir(parents=True)
    from datetime import datetime, timezone
    from src.utils.json_schemas import ProjectMetadata, ProjectPipeline
    meta = ProjectMetadata(
        project_id="test",
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


# ---------------------------------------------------------------------------
# Story 4.1 — Asset-to-segment matching
# ---------------------------------------------------------------------------

def test_mood_match_boosts_score():
    m = OrchestrationModule.__new__(OrchestrationModule)
    dark_asset = _make_video_asset("a1", Mood.DARK, ["haunted", "ruins"])
    bright_asset = _make_video_asset("a2", Mood.UPLIFTING, ["garden", "flowers"])
    seg = _make_segment(1, mood="dark", keywords=["haunted", "ruins"])

    score_dark = m._score_asset(dark_asset, seg)
    score_bright = m._score_asset(bright_asset, seg)
    assert score_dark > score_bright


def test_tag_overlap_boosts_score():
    m = OrchestrationModule.__new__(OrchestrationModule)
    tagged = _make_video_asset("a1", Mood.NEUTRAL, ["ancient ruins", "pakistan", "dark"])
    plain = _make_video_asset("a2", Mood.NEUTRAL, ["beach", "sunny"])
    seg = _make_segment(1, keywords=["ancient ruins", "pakistan"])

    assert m._score_asset(tagged, seg) > m._score_asset(plain, seg)


def test_best_asset_selected():
    m = OrchestrationModule.__new__(OrchestrationModule)
    assets = [
        _make_video_asset("low", Mood.NEUTRAL, [], quality=2.0),
        _make_video_asset("mid", Mood.DARK, [], quality=5.0),
        _make_video_asset("best", Mood.DARK, ["haunted"], quality=8.0),
    ]
    seg = _make_segment(1, mood="dark", keywords=["haunted"])
    chosen = m.match_asset_to_segment(seg, assets)
    assert chosen.id == "best"


def test_fallback_on_no_assets():
    """If assets list is empty, returns None (caller must handle)."""
    m = OrchestrationModule.__new__(OrchestrationModule)
    seg = _make_segment(1)
    result = m.match_asset_to_segment(seg, [])
    assert result is None


# ---------------------------------------------------------------------------
# Story 4.2 — Timeline construction
# ---------------------------------------------------------------------------

def test_no_gaps_in_timeline():
    segments = [_make_segment(i, duration=10.0) for i in range(1, 5)]
    script = _make_script(segments)
    assets = [_make_video_asset(f"v{i}", Mood.NEUTRAL, []) for i in range(4)]

    m = OrchestrationModule.__new__(OrchestrationModule)
    scenes = m.build_timeline(script, assets)

    for i in range(len(scenes) - 1):
        assert abs(scenes[i].end_time - scenes[i + 1].start_time) < 0.01


def test_scene_ids_sequential():
    segments = [_make_segment(i, duration=5.0) for i in range(1, 4)]
    script = _make_script(segments)
    assets = [_make_video_asset(f"v{i}", Mood.NEUTRAL, []) for i in range(3)]
    m = OrchestrationModule.__new__(OrchestrationModule)
    scenes = m.build_timeline(script, assets)
    # First scene is always the title card (id=0)
    assert scenes[0].id == 0
    assert scenes[0].is_title_card is True
    # Content scenes are 1-indexed and sequential
    content = [s for s in scenes if not s.is_title_card and not s.is_outro_card]
    for idx, scene in enumerate(content):
        assert scene.id == idx + 1


def test_total_duration_matches_script():
    segments = [_make_segment(i, duration=12.0) for i in range(1, 4)]
    script = _make_script(segments)
    assets = [_make_video_asset(f"v{i}", Mood.NEUTRAL, []) for i in range(3)]
    m = OrchestrationModule.__new__(OrchestrationModule)
    scenes = m.build_timeline(script, assets)
    total = scenes[-1].end_time
    # build_timeline prepends a 4s title card and appends a 3s outro card
    _OVERHEAD = 4.0 + 3.0
    assert abs(total - (script.duration_sec + _OVERHEAD)) <= 2.0


# ---------------------------------------------------------------------------
# Story 4.3 — Transitions & color grading
# ---------------------------------------------------------------------------

def test_dark_mood_gets_dark_grade():
    m = OrchestrationModule.__new__(OrchestrationModule)
    grade = m.assign_color_grade(Mood.DARK)
    assert grade == ColorGrade.DARK_MYSTERIOUS


def test_mysterious_mood_gets_dark_grade():
    m = OrchestrationModule.__new__(OrchestrationModule)
    grade = m.assign_color_grade(Mood.MYSTERIOUS)
    assert grade == ColorGrade.DARK_MYSTERIOUS


def test_uplifting_mood_gets_uplifting_grade():
    m = OrchestrationModule.__new__(OrchestrationModule)
    grade = m.assign_color_grade(Mood.UPLIFTING)
    assert grade == ColorGrade.UPLIFTING


def test_transitions_assigned_to_all_scenes():
    segments = [_make_segment(i, duration=8.0) for i in range(1, 4)]
    script = _make_script(segments)
    assets = [_make_video_asset(f"v{i}", Mood.NEUTRAL, []) for i in range(3)]
    m = OrchestrationModule.__new__(OrchestrationModule)
    scenes = m.build_timeline(script, assets)
    for scene in scenes:
        assert scene.transition_in is not None
        assert scene.transition_out is not None


# ---------------------------------------------------------------------------
# Story 4.4 — Orchestration JSON
# ---------------------------------------------------------------------------

def test_orchestration_json_valid_schema(tmp_project):
    # Build a minimal orchestration and save it
    segments = [
        _make_segment(1, type_="intro", duration=10.0, mood="dark"),
        _make_segment(2, type_="narration", duration=10.0, mood="neutral"),
        _make_segment(3, type_="outro", duration=10.0, mood="neutral"),
    ]
    script = _make_script(segments, mood=Mood.DARK)
    assets_json = [
        _make_video_asset("v1", Mood.DARK, ["ruins"], local_path="/tmp/fake.mp4").model_dump()
    ]

    # Write script.json + assets.json
    (tmp_project / "script.json").write_text(
        json.dumps(script.model_dump(), default=str)
    )
    (tmp_project / "assets.json").write_text(
        json.dumps(assets_json, default=str)
    )

    module = OrchestrationModule(tmp_project)
    orch = module.run()

    out = tmp_project / "orchestration.json"
    assert out.exists()
    raw = json.loads(out.read_text())
    loaded = Orchestration(**raw)
    # 1 title_card + 3 segments × 2 clips/segment + 1 outro_card = 8 scenes
    assert len(loaded.scenes) == 8


def test_voiceover_volume_is_1(tmp_project):
    segments = [_make_segment(i, duration=5.0) for i in range(1, 3)]
    script = _make_script(segments)
    # Attach fake voiceover
    for seg in script.segments:
        wav = tmp_project / "assets" / "audio" / f"vo_{seg.id}.wav"
        wav.touch()
        seg.voiceover_path = str(wav)

    assets = [_make_video_asset("v1", Mood.NEUTRAL, [], local_path="/tmp/fake.mp4")]
    (tmp_project / "script.json").write_text(json.dumps(script.model_dump(), default=str))
    (tmp_project / "assets.json").write_text(json.dumps([a.model_dump() for a in assets], default=str))

    module = OrchestrationModule(tmp_project)
    orch = module.run()

    for track in orch.voiceover_tracks:
        assert track.volume == 1.0


def test_music_volume_is_0_25(tmp_project):
    # Create a fake background music asset
    music_path = tmp_project / "assets" / "audio" / "bg_music.mp3"
    music_path.touch()
    music_asset = Asset(
        id="music_001",
        type=AssetType.AUDIO,
        role=AssetRole.MUSIC,
        source=AssetSource.LOCAL,
        local_path=str(music_path),
        duration_sec=60,
        ready_for_use=True,
        quality_score=7.0,
    )

    segments = [_make_segment(1, duration=10.0)]
    script = _make_script(segments)
    video_asset = _make_video_asset("v1", Mood.NEUTRAL, [], local_path="/tmp/fake.mp4")

    (tmp_project / "script.json").write_text(json.dumps(script.model_dump(), default=str))
    (tmp_project / "assets.json").write_text(
        json.dumps([video_asset.model_dump(), music_asset.model_dump()], default=str)
    )

    module = OrchestrationModule(tmp_project)
    orch = module.run()

    if orch.background_music:
        assert abs(orch.background_music.volume - 0.25) < 0.001
