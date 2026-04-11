"""
Tests for src/ai_director.py — ScriptDirector and VisualDirector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_director import ScriptDirector, VisualDirector, _COLOR_TEMP, _TEMP_TO_GRADE
from src.utils.json_schemas import (
    ColorGrade,
    Mood,
    Orchestration,
    Scene,
    Script,
    ScriptSegment,
    SegmentTransitions,
    TextOverlay,
    TransitionType,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_API_KEYS_WITH_OR = {
    "OPENROUTER_API_KEY": "test-key",
    "OPENROUTER_MODEL": "anthropic/claude-sonnet-4-5",
}

_API_KEYS_NO_OR = {
    "OPENROUTER_API_KEY": "",
}


def _make_script(**overrides) -> Script:
    defaults = dict(
        title="Test Script",
        topic="Test Topic",
        duration_sec=60,
        mood=Mood.EDUCATIONAL,
        visual_style="documentary",
        color_palette=["#111111"],
        segments=[
            ScriptSegment(
                id=1,
                type="intro",
                text="Introduction to the topic.",
                duration_sec=20,
                b_roll_keywords=["test keyword one", "test keyword two"],
                mood_tags=["educational"],
                transitions=SegmentTransitions(
                    in_transition=TransitionType.FADE_IN,
                    out_transition=TransitionType.DISSOLVE,
                ),
                text_overlay=TextOverlay(enabled=True, text="Test Title", start_time=0.0),
            ),
            ScriptSegment(
                id=2,
                type="narration",
                text="The main body of the script.",
                duration_sec=30,
                b_roll_keywords=["specific footage"],
                mood_tags=["educational"],
                transitions=SegmentTransitions(
                    in_transition=TransitionType.DISSOLVE,
                    out_transition=TransitionType.DISSOLVE,
                ),
                text_overlay=TextOverlay(enabled=False, text="", start_time=0.0),
            ),
            ScriptSegment(
                id=3,
                type="outro",
                text="Closing thoughts.",
                duration_sec=10,
                b_roll_keywords=["outro visual"],
                mood_tags=["uplifting"],
                transitions=SegmentTransitions(
                    in_transition=TransitionType.DISSOLVE,
                    out_transition=TransitionType.FADE_OUT,
                ),
                text_overlay=TextOverlay(enabled=False, text="", start_time=0.0),
            ),
        ],
        background_music_style="ambient",
        overall_pacing="medium",
    )
    defaults.update(overrides)
    return Script(**defaults)


def _make_orchestration(scenes: list[Scene] | None = None) -> Orchestration:
    if scenes is None:
        scenes = []
    return Orchestration(
        project_id="test",
        title="Test",
        topic="Test Topic",
        total_duration_sec=60,
        scenes=scenes,
    )


def _make_scene(
    id: int,
    color_grade: ColorGrade = ColorGrade.DOCUMENTARY,
    transition_out: TransitionType = TransitionType.DISSOLVE,
) -> Scene:
    return Scene(
        id=id,
        segment_id=id,
        asset_id=f"asset_{id}",
        asset_path=f"/fake/asset_{id}.mp4",
        start_time=float((id - 1) * 10),
        end_time=float(id * 10),
        duration_sec=10.0,
        transition_in=TransitionType.FADE_IN if id == 1 else TransitionType.DISSOLVE,
        transition_out=transition_out,
        color_grade=color_grade,
    )


# ---------------------------------------------------------------------------
# ScriptDirector — unit tests
# ---------------------------------------------------------------------------

class TestScriptDirector:

    def test_review_calls_openrouter_when_key_present(self):
        """When an OpenRouter API key is configured, it must call OpenRouter."""
        director = ScriptDirector(api_keys=_API_KEYS_WITH_OR)
        script = _make_script()

        revised_json = json.dumps(script.model_dump(), default=str)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": revised_json}}]
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = director.review(script)

        call_kwargs = mock_post.call_args
        assert "openrouter.ai" in call_kwargs[0][0]
        assert isinstance(result, Script)

    def test_review_falls_back_when_llm_unavailable(self):
        """When LLM raises, the director falls back to the original script."""
        director = ScriptDirector(api_keys=_API_KEYS_NO_OR)
        script = _make_script()

        with patch("src.ai_director.call_llm", side_effect=RuntimeError("no key")):
            result = director.review(script)

        # Falls back gracefully — returns original script unchanged
        assert isinstance(result, Script)
        assert result.topic == script.topic

    def test_review_preserves_voiceover_paths(self):
        """The director must not overwrite voiceover_path on any segment."""
        director = ScriptDirector(api_keys=_API_KEYS_WITH_OR)
        script = _make_script()
        # Attach fake paths
        segments = []
        for seg in script.segments:
            segments.append(
                seg.model_copy(
                    update={
                        "voiceover_path": f"/audio/seg_{seg.id}.wav",
                        "voiceover_duration_sec": 5.0,
                    }
                )
            )
        script = script.model_copy(update={
            "segments": segments,
            "total_voiceover_path": "/audio/voiceover.wav",
        })

        # Director returns a script with no voiceover fields
        stripped = script.model_dump(exclude={"total_voiceover_path"})
        for seg in stripped["segments"]:
            seg.pop("voiceover_path", None)
            seg.pop("voiceover_duration_sec", None)
        revised_json = json.dumps(stripped, default=str)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": revised_json}}]
        }

        with patch("requests.post", return_value=mock_resp):
            result = director.review(script)

        for orig, rev in zip(script.segments, result.segments):
            assert rev.voiceover_path == orig.voiceover_path
        assert result.total_voiceover_path == script.total_voiceover_path

    def test_review_returns_original_on_bad_json(self):
        """If the LLM returns garbage, the original script is returned."""
        director = ScriptDirector(api_keys=_API_KEYS_WITH_OR)
        script = _make_script()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "not json at all {{"}}]
        }

        with patch("requests.post", return_value=mock_resp):
            result = director.review(script)

        # Should return the original without raising
        assert isinstance(result, Script)
        assert result.title == script.title

    def test_review_returns_original_on_network_error(self):
        """Network failure must not crash the pipeline."""
        import requests as req
        director = ScriptDirector(api_keys=_API_KEYS_WITH_OR)
        script = _make_script()

        with patch("requests.post", side_effect=req.ConnectionError("timeout")):
            result = director.review(script)

        assert isinstance(result, Script)
        assert result.title == script.title

    def test_parse_handles_in_out_transition_format(self):
        """Director may return "in"/"out" transition keys instead of Pydantic names."""
        director = ScriptDirector(api_keys=_API_KEYS_NO_OR)
        script = _make_script()

        data = script.model_dump(mode="json")
        # Replace in_transition/out_transition with in/out style
        for seg in data["segments"]:
            tr = seg["transitions"]
            seg["transitions"] = {
                "in": tr["in_transition"],
                "out": tr["out_transition"],
            }

        result = director._parse_revised_script(json.dumps(data), script)
        assert isinstance(result, Script)
        for seg in result.segments:
            assert seg.transitions.in_transition in list(TransitionType)
            assert seg.transitions.out_transition in list(TransitionType)

    def test_max_two_passes_only(self):
        """ScriptDirector must not exceed _MAX_DIRECTOR_PASSES calls."""
        director = ScriptDirector(api_keys=_API_KEYS_WITH_OR)
        script = _make_script()
        revised_json = json.dumps(script.model_dump(), default=str)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": revised_json}}]
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            director.review(script)

        # 2 passes → 2 POST calls
        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# VisualDirector — unit tests
# ---------------------------------------------------------------------------

class TestVisualDirector:

    def test_no_changes_for_coherent_scenes(self):
        """Consecutive scenes with similar temperature should not be changed."""
        scenes = [
            _make_scene(1, ColorGrade.DOCUMENTARY),
            _make_scene(2, ColorGrade.DOCUMENTARY),
            _make_scene(3, ColorGrade.DRAMATIC),
        ]
        orch = _make_orchestration(scenes)
        revised = VisualDirector().review(orch)

        for orig, rev in zip(orch.scenes, revised.scenes):
            assert orig.color_grade == rev.color_grade

    def test_large_temperature_jump_is_smoothed(self):
        """A jump from DARK_MYSTERIOUS (1) to UPLIFTING (5) exceeds the limit
        and should be adjusted."""
        scenes = [
            _make_scene(1, ColorGrade.DARK_MYSTERIOUS),
            _make_scene(2, ColorGrade.UPLIFTING),
        ]
        orch = _make_orchestration(scenes)
        revised = VisualDirector().review(orch)

        # The uplifting scene should be pulled toward dramatic (2)
        assert revised.scenes[1].color_grade != ColorGrade.UPLIFTING
        # Temperature of result must be closer to scene 1 than the original
        orig_gap = abs(
            _COLOR_TEMP[ColorGrade.UPLIFTING] - _COLOR_TEMP[ColorGrade.DARK_MYSTERIOUS]
        )
        new_gap = abs(
            _COLOR_TEMP[revised.scenes[1].color_grade]
            - _COLOR_TEMP[ColorGrade.DARK_MYSTERIOUS]
        )
        assert new_gap < orig_gap

    def test_crossfade_replaced_on_energy_mismatch(self):
        """CROSSFADE between scenes with energy gap ≥ 3 should become DISSOLVE."""
        scenes = [
            _make_scene(1, ColorGrade.UPLIFTING, transition_out=TransitionType.CROSSFADE),
            _make_scene(2, ColorGrade.DARK_MYSTERIOUS),
        ]
        orch = _make_orchestration(scenes)
        revised = VisualDirector().review(orch)

        assert revised.scenes[0].transition_out == TransitionType.DISSOLVE

    def test_crossfade_kept_for_similar_energy(self):
        """CROSSFADE between DOCUMENTARY and DRAMATIC (energy 3→2) should be kept."""
        scenes = [
            _make_scene(1, ColorGrade.DOCUMENTARY, transition_out=TransitionType.CROSSFADE),
            _make_scene(2, ColorGrade.DRAMATIC),
        ]
        orch = _make_orchestration(scenes)
        revised = VisualDirector().review(orch)

        assert revised.scenes[0].transition_out == TransitionType.CROSSFADE

    def test_empty_orchestration_is_handled(self):
        """An orchestration with no scenes must not raise."""
        orch = _make_orchestration(scenes=[])
        revised = VisualDirector().review(orch)
        assert revised.scenes == []

    def test_single_scene_is_unchanged(self):
        """A single scene has no neighbours to compare — it must not be modified."""
        scenes = [_make_scene(1, ColorGrade.UPLIFTING)]
        orch = _make_orchestration(scenes)
        revised = VisualDirector().review(orch)
        assert revised.scenes[0].color_grade == ColorGrade.UPLIFTING

    def test_count_changes_returns_correct_delta(self):
        """_count_changes should tally only scenes with modified fields."""
        director = VisualDirector()
        original = [
            _make_scene(1, ColorGrade.DOCUMENTARY, TransitionType.DISSOLVE),
            _make_scene(2, ColorGrade.UPLIFTING, TransitionType.DISSOLVE),
        ]
        revised = [
            _make_scene(1, ColorGrade.DOCUMENTARY, TransitionType.DISSOLVE),  # unchanged
            _make_scene(2, ColorGrade.DRAMATIC, TransitionType.DISSOLVE),     # grade changed
        ]
        assert director._count_changes(original, revised) == 1

    def test_review_returns_orchestration_instance(self):
        """review() must always return an Orchestration object."""
        scenes = [_make_scene(i) for i in range(1, 4)]
        orch = _make_orchestration(scenes)
        result = VisualDirector().review(orch)
        assert isinstance(result, Orchestration)


# ---------------------------------------------------------------------------
# Integration: ScriptModule uses ScriptDirector when skip_director=False
# ---------------------------------------------------------------------------

class TestScriptModuleDirectorIntegration:

    def test_script_draft_saved_when_director_runs(self, tmp_path):
        """Module 3 must write script_draft.json before the director pass."""
        pytest.importorskip("pydub", reason="pydub not installed")
        import json as _json
        from datetime import datetime, timezone

        from src.module_3_script_voiceover import ScriptModule
        from src.utils.json_schemas import ProjectMetadata

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "assets" / "audio").mkdir(parents=True)

        meta = ProjectMetadata(
            project_id="test_dir",
            topic="Test Topic",
            duration_min=1,
            duration_sec=60,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            project_dir=str(project_dir),
        )
        (project_dir / "project.json").write_text(
            _json.dumps(meta.model_dump(), default=str)
        )

        # Pre-build a minimal Script that generate_script will return
        script = _make_script()
        revised_json = _json.dumps(script.model_dump(), default=str)

        module = ScriptModule(project_dir, skip_director=False)

        # Patch generate_script so no real LLM is called
        with patch.object(module, "generate_script", return_value=script):
            # Patch ScriptDirector.review to return same script
            with patch("src.ai_director.ScriptDirector.review", return_value=script):
                # Patch voiceover generation to avoid TTS
                with patch.object(module, "generate_all_voiceovers", return_value=script):
                    with patch.object(module, "concatenate_voiceovers"):
                        with patch("src.project_manager.update_pipeline_status"):
                            module.run()

        assert (project_dir / "script_draft.json").exists()

    def test_script_draft_not_saved_when_director_skipped(self, tmp_path):
        """script_draft.json must NOT be written when --skip-director is set."""
        pytest.importorskip("pydub", reason="pydub not installed")
        import json as _json
        from datetime import datetime, timezone

        from src.module_3_script_voiceover import ScriptModule
        from src.utils.json_schemas import ProjectMetadata

        project_dir = tmp_path / "proj_skip"
        project_dir.mkdir()
        (project_dir / "assets" / "audio").mkdir(parents=True)

        meta = ProjectMetadata(
            project_id="test_skip",
            topic="Test Topic",
            duration_min=1,
            duration_sec=60,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            project_dir=str(project_dir),
        )
        (project_dir / "project.json").write_text(
            _json.dumps(meta.model_dump(), default=str)
        )

        script = _make_script()
        module = ScriptModule(project_dir, skip_director=True)

        with patch.object(module, "generate_script", return_value=script):
            with patch.object(module, "generate_all_voiceovers", return_value=script):
                with patch.object(module, "concatenate_voiceovers"):
                    with patch("src.project_manager.update_pipeline_status"):
                        module.run()

        assert not (project_dir / "script_draft.json").exists()
