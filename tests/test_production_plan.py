"""
Tests for the production plan system and the fixed module execution order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub out heavy optional dependencies so tests run without a full install
import types as _types
for _mod in ("pydub", "pydub.AudioSegment"):
    if _mod not in sys.modules:
        _stub = _types.ModuleType(_mod)
        _stub.AudioSegment = MagicMock()
        sys.modules[_mod] = _stub

from src.production_plan import ProductionPlanModule
from src.utils.json_schemas import (
    ProductionPlan,
    ToneStyle,
    VisualStyleChoice,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    """Minimal project directory with project.json."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "assets" / "audio").mkdir(parents=True)
    project_json = {
        "project_id": "test_project",
        "topic": "Haunted Places in Pakistan",
        "duration_min": 5.0,
        "duration_sec": 300.0,
        "created_at": "2026-04-05T00:00:00",
        "updated_at": "2026-04-05T00:00:00",
        "project_dir": str(project_dir),
        "pipeline": {
            "module_1_research": "pending",
            "module_2_metadata": "pending",
            "module_3_script": "pending",
            "module_4_orchestration": "pending",
            "module_5_render": "pending",
            "module_6_validation": "pending",
        },
    }
    (project_dir / "project.json").write_text(json.dumps(project_json))
    return project_dir


@pytest.fixture
def fake_api_keys():
    return {
        "OPENROUTER_API_KEY": "fake-test-key",
        "OPENROUTER_MODEL": "anthropic/claude-sonnet-4-5",
    }


# ---------------------------------------------------------------------------
# ProductionPlan Pydantic model tests
# ---------------------------------------------------------------------------

class TestProductionPlanModel:
    def test_default_values(self):
        plan = ProductionPlan(topic="Test Topic")
        assert plan.narrator_voice == "en-US-AriaNeural"
        assert plan.tone in ("educational", "documentary", ToneStyle.DOCUMENTARY)
        assert plan.visual_style == VisualStyleChoice.DOCUMENTARY
        assert plan.target_audience == "general audience"
        assert plan.avoid_list == []
        assert plan.image_search_queries == []
        assert plan.script_guidance == ""
        assert plan.testimonial_voices == []

    def test_all_tone_values_valid(self):
        for tone in ToneStyle:
            plan = ProductionPlan(topic="X", tone=tone)
            assert plan.tone == tone

    def test_all_visual_style_values_valid(self):
        for vs in VisualStyleChoice:
            plan = ProductionPlan(topic="X", visual_style=vs)
            assert plan.visual_style == vs

    def test_serialization_round_trip(self):
        plan = ProductionPlan(
            topic="AI Revolution",
            tone=ToneStyle.CINEMATIC,
            visual_style=VisualStyleChoice.DARK_MYSTERIOUS,
            avoid_list=["violence", "politics"],
            image_search_queries=["AI robot arm", "neural network glow"],
            narrator_voice="Daniel",
        )
        data = plan.model_dump()
        restored = ProductionPlan(**data)
        assert restored.topic == plan.topic
        assert restored.tone == plan.tone
        assert restored.visual_style == plan.visual_style
        assert restored.avoid_list == plan.avoid_list
        assert restored.narrator_voice == plan.narrator_voice

    def test_cultural_context_optional(self):
        plan = ProductionPlan(topic="X", cultural_context="South Asian cultural norms")
        assert plan.cultural_context == "South Asian cultural norms"


# ---------------------------------------------------------------------------
# ProductionPlanModule tests
# ---------------------------------------------------------------------------

class TestProductionPlanModule:
    def test_exists_false_when_no_file(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        assert not pm.exists()

    def test_exists_true_after_save(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        plan = ProductionPlan(topic="Test")
        pm.save(plan)
        assert pm.exists()
        assert (tmp_project / "production_plan.json").exists()

    def test_load_after_save(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        original = ProductionPlan(
            topic="Haunted Places",
            tone=ToneStyle.DRAMATIC,
            narrator_voice="Daniel",
            image_search_queries=["dark ruins night", "abandoned mansion"],
        )
        pm.save(original)
        loaded = pm.load()
        assert loaded.topic == "Haunted Places"
        assert loaded.tone == ToneStyle.DRAMATIC
        assert loaded.narrator_voice == "Daniel"
        assert "dark ruins night" in loaded.image_search_queries

    def test_load_raises_when_no_file(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        with pytest.raises(FileNotFoundError):
            pm.load()

    def test_default_plan_fallback(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        plan = pm._default_plan("Haunted Places in Pakistan", 5.0)
        assert plan.topic == "Haunted Places in Pakistan"
        assert plan.duration_minutes == 5.0
        assert len(plan.image_search_queries) > 0
        # Should include the topic itself
        assert any("Pakistan" in q or "Haunted" in q for q in plan.image_search_queries)

    def test_parse_plan_valid_json(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        raw = json.dumps({
            "topic": "AI in Healthcare",
            "narrator_voice": "Samantha",
            "testimonial_voices": [],
            "tone": "documentary",
            "visual_style": "documentary",
            "target_audience": "medical professionals",
            "cultural_context": "",
            "duration_minutes": 5.0,
            "avoid_list": ["misinformation"],
            "image_search_queries": ["doctor hospital corridor", "MRI scanner glow"],
            "script_guidance": "Be factual and calm",
        })
        plan = pm._parse_plan(raw, "AI in Healthcare", 5.0)
        assert plan.topic == "AI in Healthcare"
        assert plan.tone == ToneStyle.DOCUMENTARY
        assert "doctor hospital corridor" in plan.image_search_queries

    def test_parse_plan_with_code_fences(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        raw = '```json\n{"topic": "Test", "duration_minutes": 3.0}\n```'
        plan = pm._parse_plan(raw, "Test", 3.0)
        assert plan.topic == "Test"

    def test_parse_plan_invalid_tone_falls_back(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        raw = json.dumps({
            "topic": "X",
            "tone": "INVALID_TONE",
            "visual_style": "documentary",
        })
        plan = pm._parse_plan(raw, "X", 5.0)
        assert plan.tone == ToneStyle.DOCUMENTARY

    def test_parse_plan_invalid_visual_style_falls_back(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        raw = json.dumps({
            "topic": "X",
            "tone": "cinematic",
            "visual_style": "INVALID",
        })
        plan = pm._parse_plan(raw, "X", 5.0)
        assert plan.visual_style == VisualStyleChoice.DOCUMENTARY

    def test_generate_uses_default_on_llm_failure(self, tmp_project, fake_api_keys):
        pm = ProductionPlanModule(tmp_project, api_keys=fake_api_keys)
        with patch.object(pm, "_call_llm", side_effect=RuntimeError("Ollama down")):
            plan = pm.generate("Space Exploration", 3.0)
        assert plan.topic == "Space Exploration"
        assert (tmp_project / "production_plan.json").exists()

    def test_generate_calls_openrouter_when_configured(self, tmp_project):
        api_keys = {
            "OPENROUTER_API_KEY": "sk-test-key",
            "OPENROUTER_MODEL": "anthropic/claude-sonnet-4-5",
        }
        pm = ProductionPlanModule(tmp_project, api_keys=api_keys)
        good_response = json.dumps({
            "topic": "Climate Change",
            "narrator_voice": "Samantha",
            "testimonial_voices": [],
            "tone": "documentary",
            "visual_style": "documentary",
            "target_audience": "general",
            "cultural_context": "",
            "duration_minutes": 5.0,
            "avoid_list": [],
            "image_search_queries": ["melting glacier", "polar bear ice"],
            "script_guidance": "factual",
        })
        with patch("src.production_plan.call_llm", return_value=good_response) as mock_llm:
            plan = pm.generate("Climate Change", 5.0)
        mock_llm.assert_called_once()
        assert plan.topic == "Climate Change"


# ---------------------------------------------------------------------------
# Module 1 integration: reads production_plan.json queries
# ---------------------------------------------------------------------------

class TestModule1ReadsProductionPlan:
    def test_uses_plan_image_queries(self, tmp_project, fake_api_keys):
        # Write a production plan with specific queries
        plan = ProductionPlan(
            topic="Haunted Places in Pakistan",
            image_search_queries=["Pakistan ancient fort night", "Lahore ruins fog", "abandoned haveli"],
        )
        (tmp_project / "production_plan.json").write_text(
            json.dumps(plan.model_dump(), indent=2)
        )
        # Write minimal project.json
        project_json = {
            "topic": "Haunted Places in Pakistan",
            "duration_min": 5.0,
        }
        (tmp_project / "project.json").write_text(json.dumps(project_json))

        from src.module_1_research import ResearchModule
        m = ResearchModule(tmp_project, api_keys=fake_api_keys)
        queries = m._build_search_queries("Haunted Places in Pakistan")

        assert "Pakistan ancient fort night" in queries
        assert "Lahore ruins fog" in queries
        assert "abandoned haveli" in queries

    def test_falls_back_to_topic_without_plan(self, tmp_project, fake_api_keys):
        project_json = {"topic": "Deep Sea Creatures", "duration_min": 3.0}
        (tmp_project / "project.json").write_text(json.dumps(project_json))

        from src.module_1_research import ResearchModule
        m = ResearchModule(tmp_project, api_keys=fake_api_keys)
        queries = m._build_search_queries("Deep Sea Creatures")

        assert "Deep Sea Creatures" in queries

    def test_merges_script_keywords_with_plan_queries(self, tmp_project, fake_api_keys):
        # Plan has image queries
        plan = ProductionPlan(
            topic="Space Exploration",
            image_search_queries=["rocket launch flames", "astronaut spacewalk"],
        )
        (tmp_project / "production_plan.json").write_text(
            json.dumps(plan.model_dump(), indent=2)
        )
        # Script.json has b_roll_keywords
        script_data = {
            "segments": [
                {"b_roll_keywords": ["NASA control room", "ISS station orbit"]},
            ]
        }
        (tmp_project / "script.json").write_text(json.dumps(script_data))
        (tmp_project / "project.json").write_text(
            json.dumps({"topic": "Space Exploration", "duration_min": 5.0})
        )

        from src.module_1_research import ResearchModule
        m = ResearchModule(tmp_project, api_keys=fake_api_keys)
        queries = m._build_search_queries("Space Exploration")

        assert "rocket launch flames" in queries
        assert "NASA control room" in queries

    def test_cap_is_higher_with_plan(self, tmp_project, fake_api_keys):
        """When production plan exists, up to 8 queries are used (vs 4 without)."""
        plan = ProductionPlan(
            topic="X",
            image_search_queries=[f"query {i}" for i in range(10)],
        )
        (tmp_project / "production_plan.json").write_text(
            json.dumps(plan.model_dump(), indent=2)
        )
        (tmp_project / "project.json").write_text(json.dumps({"topic": "X", "duration_min": 1.0}))

        from src.module_1_research import ResearchModule
        m = ResearchModule(tmp_project, api_keys=fake_api_keys)
        queries = m._build_search_queries("X")
        assert len(queries) <= 8
        assert len(queries) > 4  # more than legacy cap


# ---------------------------------------------------------------------------
# Module 3 integration: uses plan settings
# ---------------------------------------------------------------------------

class TestModule3UsesPlan:
    def test_prompt_augmented_with_plan_settings(self, tmp_project, fake_api_keys):
        from src.module_3_script_voiceover import ScriptModule
        m = ScriptModule(tmp_project, api_keys=fake_api_keys)

        plan = ProductionPlan(
            topic="AI in Healthcare",
            tone=ToneStyle.CINEMATIC,
            target_audience="medical students",
            cultural_context="Western medical context",
            avoid_list=["misinformation", "graphic imagery"],
            script_guidance="Use authoritative but warm narration",
        )

        # We capture the user_prompt by mocking call_llm
        captured = {}

        def fake_llm(system_prompt, user_prompt, api_keys=None, **kwargs):
            captured["user_prompt"] = user_prompt
            return _minimal_script_json("AI in Healthcare", 300)

        with patch("src.module_3_script_voiceover.call_llm", side_effect=fake_llm):
            m.generate_script("AI in Healthcare", 5.0, plan=plan)

        # Verify the user_prompt was augmented with plan notes
        assert "user_prompt" in captured
        assert "cinematic" in captured["user_prompt"].lower()
        assert "medical students" in captured["user_prompt"]
        assert "ADDITIONAL PRODUCTION NOTES" in captured["user_prompt"]

    def test_no_plan_generates_script_normally(self, tmp_project, fake_api_keys):
        """Without production plan, generate_script works as before."""
        from src.module_3_script_voiceover import ScriptModule
        m = ScriptModule(tmp_project, api_keys=fake_api_keys)
        script_json = _minimal_script_json("Deep Ocean", 180)

        with patch("src.module_3_script_voiceover.call_llm", return_value=script_json):
            script = m.generate_script("Deep Ocean", 3.0, plan=None)

        assert script.topic == "Deep Ocean"

    def test_load_production_plan_returns_none_when_missing(self, tmp_project, fake_api_keys):
        from src.module_3_script_voiceover import ScriptModule
        m = ScriptModule(tmp_project, api_keys=fake_api_keys)
        assert m._load_production_plan() is None

    def test_load_production_plan_returns_plan_when_exists(self, tmp_project, fake_api_keys):
        plan = ProductionPlan(topic="Test", narrator_voice="Daniel")
        (tmp_project / "production_plan.json").write_text(
            json.dumps(plan.model_dump(), indent=2)
        )
        from src.module_3_script_voiceover import ScriptModule
        m = ScriptModule(tmp_project, api_keys=fake_api_keys)
        loaded = m._load_production_plan()
        assert loaded is not None
        assert loaded.narrator_voice == "Daniel"


# ---------------------------------------------------------------------------
# Module 4 integration: uses plan visual_style
# ---------------------------------------------------------------------------

class TestModule4UsesPlan:
    def test_plan_color_grade_mapping(self, tmp_project):
        from src.module_4_orchestration import OrchestrationModule
        from src.utils.json_schemas import ColorGrade

        m = OrchestrationModule(tmp_project)

        cases = [
            ("dark_mysterious", ColorGrade.DARK_MYSTERIOUS),
            ("cinematic_warm", ColorGrade.CINEMATIC_WARM),
            ("documentary", ColorGrade.DOCUMENTARY),
            ("dramatic", ColorGrade.DRAMATIC),
            ("bright_modern", ColorGrade.UPLIFTING),
        ]
        for style_value, expected_grade in cases:
            plan = ProductionPlan(topic="X", visual_style=style_value)
            grade = m._plan_color_grade(plan)
            assert grade == expected_grade, f"visual_style={style_value!r} → expected {expected_grade}"

    def test_plan_color_grade_none_when_no_plan(self, tmp_project):
        from src.module_4_orchestration import OrchestrationModule
        m = OrchestrationModule(tmp_project)
        assert m._plan_color_grade(None) is None

    def test_load_production_plan_returns_none_when_missing(self, tmp_project):
        from src.module_4_orchestration import OrchestrationModule
        m = OrchestrationModule(tmp_project)
        assert m._load_production_plan() is None

    def test_load_production_plan_returns_plan_when_exists(self, tmp_project):
        plan = ProductionPlan(topic="Test", visual_style=VisualStyleChoice.DRAMATIC)
        (tmp_project / "production_plan.json").write_text(
            json.dumps(plan.model_dump(), indent=2)
        )
        from src.module_4_orchestration import OrchestrationModule
        m = OrchestrationModule(tmp_project)
        loaded = m._load_production_plan()
        assert loaded is not None
        assert loaded.visual_style == VisualStyleChoice.DRAMATIC


# ---------------------------------------------------------------------------
# CLI: --no-plan flag
# ---------------------------------------------------------------------------

class TestCLINoplan:
    def test_no_plan_flag_uses_legacy_order(self, tmp_path):
        """With --no-plan, modules run in order 1→2→3→4→5→6."""
        from click.testing import CliRunner
        from main import cli

        runner = CliRunner()
        # We just verify that the module dispatch order is correct in the source;
        # a lightweight integration check: --no-plan doesn't call production plan.
        with patch("main._ensure_production_plan") as mock_plan, \
             patch("main._run_module") as mock_run:
            mock_run.return_value = None
            result = runner.invoke(cli, [
                "--run", "--no-plan", "--project", "test_project",
                "--projects-root", str(tmp_path),
            ])
        # _ensure_production_plan should NOT be called with --no-plan
        mock_plan.assert_not_called()

    def test_without_no_plan_ensures_production_plan(self, tmp_path):
        """Without --no-plan, _ensure_production_plan is called before running."""
        from click.testing import CliRunner
        from main import cli

        # Create minimal project
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        project_json = {
            "project_id": "test_project",
            "topic": "Test Topic",
            "duration_min": 2.0,
            "duration_sec": 120.0,
            "created_at": "2026-04-05T00:00:00",
            "updated_at": "2026-04-05T00:00:00",
            "project_dir": str(project_dir),
            "pipeline": {
                "module_1_research": "pending",
                "module_2_metadata": "pending",
                "module_3_script": "pending",
                "module_4_orchestration": "pending",
                "module_5_render": "pending",
                "module_6_validation": "pending",
            },
        }
        (project_dir / "project.json").write_text(json.dumps(project_json))

        runner = CliRunner()
        with patch("main._ensure_production_plan") as mock_plan, \
             patch("main._run_module") as mock_run:
            mock_run.return_value = None
            runner.invoke(cli, [
                "--run", "--project", "test_project",
                "--projects-root", str(tmp_path),
            ])
        mock_plan.assert_called_once()

    def test_plan_order_is_3_1_2_4_5_6(self):
        """Verify the new module order constant."""
        from main import _PLAN_ORDER, _LEGACY_ORDER
        assert _PLAN_ORDER == [3, 1, 2, 4, 5, 6]
        assert _LEGACY_ORDER == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_script_json(topic: str, duration_sec: int) -> str:
    """Return a minimal valid script JSON string for mocking LLM responses."""
    return json.dumps({
        "title": f"Video about {topic}",
        "topic": topic,
        "duration_sec": duration_sec,
        "mood": "educational",
        "visual_style": "documentary",
        "color_palette": ["#333333"],
        "segments": [
            {
                "id": 1,
                "type": "intro",
                "text": f"Welcome to our exploration of {topic}.",
                "duration_sec": duration_sec,
                "visual_cues": ["opening shot"],
                "mood_tags": ["educational"],
                "b_roll_keywords": [f"{topic} overview"],
                "sfx_cues": [],
                "music_cues": ["ambient"],
                "transitions": {"in": "fade_in", "out": "dissolve"},
                "text_overlay": {
                    "enabled": True,
                    "text": topic,
                    "position": "bottom_third",
                    "style": "lower_third",
                    "start_time": 0.0,
                },
            }
        ],
        "background_music_style": "ambient",
        "overall_pacing": "medium",
    })
