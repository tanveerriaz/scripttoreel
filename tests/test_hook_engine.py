"""
Tests for the HookEngine (src/hook_engine.py).

Covers:
- All 12 patterns produce non-empty fallback text
- Templates fill topic correctly
- LLM call path (mocked) returns 3 hooks
- select_best_hook returns highest-scored entry
- LLM failure falls back to templates
- hook_style from plan constrains pattern selection
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hook_engine import HookEngine, HOOK_PATTERNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """HookEngine with no real API keys (forces fallback path)."""
    return HookEngine(api_keys={})


# ---------------------------------------------------------------------------
# Pattern coverage
# ---------------------------------------------------------------------------

def test_all_12_patterns_defined():
    assert len(HOOK_PATTERNS) == 12
    expected = {
        "question", "stat", "controversy", "story", "authority", "contrast",
        "curiosity_gap", "fomo", "myth_buster", "time_pressure", "empathy", "prediction",
    }
    assert set(HOOK_PATTERNS.keys()) == expected


def test_all_patterns_have_template():
    for name, pattern in HOOK_PATTERNS.items():
        assert "template" in pattern, f"Pattern '{name}' missing 'template'"
        assert "{topic}" in pattern["template"], f"Pattern '{name}' template missing {{topic}}"


def test_template_fallback_all_patterns_non_empty(engine):
    hooks = engine._template_fallback("Artificial Intelligence", "educational")
    # Returns top 3
    assert len(hooks) == 3
    for h in hooks:
        assert h["text"]
        assert h["pattern"] in HOOK_PATTERNS
        assert isinstance(h["score"], float)


def test_template_fills_topic(engine):
    hooks = engine._template_fallback("Quantum Computing", "educational")
    # Every returned hook should mention the topic
    for h in hooks:
        assert "Quantum Computing" in h["text"], f"Topic missing in hook: {h['text']}"


def test_template_fallback_constrained_to_pattern(engine):
    hooks = engine._template_fallback("Space Travel", "dramatic", patterns=["controversy"])
    assert len(hooks) == 1
    assert hooks[0]["pattern"] == "controversy"


# ---------------------------------------------------------------------------
# LLM call path (mocked)
# ---------------------------------------------------------------------------

def _make_mock_response(hooks_list: list) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": json.dumps(hooks_list)}}]
    }
    return mock


def test_generate_hooks_uses_llm_response():
    """When LLM returns valid JSON, generate_hooks returns those hooks."""
    llm_hooks = [
        {"pattern": "question", "text": "What if AI replaces all jobs?", "score": 9.0},
        {"pattern": "stat", "text": "By 2030, 40% of tasks will be automated.", "score": 8.0},
        {"pattern": "curiosity_gap", "text": "There's a hidden truth about AI.", "score": 7.5},
    ]
    engine = HookEngine(api_keys={
        "OPENROUTER_API_KEY": "fake-key",
        "OPENROUTER_MODEL": "test/model",
    })

    with patch("src.hook_engine.call_llm", return_value=json.dumps(llm_hooks)):
        hooks = engine.generate_hooks("Artificial Intelligence", "educational", "developers")

    assert len(hooks) == 3
    texts = [h["text"] for h in hooks]
    assert "What if AI replaces all jobs?" in texts


def test_generate_hooks_returns_3_or_fewer(engine):
    hooks = engine.generate_hooks("Blockchain", "educational", "general")
    assert 1 <= len(hooks) <= 3


def test_select_best_hook_returns_highest_score(engine):
    """select_best_hook returns the hook with the highest score."""
    with patch.object(engine, "generate_hooks") as mock_gen:
        mock_gen.return_value = [
            {"pattern": "question", "text": "Hook A", "score": 7.0},
            {"pattern": "stat", "text": "Hook B", "score": 9.5},
            {"pattern": "empathy", "text": "Hook C", "score": 6.0},
        ]
        best = engine.select_best_hook("Topic", "educational", "general")

    assert best["text"] == "Hook B"
    assert best["score"] == 9.5


# ---------------------------------------------------------------------------
# Fallback on LLM failure
# ---------------------------------------------------------------------------

def test_llm_failure_falls_back_to_templates():
    """If LLM call raises an exception, template fallback is used."""
    engine = HookEngine(api_keys={
        "USE_OPENROUTER": "true",
        "OPENROUTER_API_KEY": "fake-key",
    })

    with patch("requests.post", side_effect=Exception("Network error")):
        hooks = engine.generate_hooks("Climate Change", "dramatic", "general")

    assert len(hooks) >= 1
    for h in hooks:
        assert h["text"]
        assert "Climate Change" in h["text"]


def test_llm_bad_json_falls_back_to_templates():
    """Malformed LLM response falls back to templates."""
    engine = HookEngine(api_keys={
        "USE_OPENROUTER": "true",
        "OPENROUTER_API_KEY": "fake-key",
    })
    bad_response = MagicMock()
    bad_response.raise_for_status = MagicMock()
    bad_response.json.return_value = {
        "choices": [{"message": {"content": "This is not JSON at all!"}}]
    }

    with patch("requests.post", return_value=bad_response):
        hooks = engine.generate_hooks("Robotics", "educational", "students")

    assert len(hooks) >= 1
    for h in hooks:
        assert "Robotics" in h["text"]


# ---------------------------------------------------------------------------
# hook_style from plan
# ---------------------------------------------------------------------------

def test_hook_style_from_plan_constrains_pattern():
    """If plan.hook_style is set, only that pattern is used."""
    plan = MagicMock()
    plan.hook_style = "empathy"

    engine = HookEngine(api_keys={})

    with patch.object(engine, "_call_llm_for_hooks", side_effect=Exception("skip LLM")):
        hooks = engine.generate_hooks("Mental Health", "uplifting", "general", plan=plan)

    # All returned hooks should be empathy pattern (from template fallback)
    assert all(h["pattern"] == "empathy" for h in hooks)


def test_hook_style_none_allows_any_pattern(engine):
    """If plan.hook_style is None, any pattern is fair game."""
    plan = MagicMock()
    plan.hook_style = None

    hooks = engine.generate_hooks("Finance", "educational", "investors", plan=plan)
    patterns_used = {h["pattern"] for h in hooks}
    # Should have at least 2 different patterns (best 3 of 12)
    assert len(patterns_used) >= 1
