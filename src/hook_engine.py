"""
Hook Engine — generates attention-grabbing opening hooks for video scripts.

12 proven hook patterns are used to generate or select the best opening
for a video, based on topic, tone, and target audience.

Usage:
    engine = HookEngine(api_keys)
    hooks = engine.generate_hooks("Agentic AI", "educational", "tech enthusiasts")
    best = engine.select_best_hook("Agentic AI", "educational", "tech enthusiasts")
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import requests

from src.utils.config_loader import load_api_keys, load_ollama_prompts
from src.utils.llm_client import call_llm

logger = logging.getLogger(__name__)

# 12 proven hook patterns with fill-in-the-blank templates
HOOK_PATTERNS: dict[str, dict] = {
    "question": {
        "template": "What if everything you knew about {topic} was wrong?",
        "description": "Challenges assumptions, sparks curiosity",
        "best_for": ["educational", "dramatic", "suspenseful"],
    },
    "stat": {
        "template": "By 2027, {topic} will reshape how we live — and most people aren't ready.",
        "description": "Uses data/timeframe to create urgency",
        "best_for": ["educational", "dramatic"],
    },
    "controversy": {
        "template": "Nobody wants to talk about the dark side of {topic}.",
        "description": "Provocative, challenges consensus",
        "best_for": ["dramatic", "suspenseful"],
    },
    "story": {
        "template": "Three years ago, a breakthrough in {topic} changed everything we thought we knew.",
        "description": "Narrative-driven, human element",
        "best_for": ["educational", "uplifting", "dramatic"],
    },
    "authority": {
        "template": "The world's leading experts agree: {topic} is the most important shift of our generation.",
        "description": "Credibility through authority",
        "best_for": ["educational"],
    },
    "contrast": {
        "template": "Everyone's focused on the obvious story. The real revolution in {topic} is hiding in plain sight.",
        "description": "Hidden truth, reframes perspective",
        "best_for": ["educational", "dramatic", "suspenseful"],
    },
    "curiosity_gap": {
        "template": "There's a reason {topic} keeps failing — and it's not what you think.",
        "description": "Creates information gap the viewer must fill",
        "best_for": ["educational", "dramatic", "suspenseful"],
    },
    "fomo": {
        "template": "While you were sleeping, {topic} quietly changed the game.",
        "description": "Fear of missing out, urgency",
        "best_for": ["educational", "dramatic"],
    },
    "myth_buster": {
        "template": "The biggest myth about {topic}? That you already understand it.",
        "description": "Debunks common misconceptions",
        "best_for": ["educational", "uplifting"],
    },
    "time_pressure": {
        "template": "You have 18 months before {topic} makes everything you know obsolete.",
        "description": "Urgency through deadline",
        "best_for": ["dramatic", "suspenseful"],
    },
    "empathy": {
        "template": "If you've ever felt overwhelmed trying to keep up with {topic}, you're not alone.",
        "description": "Emotional connection, validates feeling",
        "best_for": ["uplifting", "educational"],
    },
    "prediction": {
        "template": "Here's what {topic} will look like in 2028 — and why it matters to you right now.",
        "description": "Forward-looking, creates anticipation",
        "best_for": ["educational", "uplifting", "dramatic"],
    },
}

# Tone affinity scores (pattern → tone → score boost)
_TONE_AFFINITY: dict[str, dict[str, float]] = {
    "question":       {"educational": 1.5, "dramatic": 1.2, "suspenseful": 1.3},
    "stat":           {"educational": 1.5, "dramatic": 1.2},
    "controversy":    {"dramatic": 1.5, "suspenseful": 1.3},
    "story":          {"educational": 1.2, "uplifting": 1.4, "dramatic": 1.3},
    "authority":      {"educational": 1.6},
    "contrast":       {"educational": 1.3, "dramatic": 1.4, "suspenseful": 1.2},
    "curiosity_gap":  {"educational": 1.4, "dramatic": 1.3, "suspenseful": 1.5},
    "fomo":           {"educational": 1.2, "dramatic": 1.3},
    "myth_buster":    {"educational": 1.5, "uplifting": 1.2},
    "time_pressure":  {"dramatic": 1.5, "suspenseful": 1.4},
    "empathy":        {"uplifting": 1.5, "educational": 1.2},
    "prediction":     {"educational": 1.4, "uplifting": 1.3, "dramatic": 1.2},
}


class HookEngine:
    """Generates attention-grabbing opening hooks using 12 proven patterns."""

    def __init__(self, api_keys: Optional[dict] = None):
        self.api_keys = api_keys if api_keys is not None else load_api_keys()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_hooks(
        self,
        topic: str,
        tone: str = "educational",
        audience: str = "general",
        plan=None,  # Optional[ProductionPlan]
        patterns: Optional[list[str]] = None,
    ) -> list[dict]:
        """Generate top 3 hooks for the topic using LLM or template fallback.

        Returns list of dicts with keys: pattern, text, score.
        """
        hook_style = getattr(plan, "hook_style", None) if plan else None

        # If a specific pattern is requested (from plan or arg), constrain to it
        active_patterns = patterns or (
            [hook_style] if hook_style and hook_style in HOOK_PATTERNS else None
        )

        try:
            hooks = self._call_llm_for_hooks(topic, tone, audience, active_patterns)
            if hooks:
                return hooks[:3]
        except Exception as e:
            logger.warning("HookEngine LLM call failed: %s — using template fallback", e)

        return self._template_fallback(topic, tone, patterns=active_patterns)

    def select_best_hook(
        self,
        topic: str,
        tone: str = "educational",
        audience: str = "general",
        plan=None,
    ) -> dict:
        """Return the single best hook for the topic/tone."""
        hooks = self.generate_hooks(topic, tone, audience, plan=plan)
        if not hooks:
            return {"pattern": "question", "text": HOOK_PATTERNS["question"]["template"].format(topic=topic), "score": 5.0}
        return max(hooks, key=lambda h: h.get("score", 0))

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def _call_llm_for_hooks(
        self,
        topic: str,
        tone: str,
        audience: str,
        patterns: Optional[list[str]] = None,
    ) -> list[dict]:
        """Ask the LLM to generate 3 hooks and return parsed list."""
        try:
            prompts = load_ollama_prompts()
            hook_prompts = prompts.get("hook_engine", {})
            system_prompt = hook_prompts.get("system", _DEFAULT_SYSTEM_PROMPT)
            user_template = hook_prompts.get("user_template", _DEFAULT_USER_TEMPLATE)
        except Exception:
            system_prompt = _DEFAULT_SYSTEM_PROMPT
            user_template = _DEFAULT_USER_TEMPLATE

        pattern_constraint = ""
        if patterns:
            pattern_constraint = f"\nOnly use these hook patterns: {', '.join(patterns)}"

        user_prompt = user_template.format(
            topic=topic,
            tone=tone,
            audience=audience,
            pattern_constraint=pattern_constraint,
        )

        raw = call_llm(system_prompt, user_prompt, self.api_keys, temperature=0.8, max_tokens=1024)
        return self._parse_hooks_json(raw, topic, tone)

    def _parse_hooks_json(self, raw: str, topic: str, tone: str) -> list[dict]:
        """Parse LLM JSON response into list of hook dicts."""
        text = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Find JSON array in response
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                return []
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return []

        if isinstance(data, dict) and "hooks" in data:
            data = data["hooks"]
        if not isinstance(data, list):
            return []

        hooks = []
        for item in data:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern", "question")
            text_val = item.get("text", "")
            score = float(item.get("score", 5.0))
            if not text_val:
                continue
            # Apply tone affinity boost
            affinity = _TONE_AFFINITY.get(pattern, {}).get(tone, 1.0)
            hooks.append({"pattern": pattern, "text": text_val, "score": round(score * affinity, 2)})

        return sorted(hooks, key=lambda h: h["score"], reverse=True)

    # ------------------------------------------------------------------
    # Template fallback
    # ------------------------------------------------------------------

    def _template_fallback(
        self,
        topic: str,
        tone: str = "educational",
        patterns: Optional[list[str]] = None,
    ) -> list[dict]:
        """Fill hook templates and score by tone affinity. No LLM needed."""
        results = []
        active = patterns if patterns else list(HOOK_PATTERNS.keys())

        for name in active:
            if name not in HOOK_PATTERNS:
                continue
            pattern = HOOK_PATTERNS[name]
            text = pattern["template"].format(topic=topic)
            base_score = 5.0
            affinity = _TONE_AFFINITY.get(name, {}).get(tone, 1.0)
            score = round(base_score * affinity, 2)
            results.append({"pattern": name, "text": text, "score": score})

        return sorted(results, key=lambda h: h["score"], reverse=True)[:3]


# ---------------------------------------------------------------------------
# Default LLM prompts (used if ollama_prompts.yaml has no hook_engine section)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are a viral video hook writer. Generate 3 compelling opening hooks for a video.
Return ONLY a JSON array — no markdown, no extra text.
Each hook: {"pattern": "<name>", "text": "<1-2 sentence hook>", "score": <float 1-10>}
Pattern names: question, stat, controversy, story, authority, contrast, curiosity_gap,
               fomo, myth_buster, time_pressure, empathy, prediction"""

_DEFAULT_USER_TEMPLATE = """\
Topic: {topic}
Tone: {tone}
Audience: {audience}{pattern_constraint}

Generate 3 hooks that would make someone stop scrolling. Each hook must be 1-2 sentences,
punchy, and specific to the topic. Score each 1-10 for effectiveness.

Return JSON array only:
[
  {{"pattern": "question", "text": "...", "score": 8.5}},
  {{"pattern": "stat", "text": "...", "score": 7.2}},
  {{"pattern": "curiosity_gap", "text": "...", "score": 9.0}}
]"""
