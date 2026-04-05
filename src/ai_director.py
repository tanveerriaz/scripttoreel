"""
AI Director layer for ScriptToReel.

AIDirector: Generates pre-production plan (production_plan.json) during --init.
ScriptDirector: Reviews and improves the generated script (narrative flow,
pacing, emotional arc, b-roll keyword quality, segment duration balance).
VisualDirector: Reviews visual coherence of the orchestration plan (color
temperature consistency, transition narrative logic).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from src.utils.config_loader import load_api_keys, load_ollama_prompts
from src.utils.json_schemas import (
    ColorGrade,
    Mood,
    Orchestration,
    ProductionPlan,
    Scene,
    Script,
    TransitionType,
)

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 3
_MAX_DIRECTOR_PASSES = 2

_VALID_MOODS = {"dark", "mysterious", "uplifting", "dramatic", "educational",
                "horror", "neutral", "suspenseful", "melancholic"}
_VALID_VISUAL_STYLES = {"documentary", "cinematic", "minimalist"}
_VALID_TONES = {"educational", "dramatic", "uplifting", "suspenseful"}
_VALID_OVERLAY_STYLES = {"lower_third", "minimal", "bold", "subtitle"}

_TRANSITION_MAP = {
    "fade_in": "fade_in",
    "fade_out": "fade_out",
    "dissolve": "dissolve",
    "crossfade": "crossfade",
    "cut": "cut",
    "none": "none",
}


def _normalize_transition(value: str) -> str:
    return _TRANSITION_MAP.get(str(value).lower(), "dissolve")


# Color temperature scale: 1 = cool/dark, 5 = warm/uplifting
_COLOR_TEMP: dict[ColorGrade, int] = {
    ColorGrade.DARK_MYSTERIOUS: 1,
    ColorGrade.DRAMATIC: 2,
    ColorGrade.DOCUMENTARY: 3,
    ColorGrade.CINEMATIC_WARM: 4,
    ColorGrade.UPLIFTING: 5,
}

# Inverse map: temperature int → ColorGrade (nearest match)
_TEMP_TO_GRADE: dict[int, ColorGrade] = {v: k for k, v in _COLOR_TEMP.items()}

# Mood energy used for transition mismatch detection (1=somber, 5=upbeat)
_GRADE_ENERGY: dict[ColorGrade, int] = {
    ColorGrade.DARK_MYSTERIOUS: 1,
    ColorGrade.DRAMATIC: 2,
    ColorGrade.DOCUMENTARY: 3,
    ColorGrade.CINEMATIC_WARM: 4,
    ColorGrade.UPLIFTING: 5,
}


class AIDirector:
    """Generates a pre-production plan (production_plan.json) for a project."""

    def __init__(self, project_dir: Path, api_keys: Optional[dict] = None):
        self.project_dir = Path(project_dir)
        self.api_keys = api_keys if api_keys is not None else load_api_keys()
        self.ollama_url = self.api_keys.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = self.api_keys.get("OLLAMA_MODEL", "llama3.2")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, topic: str, duration_min: float) -> ProductionPlan:
        """Generate and save production_plan.json. Returns the plan."""
        logger.info("AI Director: generating production plan for %r", topic)
        plan = self._generate_plan(topic, duration_min)
        self._save(plan)
        logger.info("AI Director: production_plan.json written")
        return plan

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_plan(self, topic: str, duration_min: float) -> ProductionPlan:
        prompts = load_ollama_prompts()
        director_cfg = prompts.get("ai_director", {})
        system_prompt = director_cfg.get("system", "You are an AI video director. Return JSON only.")
        user_template = director_cfg.get("user_template", "")
        duration_sec = int(duration_min * 60)
        user_prompt = user_template.format(
            topic=topic,
            duration=duration_min,
            duration_sec=duration_sec,
        )

        use_openrouter = self.api_keys.get("USE_OPENROUTER", "").lower() == "true"
        or_key = self.api_keys.get("OPENROUTER_API_KEY", "")
        or_model = self.api_keys.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")

        last_error: Exception = Exception("Unknown LLM error")
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                if use_openrouter and or_key:
                    raw = self._call_openrouter_plan(system_prompt, user_prompt, or_key, or_model)
                else:
                    raw = self._call_ollama_plan(system_prompt, user_prompt)
                return self._parse_plan(raw, topic, duration_sec)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning("AI Director attempt %d failed: %s", attempt + 1, e)
                continue

        # Fallback: return a sensible default plan so --init never hard-fails
        logger.warning("AI Director LLM failed after %d attempts; using fallback plan", _MAX_LLM_RETRIES)
        return self._fallback_plan(topic, duration_sec)

    def _call_ollama_plan(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "options": {"temperature": 0.6, "num_predict": -1},
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_url}") from e

    def _call_openrouter_plan(
        self, system_prompt: str, user_prompt: str, api_key: str, model: str
    ) -> str:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://scripttoreel.local",
                "X-Title": "ScriptToReel",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.6,
                "max_tokens": 2048,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_plan(self, raw_text: str, topic: str, duration_sec: int) -> ProductionPlan:
        text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON in AI Director response: {raw_text[:200]!r}")
            data = json.loads(text[start:end + 1])

        # Normalise enum-like fields
        mood_raw = str(data.get("mood", "educational"))
        data["mood"] = next(
            (m for m in (p.strip() for p in mood_raw.replace("|", ",").split(","))
             if m in _VALID_MOODS),
            "educational",
        )
        vs = str(data.get("visual_style", "documentary")).lower()
        data["visual_style"] = vs if vs in _VALID_VISUAL_STYLES else "documentary"

        tone = str(data.get("tone", "educational")).lower()
        data["tone"] = tone if tone in _VALID_TONES else "educational"

        ols = str(data.get("text_overlay_style", "lower_third")).lower()
        data["text_overlay_style"] = ols if ols in _VALID_OVERLAY_STYLES else "lower_third"

        # Ensure lists are actually lists
        for list_key in ("search_keywords", "scene_breakdown",
                         "background_music_keywords", "sfx_keywords", "color_palette"):
            if not isinstance(data.get(list_key), list):
                data[list_key] = []

        data["topic"] = topic
        data["duration_sec"] = float(duration_sec)
        data["generated_at"] = datetime.now(timezone.utc).isoformat()

        return ProductionPlan(**data)

    def _fallback_plan(self, topic: str, duration_sec: int) -> ProductionPlan:
        """Minimal sensible plan used when the LLM call fails completely."""
        slug_words = topic.lower().split()[:4]
        keywords = [
            f"{' '.join(slug_words[:2])} visualization",
            f"{' '.join(slug_words)} technology",
            "artificial intelligence neural network",
            "computer data processing",
            "futuristic technology screen",
            "circuit board closeup",
        ]
        return ProductionPlan(
            topic=topic,
            duration_sec=float(duration_sec),
            visual_style="documentary",
            tone="educational",
            mood=Mood.EDUCATIONAL,
            color_palette=["#0d1117", "#1f6feb", "#58a6ff"],
            narrator_voice="en-US-GuyNeural",
            search_keywords=keywords,
            scene_breakdown=[
                "Scene 1: Open with striking visuals of the topic",
                "Scene 2: Core concept explanation with supporting footage",
                "Scene 3: Real-world applications and examples",
                "Scene 4: Future implications and closing thoughts",
            ],
            background_music_keywords=["ambient electronic", "cinematic documentary"],
            sfx_keywords=["digital beep", "server hum", "keyboard typing"],
            text_overlay_style="lower_third",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _save(self, plan: ProductionPlan) -> None:
        out = self.project_dir / "production_plan.json"
        out.write_text(json.dumps(plan.model_dump(), indent=2, default=str))


class ScriptDirector:
    """
    Reviews the raw script from Module 3 using a higher-tier LLM via
    OpenRouter (or local Ollama as fallback), improving narration text,
    b-roll keywords, and segment duration balance.

    Up to _MAX_DIRECTOR_PASSES review passes are run. If any pass fails
    (network error, bad JSON, etc.) the current best script is returned.
    """

    def __init__(self, api_keys: Optional[dict] = None):
        self.api_keys = api_keys if api_keys is not None else load_api_keys()
        self._prompts = load_ollama_prompts()
        self.ollama_url = self.api_keys.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = self.api_keys.get("OLLAMA_MODEL", "llama3.2")
        # Prefer a higher-tier model for the director pass
        self.director_model = self.api_keys.get(
            "DIRECTOR_MODEL", "anthropic/claude-sonnet-4-5"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, script: Script) -> Script:
        """
        Run up to _MAX_DIRECTOR_PASSES review passes on the script.
        Returns the final revised Script, or the original on total failure.
        """
        current = script
        for pass_num in range(1, _MAX_DIRECTOR_PASSES + 1):
            try:
                revised = self._run_review_pass(current, pass_num)
                logger.info("ScriptDirector: pass %d complete", pass_num)
                current = revised
            except Exception as e:
                logger.warning(
                    "ScriptDirector: pass %d failed (%s) — keeping current script",
                    pass_num, e,
                )
                break
        return current

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_review_pass(self, script: Script, pass_num: int) -> Script:
        system_prompt = self._prompts.get("script_director", {}).get("system", "")
        if not system_prompt:
            raise ValueError(
                "script_director.system prompt missing from ollama_prompts.yaml"
            )

        script_json = json.dumps(script.model_dump(), indent=2, default=str)
        user_prompt = (
            f"Review pass {pass_num}. Improve the script below and return ONLY "
            f"the revised JSON with the identical structure:\n\n{script_json}"
        )

        or_key = self.api_keys.get("OPENROUTER_API_KEY", "")
        if or_key:
            raw = self._call_openrouter(system_prompt, user_prompt, or_key)
        else:
            raw = self._call_ollama(system_prompt, user_prompt)

        return self._parse_revised_script(raw, script)

    def _call_openrouter(
        self, system_prompt: str, user_prompt: str, api_key: str
    ) -> str:
        logger.info("ScriptDirector: calling OpenRouter model %s", self.director_model)
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://scripttoreel.local",
                "X-Title": "ScriptToReel-Director",
            },
            json={
                "model": self.director_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 6000,
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        logger.info(
            "ScriptDirector: OpenRouter key absent — using Ollama (%s)",
            self.ollama_model,
        )
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": -1},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _parse_revised_script(self, raw_text: str, original: Script) -> Script:
        """Parse the director's JSON response back into a Script, preserving
        voiceover paths from the original (director doesn't re-generate audio)."""
        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in director response")
            data = json.loads(text[start:end + 1])

        # Re-attach voiceover paths from original (director rewrites narration,
        # not audio files)
        orig_by_id = {s.id: s for s in original.segments}
        for seg in data.get("segments", []):
            orig = orig_by_id.get(seg.get("id"))
            if orig:
                seg.setdefault("voiceover_path", orig.voiceover_path)
                seg.setdefault(
                    "voiceover_duration_sec", orig.voiceover_duration_sec
                )

        data.setdefault("total_voiceover_path", original.total_voiceover_path)
        data.setdefault(
            "total_voiceover_duration_sec", original.total_voiceover_duration_sec
        )

        # Normalise mood and transitions (same logic as module_3)
        mood_raw = str(data.get("mood", "neutral"))
        data["mood"] = next(
            (
                m
                for m in (p.strip() for p in mood_raw.replace("|", ",").split(","))
                if m in _VALID_MOODS
            ),
            "neutral",
        )

        for seg in data.get("segments", []):
            tr = seg.get("transitions", {})
            if isinstance(tr, dict):
                # Accept both "in"/"out" (original LLM format) and
                # "in_transition"/"out_transition" (Pydantic serialised format)
                in_val = tr.get("in") or tr.get("in_transition", "dissolve")
                out_val = tr.get("out") or tr.get("out_transition", "dissolve")
                seg["transitions"] = {
                    "in_transition": _normalize_transition(str(in_val)),
                    "out_transition": _normalize_transition(str(out_val)),
                }
            overlay = seg.get("text_overlay", {})
            if isinstance(overlay, dict) and "start_time" not in overlay:
                overlay.setdefault("start_time", 0.0)

        return Script(**data)


class VisualDirector:
    """
    Reviews the orchestration plan produced by Module 4 for visual coherence:

    1. Color temperature consistency — avoids jarring cool→warm jumps between
       consecutive scenes (max allowed gap: _MAX_TEMP_JUMP steps).
    2. Transition narrative logic — replaces CROSSFADE transitions that bridge
       scenes with very different energy levels with the softer DISSOLVE.

    Both checks are rule-based (no LLM call needed) for reliability and speed.
    """

    _MAX_TEMP_JUMP = 2  # max colour-temperature steps between consecutive scenes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, orch: Orchestration) -> Orchestration:
        """Return a revised Orchestration with coherent color grades and
        transitions.  Logs what was changed."""
        scenes = list(orch.scenes)
        # Fix transitions first (evaluated against original color grades), then
        # smooth colors — prevents color fix from reducing energy gaps before
        # transitions are checked.
        scenes = self._fix_transition_mismatches(scenes)
        scenes = self._fix_color_temperature(scenes)
        n_changed = self._count_changes(orch.scenes, scenes)
        if n_changed:
            logger.info("VisualDirector: revised %d scene(s)", n_changed)
        else:
            logger.info("VisualDirector: no changes needed")
        return orch.model_copy(update={"scenes": scenes})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fix_color_temperature(self, scenes: list[Scene]) -> list[Scene]:
        """Smooth out color temperature jumps larger than _MAX_TEMP_JUMP."""
        result = list(scenes)
        for i in range(1, len(result)):
            prev_temp = _COLOR_TEMP.get(result[i - 1].color_grade, 3)
            curr_temp = _COLOR_TEMP.get(result[i].color_grade, 3)
            if abs(curr_temp - prev_temp) > self._MAX_TEMP_JUMP:
                # Shift one step toward the previous scene's temperature
                adjusted_temp = prev_temp + (1 if curr_temp > prev_temp else -1)
                adjusted_grade = _TEMP_TO_GRADE.get(adjusted_temp, ColorGrade.DOCUMENTARY)
                logger.debug(
                    "VisualDirector: scene %d color_grade %s → %s "
                    "(temperature clash with scene %d)",
                    result[i].id,
                    result[i].color_grade.value,
                    adjusted_grade.value,
                    result[i - 1].id,
                )
                result[i] = result[i].model_copy(
                    update={"color_grade": adjusted_grade}
                )
        return result

    def _fix_transition_mismatches(self, scenes: list[Scene]) -> list[Scene]:
        """Replace CROSSFADE between scenes whose energy differs by >= 3 with
        DISSOLVE — avoids an upbeat crossfade cutting into a somber scene."""
        result = list(scenes)
        for i in range(len(result) - 1):
            curr = result[i]
            nxt = result[i + 1]
            curr_energy = _GRADE_ENERGY.get(curr.color_grade, 3)
            nxt_energy = _GRADE_ENERGY.get(nxt.color_grade, 3)
            if (
                curr.transition_out == TransitionType.CROSSFADE
                and abs(curr_energy - nxt_energy) >= 3
            ):
                logger.debug(
                    "VisualDirector: scene %d→%d replacing CROSSFADE with DISSOLVE "
                    "(energy jump %d→%d)",
                    curr.id, nxt.id, curr_energy, nxt_energy,
                )
                result[i] = curr.model_copy(
                    update={"transition_out": TransitionType.DISSOLVE}
                )
        return result

    @staticmethod
    def _count_changes(original: list[Scene], revised: list[Scene]) -> int:
        count = 0
        for o, r in zip(original, revised):
            if o.color_grade != r.color_grade or o.transition_out != r.transition_out:
                count += 1
        return count
