"""
AI Director — pre-production planning module.

Called during `--init` to generate production_plan.json before any other
module runs.  Uses the same LLM backend as Module 3 (OpenRouter or Ollama).

Output: <project_dir>/production_plan.json
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
from src.utils.json_schemas import Mood, ProductionPlan

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 3

_VALID_MOODS = {"dark", "mysterious", "uplifting", "dramatic", "educational",
                "horror", "neutral", "suspenseful", "melancholic"}
_VALID_VISUAL_STYLES = {"documentary", "cinematic", "minimalist"}
_VALID_TONES = {"educational", "dramatic", "uplifting", "suspenseful"}
_VALID_OVERLAY_STYLES = {"lower_third", "minimal", "bold", "subtitle"}


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
                    raw = self._call_openrouter(system_prompt, user_prompt, or_key, or_model)
                else:
                    raw = self._call_ollama(system_prompt, user_prompt)
                return self._parse(raw, topic, duration_sec)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning("AI Director attempt %d failed: %s", attempt + 1, e)
                continue

        # Fallback: return a sensible default plan so --init never hard-fails
        logger.warning("AI Director LLM failed after %d attempts; using fallback plan", _MAX_LLM_RETRIES)
        return self._fallback_plan(topic, duration_sec)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
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

    def _call_openrouter(
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

    def _parse(self, raw_text: str, topic: str, duration_sec: int) -> ProductionPlan:
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
