"""
Production Plan — Pre-production configuration generator.

Generates production_plan.json with intelligent defaults via LLM call.
The user can edit the JSON before running the pipeline.

Usage:
    from src.production_plan import ProductionPlanModule
    pm = ProductionPlanModule(project_dir)
    plan = pm.generate(topic, duration_min)
    # user edits production_plan.json
    plan = pm.load()
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import requests

from src.utils.config_loader import load_api_keys
from src.utils.json_schemas import ProductionPlan, ToneStyle, VisualStyleChoice

logger = logging.getLogger(__name__)

_PLAN_FILENAME = "production_plan.json"

_SYSTEM_PROMPT = (
    "You are a professional video producer and creative director. "
    "Given a video topic, generate a detailed production plan with intelligent defaults. "
    "You MUST respond with valid JSON only — no markdown, no prose, no code fences."
)

_USER_TEMPLATE = """\
Generate a production plan for this video topic: "{topic}"
Duration: {duration_min} minutes

Return a JSON object with EXACTLY these fields:
{{
  "topic": "{topic}",
  "narrator_voice": "Samantha",
  "testimonial_voices": [],
  "tone": "documentary",
  "visual_style": "documentary",
  "target_audience": "describe the target audience here",
  "cultural_context": "relevant cultural notes or empty string",
  "duration_minutes": {duration_min},
  "avoid_list": ["things to avoid showing or mentioning"],
  "image_search_queries": ["8-12 specific visual search queries for stock footage"],
  "script_guidance": "specific instructions for the script writer"
}}

RULES:
- tone: exactly one of: cinematic, documentary, dramatic, uplifting, casual
- visual_style: exactly one of: dark_mysterious, cinematic_warm, documentary, dramatic, bright_modern
- narrator_voice: use macOS voice names like Samantha, Daniel, Alex, Victoria, Karen
- image_search_queries: be SPECIFIC and VISUAL — think stock footage librarian.
  BAD: ["history", "people", "technology"]
  GOOD for "Haunted Places in Pakistan": ["Pakistan ancient fort night", "Lahore Mughal ruins fog",
    "abandoned haveli Pakistan", "dark stone corridor moonlight", "cemetery grave stones Asia dusk"]
- avoid_list: things that would be inappropriate, off-brand, or factually wrong for this topic
- script_guidance: tone, pacing, narrative style instructions (e.g. "conversational, upbeat, use second person")
"""


class ProductionPlanModule:
    def __init__(self, project_dir: Path, api_keys: Optional[dict] = None):
        self.project_dir = Path(project_dir)
        self.api_keys = api_keys if api_keys is not None else load_api_keys()
        self.ollama_url = self.api_keys.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = self.api_keys.get("OLLAMA_MODEL", "llama3.2")

    @property
    def plan_path(self) -> Path:
        return self.project_dir / _PLAN_FILENAME

    def exists(self) -> bool:
        return self.plan_path.exists()

    def generate(self, topic: str, duration_min: float) -> ProductionPlan:
        """Generate a production plan via LLM and save to production_plan.json."""
        logger.info("Generating production plan for topic %r", topic)
        try:
            plan = self._call_llm(topic, duration_min)
        except Exception as e:
            logger.warning("LLM call failed for production plan, using defaults: %s", e)
            plan = self._default_plan(topic, duration_min)
        self.save(plan)
        return plan

    def load(self) -> ProductionPlan:
        """Load existing production_plan.json."""
        if not self.plan_path.exists():
            raise FileNotFoundError(
                f"production_plan.json not found in {self.project_dir}"
            )
        return ProductionPlan(**json.loads(self.plan_path.read_text()))

    def save(self, plan: ProductionPlan) -> None:
        self.plan_path.write_text(json.dumps(plan.model_dump(), indent=2, default=str))
        logger.info("Saved production_plan.json")

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, topic: str, duration_min: float) -> ProductionPlan:
        user_prompt = _USER_TEMPLATE.format(topic=topic, duration_min=duration_min)

        use_openrouter = self.api_keys.get("USE_OPENROUTER", "").lower() == "true"
        or_key = self.api_keys.get("OPENROUTER_API_KEY", "")
        or_model = self.api_keys.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")

        if use_openrouter and or_key:
            raw = self._call_openrouter(_SYSTEM_PROMPT, user_prompt, or_key, or_model)
        else:
            raw = self._call_ollama(_SYSTEM_PROMPT, user_prompt)

        return self._parse_plan(raw, topic, duration_min)

    def _call_openrouter(
        self, system_prompt: str, user_prompt: str, api_key: str, model: str
    ) -> str:
        logger.info("Production plan: using OpenRouter model %s", model)
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
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        logger.info("Production plan: using Ollama model %s", self.ollama_model)
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 1024},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _parse_plan(
        self, raw: str, topic: str, duration_min: float
    ) -> ProductionPlan:
        text = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")
            data = json.loads(text[start : end + 1])

        # Ensure required fields
        data.setdefault("topic", topic)
        data.setdefault("duration_minutes", duration_min)

        # Validate enum values — fall back to defaults if LLM returned something invalid
        valid_tones = {t.value for t in ToneStyle}
        if data.get("tone", "") not in valid_tones:
            data["tone"] = "documentary"
        valid_styles = {v.value for v in VisualStyleChoice}
        if data.get("visual_style", "") not in valid_styles:
            data["visual_style"] = "documentary"

        return ProductionPlan(**data)

    def _default_plan(self, topic: str, duration_min: float) -> ProductionPlan:
        """Return sensible defaults without LLM (used when LLM is unavailable)."""
        # Import here to avoid circular dependency
        from src.module_1_research import _topic_to_queries

        topic_queries = _topic_to_queries(topic)
        queries = [topic] + [q for q in topic_queries if q != topic]
        return ProductionPlan(
            topic=topic,
            duration_minutes=duration_min,
            image_search_queries=queries[:6],
        )
