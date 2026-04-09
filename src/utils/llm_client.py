"""
Shared LLM client — OpenRouter only.

All modules use call_llm() instead of managing their own HTTP calls.
Model is read from OPENROUTER_MODEL in config/api_keys.env.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from src.utils.config_loader import load_api_keys

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL  = "anthropic/claude-sonnet-4-5"


def call_llm(
    system_prompt: str,
    user_prompt: str,
    api_keys: Optional[dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Send a chat request to OpenRouter and return the response text.

    Raises:
        RuntimeError: if OPENROUTER_API_KEY is missing or the request fails.
    """
    keys = api_keys or load_api_keys()
    api_key = keys.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set in config/api_keys.env. "
            "Get a free key at https://openrouter.ai and add it to the config."
        )

    model = keys.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    logger.info("LLM → OpenRouter  model=%s", model)

    resp = requests.post(
        _OPENROUTER_URL,
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
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
