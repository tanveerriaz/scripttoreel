"""
Shared LLM client — OpenRouter only.

All modules use call_llm() instead of managing their own HTTP calls.
Model is read from OPENROUTER_MODEL in config/api_keys.env.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from src.utils.config_loader import load_api_keys

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL  = "anthropic/claude-sonnet-4-5"


def _openrouter_trust_env() -> bool:
    """Honor HTTP(S)_PROXY only when OPENROUTER_USE_SYSTEM_PROXY is enabled.

    Many environments set a global proxy (VPN, IDE, old shell config) that returns
    403 for CONNECT to api hosts. OpenRouter calls default to a direct connection.
    """
    v = os.environ.get("OPENROUTER_USE_SYSTEM_PROXY", "").strip().lower()
    return v in ("1", "true", "yes", "on")


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

    session = requests.Session()
    session.trust_env = _openrouter_trust_env()
    resp = session.post(
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
