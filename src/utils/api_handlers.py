"""
API client wrappers for Freesound.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.json_schemas import Asset, AssetRole, AssetSource, AssetType, LicensingInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class APIKeyError(Exception):
    """Raised when an API key is missing or rejected."""


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------

class _BaseClient:
    BASE_URL: str = ""
    _MAX_RETRIES = 3
    _BACKOFF_FACTOR = 1.0

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.session = requests.Session()
        # urllib3 retry for 5xx only; 429 handled manually
        retry = Retry(total=0)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, url: str, params: dict, headers: dict | None = None) -> dict:
        """GET with manual 429 backoff retry."""
        hdrs = headers or {}
        for attempt in range(self._MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, headers=hdrs, timeout=15)
                if resp.status_code == 429:
                    wait = self._BACKOFF_FACTOR * (2 ** attempt)
                    logger.warning("Rate limited by %s, waiting %.1fs", self.BASE_URL, wait)
                    time.sleep(wait)
                    continue
                if resp.status_code == 401:
                    raise APIKeyError(
                        f"API key rejected by {self.__class__.__name__}. "
                        "Check config/api_keys.env."
                    )
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                if hasattr(e, "response") and e.response is not None and e.response.status_code == 401:
                    raise APIKeyError(
                        f"API key rejected by {self.__class__.__name__}."
                    ) from e
                raise
        raise requests.RequestException(f"Max retries exceeded for {url}")


# ---------------------------------------------------------------------------
# Freesound
# ---------------------------------------------------------------------------

class FreesoundClient(_BaseClient):
    BASE_URL = "https://freesound.org/apiv2"

    def search_sounds(self, query: str, per_page: int = 5) -> list[Asset]:
        if not self.api_key:
            logger.warning("FREESOUND_API_KEY not set — skipping Freesound search")
            return []
        data = self._get(
            f"{self.BASE_URL}/search/text/",
            params={
                "token": self.api_key,
                "query": query,
                "page_size": per_page,
                "fields": "id,name,duration,previews,license,username,tags,description",
            },
        )
        return [self._map_sound(s, query) for s in data.get("results", [])]

    def search_music(self, query: str, per_page: int = 3) -> list[Asset]:
        """Search specifically for background music tracks (role=MUSIC, duration≥45s)."""
        if not self.api_key:
            logger.warning("FREESOUND_API_KEY not set — skipping Freesound music search")
            return []
        music_query = f"ambient background music {query}"
        data = self._get(
            f"{self.BASE_URL}/search/text/",
            params={
                "token": self.api_key,
                "query": music_query,
                "page_size": per_page,
                "filter": "duration:[45 TO *]",  # at least 45 seconds
                "fields": "id,name,duration,previews,license,username,tags,description",
                "sort": "rating_desc",
            },
        )
        results = data.get("results", [])
        # Fallback: if nothing found with long-duration filter, try without
        if not results:
            data = self._get(
                f"{self.BASE_URL}/search/text/",
                params={
                    "token": self.api_key,
                    "query": "ambient background instrumental music",
                    "page_size": per_page,
                    "fields": "id,name,duration,previews,license,username,tags,description",
                    "sort": "rating_desc",
                },
            )
            results = data.get("results", [])
        return [self._map_sound(s, music_query, role=AssetRole.MUSIC) for s in results]

    def _map_sound(self, s: dict, query: str, role: AssetRole = AssetRole.SFX) -> Asset:
        previews = s.get("previews", {})
        url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
        tags = s.get("tags", [])
        return Asset(
            id=f"freesound_{s['id']}",
            type=AssetType.AUDIO,
            role=role,
            source=AssetSource.FREESOUND,
            source_id=str(s["id"]),
            source_url=url,
            duration_sec=float(s.get("duration", 0)),
            licensing=LicensingInfo(
                license_type=s.get("license", "CC"),
                attribution_required=True,
                commercial_use="creativecommons.org/licenses/by" in s.get("license", ""),
            ),
            attribution=f"{s.get('username', '')} via Freesound",
            search_query=query,
            visual_tags=tags if isinstance(tags, list) else [],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aspect(w: int, h: int) -> Optional[str]:
    if not w or not h:
        return None
    from math import gcd
    g = gcd(w, h)
    return f"{w // g}:{h // g}"
