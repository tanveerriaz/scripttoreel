"""
API client wrappers for Pexels, Pixabay, Unsplash, and Freesound.
Each client handles auth, rate limits (exponential backoff), and maps responses
to ScriptToReel Asset Pydantic objects.
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
# Pexels
# ---------------------------------------------------------------------------

class PexelsClient(_BaseClient):
    BASE_URL = "https://api.pexels.com/v1"

    def _headers(self) -> dict:
        if not self.api_key:
            raise APIKeyError("PEXELS_API_KEY is not set. Add it to config/api_keys.env")
        return {"Authorization": self.api_key}

    def search_videos(self, query: str, per_page: int = 10) -> list[Asset]:
        data = self._get(
            f"{self.BASE_URL}/videos/search",
            params={"query": query, "per_page": per_page, "orientation": "landscape"},
            headers=self._headers(),
        )
        return [self._map_video(v, query) for v in data.get("videos", [])]

    def search_images(self, query: str, per_page: int = 10) -> list[Asset]:
        data = self._get(
            f"{self.BASE_URL}/search",
            params={"query": query, "per_page": per_page, "orientation": "landscape"},
            headers=self._headers(),
        )
        return [self._map_photo(p, query) for p in data.get("photos", [])]

    def _map_video(self, v: dict, query: str) -> Asset:
        best_file = max(
            v.get("video_files", [{}]),
            key=lambda f: f.get("width", 0),
            default={},
        )
        w = best_file.get("width", 0)
        h = best_file.get("height", 0)
        return Asset(
            id=f"pexels_vid_{v['id']}",
            type=AssetType.VIDEO,
            role=AssetRole.B_ROLL,
            source=AssetSource.PEXELS,
            source_id=str(v["id"]),
            source_url=best_file.get("link") or v.get("url"),
            duration_sec=float(v.get("duration", 0)),
            resolution=f"{w}x{h}" if w and h else None,
            aspect_ratio=_aspect(w, h),
            licensing=LicensingInfo(license_type="Pexels License", attribution_required=False, commercial_use=True),
            attribution=f"{v.get('user', {}).get('name', '')} via Pexels",
            search_query=query,
        )

    def _map_photo(self, p: dict, query: str) -> Asset:
        w, h = p.get("width", 0), p.get("height", 0)
        return Asset(
            id=f"pexels_img_{p['id']}",
            type=AssetType.IMAGE,
            role=AssetRole.B_ROLL,
            source=AssetSource.PEXELS,
            source_id=str(p["id"]),
            source_url=p.get("src", {}).get("original") or p.get("src", {}).get("large"),
            resolution=f"{w}x{h}" if w and h else None,
            aspect_ratio=_aspect(w, h),
            licensing=LicensingInfo(license_type="Pexels License", attribution_required=False, commercial_use=True),
            attribution=f"{p.get('photographer', '')} via Pexels",
            search_query=query,
        )


# ---------------------------------------------------------------------------
# Pixabay
# ---------------------------------------------------------------------------

class PixabayClient(_BaseClient):
    BASE_URL = "https://pixabay.com/api"

    def search_videos(self, query: str, per_page: int = 10) -> list[Asset]:
        if not self.api_key:
            logger.warning("PIXABAY_API_KEY not set — skipping Pixabay video search")
            return []
        data = self._get(
            f"{self.BASE_URL}/videos/",
            params={"key": self.api_key, "q": query, "per_page": per_page, "orientation": "horizontal", "safesearch": "true"},
        )
        return [self._map_video(v, query) for v in data.get("hits", [])]

    def search_images(self, query: str, per_page: int = 10) -> list[Asset]:
        if not self.api_key:
            logger.warning("PIXABAY_API_KEY not set — skipping Pixabay image search")
            return []
        data = self._get(
            f"{self.BASE_URL}/",
            params={"key": self.api_key, "q": query, "per_page": per_page, "orientation": "horizontal", "image_type": "photo", "safesearch": "true"},
        )
        return [self._map_image(img, query) for img in data.get("hits", [])]

    def _map_video(self, v: dict, query: str) -> Asset:
        large = v.get("videos", {}).get("large", {})
        w, h = large.get("width", 0), large.get("height", 0)
        return Asset(
            id=f"pixabay_vid_{v['id']}",
            type=AssetType.VIDEO,
            role=AssetRole.B_ROLL,
            source=AssetSource.PIXABAY,
            source_id=str(v["id"]),
            source_url=large.get("url") or v.get("pageURL"),
            duration_sec=float(v.get("duration", 0)),
            resolution=f"{w}x{h}" if w and h else None,
            aspect_ratio=_aspect(w, h),
            licensing=LicensingInfo(license_type="Pixabay License", commercial_use=True),
            attribution=f"{v.get('user', '')} via Pixabay",
            search_query=query,
            visual_tags=[t.strip() for t in v.get("tags", "").split(",") if t.strip()],
        )

    def _map_image(self, img: dict, query: str) -> Asset:
        w, h = img.get("imageWidth", 0), img.get("imageHeight", 0)
        return Asset(
            id=f"pixabay_img_{img['id']}",
            type=AssetType.IMAGE,
            role=AssetRole.B_ROLL,
            source=AssetSource.PIXABAY,
            source_id=str(img["id"]),
            source_url=img.get("largeImageURL") or img.get("webformatURL"),
            resolution=f"{w}x{h}" if w and h else None,
            aspect_ratio=_aspect(w, h),
            licensing=LicensingInfo(license_type="Pixabay License", commercial_use=True),
            attribution=f"{img.get('user', '')} via Pixabay",
            search_query=query,
            visual_tags=[t.strip() for t in img.get("tags", "").split(",") if t.strip()],
        )


# ---------------------------------------------------------------------------
# Unsplash
# ---------------------------------------------------------------------------

class UnsplashClient(_BaseClient):
    BASE_URL = "https://api.unsplash.com"

    def __init__(self, access_key: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(api_key=access_key or api_key)

    def search_photos(self, query: str, per_page: int = 10) -> list[Asset]:
        if not self.api_key:
            logger.warning("UNSPLASH_ACCESS_KEY not set — skipping Unsplash search")
            return []
        data = self._get(
            f"{self.BASE_URL}/search/photos",
            params={"query": query, "per_page": per_page, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {self.api_key}"},
        )
        return [self._map_photo(p, query) for p in data.get("results", [])]

    def _map_photo(self, p: dict, query: str) -> Asset:
        w, h = p.get("width", 0), p.get("height", 0)
        urls = p.get("urls", {})
        return Asset(
            id=f"unsplash_{p['id']}",
            type=AssetType.IMAGE,
            role=AssetRole.B_ROLL,
            source=AssetSource.UNSPLASH,
            source_id=p["id"],
            source_url=urls.get("raw") or urls.get("full") or urls.get("regular"),
            resolution=f"{w}x{h}" if w and h else None,
            aspect_ratio=_aspect(w, h),
            licensing=LicensingInfo(license_type="Unsplash License", commercial_use=True),
            attribution=f"{p.get('user', {}).get('name', '')} via Unsplash",
            search_query=query,
        )


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
