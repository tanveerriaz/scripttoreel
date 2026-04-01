"""
Module 1 — Research & Asset Discovery.

Searches Pexels, Pixabay, Unsplash, and Freesound for assets related to
the project topic. Downloads all found assets and writes assets_raw.json.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from src.project_manager import load_project, update_pipeline_status
from src.utils.api_handlers import (
    APIKeyError,
    FreesoundClient,
    PexelsClient,
    PixabayClient,
    UnsplashClient,
)
from src.utils.config_loader import load_api_keys
from src.utils.json_schemas import Asset, AssetType, ModuleStatus

logger = logging.getLogger(__name__)

_ASSETS_PER_SOURCE = 8   # how many assets to fetch per query per source


class ResearchModule:
    def __init__(self, project_dir: Path, api_keys: Optional[dict] = None):
        self.project_dir = Path(project_dir)
        self.api_keys = api_keys if api_keys is not None else load_api_keys()
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[Asset]:
        """Run the full research pipeline for this project."""
        project_json = self.project_dir / "project.json"
        if not project_json.exists():
            raise FileNotFoundError(f"project.json not found in {self.project_dir}")

        meta = json.loads(project_json.read_text())
        topic = meta.get("topic", "")
        logger.info("Module 1: researching topic %r", topic)

        all_assets: list[Asset] = []

        # --- Pexels ---
        pexels = PexelsClient(self.api_keys.get("PEXELS_API_KEY"))
        all_assets.extend(self._safe_search(pexels.search_videos, topic, "Pexels videos"))
        all_assets.extend(self._safe_search(pexels.search_images, topic, "Pexels images"))

        # --- Pixabay ---
        pixabay = PixabayClient(self.api_keys.get("PIXABAY_API_KEY"))
        all_assets.extend(self._safe_search(pixabay.search_videos, topic, "Pixabay videos"))
        all_assets.extend(self._safe_search(pixabay.search_images, topic, "Pixabay images"))

        # --- Unsplash ---
        unsplash = UnsplashClient(self.api_keys.get("UNSPLASH_ACCESS_KEY"))
        all_assets.extend(self._safe_search(unsplash.search_photos, topic, "Unsplash photos"))

        # --- Freesound ---
        freesound = FreesoundClient(self.api_keys.get("FREESOUND_API_KEY"))
        all_assets.extend(self._safe_search(freesound.search_sounds, topic, "Freesound audio"))

        logger.info("Found %d assets total; starting downloads", len(all_assets))

        # Download all assets
        downloaded: list[Asset] = []
        for asset in tqdm(all_assets, desc="Downloading assets", unit="file"):
            updated = self.download_asset(asset)
            downloaded.append(updated)

        self.save_assets_raw(downloaded)

        # Update pipeline status
        self._update_status(ModuleStatus.COMPLETE, total_assets=len(downloaded))

        return downloaded

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _safe_search(self, fn, query: str, label: str) -> list[Asset]:
        """Call a search function; on APIKeyError or network error, warn and return []."""
        try:
            results = fn(query, per_page=_ASSETS_PER_SOURCE)
            logger.info("%s: found %d results", label, len(results))
            return results
        except APIKeyError as e:
            logger.warning("Skipping %s — %s", label, e)
            return []
        except Exception as e:
            logger.warning("Error searching %s: %s", label, e)
            return []

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_asset(self, asset: Asset) -> Asset:
        """
        Download the asset's source_url to the appropriate local folder.
        Idempotent: skips if local_path already set and file exists.
        On failure: logs warning, returns asset with local_path=None.
        """
        # Already downloaded
        if asset.local_path and Path(asset.local_path).exists():
            return asset

        if not asset.source_url:
            logger.warning("Asset %s has no source_url — skipping", asset.id)
            return asset

        dest_dir = self._dest_dir(asset.type)
        ext = _guess_extension(asset.source_url, asset.type)
        filename = f"{asset.id}{ext}"
        dest = dest_dir / filename

        try:
            with requests.get(asset.source_url, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        fh.write(chunk)
            logger.info("Downloaded %s → %s", asset.id, dest)
            asset = asset.model_copy(update={"local_path": str(dest), "filename": filename})
        except Exception as e:
            logger.warning("Failed to download %s (%s): %s", asset.id, asset.source_url, e)

        return asset

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_assets_raw(self, assets: list[Asset]) -> None:
        out = self.project_dir / "assets_raw.json"
        out.write_text(
            json.dumps([a.model_dump() for a in assets], indent=2, default=str)
        )
        logger.info("Saved assets_raw.json with %d assets", len(assets))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        for sub in ["assets/raw/video", "assets/raw/image", "assets/raw/audio", "assets/processed", "output"]:
            (self.project_dir / sub).mkdir(parents=True, exist_ok=True)

    def _dest_dir(self, asset_type: AssetType) -> Path:
        mapping = {
            AssetType.VIDEO: "assets/raw/video",
            AssetType.IMAGE: "assets/raw/image",
            AssetType.AUDIO: "assets/raw/audio",
            AssetType.SFX: "assets/raw/audio",
        }
        return self.project_dir / mapping.get(asset_type, "assets/raw")

    def _update_status(self, status: ModuleStatus, **kwargs) -> None:
        project_json = self.project_dir / "project.json"
        if not project_json.exists():
            return
        meta = json.loads(project_json.read_text())
        project_id = meta.get("project_id")
        if not project_id:
            return
        try:
            update_pipeline_status(
                project_id,
                "module_1_research",
                status,
                projects_root=self.project_dir.parent,
                **kwargs,
            )
        except Exception as e:
            logger.warning("Could not update pipeline status: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_extension(url: str, asset_type: AssetType) -> str:
    """Guess file extension from URL, falling back to type-appropriate default."""
    parsed = urlparse(url)
    path = parsed.path.split("?")[0]
    ext = Path(path).suffix
    if ext and len(ext) <= 5:
        return ext
    defaults = {
        AssetType.VIDEO: ".mp4",
        AssetType.IMAGE: ".jpg",
        AssetType.AUDIO: ".mp3",
        AssetType.SFX: ".mp3",
    }
    return defaults.get(asset_type, ".bin")
