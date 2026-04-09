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
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

from src.project_manager import load_project, update_pipeline_status
from src.utils.api_handlers import (
    APIKeyError,
    FreesoundClient,
)
from src.utils.config_loader import load_api_keys
from src.utils.json_schemas import Asset, AssetSource, AssetType, ModuleStatus
from src.utils.local_image_generator import LocalSDXLClient, build_image_prompt

logger = logging.getLogger(__name__)

_ASSETS_PER_SOURCE = 8   # how many assets to fetch per query per source

# Tags that indicate an asset is a cartoon, 3D render, illustration, or contains
# inappropriate content. Assets matching any of these are silently dropped.
_FORBIDDEN_TAGS: frozenset[str] = frozenset({
    "3d", "3d render", "3d rendering", "3d rendered", "cgi",
    "cartoon", "animated", "animation", "anime",
    "illustration", "clipart", "clip art", "drawing", "sketch",
    "vector", "vector art", "digital art", "artwork",
    "nude", "naked", "nudity", "topless", "adult",
})


class ResearchModule:
    def __init__(self, project_dir: Path, api_keys: Optional[dict] = None):
        self.project_dir = Path(project_dir)
        self.api_keys = api_keys if api_keys is not None else load_api_keys()
        self._warned_sources: set[str] = set()  # tracks which sources already warned
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
        duration_min = float(meta.get("duration_min", 5.0))
        logger.info("Module 1: researching topic %r (%.1f min)", topic, duration_min)

        # Scale fetch budget to video length.
        # Rule: need 1 video + 1 image per 5 sec, fetch ~4× that for variety after filtering.
        #   ≤0.5 min (30s) → 6 clips → need 12 each type → 5/source, 3 queries
        #   ≤2 min         → 24 clips → 8/source, 4 queries
        #   >2 min         → 48+ clips → 10/source, 5 queries
        if duration_min <= 0.5:
            per_source = 5
            max_queries = 3
        elif duration_min <= 2.0:
            per_source = 8
            max_queries = 4
        else:
            per_source = 10
            max_queries = 5

        # Build a diverse set of search queries — topic + extracted sub-queries
        # Also incorporate B-roll keywords from script.json if already generated
        queries = self._build_search_queries(topic)[:max_queries]
        logger.info("Search queries (%d): %s", len(queries), queries)

        all_assets: list[Asset] = []
        seen_ids: set = set()  # deduplicate across queries

        def _add(assets):
            for a in assets:
                if a.id not in seen_ids:
                    seen_ids.add(a.id)
                    all_assets.append(a)

        freesound = FreesoundClient(self.api_keys.get("FREESOUND_API_KEY"))

        # SFX: ambient sounds for scene atmosphere
        _add(self._safe_search(freesound.search_sounds, topic, "Freesound SFX"))
        if len(queries) > 1:
            _add(self._safe_search(freesound.search_sounds, queries[1], f"Freesound SFX [{queries[1]}]"))

        # Background music: tone-aware queries for variety
        # Map tone → diverse Freesound search queries
        _TONE_MUSIC_QUERIES: dict[str, list[str]] = {
            "documentary":  ["ambient cinematic background", "documentary score", "atmospheric pad"],
            "educational":  ["ambient cinematic background", "documentary score", "atmospheric pad"],
            "dramatic":     ["dramatic orchestral", "tension building music", "epic cinematic"],
            "suspenseful":  ["suspenseful ambient", "dark tension music", "thriller soundtrack"],
            "uplifting":    ["uplifting acoustic", "positive background music", "inspiring piano"],
            "cinematic":    ["cinematic soundtrack", "film score ambient", "orchestral background"],
            "casual":       ["lo-fi background", "chill acoustic", "light background music"],
        }
        plan_tone = ""
        plan_music_kw: list[str] = []
        plan_path_music = self.project_dir / "production_plan.json"
        if plan_path_music.exists():
            try:
                plan_data = json.loads(plan_path_music.read_text())
                plan_tone = plan_data.get("tone", "")
                plan_music_kw = plan_data.get("background_music_keywords", [])
            except Exception:
                pass

        music_queries = _TONE_MUSIC_QUERIES.get(plan_tone, [f"ambient background music {topic}"])
        # Also add plan-specified music keywords
        if plan_music_kw:
            music_queries = plan_music_kw[:2] + music_queries[:1]

        for mq in music_queries[:3]:
            _add(self._safe_search(freesound.search_music, mq, f"Freesound music [{mq}]"))

        # AI image generation — locally via SDXL + MPS (graceful skip if not installed)
        tone = ""
        plan_path = self.project_dir / "production_plan.json"
        if plan_path.exists():
            try:
                tone = json.loads(plan_path.read_text()).get("tone", "")
            except Exception:
                pass

        # Collect per-segment sdxl_prompt fields from script.json (highest quality prompts)
        segment_sdxl_prompts: list[str] = []
        script_path = self.project_dir / "script.json"
        if script_path.exists():
            try:
                script_data = json.loads(script_path.read_text())
                for seg in script_data.get("segments", []):
                    sp = seg.get("sdxl_prompt", "")
                    if sp and len(sp) > 20:
                        segment_sdxl_prompts.append(sp)
            except Exception:
                pass

        # SDXL is the sole visual source — generate enough images for full coverage.
        # Rule: 1 image per 5s of video + 50% buffer. Cap per query at 8 for speed.
        import math as _math  # noqa: PLC0415
        total_needed = (duration_min * 60) / 5
        images_per_query = max(2, min(8, _math.ceil(total_needed * 1.5 / max_queries)))

        # Use fewer inference steps for short videos to keep generation time reasonable:
        # ≤1min → 22 steps (~30s/image), >1min → 32 steps (~50s/image, sharper detail)
        num_steps = 22 if duration_min <= 1.0 else 32

        # Build SDXL prompt list: prefer per-segment sdxl_prompts, then plan prompts, then queries
        sdxl_prompts: list[str] = []
        if segment_sdxl_prompts:
            # Per-segment prompts are most specific — generate 2 candidates each
            sdxl_prompts = segment_sdxl_prompts
            images_per_query = 2  # 2 candidates per scene, orchestrator picks best
            logger.info("SDXL: using %d per-segment sdxl_prompt fields", len(sdxl_prompts))
        else:
            # Fall back to queries + style modifier
            sdxl_prompts = [build_image_prompt(q, topic, tone) for q in queries]

        try:
            img_dir = self.project_dir / "assets" / "raw" / "image"
            sdxl = LocalSDXLClient(output_dir=img_dir)
            total_generated = 0
            with Progress(
                TextColumn("[magenta]   Generating AI images[/magenta]"),
                BarColumn(bar_width=36),
                TaskProgressColumn(),
                TextColumn("[dim]{task.completed}/{task.total} prompts[/dim]"),
                transient=True,
            ) as progress:
                task = progress.add_task("", total=len(sdxl_prompts))
                for prompt in sdxl_prompts:
                    ai_assets = sdxl.generate(prompt, num_images=images_per_query, num_steps=num_steps)
                    _add(ai_assets)
                    total_generated += len(ai_assets)
                    logger.debug("SDXL: %d images for %r", len(ai_assets), prompt[:60])
                    progress.advance(task)
            logger.info("SDXL: generated %d AI images total", total_generated)
        except RuntimeError as e:
            logger.warning("SDXL skipped — %s", e)
        except Exception as e:
            logger.warning("SDXL generation failed: %s", e)

        # Re-register any SDXL images already on disk from a previous run.
        # This ensures orphaned images are picked up even when SDXL is skipped.
        img_dir = self.project_dir / "assets" / "raw" / "image"
        existing_imgs = [p for p in img_dir.glob("sdxl_*.jpg") if p.stat().st_size > 1000]
        if existing_imgs:
            logger.info("SDXL: found %d existing images on disk — re-registering", len(existing_imgs))
            for p in existing_imgs:
                orphan = Asset(
                    id=p.stem,
                    type=AssetType.IMAGE,
                    source=AssetSource.GENERATED,
                    local_path=str(p),
                )
                _add([orphan])

        logger.info("Found %d unique assets total; starting downloads", len(all_assets))

        # Download all assets — single-line Rich progress bar
        downloaded: list[Asset] = []
        with Progress(
            TextColumn("[cyan]   Downloading assets[/cyan]"),
            BarColumn(bar_width=36),
            TaskProgressColumn(),
            TextColumn("[dim]{task.completed}/{task.total} files[/dim]"),
            TimeRemainingColumn(),
            transient=True,  # clears the bar line when done
        ) as progress:
            task = progress.add_task("", total=len(all_assets))
            for asset in all_assets:
                updated = self.download_asset(asset)
                downloaded.append(updated)
                progress.advance(task)

        self.save_assets_raw(downloaded)

        # Update pipeline status
        self._update_status(ModuleStatus.COMPLETE, total_assets=len(downloaded))

        return downloaded

    # ------------------------------------------------------------------
    # Search query building
    # ------------------------------------------------------------------

    def _build_search_queries(self, topic: str) -> list[str]:
        """Return specific search queries derived from the topic.

        Priority order:
        1. image_search_queries from production_plan.json (most specific, LLM-generated)
        2. B-roll keywords from script.json (if Module 3 already ran — new order or re-runs)
        3. Key noun phrases extracted from the topic
        4. The raw topic as fallback
        """
        queries: list[str] = []

        # 1. Load sdxl_visual_prompts and search_keywords from production_plan.json
        #    sdxl_visual_prompts are rich camera-ready descriptions → best SDXL input
        #    search_keywords are fallback if visual prompts not present
        plan_path = self.project_dir / "production_plan.json"
        if plan_path.exists():
            try:
                plan_data = json.loads(plan_path.read_text())

                # HIGHEST PRIORITY: rich visual prompts designed for SDXL
                visual_prompts = [
                    p.strip() for p in plan_data.get("sdxl_visual_prompts", [])
                    if p.strip()
                ]
                if visual_prompts:
                    queries = visual_prompts[:6]
                    logger.info(
                        "Using %d SDXL visual prompts from production_plan.json", len(queries)
                    )
                else:
                    # Fallback: plain search_keywords from AI Director
                    plan_keywords = [
                        kw.strip() for kw in plan_data.get("search_keywords", [])
                        if kw.strip()
                    ]
                    if plan_keywords:
                        queries = plan_keywords[:4]
                        logger.info(
                            "Using %d AI Director keywords from production_plan.json", len(queries)
                        )

                # Also pull image_search_queries if present
                plan_queries = plan_data.get("image_search_queries", [])
                for q in plan_queries:
                    q = q.strip()
                    if q and q not in queries:
                        queries.append(q)
            except Exception as e:
                logger.warning("Could not read production_plan.json for keywords: %s", e)

        # 2. Pull B-roll keywords from existing script.json
        # (populated when Module 3 runs before Module 1 in the new order)
        script_path = self.project_dir / "script.json"
        if script_path.exists():
            try:
                script_data = json.loads(script_path.read_text())
                kw_set: dict[str, None] = {}  # ordered dedup
                for seg in script_data.get("segments", []):
                    for kw in seg.get("b_roll_keywords", []):
                        kw = kw.strip()
                        if kw and kw.lower() not in ("technology", "science", "people",
                                                       "nature", "business", "future",
                                                       "innovation", "modern"):
                            kw_set[kw] = None
                script_kws = [k for k in kw_set.keys() if k not in queries]
                if script_kws:
                    queries.extend(script_kws[:4])
                    logger.info("Added %d B-roll keywords from script.json", len(script_kws[:4]))
            except Exception as e:
                logger.warning("Could not read script.json for keywords: %s", e)

        # 3. Extract specific terms from the topic itself
        topic_queries = _topic_to_queries(topic)
        for q in topic_queries:
            if q not in queries:
                queries.append(q)

        # 4. Ensure the raw topic is always searched
        if topic not in queries:
            queries.insert(0, topic)

        # Cap: 8 queries when plan exists (more specific), 4 otherwise
        cap = 8 if plan_path.exists() else 4
        return queries[:cap]

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _safe_search(self, fn, query: str, label: str, per_page: int = _ASSETS_PER_SOURCE) -> list[Asset]:
        """Call a search function; on APIKeyError or network error, warn and return []."""
        try:
            results = fn(query, per_page=per_page)
            before = len(results)
            results = [a for a in results if not self._has_forbidden_tags(a)]
            dropped = before - len(results)
            if dropped:
                logger.debug("%s: dropped %d asset(s) with forbidden tags", label, dropped)
            logger.debug("%s: found %d results", label, len(results))
            return results
        except APIKeyError as e:
            # Log once per source class (strip the query suffix for dedup key)
            source = label.split("[")[0].strip()
            if source not in self._warned_sources:
                self._warned_sources.add(source)
                logger.warning("⚠  %s unavailable — %s", source, e)
            return []
        except Exception as e:
            logger.debug("Error searching %s: %s", label, e)
            return []

    @staticmethod
    def _has_forbidden_tags(asset: Asset) -> bool:
        """Return True if any of the asset's visual_tags match _FORBIDDEN_TAGS."""
        tags = {t.lower().strip() for t in (asset.visual_tags or [])}
        return bool(tags & _FORBIDDEN_TAGS)

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
            logger.debug("Downloaded %s → %s", asset.id, dest)
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


def _topic_to_queries(topic: str) -> list[str]:
    """Derive 2-3 specific search queries from the topic string.

    Strategy:
    - Strip question words (what, how, why, is, are, does, etc.)
    - Remove stop words and short words to get content nouns
    - Build focused compound queries
    """
    # Common stop words to filter — includes question words so
    # "What is Vibe Coding" → ["Vibe", "Coding"] not ["What", "Vibe"]
    _STOPS = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        # question/linking words
        "what", "how", "why", "when", "where", "who", "which",
        "is", "are", "was", "were", "does", "did", "will", "can",
        "its", "this", "that", "with", "from", "into", "about",
        "but", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "about", "with", "from", "by", "as",
        "its", "their", "our", "my", "your", "this", "that", "these", "those",
        "how", "what", "when", "where", "why", "who", "rise", "rise", "story",
        "history", "world", "guide", "complete", "top", "best",
    }

    words = topic.split()
    # Content words (4+ chars, not stop words)
    content_words = [w for w in words if len(w) >= 4 and w.lower() not in _STOPS]

    queries: list[str] = []

    # Pair first two content words for a compound query
    if len(content_words) >= 2:
        queries.append(f"{content_words[0]} {content_words[1]}")
    elif content_words:
        queries.append(content_words[0])

    # Add each remaining content word as a standalone query
    for w in content_words[2:4]:
        queries.append(w)

    return queries
