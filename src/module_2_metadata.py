"""
Module 2 — Asset Metadata Extraction.

Enriches every asset with:
- Video: ffprobe (duration, resolution, FPS, codec)
- Image: Pillow (dimensions, format, aspect ratio)
- Audio: librosa (duration, sample rate, BPM)
- Visual: OpenCV K-means dominant colors
- quality_score (0-10)
- Writes assets.json
"""
from __future__ import annotations

import json
import logging
import subprocess
from math import gcd
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.project_manager import update_pipeline_status
from src.utils.json_schemas import (
    Asset,
    AssetType,
    AudioMetadata,
    ImageMetadata,
    Mood,
    ModuleStatus,
    VideoMetadata,
    VisualDNA,
)

logger = logging.getLogger(__name__)

_QUALITY_THRESHOLD = 5.0   # assets with score >= this are marked ready_for_use


class MetadataModule:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[Asset]:
        raw_path = self.project_dir / "assets_raw.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"assets_raw.json not found in {self.project_dir}")

        raw_data = json.loads(raw_path.read_text())
        assets = [Asset(**item) for item in raw_data]
        logger.info("Module 2: processing metadata for %d assets", len(assets))

        enriched: list[Asset] = []
        for asset in tqdm(assets, desc="Extracting metadata", unit="asset"):
            enriched.append(self._enrich(asset))

        self.save_assets_json(enriched)
        self._update_status(ModuleStatus.COMPLETE)
        return enriched

    # ------------------------------------------------------------------
    # Per-asset enrichment
    # ------------------------------------------------------------------

    def _enrich(self, asset: Asset) -> Asset:
        if not asset.local_path or not Path(asset.local_path).exists():
            logger.warning("Asset %s has no local file — skipping metadata", asset.id)
            return asset

        path = Path(asset.local_path)
        updates: dict = {}

        try:
            if asset.type == AssetType.VIDEO:
                vm = self.extract_video_metadata(path)
                updates["video_metadata"] = vm
                updates["duration_sec"] = vm.duration_sec
                updates["resolution"] = f"{vm.width}x{vm.height}" if vm.width else asset.resolution
                updates["fps"] = vm.fps
                updates["aspect_ratio"] = self._compute_aspect_ratio(vm.width, vm.height)
                # extract colors from first frame
                colors = self.extract_dominant_colors(path, n=5)
                updates["color_palette"] = colors

            elif asset.type == AssetType.IMAGE:
                im = self.extract_image_metadata(path)
                updates["image_metadata"] = im
                updates["resolution"] = f"{im.width}x{im.height}"
                updates["aspect_ratio"] = self._compute_aspect_ratio(im.width, im.height)
                colors = self.extract_dominant_colors(path, n=5)
                updates["color_palette"] = colors

            elif asset.type in (AssetType.AUDIO, AssetType.SFX):
                am = self.extract_audio_metadata(path)
                updates["audio_metadata"] = am
                updates["duration_sec"] = am.duration_sec

            # Build VisualDNA for visual assets
            if asset.type in (AssetType.VIDEO, AssetType.IMAGE):
                vdna = self.compute_visual_dna(asset.model_copy(update=updates))
                updates["visual_dna"] = vdna
                updates["dominant_mood"] = vdna.dominant_mood
                updates["visual_tags"] = vdna.visual_tags

        except Exception as e:
            logger.warning("Metadata extraction failed for %s: %s", asset.id, e)

        # Compute quality score on updated asset
        updated_asset = asset.model_copy(update=updates)
        score = self.compute_quality_score(updated_asset)
        updated_asset = updated_asset.model_copy(update={
            "quality_score": score,
            "ready_for_use": score >= _QUALITY_THRESHOLD,
        })
        return updated_asset

    # ------------------------------------------------------------------
    # Story 2.1 — Video metadata (ffprobe)
    # ------------------------------------------------------------------

    def extract_video_metadata(self, path: Path) -> VideoMetadata:
        if not path.exists():
            raise FileNotFoundError(path)

        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        fmt = probe.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        w = int(video_stream.get("width", 0))
        h = int(video_stream.get("height", 0))
        fps = _parse_fps(video_stream.get("r_frame_rate", "0/1"))
        duration = float(fmt.get("duration") or video_stream.get("duration", 0))
        codec = video_stream.get("codec_name", "")
        nb_frames = int(video_stream.get("nb_frames") or 0)
        if not nb_frames and fps and duration:
            nb_frames = int(fps * duration)

        bitrate = int(fmt.get("bit_rate", 0)) // 1000 if fmt.get("bit_rate") else None

        return VideoMetadata(
            duration_sec=round(duration, 3),
            width=w,
            height=h,
            fps=round(fps, 3),
            codec=codec,
            bitrate_kbps=bitrate,
            has_audio=audio_stream is not None,
            frame_count=nb_frames,
        )

    # ------------------------------------------------------------------
    # Story 2.2 — Image metadata (Pillow)
    # ------------------------------------------------------------------

    def extract_image_metadata(self, path: Path) -> ImageMetadata:
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as img:
            w, h = img.size
            fmt = img.format or path.suffix.lstrip(".").upper()
            mode = img.mode
        size = path.stat().st_size
        return ImageMetadata(width=w, height=h, format=fmt, mode=mode, file_size_bytes=size)

    # ------------------------------------------------------------------
    # Story 2.3 — Audio metadata (librosa)
    # ------------------------------------------------------------------

    def extract_audio_metadata(self, path: Path) -> AudioMetadata:
        if not path.exists():
            raise FileNotFoundError(path)

        import concurrent.futures
        _LIBROSA_TIMEOUT = 30  # seconds

        def _load_with_librosa():
            y, sr = librosa.load(str(path), sr=None, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = float(tempo) if tempo else None
            except Exception:
                bpm = None
            return duration, int(sr), bpm

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_load_with_librosa)
                duration, sr, bpm = future.result(timeout=_LIBROSA_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.warning("librosa.load() timed out for %s — returning default AudioMetadata", path)
            return AudioMetadata()
        except Exception as e:
            logger.warning("librosa failed for %s: %s — returning default AudioMetadata", path, e)
            return AudioMetadata()

        return AudioMetadata(duration_sec=round(duration, 3), sample_rate=sr, bpm=bpm)

    # ------------------------------------------------------------------
    # Story 2.4 — Dominant color extraction (OpenCV K-means)
    # ------------------------------------------------------------------

    def extract_dominant_colors(self, path: Path, n: int = 5) -> list[str]:
        """Return n dominant hex colors using K-means clustering."""
        try:
            if path.suffix.lower() in (".mp4", ".mov", ".webm", ".avi"):
                cap = cv2.VideoCapture(str(path))
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    return ["#000000"] * n
            else:
                frame = cv2.imread(str(path))
                if frame is None:
                    # fallback to Pillow
                    with Image.open(path) as img:
                        img = img.convert("RGB").resize((150, 150))
                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Resize for speed
            small = cv2.resize(frame, (150, 150))
            data = small.reshape(-1, 3).astype(np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, _, centers = cv2.kmeans(data, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            colors = []
            for c in centers:
                b, g, r = int(c[0]), int(c[1]), int(c[2])
                colors.append(f"#{r:02x}{g:02x}{b:02x}")
            return colors

        except Exception as e:
            logger.warning("Color extraction failed for %s: %s", path, e)
            return ["#808080"] * n

    # ------------------------------------------------------------------
    # Story 2.4 — Visual DNA
    # ------------------------------------------------------------------

    def compute_visual_dna(self, asset: Asset) -> VisualDNA:
        colors = asset.color_palette or []
        mood = _infer_mood_from_colors(colors)
        return VisualDNA(
            dominant_colors=colors[:3],
            color_palette=colors,
            dominant_mood=mood,
            visual_tags=_infer_tags(asset),
        )

    # ------------------------------------------------------------------
    # Story 2.5 — Quality score
    # ------------------------------------------------------------------

    def compute_quality_score(self, asset: Asset) -> float:
        """
        Scoring (max 10):
          Resolution match  0-4 pts  (1080p=4, 720p=2, <720p=1)
          Aspect ratio      0-2 pts  (16:9=2, other=1, portrait=0)
          Duration          0-2 pts  (>5s video=2, >2s=1; any image=2)
          Codec/format      0-2 pts  (h264/hevc/jpg=2, other=1)
        """
        score = 0.0

        # Resolution
        if asset.resolution:
            try:
                w, h = map(int, asset.resolution.split("x"))
                long_side = max(w, h)
                if long_side >= 1920:
                    score += 4
                elif long_side >= 1280:
                    score += 2
                elif long_side >= 640:
                    score += 1
            except ValueError:
                pass

        # Aspect ratio
        ar = asset.aspect_ratio or ""
        if ar in ("16:9", "1920:1080"):
            score += 2
        elif ar and ":" in ar:
            score += 1

        # Duration / type
        if asset.type == AssetType.IMAGE:
            score += 2
        elif asset.duration_sec >= 5:
            score += 2
        elif asset.duration_sec >= 2:
            score += 1

        # Codec / format
        vm = asset.video_metadata
        if vm and vm.codec in ("h264", "hevc", "vp9", "av1"):
            score += 2
        elif vm:
            score += 1
        elif asset.type == AssetType.IMAGE:
            score += 2
        elif asset.type in (AssetType.AUDIO, AssetType.SFX):
            score += 2

        return round(min(score, 10.0), 2)

    # ------------------------------------------------------------------
    # Persistence & helpers
    # ------------------------------------------------------------------

    def save_assets_json(self, assets: list[Asset]) -> None:
        out = self.project_dir / "assets.json"
        out.write_text(json.dumps([a.model_dump() for a in assets], indent=2, default=str))
        logger.info("Saved assets.json with %d assets", len(assets))

    def _compute_aspect_ratio(self, w: int, h: int) -> Optional[str]:
        if not w or not h:
            return None
        g = gcd(w, h)
        return f"{w // g}:{h // g}"

    def _update_status(self, status: ModuleStatus) -> None:
        project_json = self.project_dir / "project.json"
        if not project_json.exists():
            return
        meta = json.loads(project_json.read_text())
        project_id = meta.get("project_id")
        if not project_id:
            return
        try:
            update_pipeline_status(
                project_id, "module_2_metadata", status,
                projects_root=self.project_dir.parent,
            )
        except Exception as e:
            logger.warning("Could not update pipeline status: %s", e)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_fps(rate_str: str) -> float:
    try:
        if "/" in rate_str:
            n, d = rate_str.split("/")
            return float(n) / float(d) if float(d) else 0.0
        return float(rate_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _infer_mood_from_colors(hex_colors: list[str]) -> Mood:
    """Heuristic mood from average color brightness and saturation."""
    if not hex_colors:
        return Mood.NEUTRAL
    try:
        brightnesses = []
        for h in hex_colors[:5]:
            h = h.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            brightnesses.append(0.299 * r + 0.587 * g + 0.114 * b)
        avg = sum(brightnesses) / len(brightnesses)
        if avg < 60:
            return Mood.DARK
        elif avg < 120:
            return Mood.MYSTERIOUS
        elif avg > 180:
            return Mood.UPLIFTING
        else:
            return Mood.NEUTRAL
    except Exception:
        return Mood.NEUTRAL


def _infer_tags(asset: Asset) -> list[str]:
    tags = list(asset.visual_tags or [])
    if asset.search_query:
        tags += [w.lower() for w in asset.search_query.split() if len(w) > 2]
    return list(dict.fromkeys(tags))[:10]  # deduplicate, cap at 10
