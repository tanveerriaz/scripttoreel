"""
Module 4 — Scene Planning & Orchestration.

Maps script segments → best-matching assets → Scene objects with transitions,
color grades, text overlays, and audio mix levels.
Writes orchestration.json — the full edit plan for Module 5 to render.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.project_manager import update_pipeline_status
from src.utils.json_schemas import (
    Asset,
    AssetRole,
    AssetType,
    AudioTrack,
    ColorGrade,
    Mood,
    ModuleStatus,
    Orchestration,
    Scene,
    Script,
    ScriptSegment,
    TextOverlay,
    TransitionType,
)

logger = logging.getLogger(__name__)

# Audio levels
_VOICEOVER_VOLUME = 1.0
_MUSIC_VOLUME = 0.12
_SFX_VOLUME = 0.4


class OrchestrationModule:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Orchestration:
        script = self._load_script()
        assets = self._load_assets(min_quality=0.0)  # accept all ready assets

        logger.info("Module 4: mapping %d segments to %d assets", len(script.segments), len(assets))

        voiceover_tracks = self._build_voiceover_tracks(script)
        # Use actual voiceover duration so video length matches the audio
        total_duration = self._voiceover_duration(script) or script.duration_sec
        scenes = self.build_timeline(script, assets, total_duration=total_duration)
        background_music = self._find_background_music(assets)

        meta = self._load_project_meta()
        orch = Orchestration(
            project_id=meta.get("project_id", "unknown"),
            title=script.title,
            topic=script.topic,
            total_duration_sec=total_duration,
            color_grade=self.assign_color_grade(script.mood),
            scenes=scenes,
            background_music=background_music,
            voiceover_tracks=voiceover_tracks,
        )

        self.save_orchestration(orch)
        self._update_status(ModuleStatus.COMPLETE)
        return orch

    # ------------------------------------------------------------------
    # Story 4.1 — Asset matching
    # ------------------------------------------------------------------

    def _score_asset(self, asset: Asset, segment: ScriptSegment) -> float:
        """Score an asset's suitability for a script segment."""
        score = 0.0

        # Mood match
        seg_moods = [m.lower() for m in (segment.mood_tags or [])]
        if asset.dominant_mood and asset.dominant_mood.value in seg_moods:
            score += 3.0

        # Visual tag / keyword overlap
        asset_tags = {t.lower() for t in (asset.visual_tags or [])}
        for kw in segment.b_roll_keywords:
            if kw.lower() in asset_tags:
                score += 1.0
            # partial match
            elif any(kw.lower() in t for t in asset_tags):
                score += 0.5

        # Duration fit (for video assets)
        if asset.type == AssetType.VIDEO and asset.duration_sec > 0:
            diff = abs(asset.duration_sec - segment.duration_sec)
            if diff < 2:
                score += 2.0
            elif diff < 5:
                score += 1.0

        # Quality bonus
        score += asset.quality_score / 2.0

        return score

    def match_asset_to_segment(
        self, segment: ScriptSegment, assets: list[Asset]
    ) -> Optional[Asset]:
        if not assets:
            return None
        return max(assets, key=lambda a: self._score_asset(a, segment))

    # ------------------------------------------------------------------
    # Story 4.2 — Timeline construction
    # ------------------------------------------------------------------

    def build_timeline(
        self, script: Script, assets: list[Asset], total_duration: float = 0.0
    ) -> list[Scene]:
        visual_assets = [
            a for a in assets
            if a.type in (AssetType.VIDEO, AssetType.IMAGE) and a.local_path
        ]

        # Distribute total_duration proportionally across segments if provided
        script_total = sum(s.duration_sec for s in script.segments) or 1.0
        scale = (total_duration / script_total) if total_duration > 0 else 1.0

        scenes: list[Scene] = []
        cursor = 0.0

        for idx, segment in enumerate(script.segments):
            asset = self.match_asset_to_segment(segment, visual_assets)
            if asset is None:
                logger.warning("No asset matched for segment %d — using placeholder", segment.id)
                asset = visual_assets[0] if visual_assets else None
                if asset is None:
                    continue

            grade = self.assign_color_grade(
                Mood(segment.mood_tags[0]) if segment.mood_tags and _is_valid_mood(segment.mood_tags[0]) else script.mood
            )

            t_in = TransitionType.FADE_IN if idx == 0 else TransitionType.DISSOLVE
            t_out = TransitionType.DISSOLVE if idx < len(script.segments) - 1 else TransitionType.FADE_OUT

            overlays = self._build_text_overlays(segment)
            scene_dur = round(segment.duration_sec * scale, 3)

            scene = Scene(
                id=idx + 1,
                segment_id=segment.id,
                asset_id=asset.id,
                asset_path=asset.local_path or "",
                start_time=cursor,
                end_time=cursor + scene_dur,
                duration_sec=scene_dur,
                transition_in=t_in,
                transition_out=t_out,
                color_grade=grade,
                text_overlays=overlays,
                voiceover=None,
            )
            scenes.append(scene)
            cursor += scene_dur

        return scenes

    # ------------------------------------------------------------------
    # Story 4.3 — Color grade assignment
    # ------------------------------------------------------------------

    def assign_color_grade(self, mood: Mood) -> ColorGrade:
        mapping = {
            Mood.DARK: ColorGrade.DARK_MYSTERIOUS,
            Mood.MYSTERIOUS: ColorGrade.DARK_MYSTERIOUS,
            Mood.HORROR: ColorGrade.DARK_MYSTERIOUS,
            Mood.MELANCHOLIC: ColorGrade.DARK_MYSTERIOUS,
            Mood.DRAMATIC: ColorGrade.DRAMATIC,
            Mood.SUSPENSEFUL: ColorGrade.DRAMATIC,
            Mood.UPLIFTING: ColorGrade.UPLIFTING,
            Mood.EDUCATIONAL: ColorGrade.DOCUMENTARY,
            Mood.NEUTRAL: ColorGrade.DOCUMENTARY,
        }
        return mapping.get(mood, ColorGrade.DOCUMENTARY)

    def _build_text_overlays(self, segment: ScriptSegment) -> list[TextOverlay]:
        overlay = segment.text_overlay
        if overlay and overlay.enabled and overlay.text:
            return [overlay]
        return []

    # ------------------------------------------------------------------
    # Story 4.4 — Audio plan
    # ------------------------------------------------------------------

    def _build_voiceover_tracks(self, script: Script) -> list[AudioTrack]:
        # Use the combined voiceover.wav produced by module 3 as a single track.
        # Using individual segment files caused all segments to overlap at t=0
        # because _mix_audio does not honour start_time offsets.
        combined = script.total_voiceover_path
        if combined and Path(combined).exists():
            return [AudioTrack(
                asset_id="vo_combined",
                local_path=combined,
                start_time=0.0,
                volume=_VOICEOVER_VOLUME,
            )]
        # Fallback: return first available segment path (rare edge-case).
        for seg in script.segments:
            if seg.voiceover_path and Path(seg.voiceover_path).exists():
                return [AudioTrack(
                    asset_id="vo_fallback",
                    local_path=seg.voiceover_path,
                    start_time=0.0,
                    volume=_VOICEOVER_VOLUME,
                )]
        return []

    def _find_background_music(self, assets: list[Asset]) -> Optional[AudioTrack]:
        music_assets = [
            a for a in assets
            if a.type == AssetType.AUDIO and a.role == AssetRole.MUSIC and a.local_path
        ]
        if not music_assets:
            return None
        best = max(music_assets, key=lambda a: a.quality_score)
        return AudioTrack(
            asset_id=best.id,
            local_path=best.local_path or "",
            volume=_MUSIC_VOLUME,
            loop=True,
            fade_in=1.0,
            fade_out=2.0,
        )

    def _voiceover_duration(self, script: Script) -> float:
        """Return duration of the combined voiceover WAV, or 0 if unavailable."""
        combined = script.total_voiceover_path
        if not combined:
            return 0.0
        p = Path(combined)
        if not p.exists():
            return 0.0
        try:
            from pydub import AudioSegment as _AS
            return _AS.from_file(str(p)).duration_seconds
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Persistence & helpers
    # ------------------------------------------------------------------

    def save_orchestration(self, orch: Orchestration) -> None:
        out = self.project_dir / "orchestration.json"
        out.write_text(json.dumps(orch.model_dump(), indent=2, default=str))
        logger.info("Saved orchestration.json (%d scenes)", len(orch.scenes))

    def _load_script(self) -> Script:
        p = self.project_dir / "script.json"
        if not p.exists():
            raise FileNotFoundError(f"script.json not found in {self.project_dir}")
        return Script(**json.loads(p.read_text()))

    def _load_assets(self, min_quality: float = 4.0) -> list[Asset]:
        p = self.project_dir / "assets.json"
        if not p.exists():
            p = self.project_dir / "assets_raw.json"
        if not p.exists():
            return []
        data = json.loads(p.read_text())
        return [Asset(**a) for a in data if a.get("quality_score", 0) >= min_quality]

    def _load_project_meta(self) -> dict:
        p = self.project_dir / "project.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _update_status(self, status: ModuleStatus) -> None:
        meta = self._load_project_meta()
        project_id = meta.get("project_id")
        if not project_id:
            return
        try:
            update_pipeline_status(
                project_id, "module_4_orchestration", status,
                projects_root=self.project_dir.parent,
            )
        except Exception as e:
            logger.warning("Could not update pipeline status: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_mood(value: str) -> bool:
    try:
        Mood(value.lower())
        return True
    except ValueError:
        return False
