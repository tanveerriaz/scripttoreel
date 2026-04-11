"""
Module 4 — Scene Planning & Orchestration.

Maps script segments → best-matching assets → Scene objects with transitions,
color grades, text overlays, and audio mix levels.
Writes orchestration.json — the full edit plan for Module 5 to render.
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
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
_MUSIC_VOLUME = 0.25
_SFX_VOLUME = 0.4

# Visual cut cadence: each scene is at most this many seconds before switching asset.
# Alternates video → image → video → image for visual variety.
_MAX_CLIP_SEC = 5.0

# Genre-aware clip duration ranges (min_sec, max_sec).
# Detected from script.topic + plan.tone at timeline build time.
_GENRE_CLIP_RANGES: dict[str, tuple[float, float]] = {
    "thriller":    (2.5, 4.0),   # fast cuts — high tension
    "horror":      (3.0, 5.0),   # slightly longer for dread buildup
    "action":      (2.0, 3.5),   # very fast — kinetic energy
    "celebration": (3.5, 5.5),   # moderate, energetic
    "documentary": (4.5, 7.0),   # slower, contemplative
    "default":     (3.0, 6.0),   # varied — avoids mechanical uniform pacing
}


def _detect_genre(topic: str, plan=None) -> str:
    """Detect genre from topic text and production plan tone.

    Returns one of: 'thriller', 'horror', 'action', 'celebration', 'documentary', 'default'.
    """
    combined = topic.lower()
    if plan:
        tone = getattr(plan, "tone", "") or ""
        combined += " " + (tone.value if hasattr(tone, "value") else str(tone)).lower()

    if any(k in combined for k in ("thriller", "suspense", "spy", "heist", "chase")):
        return "thriller"
    if any(k in combined for k in ("horror", "haunted", "ghost", "terror", "scary")):
        return "horror"
    if any(k in combined for k in ("action", "fight", "combat", "battle", "war")):
        return "action"
    if any(k in combined for k in ("birthday", "celebration", "wedding", "party", "anniversary")):
        return "celebration"
    if any(k in combined for k in ("documentary", "history", "science", "education", "nature")):
        return "documentary"
    return "default"


class OrchestrationModule:
    def __init__(self, project_dir: Path, skip_director: bool = False):
        self.project_dir = Path(project_dir)
        self.skip_director = skip_director

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Orchestration:
        script = self._load_script()
        assets = self._load_assets()  # uses default min_quality=4.0

        logger.info("Module 4: mapping %d segments to %d assets", len(script.segments), len(assets))

        # Load production plan for visual_style / tone overrides
        plan = self._load_production_plan()

        voiceover_tracks = self._build_voiceover_tracks(script)
        # Use actual voiceover duration so video length matches the audio
        # Add 4s title card + 3s outro card
        vo_duration = self._voiceover_duration(script) or script.duration_sec
        total_duration = vo_duration + 4.0 + 3.0
        background_music = self._find_background_music(assets)
        beat_grid = self._extract_beat_grid(background_music)
        scenes = self.build_timeline(script, assets, total_duration=vo_duration, plan=plan, beat_grid=beat_grid)
        scenes = self._assign_sfx_to_scenes(scenes, script, assets)

        meta = self._load_project_meta()
        global_grade = self._plan_color_grade(plan) or self.assign_color_grade(script.mood)
        orch = Orchestration(
            project_id=meta.get("project_id", "unknown"),
            title=script.title,
            topic=script.topic,
            total_duration_sec=total_duration,
            color_grade=global_grade,
            scenes=scenes,
            background_music=background_music,
            voiceover_tracks=voiceover_tracks,
        )

        if not self.skip_director and total_duration > 60:
            orch = self._run_visual_director(orch)

        self.save_orchestration(orch)
        self._update_status(ModuleStatus.COMPLETE)
        return orch

    def _run_visual_director(self, orch: Orchestration) -> Orchestration:
        """Run VisualDirector coherence review on the orchestration plan."""
        try:
            from src.ai_director import VisualDirector
            return VisualDirector().review(orch)
        except Exception as e:
            logger.warning("VisualDirector review failed: %s — using original plan", e)
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
        self,
        script: Script,
        assets: list[Asset],
        total_duration: float = 0.0,
        plan=None,  # Optional[ProductionPlan]
        beat_grid: list[float] | None = None,
    ) -> list[Scene]:
        visual_assets = [
            a for a in assets
            if a.type in (AssetType.VIDEO, AssetType.IMAGE) and a.local_path
        ]

        # Separate pools for alternation — video on even clips, image on odd.
        # Sort by quality descending so the first rotation cycle shows best assets.
        # Use deque for O(1) round-robin: rotate(-1) after each use so every asset
        # is shown once before any asset repeats.
        video_pool = sorted(
            [a for a in visual_assets if a.type == AssetType.VIDEO],
            key=lambda a: -a.quality_score,
        )
        image_pool = sorted(
            [a for a in visual_assets if a.type == AssetType.IMAGE],
            key=lambda a: -a.quality_score,
        )
        video_deque: deque = deque(video_pool)
        image_deque: deque = deque(image_pool)

        # Distribute total_duration proportionally across segments if provided
        script_total = sum(s.duration_sec for s in script.segments) or 1.0
        scale = (total_duration / script_total) if total_duration > 0 else 1.0

        # Genre-aware clip pacing — detect once, use throughout
        genre = _detect_genre(script.topic, plan)
        min_clip, max_clip = _GENRE_CLIP_RANGES.get(genre, _GENRE_CLIP_RANGES["default"])
        logger.info("build_timeline: genre=%s  clip_range=%.1f–%.1fs", genre, min_clip, max_clip)

        def _clip_target(seg_idx: int) -> float:
            """Target clip duration for a segment — oscillates within genre range
            to avoid the mechanical uniform-cut feel."""
            return min_clip + (max_clip - min_clip) * (0.5 + 0.5 * math.sin(seg_idx))

        # Pre-compute total clip count for first/last detection
        total_clips = sum(
            max(1, round(round(s.duration_sec * scale, 3) / _clip_target(i)))
            for i, s in enumerate(script.segments)
        )

        # Mood → color grade mapping for per-scene grading
        _MOOD_GRADE: dict[str, ColorGrade] = {
            "dark": ColorGrade.DARK_MYSTERIOUS,
            "mysterious": ColorGrade.DARK_MYSTERIOUS,
            "horror": ColorGrade.DARK_MYSTERIOUS,
            "dramatic": ColorGrade.DRAMATIC,
            "suspenseful": ColorGrade.DRAMATIC,
            "educational": ColorGrade.DOCUMENTARY,
            "neutral": ColorGrade.DOCUMENTARY,
            "uplifting": ColorGrade.CINEMATIC_WARM,
            "melancholic": ColorGrade.DRAMATIC,
        }

        # Genre → prefer cuts vs dissolves
        fast_genres = {"thriller", "action", "horror"}
        use_fast_cuts = genre in fast_genres

        scenes: list[Scene] = []
        cursor = 0.0
        scene_id = 0
        clip_counter = 0  # increments per clip; even → video, odd → image

        # Track cumulative VO timing for caption sync
        # VO starts after title card; each segment plays back-to-back
        _TITLE_DUR = 4.0
        vo_cursor = _TITLE_DUR  # where the next segment's VO begins in the timeline

        # Prepend title card scene (4s)
        title_card = Scene(
            id=0,
            segment_id=0,
            asset_id="title_card",
            asset_path="",
            start_time=0.0,
            end_time=_TITLE_DUR,
            duration_sec=_TITLE_DUR,
            transition_in=TransitionType.FADE_IN,
            transition_out=TransitionType.DISSOLVE,
            color_grade=self._plan_color_grade(plan) or ColorGrade.DARK_MYSTERIOUS,
            text_overlays=[],  # title card renders its own styled text
            is_title_card=True,
            caption_text=script.topic,
        )
        scenes.append(title_card)
        cursor = _TITLE_DUR

        prev_segment_id = None

        for seg_idx, segment in enumerate(script.segments):
            seg_dur = round(segment.duration_sec * scale, 3)

            # Subdivide segment into clips, targeting genre-appropriate duration
            n_clips = max(1, round(seg_dur / _clip_target(seg_idx)))
            base_clip_dur = round(seg_dur / n_clips, 3)

            # Per-scene color grade: prefer mood-based grade, fall back to plan grade
            plan_grade = self._plan_color_grade(plan)
            mood_key = segment.mood_tags[0].lower() if segment.mood_tags and _is_valid_mood(segment.mood_tags[0]) else ""
            grade = _MOOD_GRADE.get(mood_key) or plan_grade or self.assign_color_grade(script.mood)

            overlays = self._build_text_overlays(segment)

            # Caption timing: sync to actual VO timing, not scene visual timing
            seg_vo_dur = segment.voiceover_duration_sec or seg_dur
            seg_text = (segment.text or "").strip()
            seg_words = seg_text.split()
            words_per_clip = max(1, len(seg_words) // n_clips) if seg_words else 0

            for clip_idx in range(n_clips):
                scene_id += 1

                # Beat-snap: nudge clip end to nearest beat (within ±0.4s)
                clip_dur = base_clip_dur
                if beat_grid:
                    proposed_end = cursor + clip_dur
                    nearest = min(beat_grid, key=lambda b: abs(b - proposed_end), default=proposed_end)
                    if abs(nearest - proposed_end) < 0.4 and nearest > cursor + 1.0:
                        clip_dur = round(nearest - cursor, 3)

                # Alternate: even clip_counter → video, odd → image.
                prefer_video = (clip_counter % 2 == 0)
                if prefer_video and video_deque:
                    asset = video_deque[0]
                    video_deque.rotate(-1)
                elif image_deque:
                    asset = image_deque[0]
                    image_deque.rotate(-1)
                elif video_deque:
                    asset = video_deque[0]
                    video_deque.rotate(-1)
                elif visual_assets:
                    asset = visual_assets[clip_counter % len(visual_assets)]
                else:
                    logger.error("No visual assets available — skipping clip %d", scene_id)
                    clip_counter += 1
                    cursor += clip_dur
                    continue

                is_last = (seg_idx == len(script.segments) - 1 and clip_idx == n_clips - 1)

                # Contextual transition: CUT within segment, DISSOLVE across segments
                if scene_id == 1:
                    t_in = TransitionType.DISSOLVE  # after title card
                elif segment.id == prev_segment_id and use_fast_cuts:
                    t_in = TransitionType.CUT
                elif segment.id != prev_segment_id:
                    t_in = TransitionType.DISSOLVE
                else:
                    t_in = TransitionType.CROSSFADE if not use_fast_cuts else TransitionType.DISSOLVE
                t_out = TransitionType.FADE_OUT if is_last else TransitionType.DISSOLVE

                # Only show text overlay on the first clip of each segment
                clip_overlays = overlays if clip_idx == 0 else []

                # Caption text + VO-synced timing for this clip
                w_start = clip_idx * words_per_clip
                w_end = len(seg_words) if clip_idx == n_clips - 1 else (clip_idx + 1) * words_per_clip
                clip_caption = " ".join(seg_words[w_start:w_end]) if words_per_clip > 0 else ""

                # VO-synced timing: distribute VO duration proportionally across clips
                vo_clip_dur = seg_vo_dur / n_clips
                cap_start = vo_cursor + clip_idx * vo_clip_dur
                cap_end = vo_cursor + (clip_idx + 1) * vo_clip_dur

                scene = Scene(
                    id=scene_id,
                    segment_id=segment.id,
                    asset_id=asset.id,
                    asset_path=asset.local_path or "",
                    start_time=cursor,
                    end_time=cursor + clip_dur,
                    duration_sec=clip_dur,
                    transition_in=t_in,
                    transition_out=t_out,
                    color_grade=grade,
                    text_overlays=clip_overlays,
                    voiceover=None,
                    caption_text=clip_caption,
                    caption_start_sec=cap_start,
                    caption_end_sec=cap_end,
                )
                scenes.append(scene)
                cursor += clip_dur
                clip_counter += 1
                prev_segment_id = segment.id

            # Advance VO cursor by the actual voiceover duration for this segment
            vo_cursor += seg_vo_dur

        # Post-processing: dedup + temperature smoothing + coherence scoring
        content_scenes = [s for s in scenes if not getattr(s, "is_title_card", False)]
        content_scenes, asset_map = self._post_process_timeline(content_scenes, script.segments, visual_assets)

        # Append outro card (3s)
        _OUTRO_DUR = 3.0
        # Use the last segment's outro text or a default
        outro_text = script.topic
        for seg in reversed(script.segments):
            if seg.type and seg.type.value == "outro" and seg.text:
                outro_text = seg.text.split(".")[0].strip()  # first sentence
                break
        outro_card = Scene(
            id=scene_id + 1,
            segment_id=0,
            asset_id="outro_card",
            asset_path="",
            start_time=cursor,
            end_time=cursor + _OUTRO_DUR,
            duration_sec=_OUTRO_DUR,
            transition_in=TransitionType.DISSOLVE,
            transition_out=TransitionType.FADE_OUT,
            color_grade=self._plan_color_grade(plan) or ColorGrade.DARK_MYSTERIOUS,
            is_outro_card=True,
            caption_text=outro_text,
        )

        title_scenes = [s for s in scenes if getattr(s, "is_title_card", False)]
        scenes = title_scenes + content_scenes + [outro_card]

        return scenes

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

    _TITLE_CARD_DUR = 4.0  # must match build_timeline title card duration

    def _build_voiceover_tracks(self, script: Script) -> list[AudioTrack]:
        # Use the combined voiceover.wav produced by module 3 as a single track.
        # VO starts after the title card so narration aligns with content scenes.
        combined = script.total_voiceover_path
        if combined and Path(combined).exists():
            return [AudioTrack(
                asset_id="vo_combined",
                local_path=combined,
                start_time=self._TITLE_CARD_DUR,
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

    def _extract_beat_grid(self, music_track: Optional[AudioTrack]) -> list[float]:
        """Extract beat times from background music for beat-synced cuts."""
        if not music_track or not music_track.local_path:
            return []
        try:
            import librosa  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            y, sr = librosa.load(music_track.local_path, sr=22050, mono=True, duration=120)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
            logger.info("Beat grid: %.0f BPM, %d beats extracted", float(tempo) if np.ndim(tempo) == 0 else float(tempo[0]), len(beat_times))
            return beat_times
        except Exception as e:
            logger.warning("Beat extraction failed: %s — using time-based cuts", e)
            return []

    def _assign_sfx_to_scenes(
        self, scenes: list[Scene], script: Script, assets: list[Asset]
    ) -> list[Scene]:
        """Assign downloaded SFX assets to the first scene of each segment."""
        sfx_assets = [
            a for a in assets
            if a.type in (AssetType.AUDIO, AssetType.SFX)
            and a.local_path and Path(a.local_path).exists()
            and 0.5 < a.duration_sec < 30  # short sounds, not music loops
            and a.role in (AssetRole.SFX, AssetRole.MUSIC)  # accept short music clips as SFX
        ]
        if not sfx_assets:
            return scenes

        # Build mapping: segment_id → sfx_cues keywords
        seg_cues: dict[int, list[str]] = {}
        for seg in script.segments:
            seg_cues[seg.id] = [c.lower() for c in (seg.sfx_cues or [])]

        assigned_ids: set[str] = set()
        updated = []
        for scene in scenes:
            if scene.is_title_card or getattr(scene, "is_outro_card", False):
                updated.append(scene)
                continue

            cues = seg_cues.get(scene.segment_id, [])
            if not cues:
                updated.append(scene)
                continue

            # Find best matching SFX (not already used) based on keyword overlap
            best, best_score = None, 0
            for sfx in sfx_assets:
                if sfx.id in assigned_ids:
                    continue
                tags = {t.lower() for t in (sfx.visual_tags or [])}
                score = sum(1 for c in cues if any(c in t for t in tags))
                if score > best_score:
                    best, best_score = sfx, score

            if best and best_score > 0:
                assigned_ids.add(best.id)
                sfx_track = AudioTrack(
                    asset_id=best.id,
                    local_path=best.local_path or "",
                    start_time=scene.start_time,
                    volume=_SFX_VOLUME,
                )
                scene = scene.model_copy(update={"sfx_tracks": [sfx_track]})
                logger.info("SFX %s assigned to scene %d (seg %d)", best.id, scene.id, scene.segment_id)

            updated.append(scene)
        return updated

    def _find_background_music(self, assets: list[Asset]) -> Optional[AudioTrack]:
        # Prefer dedicated music assets
        music_assets = [
            a for a in assets
            if a.type == AssetType.AUDIO and a.role == AssetRole.MUSIC and a.local_path
        ]
        # Fallback: long SFX/audio tracks (≥30s) can serve as ambient background
        if not music_assets:
            music_assets = [
                a for a in assets
                if a.type in (AssetType.AUDIO, AssetType.SFX)
                and a.local_path
                and a.duration_sec >= 30
            ]
            if music_assets:
                logger.info("No dedicated music assets — using long audio asset as background")
        if not music_assets:
            logger.info("No background music found — video will be voiceover only")
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

    # ------------------------------------------------------------------
    # Story 4.5 — Visual coherence scoring
    # ------------------------------------------------------------------

    def _post_process_timeline(
        self,
        scenes: list[Scene],
        segments,
        visual_assets: list,
    ) -> tuple[list[Scene], dict[int, object]]:
        """Dedup consecutive same-asset scenes and penalise temperature jumps.

        Returns (updated_scenes, asset_by_scene_id) where asset_by_scene_id maps
        scene.id → Asset for downstream logging.
        """
        # Build scene_id → Asset lookup
        asset_by_id: dict[str, object] = {a.id: a for a in visual_assets}
        asset_by_scene: dict[int, object] = {}
        for scene in scenes:
            asset_by_scene[scene.id] = asset_by_id.get(scene.asset_id)

        # --- Dedup: same asset in consecutive scenes → swap to next-best ---
        for i in range(1, len(scenes)):
            if scenes[i].asset_id == scenes[i - 1].asset_id and len(scenes) > i:
                seg = segments[i] if i < len(segments) else segments[-1]
                # Exclude already-used asset
                candidates = [a for a in visual_assets if a.id != scenes[i].asset_id]
                if candidates:
                    best = max(candidates, key=lambda a: self._score_asset(a, seg))
                    scenes[i] = scenes[i].model_copy(update={
                        "asset_id": best.id,
                        "asset_path": best.local_path or "",
                    })
                    asset_by_scene[scenes[i].id] = best
                    logger.info(
                        "Dedup: scene %d swapped from %s → %s",
                        scenes[i].id, scenes[i - 1].asset_id, best.id,
                    )

        # --- Temperature smoothing: penalise delta > 2.0 ---
        for i in range(1, len(scenes)):
            asset_a = asset_by_scene.get(scenes[i - 1].id)
            asset_b = asset_by_scene.get(scenes[i].id)
            if asset_a is None or asset_b is None:
                continue
            temp_a = _compute_color_temperature(getattr(asset_a, "color_palette", []))
            temp_b = _compute_color_temperature(getattr(asset_b, "color_palette", []))
            if abs(temp_a - temp_b) > 2.0:
                seg = segments[i] if i < len(segments) else segments[-1]
                # Try to find an asset whose temperature is closer to temp_a
                candidates = [
                    a for a in visual_assets if a.id != scenes[i - 1].asset_id
                ]
                if candidates:
                    best = min(
                        candidates,
                        key=lambda a: abs(
                            _compute_color_temperature(getattr(a, "color_palette", [])) - temp_a
                        ),
                    )
                    best_temp = _compute_color_temperature(getattr(best, "color_palette", []))
                    if abs(best_temp - temp_a) < abs(temp_b - temp_a):
                        scenes[i] = scenes[i].model_copy(update={
                            "asset_id": best.id,
                            "asset_path": best.local_path or "",
                        })
                        asset_by_scene[scenes[i].id] = best
                        logger.info(
                            "Temperature smooth: scene %d temp %.1f→%.1f (delta was %.1f)",
                            scenes[i].id, temp_b, best_temp, abs(temp_a - temp_b),
                        )

        # --- Coherence scoring ---
        score = 10.0
        for i in range(1, len(scenes)):
            if scenes[i].asset_id == scenes[i - 1].asset_id:
                score -= 2.0
            asset_a = asset_by_scene.get(scenes[i - 1].id)
            asset_b = asset_by_scene.get(scenes[i].id)
            if asset_a and asset_b:
                temp_a = _compute_color_temperature(getattr(asset_a, "color_palette", []))
                temp_b = _compute_color_temperature(getattr(asset_b, "color_palette", []))
                if abs(temp_a - temp_b) > 2.0:
                    score -= 1.0
        score = max(0.0, min(10.0, score))
        logger.info("Visual coherence score: %.1f/10", score)

        return scenes, asset_by_scene

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

    def _load_production_plan(self):
        """Load production_plan.json if present. Returns None otherwise."""
        plan_path = self.project_dir / "production_plan.json"
        if not plan_path.exists():
            return None
        try:
            from src.utils.json_schemas import ProductionPlan
            return ProductionPlan(**json.loads(plan_path.read_text()))
        except Exception as e:
            logger.warning("Could not load production_plan.json: %s", e)
            return None

    def _plan_color_grade(self, plan) -> Optional[ColorGrade]:
        """Map production plan visual_style to a ColorGrade, or None if no plan."""
        if plan is None:
            return None
        mapping = {
            "dark_mysterious": ColorGrade.DARK_MYSTERIOUS,
            "cinematic_warm": ColorGrade.CINEMATIC_WARM,
            "documentary": ColorGrade.DOCUMENTARY,
            "dramatic": ColorGrade.DRAMATIC,
            "bright_modern": ColorGrade.UPLIFTING,
        }
        return mapping.get(plan.visual_style.value if hasattr(plan.visual_style, "value") else str(plan.visual_style))

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


def _compute_color_temperature(hex_colors: list[str]) -> float:
    """Compute warmth score (0=cool, 10=warm) from a list of hex color strings.

    Warmth = average of per-color (R - B + 255) / 510 * 10.
    Returns 5.0 (neutral) for empty input.
    """
    if not hex_colors:
        return 5.0

    scores = []
    for hex_color in hex_colors:
        try:
            h = hex_color.lstrip("#")
            if len(h) < 6:
                continue
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            warmth = (r - b + 255) / 510 * 10
            scores.append(warmth)
        except (ValueError, IndexError):
            continue

    return round(sum(scores) / len(scores), 2) if scores else 5.0
