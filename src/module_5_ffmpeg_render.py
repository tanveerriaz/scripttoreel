"""
Module 5 — Video Rendering (MoviePy v2.0).

Parses orchestration.json and produces a final 1080p H.264 MP4.

Pipeline:
  1. Per-scene: build MoviePy clip from image/video asset
       - Images  → Ken Burns motion (numpy crop on padded canvas)
       - Videos  → Scale/pad/loop/trim to target duration
       - Apply color grade, vignette, letterbox, fade in/out, text overlays
  2. Concatenate scene clips with dissolve transitions (crossfadein/out)
  3. Mix audio via ffmpeg subprocess (LUFS normalization, amix)
  4. Attach mixed audio → write final MP4 via write_videofile()
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np

from src.project_manager import update_pipeline_status
from src.utils.config_loader import load_ffmpeg_presets
from src.utils.ffmpeg_builder import (
    FFmpegCommand,
    build_audio_amix_filter,
    build_color_grade_filter,
    build_scale_pad_filter,
    build_xfade_filter,
)
from src.utils.json_schemas import (
    ColorGrade,
    ModuleStatus,
    Orchestration,
    Scene,
    TextOverlay,
    TransitionType,
)

logger = logging.getLogger(__name__)

_W, _H = 1920, 1080
_FPS = 30

# Grades that get vignette overlay (dark, cinematic)
_VIGNETTE_GRADES = {ColorGrade.DARK_MYSTERIOUS, ColorGrade.DRAMATIC}
# Grades that get 2.39:1 letterbox bars
_LETTERBOX_GRADES = {ColorGrade.DARK_MYSTERIOUS, ColorGrade.DRAMATIC}

_TRANSITION_DUR = 0.5  # seconds for crossfade between scenes


class RenderModule:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.output_dir = self.project_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._presets = load_ffmpeg_presets()
        self._hw_encoder = self._detect_encoder()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> str:
        orch = self._load_orchestration()
        logger.info(
            "Module 5: rendering %d scenes, total %.1fs",
            len(orch.scenes), orch.total_duration_sec,
        )

        with tempfile.TemporaryDirectory(prefix="vf_render_") as tmpdir:
            tmp = Path(tmpdir)
            final_out = self.output_dir / "final_video.mp4"

            # Step 1 + 2: build scene clips and concatenate in ONE encode pass
            video_only = tmp / "video_only.mp4"
            if orch.scenes:
                clips = self._build_all_scene_clips(orch, tmp)
                self._write_concat(clips, str(video_only), orch.total_duration_sec,
                                   orch.scenes)
            else:
                logger.warning(
                    "No scenes — rendering black placeholder for %.1fs",
                    orch.total_duration_sec,
                )
                self._render_placeholder(orch.total_duration_sec, video_only)

            # Step 3: mix audio via ffmpeg (LUFS, amix)
            audio_out = tmp / "mixed_audio.aac"
            self._mix_audio(orch, audio_out, total_duration=orch.total_duration_sec)

            # Step 4: final encode — mux video + audio
            self._final_encode(str(video_only), str(audio_out), str(final_out))

        self._update_status(ModuleStatus.COMPLETE, output_file=str(final_out))
        logger.info("Render complete: %s", final_out)
        return str(final_out)

    # ------------------------------------------------------------------
    # Scene clip building (MoviePy)
    # ------------------------------------------------------------------

    def _build_all_scene_clips(self, orch: Orchestration, tmp: Path) -> list:
        """Build a MoviePy clip for every scene and return the list."""
        from moviepy import VideoFileClip, ImageClip, ColorClip, CompositeVideoClip, vfx

        n = len(orch.scenes)
        clips = []
        for idx, scene in enumerate(orch.scenes):
            clip = self._build_scene_clip(
                scene, tmp,
                is_first=(idx == 0),
                is_last=(idx == n - 1),
            )
            clips.append(clip)
        return clips

    def _build_scene_clip(
        self,
        scene: Scene,
        tmp: Path,
        is_first: bool = False,
        is_last: bool = False,
    ):
        """Return a fully-processed MoviePy clip for one scene."""
        from moviepy import ColorClip

        # Title/outro card scenes get a special render path
        if getattr(scene, "is_title_card", False):
            return self._build_title_card(scene, tmp)
        if getattr(scene, "is_outro_card", False):
            return self._build_outro_card(scene, tmp)

        asset_path = Path(scene.asset_path)
        duration = scene.duration_sec

        if not asset_path.exists():
            logger.warning("Asset not found: %s — black placeholder", asset_path)
            return ColorClip(size=(_W, _H), color=(0, 0, 0)).with_duration(duration).with_fps(_FPS)

        suffix = asset_path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"):
            clip = self._image_to_clip(str(asset_path), duration, scene.id)
        else:
            clip = self._video_to_clip(str(asset_path), duration)

        # Apply color grade (brightness/contrast/saturation/gamma)
        clip = self._apply_color_grade(clip, scene.color_grade)

        # Vignette for dark/dramatic grades
        if scene.color_grade in _VIGNETTE_GRADES:
            clip = self._apply_vignette(clip)

        # Letterbox bars (2.39:1) for dark/dramatic grades
        if scene.color_grade in _LETTERBOX_GRADES:
            clip = self._apply_letterbox(clip)

        # Fade in on first scene, fade out on last
        clip = self._apply_fades(clip, duration, is_first, is_last)

        # Text overlays (Pillow PNG composited)
        clip = self._apply_text_overlays(clip, scene.text_overlays, duration, tmp, scene.id)

        return clip

    # ------------------------------------------------------------------
    # Image → clip with Ken Burns effect
    # ------------------------------------------------------------------

    # Padded canvas: 15% larger than output for smooth zoom/pan headroom
    _KB_W = 2208   # 1920 * 1.15 (even)
    _KB_H = 1242   # 1080 * 1.15 (even)

    def _image_to_clip(self, image_path: str, duration: float, scene_id: int = 1):
        """Convert a still image to a video clip with Ken Burns motion."""
        from moviepy import VideoClip
        from PIL import Image as PILImage

        style = (scene_id - 1) % 4
        kw, kh = self._KB_W, self._KB_H
        n_frames = max(2, int(duration * _FPS))
        denom = max(1, n_frames - 1)

        # Load and pad to canvas size once (outside make_frame closure)
        try:
            img = PILImage.open(image_path).convert("RGB")
            # Fit into KB canvas preserving aspect ratio
            img.thumbnail((kw, kh), PILImage.LANCZOS)
            canvas = PILImage.new("RGB", (kw, kh), (0, 0, 0))
            ox = (kw - img.width) // 2
            oy = (kh - img.height) // 2
            canvas.paste(img, (ox, oy))
            canvas_arr = np.array(canvas, dtype=np.uint8)
        except Exception as e:
            logger.warning("Image load failed (%s): %s — black frame", image_path, e)
            return self._black_clip(duration)

        pan_cy = (kh - _H) // 2

        def make_frame(t: float) -> np.ndarray:
            progress = min(t / duration, 1.0) if duration > 0 else 0.0

            if style == 0:  # zoom in
                cw = int(kw + (_W - kw) * progress)
                ch = int(kh + (_H - kh) * progress)
                cx = (kw - cw) // 2
                cy = (kh - ch) // 2
            elif style == 1:  # zoom out
                cw = int(_W + (kw - _W) * progress)
                ch = int(_H + (kh - _H) * progress)
                cx = (kw - cw) // 2
                cy = (kh - ch) // 2
            elif style == 2:  # pan right
                max_x = kw - _W
                cx = int(max_x * progress)
                cy = pan_cy
                cw, ch = _W, _H
            else:  # pan left
                max_x = kw - _W
                cx = int(max_x * (1.0 - progress))
                cy = pan_cy
                cw, ch = _W, _H

            # Clamp
            cx = max(0, min(cx, kw - max(cw, 1)))
            cy = max(0, min(cy, kh - max(ch, 1)))
            cw = max(1, min(cw, kw - cx))
            ch = max(1, min(ch, kh - cy))

            cropped = canvas_arr[cy:cy + ch, cx:cx + cw]
            # Resize to output dimensions
            pil_crop = PILImage.fromarray(cropped).resize((_W, _H), PILImage.BICUBIC)
            return np.array(pil_crop, dtype=np.uint8)

        from moviepy import VideoClip
        return VideoClip(make_frame, duration=duration).with_fps(_FPS)

    # ------------------------------------------------------------------
    # Video → clip (scale/pad/loop/trim)
    # ------------------------------------------------------------------

    def _video_to_clip(self, video_path: str, duration: float):
        """Load a video, loop if too short, trim to duration, resize to 1920x1080."""
        from moviepy import VideoFileClip, ColorClip, concatenate_videoclips

        try:
            raw = VideoFileClip(video_path)
        except Exception as e:
            logger.warning("VideoFileClip failed (%s): %s — black frame", video_path, e)
            return self._black_clip(duration)

        # Loop if shorter than needed
        if raw.duration < duration:
            reps = int(duration / raw.duration) + 1
            raw = concatenate_videoclips([raw] * reps)

        # Trim to target duration
        clip = raw.subclipped(0, duration)

        # Resize to fill 1920x1080 (letterbox/pillarbox with black bars)
        clip_w, clip_h = clip.size
        scale = min(_W / clip_w, _H / clip_h)
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)

        resized = clip.resized((new_w, new_h))

        # Composite onto black 1920x1080 background
        if new_w != _W or new_h != _H:
            from moviepy import CompositeVideoClip
            bg = ColorClip(size=(_W, _H), color=(0, 0, 0)).with_duration(duration)
            ox = (_W - new_w) // 2
            oy = (_H - new_h) // 2
            resized = resized.with_position((ox, oy))
            clip = CompositeVideoClip([bg, resized])
        else:
            clip = resized

        return clip.with_fps(_FPS)

    # ------------------------------------------------------------------
    # Color grade (numpy image_transform)
    # ------------------------------------------------------------------

    def _apply_color_grade(self, clip, color_grade: ColorGrade):
        """Apply eq-style color grading via numpy image_transform."""
        grade = self._grade_params(color_grade)
        brightness = float(grade.get("brightness", 0.0))
        contrast = float(grade.get("contrast", 1.0))
        saturation = float(grade.get("saturation", 1.0))
        gamma = float(grade.get("gamma", 1.0))

        # Skip if all defaults
        if brightness == 0.0 and contrast == 1.0 and saturation == 1.0 and gamma == 1.0:
            return clip

        def grade_frame(frame: np.ndarray) -> np.ndarray:
            img = frame.astype(np.float32) / 255.0
            # Gamma
            if gamma != 1.0:
                img = np.power(np.clip(img, 0, 1), 1.0 / gamma)
            # Contrast
            if contrast != 1.0:
                img = (img - 0.5) * contrast + 0.5
            # Brightness
            if brightness != 0.0:
                img = img + brightness
            # Saturation
            if saturation != 1.0:
                gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
                gray = gray[..., np.newaxis]
                img = gray + (img - gray) * saturation
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)

        return clip.image_transform(grade_frame)

    # ------------------------------------------------------------------
    # Vignette (radial darkening mask)
    # ------------------------------------------------------------------

    def _apply_vignette(self, clip):
        """Apply a radial darkening vignette via numpy image_transform."""
        # Pre-compute vignette mask at output size (applied to every frame)
        cy, cx = _H / 2, _W / 2
        y = np.linspace(0, _H - 1, _H)
        x = np.linspace(0, _W - 1, _W)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        # Falloff: 1.0 at center, ~0.5 at corners
        mask = np.clip(1.0 - 0.5 * dist ** 1.5, 0.0, 1.0).astype(np.float32)
        mask_rgb = mask[:, :, np.newaxis]  # (H, W, 1) — broadcasts over RGB channels

        def vignette_frame(frame: np.ndarray) -> np.ndarray:
            return (frame.astype(np.float32) * mask_rgb).clip(0, 255).astype(np.uint8)

        return clip.image_transform(vignette_frame)

    # ------------------------------------------------------------------
    # Letterbox (2.39:1 black bars)
    # ------------------------------------------------------------------

    def _apply_letterbox(self, clip):
        """Add 2.39:1 widescreen letterbox (140px black bars top/bottom)."""
        bar_h = 140

        def letterbox_frame(frame: np.ndarray) -> np.ndarray:
            out = frame.copy()
            out[:bar_h, :] = 0
            out[-bar_h:, :] = 0
            return out

        return clip.image_transform(letterbox_frame)

    # ------------------------------------------------------------------
    # Fades
    # ------------------------------------------------------------------

    def _apply_fades(self, clip, duration: float, is_first: bool, is_last: bool):
        from moviepy import vfx
        effects = []
        # No fade-in on scene 0 — prevents a black opening frame
        # (thumbnails and autoplay need a real image at frame 0).
        # Text overlays get their own fade-in separately.
        if is_last:
            effects.append(vfx.FadeOut(0.5))
        if effects:
            clip = clip.with_effects(effects)
        return clip

    # ------------------------------------------------------------------
    # Text overlays (Pillow PNG + composite)
    # ------------------------------------------------------------------

    def _apply_text_overlays(
        self,
        clip,
        overlays: list[TextOverlay] | None,
        duration: float,
        tmp: Path,
        scene_id: int,
    ):
        """Composite Pillow-rendered text PNGs onto the clip."""
        enabled = [o for o in (overlays or []) if o.enabled and o.text.strip()]
        if not enabled:
            return clip

        from moviepy import ImageClip, CompositeVideoClip

        layers = [clip]
        for i, overlay in enumerate(enabled):
            png_path = str(tmp / f"overlay_{scene_id}_{i}.png")
            try:
                self._render_overlay_png(overlay, png_path)
            except Exception as e:
                logger.warning("Overlay PNG render failed: %s", e)
                continue

            start = getattr(overlay, "start_time", 0.0) or 0.0
            end = getattr(overlay, "end_time", None) or duration

            ov_clip = (
                ImageClip(png_path)
                .with_duration(end - start)
                .with_start(start)
            )
            layers.append(ov_clip)

        if len(layers) == 1:
            return clip

        return CompositeVideoClip(layers, size=(_W, _H))

    # ------------------------------------------------------------------
    # Concatenation (MoviePy)
    # ------------------------------------------------------------------

    def _write_concat(self, clips: list, out_path: str, total_duration: float,
                       scenes: list[Scene] | None = None,
                       caption_clips: list | None = None) -> None:
        """Concatenate scene clips with transitions in a single encode pass."""
        from moviepy import concatenate_videoclips, CompositeVideoClip

        if not clips:
            self._render_placeholder(total_duration, Path(out_path))
            return

        if len(clips) == 1:
            clips[0].write_videofile(
                out_path,
                fps=_FPS,
                codec=self._hw_encoder,
                audio=False,
                ffmpeg_params=self._video_ffmpeg_params(),
                logger=None,
            )
            clips[0].close()
            return

        # Apply per-scene transitions between consecutive clips
        transitioned = []
        for i, clip in enumerate(clips):
            if i == 0:
                transitioned.append(clip)
            else:
                # Determine transition from scene metadata
                t_type = TransitionType.DISSOLVE
                if scenes and i < len(scenes):
                    t_type = scenes[i].transition_in

                if t_type == TransitionType.CUT:
                    # Hard cut — no overlap
                    clip_cut = clip.with_start(
                        sum(c.duration for c in clips[:i])
                    )
                    transitioned.append(clip_cut)
                else:
                    # DISSOLVE or CROSSFADE — use CrossFadeIn
                    base_dur = 0.8 if t_type == TransitionType.CROSSFADE else _TRANSITION_DUR
                    fade_dur = min(base_dur, clips[i - 1].duration / 2, clip.duration / 2)
                    from moviepy import vfx
                    clip_faded = clip.with_effects([vfx.CrossFadeIn(fade_dur)])
                    clip_faded = clip_faded.with_start(
                        sum(c.duration for c in clips[:i]) - fade_dur * i
                    )
                    transitioned.append(clip_faded)

        try:
            final = concatenate_videoclips(transitioned, method="compose")
            final.write_videofile(
                out_path,
                fps=_FPS,
                codec=self._hw_encoder,
                audio=False,
                ffmpeg_params=self._video_ffmpeg_params(),
                logger=None,
            )
        finally:
            for c in clips:
                try:
                    c.close()
                except Exception:
                    pass

    # Public method kept for test compatibility
    def concat_clips(
        self,
        clip_paths: list[str],
        out_path: str,
        scenes: list[Scene] | None = None,
    ) -> None:
        """Concatenate pre-rendered clip files (test-facing API)."""
        from moviepy import VideoFileClip, concatenate_videoclips

        if not clip_paths:
            raise ValueError("concat_clips called with empty clip list")

        if len(clip_paths) == 1:
            shutil.copy(clip_paths[0], out_path)
            return

        loaded = [VideoFileClip(p) for p in clip_paths]
        try:
            final = concatenate_videoclips(loaded, method="compose")
            final.write_videofile(
                out_path,
                fps=_FPS,
                codec=self._hw_encoder,
                audio=False,
                ffmpeg_params=self._video_ffmpeg_params(),
                logger=None,
            )
        finally:
            for c in loaded:
                try:
                    c.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Public per-clip methods (test-facing API)
    # ------------------------------------------------------------------

    def render_image_to_clip(
        self,
        image_path: str,
        out_path: str,
        duration_sec: float,
        color_grade: ColorGrade = ColorGrade.DOCUMENTARY,
        text_overlays: list[TextOverlay] | None = None,
        scene_id: int = 1,
        is_first: bool = False,
        is_last: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """Convert a still image to a video clip. Writes file to out_path."""
        tmp = tmp_dir or Path(tempfile.mkdtemp(prefix="vf_img_"))
        clip = self._image_to_clip(image_path, duration_sec, scene_id)
        clip = self._apply_color_grade(clip, color_grade)
        if color_grade in _VIGNETTE_GRADES:
            clip = self._apply_vignette(clip)
        if color_grade in _LETTERBOX_GRADES:
            clip = self._apply_letterbox(clip)
        clip = self._apply_fades(clip, duration_sec, is_first, is_last)
        if text_overlays:
            clip = self._apply_text_overlays(clip, text_overlays, duration_sec, tmp, scene_id)
        clip.write_videofile(
            out_path,
            fps=_FPS,
            codec=self._hw_encoder,
            audio=False,
            ffmpeg_params=self._video_ffmpeg_params(),
            logger=None,
        )
        clip.close()

    def render_video_clip(
        self,
        video_path: str,
        out_path: str,
        duration_sec: float,
        color_grade: ColorGrade = ColorGrade.DOCUMENTARY,
        text_overlays: list[TextOverlay] | None = None,
        scene_id: int = 1,
        is_first: bool = False,
        is_last: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """Scale and color-grade a video clip to 1920x1080. Writes file to out_path."""
        tmp = tmp_dir or Path(tempfile.mkdtemp(prefix="vf_vid_"))
        clip = self._video_to_clip(video_path, duration_sec)
        clip = self._apply_color_grade(clip, color_grade)
        if color_grade in _VIGNETTE_GRADES:
            clip = self._apply_vignette(clip)
        if color_grade in _LETTERBOX_GRADES:
            clip = self._apply_letterbox(clip)
        clip = self._apply_fades(clip, duration_sec, is_first, is_last)
        if text_overlays:
            clip = self._apply_text_overlays(clip, text_overlays, duration_sec, tmp, scene_id)
        clip.write_videofile(
            out_path,
            fps=_FPS,
            codec=self._hw_encoder,
            audio=False,
            ffmpeg_params=self._video_ffmpeg_params(),
            logger=None,
        )
        clip.close()

    # ------------------------------------------------------------------
    # Placeholder (black screen)
    # ------------------------------------------------------------------

    def _render_placeholder(self, duration_sec: float, out: Path) -> None:
        """Render a black 1920x1080 clip for the given duration."""
        from moviepy import ColorClip
        clip = ColorClip(size=(_W, _H), color=(0, 0, 0)).with_duration(duration_sec)
        clip.write_videofile(
            str(out),
            fps=_FPS,
            codec=self._hw_encoder,
            audio=False,
            ffmpeg_params=self._video_ffmpeg_params(),
            logger=None,
        )
        clip.close()

    def _black_clip(self, duration: float):
        from moviepy import ColorClip
        return ColorClip(size=(_W, _H), color=(0, 0, 0)).with_duration(duration).with_fps(_FPS)

    # ------------------------------------------------------------------
    # Audio mixing (ffmpeg subprocess — LUFS normalization)
    # ------------------------------------------------------------------

    def _mix_audio(
        self,
        orch: Orchestration,
        out_path: Path,
        total_duration: float,
    ) -> None:
        inputs: list[str] = []
        volumes: list[float] = []
        start_times: list[float] = []
        bg_input_idx: int | None = None

        if orch.voiceover_tracks:
            for track in orch.voiceover_tracks:
                p = Path(track.local_path)
                if p.exists():
                    inputs.append(track.local_path)
                    volumes.append(track.volume)
                    start_times.append(getattr(track, "start_time", 0.0) or 0.0)

        if orch.background_music and Path(orch.background_music.local_path).exists():
            bg_input_idx = len(inputs)
            inputs.append(orch.background_music.local_path)
            volumes.append(orch.background_music.volume)
            start_times.append(0.0)

        for scene in orch.scenes:
            for sfx in scene.sfx_tracks:
                p = Path(sfx.local_path)
                if p.exists():
                    inputs.append(sfx.local_path)
                    volumes.append(sfx.volume)
                    start_times.append(0.0)

        if not inputs:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r=44100:cl=stereo:duration={total_duration}",
                "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ], capture_output=True, check=True)
            return

        # Voice EQ: presence boost + de-mud + light compression for broadcast clarity
        _VO_EQ = "highpass=f=80,equalizer=f=200:t=q:w=1:g=-1,equalizer=f=3000:t=q:w=1.5:g=2,acompressor=threshold=-20dB:ratio=3:attack=5:release=50"
        # Voiceover → broadcast speech level (-14 LUFS)
        _VO_LUFS    = "loudnorm=I=-14:TP=-1.5:LRA=11"
        # Music/SFX → background level (-35 LUFS)
        _MUSIC_LUFS = "loudnorm=I=-24:TP=-2:LRA=9"

        if len(inputs) == 1:
            is_music = (bg_input_idx == 0 and orch.background_music)
            lufs = _MUSIC_LUFS if is_music else _VO_LUFS
            eq = "" if is_music else f"{_VO_EQ},"
            af = f"volume={volumes[0]:.4f},{eq}{lufs}"
            if is_music:
                af += self._bg_music_afade(orch.background_music, total_duration)
            af += f",apad=whole_dur={total_duration}"
            subprocess.run([
                "ffmpeg", "-y",
                "-i", inputs[0],
                "-af", af,
                "-ar", "44100", "-ac", "2",
                "-t", str(total_duration),
                "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ], capture_output=True, check=True)
            return

        cmd = ["ffmpeg", "-y"]
        for inp in inputs:
            cmd += ["-i", inp]

        # Build per-input filter chains
        vol_parts = []
        vo_label = None
        music_label = None
        for i, vol in enumerate(volumes):
            is_music = (i == bg_input_idx and orch.background_music)
            lufs = _MUSIC_LUFS if is_music else _VO_LUFS
            eq = "" if is_music else f"{_VO_EQ},"
            fade_filters = self._bg_music_afade(orch.background_music, total_duration) if is_music else ""
            # Delay track if start_time > 0 (e.g. VO starts after title card)
            delay_ms = int((start_times[i] if i < len(start_times) else 0) * 1000)
            delay_filter = f"adelay={delay_ms}|{delay_ms}," if delay_ms > 0 else ""
            label = f"av{i}"
            vol_parts.append(
                f"[{i}:a]{delay_filter}volume={vol:.4f},{eq}{lufs}{fade_filters},"
                f"apad=whole_dur={total_duration}[{label}]"
            )
            if is_music:
                music_label = label
            elif vo_label is None:
                vo_label = label

        # Audio ducking: use sidechaincompress so music ducks when VO is present
        if vo_label and music_label and len(inputs) == 2:
            # Sidechain: music is compressed by VO signal
            fc = "; ".join(vol_parts) + (
                f"; [{music_label}][{vo_label}]sidechaincompress="
                f"threshold=0.02:ratio=4:attack=200:release=800[ducked]"
                f"; [{vo_label}][ducked]amix=inputs=2:normalize=0[aout]"
            )
        else:
            # Fallback: simple amix for >2 inputs or no clear VO/music split
            all_labels = "".join(f"[av{i}]" for i in range(len(inputs)))
            fc = "; ".join(vol_parts) + f"; {all_labels}amix=inputs={len(inputs)}:normalize=0[aout]"

        cmd += [
            "-filter_complex", fc,
            "-map", "[aout]",
            "-ar", "44100", "-ac", "2",
            "-t", str(total_duration),
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "Audio mix with ducking failed — retrying without ducking.\n"
                "FFmpeg stderr:\n%s",
                result.stderr[-600:],
            )
            # Fallback: simple amix without sidechain
            all_labels = "".join(f"[av{i}]" for i in range(len(inputs)))
            fc_simple = "; ".join(vol_parts) + f"; {all_labels}amix=inputs={len(inputs)}:normalize=0[aout]"
            cmd_retry = ["ffmpeg", "-y"]
            for inp in inputs:
                cmd_retry += ["-i", inp]
            cmd_retry += [
                "-filter_complex", fc_simple,
                "-map", "[aout]",
                "-ar", "44100", "-ac", "2",
                "-t", str(total_duration),
                "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ]
            result2 = subprocess.run(cmd_retry, capture_output=True, text=True)
            if result2.returncode != 0:
                # Last resort: voiceover only
                subprocess.run([
                    "ffmpeg", "-y", "-i", inputs[0],
                    "-ar", "44100", "-ac", "2",
                    "-t", str(total_duration), "-c:a", "aac", "-b:a", "192k",
                    str(out_path),
                ], capture_output=True, check=True)

    @staticmethod
    def _bg_music_afade(track, total_duration: float) -> str:
        parts = []
        fade_in = getattr(track, "fade_in", 0.0) or 0.0
        fade_out = getattr(track, "fade_out", 0.0) or 0.0
        if fade_in > 0:
            parts.append(f"afade=t=in:d={fade_in:.1f}")
        if fade_out > 0:
            st = max(0.0, total_duration - fade_out)
            parts.append(f"afade=t=out:st={st:.3f}:d={fade_out:.1f}")
        return "," + ",".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Final encode — mux video + audio
    # ------------------------------------------------------------------

    def _final_encode(self, video_path: str, audio_path: str, out_path: str) -> None:
        default = self._presets["output"]["default"]
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",   # video already encoded — just remux
            "-c:a", "aac", "-b:a", default["audio_bitrate"],
            "-movflags", "+faststart",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Final mux failed:\n{result.stderr[-500:]}")

    # ------------------------------------------------------------------
    # Text overlay helpers (Pillow)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Title Card (professional intro sequence)
    # ------------------------------------------------------------------

    def _brand(self, key: str, default):
        """Read a branding config value with fallback."""
        brand = self._presets.get("branding", {})
        val = brand.get(key, default)
        if isinstance(val, list) and len(val) >= 3:
            return tuple(val) + (255,) if len(val) == 3 else tuple(val)
        return val

    def _build_title_card(self, scene: Scene, tmp: Path):
        """Render a cinematic title card: serif title + gold divider + subtitle."""
        from PIL import Image, ImageDraw
        from moviepy import ImageClip, vfx

        duration = scene.duration_sec
        bg = self._brand("background_color", [0, 0, 0])
        img = Image.new("RGBA", (_W, _H), bg)
        draw = ImageDraw.Draw(img)

        primary = self._brand("primary_color", [240, 235, 220])
        secondary = self._brand("secondary_color", [201, 168, 76])
        accent = self._brand("accent_color", [140, 135, 125])

        # Fonts
        title_font = self._resolve_font("title_serif", 76)
        sub_font = self._resolve_font("overlay_sans", 32)

        # Title text (from the first text overlay or caption)
        title_text = ""
        for ov in (scene.text_overlays or []):
            if ov.text.strip():
                title_text = ov.text.strip()
                break
        if not title_text and scene.caption_text:
            title_text = scene.caption_text

        # Measure and position title above center
        title_lines = textwrap.wrap(title_text, width=35) or [title_text]
        line_h = 76 + 12
        title_h = len(title_lines) * line_h
        title_y = int(_H / 2) - title_h - 30

        for j, line in enumerate(title_lines):
            lw = draw.textlength(line, font=title_font)
            lx = int((_W - lw) / 2)
            self._draw_text_outlined(
                draw, (lx, title_y + j * line_h), line, font=title_font,
                fill=primary, outline_width=2, shadow=True,
            )

        # Divider accent line
        divider_y = int(_H / 2) + 5
        divider_w = 220
        div_x = int((_W - divider_w) / 2)
        draw.rectangle([div_x, divider_y, div_x + divider_w, divider_y + 2], fill=secondary)

        # Subtitle below divider (topic or tagline)
        sub_text = scene.caption_text or ""
        if sub_text:
            sw = draw.textlength(sub_text, font=sub_font)
            sx = int((_W - sw) / 2)
            sy = divider_y + 25
            self._draw_text_outlined(
                draw, (sx, sy), sub_text, font=sub_font,
                fill=accent, outline_width=1, shadow=False,
            )

        # Optional logo
        self._composite_logo(img)

        # Save as PNG and create clip
        card_path = str(tmp / "title_card.png")
        img.save(card_path, "PNG")

        clip = ImageClip(card_path).with_duration(duration).with_fps(_FPS)
        clip = clip.with_effects([vfx.FadeIn(1.5), vfx.FadeOut(0.5)])
        return clip

    def _composite_logo(self, img) -> None:
        """Composite brand logo onto an image if logo_path is configured."""
        from PIL import Image as PILImage
        logo_path = self._presets.get("branding", {}).get("logo_path", "")
        if not logo_path or not Path(logo_path).exists():
            return
        try:
            logo = PILImage.open(logo_path).convert("RGBA")
            target_h = int(self._presets.get("branding", {}).get("logo_size", 60))
            ratio = target_h / logo.height
            logo = logo.resize((int(logo.width * ratio), target_h), PILImage.LANCZOS)
            pos_name = self._presets.get("branding", {}).get("logo_position", "bottom_right")
            margin = 30
            if pos_name == "bottom_right":
                pos = (_W - logo.width - margin, _H - logo.height - margin)
            elif pos_name == "bottom_left":
                pos = (margin, _H - logo.height - margin)
            elif pos_name == "top_right":
                pos = (_W - logo.width - margin, margin)
            else:
                pos = (margin, margin)
            img.paste(logo, pos, logo)
        except Exception as e:
            logger.warning("Logo composite failed: %s", e)

    def _build_outro_card(self, scene: Scene, tmp: Path):
        """Render a cinematic outro card: closing text + gold divider + attribution."""
        from PIL import Image, ImageDraw
        from moviepy import ImageClip, vfx

        duration = scene.duration_sec
        bg = self._brand("background_color", [0, 0, 0])
        img = Image.new("RGBA", (_W, _H), bg)
        draw = ImageDraw.Draw(img)

        primary = self._brand("primary_color", [240, 235, 220])
        secondary = self._brand("secondary_color", [201, 168, 76])
        accent = self._brand("accent_color", [140, 135, 125])

        title_font = self._resolve_font("title_serif", 56)
        attr_font = self._resolve_font("overlay_sans", 24)

        # Closing text
        closing = scene.caption_text or "Thank you for watching"
        lines = textwrap.wrap(closing, width=40) or [closing]
        line_h = 56 + 10
        total_h = len(lines) * line_h
        y_start = int(_H / 2) - total_h - 20

        for j, line in enumerate(lines):
            lw = draw.textlength(line, font=title_font)
            lx = int((_W - lw) / 2)
            self._draw_text_outlined(
                draw, (lx, y_start + j * line_h), line, font=title_font,
                fill=primary, outline_width=2, shadow=True,
            )

        # Divider accent line
        divider_y = int(_H / 2) + 10
        div_x = int((_W - 160) / 2)
        draw.rectangle([div_x, divider_y, div_x + 160, divider_y + 2], fill=secondary)

        # Attribution from config
        attr_text = self._presets.get("branding", {}).get("attribution_text", "Generated by ScriptToReel")
        aw = draw.textlength(attr_text, font=attr_font)
        ax = int((_W - aw) / 2)
        self._draw_text_outlined(
            draw, (ax, divider_y + 25), attr_text, font=attr_font,
            fill=accent, outline_width=1, shadow=False,
        )

        # Optional logo
        self._composite_logo(img)

        card_path = str(tmp / "outro_card.png")
        img.save(card_path, "PNG")

        clip = ImageClip(card_path).with_duration(duration).with_fps(_FPS)
        clip = clip.with_effects([vfx.FadeIn(0.5), vfx.FadeOut(1.0)])
        return clip

    # ------------------------------------------------------------------
    # Burned-in Captions
    # ------------------------------------------------------------------

    def _build_caption_clips(self, scenes: list[Scene], tmp: Path) -> list:
        """Build caption ImageClips for all scenes that have caption_text."""
        from PIL import Image, ImageDraw
        from moviepy import ImageClip

        caption_clips = []
        for scene in scenes:
            # Skip title/outro cards — they render their own styled text
            if getattr(scene, "is_title_card", False) or getattr(scene, "is_outro_card", False):
                continue
            text = getattr(scene, "caption_text", None)
            if not text or not text.strip():
                continue

            # Split into chunks of ~7 words, sentence-aware
            words = text.split()
            chunks = []
            chunk = []
            for w in words:
                chunk.append(w)
                if len(chunk) >= 7 or w.endswith((".", "?", "!", ",", ";", ":")):
                    chunks.append(" ".join(chunk))
                    chunk = []
            if chunk:
                chunks.append(" ".join(chunk))
            if not chunks:
                continue

            # Use VO-synced timing (caption_start/end_sec) instead of scene visual timing
            cap_start = getattr(scene, "caption_start_sec", None) or scene.start_time
            cap_end = getattr(scene, "caption_end_sec", None) or scene.end_time
            cap_dur = cap_end - cap_start
            start = cap_start
            chunk_dur = cap_dur / len(chunks) if chunks else cap_dur

            font = self._resolve_font("caption_bold", 46)

            for ci, chunk_text in enumerate(chunks):
                img = Image.new("RGBA", (_W, _H), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                # Wrap if very long
                lines = textwrap.wrap(chunk_text, width=45) or [chunk_text]
                line_h = 46 + 8
                total_h = len(lines) * line_h

                # Position: 82% down (above letterbox bars)
                y_base = int(_H * 0.82) - total_h // 2

                for j, line in enumerate(lines):
                    lw = draw.textlength(line, font=font)
                    lx = int((_W - lw) / 2)
                    self._draw_text_outlined(
                        draw, (lx, y_base + j * line_h), line, font=font,
                        fill=(255, 255, 255, 255), outline_width=2,
                        shadow=True, shadow_offset=(2, 2), shadow_alpha=120,
                    )

                cap_path = str(tmp / f"cap_{scene.id}_{ci}.png")
                img.save(cap_path, "PNG")

                cap_clip = (
                    ImageClip(cap_path)
                    .with_duration(chunk_dur)
                    .with_start(start + ci * chunk_dur)
                )
                caption_clips.append(cap_clip)

        return caption_clips

    # ------------------------------------------------------------------
    # Text overlay rendering (Pillow)
    # ------------------------------------------------------------------

    def _resolve_font(self, purpose: str, size: int):
        """Load a Pillow font by purpose (title_serif, caption_bold, overlay_sans)."""
        from PIL import ImageFont
        font_config = self._presets.get("fonts", {})
        candidates = font_config.get(purpose, [])
        # Append generic fallbacks
        candidates = list(candidates) + [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    @staticmethod
    def _draw_text_outlined(
        draw, xy: tuple[int, int], text: str, font, fill=(255, 255, 255, 255),
        outline_color=(0, 0, 0, 255), outline_width: int = 2,
        shadow: bool = True, shadow_offset: tuple[int, int] = (3, 3),
        shadow_alpha: int = 100,
    ) -> None:
        """Draw text with outline and optional drop shadow for professional look."""
        x, y = xy
        # Drop shadow
        if shadow:
            draw.text(
                (x + shadow_offset[0], y + shadow_offset[1]),
                text, font=font,
                fill=(0, 0, 0, shadow_alpha),
            )
        # Outline: 8 compass directions
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        # Fill
        draw.text((x, y), text, font=font, fill=fill)

    def _render_overlay_png(self, overlay: TextOverlay, out_path: str) -> None:
        """Render a TextOverlay to a transparent 1920x1080 RGBA PNG via Pillow."""
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (_W, _H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        size, use_box, _ = self._overlay_style(overlay.style)
        font_purpose = "title_serif" if overlay.style == "title" else "overlay_sans"
        font = self._resolve_font(font_purpose, size)

        lines = textwrap.wrap(overlay.text, width=50) or [overlay.text]
        line_height = size + 8

        max_w = max(draw.textlength(ln, font=font) for ln in lines)
        total_h = len(lines) * line_height

        x = int((_W - max_w) / 2)
        if overlay.position == "bottom_third":
            y = int(_H * 2 / 3)
        elif overlay.position == "center":
            y = int((_H - total_h) / 2)
        elif overlay.position == "top":
            y = 50
        elif overlay.position == "bottom":
            y = _H - total_h - 50
        else:
            y = _H - total_h - 80

        if use_box:
            pad = 18
            box_layer = Image.new("RGBA", (_W, _H), (0, 0, 0, 0))
            box_draw = ImageDraw.Draw(box_layer)
            box_draw.rectangle(
                [x - pad, y - pad, x + int(max_w) + pad, y + total_h + pad],
                fill=(0, 0, 0, int(0.55 * 255)),
            )
            img = Image.alpha_composite(img, box_layer)
            draw = ImageDraw.Draw(img)

        for j, line in enumerate(lines):
            lw = draw.textlength(line, font=font)
            lx = int((_W - lw) / 2)
            self._draw_text_outlined(draw, (lx, y + j * line_height), line, font=font)

        img.save(out_path, "PNG")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _overlay_style(style: str) -> tuple[int, bool, str]:
        styles = {
            "lower_third": (42, True, "black@0.6"),
            "title":       (72, True, "black@0.5"),
            "subtitle":    (36, False, "black@0.6"),
        }
        return styles.get(style, (42, True, "black@0.6"))

    def _video_ffmpeg_params(self) -> list[str]:
        params = ["-pix_fmt", "yuv420p"]
        if self._hw_encoder == "h264_videotoolbox":
            params += ["-b:v", "5000k"]
        else:
            params += ["-crf", "22", "-preset", "fast"]
        return params

    def _grade_params(self, grade: ColorGrade) -> dict:
        defaults = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "gamma": 1.0}
        try:
            params = self._presets["color_grades"].get(grade.value, {})
            defaults.update(params)
        except Exception:
            pass
        return defaults

    def _detect_encoder(self) -> str:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_videotoolbox" in result.stdout:
            logger.info("Using h264_videotoolbox (Apple Silicon hardware)")
            return "h264_videotoolbox"
        logger.info("Falling back to libx264")
        return "libx264"

    def _load_orchestration(self) -> Orchestration:
        p = self.project_dir / "orchestration.json"
        if not p.exists():
            raise FileNotFoundError(f"orchestration.json not found in {self.project_dir}")
        return Orchestration(**json.loads(p.read_text()))

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
                project_id, "module_5_render", status,
                projects_root=self.project_dir.parent,
                **kwargs,
            )
        except Exception as e:
            logger.warning("Could not update pipeline status: %s", e)
