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

            # Step 1 + 2: build and concatenate scene clips
            video_only = tmp / "video_only.mp4"
            if orch.scenes:
                clips = self._build_all_scene_clips(orch, tmp)
                self._write_concat(clips, str(video_only), orch.total_duration_sec)
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
        if is_first:
            effects.append(vfx.FadeIn(0.4))
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

    def _write_concat(self, clips: list, out_path: str, total_duration: float) -> None:
        """Concatenate scene clips with dissolve transitions and write to file."""
        from moviepy import concatenate_videoclips

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

        # Apply crossfade transitions between consecutive clips
        transitioned = []
        for i, clip in enumerate(clips):
            if i == 0:
                transitioned.append(clip)
            else:
                # Each subsequent clip starts _TRANSITION_DUR earlier
                # and fades in over the transition window
                fade_dur = min(_TRANSITION_DUR, clips[i - 1].duration / 2, clip.duration / 2)
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
        bg_input_idx: int | None = None

        if orch.voiceover_tracks:
            for track in orch.voiceover_tracks:
                p = Path(track.local_path)
                if p.exists():
                    inputs.append(track.local_path)
                    volumes.append(track.volume)

        if orch.background_music and Path(orch.background_music.local_path).exists():
            bg_input_idx = len(inputs)
            inputs.append(orch.background_music.local_path)
            volumes.append(orch.background_music.volume)

        for scene in orch.scenes:
            for sfx in scene.sfx_tracks:
                p = Path(sfx.local_path)
                if p.exists():
                    inputs.append(sfx.local_path)
                    volumes.append(sfx.volume)

        if not inputs:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r=44100:cl=stereo:duration={total_duration}",
                "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ], capture_output=True, check=True)
            return

        # Voiceover → normalise to broadcast speech level (-14 LUFS)
        # Music/SFX  → normalise to background level (-35 LUFS) then apply volume
        #              This prevents loudnorm from re-amplifying the music back up.
        _VO_LUFS    = "loudnorm=I=-14:TP=-1.5:LRA=11"
        _MUSIC_LUFS = "loudnorm=I=-35:TP=-3:LRA=7"

        if len(inputs) == 1:
            is_music = (bg_input_idx == 0 and orch.background_music)
            lufs = _MUSIC_LUFS if is_music else _VO_LUFS
            af = f"volume={volumes[0]:.4f},{lufs}"
            if is_music:
                af += self._bg_music_afade(orch.background_music, total_duration)
            af += f",apad=whole_dur={total_duration}"
            subprocess.run([
                "ffmpeg", "-y",
                "-i", inputs[0],
                "-af", af,
                "-t", str(total_duration),
                "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ], capture_output=True, check=True)
            return

        cmd = ["ffmpeg", "-y"]
        for inp in inputs:
            cmd += ["-i", inp]

        vol_parts = []
        vol_outs = []
        for i, vol in enumerate(volumes):
            vol_outs.append(f"av{i}")
            is_music = (i == bg_input_idx and orch.background_music)
            lufs = _MUSIC_LUFS if is_music else _VO_LUFS
            fade_filters = self._bg_music_afade(orch.background_music, total_duration) if is_music else ""
            vol_parts.append(
                f"[{i}:a]volume={vol:.4f},{lufs}{fade_filters},"
                f"apad=whole_dur={total_duration}[av{i}]"
            )

        mix_inputs = "".join(f"[{v}]" for v in vol_outs)
        fc = "; ".join(vol_parts) + f"; {mix_inputs}amix=inputs={len(inputs)}:normalize=0[aout]"
        cmd += [
            "-filter_complex", fc,
            "-map", "[aout]",
            "-t", str(total_duration),
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "Multi-track audio mix failed (falling back to voiceover only).\n"
                "FFmpeg stderr:\n%s",
                result.stderr[-600:],
            )
            subprocess.run([
                "ffmpeg", "-y", "-i", inputs[0],
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

    def _render_overlay_png(self, overlay: TextOverlay, out_path: str) -> None:
        """Render a TextOverlay to a transparent 1920x1080 RGBA PNG via Pillow."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGBA", (_W, _H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        size, use_box, _ = self._overlay_style(overlay.style)

        font = None
        for candidate in [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]:
            if Path(candidate).exists():
                try:
                    font = ImageFont.truetype(candidate, size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

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
                fill=(0, 0, 0, int(0.65 * 255)),
            )
            img = Image.alpha_composite(img, box_layer)
            draw = ImageDraw.Draw(img)

        for j, line in enumerate(lines):
            lw = draw.textlength(line, font=font)
            lx = int((_W - lw) / 2)
            draw.text((lx, y + j * line_height), line, font=font, fill=(255, 255, 255, 255))

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
