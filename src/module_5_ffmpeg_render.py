"""
Module 5 — FFmpeg Rendering.

Parses orchestration.json and produces a final 1080p H.264 MP4 using
h264_videotoolbox (M4 Pro hardware encoder) with libx264 fallback.

Pipeline:
  1. Per-scene: image/video → scaled 1920x1080 clip with color grade,
     Ken Burns motion (images), vignette, letterbox, fade in/out
  2. Concat scene clips with dissolve transitions
  3. Mix audio (voiceover + background music with fade in/out)
  4. Final encode with VideoToolbox, BT.709, -movflags +faststart
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

from src.project_manager import update_pipeline_status
from src.utils.config_loader import load_ffmpeg_presets
from src.utils.ffmpeg_builder import (
    FFmpegCommand,
    build_audio_amix_filter,
    build_color_grade_filter,
    build_concat_filter,
    build_drawtext_filter,
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


class RenderModule:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.output_dir = self.project_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._presets = load_ffmpeg_presets()
        self._hw_encoder = self._detect_encoder()
        self._has_drawtext = self._detect_drawtext()
        self._has_drawbox = self._detect_drawbox()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> str:
        orch = self._load_orchestration()
        logger.info("Module 5: rendering %d scenes, total %.1fs", len(orch.scenes), orch.total_duration_sec)

        with tempfile.TemporaryDirectory(prefix="vf_render_") as tmpdir:
            tmp = Path(tmpdir)
            scenes = orch.scenes
            n_scenes = len(scenes)

            # Step 1: render each scene to a clip
            scene_clips = []
            for idx, scene in enumerate(scenes):
                clip = tmp / f"scene_{scene.id:03d}.mp4"
                self._render_scene(
                    scene, clip, tmp,
                    is_first=(idx == 0),
                    is_last=(idx == n_scenes - 1),
                )
                scene_clips.append(str(clip))

            # Step 2: concat scene clips (with xfade transitions where applicable)
            concat_video = tmp / "concat.mp4"
            if not scene_clips:
                # No visual assets — render a solid black placeholder for the full duration
                logger.warning(
                    "No scenes available — rendering black placeholder for %.1fs",
                    orch.total_duration_sec,
                )
                self._render_placeholder(orch.total_duration_sec, concat_video)
            else:
                self.concat_clips(scene_clips, str(concat_video), scenes=orch.scenes)

            # Step 3: mix audio
            audio_out = tmp / "mixed_audio.aac"
            self._mix_audio(orch, audio_out, total_duration=orch.total_duration_sec)

            # Step 4: final encode
            final_out = self.output_dir / "final_video.mp4"
            self._final_encode(str(concat_video), str(audio_out), str(final_out))

        self._update_status(ModuleStatus.COMPLETE, output_file=str(final_out))
        logger.info("Render complete: %s", final_out)
        return str(final_out)

    # ------------------------------------------------------------------
    # Story 5.2 — Per-scene rendering
    # ------------------------------------------------------------------

    def _render_scene(
        self,
        scene: Scene,
        out: Path,
        tmp_dir: Path,
        is_first: bool = False,
        is_last: bool = False,
    ) -> None:
        asset_path = Path(scene.asset_path)
        if not asset_path.exists():
            logger.warning("Asset not found: %s — using color placeholder", asset_path)
            self._render_placeholder(scene.duration_sec, out)
            return

        suffix = asset_path.suffix.lower()
        overlays = scene.text_overlays
        if suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            self.render_image_to_clip(
                str(asset_path), str(out), scene.duration_sec, scene.color_grade,
                overlays,
                scene_id=scene.id,
                is_first=is_first,
                is_last=is_last,
                tmp_dir=tmp_dir,
            )
        else:
            self.render_video_clip(
                str(asset_path), str(out), scene.duration_sec, scene.color_grade,
                overlays,
                scene_id=scene.id,
                is_first=is_first,
                is_last=is_last,
                tmp_dir=tmp_dir,
            )

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
        """Convert a still image to a video clip with Ken Burns motion + grade."""
        grade = self._grade_params(color_grade)
        n = max(2, int(duration_sec * _FPS))

        # C1 — Ken Burns (scale+pad+crop — fast, no zoompan buffering)
        kb = self._ken_burns_filter(scene_id, n)

        # C2 — Vignette
        vignette = ",vignette=PI/4" if grade.get("vignette") else ""

        # C5 — Fade in / out
        fade = self._fade_filter(duration_sec, is_first, is_last)

        # C4 — Letterbox bars (2.39:1 widescreen)
        letterbox = self._letterbox_filter(color_grade) if self._has_drawbox else ""

        # kb already includes the full scale→pad→crop→scale→fps chain
        filter_chain = (
            f"[0:v]{kb},"
            f"eq=brightness={grade['brightness']:.3f}:contrast={grade['contrast']:.3f}:"
            f"saturation={grade['saturation']:.3f}:gamma={grade['gamma']:.3f}"
            f"{vignette}"
            f"{fade}"
            f"{letterbox}"
            f"[grade]"
        )

        # C3 — Text overlays via Pillow PNG + overlay filter
        extra_inputs, overlay_frag = self._build_pillow_overlays(
            text_overlays, duration_sec, tmp_dir, input_offset=1
        )
        filter_chain += overlay_frag

        cmd = self._build_scene_cmd(image_path, out_path, filter_chain, duration_sec, extra_inputs)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg image→clip error:\n{result.stderr[-400:]}")

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
        """Scale and color-grade a video clip to 1920x1080 with polish effects."""
        grade = self._grade_params(color_grade)

        # C2 — Vignette
        vignette = ",vignette=PI/4" if grade.get("vignette") else ""

        # C5 — Fade in / out
        fade = self._fade_filter(duration_sec, is_first, is_last)

        # C4 — Letterbox
        letterbox = self._letterbox_filter(color_grade) if self._has_drawbox else ""

        filter_chain = (
            f"[0:v]scale={_W}:{_H}:force_original_aspect_ratio=decrease,"
            f"pad={_W}:{_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"eq=brightness={grade['brightness']:.3f}:contrast={grade['contrast']:.3f}:"
            f"saturation={grade['saturation']:.3f}:gamma={grade['gamma']:.3f}"
            f"{vignette}"
            f"{fade}"
            f"{letterbox}"
            f"[grade]"
        )

        # C3 — Text overlays
        extra_inputs, overlay_frag = self._build_pillow_overlays(
            text_overlays, duration_sec, tmp_dir, input_offset=1
        )
        filter_chain += overlay_frag

        cmd = self._build_scene_cmd(video_path, out_path, filter_chain, duration_sec, extra_inputs)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg video clip error:\n{result.stderr[-400:]}")

    def _render_placeholder(self, duration_sec: float, out: Path) -> None:
        cmd = (
            FFmpegCommand()
            .input(
                f"color=c=black:size={_W}x{_H}:duration={duration_sec}:rate={_FPS}",
                f="lavfi",
            )
            .output(str(out), **self._video_encode_opts())
        )
        self._run(cmd)

    # ------------------------------------------------------------------
    # Story 5.3 — Concat
    # ------------------------------------------------------------------

    def concat_clips(
        self,
        clip_paths: list[str],
        out_path: str,
        scenes: list[Scene] | None = None,
    ) -> None:
        """Concatenate scene clips.

        When scenes are provided and consecutive scenes have dissolve/crossfade
        transitions, xfade filters are applied. Falls back to concat demuxer
        (hard cuts, no re-encode) when transitions are not needed.
        """
        if not clip_paths:
            raise ValueError("concat_clips called with empty clip list")
        if len(clip_paths) == 1:
            shutil.copy(clip_paths[0], out_path)
            return

        # Determine if any scene requires a smooth transition
        use_xfade = False
        if scenes and len(scenes) == len(clip_paths):
            xfade_types = {TransitionType.DISSOLVE, TransitionType.CROSSFADE}
            use_xfade = any(
                s.transition_in in xfade_types or s.transition_out in xfade_types
                for s in scenes[1:]  # first scene has no "previous" to transition from
            )

        if use_xfade and scenes:
            self._concat_with_xfade(clip_paths, out_path, scenes)
            return

        # Default: fast concat demuxer (c=copy, no re-encode)
        import tempfile as _tf
        with _tf.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            list_path = fh.name
            for p in clip_paths:
                fh.write(f"file '{p}'\n")

        cmd = (
            FFmpegCommand()
            .input(list_path, f="concat", safe="0")
            .output(out_path, c="copy")
        )
        self._run(cmd)
        Path(list_path).unlink(missing_ok=True)

    def _concat_with_xfade(
        self,
        clip_paths: list[str],
        out_path: str,
        scenes: list[Scene],
    ) -> None:
        """Concatenate clips using xfade filter for smooth transitions."""
        xfade_types = {TransitionType.DISSOLVE, TransitionType.CROSSFADE}
        transition_dur = 0.5  # seconds per transition

        cmd = ["ffmpeg", "-y"]
        for path in clip_paths:
            cmd += ["-i", path]

        # Build filter: setpts normalisation then chain xfade between pairs
        parts: list[str] = []
        for i in range(len(clip_paths)):
            parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

        current = "v0"
        offset = 0.0
        for i, scene in enumerate(scenes[1:], start=1):
            prev_dur = scenes[i - 1].duration_sec
            offset += prev_dur - transition_dur
            offset = max(offset, 0.0)

            do_xfade = (
                scene.transition_in in xfade_types
                or scenes[i - 1].transition_out in xfade_types
            )
            out_stream = f"xf{i}" if i < len(scenes) - 1 else "vout"

            if do_xfade:
                parts.append(
                    f"[{current}][v{i}]xfade=transition=dissolve:"
                    f"duration={transition_dur:.3f}:offset={offset:.3f}[{out_stream}]"
                )
            else:
                parts.append(f"[{current}][v{i}]concat=n=2:v=1:a=0[{out_stream}]")
                offset += scenes[i].duration_sec - transition_dur

            current = out_stream

        filter_complex = "; ".join(parts)
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", self._hw_encoder,
            "-pix_fmt", "yuv420p",
            "-r", str(_FPS),
        ]
        if self._hw_encoder == "h264_videotoolbox":
            cmd += ["-b:v", "5000k"]
        else:
            cmd += ["-crf", "22", "-preset", "fast"]
        cmd.append(out_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "xfade concat failed (falling back to hard cuts).\nFFmpeg stderr:\n%s",
                result.stderr[-600:],
            )
            self.concat_clips(clip_paths, out_path, scenes=None)

    # ------------------------------------------------------------------
    # Story 5.4 — Audio mixing
    # ------------------------------------------------------------------

    def _mix_audio(
        self,
        orch: Orchestration,
        out_path: Path,
        total_duration: float,
    ) -> None:
        inputs: list[str] = []
        volumes: list[float] = []
        # Track which input index is background music for fade application
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

        # Collect SFX tracks from all scenes
        for scene in orch.scenes:
            for sfx in scene.sfx_tracks:
                p = Path(sfx.local_path)
                if p.exists():
                    inputs.append(sfx.local_path)
                    volumes.append(sfx.volume)
                else:
                    logger.debug("SFX track not found, skipping: %s", sfx.local_path)

        if not inputs:
            # Generate silence
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:duration={total_duration}",
                "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ], capture_output=True, check=True)
            return

        if len(inputs) == 1:
            # C6 — audio fade on background music when it's the only track
            af = f"volume={volumes[0]:.4f}"
            if bg_input_idx == 0 and orch.background_music:
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

        # Multiple inputs — build input list + amix
        cmd = ["ffmpeg", "-y"]
        for inp in inputs:
            cmd += ["-i", inp]

        vol_parts = []
        vol_outs = []
        for i, vol in enumerate(volumes):
            vol_outs.append(f"av{i}")
            # C6 — audio fade on background music
            if i == bg_input_idx and orch.background_music:
                fade_filters = self._bg_music_afade(orch.background_music, total_duration)
            else:
                fade_filters = ""
            vol_parts.append(
                f"[{i}:a]volume={vol:.4f}{fade_filters},apad=whole_dur={total_duration}[av{i}]"
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
        """Return afade filter string for background music fade in/out."""
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
    # Story 5.5 — Final encode
    # ------------------------------------------------------------------

    def _final_encode(self, video_path: str, audio_path: str, out_path: str) -> None:
        default = self._presets["output"]["default"]

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", self._hw_encoder,
        ]

        if self._hw_encoder == "h264_videotoolbox":
            cmd += ["-b:v", default["bitrate"]]
        else:
            cmd += ["-crf", "22", "-preset", "fast"]

        cmd += [
            "-c:a", "aac", "-b:a", default["audio_bitrate"],
            "-pix_fmt", default["pixel_format"],
            "-color_primaries", default["color_primaries"],
            "-color_trc", default["color_trc"],
            "-colorspace", default["colorspace"],
            "-movflags", "+faststart",
            "-threads", str(default["threads"]),
            out_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Final encode failed:\n{result.stderr[-500:]}")

    # ------------------------------------------------------------------
    # C1 — Ken Burns motion helpers
    # ------------------------------------------------------------------

    # Padded canvas: 15% larger than output gives smooth zoom/pan headroom
    _KB_W = 2208   # 1920 * 1.15  (even)
    _KB_H = 1242   # 1080 * 1.15  (even)

    def _ken_burns_filter(self, scene_id: int, n_frames: int) -> str:
        """Return a filter chain fragment for Ken Burns motion on still images.

        Uses scale+pad+crop (fast, no look-ahead buffering like zoompan).
        Returns the full chain from raw input to 1920x1080 output at _FPS,
        ready to be prefixed with '[0:v]' and followed by ',eq=...'.

        4 styles cycle by scene_id:
          0 — zoom in toward center
          1 — zoom out from center
          2 — pan right
          3 — pan left
        """
        style = (scene_id - 1) % 4
        denom = max(1, n_frames - 1)
        kw, kh = self._KB_W, self._KB_H
        pan_cy = (kh - _H) // 2  # vertical center offset for pan styles

        # loop=n_frames bounds the filter pipeline so it terminates cleanly;
        # without this, -stream_loop -1 creates an infinite stream that -t
        # may not reliably cut inside a filter_complex graph.
        loop_fps = f"loop={n_frames}:size=1:start=0,fps={_FPS}"

        # Scale source to padded canvas (handles any aspect ratio, letterboxed)
        scale_pad = (
            f"{loop_fps},"
            f"scale={kw}:{kh}:force_original_aspect_ratio=decrease,"
            f"pad={kw}:{kh}:(ow-iw)/2:(oh-ih)/2:black"
        )

        if style == 0:  # zoom in — crop window shrinks from canvas to output size
            cw = f"{kw}+({_W}-{kw})*n/{denom}"
            ch = f"{kh}+({_H}-{kh})*n/{denom}"
            return (
                f"{scale_pad},"
                f"crop=w='{cw}':h='{ch}':x='({kw}-out_w)/2':y='({kh}-out_h)/2',"
                f"scale={_W}:{_H}:flags=bicubic"
            )

        elif style == 1:  # zoom out — crop window expands from output to canvas size
            cw = f"{_W}+({kw}-{_W})*n/{denom}"
            ch = f"{_H}+({kh}-{_H})*n/{denom}"
            return (
                f"{scale_pad},"
                f"crop=w='{cw}':h='{ch}':x='({kw}-out_w)/2':y='({kh}-out_h)/2',"
                f"scale={_W}:{_H}:flags=bicubic"
            )

        elif style == 2:  # pan right — fixed 1920x1080 crop, x moves right→left
            max_x = kw - _W   # = 288
            cx = f"{max_x}*n/{denom}"
            return (
                f"{scale_pad},"
                f"crop=w={_W}:h={_H}:x='{cx}':y={pan_cy}"
            )

        else:  # pan left — fixed 1920x1080 crop, x moves left→right
            max_x = kw - _W
            cx = f"{max_x}*(1-n/{denom})"
            return (
                f"{scale_pad},"
                f"crop=w={_W}:h={_H}:x='{cx}':y={pan_cy}"
            )

    # ------------------------------------------------------------------
    # C2 / C4 / C5 — filter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fade_filter(duration_sec: float, is_first: bool, is_last: bool) -> str:
        """Return comma-prefixed fade filter(s) or empty string."""
        parts = []
        if is_first:
            parts.append("fade=t=in:st=0:d=0.4")
        if is_last:
            st = max(0.0, duration_sec - 0.5)
            parts.append(f"fade=t=out:st={st:.3f}:d=0.5")
        return "," + ",".join(parts) if parts else ""

    def _letterbox_filter(self, color_grade: ColorGrade) -> str:
        """Return comma-prefixed drawbox letterbox or empty string."""
        if color_grade not in _LETTERBOX_GRADES:
            return ""
        return (
            ",drawbox=x=0:y=0:w=iw:h=140:c=black:t=fill"
            ",drawbox=x=0:y=ih-140:w=iw:h=140:c=black:t=fill"
        )

    # ------------------------------------------------------------------
    # C3 — Pillow-based text overlay
    # ------------------------------------------------------------------

    def _build_pillow_overlays(
        self,
        overlays: list[TextOverlay] | None,
        duration_sec: float,
        tmp_dir: Path | None,
        input_offset: int = 1,
    ) -> tuple[list[str], str]:
        """Return (extra_input_paths, filter_fragment) for text overlays.

        Falls back to null (pass-through) if Pillow unavailable, no tmp_dir,
        or no enabled overlays.
        """
        enabled = [o for o in (overlays or []) if o.enabled and o.text.strip()]
        if not enabled or tmp_dir is None:
            return [], "; [grade]null[v]"

        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            logger.warning("Pillow not installed — skipping text overlays")
            return [], "; [grade]null[v]"

        extra_inputs: list[str] = []
        parts: list[str] = []
        current = "grade"

        for i, overlay in enumerate(enabled):
            png_path = str(tmp_dir / f"overlay_{i}.png")
            try:
                self._render_overlay_png(overlay, png_path)
            except Exception as e:
                logger.warning("Failed to render overlay PNG: %s — skipping", e)
                continue

            extra_inputs.append(png_path)
            idx = input_offset + len(extra_inputs) - 1

            start = getattr(overlay, "start_time", 0.0) or 0.0
            end = getattr(overlay, "end_time", None) or duration_sec

            out_stream = "v" if i == len(enabled) - 1 else f"ov{i}"
            parts.append(
                f"[{current}][{idx}:v]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'[{out_stream}]"
            )
            current = out_stream

        if not parts:
            return [], "; [grade]null[v]"

        return extra_inputs, "; " + "; ".join(parts)

    def _render_overlay_png(self, overlay: TextOverlay, out_path: str) -> None:
        """Render a TextOverlay to a transparent 1920x1080 RGBA PNG via Pillow."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGBA", (_W, _H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        size, use_box, _ = self._overlay_style(overlay.style)

        # Find a usable TTF font
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
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

        # Wrap long text
        lines = textwrap.wrap(overlay.text, width=50) or [overlay.text]
        line_height = size + 8

        # Measure total block
        max_w = max(draw.textlength(ln, font=font) for ln in lines)
        total_h = len(lines) * line_height

        # Compute position
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

        # Background box
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

        # Draw each line
        for j, line in enumerate(lines):
            lw = draw.textlength(line, font=font)
            lx = int((_W - lw) / 2)
            draw.text((lx, y + j * line_height), line, font=font, fill=(255, 255, 255, 255))

        img.save(out_path, "PNG")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_scene_cmd(
        self,
        input_path: str,
        out_path: str,
        filter_chain: str,
        duration_sec: float,
        extra_inputs: list[str] | None = None,
    ) -> list[str]:
        """Build a subprocess argv for a per-scene encode."""
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",   # loop main input to fill duration
            "-i", input_path,
        ]
        # Extra inputs (e.g. overlay PNGs) — no stream_loop
        for ei in (extra_inputs or []):
            cmd += ["-i", ei]
        cmd += [
            "-filter_complex", filter_chain,
            "-map", "[v]",
            "-t", str(duration_sec),
            "-c:v", self._hw_encoder,
            "-pix_fmt", "yuv420p",
            "-r", str(_FPS),
        ]
        if self._hw_encoder == "h264_videotoolbox":
            cmd += ["-b:v", "3000k"]
        else:
            cmd += ["-crf", "22", "-preset", "fast"]
        cmd.append(out_path)
        return cmd

    # Keep for backward compat (tests that call directly)
    def _build_overlay_chain(self, overlays: list[TextOverlay] | None) -> str:
        """Legacy helper — returns null pass-through chain."""
        return "; [grade]null[v]"

    @staticmethod
    def _find_system_font() -> str:
        candidates = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for p in candidates:
            if Path(p).exists():
                return p
        return ""

    @staticmethod
    def _overlay_style(style: str) -> tuple[int, bool, str]:
        styles = {
            "lower_third": (42, True, "black@0.6"),
            "title":       (72, True, "black@0.5"),
            "subtitle":    (36, False, "black@0.6"),
        }
        return styles.get(style, (42, True, "black@0.6"))

    @staticmethod
    def _overlay_xy(position: str) -> tuple[str, str]:
        positions = {
            "bottom_third": ("(w-text_w)/2", "h*2/3"),
            "center":       ("(w-text_w)/2", "(h-text_h)/2"),
            "top":          ("(w-text_w)/2", "50"),
            "bottom":       ("(w-text_w)/2", "h-text_h-50"),
        }
        return positions.get(position, ("(w-text_w)/2", "h-text_h-80"))

    def _detect_encoder(self) -> str:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_videotoolbox" in result.stdout:
            logger.info("Using h264_videotoolbox (M4 Pro hardware)")
            return "h264_videotoolbox"
        logger.info("Falling back to libx264")
        return "libx264"

    def _detect_drawtext(self) -> bool:
        result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
        available = "drawtext" in result.stdout
        if not available:
            logger.info("FFmpeg drawtext unavailable — using Pillow for text overlays")
        return available

    def _detect_drawbox(self) -> bool:
        result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
        available = "drawbox" in result.stdout
        if not available:
            logger.info("FFmpeg drawbox unavailable — letterbox disabled")
        return available

    def _video_encode_opts(self) -> dict:
        opts: dict = {
            "c_v": self._hw_encoder,
            "pix_fmt": "yuv420p",
            "r": str(_FPS),
        }
        if self._hw_encoder == "h264_videotoolbox":
            opts["b_v"] = "3000k"
        else:
            opts["crf"] = "22"
            opts["preset"] = "fast"
        return opts

    def _grade_params(self, grade: ColorGrade) -> dict:
        defaults = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "gamma": 1.0}
        try:
            params = self._presets["color_grades"].get(grade.value, {})
            defaults.update(params)
        except Exception:
            pass
        return defaults

    def _run(self, cmd: FFmpegCommand) -> None:
        result = cmd.run()
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr[-400:]}")

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
