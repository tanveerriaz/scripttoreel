"""
Module 5 — FFmpeg Rendering.

Parses orchestration.json and produces a final 1080p H.264 MP4 using
h264_videotoolbox (M4 Pro hardware encoder) with libx264 fallback.

Pipeline:
  1. Per-scene: image/video → scaled 1920x1080 clip with color grade
  2. Concat scene clips with dissolve transitions
  3. Mix audio (voiceover + background music)
  4. Final encode with VideoToolbox, BT.709, -movflags +faststart
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from src.project_manager import update_pipeline_status
from src.utils.config_loader import load_ffmpeg_presets
from src.utils.ffmpeg_builder import (
    FFmpegCommand,
    build_audio_amix_filter,
    build_color_grade_filter,
    build_concat_filter,
    build_scale_pad_filter,
    build_xfade_filter,
)
from src.utils.json_schemas import (
    ColorGrade,
    ModuleStatus,
    Orchestration,
    Scene,
    TransitionType,
)

logger = logging.getLogger(__name__)

_W, _H = 1920, 1080
_FPS = 30


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
        logger.info("Module 5: rendering %d scenes, total %.1fs", len(orch.scenes), orch.total_duration_sec)

        with tempfile.TemporaryDirectory(prefix="vf_render_") as tmpdir:
            tmp = Path(tmpdir)

            # Step 1: render each scene to a clip
            scene_clips = []
            for scene in orch.scenes:
                clip = tmp / f"scene_{scene.id:03d}.mp4"
                self._render_scene(scene, clip)
                scene_clips.append(str(clip))

            # Step 2: concat scene clips
            concat_video = tmp / "concat.mp4"
            self.concat_clips(scene_clips, str(concat_video))

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

    def _render_scene(self, scene: Scene, out: Path) -> None:
        asset_path = Path(scene.asset_path)
        if not asset_path.exists():
            logger.warning("Asset not found: %s — using color placeholder", asset_path)
            self._render_placeholder(scene.duration_sec, out)
            return

        suffix = asset_path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            self.render_image_to_clip(str(asset_path), str(out), scene.duration_sec, scene.color_grade)
        else:
            self.render_video_clip(str(asset_path), str(out), scene.duration_sec, scene.color_grade)

    def render_image_to_clip(
        self,
        image_path: str,
        out_path: str,
        duration_sec: float,
        color_grade: ColorGrade = ColorGrade.DOCUMENTARY,
    ) -> None:
        """Convert a still image to a video clip of given duration."""
        grade = self._grade_params(color_grade)
        filter_chain = (
            f"[0:v]loop={int(duration_sec * _FPS)}:size=1:start=0,fps={_FPS},"
            f"scale={_W}:{_H}:force_original_aspect_ratio=decrease,"
            f"pad={_W}:{_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"eq=brightness={grade['brightness']:.3f}:contrast={grade['contrast']:.3f}:"
            f"saturation={grade['saturation']:.3f}:gamma={grade['gamma']:.3f}[v]"
        )
        cmd = self._build_scene_cmd(image_path, out_path, filter_chain, duration_sec)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg image→clip error:\n{result.stderr[-400:]}")

    def render_video_clip(
        self,
        video_path: str,
        out_path: str,
        duration_sec: float,
        color_grade: ColorGrade = ColorGrade.DOCUMENTARY,
    ) -> None:
        """Scale and color-grade a video clip to 1920x1080."""
        grade = self._grade_params(color_grade)
        filter_chain = (
            f"[0:v]scale={_W}:{_H}:force_original_aspect_ratio=decrease,"
            f"pad={_W}:{_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"eq=brightness={grade['brightness']:.3f}:contrast={grade['contrast']:.3f}:"
            f"saturation={grade['saturation']:.3f}:gamma={grade['gamma']:.3f}[v]"
        )
        cmd = self._build_scene_cmd(video_path, out_path, filter_chain, duration_sec)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg video clip error:\n{result.stderr[-400:]}")

    def _render_placeholder(self, duration_sec: float, out: Path) -> None:
        cmd = (
            FFmpegCommand()
            .input(f"color=c=black:size={_W}x{_H}:duration={duration_sec}:rate={_FPS}", f=True)
            .output(str(out), **self._video_encode_opts())
        )
        self._run(cmd)

    # ------------------------------------------------------------------
    # Story 5.3 — Concat
    # ------------------------------------------------------------------

    def concat_clips(self, clip_paths: list[str], out_path: str) -> None:
        """Concatenate scene clips. Uses concat demuxer for reliability."""
        if len(clip_paths) == 1:
            shutil.copy(clip_paths[0], out_path)
            return

        # Write a concat list file
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

        if orch.voiceover_tracks:
            for track in orch.voiceover_tracks:
                p = Path(track.local_path)
                if p.exists():
                    inputs.append(track.local_path)
                    volumes.append(track.volume)

        if orch.background_music and Path(orch.background_music.local_path).exists():
            inputs.append(orch.background_music.local_path)
            volumes.append(orch.background_music.volume)

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
            subprocess.run([
                "ffmpeg", "-y",
                "-i", inputs[0],
                "-af", f"volume={volumes[0]:.4f},apad=whole_dur={total_duration}",
                "-t", str(total_duration),
                "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ], capture_output=True, check=True)
            return

        # Multiple inputs — build input list + amix
        cmd = ["ffmpeg", "-y"]
        for inp in inputs:
            cmd += ["-i", inp]
        # Build filter
        vol_parts = []
        vol_outs = []
        for i, (vol) in enumerate(volumes):
            vol_outs.append(f"av{i}")
            vol_parts.append(f"[{i}:a]volume={vol:.4f},apad=whole_dur={total_duration}[av{i}]")
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
            logger.warning("Audio mix failed: %s — using first track only", result.stderr[-300:])
            subprocess.run([
                "ffmpeg", "-y", "-i", inputs[0],
                "-t", str(total_duration), "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ], capture_output=True, check=True)

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
    # Helpers
    # ------------------------------------------------------------------

    def _build_scene_cmd(self, input_path: str, out_path: str, filter_chain: str, duration_sec: float) -> list[str]:
        """Build a direct subprocess argv for a per-scene encode."""
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",  # loop input so short clips fill the full duration
            "-i", input_path,
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

    def _detect_encoder(self) -> str:
        """Check if h264_videotoolbox is available; fall back to libx264."""
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True
        )
        if "h264_videotoolbox" in result.stdout:
            logger.info("Using h264_videotoolbox (M4 Pro hardware)")
            return "h264_videotoolbox"
        logger.info("Falling back to libx264")
        return "libx264"

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
