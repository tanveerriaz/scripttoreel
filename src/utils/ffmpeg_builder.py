"""
FFmpeg command builder — fluent API for constructing complex ffmpeg invocations.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FFmpegCommand:
    """Fluent builder for ffmpeg commands."""

    def __init__(self):
        self._inputs: list[tuple[str, dict]] = []   # (path, kwargs)
        self._filter_complex: Optional[str] = None
        self._output_path: Optional[str] = None
        self._output_opts: dict = {}
        self._global_opts: dict = {"threads": 8, "y": True}

    def input(self, path: str, **opts) -> "FFmpegCommand":
        self._inputs.append((path, opts))
        return self

    def filter_complex(self, graph: str) -> "FFmpegCommand":
        self._filter_complex = graph
        return self

    def output(self, path: str, **opts) -> "FFmpegCommand":
        self._output_path = path
        self._output_opts = opts
        return self

    def global_opt(self, key: str, value) -> "FFmpegCommand":
        self._global_opts[key] = value
        return self

    @staticmethod
    def _ffmpeg_key(k: str) -> str:
        """Convert Python kwarg names to ffmpeg option names.

        Rules:
        - Keys matching ``^[a-z]{1,2}_[vads]$`` map the trailing ``_X`` → ``:X``
          so that e.g. ``c_v`` → ``c:v``, ``b_v`` → ``b:v``, ``c_a`` → ``c:a``.
        - All other keys are passed through unchanged (``pix_fmt``, ``preset``, …).
        """
        import re
        return re.sub(r"^([a-z]{1,2})_([vads])$", r"\1:\2", k)

    def build(self) -> list[str]:
        """Return the full argv list for ffmpeg."""
        cmd = ["ffmpeg"]

        # Global options
        for k, v in self._global_opts.items():
            if v is True:
                cmd.append(f"-{self._ffmpeg_key(k)}")
            elif v is not False and v is not None:
                cmd.extend([f"-{self._ffmpeg_key(k)}", str(v)])

        # Inputs
        for path, opts in self._inputs:
            for k, v in opts.items():
                fk = self._ffmpeg_key(k)
                if v is True:
                    cmd.append(f"-{fk}")
                elif v is not False and v is not None:
                    cmd.extend([f"-{fk}", str(v)])
            cmd.extend(["-i", path])

        # Filter complex
        if self._filter_complex:
            cmd.extend(["-filter_complex", self._filter_complex])

        # Output options
        for k, v in self._output_opts.items():
            fk = self._ffmpeg_key(k)
            if v is True:
                cmd.append(f"-{fk}")
            elif v is not False and v is not None:
                cmd.extend([f"-{fk}", str(v)])

        # Output path
        if self._output_path:
            cmd.append(self._output_path)

        return cmd

    def run(self, dry_run: bool = False) -> subprocess.CompletedProcess:
        argv = self.build()
        if dry_run:
            logger.info("DRY RUN: %s", " ".join(argv))
            return subprocess.CompletedProcess(argv, returncode=0)
        logger.info("Running: %s", " ".join(argv))
        return subprocess.run(argv, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Filter fragment builders
# ---------------------------------------------------------------------------

def build_scale_pad_filter(w: int, h: int, stream: str = "0:v") -> str:
    """Scale input to WxH, padding with black bars if needed (keep aspect)."""
    return (
        f"[{stream}]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[scaled]"
    )


def build_color_grade_filter(
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    in_stream: str = "scaled",
    out_stream: str = "graded",
) -> str:
    return (
        f"[{in_stream}]eq=brightness={brightness:.3f}:"
        f"contrast={contrast:.3f}:saturation={saturation:.3f}:gamma={gamma:.3f}"
        f"[{out_stream}]"
    )


def build_drawtext_filter(
    text: str,
    fontfile: str,
    fontsize: int,
    fontcolor: str,
    x: str,
    y: str,
    box: bool = True,
    boxcolor: str = "black@0.6",
    boxborderw: int = 10,
    in_stream: str = "graded",
    out_stream: str = "titled",
) -> str:
    safe_text = text.replace("'", "\\'").replace(":", "\\:")
    box_part = f":box=1:boxcolor={boxcolor}:boxborderw={boxborderw}" if box else ""
    return (
        f"[{in_stream}]drawtext=fontfile='{fontfile}':fontsize={fontsize}:"
        f"fontcolor={fontcolor}:x={x}:y={y}:text='{safe_text}'{box_part}"
        f"[{out_stream}]"
    )


def build_xfade_filter(
    in1: str, in2: str, out: str, duration: float = 0.8, offset: float = 0.0
) -> str:
    return f"[{in1}][{in2}]xfade=transition=dissolve:duration={duration:.3f}:offset={offset:.3f}[{out}]"


def build_concat_filter(inputs: list[str], out: str = "vout") -> str:
    """Simple concat (no transitions) for N inputs."""
    joined = "".join(f"[{i}]" for i in inputs)
    return f"{joined}concat=n={len(inputs)}:v=1:a=0[{out}]"


def build_loop_filter(duration_sec: float, in_stream: str = "0:v", out_stream: str = "looped") -> str:
    """Loop an image to create a video clip of the given duration."""
    frames = int(duration_sec * 30)
    return f"[{in_stream}]loop={frames}:size=1:start=0,fps=30[{out_stream}]"


def build_audio_amix_filter(
    inputs: list[str],
    volumes: list[float],
    out: str = "aout",
) -> str:
    """Mix N audio inputs with given volumes."""
    if len(inputs) == 1:
        return f"[{inputs[0]}]volume={volumes[0]:.4f}[{out}]"

    vol_filters = []
    vol_outs = []
    for i, (inp, vol) in enumerate(zip(inputs, volumes)):
        vout = f"av{i}"
        vol_filters.append(f"[{inp}]volume={vol:.4f}[{vout}]")
        vol_outs.append(vout)

    mix_in = "".join(f"[{v}]" for v in vol_outs)
    return "; ".join(vol_filters) + f"; {mix_in}amix=inputs={len(inputs)}:normalize=0[{out}]"
