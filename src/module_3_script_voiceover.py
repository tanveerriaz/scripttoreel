"""
Module 3 — Script Generation & Voiceover.

1. Calls local Ollama to generate a structured script (JSON)
2. Parses and validates the script with Pydantic
3. Generates per-segment WAV voiceover (Coqui TTS → macOS `say` fallback)
4. Concatenates segments with pauses → voiceover.wav
5. Writes script.json
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import requests
from pydub import AudioSegment

from src.project_manager import update_pipeline_status
from src.utils.config_loader import load_api_keys, load_ollama_prompts
from src.utils.json_schemas import (
    ModuleStatus,
    Script,
    ScriptSegment,
    SegmentTransitions,
    TextOverlay,
    TransitionType,
)

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 3
_PAUSE_BETWEEN_SEGMENTS_MS = 500


class OllamaNotAvailableError(Exception):
    """Raised when Ollama is unreachable or returns an error."""


class ScriptModule:
    def __init__(self, project_dir: Path, api_keys: Optional[dict] = None):
        self.project_dir = Path(project_dir)
        self.api_keys = api_keys if api_keys is not None else load_api_keys()
        self.ollama_url = self.api_keys.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = self.api_keys.get("OLLAMA_MODEL", "llama3.2")
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _audio_dir(self) -> Path:
        return self.project_dir / "assets" / "audio"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Script:
        meta = self._load_project_meta()
        topic = meta.get("topic", "")
        duration_min = float(meta.get("duration_min", 2))

        logger.info("Module 3: generating script for %r (%.0f min)", topic, duration_min)

        script = self.generate_script_ollama(topic, duration_min)
        script = self.generate_all_voiceovers(script)
        combined_path = self._audio_dir / "voiceover.wav"
        self.concatenate_voiceovers(script, combined_path)
        script = script.model_copy(update={"total_voiceover_path": str(combined_path)})
        self.save_script(script)
        self._update_status(ModuleStatus.COMPLETE)
        return script

    # ------------------------------------------------------------------
    # Story 3.1 — Ollama script generation
    # ------------------------------------------------------------------

    def generate_script_ollama(self, topic: str, duration_min: float) -> Script:
        """POST to Ollama, retry up to 3x on bad JSON, return validated Script."""
        prompts = load_ollama_prompts()
        system_prompt = prompts["script_generation"]["system"]
        user_template = prompts["script_generation"]["user_template"]
        duration_sec = int(duration_min * 60)
        user_prompt = user_template.format(
            topic=topic,
            duration=duration_min,
            duration_sec=duration_sec,
        )

        payload = {
            "model": self.ollama_model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": 0.7},
        }

        last_error: Exception = Exception("Unknown error")
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                raw_text = resp.json().get("response", "")
                return self.parse_script_json(raw_text)
            except requests.ConnectionError as e:
                raise OllamaNotAvailableError(
                    f"Cannot connect to Ollama at {self.ollama_url}. "
                    "Is it running? Start with: ollama serve"
                ) from e
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning("Attempt %d: bad JSON from Ollama: %s", attempt + 1, e)
                continue

        raise last_error

    def parse_script_json(self, raw_text: str) -> Script:
        """Extract JSON from LLM response, validate with Pydantic."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        # Find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in LLM response: {raw_text[:200]!r}")
        json_str = text[start : end + 1]
        data = json.loads(json_str)

        # Normalise segment transitions (may be nested dicts with "in"/"out" keys)
        for seg in data.get("segments", []):
            tr = seg.get("transitions", {})
            if isinstance(tr, dict):
                seg["transitions"] = {
                    "in_transition": _normalize_transition(tr.get("in", "dissolve")),
                    "out_transition": _normalize_transition(tr.get("out", "dissolve")),
                }
            overlay = seg.get("text_overlay", {})
            if isinstance(overlay, dict) and "start_time" not in overlay:
                overlay.setdefault("start_time", 0.0)

        return Script(**data)

    # ------------------------------------------------------------------
    # Story 3.3 — TTS voiceover
    # ------------------------------------------------------------------

    # Path to piper model — falls back to macOS say if not found
    _PIPER_MODEL = Path(__file__).parent.parent / "models" / "piper" / "en_US-lessac-high.onnx"

    def generate_voiceover_segment(self, text: str, segment_id: int) -> Path:
        """Generate WAV for one segment. Tries piper → macOS say."""
        out = self._audio_dir / f"voiceover_{segment_id}.wav"
        if out.exists() and out.stat().st_size > 1000:
            return out
        try:
            self._piper_tts(text, out)
        except Exception as e:
            logger.warning("Piper TTS failed (segment %d): %s — using macOS say", segment_id, e)
            self._macos_say_fallback(text, out)
        return out

    def _piper_tts(self, text: str, out_path: Path) -> None:
        """Generate audio with piper-tts (high-quality neural TTS)."""
        model = self._PIPER_MODEL
        if not model.exists():
            raise FileNotFoundError(f"Piper model not found: {model}")
        piper_bin = shutil.which("piper")
        if not piper_bin:
            raise RuntimeError("`piper` command not found")
        result = subprocess.run(
            [piper_bin, "-m", str(model), "-f", str(out_path)],
            input=text,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"piper failed: {result.stderr[:200]}")

    def _macos_say_fallback(self, text: str, out_path: Path) -> None:
        """Use macOS built-in `say` to produce an AIFF then convert to WAV."""
        if not shutil.which("say"):
            raise RuntimeError("`say` command not found — not on macOS?")

        aiff = out_path.with_suffix(".aiff")
        subprocess.run(
            ["say", "-v", "Samantha", "-r", "150", "-o", str(aiff), text],
            check=True,
            capture_output=True,
        )
        # Convert AIFF → WAV via ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(aiff), "-ar", "22050", "-ac", "1", str(out_path)],
            check=True,
            capture_output=True,
        )
        aiff.unlink(missing_ok=True)

    def generate_all_voiceovers(self, script: Script) -> Script:
        updated_segments = []
        for seg in script.segments:
            wav = self.generate_voiceover_segment(seg.text, seg.id)
            dur = _wav_duration(wav)
            updated_segments.append(
                seg.model_copy(update={"voiceover_path": str(wav), "voiceover_duration_sec": dur})
            )
        return script.model_copy(update={"segments": updated_segments})

    # ------------------------------------------------------------------
    # Story 3.4 — Concatenation + script.json
    # ------------------------------------------------------------------

    def concatenate_voiceovers(self, script: Script, out_path: Path) -> Path:
        wav_paths = [
            Path(seg.voiceover_path)
            for seg in script.segments
            if seg.voiceover_path and Path(seg.voiceover_path).exists()
        ]
        self._concatenate_wavs(wav_paths, out_path, pause_ms=_PAUSE_BETWEEN_SEGMENTS_MS)
        return out_path

    def _concatenate_wavs(self, paths: list[Path], out: Path, pause_ms: int = 500) -> None:
        if not paths:
            # Write a silent 1s WAV
            silence = AudioSegment.silent(duration=1000)
            silence.export(str(out), format="wav")
            return
        combined = AudioSegment.empty()
        pause = AudioSegment.silent(duration=pause_ms)
        for i, p in enumerate(paths):
            seg = AudioSegment.from_file(str(p))
            combined += seg
            if i < len(paths) - 1:
                combined += pause
        combined.export(str(out), format="wav")

    def save_script(self, script: Script) -> None:
        out = self.project_dir / "script.json"
        out.write_text(json.dumps(script.model_dump(), indent=2, default=str))
        logger.info("Saved script.json (%d segments)", len(script.segments))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_project_meta(self) -> dict:
        p = self.project_dir / "project.json"
        if not p.exists():
            raise FileNotFoundError(f"project.json not found in {self.project_dir}")
        return json.loads(p.read_text())

    def _update_status(self, status: ModuleStatus) -> None:
        meta = self._load_project_meta()
        project_id = meta.get("project_id")
        if not project_id:
            return
        try:
            update_pipeline_status(
                project_id, "module_3_script", status,
                projects_root=self.project_dir.parent,
            )
        except Exception as e:
            logger.warning("Could not update pipeline status: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wav_duration(path: Path) -> float:
    try:
        seg = AudioSegment.from_file(str(path))
        return seg.duration_seconds
    except Exception:
        return 0.0


def _normalize_transition(value: str) -> str:
    """Map YAML transition names to TransitionType enum values."""
    mapping = {
        "fade_in": "fade_in",
        "fade_out": "fade_out",
        "dissolve": "dissolve",
        "crossfade": "crossfade",
        "cut": "cut",
        "none": "none",
    }
    return mapping.get(str(value).lower(), "dissolve")
