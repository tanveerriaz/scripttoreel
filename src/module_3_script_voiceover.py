"""
Module 3 — Script Generation & Voiceover.

1. Calls OpenRouter via `call_llm()` to generate a structured script (JSON)
2. Parses and validates the script with Pydantic
3. Generates per-segment WAV voiceover (Kokoro-ONNX → edge-tts → piper → macOS `say`)
4. Concatenates segments with pauses → voiceover.wav
5. Writes script.json
"""
from __future__ import annotations

import asyncio
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

# Available edge-tts neural voices for casting
AVAILABLE_VOICES = [
    "en-US-GuyNeural",          # Deep, documentary narrator (default)
    "en-US-ChristopherNeural",  # Authoritative male
    "en-US-JennyNeural",        # Friendly female
    "en-US-AriaNeural",         # Expressive female
    "en-GB-RyanNeural",         # British male
    "en-GB-SoniaNeural",        # British female
]
DEFAULT_NARRATOR_VOICE = "en-US-AriaNeural"

# ---------------------------------------------------------------------------
# Kokoro-ONNX TTS (local, free, natural voice)
# ---------------------------------------------------------------------------
_KOKORO_MODEL  = Path(__file__).parent.parent / "models" / "kokoro" / "kokoro-v1.0.int8.onnx"
_KOKORO_VOICES = Path(__file__).parent.parent / "models" / "kokoro" / "voices-v1.0.bin"

# Maps edge-tts voice names → Kokoro voice IDs
_KOKORO_VOICE_MAP: dict[str, str] = {
    "en-US-GuyNeural":         "am_adam",    # American male — documentary narrator
    "en-US-ChristopherNeural": "am_michael", # American male — authoritative
    "en-US-JennyNeural":       "af_sarah",   # American female
    "en-US-AriaNeural":        "af_bella",   # American female — expressive
    "en-GB-RyanNeural":        "bm_george",  # British male
    "en-GB-SoniaNeural":       "bf_emma",    # British female
}
_KOKORO_DEFAULT_VOICE = "am_adam"

from src.project_manager import update_pipeline_status
from src.utils.config_loader import load_api_keys, load_script_prompts
from src.utils.llm_client import call_llm
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


class ScriptModule:
    def __init__(
        self,
        project_dir: Path,
        api_keys: Optional[dict] = None,
        skip_director: bool = False,
    ):
        self.project_dir = Path(project_dir)
        self.api_keys = api_keys if api_keys is not None else load_api_keys()
        self.skip_director = skip_director
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

        # Load production plan settings if available
        plan = self._load_production_plan()

        script = self.generate_script(topic, duration_min, plan=plan)

        # Duration pacing check — revise script if estimated VO length is off target
        script = self._enforce_duration_pacing(script, duration_min, plan=plan)

        # Enhance the opening hook
        script = self._enhance_hook(script, plan)

        # Run ScriptDirector review — skip for short videos (≤1 min) to save ~40-60s
        if not self.skip_director and duration_min > 1.0:
            script = self._run_script_director(script)

        # Clear cached voiceover files so current TTS engine is always used
        for old_wav in self._audio_dir.glob("voiceover_*.wav"):
            old_wav.unlink(missing_ok=True)
        combined = self._audio_dir / "voiceover.wav"
        combined.unlink(missing_ok=True)

        script = self.generate_all_voiceovers(script)
        combined_path = self._audio_dir / "voiceover.wav"
        self.concatenate_voiceovers(script, combined_path)

        # Log actual vs target duration
        total_vo_dur = sum(
            s.voiceover_duration_sec or 0.0 for s in script.segments
        )
        target_sec = duration_min * 60
        logger.info(
            "Target duration: %.0fs, Actual TTS: %.1fs (%.0f%%)",
            target_sec, total_vo_dur, (total_vo_dur / target_sec * 100) if target_sec > 0 else 0,
        )
        script = script.model_copy(update={
            "total_voiceover_path": str(combined_path),
            "total_voiceover_duration_sec": total_vo_dur,
        })
        self.save_script(script)
        self._update_status(ModuleStatus.COMPLETE)
        return script

    def _enhance_hook(self, script: Script, plan) -> Script:
        """Replace or enhance the intro segment's opening with a generated hook.

        If the intro already starts with a strong hook (ends with '?' or contains
        a number), it is kept as-is. Otherwise the HookEngine selects the best
        hook pattern for the topic/tone and prepends it to the intro text.
        """
        try:
            intro_idx = next(
                (i for i, s in enumerate(script.segments) if s.type.value == "intro"),
                None,
            )
            if intro_idx is None:
                return script

            intro = script.segments[intro_idx]
            first_sentence = (intro.text or "").split(".")[0].strip()

            # Skip if already looks like a strong hook
            if (
                first_sentence.endswith("?")
                or any(ch.isdigit() for ch in first_sentence[:60])
            ):
                logger.info("Module 3: intro already has a strong hook — skipping enhance")
                return script

            tone = getattr(plan, "tone", "educational") if plan else "educational"
            audience = getattr(plan, "target_audience", "general") if plan else "general"

            from src.hook_engine import HookEngine
            engine = HookEngine(self.api_keys)
            hook = engine.select_best_hook(script.topic, str(tone), str(audience), plan=plan)
            hook_text = hook.get("text", "")

            if not hook_text or hook_text in intro.text:
                return script

            new_text = hook_text + " " + intro.text
            updated_segments = list(script.segments)
            updated_segments[intro_idx] = intro.model_copy(update={"text": new_text})
            logger.info(
                "Module 3: enhanced hook (pattern=%s) prepended to intro segment",
                hook.get("pattern", "?"),
            )
            return script.model_copy(update={"segments": updated_segments})

        except Exception as e:
            logger.warning("Hook enhancement failed: %s — using original intro", e)
            return script

    def _enforce_duration_pacing(self, script: Script, duration_min: float, plan=None) -> Script:
        """Check estimated voiceover duration vs target; ask LLM to revise if off.

        Rule of thumb: ~2.0 words/second for natural TTS narration.
        Conservative estimate ensures scripts come in under target, not over.
        If estimated is >110% or <80% of target, request revision passes.
        """
        _WPS = 2.0  # conservative: Kokoro/edge-tts measured at 2.8-3.2 WPS
        target_sec = duration_min * 60
        total_words = sum(len((s.text or "").split()) for s in script.segments)
        estimated_sec = total_words / _WPS

        logger.info(
            "Duration check: target=%.0fs, estimated=%.1fs (%d words @ %.1f wps)",
            target_sec, estimated_sec, total_words, _WPS,
        )

        if 0.8 * target_sec <= estimated_sec <= 1.1 * target_sec:
            return script  # within acceptable range

        lo, hi = 0.8 * target_sec, 1.1 * target_sec
        for attempt in range(3):
            direction = "too long" if estimated_sec > target_sec else "too short"
            if estimated_sec > hi:
                target_words_hint = int(target_sec / _WPS)
                instruction = (
                    f"This script is ~{estimated_sec:.0f}s but the target is {target_sec:.0f}s. "
                    f"Tighten to {target_words_hint} words or fewer — cut filler, combine ideas, keep the hook. "
                    "Return the complete revised script as JSON."
                )
            else:
                instruction = (
                    f"This script is ~{estimated_sec:.0f}s but the target is {target_sec:.0f}s. "
                    "Expand with more detail, examples, or transitional narration. "
                    "Return the complete revised script as JSON."
                )

            logger.info("Script is %s (attempt %d/3) — requesting LLM revision", direction, attempt + 1)
            try:
                script_json = json.dumps(script.model_dump(), indent=2, default=str)
                system_prompt = "You are a professional video scriptwriter. Revise the script duration. Return ONLY valid JSON."
                user_prompt = f"{instruction}\n\n{script_json}"
                raw = call_llm(system_prompt, user_prompt, self.api_keys, max_tokens=6000)
                revised = self.parse_script_json(raw)
                revised_words = sum(len((s.text or "").split()) for s in revised.segments)
                revised_estimated = revised_words / _WPS
                logger.info("Revision %d: %d words → ~%.1fs (target %.0fs)", attempt + 1, revised_words, revised_estimated, target_sec)
                script = revised
                estimated_sec = revised_estimated
                if lo <= estimated_sec <= hi:
                    logger.info("Duration within target range — done")
                    return script
            except Exception as e:
                logger.warning("Duration revision %d failed: %s", attempt + 1, e)
                break

        logger.warning("Could not reach target duration after revisions — using best attempt (%.1fs)", estimated_sec)
        return script

    def _run_script_director(self, draft: Script) -> Script:
        """Save script_draft.json, run ScriptDirector review, return revised script."""
        # Persist the raw draft so the user can compare before/after
        draft_path = self.project_dir / "script_draft.json"
        draft_path.write_text(
            json.dumps(draft.model_dump(), indent=2, default=str)
        )
        logger.info("Saved script_draft.json — running ScriptDirector review")

        try:
            from src.ai_director import ScriptDirector
            director = ScriptDirector(api_keys=self.api_keys)
            revised = director.review(draft)
            return revised
        except Exception as e:
            logger.warning("ScriptDirector review failed: %s — using original script", e)
            return draft

    # ------------------------------------------------------------------
    # Story 3.1 — Script generation via OpenRouter
    # ------------------------------------------------------------------

    def generate_script(
        self,
        topic: str,
        duration_min: float,
        plan=None,  # Optional[ProductionPlan] — avoid circular import
    ) -> Script:
        """Generate script via OpenRouter (call_llm)."""
        prompts = load_script_prompts()
        system_prompt = prompts["script_generation"]["system"]
        user_template = prompts["script_generation"]["user_template"]
        duration_sec = int(duration_min * 60)
        target_words = int(duration_sec / 2.0)
        user_prompt = user_template.format(
            topic=topic,
            duration=duration_min,
            duration_sec=duration_sec,
            target_words=target_words,
        )

        # Augment prompt with production plan settings
        if plan is not None:
            plan_hints: list[str] = []
            if plan.tone:
                plan_hints.append(f"Tone: {plan.tone}")
            if plan.target_audience:
                plan_hints.append(f"Target audience: {plan.target_audience}")
            if plan.cultural_context:
                plan_hints.append(f"Cultural context: {plan.cultural_context}")
            if plan.avoid_list:
                plan_hints.append(f"AVOID these topics/visuals: {', '.join(plan.avoid_list)}")
            if plan.script_guidance:
                plan_hints.append(f"Script guidance: {plan.script_guidance}")
            if plan_hints:
                user_prompt += "\n\nADDITIONAL PRODUCTION NOTES:\n" + "\n".join(
                    f"- {h}" for h in plan_hints
                )

        last_error: Exception = Exception("Unknown error")
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                raw_text = call_llm(system_prompt, user_prompt, self.api_keys)
                return self.parse_script_json(raw_text)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning("Attempt %d: bad JSON from LLM: %s", attempt + 1, e)
                continue

        raise last_error

    def parse_script_json(self, raw_text: str) -> Script:
        """Extract JSON from LLM response, validate with Pydantic."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()

        # Fast path: try direct parse first (LLM returned clean JSON)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: find the first { ... } block (handles leading/trailing prose)
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON object found in LLM response: {raw_text[:200]!r}")
            data = json.loads(text[start : end + 1])

        # Normalise top-level enum fields (LLM may output "dark|mysterious" style)
        _VALID_MOODS = {"dark", "mysterious", "uplifting", "dramatic", "educational",
                        "horror", "neutral", "suspenseful", "melancholic"}
        mood_raw = str(data.get("mood", "neutral"))
        data["mood"] = next(
            (m for m in (p.strip() for p in mood_raw.replace("|", ",").split(",")) if m in _VALID_MOODS),
            "neutral"
        )

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

    def generate_voiceover_segment(
        self, text: str, segment_id: int, voice: Optional[str] = None
    ) -> Path:
        """Generate WAV for one segment.

        Priority: Kokoro-ONNX (local neural) → edge-tts → piper → macOS say.
        Kokoro gives the most natural prosody when model files are present.
        Each step falls back gracefully if unavailable.
        """
        out = self._audio_dir / f"voiceover_{segment_id}.wav"
        if out.exists() and out.stat().st_size > 1000:
            return out

        # Skip TTS for empty or whitespace-only text — write 500ms silence instead
        if not text or not text.strip():
            logger.warning("Segment %d has empty text — writing silence", segment_id)
            AudioSegment.silent(duration=500).export(str(out), format="wav")
            return out

        selected_voice = voice or DEFAULT_NARRATOR_VOICE

        # 1. Kokoro-ONNX (local, free, most natural)
        try:
            logger.info("Segment %d: Using Kokoro-ONNX TTS", segment_id)
            self._kokoro_tts(text, out, selected_voice)
            logger.info("Segment %d: Kokoro-ONNX succeeded", segment_id)
            return out
        except Exception as e:
            logger.warning("Kokoro TTS failed (segment %d): %s — trying edge-tts", segment_id, e)

        # 2. edge-tts (neural cloud voices)
        try:
            logger.info("Segment %d: Using edge-tts (%s)", segment_id, selected_voice)
            self._edge_tts(text, out, selected_voice)
            logger.info("Segment %d: edge-tts succeeded", segment_id)
            return out
        except Exception as e:
            logger.warning("edge-tts failed (segment %d): %s — trying piper", segment_id, e)

        # 3. piper (local neural fallback)
        try:
            logger.info("Segment %d: Using piper TTS", segment_id)
            self._piper_tts(text, out)
            logger.info("Segment %d: piper succeeded", segment_id)
            return out
        except Exception as e:
            logger.warning("Piper TTS failed (segment %d): %s — using macOS say", segment_id, e)

        # 4. macOS say (last resort)
        logger.info("Segment %d: Using macOS say (fallback)", segment_id)
        self._macos_say_fallback(text, out)
        return out

    def _kokoro_tts(
        self, text: str, out_path: Path, voice: str = DEFAULT_NARRATOR_VOICE
    ) -> None:
        """Generate audio with Kokoro-ONNX (local, free, natural voice).

        Requires model files in models/kokoro/:
            kokoro-v0_19.onnx  (~90 MB)
            voices.bin         (~5 MB)
        Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/latest

        Raises FileNotFoundError if model files missing → triggers next fallback.
        Raises RuntimeError  if kokoro-onnx not installed → triggers next fallback.
        """
        if not _KOKORO_MODEL.exists() or not _KOKORO_VOICES.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found at {_KOKORO_MODEL.parent}. "
                "Download kokoro-v1.0.int8.onnx and voices-v1.0.bin from "
                "https://github.com/thewh1teagle/kokoro-onnx/releases/latest"
            )

        try:
            from kokoro_onnx import Kokoro  # noqa: PLC0415
            import soundfile as sf          # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx not installed: pip install kokoro-onnx soundfile"
            ) from exc

        kokoro_voice = _KOKORO_VOICE_MAP.get(voice, _KOKORO_DEFAULT_VOICE)
        k = Kokoro(str(_KOKORO_MODEL), str(_KOKORO_VOICES))
        try:
            samples, sample_rate = k.create(text, voice=kokoro_voice, speed=1.0)
        except AttributeError as exc:
            raise RuntimeError(
                f"Kokoro espeak-ng version mismatch ({exc}). "
                "Fix: pip install --upgrade kokoro-onnx  or reinstall espeak-ng."
            ) from exc

        # Write raw samples to a temp file then re-encode to 22050 Hz mono
        # (consistent with edge-tts / piper output format)
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        try:
            sf.write(str(tmp), samples, sample_rate)
            subprocess.run(
                [
                    self._ffmpeg_bin(),
                    "-y", "-i", str(tmp),
                    "-ar", "44100", "-ac", "1",
                    # highpass removes mic rumble; presence boost at 3kHz adds speech clarity
                    # No lowpass — keeping full bandwidth makes voice sound natural, not telephonic
                    "-af", "highpass=f=80,equalizer=f=3000:width_type=o:width=2:g=2",
                    str(out_path),
                ],
                check=True,
                capture_output=True,
            )
        finally:
            tmp.unlink(missing_ok=True)

    def _edge_tts(self, text: str, out_path: Path, voice: str = DEFAULT_NARRATOR_VOICE) -> None:
        """Generate audio with Microsoft Edge TTS (neural voices via edge-tts package)."""
        try:
            import edge_tts  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("edge-tts not installed: pip install edge-tts") from exc

        async def _generate() -> None:
            communicate = edge_tts.Communicate(text, voice, rate="+10%")
            mp3_path = out_path.with_suffix(".mp3")
            mp3_path.parent.mkdir(parents=True, exist_ok=True)
            await communicate.save(str(mp3_path))
            if not mp3_path.exists() or mp3_path.stat().st_size < 100:
                raise RuntimeError(f"edge-tts produced no output at {mp3_path}")
            # Convert MP3 → WAV (44100 Hz mono)
            # No lowpass: keeping full bandwidth prevents the telephone/robotic effect
            # Presence boost at 3kHz adds speech clarity and warmth
            subprocess.run(
                [
                    self._ffmpeg_bin(),
                    "-y", "-i", str(mp3_path),
                    "-ar", "44100", "-ac", "1",
                    "-af", "highpass=f=80,equalizer=f=3000:width_type=o:width=2:g=2",
                    str(out_path),
                ],
                check=True,
                capture_output=True,
            )
            mp3_path.unlink(missing_ok=True)

        try:
            asyncio.run(_generate())
        except RuntimeError as exc:
            # Already inside a running event loop (e.g. Jupyter) — use a thread
            if "cannot run nested" in str(exc).lower() or "event loop is already running" in str(exc).lower():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(asyncio.run, _generate()).result()
            else:
                raise

    @staticmethod
    def _ffmpeg_bin() -> str:
        """Return path to ffmpeg, checking homebrew if not in PATH."""
        found = shutil.which("ffmpeg")
        if found:
            return found
        for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if Path(candidate).exists():
                return candidate
        return "ffmpeg"  # let subprocess raise a clear error

    def _piper_tts(self, text: str, out_path: Path) -> None:
        """Generate audio with piper-tts (high-quality neural TTS).

        Tuning notes:
          --length-scale 1.1   Slightly slower → more natural pacing
          --noise-scale  0.667 Expressiveness (default); increase for more variation
          --noise-w-scale 0.8  Phoneme duration variation; keeps rhythm natural
        """
        model = self._PIPER_MODEL
        if not model.exists():
            raise FileNotFoundError(f"Piper model not found: {model}")
        piper_bin = shutil.which("piper")
        if not piper_bin:
            raise RuntimeError("`piper` command not found")
        result = subprocess.run(
            [
                piper_bin,
                "-m", str(model),
                "-f", str(out_path),
                "--length-scale", "1.1",    # 10% slower → less rushed
                "--noise-scale", "0.667",   # natural expressiveness
                "--noise-w-scale", "0.8",   # stable phoneme timing
                "--sentence-silence", "0.3", # 300ms pause between sentences
            ],
            input=text,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"piper failed: {result.stderr[:200]}")

    def _macos_say_fallback(
        self,
        text: str,
        out_path: Path,
        preferred_voice: Optional[str] = None,
    ) -> None:
        """Use macOS built-in `say` to produce an AIFF then convert to WAV."""
        if not shutil.which("say"):
            raise RuntimeError("`say` command not found — not on macOS?")

        # Use Samantha (female) as default, matching the pipeline's AriaNeural preference.
        # Rate 175 wpm sounds natural for documentary-style narration.
        aiff = out_path.with_suffix(".aiff")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        available_voices_result = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True
        )
        available = available_voices_result.stdout

        voice = "Samantha"
        candidates = []
        if preferred_voice:
            candidates.append(preferred_voice)
        candidates.extend(("Samantha", "Daniel"))
        for candidate in candidates:
            if candidate in available:
                voice = candidate
                break

        # Write text to temp file to avoid CLI arg length limits with long narration
        txt_file = out_path.with_suffix(".txt")
        txt_file.write_text(text, encoding="utf-8")
        try:
            subprocess.run(
                ["say", "-v", voice, "-r", "175", "-o", str(aiff), "-f", str(txt_file)],
                check=True,
                capture_output=True,
            )
        finally:
            txt_file.unlink(missing_ok=True)
        # Convert AIFF → WAV, upsample to 22050 Hz mono for consistency with piper
        subprocess.run(
            [self._ffmpeg_bin(), "-y", "-i", str(aiff),
             "-ar", "44100", "-ac", "1",
             "-af", "highpass=f=80,equalizer=f=3000:width_type=o:width=2:g=2",
             str(out_path)],
            check=True,
            capture_output=True,
        )
        aiff.unlink(missing_ok=True)

    def generate_all_voiceovers(
        self, script: Script, narrator_voice: Optional[str] = None
    ) -> Script:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        narrator = script.narrator_voice or DEFAULT_NARRATOR_VOICE

        def _gen_one(seg):
            voice = narrator
            if seg.voice and seg.voice in AVAILABLE_VOICES:
                voice = seg.voice
            wav = self.generate_voiceover_segment(seg.text, seg.id, voice=voice)
            dur = _wav_duration(wav)
            return seg.model_copy(update={"voiceover_path": str(wav), "voiceover_duration_sec": dur})

        # Parallel TTS: segments are independent, subprocess calls release GIL
        try:
            with ThreadPoolExecutor(max_workers=min(len(script.segments), 4)) as pool:
                updated_segments = list(pool.map(_gen_one, script.segments))
            return script.model_copy(update={"segments": updated_segments})
        except Exception as e:
            logger.warning("Parallel TTS failed (%s) — falling back to sequential", e)

        # Sequential fallback
        updated_segments = []
        for seg in script.segments:
            voice = narrator
            if seg.voice and seg.voice in AVAILABLE_VOICES:
                voice = seg.voice
            wav = self.generate_voiceover_segment(seg.text, seg.id, voice=voice)
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

    def _load_production_plan(self):
        """Load production_plan.json if it exists. Returns None if not present."""
        plan_path = self.project_dir / "production_plan.json"
        if not plan_path.exists():
            return None
        try:
            from src.utils.json_schemas import ProductionPlan
            return ProductionPlan(**json.loads(plan_path.read_text()))
        except Exception as e:
            logger.warning("Could not load production_plan.json: %s", e)
            return None

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
