"""
Standalone script: generates TTS voiceover with edge-tts, then renders final video with ffmpeg.
No ScriptToReel module imports — stdlib + edge_tts + pydub + subprocess only.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

# Ensure /opt/homebrew/bin is on PATH so pydub/ffprobe can be found
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

import edge_tts
from pydub import AudioSegment

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("projects/the_rise_of_agentic_ai")
SCRIPT_JSON = PROJECT_DIR / "script.json"
VIDEO_ASSET_DIR = PROJECT_DIR / "assets" / "raw" / "video"
OUTPUT_DIR = PROJECT_DIR / "output"
VOICEOVER_WAV = PROJECT_DIR / "voiceover.wav"
FINAL_VIDEO = OUTPUT_DIR / "final_video.mp4"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

VOICE = "en-US-GuyNeural"
PAUSE_MS = 500
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ── Step 1: Parse script.json ──────────────────────────────────────────────────
def load_segments(script_path: Path) -> list[dict]:
    with open(script_path) as f:
        data = json.load(f)
    segments = []
    for seg in data.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            segments.append({"id": seg["id"], "text": text})
    return segments


# ── Step 2: Generate TTS MP3 per segment ──────────────────────────────────────
async def tts_segment(text: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(out_path)


async def generate_all_tts(segments: list[dict], tmp_dir: str) -> list[str]:
    mp3_paths = []
    for seg in segments:
        mp3_path = os.path.join(tmp_dir, f"seg_{seg['id']}.mp3")
        print(f"  TTS segment {seg['id']}: {seg['text'][:60]}...")
        await tts_segment(seg["text"], mp3_path)
        mp3_paths.append(mp3_path)
    return mp3_paths


# ── Step 3 & 4: Convert MP3 → WAV and concatenate with pauses ─────────────────
def build_voiceover(mp3_paths: list[str], output_wav: Path) -> float:
    silence = AudioSegment.silent(duration=PAUSE_MS)
    combined = AudioSegment.empty()
    for i, mp3_path in enumerate(mp3_paths):
        seg_audio = AudioSegment.from_mp3(mp3_path)
        if i > 0:
            combined += silence
        combined += seg_audio
        print(f"  Appended segment {i+1}/{len(mp3_paths)} ({len(seg_audio)/1000:.1f}s)")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_wav), format="wav")
    duration_sec = len(combined) / 1000.0
    print(f"  Voiceover saved → {output_wav} ({duration_sec:.1f}s)")
    return duration_sec


# ── Step 5: Find video assets ──────────────────────────────────────────────────
def find_videos(asset_dir: Path) -> list[Path]:
    if not asset_dir.exists():
        return []
    videos = sorted(
        p for p in asset_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos


# ── Step 6 & 7: Build ffmpeg command ──────────────────────────────────────────
def render_video(voiceover_wav: Path, video_files: list[Path],
                 output_mp4: Path, voiceover_duration: float) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    if video_files:
        _render_with_videos(voiceover_wav, video_files, output_mp4, voiceover_duration)
    else:
        _render_black_screen(voiceover_wav, output_mp4, voiceover_duration)


def _render_with_videos(voiceover_wav: Path, video_files: list[Path],
                        output_mp4: Path, voiceover_duration: float) -> None:
    """
    Scale each clip to 1920x1080, concatenate them in a loop until we have
    enough footage for the full voiceover, then mux with the voiceover audio.
    """
    print(f"  Using {len(video_files)} video file(s) as b-roll")

    # --- Step A: normalise every clip to 1920x1080 30fps in a temp dir --------
    with tempfile.TemporaryDirectory() as tmp:
        normalised: list[str] = []
        for idx, vf in enumerate(video_files):
            out = os.path.join(tmp, f"norm_{idx:03d}.mp4")
            cmd = [
                FFMPEG, "-y", "-i", str(vf),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                       "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-r", "30",
                "-c:v", "h264_videotoolbox",
                "-b:v", "4M",
                "-an",          # drop original audio — we supply voiceover
                out,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            normalised.append(out)
            print(f"    Normalised {vf.name}")

        # --- Step B: write concat list, looping clips until we fill duration --
        concat_list = os.path.join(tmp, "concat.txt")
        total = 0.0
        entries: list[str] = []
        while total < voiceover_duration:
            for p in normalised:
                if total >= voiceover_duration:
                    break
                # probe clip duration
                probe = subprocess.run(
                    ["/opt/homebrew/bin/ffprobe", "-v", "error",
                     "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", p],
                    capture_output=True, text=True
                )
                clip_dur = float(probe.stdout.strip() or "5")
                entries.append(f"file '{p}'")
                total += clip_dur

        with open(concat_list, "w") as f:
            f.write("\n".join(entries) + "\n")

        # --- Step C: concat + mux voiceover ------------------------------------
        concat_mp4 = os.path.join(tmp, "concat_raw.mp4")
        cmd_concat = [
            FFMPEG, "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c", "copy",
            concat_mp4,
        ]
        subprocess.run(cmd_concat, check=True, capture_output=True)

        cmd_mux = [
            FFMPEG, "-y",
            "-i", concat_mp4,
            "-i", str(voiceover_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(voiceover_duration),
            "-c:v", "h264_videotoolbox",
            "-b:v", "4M",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_mp4),
        ]
        subprocess.run(cmd_mux, check=True, capture_output=True)


def _render_black_screen(voiceover_wav: Path, output_mp4: Path,
                         voiceover_duration: float) -> None:
    print("  No video assets found — generating black-screen video")
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=30",
        "-i", str(voiceover_wav),
        "-map", "0:v", "-map", "1:a",
        "-t", str(voiceover_duration),
        "-c:v", "h264_videotoolbox",
        "-b:v", "1M",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ── Step 8: Report ─────────────────────────────────────────────────────────────
def report(output_mp4: Path) -> None:
    probe = subprocess.run(
        ["/opt/homebrew/bin/ffprobe", "-v", "quiet",
         "-print_format", "json", "-show_format", "-show_streams",
         str(output_mp4)],
        capture_output=True, text=True, check=True
    )
    info = json.loads(probe.stdout)
    fmt = info.get("format", {})
    streams = info.get("streams", [])

    print("\n── Final video report ──────────────────────────────────────")
    print(f"  File     : {output_mp4}")
    print(f"  Size     : {int(fmt.get('size', 0)) / 1_048_576:.1f} MB")
    print(f"  Duration : {float(fmt.get('duration', 0)):.2f}s")
    for s in streams:
        if s["codec_type"] == "video":
            print(f"  Video    : {s['codec_name']} {s.get('width')}x{s.get('height')} "
                  f"@ {s.get('r_frame_rate')} fps")
        elif s["codec_type"] == "audio":
            print(f"  Audio    : {s['codec_name']} {s.get('sample_rate')}Hz "
                  f"{s.get('channels')}ch")
    print("────────────────────────────────────────────────────────────")


# ── Main ───────────────────────────────────────────────────────────────────────
async def main() -> None:
    print("=== render_with_edge_tts ===\n")

    print("[1] Loading script.json …")
    segments = load_segments(SCRIPT_JSON)
    print(f"    {len(segments)} segments found\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("[2] Generating TTS audio …")
        mp3_paths = await generate_all_tts(segments, tmp_dir)

        print("\n[3] Building voiceover WAV …")
        voiceover_duration = build_voiceover(mp3_paths, VOICEOVER_WAV)

    print("\n[4] Scanning for video assets …")
    videos = find_videos(VIDEO_ASSET_DIR)
    print(f"    {len(videos)} video file(s) found")

    print("\n[5] Rendering final video …")
    render_video(VOICEOVER_WAV, videos, FINAL_VIDEO, voiceover_duration)
    print(f"    Done → {FINAL_VIDEO}")

    print("\n[6] Probing output …")
    report(FINAL_VIDEO)


if __name__ == "__main__":
    asyncio.run(main())
