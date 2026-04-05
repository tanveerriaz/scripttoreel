"""
Pydantic models for all ScriptToReel JSON contracts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AssetType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    SFX = "sfx"


class AssetRole(str, Enum):
    B_ROLL = "b_roll"
    BACKGROUND = "background"
    INTRO = "intro"
    OUTRO = "outro"
    OVERLAY = "overlay"
    MUSIC = "music"
    SFX = "sfx"
    VOICEOVER = "voiceover"


class AssetSource(str, Enum):
    PEXELS = "pexels"
    PIXABAY = "pixabay"
    UNSPLASH = "unsplash"
    FREESOUND = "freesound"
    GENERATED = "generated"
    LOCAL = "local"


class Mood(str, Enum):
    DARK = "dark"
    MYSTERIOUS = "mysterious"
    UPLIFTING = "uplifting"
    DRAMATIC = "dramatic"
    EDUCATIONAL = "educational"
    HORROR = "horror"
    NEUTRAL = "neutral"
    SUSPENSEFUL = "suspenseful"
    MELANCHOLIC = "melancholic"


class TransitionType(str, Enum):
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    DISSOLVE = "dissolve"
    CROSSFADE = "crossfade"
    CUT = "cut"
    NONE = "none"


class SegmentType(str, Enum):
    INTRO = "intro"
    NARRATION = "narration"
    TRANSITION = "transition"
    OUTRO = "outro"


class ColorGrade(str, Enum):
    DARK_MYSTERIOUS = "dark_mysterious"
    CINEMATIC_WARM = "cinematic_warm"
    DOCUMENTARY = "documentary"
    DRAMATIC = "dramatic"
    UPLIFTING = "uplifting"


class ToneStyle(str, Enum):
    CINEMATIC = "cinematic"
    DOCUMENTARY = "documentary"
    DRAMATIC = "dramatic"
    UPLIFTING = "uplifting"
    CASUAL = "casual"


class VisualStyleChoice(str, Enum):
    DARK_MYSTERIOUS = "dark_mysterious"
    CINEMATIC_WARM = "cinematic_warm"
    DOCUMENTARY = "documentary"
    DRAMATIC = "dramatic"
    BRIGHT_MODERN = "bright_modern"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class LicensingInfo(BaseModel):
    license_type: str = "royalty_free"
    attribution_required: bool = False
    commercial_use: bool = True
    source_name: str = ""
    license_url: Optional[str] = None


class AudioMetadata(BaseModel):
    duration_sec: float = 0.0
    sample_rate: int = 44100
    channels: int = 2
    bpm: Optional[float] = None
    key: Optional[str] = None
    loudness_lufs: Optional[float] = None
    codec: Optional[str] = None
    bitrate_kbps: Optional[int] = None


class VideoMetadata(BaseModel):
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    bitrate_kbps: Optional[int] = None
    has_audio: bool = False
    frame_count: int = 0


class ImageMetadata(BaseModel):
    width: int = 0
    height: int = 0
    format: str = ""
    mode: str = ""
    file_size_bytes: int = 0


class VisualDNA(BaseModel):
    """Visual fingerprint of an asset."""
    dominant_colors: list[str] = Field(default_factory=list)
    color_palette: list[str] = Field(default_factory=list)
    dominant_mood: Mood = Mood.NEUTRAL
    brightness: float = 0.5          # 0-1
    contrast: float = 0.5            # 0-1
    saturation: float = 0.5          # 0-1
    visual_tags: list[str] = Field(default_factory=list)
    scene_type: Optional[str] = None  # indoor|outdoor|abstract
    time_of_day: Optional[str] = None # day|night|dusk|dawn


# ---------------------------------------------------------------------------
# Core Asset Model
# ---------------------------------------------------------------------------

class Asset(BaseModel):
    id: str
    type: AssetType
    role: AssetRole = AssetRole.B_ROLL
    source: AssetSource = AssetSource.LOCAL
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    local_path: Optional[str] = None
    filename: Optional[str] = None

    # Licensing
    licensing: LicensingInfo = Field(default_factory=LicensingInfo)
    attribution: Optional[str] = None

    # Duration & geometry
    duration_sec: float = 0.0
    resolution: Optional[str] = None   # "1920x1080"
    aspect_ratio: Optional[str] = None  # "16:9"
    fps: Optional[float] = None

    # Visual properties
    color_palette: list[str] = Field(default_factory=list)
    dominant_mood: Mood = Mood.NEUTRAL
    visual_tags: list[str] = Field(default_factory=list)
    visual_dna: Optional[VisualDNA] = None

    # Audio metadata (populated for audio/video assets)
    audio_metadata: Optional[AudioMetadata] = None
    video_metadata: Optional[VideoMetadata] = None
    image_metadata: Optional[ImageMetadata] = None

    # Quality
    quality_score: float = Field(default=0.0, ge=0.0, le=10.0)
    ready_for_use: bool = False

    # Search context
    search_query: Optional[str] = None
    search_tags: list[str] = Field(default_factory=list)

    @field_validator("quality_score")
    @classmethod
    def round_quality(cls, v: float) -> float:
        return round(v, 2)


# ---------------------------------------------------------------------------
# Script & Segment Models
# ---------------------------------------------------------------------------

class TextOverlay(BaseModel):
    enabled: bool = False
    text: str = ""
    position: str = "bottom_third"
    style: str = "lower_third"
    start_time: float = 0.0
    end_time: Optional[float] = None
    fade_in: float = 0.3
    fade_out: float = 0.3


class SegmentTransitions(BaseModel):
    in_transition: TransitionType = TransitionType.FADE_IN
    out_transition: TransitionType = TransitionType.DISSOLVE


class ScriptSegment(BaseModel):
    id: int
    type: SegmentType = SegmentType.NARRATION
    text: str
    duration_sec: float
    visual_cues: list[str] = Field(default_factory=list)
    mood_tags: list[str] = Field(default_factory=list)
    b_roll_keywords: list[str] = Field(default_factory=list)
    sfx_cues: list[str] = Field(default_factory=list)
    music_cues: list[str] = Field(default_factory=list)
    transitions: SegmentTransitions = Field(default_factory=SegmentTransitions)
    text_overlay: TextOverlay = Field(default_factory=TextOverlay)

    # Populated after voiceover generation
    voiceover_path: Optional[str] = None
    voiceover_duration_sec: Optional[float] = None


class Script(BaseModel):
    title: str
    topic: str
    duration_sec: float
    mood: Mood = Mood.NEUTRAL
    visual_style: str = "documentary"
    color_palette: list[str] = Field(default_factory=list)
    segments: list[ScriptSegment] = Field(default_factory=list)
    background_music_style: str = "ambient"
    overall_pacing: str = "medium"

    # Populated after voiceover
    total_voiceover_path: Optional[str] = None
    total_voiceover_duration_sec: Optional[float] = None


# ---------------------------------------------------------------------------
# Scene / Orchestration Models
# ---------------------------------------------------------------------------

class AudioTrack(BaseModel):
    asset_id: str
    local_path: str
    start_time: float = 0.0
    end_time: Optional[float] = None
    volume: float = 1.0
    fade_in: float = 0.5
    fade_out: float = 0.5
    loop: bool = False


class Scene(BaseModel):
    id: int
    segment_id: int
    asset_id: str                  # primary visual asset
    asset_path: str
    start_time: float              # in final video
    end_time: float
    duration_sec: float
    transition_in: TransitionType = TransitionType.FADE_IN
    transition_out: TransitionType = TransitionType.DISSOLVE
    transition_in_duration: float = 0.5
    transition_out_duration: float = 0.8
    color_grade: ColorGrade = ColorGrade.DOCUMENTARY
    text_overlays: list[TextOverlay] = Field(default_factory=list)
    voiceover: Optional[AudioTrack] = None
    sfx_tracks: list[AudioTrack] = Field(default_factory=list)


class Orchestration(BaseModel):
    project_id: str
    title: str
    topic: str
    total_duration_sec: float
    output_resolution: str = "1920x1080"
    output_fps: int = 30
    color_grade: ColorGrade = ColorGrade.DOCUMENTARY
    scenes: list[Scene] = Field(default_factory=list)
    background_music: Optional[AudioTrack] = None
    voiceover_tracks: list[AudioTrack] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Project Metadata
# ---------------------------------------------------------------------------

class ModuleStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProjectPipeline(BaseModel):
    module_1_research: ModuleStatus = ModuleStatus.PENDING
    module_2_metadata: ModuleStatus = ModuleStatus.PENDING
    module_3_script: ModuleStatus = ModuleStatus.PENDING
    module_4_orchestration: ModuleStatus = ModuleStatus.PENDING
    module_5_render: ModuleStatus = ModuleStatus.PENDING
    module_6_validation: ModuleStatus = ModuleStatus.PENDING


class ProjectMetadata(BaseModel):
    project_id: str
    topic: str
    duration_min: float
    duration_sec: float
    created_at: str
    updated_at: str
    project_dir: str
    pipeline: ProjectPipeline = Field(default_factory=ProjectPipeline)
    output_file: Optional[str] = None
    total_assets: int = 0
    script_title: Optional[str] = None


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Production Plan
# ---------------------------------------------------------------------------

class ProductionPlan(BaseModel):
    topic: str
    narrator_voice: str = "Samantha"
    testimonial_voices: list[str] = Field(default_factory=list)
    tone: ToneStyle = ToneStyle.DOCUMENTARY
    visual_style: VisualStyleChoice = VisualStyleChoice.DOCUMENTARY
    target_audience: str = "general audience"
    cultural_context: str = ""
    duration_minutes: float = 5.0
    avoid_list: list[str] = Field(default_factory=list)
    image_search_queries: list[str] = Field(default_factory=list)
    script_guidance: str = ""


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------

class ValidationCheck(BaseModel):
    name: str
    passed: bool
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    message: str = ""


class ValidationReport(BaseModel):
    project_id: str
    output_file: str
    passed: bool
    checks: list[ValidationCheck] = Field(default_factory=list)
    generated_at: str
    file_size_mb: Optional[float] = None
