# VideoForge — Agile MVP Plan

## Environment Audit
| Tool | Version | Status |
|------|---------|--------|
| ffmpeg (VideoToolBox) | 8.1 | ✅ |
| ffprobe | 8.1 | ✅ |
| ollama | 0.18.3 | ✅ |
| python3 | 3.14.3 | ✅ |

## Methodology
- Each module = 1 MVP
- Each MVP has User Stories with acceptance criteria + automated tests
- Story is "Done" only when: code ✅ + tests pass ✅ + output verified ✅
- Only advance to next story after current story is verified
- Only advance to next MVP after ALL stories in current MVP pass

---

## MVP 0 — Scaffold & Infrastructure
_Goal: project skeleton, Pydantic schemas, CLI shell, API wrappers_

### Story 0.1 — Project Initialization
**As a user, I can run `python main.py --init --topic "X" --duration 5` to create a new project.**

Acceptance Criteria:
- Running `--init` creates `projects/<slug>/` directory
- Creates `project.json` with topic, duration, timestamps, pipeline status
- Prints the new project ID/path to the terminal
- Running `--init` twice on the same topic appends a counter suffix

Features:
- `main.py` with Click CLI
- `src/utils/json_schemas.py` — all Pydantic models (Asset, Script, Scene, Orchestration, ProjectMetadata, ValidationReport)
- Project slug = snake_case of topic

Tests (`tests/test_story_0_1.py`):
- [ ] `test_init_creates_directory` — project dir exists after init
- [ ] `test_init_creates_project_json` — `project.json` is valid ProjectMetadata
- [ ] `test_init_slug_format` — slug is snake_case, no special chars
- [ ] `test_init_duplicate_topic_suffix` — second init appends _2

User sees:
```
✅ Project created: projects/haunted_places_in_pakistan/
   Project ID: haunted_places_in_pakistan
   Topic: Haunted Places in Pakistan
   Duration: 5 min
   Run: python main.py --run --project haunted_places_in_pakistan
```

---

### Story 0.2 — Project Status Command
**As a user, I can run `python main.py --status --project <id>` to see pipeline progress.**

Acceptance Criteria:
- Prints a table of all 6 modules with their status (pending/running/complete/failed)
- Shows project metadata (topic, duration, created date)
- Returns non-zero exit code if project does not exist

Features:
- `--status` flag in main.py
- Rich table output

Tests:
- [ ] `test_status_shows_all_modules` — all 6 modules in output
- [ ] `test_status_nonexistent_project_exits_1` — sys.exit(1) if missing

---

### Story 0.3 — Configuration Loading
**As a developer, all config files load without errors and API keys are read from env.**

Acceptance Criteria:
- `ollama_prompts.yaml` loads and has `script_generation.system` key
- `ffmpeg_presets.yaml` loads and has `output.default.codec` = `h264_videotoolbox`
- `api_keys.env` is loaded via dotenv; missing keys return `None` (not exception)

Features:
- `src/utils/config_loader.py` — `load_config()`, `load_api_keys()`

Tests:
- [ ] `test_ollama_prompts_load` — yaml valid, key present
- [ ] `test_ffmpeg_presets_load` — yaml valid, codec key present
- [ ] `test_missing_api_key_returns_none` — no KeyError on missing key

---

## MVP 1 — Module 1: Research & Asset Discovery
_Goal: search 4 APIs, download assets, write assets_raw.json_

### Story 1.1 — Pexels Video & Image Search
**As a user, I can search Pexels for videos and images related to my topic.**

Acceptance Criteria:
- `PexelsClient.search_videos(query, per_page=5)` returns ≥1 result when key is set
- `PexelsClient.search_images(query, per_page=5)` returns ≥1 result
- Each result maps to a valid `Asset` Pydantic object
- If API key is missing/invalid, raises `APIKeyError` (not crash)
- Rate limit 429 triggers exponential backoff (tested with mock)

Features:
- `src/utils/api_handlers.py` — `PexelsClient`

Tests (`tests/test_story_1_1.py`):
- [ ] `test_pexels_video_search_returns_assets` — integration (skipped if no key)
- [ ] `test_pexels_image_search_returns_assets` — integration (skipped if no key)
- [ ] `test_pexels_asset_schema_valid` — each result validates against Asset
- [ ] `test_pexels_missing_key_raises_error` — mock 401 → APIKeyError
- [ ] `test_pexels_rate_limit_retries` — mock 429 → retries with backoff

---

### Story 1.2 — Pixabay & Unsplash Search
**As a user, I can search Pixabay and Unsplash for additional stock assets.**

Acceptance Criteria:
- `PixabayClient.search_videos()` and `.search_images()` return valid Assets
- `UnsplashClient.search_photos()` returns valid Assets
- Missing key = skip source + log warning (not crash)

Features:
- `PixabayClient`, `UnsplashClient` in `api_handlers.py`

Tests:
- [ ] `test_pixabay_search_returns_assets` — integration (skipped if no key)
- [ ] `test_unsplash_search_returns_assets` — integration (skipped if no key)
- [ ] `test_pixabay_missing_key_skips_gracefully`
- [ ] `test_unsplash_missing_key_skips_gracefully`

---

### Story 1.3 — Freesound Audio Search
**As a user, I can search Freesound for ambient audio and SFX.**

Acceptance Criteria:
- `FreesoundClient.search_sounds(query)` returns valid Audio Assets
- Assets have `type=audio`, `role=sfx` or `role=music`
- Download url is populated

Features:
- `FreesoundClient` in `api_handlers.py`

Tests:
- [ ] `test_freesound_search_returns_audio_assets`
- [ ] `test_freesound_asset_type_is_audio`
- [ ] `test_freesound_missing_key_skips`

---

### Story 1.4 — Asset Download & File Management
**As a user, all discovered assets are downloaded to the project's assets/ folder.**

Acceptance Criteria:
- Files downloaded to `projects/<id>/assets/raw/{video,image,audio}/`
- Filename = `<asset_id>.<ext>`
- Download resumes / skips if file already exists (idempotent)
- Progress bar displayed via tqdm
- Failed downloads logged, not crash-fatal

Features:
- `download_asset(asset)` in `module_1_research.py`
- Folder creation handled automatically

Tests:
- [ ] `test_download_saves_file` — mock HTTP → file on disk
- [ ] `test_download_idempotent` — second call skips (no re-download)
- [ ] `test_download_failed_logged_not_raised`
- [ ] `test_folder_structure_created`

---

### Story 1.5 — assets_raw.json Output
**As a user, after Module 1 runs, I have a complete `assets_raw.json` with all found assets.**

Acceptance Criteria:
- `assets_raw.json` is valid JSON
- All assets validate against `Asset` Pydantic schema
- `ready_for_use = false` (metadata not yet extracted)
- Module updates `project.json` pipeline status → `module_1_research: complete`

Features:
- `ResearchModule.save_assets_raw(assets)`
- `ResearchModule.run()` — orchestrates all sources

Tests:
- [ ] `test_assets_raw_json_valid_schema` — Pydantic validates all items
- [ ] `test_pipeline_status_updated_to_complete`
- [ ] `test_run_with_no_keys_produces_empty_assets_gracefully`

MVP 1 Complete Verification: Run `python main.py --module 1 --project test` → show assets_raw.json

---

## MVP 2 — Module 2: Metadata Extraction
_Goal: enrich every asset with ffprobe/Pillow/librosa metadata + quality scores_

### Story 2.1 — Video Metadata via ffprobe
**As a user, every video asset has duration, resolution, FPS, codec populated.**

Acceptance Criteria:
- `extract_video_metadata(path)` returns `VideoMetadata` with all fields
- Uses `ffprobe -v quiet -print_format json -show_streams`
- Works with MP4, WebM, MOV

Tests:
- [ ] `test_ffprobe_mp4_metadata` — fixture video → correct duration/resolution
- [ ] `test_ffprobe_missing_file_raises` — FileNotFoundError
- [ ] `test_ffprobe_output_maps_to_pydantic`

---

### Story 2.2 — Image Metadata via Pillow
**As a user, every image asset has dimensions, format, aspect ratio populated.**

Tests:
- [ ] `test_pillow_image_metadata` — fixture JPEG → correct width/height
- [ ] `test_pillow_aspect_ratio_computed` — 1920x1080 → "16:9"

---

### Story 2.3 — Audio Metadata via librosa
**As a user, every audio asset has duration, sample rate, BPM populated.**

Tests:
- [ ] `test_librosa_audio_duration` — fixture WAV → correct duration
- [ ] `test_librosa_bpm_detected` — fixture music → BPM > 0

---

### Story 2.4 — Dominant Color Extraction via OpenCV
**As a user, every visual asset has a color palette of 5 dominant hex colors.**

Tests:
- [ ] `test_dominant_colors_returns_5_hex` — fixture image → 5 #RRGGBB strings
- [ ] `test_dominant_colors_pure_red` — red image → palette is red-dominant

---

### Story 2.5 — Quality Score & assets.json
**As a user, each asset has a quality_score 0-10 and assets.json is written.**

Acceptance Criteria:
- Quality score formula: resolution (0-4pts) + aspect ratio (0-2pts) + duration (0-2pts) + codec (0-2pts)
- `assets.json` replaces `assets_raw.json` with full metadata
- `ready_for_use = true` for assets passing quality threshold (≥5.0)
- Pipeline status → `module_2_metadata: complete`

Tests:
- [ ] `test_quality_score_1080p_h264_is_high`
- [ ] `test_quality_score_low_res_is_low`
- [ ] `test_assets_json_written_and_valid`
- [ ] `test_pipeline_status_updated`

---

## MVP 3 — Module 3: Script Generation & Voiceover
_Goal: Ollama LLM → script.json + Coqui TTS → voiceover.wav_

### Story 3.1 — Ollama Connection & Script Generation
**As a user, `module_3` calls local Ollama and returns a structured script.**

Acceptance Criteria:
- POST to `localhost:11434/api/generate` with system + user prompt
- Response parsed as JSON → validates against `Script` Pydantic model
- Retries up to 3x on JSON parse failure
- If Ollama unreachable → `OllamaNotAvailableError` with clear message

Tests:
- [ ] `test_ollama_request_format` — mock server captures correct payload
- [ ] `test_script_json_parses_to_pydantic` — fixture JSON → Script model
- [ ] `test_ollama_unreachable_raises_clear_error`
- [ ] `test_retry_on_bad_json` — mock returns bad JSON twice, good on 3rd

---

### Story 3.2 — Script Segment Validation
**As a user, the generated script has correct total duration and valid segment types.**

Acceptance Criteria:
- Sum of segment durations ≈ target duration ±10%
- First segment type = `intro`, last = `outro`
- Each segment has non-empty `text` and `b_roll_keywords`

Tests:
- [ ] `test_segment_durations_sum_to_target`
- [ ] `test_first_segment_is_intro`
- [ ] `test_last_segment_is_outro`
- [ ] `test_no_empty_segment_text`

---

### Story 3.3 — TTS Voiceover Generation (Coqui)
**As a user, each script segment has a corresponding WAV voiceover file.**

Acceptance Criteria:
- Coqui `TTS` generates `voiceover_<N>.wav` for each segment
- If Coqui fails → macOS `say` fallback produces valid WAV
- Output WAVs are 22050Hz or 44100Hz, mono acceptable

Tests:
- [ ] `test_tts_produces_wav_file` — integration (skipped if no GPU/model)
- [ ] `test_macos_say_fallback_produces_wav`
- [ ] `test_wav_file_is_valid_audio` — soundfile.read() works

---

### Story 3.4 — Voiceover Concatenation & script.json
**As a user, all segment WAVs are merged into a single voiceover.wav with natural pauses.**

Acceptance Criteria:
- `voiceover.wav` duration ≈ sum of segment durations + (n-1 × 0.5s pause)
- `script.json` written with `voiceover_path` populated on each segment
- Pipeline status → `module_3_script: complete`

Tests:
- [ ] `test_concatenation_duration_correct`
- [ ] `test_pause_between_segments`
- [ ] `test_script_json_voiceover_paths_exist`
- [ ] `test_pipeline_status_updated`

---

## MVP 4 — Module 4: Scene Planning & Orchestration
_Goal: map assets → segments → scenes → orchestration.json_

### Story 4.1 — Asset-to-Segment Matching
**As a user, each script segment is matched to the best available asset.**

Acceptance Criteria:
- Matching score considers: mood overlap, visual tag overlap, duration fit, quality score
- Returns highest-scoring asset per segment
- Falls back to random valid asset if no good match (score=0)

Tests:
- [ ] `test_mood_match_boosts_score`
- [ ] `test_tag_overlap_boosts_score`
- [ ] `test_best_asset_selected`
- [ ] `test_fallback_on_no_match`

---

### Story 4.2 — Timeline Construction
**As a user, scenes are arranged in a continuous timeline with no gaps.**

Acceptance Criteria:
- `scene.start_time` + `scene.duration_sec` = next `scene.start_time`
- Total timeline ≈ script duration ±1s
- Scene IDs are sequential (1, 2, 3...)

Tests:
- [ ] `test_no_gaps_in_timeline`
- [ ] `test_scene_ids_sequential`
- [ ] `test_total_duration_matches_script`

---

### Story 4.3 — Transitions & Color Grading
**As a user, transitions are assigned between scenes and color grades match the mood.**

Acceptance Criteria:
- `dark`/`mysterious` mood → `dark_mysterious` color grade
- Transitions alternate between `dissolve` (default) and `crossfade`
- Intro scene always has `fade_in` transition_in

Tests:
- [ ] `test_dark_mood_gets_dark_grade`
- [ ] `test_intro_scene_fade_in`
- [ ] `test_transitions_assigned_to_all_scenes`

---

### Story 4.4 — Audio Mix Plan & orchestration.json
**As a user, orchestration.json has audio tracks with correct volumes.**

Acceptance Criteria:
- Voiceover volume = 1.0
- Background music volume = 0.12
- SFX volume = 0.4
- `orchestration.json` validates against Orchestration Pydantic model
- Pipeline status → `module_4_orchestration: complete`

Tests:
- [ ] `test_voiceover_volume_is_1`
- [ ] `test_music_volume_is_0_12`
- [ ] `test_orchestration_json_valid_schema`
- [ ] `test_pipeline_status_updated`

---

## MVP 5 — Module 5: FFmpeg Rendering
_Goal: orchestration.json → final_video.mp4 via h264_videotoolbox_

### Story 5.1 — FFmpegCommand Builder
**As a developer, I can build complex FFmpeg commands using a fluent builder API.**

Acceptance Criteria:
- `FFmpegCommand().input(path).output(path).build()` returns valid argv list
- `filter_complex(graph)` appends filter graph
- `run()` executes and returns returncode
- `dry_run=True` prints command without executing

Tests:
- [ ] `test_builder_produces_valid_argv`
- [ ] `test_filter_complex_appended`
- [ ] `test_dry_run_does_not_execute`
- [ ] `test_run_returns_zero_on_success` — runs `ffmpeg -version`

---

### Story 5.2 — Scene Rendering (scale + color grade)
**As a user, each scene clip is scaled to 1920x1080 with color grading applied.**

Acceptance Criteria:
- Output clips are exactly 1920x1080
- Color grade `eq` filter values match `ffmpeg_presets.yaml`
- Works for both video and image (image → looped video)

Tests:
- [ ] `test_scene_output_is_1080p` — render fixture clip → ffprobe resolution
- [ ] `test_image_input_creates_video_clip`
- [ ] `test_color_grade_filter_string_valid` — parseable by ffmpeg

---

### Story 5.3 — Concat & Transitions
**As a user, scene clips are concatenated with smooth xfade transitions.**

Acceptance Criteria:
- `xfade` filter applied between adjacent clips
- Total output duration = sum(scene durations) ± 1s
- No black frames between clips

Tests:
- [ ] `test_concat_two_clips_duration` — 3s + 3s dissolve 0.8s = 5.2s
- [ ] `test_xfade_filter_string_format`

---

### Story 5.4 — Audio Mixing
**As a user, voiceover is mixed with background music at correct levels.**

Acceptance Criteria:
- `amix` filter combines voiceover (vol 1.0) + music (vol 0.12)
- Output audio is AAC 192k stereo
- Music fades out in last 2s

Tests:
- [ ] `test_amix_filter_string_contains_volumes`
- [ ] `test_output_has_audio_stream` — ffprobe on output
- [ ] `test_audio_codec_is_aac`

---

### Story 5.5 — Final Encode & Output
**As a user, `python main.py --module 5` produces a valid 1080p MP4.**

Acceptance Criteria:
- Output: `projects/<id>/output/final_video.mp4`
- Codec: `h264` (videotoolbox or libx264 fallback)
- Bitrate: 4000k–6000k
- Duration within ±5% of target
- `-movflags +faststart` for web compatibility
- Pipeline status → `module_5_render: complete`

Tests:
- [ ] `test_output_file_exists`
- [ ] `test_output_codec_is_h264`
- [ ] `test_output_duration_within_tolerance`
- [ ] `test_output_has_faststart_flag` — moov atom at start
- [ ] `test_pipeline_status_updated`

---

## MVP 6 — Module 6: Quality Validation
_Goal: 10 automated checks + metadata embedding + validation report_

### Story 6.1 — ffprobe-based Checks
**As a user, the output video is verified against technical specs.**

Acceptance Criteria (10 checks):
1. File exists
2. Codec = h264
3. Resolution = 1920x1080
4. FPS = 29.97 or 30
5. Duration within ±5% of target
6. Bitrate between 2000k–8000k
7. At least 1 audio stream (AAC)
8. File size > 1MB and < 5GB
9. Frame count ≈ duration × fps ±10%
10. Colorspace = bt709

Tests:
- [ ] `test_all_10_checks_pass_on_valid_output` — fixture MP4 passes all
- [ ] `test_codec_check_fails_on_wrong_codec`
- [ ] `test_duration_check_fails_on_wrong_duration`

---

### Story 6.2 — Metadata Embedding
**As a user, the final video has title, description, and keywords embedded.**

Acceptance Criteria:
- `ffmpeg -metadata title="..." -metadata comment="..." -metadata genre="..."`
- Reading back with ffprobe confirms metadata present

Tests:
- [ ] `test_metadata_embedded_title` — ffprobe reads back correct title
- [ ] `test_metadata_embedded_keywords`

---

### Story 6.3 — Validation Report
**As a user, I see a rich table of pass/fail checks and a saved JSON report.**

Acceptance Criteria:
- Console: rich table with ✅/❌ per check, expected vs actual values
- `validation_report.json` written to project dir
- Pipeline status → `module_6_validation: complete`
- Non-zero exit code if any check fails

Tests:
- [ ] `test_report_json_valid_schema`
- [ ] `test_report_all_pass_returns_exit_0`
- [ ] `test_report_one_fail_returns_exit_1`
- [ ] `test_pipeline_status_updated`

---

## Execution Tracker

| MVP | Stories | Status |
|-----|---------|--------|
| 0 — Scaffold | 0.1, 0.2, 0.3 | ⬜ |
| 1 — Research | 1.1, 1.2, 1.3, 1.4, 1.5 | ⬜ |
| 2 — Metadata | 2.1, 2.2, 2.3, 2.4, 2.5 | ⬜ |
| 3 — Script+TTS | 3.1, 3.2, 3.3, 3.4 | ⬜ |
| 4 — Orchestration | 4.1, 4.2, 4.3, 4.4 | ⬜ |
| 5 — Render | 5.1, 5.2, 5.3, 5.4, 5.5 | ⬜ |
| 6 — Validation | 6.1, 6.2, 6.3 | ⬜ |
