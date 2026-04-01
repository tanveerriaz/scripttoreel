"""
MVP 1, Stories 1.4 & 1.5 — Asset Download & assets_raw.json tests.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.json_schemas import Asset, AssetType, AssetSource, AssetRole
from src.module_1_research import ResearchModule


def _make_asset(asset_id="vid_001", url="https://example.com/video.mp4", type_=AssetType.VIDEO):
    return Asset(
        id=asset_id,
        type=type_,
        role=AssetRole.B_ROLL,
        source=AssetSource.PEXELS,
        source_url=url,
        source_id="1234",
    )


# ---------------------------------------------------------------------------
# Story 1.4 — Download & file management
# ---------------------------------------------------------------------------

def test_download_saves_file(tmp_path):
    project_dir = tmp_path / "project"
    (project_dir / "assets/raw/video").mkdir(parents=True)

    asset = _make_asset(url="https://example.com/video.mp4")
    fake_content = b"fake video content"

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [fake_content]
    mock_resp.headers = {"content-length": str(len(fake_content))}
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    module = ResearchModule(project_dir, api_keys={})
    with patch("requests.get", return_value=mock_resp):
        updated = module.download_asset(asset)

    assert updated.local_path is not None
    assert Path(updated.local_path).exists()


def test_download_idempotent(tmp_path):
    """Calling download_asset twice for same asset should not re-download."""
    project_dir = tmp_path / "project"
    (project_dir / "assets/raw/video").mkdir(parents=True)

    asset = _make_asset(url="https://example.com/clip.mp4")
    fake_content = b"content"

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [fake_content]
    mock_resp.headers = {"content-length": "7"}
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    module = ResearchModule(project_dir, api_keys={})
    with patch("requests.get", return_value=mock_resp) as mock_get:
        updated1 = module.download_asset(asset)
        updated2 = module.download_asset(updated1)  # second call with local_path set
    # requests.get should only be called once (file already exists on second call)
    assert mock_get.call_count == 1


def test_download_failed_logs_not_raises(tmp_path):
    """A failed download should be logged and return the asset unchanged, not raise."""
    project_dir = tmp_path / "project"
    (project_dir / "assets/raw/video").mkdir(parents=True)

    asset = _make_asset(url="https://example.com/missing.mp4")
    module = ResearchModule(project_dir, api_keys={})

    import requests
    with patch("requests.get", side_effect=requests.ConnectionError("Network error")):
        result = module.download_asset(asset)

    assert result.local_path is None  # not downloaded, but no exception


def test_folder_structure_created(tmp_path):
    project_dir = tmp_path / "newproject"
    module = ResearchModule(project_dir, api_keys={})
    module._ensure_dirs()
    assert (project_dir / "assets/raw/video").exists()
    assert (project_dir / "assets/raw/image").exists()
    assert (project_dir / "assets/raw/audio").exists()


# ---------------------------------------------------------------------------
# Story 1.5 — assets_raw.json
# ---------------------------------------------------------------------------

def test_assets_raw_json_written(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    module = ResearchModule(project_dir, api_keys={})

    assets = [
        _make_asset("v1", type_=AssetType.VIDEO),
        _make_asset("i1", type_=AssetType.IMAGE),
    ]
    module.save_assets_raw(assets)

    out = project_dir / "assets_raw.json"
    assert out.exists()


def test_assets_raw_json_valid_schema(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    module = ResearchModule(project_dir, api_keys={})

    assets = [_make_asset("v1"), _make_asset("i1", type_=AssetType.IMAGE)]
    module.save_assets_raw(assets)

    raw = json.loads((project_dir / "assets_raw.json").read_text())
    assert isinstance(raw, list)
    for item in raw:
        Asset(**item)  # validates Pydantic schema


def test_run_with_no_keys_produces_empty_gracefully(tmp_path):
    """If all API keys are None, run() should return empty list without crashing."""
    project_dir = tmp_path / "project"
    (project_dir / "assets/raw/video").mkdir(parents=True)
    (project_dir / "assets/raw/image").mkdir(parents=True)
    (project_dir / "assets/raw/audio").mkdir(parents=True)

    # Write a minimal project.json
    import json as _json
    from datetime import datetime, timezone
    from src.utils.json_schemas import ProjectMetadata, ProjectPipeline
    meta = ProjectMetadata(
        project_id="test",
        topic="Test Topic",
        duration_min=2,
        duration_sec=120,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        project_dir=str(project_dir),
    )
    (project_dir / "project.json").write_text(_json.dumps(meta.model_dump(), default=str))

    module = ResearchModule(project_dir, api_keys={
        "PEXELS_API_KEY": None,
        "PIXABAY_API_KEY": None,
        "UNSPLASH_ACCESS_KEY": None,
        "FREESOUND_API_KEY": None,
    })
    result = module.run()
    # Should not raise; assets_raw.json should be written (possibly empty)
    assert isinstance(result, list)
    assert (project_dir / "assets_raw.json").exists()
