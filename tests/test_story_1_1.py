"""
MVP 1, Story 1.1 — Pexels Video & Image Search tests.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.api_handlers import PexelsClient, APIKeyError
from src.utils.json_schemas import Asset, AssetType, AssetSource


PEXELS_VIDEO_FIXTURE = {
    "videos": [
        {
            "id": 1234,
            "duration": 15,
            "width": 1920,
            "height": 1080,
            "url": "https://www.pexels.com/video/1234",
            "video_files": [
                {"link": "https://cdn.pexels.com/v/1234.mp4", "width": 1920, "height": 1080, "file_type": "video/mp4"}
            ],
            "user": {"name": "John Doe", "url": "https://pexels.com/@john"},
            "tags": [],
        }
    ],
    "total_results": 1,
}

PEXELS_PHOTO_FIXTURE = {
    "photos": [
        {
            "id": 5678,
            "width": 3000,
            "height": 2000,
            "url": "https://www.pexels.com/photo/5678",
            "src": {"original": "https://images.pexels.com/photos/5678/photo.jpg"},
            "photographer": "Jane Smith",
            "photographer_url": "https://pexels.com/@jane",
            "alt": "Dark misty forest",
        }
    ],
    "total_results": 1,
}


def make_mock_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pexels_missing_key_raises_error():
    client = PexelsClient(api_key=None)
    with pytest.raises(APIKeyError):
        client.search_videos("haunted forest")


def test_pexels_video_search_returns_assets():
    client = PexelsClient(api_key="test_key_123")
    with patch.object(client.session, "get", return_value=make_mock_response(PEXELS_VIDEO_FIXTURE)):
        assets = client.search_videos("haunted forest", per_page=1)
    assert len(assets) == 1
    a = assets[0]
    assert isinstance(a, Asset)
    assert a.type == AssetType.VIDEO
    assert a.source == AssetSource.PEXELS
    assert a.source_url is not None


def test_pexels_image_search_returns_assets():
    client = PexelsClient(api_key="test_key_123")
    with patch.object(client.session, "get", return_value=make_mock_response(PEXELS_PHOTO_FIXTURE)):
        assets = client.search_images("dark forest", per_page=1)
    assert len(assets) == 1
    a = assets[0]
    assert a.type == AssetType.IMAGE
    assert a.source == AssetSource.PEXELS


def test_pexels_asset_schema_valid():
    """Each returned asset validates as a full Pydantic Asset."""
    client = PexelsClient(api_key="test_key_123")
    with patch.object(client.session, "get", return_value=make_mock_response(PEXELS_VIDEO_FIXTURE)):
        assets = client.search_videos("test", per_page=1)
    # If the model validation would fail, this would raise
    for a in assets:
        dumped = a.model_dump()
        Asset(**dumped)  # re-validate


def test_pexels_rate_limit_retries():
    """429 response triggers retry logic; success on second attempt."""
    client = PexelsClient(api_key="test_key_123")
    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.raise_for_status.side_effect = Exception("429")
    ok_response = make_mock_response(PEXELS_VIDEO_FIXTURE)

    with patch.object(client.session, "get", side_effect=[rate_limit_response, ok_response]):
        with patch("time.sleep"):  # don't actually sleep
            assets = client.search_videos("test", per_page=1)
    assert len(assets) == 1


def test_pexels_401_raises_api_key_error():
    client = PexelsClient(api_key="bad_key")
    import requests
    resp = MagicMock()
    resp.status_code = 401
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp)

    with patch.object(client.session, "get", return_value=resp):
        with pytest.raises(APIKeyError):
            client.search_videos("test")
