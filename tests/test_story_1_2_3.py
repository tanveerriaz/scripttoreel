"""
MVP 1, Stories 1.2 & 1.3 — Pixabay, Unsplash, Freesound tests.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.api_handlers import PixabayClient, UnsplashClient, FreesoundClient
from src.utils.json_schemas import Asset, AssetType, AssetSource


def make_mock_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


PIXABAY_VIDEO_FIXTURE = {
    "hits": [
        {
            "id": 111,
            "duration": 20,
            "videos": {
                "large": {"url": "https://cdn.pixabay.com/vimeo/111/large.mp4", "width": 1920, "height": 1080}
            },
            "pageURL": "https://pixabay.com/videos/111",
            "user": "PhotoUser",
            "tags": "haunted, forest, dark",
        }
    ],
    "total": 1,
}

PIXABAY_IMAGE_FIXTURE = {
    "hits": [
        {
            "id": 222,
            "webformatURL": "https://cdn.pixabay.com/photo/222.jpg",
            "largeImageURL": "https://cdn.pixabay.com/photo/222_large.jpg",
            "imageWidth": 1920,
            "imageHeight": 1080,
            "pageURL": "https://pixabay.com/images/222",
            "user": "ImgUser",
            "tags": "castle, dark",
        }
    ],
    "total": 1,
}

UNSPLASH_FIXTURE = {
    "results": [
        {
            "id": "abc123",
            "urls": {"raw": "https://images.unsplash.com/photo-abc123?raw=true", "regular": "https://images.unsplash.com/photo-abc123?w=1080"},
            "width": 4000,
            "height": 3000,
            "description": "dark foggy forest",
            "user": {"name": "Unsplash Photographer", "links": {"html": "https://unsplash.com/@photographer"}},
            "links": {"html": "https://unsplash.com/photos/abc123"},
        }
    ],
    "total": 1,
}

FREESOUND_FIXTURE = {
    "results": [
        {
            "id": 999,
            "name": "wind_howl.wav",
            "duration": 8.5,
            "previews": {"preview-hq-mp3": "https://freesound.org/previews/999/999_hq.mp3"},
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "username": "AudioUser",
            "tags": ["wind", "howl", "ambient"],
            "description": "Howling wind at night",
        }
    ],
    "count": 1,
}


# ---------------------------------------------------------------------------
# Pixabay tests
# ---------------------------------------------------------------------------

def test_pixabay_missing_key_skips_gracefully():
    client = PixabayClient(api_key=None)
    assets = client.search_videos("test")
    assert assets == []


def test_pixabay_video_search_returns_assets():
    client = PixabayClient(api_key="test_key")
    with patch.object(client.session, "get", return_value=make_mock_response(PIXABAY_VIDEO_FIXTURE)):
        assets = client.search_videos("haunted", per_page=1)
    assert len(assets) == 1
    assert assets[0].type == AssetType.VIDEO
    assert assets[0].source == AssetSource.PIXABAY


def test_pixabay_image_search_returns_assets():
    client = PixabayClient(api_key="test_key")
    with patch.object(client.session, "get", return_value=make_mock_response(PIXABAY_IMAGE_FIXTURE)):
        assets = client.search_images("castle", per_page=1)
    assert len(assets) == 1
    assert assets[0].type == AssetType.IMAGE


# ---------------------------------------------------------------------------
# Unsplash tests
# ---------------------------------------------------------------------------

def test_unsplash_missing_key_skips_gracefully():
    client = UnsplashClient(access_key=None)
    assets = client.search_photos("test")
    assert assets == []


def test_unsplash_search_returns_assets():
    client = UnsplashClient(access_key="test_key")
    with patch.object(client.session, "get", return_value=make_mock_response(UNSPLASH_FIXTURE)):
        assets = client.search_photos("forest", per_page=1)
    assert len(assets) == 1
    assert assets[0].type == AssetType.IMAGE
    assert assets[0].source == AssetSource.UNSPLASH


# ---------------------------------------------------------------------------
# Freesound tests
# ---------------------------------------------------------------------------

def test_freesound_missing_key_skips_gracefully():
    client = FreesoundClient(api_key=None)
    assets = client.search_sounds("test")
    assert assets == []


def test_freesound_search_returns_audio_assets():
    client = FreesoundClient(api_key="test_key")
    with patch.object(client.session, "get", return_value=make_mock_response(FREESOUND_FIXTURE)):
        assets = client.search_sounds("wind", per_page=1)
    assert len(assets) == 1
    assert assets[0].type == AssetType.AUDIO


def test_freesound_asset_has_source_url():
    client = FreesoundClient(api_key="test_key")
    with patch.object(client.session, "get", return_value=make_mock_response(FREESOUND_FIXTURE)):
        assets = client.search_sounds("wind", per_page=1)
    assert assets[0].source_url is not None
