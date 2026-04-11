"""
MVP 1, Story 1.3 — Freesound client tests.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.api_handlers import FreesoundClient
from src.utils.json_schemas import Asset, AssetType, AssetSource


def make_mock_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


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
